#!/usr/bin/env python3
"""
linearization_error_patchtokens.py

Compute local linearization error of the learned predictor in patch-token space.

Key choices:
- perturb ONLY the visual patch-token dimensions
- do NOT perturb control/action dimensions
- evaluate predictor local linearization around encoded z_src
- use JVP instead of forming the full Jacobian

Linearization tested:
    f(z0 + d)  vs  f(z0) + J(z0) d

where d only lives in the visual patch-token subspace.

Recommended metric:
    rel_err = ||f(z0+d) - (f(z0)+Jd)|| / (||f(z0+d)-f(z0)|| + eps)

You can also inspect:
    raw_err = ||f(z0+d) - (f(z0)+Jd)||
    curv_err = ||...|| / (||d||^2 + eps)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Predictor import ViTPredictor
from visual_world_model import VWorldModel
from ControlEncoderMLP import ControlEncoderMLP
from unicycle_dataloader import load_unicycle_slice_train_val


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ControlEncoderSingle(nn.Module):
    def __init__(self, action_dim, emb_dim=10):
        super().__init__()
        self.fc = nn.Linear(action_dim, emb_dim)

    def forward(self, a):
        return self.fc(a)

class PatchTokenEncoder(nn.Module):
    """
    Dummy encoder because dataset already stores patch tokens.
    """

    def __init__(self, emb_dim: int = 768):
        super().__init__()
        self.emb_dim = emb_dim
        self.name = "dummy_encoder"
        self.patch_size = 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyProprio(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_model(device: torch.device,
                num_hist: int,
                num_pred: int,
                num_patches: int = 256,
                token_dim: int = 768,
                action_dim_raw: int = 2,
                action_emb_dim: int = 10,
                num_action_repeat: int = 7,
                num_proprio_repeat: int = 7,
                predictor_depth: int = 6,
                predictor_heads: int = 16,
                predictor_mlp_dim: int = 2048) -> VWorldModel:
    """
    Mirrors your training setup, except uses ControlEncoderMLP because that file
    is available here.
    """
    encoder = PatchTokenEncoder(token_dim)
    proprio_encoder = DummyProprio()
    #action_encoder = ControlEncoderMLP(action_dim=action_dim_raw, emb_dim=action_emb_dim)
    action_encoder = ControlEncoderSingle(action_dim=action_dim_raw, emb_dim=action_emb_dim)

    predictor_dim = token_dim + action_emb_dim * num_action_repeat

    predictor = ViTPredictor(
        num_patches=num_patches,
        num_frames=num_hist,
        dim=predictor_dim,
        depth=predictor_depth,
        heads=predictor_heads,
        mlp_dim=predictor_mlp_dim,
    )

    model = VWorldModel(
        image_size=224,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=0,
        action_dim=action_emb_dim,
        concat_dim=1,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ).to(device)

    return model


def load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)


def get_batch(data_root: str,
              n_rollout: int,
              split_ratio: float,
              num_hist: int,
              num_pred: int,
              frameskip: int,
              batch_size: int,
              num_workers: int,
              device: torch.device):
    datasets, _ = load_unicycle_slice_train_val(
        data_root=data_root,
        n_rollout=n_rollout,
        split_ratio=split_ratio,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
    )

    loader = DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    obs, act, state = next(iter(loader))
    for k in obs:
        obs[k] = obs[k].to(device, non_blocking=True)
    act = act.to(device, non_blocking=True)
    return obs, act, state


def make_visual_only_delta(z_src: torch.Tensor,
                           visual_dim: int,
                           sigma: float,
                           distribution: str = "gaussian",
                           normalize_to_radius: float | None = None) -> torch.Tensor:
    """
    Create perturbation with same shape as z_src, but only on visual patch-token dims.
    Action/proprio appended dims are left exactly zero.

    z_src shape: (B, T, P, D_total)
    visual dims are z_src[..., :visual_dim]
    """
    delta = torch.zeros_like(z_src)

    if distribution == "gaussian":
        delta_visual = sigma * torch.randn_like(z_src[..., :visual_dim])
    elif distribution == "rademacher":
        delta_visual = sigma * torch.sign(torch.randn_like(z_src[..., :visual_dim]))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    if normalize_to_radius is not None:
        flat = delta_visual.reshape(delta_visual.shape[0], -1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        flat = flat / norms * normalize_to_radius
        delta_visual = flat.reshape_as(delta_visual)

    delta[..., :visual_dim] = delta_visual
    return delta


def batch_fro_norm(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).norm(dim=1)


def compute_linearization_errors(
    model: VWorldModel,
    obs: dict[str, torch.Tensor],
    act: torch.Tensor,
    num_samples: int,
    sigma: float,
    distribution: str,
    normalize_radius: float | None,
    output_slice: str = "visual_only",
):
    """
    Compute local linearization error of predictor around z_src.

    output_slice:
      - "visual_only": error only on visual dimensions
      - "all_nonaction": all dims except final action dims
      - "all": all output dims
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(obs, act)                 # (B, T_total, P, D_total)
        z_src = z[:, :model.num_hist, :, :].clone()

    # For concat_dim=1, visual dims come first, then proprio dims, then action dims.
    if model.concat_dim != 1:
        raise NotImplementedError("This script assumes concat_dim=1.")

    visual_dim = z_src.shape[-1] - (model.proprio_dim + model.action_dim)
    total_dim = z_src.shape[-1]

    if output_slice == "visual_only":
        out_lo = 0
        out_hi = visual_dim
    elif output_slice == "all_nonaction":
        out_lo = 0
        out_hi = total_dim - model.action_dim
    elif output_slice == "all":
        out_lo = 0
        out_hi = total_dim
    else:
        raise ValueError(f"Unknown output_slice: {output_slice}")

    def predictor_only(z_in: torch.Tensor) -> torch.Tensor:
        return model.predict(z_in)

    z0 = z_src.detach().clone().requires_grad_(True)
    y0 = predictor_only(z0)

    raw_errs = []
    rel_errs = []
    per_step_errs = []
    curvature_errs = []
    delta_norms = []
    nonlinear_change_norms = []
    jvp_norms = []

    for _ in range(num_samples):
        delta = make_visual_only_delta(
            z_src=z0,
            visual_dim=visual_dim,
            sigma=sigma,
            distribution=distribution,
            normalize_to_radius=normalize_radius,
        )

        # JVP: J(z0) * delta
        _, jvp = torch.autograd.functional.jvp(
            predictor_only,
            (z0,),
            (delta,),
            create_graph=False,
            strict=False,
        )

        with torch.no_grad():
            y_true = predictor_only(z0 + delta)
            y_lin = y0 + jvp

            y_true_eval = y_true[..., out_lo:out_hi]
            y_lin_eval = y_lin[..., out_lo:out_hi]
            y0_eval = y0[..., out_lo:out_hi]

            err = y_true_eval - y_lin_eval
            nonlinear_change = y_true_eval - y0_eval

            raw = batch_fro_norm(err)
            rel = raw / (batch_fro_norm(nonlinear_change) + 1e-12)
            dnorm = batch_fro_norm(delta[..., :visual_dim])
            per_step = raw / (dnorm + 1e-12)
            curvature = raw / (dnorm.pow(2) + 1e-12)
            jnorm = batch_fro_norm(jvp[..., out_lo:out_hi])

            raw_errs.append(raw.cpu())
            rel_errs.append(rel.cpu())
            per_step_errs.append(per_step.cpu())
            curvature_errs.append(curvature.cpu())
            delta_norms.append(dnorm.cpu())
            nonlinear_change_norms.append(batch_fro_norm(nonlinear_change).cpu())
            jvp_norms.append(jnorm.cpu())

    def cat_mean_std(lst):
        x = torch.cat(lst, dim=0)
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "median": float(x.median().item()),
        }

    summary = {
        "num_samples": num_samples,
        "sigma": sigma,
        "normalize_radius": normalize_radius,
        "distribution": distribution,
        "output_slice": output_slice,
        "visual_dim": int(visual_dim),
        "total_dim": int(total_dim),
        "action_dim_total": int(model.action_dim),
        "proprio_dim_total": int(model.proprio_dim),
        "raw_error": cat_mean_std(raw_errs),
        "relative_error": cat_mean_std(rel_errs),
        "error_over_delta_norm": cat_mean_std(per_step_errs),
        "error_over_delta_norm_sq": cat_mean_std(curvature_errs),
        "delta_norm": cat_mean_std(delta_norms),
        "nonlinear_change_norm": cat_mean_std(nonlinear_change_norms),
        "jvp_norm": cat_mean_std(jvp_norms),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    # for 768
    #data_root_768 = "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset"
    #ckpt_768 = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_5/model_0070.pt"
    
    #for 384:
    data_root_384 = "/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset"
    ckpt_384 = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_14_1/model_0070.pt"
    
    
    parser.add_argument("--data_root", type=str, default = data_root_384)
    parser.add_argument("--ckpt", type=str, default = ckpt_384)

    parser.add_argument("--n_rollout", type=int, default=10000)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--frameskip", type=int, default=1)

    parser.add_argument("--num_hist", type=int, default=1)
    parser.add_argument("--num_pred", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=1e-2)
    parser.add_argument("--distribution", type=str, default="gaussian",
                        choices=["gaussian", "rademacher"])
    parser.add_argument("--normalize_radius", type=float, default=None)

    parser.add_argument("--output_slice", type=str, default="visual_only",
                        choices=["visual_only", "all_nonaction", "all"])

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = build_model(
        device=device,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        token_dim = 384
    )
    load_checkpoint(model, args.ckpt)
    model.eval()

    obs, act, _ = get_batch(
        data_root=args.data_root,
        n_rollout=args.n_rollout,
        split_ratio=args.split_ratio,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    summary = compute_linearization_errors(
        model=model,
        obs=obs,
        act=act,
        num_samples=args.num_samples,
        sigma=args.sigma,
        distribution=args.distribution,
        normalize_radius=args.normalize_radius,
        output_slice=args.output_slice,
    )

    print(json.dumps(summary, indent=2))

    if args.save_json is not None:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()