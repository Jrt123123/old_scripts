#!/usr/bin/env python3
"""
linearization_error_patchtokens_vs_radius.py

Compute local linearization error of the learned predictor in patch-token space,
and plot error metrics as a function of ||delta||.

Linearization tested:
    f(z0 + d)  vs  f(z0) + J(z0) d

where d only lives in the visual patch-token subspace.

This version:
- sweeps perturbation radii
- samples many random directions per radius
- plots mean/std error vs ||delta||
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Predictor import ViTPredictor
from visual_world_model import VWorldModel
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


def build_model(
    device: torch.device,
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
    predictor_mlp_dim: int = 2048,
) -> VWorldModel:
    encoder = PatchTokenEncoder(token_dim)
    proprio_encoder = DummyProprio()
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


def get_batch(
    data_root: str,
    n_rollout: int,
    split_ratio: float,
    num_hist: int,
    num_pred: int,
    frameskip: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
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


def batch_fro_norm(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).norm(dim=1)


def make_visual_direction(
    z_src: torch.Tensor,
    visual_dim: int,
    distribution: str = "gaussian",
) -> torch.Tensor:
    """
    Create a random direction in the visual subspace only, then normalize later.
    """
    delta = torch.zeros_like(z_src)

    if distribution == "gaussian":
        delta_visual = torch.randn_like(z_src[..., :visual_dim])
    elif distribution == "rademacher":
        delta_visual = torch.sign(torch.randn_like(z_src[..., :visual_dim]))
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    delta[..., :visual_dim] = delta_visual
    return delta


def summarize_temporal_step_norms(z: torch.Tensor, visual_dim: int):
    """
    Compute empirical E||z_{t+1} - z_t|| over the batch/time sequence.

    z shape: (B, T, P, D_total)
    """
    if z.shape[1] < 2:
        return {
            "all_dims": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0},
            "visual_only": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0},
        }

    dz_all = z[:, 1:, :, :] - z[:, :-1, :, :]
    dz_vis = z[:, 1:, :, :visual_dim] - z[:, :-1, :, :visual_dim]

    # flatten each (B, T-1, ...) step into one vector norm
    step_norms_all = dz_all.reshape(-1, dz_all.shape[2] * dz_all.shape[3]).norm(dim=1)
    step_norms_vis = dz_vis.reshape(-1, dz_vis.shape[2] * dz_vis.shape[3]).norm(dim=1)

    def summarize_1d(x: torch.Tensor):
        return {
            "mean": float(x.mean().item()),
            "std": float(x.std(unbiased=False).item()),
            "min": float(x.min().item()),
            "max": float(x.max().item()),
            "median": float(x.median().item()),
        }

    return {
        "all_dims": summarize_1d(step_norms_all),
        "visual_only": summarize_1d(step_norms_vis),
    }
    
    

def normalize_visual_delta_to_radius(
    delta: torch.Tensor,
    visual_dim: int,
    radius: float,
) -> torch.Tensor:
    """
    Normalize only the visual part of delta to have Frobenius norm = radius
    for each batch element.
    """
    delta_visual = delta[..., :visual_dim]
    flat = delta_visual.reshape(delta_visual.shape[0], -1)
    norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat = flat / norms * radius
    delta_visual = flat.reshape_as(delta_visual)

    out = torch.zeros_like(delta)
    out[..., :visual_dim] = delta_visual
    return out


def summarize_tensor_list(lst):
    x = torch.cat(lst, dim=0)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "median": float(x.median().item()),
    }


def compute_errors_for_radius(
    model: VWorldModel,
    z0: torch.Tensor,
    y0: torch.Tensor,
    visual_dim: int,
    radius: float,
    num_samples: int,
    distribution: str,
    output_slice: str = "visual_only",
):
    if model.concat_dim != 1:
        raise NotImplementedError("This script assumes concat_dim=1.")

    total_dim = z0.shape[-1]

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

    raw_errs = []
    rel_errs = []
    err_over_d = []
    err_over_d2 = []
    delta_norms = []
    nonlinear_change_norms = []
    jvp_norms = []

    for _ in range(num_samples):
        direction = make_visual_direction(
            z_src=z0,
            visual_dim=visual_dim,
            distribution=distribution,
        )
        delta = normalize_visual_delta_to_radius(
            delta=direction,
            visual_dim=visual_dim,
            radius=radius,
        )

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
            dnorm = batch_fro_norm(delta[..., :visual_dim])
            rel = raw / (batch_fro_norm(nonlinear_change) + 1e-12)
            per_d = raw / (dnorm + 1e-12)
            per_d2 = raw / (dnorm.pow(2) + 1e-12)
            jnorm = batch_fro_norm(jvp[..., out_lo:out_hi])

            raw_errs.append(raw.cpu())
            rel_errs.append(rel.cpu())
            err_over_d.append(per_d.cpu())
            err_over_d2.append(per_d2.cpu())
            delta_norms.append(dnorm.cpu())
            nonlinear_change_norms.append(batch_fro_norm(nonlinear_change).cpu())
            jvp_norms.append(jnorm.cpu())

    return {
        "radius": float(radius),
        "raw_error": summarize_tensor_list(raw_errs),
        "relative_error": summarize_tensor_list(rel_errs),
        "error_over_delta_norm": summarize_tensor_list(err_over_d),
        "error_over_delta_norm_sq": summarize_tensor_list(err_over_d2),
        "delta_norm": summarize_tensor_list(delta_norms),
        "nonlinear_change_norm": summarize_tensor_list(nonlinear_change_norms),
        "jvp_norm": summarize_tensor_list(jvp_norms),
    }


def plot_metric_vs_radius(results, metric_key, ylabel, out_path, loglog=False,y_max = 114514):
    radii = np.array([r["delta_norm"]["mean"] for r in results], dtype=np.float64)
    means = np.array([r[metric_key]["mean"] for r in results], dtype=np.float64)
    stds = np.array([r[metric_key]["std"] for r in results], dtype=np.float64)

    plt.figure(figsize=(7, 5))
    plt.errorbar(radii, means, yerr=stds, marker="o", capsize=3)
    plt.xlabel(r"$\|\delta\|$")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs " + r"$\|\delta\|$")
    
    if (y_max != 114514):
        plt.ylim(-0.001, y_max) 

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    #data_root_768 = "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset"
    #ckpt_768 = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_5/model_0070.pt"
    
    #for 384:
    data_root_384 = "/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset"
    ckpt_384 = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_18_2/model_0020.pt"

    parser.add_argument("--data_root", type=str, default=data_root_384)
    parser.add_argument("--ckpt", type=str, default=ckpt_384)

    parser.add_argument("--n_rollout", type=int, default=10000)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--frameskip", type=int, default=1)

    parser.add_argument("--num_hist", type=int, default=1)
    parser.add_argument("--num_pred", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--num_samples_per_radius", type=int, default=100)
    parser.add_argument("--distribution", type=str, default="gaussian",
                        choices=["gaussian", "rademacher"])

    parser.add_argument("--radius_min", type=float, default=3e-3)
    parser.add_argument("--radius_max", type=float, default=2e+2)
    parser.add_argument("--num_radii", type=int, default=20)
    parser.add_argument("--radius_spacing", type=str, default="log",
                        choices=["log", "linear"])

    parser.add_argument("--output_slice", type=str, default="visual_only",
                        choices=["visual_only", "all_nonaction", "all"])

    parser.add_argument("--token_dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="linearization_radius_plots")


    args = parser.parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    model = build_model(
        device=device,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        token_dim=args.token_dim,
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

    with torch.no_grad():
        z = model.encode(obs, act)
        z_src = z[:, :model.num_hist, :, :].clone()
    
    z0 = z_src.detach().clone().requires_grad_(True)
    y0 = model.predict(z0)
    
    visual_dim = z0.shape[-1] - (model.proprio_dim + model.action_dim)
    total_dim = z0.shape[-1]
    
    step_stats = summarize_temporal_step_norms(z, visual_dim)
    print("[E||z_{t+1}-z_t|| all dims ]", step_stats["all_dims"]["mean"])
    print("[E||z_{t+1}-z_t|| visual   ]", step_stats["visual_only"]["mean"])

    print(f"[info] visual_dim={visual_dim}, total_dim={total_dim}, action_dim={model.action_dim}")

    if args.radius_spacing == "log":
        radii = np.logspace(np.log10(args.radius_min), np.log10(args.radius_max), args.num_radii)
    else:
        radii = np.linspace(args.radius_min, args.radius_max, args.num_radii)

    all_results = []
    for radius in radii:
        print(f"[radius] {radius:.6e}")
        summary = compute_errors_for_radius(
            model=model,
            z0=z0,
            y0=y0,
            visual_dim=visual_dim,
            radius=float(radius),
            num_samples=args.num_samples_per_radius,
            distribution=args.distribution,
            output_slice=args.output_slice,
        )
        all_results.append(summary)

    json_path = save_dir / "radius_sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": vars(args),
            "visual_dim": int(visual_dim),
            "total_dim": int(total_dim),
            "expected_temporal_step_norm": step_stats,
            "results": all_results,
        }, f, indent=2)
    print(f"[saved] {json_path}")

    plot_metric_vs_radius(
        all_results,
        metric_key="raw_error",
        ylabel="raw error",
        out_path=save_dir / "raw_error_vs_delta.png",
        loglog=True,
        
    )

    plot_metric_vs_radius(
        all_results,
        metric_key="relative_error",
        ylabel="relative error",
        out_path=save_dir / "relative_error_vs_delta.png",
        loglog=False,
        y_max = 0.1,
    )

    plot_metric_vs_radius(
        all_results,
        metric_key="error_over_delta_norm",
        ylabel=r"error / $\|\delta\|$",
        out_path=save_dir / "error_over_delta_vs_delta.png",
        loglog=False,
        y_max = 0.025
    )

    plot_metric_vs_radius(
        all_results,
        metric_key="error_over_delta_norm_sq",
        ylabel=r"error / $\|\delta\|^2$",
        out_path=save_dir / "error_over_delta_sq_vs_delta.png",
        loglog=False,
        y_max = 0.05
    )

    print("[done] plots saved in", save_dir)


if __name__ == "__main__":
    main()