#!/usr/bin/env python3
"""
cem_plan_unicycle_patchtokens_v2.py

CEM planning for your unicycle patch-token world model.

Key difference from the earlier draft:
- This version follows your VWorldModel.rollout() semantics more closely.
- It keeps a rolling latent history z, predicts from z[:, -H:], takes the newest
  predicted frame, inserts the chosen action into that frame with
  replace_actions_from_z(), and appends it.

Assumptions:
- You already have:
    1) a checkpoint from your training script
    2) a renderer from (x,y,theta) -> teardrop image
    3) a DINO wrapper that returns patch tokens (P,D) for one image
- Actions are raw (v, w), consistent with how your dataset stored "controls".
"""

from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import sys
from pathlib import Path

FILE = Path(__file__).resolve()
CUR_DIR = FILE.parent
REPO_ROOT = CUR_DIR.parent

sys.path.insert(0, str(CUR_DIR))
sys.path.insert(0, str(REPO_ROOT))


import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# your modules
from Predictor import ViTPredictor
from visual_world_model import VWorldModel
from ControlEncoderSingle import ControlEncoderSingle
# if you used MLP instead, switch import/use below
# from ControlEncoderMLP import ControlEncoderMLP

# your DINO wrapper
from DINO_vit_patch import DINOv2_ViT as Dinov2_VIT

from skimage.draw import polygon
from skimage.transform import resize


def denormalize_v_batch(v_n, xbound):
    low, high = xbound
    return v_n * ((high - low) / 2.0)
    
def denormalize_v(v_n, xbound):
    low, high = xbound
    return float(v_n) * ((high - low) / 2.0)


# ============================================================
# Rendering: keep consistent with your dataset generator
# ============================================================

def generate_teardrop_mask(
    center=(10, 112),
    radius=10,
    theta=0,
    image_size=224,
    arc_resolution=200,
    tip_scale=3.0,
    upscale_factor=4,
):
    cx, cy = center
    high_res_size = image_size * upscale_factor
    high_cx, high_cy = cx * upscale_factor, cy * upscale_factor
    high_radius = radius * upscale_factor

    arc_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, arc_resolution)
    arc = np.stack(
        [high_radius * np.cos(arc_angles), high_radius * np.sin(arc_angles)], axis=1
    )
    tip = np.array([[tip_scale * high_radius, 0.0]])
    shape = np.vstack([arc[::-1], tip])

    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotated = shape @ R.T
    translated = rotated + np.array([high_cx, high_cy])

    high_img = np.ones((high_res_size, high_res_size), dtype=np.float32)
    rr, cc = polygon(translated[:, 1], translated[:, 0], shape=high_img.shape)
    high_img[rr, cc] = 0.0

    img = resize(
        high_img, (image_size, image_size), anti_aliasing=True, preserve_range=True
    )
    return img.astype(np.float32)


# ============================================================
# Minimal encoder stubs for the world model
# ============================================================

class PatchTokenEncoder(nn.Module):
    """
    Identity encoder: input already is patch tokens (P,D) or (B*T,P,D).
    """
    def __init__(self, emb_dim: int, patch_size: int = 16, name: str = "dummy_encoder"):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.name = name

    def forward(self, x):
        return x


class DummyProprio(nn.Module):
    """
    Proprio path not used for your patch-token-only predictor build.
    """
    def forward(self, x):
        return x


@dataclass
class ModelCfg:
    H: int
    action_emb_dim: int
    num_action_repeat: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int = 64


def build_core_model(P: int, D: int, cfg: ModelCfg, device: torch.device) -> VWorldModel:
    encoder = PatchTokenEncoder(emb_dim=D, patch_size=16, name="dummy_encoder")
    proprio_encoder = DummyProprio()
    action_encoder = ControlEncoderSingle(action_dim=2, emb_dim=cfg.action_emb_dim)

    predictor_dim = D + cfg.action_emb_dim * cfg.num_action_repeat
    predictor = ViTPredictor(
        num_patches=P,
        num_frames=cfg.H,
        dim=predictor_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_dim=cfg.mlp_dim,
        dim_head=cfg.dim_head,
        dropout=0.0,
        emb_dropout=0.0,
    )

    model = VWorldModel(
        image_size=224,
        num_hist=cfg.H,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=0,
        action_dim=cfg.action_emb_dim,   # IMPORTANT: un-repeated emb dim
        concat_dim=1,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=7,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ).to(device)

    model.eval()
    return model


# ============================================================
# State -> patch tokens
# ============================================================

@torch.no_grad()
def state_to_patchtokens(
    dinov2: Dinov2_VIT,
    x: float,
    y: float,
    th: float,
    image_size: int,
    radius: int,
    dino_batch_size: int = 1,
) -> torch.Tensor:
    img = generate_teardrop_mask(
        center=(float(x), float(y)),
        radius=radius,
        theta=float(th),
        image_size=image_size,
    )
    _, patches = dinov2.extract_cls_and_patches([img], batch_size=dino_batch_size)
    # expected: (1, P, D)
    return patches[0]


# ============================================================
# Latent initialization / rollout
# ============================================================

@torch.no_grad()
def make_initial_latent_history(
    core: VWorldModel,
    z0_visual: torch.Tensor,     # (1,P,D)
    init_action: torch.Tensor,   # (1,2)
) -> torch.Tensor:
    """
    Build an initial latent history of length H.
    Since we only have one start state, we repeat it H times.
    We also assign the same init_action to each repeated frame so the latent
    shape matches what the predictor expects.
    Returns z: (1,H,P,D_total)
    """
    B, P, D = z0_visual.shape
    H = core.num_hist
    device = z0_visual.device

    # repeat visual tokens H times
    z_vis = z0_visual.unsqueeze(1).repeat(1, H, 1, 1)  # (1,H,P,D)

    # build repeated action features exactly as in your model
    act = init_action.unsqueeze(1).repeat(1, H, 1)     # (1,H,2)
    act_emb = core.encode_act(act)                     # (1,H,action_emb_dim)
    act_tiled = act_emb.unsqueeze(2).expand(B, H, P, act_emb.shape[-1])
    act_rep = act_tiled.repeat(1, 1, 1, core.num_action_repeat)

    # concat_dim=1 model: z = [visual, action_repeated]
    z = torch.cat([z_vis, act_rep], dim=-1)
    return z


@torch.no_grad()
def rollout_latent_with_actions(
    core: VWorldModel,
    z_init: torch.Tensor,   # (1,H,P,D_total)
    U: torch.Tensor,        # (B,T,2)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Roll out candidate controls using the same logic as VWorldModel.rollout():
      z_pred = predict(z[:, -H:])
      z_new  = z_pred[:, -1:]
      z_new  = replace_actions_from_z(z_new, u_t)
      z      = cat([z, z_new], dim=1)

    Returns:
      z_final_visual : (B,P,D_visual)
      z_all          : (B,H+T,P,D_total) appended rollout
    """
    device = next(core.parameters()).device
    z_init = z_init.to(device)
    U = U.to(device)

    B = U.shape[0]
    T = U.shape[1]

    z = z_init.expand(B, -1, -1, -1).clone()   # (B,H,P,D_total)

    for t in range(T):
        z_pred = core.predict(z[:, -core.num_hist:])          # (B,H,P,D_total)
        z_new = z_pred[:, -1:, ...]                           # (B,1,P,D_total)
        z_new = core.replace_actions_from_z(z_new, U[:, t:t+1, :])
        z = torch.cat([z, z_new], dim=1)

    # one more predict to get the "next visual" after the final action-conditioned frame
    z_pred = core.predict(z[:, -core.num_hist:])
    z_next = z_pred[:, -1, ...]                               # (B,P,D_total)

    # strip repeated action feature tail; self.action_dim already includes repetition
    z_next_visual = z_next[..., :-core.action_dim]            # (B,P,D_visual)

    return z_next_visual, z


# ============================================================
# Cost
# ============================================================

@torch.no_grad()
def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


@torch.no_grad()
def objective_patch_mse(z_final: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    z_final: (B,P,D)
    z_goal:  (1,P,D) or (B,P,D)
    """
    if z_goal.ndim == 2:
        z_goal = z_goal.unsqueeze(0)
    if z_goal.shape[0] == 1 and z_final.shape[0] > 1:
        z_goal = z_goal.expand(z_final.shape[0], -1, -1)
    return ((z_final - z_goal) ** 2).mean(dim=(1, 2))


@torch.no_grad()
def objective_mixed(
    z_final: torch.Tensor,          # (B,P,D)
    z_goal: torch.Tensor,           # (1,P,D)
    xT: np.ndarray,                 # (B,)
    yT: np.ndarray,                 # (B,)
    thT: np.ndarray,                # (B,)
    xg: float,
    yg: float,
    thg: float,
    lambda_state: float = 0.0,
    lambda_theta: float = 0.0,
) -> torch.Tensor:
    """
    Mostly patch-token matching, optionally with a small terminal state penalty from
    real unicycle rollout.
    """
    c_patch = objective_patch_mse(z_final, z_goal)

    if lambda_state == 0.0 and lambda_theta == 0.0:
        return c_patch

    xT_t = torch.as_tensor(xT, dtype=torch.float32, device=z_final.device)
    yT_t = torch.as_tensor(yT, dtype=torch.float32, device=z_final.device)
    thT_t = torch.as_tensor(thT, dtype=torch.float32, device=z_final.device)

    pos_cost = (xT_t - xg) ** 2 + (yT_t - yg) ** 2
    th_cost = angle_wrap(thT_t - thg) ** 2

    return c_patch + lambda_state * pos_cost + lambda_theta * th_cost


# ============================================================
# Real unicycle rollout
# ============================================================

def unicycle_step(x, y, th, v, w, dt):
    x2 = x + dt * v * math.cos(th)
    y2 = y + dt * v * math.sin(th)
    th2 = th + dt * w
    return x2, y2, th2


def rollout_real_batch(x0, y0, th0, U, dt, xbound):
    """
    U: (B,T,2), where controls are (v_n, w)
    returns xT, yT, thT: each (B,)
    """
    B, T, _ = U.shape
    xs = np.full(B, float(x0), dtype=np.float32)
    ys = np.full(B, float(y0), dtype=np.float32)
    ths = np.full(B, float(th0), dtype=np.float32)

    for t in range(T):
        v_n = U[:, t, 0]
        w = U[:, t, 1]
        v = denormalize_v_batch(v_n, xbound)

        xs = xs + dt * v * np.cos(ths)
        ys = ys + dt * v * np.sin(ths)
        ths = ths + dt * w

    return xs, ys, ths


def rollout_real_single(x0, y0, th0, U, dt, xbound):
    """
    U: (T,2), where each row is (v_n, w)
    """
    xs = [x0]
    ys = [y0]
    ths = [th0]
    x, y, th = x0, y0, th0

    for v_n, w in U:
        v = denormalize_v(v_n, xbound)
        x, y, th = unicycle_step(x, y, th, v, float(w), dt)
        xs.append(x)
        ys.append(y)
        ths.append(th)

    return np.array(xs), np.array(ys), np.array(ths)


# ============================================================
# CEM
# ============================================================

@torch.no_grad()
def cem_plan(
    core: VWorldModel,
    z0_visual: torch.Tensor,       # (1,P,D)
    zg_visual: torch.Tensor,       # (1,P,D)
    x0: float,
    y0: float,
    th0: float,
    xg: float,
    yg: float,
    thg: float,
    horizon: int,
    num_samples: int,
    topk: int,
    opt_steps: int,
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
    init_v_mean: float,
    init_w_mean: float,
    init_v_std: float,
    init_w_std: float,
    dt: float,
    seed: int = 0,
    lambda_state: float = 0.0,
    lambda_theta: float = 0.0,
    eval_batch_size: int = 128,
):
    xbound = (35, 102)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(core.parameters()).device
    z0_visual = z0_visual.to(device)
    zg_visual = zg_visual.to(device)

    mu = torch.zeros(horizon, 2, device=device)
    sigma = torch.zeros(horizon, 2, device=device)

    mu[:, 0] = init_v_mean
    mu[:, 1] = init_w_mean
    sigma[:, 0] = init_v_std
    sigma[:, 1] = init_w_std

    best_u = None
    best_cost = float("inf")
    best_cost_history = []

    # for initial latent history, use first mean action
    init_action = mu[0:1, :].clone()                  # (1,2)
    z_init = make_initial_latent_history(core, z0_visual, init_action)

    for it in range(opt_steps):
        eps = torch.randn(num_samples, horizon, 2, device=device)
        U = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps

        U[..., 0] = U[..., 0].clamp(v_min, v_max)
        U[..., 1] = U[..., 1].clamp(w_min, w_max)

        # force one sample to current mean
        U[0] = mu
        U[0, :, 0] = U[0, :, 0].clamp(v_min, v_max)
        U[0, :, 1] = U[0, :, 1].clamp(w_min, w_max)

        costs = []

        for s in range(0, num_samples, eval_batch_size):
            e = min(num_samples, s + eval_batch_size)
            U_batch = U[s:e]

            z_final, _ = rollout_latent_with_actions(core, z_init, U_batch)

            xT, yT, thT = rollout_real_batch(
                x0=x0, y0=y0, th0=th0,
                U=U_batch.detach().cpu().numpy(),
                dt=dt,
                xbound=(35, 102),
            )

            c = objective_mixed(
                z_final=z_final,
                z_goal=zg_visual,
                xT=xT, yT=yT, thT=thT,
                xg=xg, yg=yg, thg=thg,
                lambda_state=lambda_state,
                lambda_theta=lambda_theta,
            )
            costs.append(c)

        costs = torch.cat(costs, dim=0)               # (N,)
        elite_idx = torch.argsort(costs)[:topk]
        elite = U[elite_idx]

        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp_min(1e-4)

        elite_best = float(costs[elite_idx[0]].item())
        best_cost_history.append(elite_best)

        if elite_best < best_cost:
            best_cost = elite_best
            best_u = elite[0].detach().cpu().clone()

        print(
            f"[CEM] iter {it+1:02d}/{opt_steps} | "
            f"best={elite_best:.6f} | "
            f"v_mu=({mu[:,0].mean().item():.3f}) | "
            f"w_mu=({mu[:,1].mean().item():.3f})"
        )

    info = {
        "best_cost": best_cost,
        "best_cost_history": best_cost_history,
    }
    return best_u, info


# ============================================================
# Main
# ============================================================

def main():
    #ckpt_path = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_5/model_0072.pt"    #768d
    ckpt_path = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_14_1/model_0098.pt"      #384d 
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default = ckpt_path)
    ap.add_argument("--device", type=str, default="cuda")

    # start / goal
    ap.add_argument("--x0", type=float, default = 90)
    ap.add_argument("--y0", type=float, default = 40)
    ap.add_argument("--th0", type=float, default = 3.14)
    ap.add_argument("--xg", type=float, default = 40)
    ap.add_argument("--yg", type=float, default = 90)
    ap.add_argument("--thg", type=float, default = 3.14)

    # rendering / DINO
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    ap.add_argument("--dino_resize_size", type=int, default=224)

    # planning
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--opt_steps", type=int, default=20)

    # action bounds
    ap.add_argument("--v_min", type=float, default=0.2)
    ap.add_argument("--v_max", type=float, default=5.0)
    ap.add_argument("--w_min", type=float, default=-5)
    ap.add_argument("--w_max", type=float, default=5)

    # initial sampling distribution
    ap.add_argument("--init_v_mean", type=float, default=2.0)
    ap.add_argument("--init_w_mean", type=float, default=0.0)
    ap.add_argument("--init_v_std", type=float, default=1)
    ap.add_argument("--init_w_std", type=float, default=3)

    # dynamics
    ap.add_argument("--dt", type=float, default=0.1)

    # optional terminal state regularization
    ap.add_argument("--lambda_state", type=float, default=0.0)
    ap.add_argument("--lambda_theta", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=113)
    ap.add_argument("--out_dir", type=str, default="cem_out_384")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print("[device]", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    # ---------- infer config from checkpoint if available ----------
    cfg = ckpt.get("cfg", {})
    H = int(cfg.get("H", 1))
    action_emb_dim = int(cfg.get("action_emb_dim", 10))
    num_action_repeat = int(cfg.get("num_action_repeat", 7))
    depth = int(cfg.get("depth", 6))
    heads = int(cfg.get("heads", 16))
    mlp_dim = int(cfg.get("mlp_dim", 2048))
    dim_head = int(cfg.get("dim_head", 64))

    # ---------- DINO ----------
    dinov2 = Dinov2_VIT(
        model_name=args.dino_model_name,
        resize_size=args.dino_resize_size,
    )

    z0_pd = state_to_patchtokens(
        dinov2, args.x0, args.y0, args.th0, args.image_size, args.radius
    )
    zg_pd = state_to_patchtokens(
        dinov2, args.xg, args.yg, args.thg, args.image_size, args.radius
    )

    P, D = int(z0_pd.shape[0]), int(z0_pd.shape[1])
    print(f"[tokens] P={P}, D={D}")

    # sanity-check checkpoint predictor dim if present
    if "model" in ckpt and "predictor.pos_embedding" in ckpt["model"]:
        pe = ckpt["model"]["predictor.pos_embedding"]
        print("[ckpt] predictor.pos_embedding:", tuple(pe.shape))
        print(
            "[expected predictor dim]",
            D + action_emb_dim * num_action_repeat
        )

    # ---------- build model ----------
    mcfg = ModelCfg(
        H=H,
        action_emb_dim=action_emb_dim,
        num_action_repeat=num_action_repeat,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
    )

    core = build_core_model(P=P, D=D, cfg=mcfg, device=device)

    missing, unexpected = core.load_state_dict(ckpt["model"], strict=False)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)
    core.eval()

    z0 = z0_pd.unsqueeze(0).to(device)
    zg = zg_pd.unsqueeze(0).to(device)

    # ---------- plan ----------
    u_best, info = cem_plan(
        core=core,
        z0_visual=z0,
        zg_visual=zg,
        x0=args.x0,
        y0=args.y0,
        th0=args.th0,
        xg=args.xg,
        yg=args.yg,
        thg=args.thg,
        horizon=args.horizon,
        num_samples=args.num_samples,
        topk=args.topk,
        opt_steps=args.opt_steps,
        v_min=args.v_min,
        v_max=args.v_max,
        w_min=args.w_min,
        w_max=args.w_max,
        init_v_mean=args.init_v_mean,
        init_w_mean=args.init_w_mean,
        init_v_std=args.init_v_std,
        init_w_std=args.init_w_std,
        dt=args.dt,
        seed=args.seed,
        lambda_state=args.lambda_state,
        lambda_theta=args.lambda_theta,
    )

    np.save(out_dir / "u_best.npy", u_best.numpy())
    with open(out_dir / "cem_info.txt", "w") as f:
        f.write(str(info) + "\n")

    print("[saved]", out_dir / "u_best.npy")
    print("[saved]", out_dir / "cem_info.txt")

    # ---------- real rollout ----------
    xs, ys, ths = rollout_real_single(
        args.x0, args.y0, args.th0,
        u_best.numpy(),
        args.dt,
        xbound=(35, 102),
    )

    plt.figure()
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.scatter([args.x0], [args.y0], s=80, marker="o", label="start")
    plt.scatter([args.xg], [args.yg], s=80, marker="x", label="goal")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Real unicycle rollout from CEM plan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "xy_rollout.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.array(info["best_cost_history"]), marker="o")
    plt.xlabel("CEM iter")
    plt.ylabel("best elite cost")
    plt.title("CEM progress")
    plt.tight_layout()
    plt.savefig(out_dir / "cem_cost_history.png", dpi=200)
    plt.close()

    print("[saved]", out_dir / "xy_rollout.png")
    print("[saved]", out_dir / "cem_cost_history.png")


if __name__ == "__main__":
    main()