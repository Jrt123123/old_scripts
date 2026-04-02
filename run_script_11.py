#!/usr/bin/env python3
"""
cem_plan_unicycle_patchtokens.py

CEM planning for your unicycle patch-token world model.

Pipeline:
  (x,y,theta) -> render teardrop image -> DINO -> patchtokens
  use trained DINO-WM-style predictor to "imagine" rollout
  CEM optimizes action sequence to match goal patchtokens (last frame)
  then roll out best actions in *real* unicycle dynamics and plot XY

Notes / assumptions:
- Your checkpoint is from the training script you pasted (ckpt_best.pt etc).
- Your DINO wrapper provides: extract_cls_and_patches(images, batch_size=...)
  (as in your dataset generator).
- The patch-token world model was trained on patchtokens with history length H
  (ckpt cfg has H). For planning, we build the initial history by repeating
  the start patchtokens H times.

- Autoregressive rollout:
  We use the predictor as a "1-step" model by feeding the current frame
  repeated H times and the current action repeated H times, then taking the
  last predicted frame. (This is the most robust way without relying on
  VWorldModel.rollout internals.)

Run example:
  python cem_plan_unicycle_patchtokens.py \
    --ckpt /storage/scratch1/6/rjiang77/Autoencoder/2_22_1/ckpt_best.pt \
    --x0 60 --y0 60 --th0 0.2 \
    --xg 90 --yg 80 --thg -0.5 \
    --horizon 12 --num_samples 512 --topk 64 --opt_steps 8 \
    --v_min 3 --v_max 15 --w_min -0.6 --w_max 0.6 \
    --out_dir cem_out
"""

from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- your copied DINO-WM code
from DINO_WM_vit import ViTPredictor
from DINO_WM_visual_world_model import VWorldModel

# --- your DINO wrapper (same one used in dataset generator)
# must exist in your repo/PYTHONPATH
from DINO_vit_patch import DINOv2_ViT as Dinov2_VIT

from skimage.draw import polygon
from skimage.transform import resize


# -------------------------
# Rendering (copied from your dataset generator)
# -------------------------

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


# -------------------------
# Minimal adapters (same as your eval script)
# -------------------------

class PatchTokenEncoder(nn.Module):
    def __init__(self, emb_dim: int, patch_size: int = 14, name: str = "dino"):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.latent_ndim = 2
        self.name = name


class MLPSeqEncoder(nn.Module):
    def __init__(self, in_chans: int, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class ModelCfg:
    H: int
    action_emb_dim: int
    num_action_repeat: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int


def build_core_model(P: int, D: int, cfg: ModelCfg, device: torch.device) -> VWorldModel:
    encoder = PatchTokenEncoder(emb_dim=D, patch_size=14, name="dino")
    proprio_encoder = MLPSeqEncoder(in_chans=1, emb_dim=1)  # unused (proprio_dim=0)
    action_encoder = MLPSeqEncoder(in_chans=2, emb_dim=cfg.action_emb_dim)

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

    core = VWorldModel(
        image_size=224,
        num_hist=cfg.H,
        num_pred=1,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=0,
        action_dim=cfg.action_emb_dim,   # IMPORTANT: embedded-action dim (your convention)
        concat_dim=1,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=0,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ).to(device)

    core.eval()
    return core


# -------------------------
# DINO patchtokens for a single state
# -------------------------

@torch.no_grad()
def state_to_patchtokens(
    dinov2: Dinov2_VIT,
    x: float, y: float, th: float,
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
    # extract_cls_and_patches returns (cls:(T,D), patches:(T,P,D)) when fed a list of images
    _, patches = dinov2.extract_cls_and_patches([img], batch_size=dino_batch_size)
    # patches: (1, P, D) in torch
    return patches[0]  # (P, D)


# -------------------------
# World-model "1-step" prediction
# -------------------------

@torch.no_grad()
def predict_next_patchtokens_1step(core: VWorldModel, z_cur: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Treat the H-frame predictor as a 1-step model by repeating the current frame H times,
    and repeating action H times, then taking the last predicted frame.

    z_cur: (B,P,D)
    u:     (B,2) raw (v,w)
    returns: z_next (B,P,D)
    """
    device = next(core.parameters()).device
    z_cur = z_cur.to(device)
    u = u.to(device)

    B, P, D = z_cur.shape
    H = core.num_hist

    # build fake history (B,H,P,D) all identical
    z_hist = z_cur.unsqueeze(1).repeat(1, H, 1, 1)

    # actions (B,H,2) all identical
    u_hist = u.unsqueeze(1).repeat(1, H, 1)

    act_emb = core.encode_act(u_hist)  # (B,H,action_emb_dim)

    act_tiled = act_emb.unsqueeze(2).expand(B, H, P, act_emb.shape[-1])
    act_rep = act_tiled.repeat(1, 1, 1, core.num_action_repeat)

    z_src = torch.cat([z_hist, act_rep], dim=-1)  # (B,H,P,D+actrep)
    z_pred = core.predict(z_src)                  # (B,H,P,D+actrep)

    #strip = core.action_dim * core.num_action_repeat
    strip = core.action_dim
    z_pred_visual = z_pred[..., :-strip]          # (B,H,P,D)
    
    print("z_cur", z_cur.shape, z_cur.dtype, z_cur.device)
    print("act_rep", act_rep.shape, act_rep.dtype, act_rep.device)
    print("z_src", z_src.shape, z_src.dtype, z_src.device)

    return z_pred_visual[:, -1]                   # (B,P,D)


@torch.no_grad()
def rollout_wm(core: VWorldModel, z0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Autoregressive rollout using predict_next_patchtokens_1step.

    z0: (B,P,D)
    U:  (B,T,2)
    returns zT: (B,P,D) final predicted patchtokens
    """
    z = z0
    T = U.shape[1]
    for t in range(T):
        z = predict_next_patchtokens_1step(core, z, U[:, t])
    return z


# -------------------------
# Objective: last-frame MSE to goal patchtokens
# (same spirit as your create_objective_fn(..., mode="last")).
# -------------------------

@torch.no_grad()
def objective_last_mse(z_final: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    z_final: (B,P,D)
    z_goal:  (B,P,D) or (P,D)
    returns loss: (B,)
    """
    if z_goal.ndim == 2:
        z_goal = z_goal.unsqueeze(0).expand(z_final.shape[0], -1, -1)
    return ((z_final - z_goal) ** 2).mean(dim=(1, 2))


# -------------------------
# CEM
# -------------------------

@torch.no_grad()
def cem_plan(
    core: VWorldModel,
    z0: torch.Tensor,          # (1,P,D)
    zg: torch.Tensor,          # (1,P,D)
    horizon: int,
    num_samples: int,
    topk: int,
    opt_steps: int,
    var_scale: float,
    v_min: float, v_max: float,
    w_min: float, w_max: float,
    seed: int = 0,
):
    """
    Returns:
      u_best: (T,2) on CPU
      info: dict
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = next(core.parameters()).device
    z0 = z0.to(device)
    zg = zg.to(device)

    # mu/sigma over (T,2)
    mu = torch.zeros(horizon, 2, device=device)
    sigma = torch.ones(horizon, 2, device=device) * float(var_scale)

    best_u = None
    best_cost = float("inf")
    hist = []

    for it in range(opt_steps):
        eps = torch.randn(num_samples, horizon, 2, device=device)
        U = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)  # (N,T,2)

        # clamp to action bounds
        U[..., 0] = U[..., 0].clamp(v_min, v_max)
        U[..., 1] = U[..., 1].clamp(w_min, w_max)

        # optional: force first sample to be mu (common trick)
        U[0] = mu
        U[0, :, 0] = U[0, :, 0].clamp(v_min, v_max)
        U[0, :, 1] = U[0, :, 1].clamp(w_min, w_max)

        # evaluate in batches to avoid OOM
        batch = 64
        costs = []
        for s in range(0, num_samples, batch):
            e = min(num_samples, s + batch)
            zT = rollout_wm(core, z0.expand(e - s, -1, -1), U[s:e])
            c = objective_last_mse(zT, zg)  # (B,)
            costs.append(c)
        costs = torch.cat(costs, dim=0)  # (N,)

        idx = torch.argsort(costs)[:topk]
        elite = U[idx]  # (K,T,2)
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp_min(1e-6)

        elite_best = float(costs[idx[0]].item())
        hist.append(elite_best)

        if elite_best < best_cost:
            best_cost = elite_best
            best_u = elite[0].detach().cpu()

        print(f"[CEM] iter {it+1}/{opt_steps}  best_elite_cost={elite_best:.6f}")

    info = {"best_cost": best_cost, "best_cost_history": hist}
    return best_u, info


# -------------------------
# Real unicycle dynamics rollout (XY plot)
# -------------------------

def unicycle_step(x, y, th, v, w, dt):
    x2 = x + dt * v * math.cos(th)
    y2 = y + dt * v * math.sin(th)
    th2 = th + dt * w
    return x2, y2, th2


def denormalize_v(v_n, xbound):
    low, high = xbound
    return float(v_n) * ((high - low) / 2.0)


def rollout_real_unicycle(x0, y0, th0, U, dt, xbound):
    """
    U: (T,2) where each row is (v_n, w)
       v_n is normalized via v_n = 2*v/(xhigh-xlow)
       w is already in real units (same as used in generator)
    """
    xs = [x0]
    ys = [y0]
    ths = [th0]
    x, y, th = x0, y0, th0

    for v_n, w in U:
        v = denormalize_v(v_n, xbound)   # <-- key change
        x, y, th = unicycle_step(x, y, th, v, float(w), dt)
        xs.append(x); ys.append(y); ths.append(th)

    return np.array(xs), np.array(ys), np.array(ths)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="/storage/scratch1/6/rjiang77/Autoencoder/2_22_1/ckpt_best.pt")
    ap.add_argument("--device", type=str, default="cuda")

    # start/goal states
    ap.add_argument("--x0", type=float, default=90)
    ap.add_argument("--y0", type=float, default=90)
    ap.add_argument("--th0", type=float, default=3.14)
    ap.add_argument("--xg", type=float, default=40)
    ap.add_argument("--yg", type=float, default=40)
    ap.add_argument("--thg", type=float, default=4.6)

    # render / DINO
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--dino_model_name", type=str, default="dinov2_vitb14")
    ap.add_argument("--dino_resize_size", type=int, default=224)

    # CEM
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num_samples", type=int, default=512)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--opt_steps", type=int, default=16)
    ap.add_argument("--var_scale", type=float, default=2.0)

    # action bounds (match your generator’s typical scale)
    ap.add_argument("--v_min", type=float, default=0.9)
    ap.add_argument("--v_max", type=float, default=4.5)
    ap.add_argument("--w_min", type=float, default=-0.6)
    ap.add_argument("--w_max", type=float, default=0.6)

    # real rollout
    ap.add_argument("--dt", type=float, default=0.1)

    ap.add_argument("--out_dir", type=str, default="cem_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[device]", device)

    # ---- load ckpt, infer model cfg
    ckpt = torch.load(args.ckpt, map_location="cpu")


    ckpt_cfg = ckpt.get("cfg", {})

    H = int(ckpt_cfg.get("H", 3))
    action_emb_dim = int(ckpt_cfg.get("action_emb_dim", 10))
    num_action_repeat = int(ckpt_cfg.get("num_action_repeat", 7))
    depth = int(ckpt_cfg.get("depth", 6))
    heads = int(ckpt_cfg.get("heads", 16))
    mlp_dim = int(ckpt_cfg.get("mlp_dim", 2048))
    dim_head = int(ckpt_cfg.get("dim_head", 64))

    # ---- init DINO and get (P,D)
    dinov2 = Dinov2_VIT(model_name=args.dino_model_name, resize_size=args.dino_resize_size)
    z0_pd = state_to_patchtokens(dinov2, args.x0, args.y0, args.th0, args.image_size, args.radius)  # (P,D)
    P, D = int(z0_pd.shape[0]), int(z0_pd.shape[1])
    print(f"[DINO] P={P} D={D}")
    
    
    
    # What the predictor in the checkpoint expects
    pos = ckpt["model"]["predictor.pos_embedding"]          # shape: (1, N, dim)
    dim_ckpt = pos.shape[-1]
    n_tokens = pos.shape[1]                                # usually = H*P (+ maybe extras)
    print("[ckpt] predictor dim =", dim_ckpt, "pos_tokens =", n_tokens)
    
    # What your runtime is trying to feed
    print("[runtime] DINO D =", D, "action_rep =", action_emb_dim*num_action_repeat,
          "sum =", D + action_emb_dim*num_action_repeat)
          

    # ---- build/load world model
    mcfg = ModelCfg(H=H, action_emb_dim=action_emb_dim, num_action_repeat=num_action_repeat,
                    depth=depth, heads=heads, mlp_dim=mlp_dim, dim_head=dim_head)
    core = build_core_model(P=P, D=D, cfg=mcfg, device=device)
    core.load_state_dict(ckpt["model"], strict=True)
    core.eval()

    # ---- goal patchtokens
    zg_pd = state_to_patchtokens(dinov2, args.xg, args.yg, args.thg, args.image_size, args.radius)  # (P,D)

    # ---- planning inputs (B=1)
    z0 = z0_pd.unsqueeze(0)
    zg = zg_pd.unsqueeze(0)

    # ---- CEM plan
    u_best, info = cem_plan(
        core=core,
        z0=z0,
        zg=zg,
        horizon=args.horizon,
        num_samples=args.num_samples,
        topk=args.topk,
        opt_steps=args.opt_steps,
        var_scale=args.var_scale,
        v_min=args.v_min, v_max=args.v_max,
        w_min=args.w_min, w_max=args.w_max,
        seed=args.seed,
    )

    np.save(out_dir / "u_best.npy", u_best.numpy())
    with open(out_dir / "cem_info.txt", "w") as f:
        f.write(str(info) + "\n")
    print(f"[saved] {out_dir/'u_best.npy'}")
    print(f"[saved] {out_dir/'cem_info.txt'}")

    # ---- real dynamics rollout + plot
    xbound = (35, 102)
    xs, ys, ths = rollout_real_unicycle(args.x0, args.y0, args.th0, u_best.numpy(), args.dt,xbound)

    plt.figure()
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.scatter([args.x0], [args.y0], s=80, marker="o", label="start")
    plt.scatter([args.xg], [args.yg], s=80, marker="x", label="goal")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Real unicycle rollout (XY) using CEM planned controls")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "xy_rollout.png", dpi=200)
    plt.close()
    print(f"[saved] {out_dir/'xy_rollout.png'}")

    # also plot cost history
    plt.figure()
    plt.plot(np.array(info["best_cost_history"]), marker="o")
    plt.xlabel("CEM iter")
    plt.ylabel("best elite cost")
    plt.title("CEM optimization progress")
    plt.tight_layout()
    plt.savefig(out_dir / "cem_cost_history.png", dpi=200)
    plt.close()
    print(f"[saved] {out_dir/'cem_cost_history.png'}")


if __name__ == "__main__":
    main()