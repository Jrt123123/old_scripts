#!/usr/bin/env python3
"""
gd_plan_unicycle_patchtokens_v2.py

GD planning for your unicycle patch-token world model.

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
from experiments.dino_wm.models.Predictor import ViTPredictor
from experiments.dino_wm.models.visual_world_model import VWorldModel
from experiments.dino_wm.models.ControlEncoderSingle import ControlEncoderSingle
# if you used MLP instead, switch import/use below
# from ControlEncoderMLP import ControlEncoderMLP

# your DINO wrapper
from experiments.dino_wm.models.DINO_vit_patch import DINOv2_ViT as Dinov2_VIT

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



def make_initial_latent_history_diff(
    core: VWorldModel,
    z0_visual: torch.Tensor,   # (1,P,D)
    init_action: torch.Tensor, # (1,2)
) -> torch.Tensor:
    B, P, D = z0_visual.shape
    H = core.num_hist

    z_vis = z0_visual.unsqueeze(1).repeat(1, H, 1, 1)  # (1,H,P,D)

    act = init_action.unsqueeze(1).repeat(1, H, 1)     # (1,H,2)
    act_emb = core.encode_act(act)                     # (1,H,action_emb_dim)
    act_tiled = act_emb.unsqueeze(2).expand(B, H, P, act_emb.shape[-1])
    act_rep = act_tiled.repeat(1, 1, 1, core.num_action_repeat)

    z = torch.cat([z_vis, act_rep], dim=-1)
    return z


def rollout_latent_with_actions_diff(
    core: VWorldModel,
    z_init: torch.Tensor,   # (1,H,P,D_total)
    U: torch.Tensor,        # (B,T,2)
) -> tuple[torch.Tensor, torch.Tensor]:
    B = U.shape[0]
    z = z_init.expand(B, -1, -1, -1).clone()

    for t in range(U.shape[1]):
        z_pred = core.predict(z[:, -core.num_hist:])      # (B,H,P,D_total)
        z_new = z_pred[:, -1:, ...]                       # (B,1,P,D_total)
        z_new = core.replace_actions_from_z(z_new, U[:, t:t+1, :])
        z = torch.cat([z, z_new], dim=1)

    z_pred = core.predict(z[:, -core.num_hist:])
    z_next = z_pred[:, -1, ...]                           # (B,P,D_total)

    # remove repeated action tail
    z_next_visual = z_next[..., :-core.action_dim]
    return z_next_visual, z


def objective_patch_mse_diff(
    z_final: torch.Tensor,   # (B,P,D)
    z_goal: torch.Tensor,    # (1,P,D) or (B,P,D)
) -> torch.Tensor:
    if z_goal.ndim == 2:
        z_goal = z_goal.unsqueeze(0)
    if z_goal.shape[0] == 1 and z_final.shape[0] > 1:
        z_goal = z_goal.expand(z_final.shape[0], -1, -1)
    return ((z_final - z_goal) ** 2).mean(dim=(1, 2))
    
    
    
def squash_controls(
    u_raw: torch.Tensor,   # (T,2)
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
) -> torch.Tensor:
    v_mid = 0.5 * (v_max + v_min)
    v_half = 0.5 * (v_max - v_min)

    w_mid = 0.5 * (w_max + w_min)
    w_half = 0.5 * (w_max - w_min)

    u = torch.empty_like(u_raw)
    u[:, 0] = v_mid + v_half * torch.tanh(u_raw[:, 0])
    u[:, 1] = w_mid + w_half * torch.tanh(u_raw[:, 1])
    return u   
    
def plan_with_gradient_descent(
    core: VWorldModel,
    z0_visual: torch.Tensor,
    zg_visual: torch.Tensor,
    horizon: int,
    v_min: float,
    v_max: float,
    w_min: float,
    w_max: float,
    init_v_mean: float,
    init_w_mean: float,
    lr: float = 5e-2,
    num_iters: int = 200,
    optimizer_name: str = "adam",
    seed: int = 0,
    grad_clip: float | None = 1.0,
    lambda_v: float = 1e-3,
    lambda_w: float = 1e-3,
):
    torch.manual_seed(seed)
    device = z0_visual.device

    # initialize raw controls so that squashed controls start near desired mean
    u_raw = nn.Parameter(0.1 * torch.randn(horizon, 2, device=device))


    # optional better init from desired means
    with torch.no_grad():
        eps = 1e-6

        def inv_tanh_from_box(x, lo, hi):
            mid = 0.5 * (hi + lo)
            half = 0.5 * (hi - lo)
            y = (x - mid) / max(half, eps)
            y = max(min(y, 1 - 1e-4), -1 + 1e-4)
            return 0.5 * math.log((1 + y) / (1 - y))

        u_raw[:, 0] = inv_tanh_from_box(init_v_mean, v_min, v_max)
        u_raw[:, 1] = inv_tanh_from_box(init_w_mean, w_min, w_max)
        u_raw[:, 0] = u_raw[:, 0] + 1 * torch.randn(horizon, device=device)
        u_raw[:, 1] = u_raw[:, 1] + 2 * torch.randn(horizon, device=device)
        u_raw[:, 0] -= 0.0

    if optimizer_name.lower() == "adam":
        opt = torch.optim.Adam([u_raw], lr=lr)
    elif optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD([u_raw], lr=lr, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        opt = torch.optim.AdamW([u_raw], lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    loss_history = []
    best_loss = float("inf")
    best_u = None

    # initial latent history uses a fixed neutral action, not the first optimized action
    with torch.no_grad():
        neutral_action = torch.tensor(
            [[init_v_mean/10, 0.0]],
            device=device,
            dtype=z0_visual.dtype,
        )
        print("Neutral init action:", neutral_action[0].tolist())

    z_init = make_initial_latent_history_diff(core, z0_visual, neutral_action)

    for it in range(num_iters):
        opt.zero_grad()

        U = squash_controls(u_raw, v_min, v_max, w_min, w_max).unsqueeze(0)  # (1,T,2)


        z_final, _ = rollout_latent_with_actions_diff(core, z_init, U)
        latent_loss = objective_patch_mse_diff(z_final, zg_visual).mean()

        v_seq = U[..., 0]
        w_seq = U[..., 1]
        control_loss = lambda_v * (v_seq ** 2).mean() + lambda_w * (w_seq ** 2).mean()
        
        #loss = latent_loss + control_loss
        loss = latent_loss

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([u_raw], grad_clip)

        opt.step()

        loss_val = float(loss.item())
        loss_history.append(loss_val)
        
        latent_val = float(latent_loss.item())
        control_val = float(control_loss.item())
        total_val = float(loss.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_u = squash_controls(
                u_raw.detach(), v_min, v_max, w_min, w_max
            ).detach().cpu().clone()

        if (it + 1) % 10 == 0 or it == 0:
            with torch.no_grad():
                U_now = squash_controls(u_raw, v_min, v_max, w_min, w_max)
        
                print(
                    f"[GD] iter {it+1:04d}/{num_iters} | "
                    f"total={total_val:.6f} | "
                    f"latent={latent_val:.6f} | "
                    f"control={control_val:.6f} | "
                    f"v_mean={U_now[:,0].mean().item():.3f} | "
                    f"w_mean={U_now[:,1].mean().item():.3f}"
                )

    info = {
        "best_loss": best_loss,
        "loss_history": loss_history,
    }
    return best_u, info
    
    
    
    



# ============================================================
# Cost
# ============================================================

@torch.no_grad()
def angle_wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))








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
# Main
# ============================================================

def main():
    # ckpt_path = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_5/model_0072.pt"    #768d
    ckpt_path = "/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_7/model_0068.pt"      #384d 
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default = ckpt_path)
    ap.add_argument("--device", type=str, default="cuda")

    # start / goal
    ap.add_argument("--x0", type=float, default = 90)
    ap.add_argument("--y0", type=float, default = 40)
    ap.add_argument("--th0", type=float, default = 3.14)
    ap.add_argument("--xg", type=float, default = 40)
    ap.add_argument("--yg", type=float, default = 90)
    ap.add_argument("--thg", type=float, default = 2.14)

    # rendering / DINO
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    ap.add_argument("--dino_resize_size", type=int, default=224)

    # planning
    ap.add_argument("--horizon", type=int, default=12)

    # action bounds
    ap.add_argument("--v_min", type=float, default=0.2)
    ap.add_argument("--v_max", type=float, default=5.0)
    ap.add_argument("--w_min", type=float, default=-5)
    ap.add_argument("--w_max", type=float, default=5)

    # initial sampling distribution
    ap.add_argument("--init_v_mean", type=float, default=2.0)
    ap.add_argument("--init_w_mean", type=float, default=0.0)


    # dynamics
    ap.add_argument("--dt", type=float, default=0.1)

    ap.add_argument("--gd_lr", type=float, default=5e-2)
    ap.add_argument("--gd_iters", type=int, default=500)
    ap.add_argument("--gd_optimizer", type=str, default="adam")
    ap.add_argument("--gd_grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_v", type=float, default=5e-4)
    ap.add_argument("--lambda_w", type=float, default=5e-4)



    ap.add_argument("--seed", type=int, default=114514)
    ap.add_argument("--out_dir", type=str, default="gd_out_384")
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
    H = int(cfg.get("H", 3))
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

    for p in core.parameters():
        p.requires_grad_(False)
    core.eval()

    z0 = z0_pd.unsqueeze(0).to(device)
    zg = zg_pd.unsqueeze(0).to(device)

    # ---------- plan ----------
    u_best, info = plan_with_gradient_descent(
        core=core,
        z0_visual=z0,
        zg_visual=zg,
        horizon=args.horizon,
        v_min=args.v_min,
        v_max=args.v_max,
        w_min=args.w_min,
        w_max=args.w_max,
        init_v_mean=args.init_v_mean,
        init_w_mean=args.init_w_mean,
        lr=args.gd_lr,
        num_iters=args.gd_iters,
        optimizer_name=args.gd_optimizer,
        seed=args.seed,
        grad_clip=args.gd_grad_clip,
        lambda_v=args.lambda_v,
        lambda_w=args.lambda_w,
    )
    
    np.save(out_dir / "u_best.npy", u_best.numpy())
    with open(out_dir / "GD_info.txt", "w") as f:
        f.write(str(info) + "\n")

    print("[saved]", out_dir / "u_best.npy")
    print("[saved]", out_dir / "GD_info.txt")

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
    plt.title("Real unicycle rollout from GD plan")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "xy_rollout.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.array(info["loss_history"]), marker="o")
    plt.xlabel("GD iter")
    plt.ylabel("latent MSE loss")
    plt.title("Gradient-descent planning progress")
    plt.tight_layout()
    plt.savefig(out_dir / "gd_loss_history.png", dpi=200)
    plt.close()

    print("[saved]", out_dir / "xy_rollout.png")
    print("[saved]", out_dir / "gd_cost_history.png")


if __name__ == "__main__":
    main()