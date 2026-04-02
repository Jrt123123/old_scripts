#!/usr/bin/env python3
"""
eval_unicycle_dinowm_style.py

Evaluate a DINO-WM-style patch-token world model trained with:
  - ViTPredictor (from DINO_WM_vit.py)
  - VWorldModel (from DINO_WM_visual_world_model.py)
and the same concat_dim=1 + num_action_repeat conditioning.

What it does:
1) Loads a checkpoint (ckpt_best.pt / ckpt_latest.pt / ckpt_step_*.pt)
2) Opens ONE HDF5 shard (one run_*/dataset.h5)
3) Computes "relative step error" over many sampled windows:
      rel_err = ||pred - gt||_2 / (||gt||_2 + eps)
   where pred is the model's predicted patchtokens at the last step in the window
   (teacher-forced using true history frames).
4) Picks one trajectory and plots the evolution (over time) of 2 dimensions
   of a pooled representation (mean over patches) for both GT and Pred.

Assumptions about HDF5 keys:
  patchtokens: (N, T, P, D) float
  controls:    (N, T, 2) float

Notes:
- This is *teacher-forced* evaluation (no autoregressive rollout).
- "One-step" means the *last* predicted frame in a length-(H+1) window.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---- your copied DINO-WM code
from DINO_WM_vit import ViTPredictor
from DINO_WM_visual_world_model import VWorldModel


# -------------------------
# Minimal adapters (same as training)
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
        # x: (B,H,in_chans) -> (B,H,emb_dim)
        return self.net(x)


# -------------------------
# Model builder / loader
# -------------------------

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
        action_dim=cfg.action_emb_dim,   # IMPORTANT: embedded-action dim, like training
        concat_dim=1,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=0,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ).to(device)

    core.eval()
    return core


@torch.no_grad()
def predict_last_step(core: VWorldModel, patch_seq: torch.Tensor, ctrl_seq: torch.Tensor) -> torch.Tensor:
    """
    Teacher-forced window prediction.

    patch_seq: (B, H+1, P, D)    true frames
    ctrl_seq:  (B, H,   2)      true controls aligned with transitions

    Returns:
      pred_last: (B, P, D)  prediction for the LAST target frame in the window
                           i.e., predicts frame index H using history frames 0..H-1
    """
    device = next(core.parameters()).device
    patch_seq = patch_seq.to(device)
    ctrl_seq = ctrl_seq.to(device)

    B, Hp1, P, D = patch_seq.shape
    H = Hp1 - 1

    # embed actions (B,H,action_emb_dim)
    act_emb = core.encode_act(ctrl_seq)

    # visual src/target frames
    z_src_visual = patch_seq[:, :H]      # (B,H,P,D) for times t..t+H-1
    z_tgt_visual = patch_seq[:, 1:H+1]   # (B,H,P,D) for times t+1..t+H

    # tile & repeat action "trick"
    act_tiled = act_emb.unsqueeze(2).expand(B, H, P, act_emb.shape[-1])
    act_rep = act_tiled.repeat(1, 1, 1, core.num_action_repeat)  # (B,H,P,action_emb_dim*repeat)

    # concat in feature dim
    z_src = torch.cat([z_src_visual, act_rep], dim=-1)  # (B,H,P,D + actrep)
    z_pred = core.predict(z_src)                        # (B,H,P,D + actrep)

    # strip action dims back out for comparing visual prediction
    # z_pred_visual = z_pred[..., :-core.action_dim]      # action_dim is *embedded dim*, matches training convention
    
    strip = core.action_dim * core.num_action_repeat
    z_pred_visual = z_pred[..., :-strip]

    # last predicted frame in the window corresponds to the last target (time t+H)
    pred_last = z_pred_visual[:, -1]  # (B,P,D)
    return pred_last


def rel_step_error(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    pred, gt: (B,P,D)
    returns: (B,) relative L2 error
    """
    B = pred.shape[0]
    pred_f = pred.reshape(B, -1)
    gt_f = gt.reshape(B, -1)
    num = torch.linalg.norm(pred_f - gt_f, dim=1)
    den = torch.linalg.norm(gt_f, dim=1).clamp_min(eps)
    return num / den


# -------------------------
# Evaluation main
# -------------------------

def main():
    ckpt = "/storage/scratch1/6/rjiang77/Autoencoder/2_22_1/ckpt_best.pt"
    h5_dataset = "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset/run_1/dataset.h5"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default = ckpt, help="Path to ckpt_best.pt / ckpt_latest.pt / ckpt_step_*.pt")
    ap.add_argument("--h5", type=str,  default = h5_dataset, help="Path to ONE run_*/dataset.h5 for evaluation")
    ap.add_argument("--num_windows", type=int, default=2000, help="How many random windows to evaluate")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for evaluation windows")
    ap.add_argument("--traj_for_plot", type=int, default=0, help="Trajectory index (0..N-1) to plot")
    ap.add_argument("--plot_dims", type=int, nargs=2, default=[0, 1], help="Two dimensions of pooled D to plot")
    ap.add_argument("--out_dir", type=str, default="eval_out", help="Directory for saved plots/results")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu (GPU recommended)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print("[device]", device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {})
    H = int(ckpt_cfg.get("H", ckpt.get("H", 3)))  # fallback
    action_emb_dim = int(ckpt_cfg.get("action_emb_dim", 10))
    num_action_repeat = int(ckpt_cfg.get("num_action_repeat", 7))
    depth = int(ckpt_cfg.get("depth", 6))
    heads = int(ckpt_cfg.get("heads", 16))
    mlp_dim = int(ckpt_cfg.get("mlp_dim", 2048))
    dim_head = int(ckpt_cfg.get("dim_head", 64))

    # Read shapes from H5
    with h5py.File(args.h5, "r") as f:
        N, T, P, D = f["patchtokens"].shape
        assert f["controls"].shape[:2] == (N, T), f["controls"].shape
        print(f"[h5] N={N} T={T} P={P} D={D}")

    mcfg = ModelCfg(
        H=H,
        action_emb_dim=action_emb_dim,
        num_action_repeat=num_action_repeat,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dim_head=dim_head,
    )
    print("[model cfg]", mcfg)

    core = build_core_model(P=P, D=D, cfg=mcfg, device=device)
    core.load_state_dict(ckpt["model"], strict=True)
    core.eval()

    # -------------------------
    # 1) Relative step error over many random windows
    # -------------------------
    max_start = T - (H + 1)
    if max_start < 0:
        raise ValueError(f"T={T} too short for H={H} (need T >= H+1)")

    all_err = []

    def sample_batch_windows(bs: int):
        traj_idx = np.random.randint(0, N, size=(bs,))
        start_idx = np.random.randint(0, max_start + 1, size=(bs,))
        return traj_idx, start_idx

    with h5py.File(args.h5, "r") as f:
        n_done = 0
        while n_done < args.num_windows:
            bs = min(args.batch, args.num_windows - n_done)
            traj_idx, start_idx = sample_batch_windows(bs)

            # Build tensors
            patch_batch = np.empty((bs, H + 1, P, D), dtype=np.float32)
            ctrl_batch = np.empty((bs, H, 2), dtype=np.float32)

            for i in range(bs):
                tr = traj_idx[i]
                s = start_idx[i]
                patch_batch[i] = f["patchtokens"][tr, s:s + (H + 1), :, :]
                ctrl_batch[i] = f["controls"][tr, s:s + H, :]

            patch_t = torch.from_numpy(patch_batch)
            ctrl_t = torch.from_numpy(ctrl_batch)

            pred_last = predict_last_step(core, patch_t, ctrl_t)                 # (B,P,D)
            gt_last = patch_t[:, -1].to(pred_last.device)                        # (B,P,D)

            err = rel_step_error(pred_last, gt_last).detach().cpu().numpy()      # (B,)
            all_err.append(err)
            n_done += bs

    all_err = np.concatenate(all_err, axis=0)
    print(f"[eval] windows={len(all_err)}")
    print(f"[eval] rel step err: mean={all_err.mean():.6f}  median={np.median(all_err):.6f}")
    for q in [90, 95, 99]:
        print(f"[eval] p{q}={np.percentile(all_err, q):.6f}")

    # Save stats
    stats_path = out_dir / "relative_step_error_stats.txt"
    with open(stats_path, "w") as w:
        w.write(f"ckpt: {args.ckpt}\n")
        w.write(f"h5: {args.h5}\n")
        w.write(f"model_cfg: {mcfg}\n")
        w.write(f"num_windows: {len(all_err)}\n")
        w.write(f"mean: {all_err.mean():.8f}\n")
        w.write(f"median: {np.median(all_err):.8f}\n")
        for q in [90, 95, 99]:
            w.write(f"p{q}: {np.percentile(all_err, q):.8f}\n")
    np.save(out_dir / "relative_step_error_values.npy", all_err)
    print(f"[saved] {stats_path}")
    print(f"[saved] {out_dir/'relative_step_error_values.npy'}")

    # -------------------------
    # 2) Plot evolution on 2 dims for one trajectory
    # -------------------------
    d0, d1 = args.plot_dims
    if not (0 <= d0 < D and 0 <= d1 < D):
        raise ValueError(f"plot_dims must be within [0,{D-1}], got {args.plot_dims}")

    traj = int(args.traj_for_plot)
    if not (0 <= traj < N):
        raise ValueError(f"traj_for_plot must be within [0,{N-1}]")

    # We'll compute predictions for times t=0..T-H-1 (predict time t+H)
    # and compare to GT at that time, using pooled mean-over-patches representation (D,)
    t_list = []
    gt_xy = []
    pr_xy = []

    with h5py.File(args.h5, "r") as f:
        # load this one trajectory to memory (one traj is usually manageable)
        patch_traj = f["patchtokens"][traj].astype(np.float32)  # (T,P,D)
        ctrl_traj = f["controls"][traj].astype(np.float32)      # (T,2)

    # iterate and batch for speed
    Bplot = 32
    n_steps = T - H
    for t0 in range(0, n_steps, Bplot):
        t1 = min(n_steps, t0 + Bplot)
        bs = t1 - t0

        patch_batch = np.empty((bs, H + 1, P, D), dtype=np.float32)
        ctrl_batch = np.empty((bs, H, 2), dtype=np.float32)

        for i, t in enumerate(range(t0, t1)):
            patch_batch[i] = patch_traj[t:t + (H + 1)]
            ctrl_batch[i] = ctrl_traj[t:t + H]

        patch_t = torch.from_numpy(patch_batch)
        ctrl_t = torch.from_numpy(ctrl_batch)

        pred_last = predict_last_step(core, patch_t, ctrl_t).detach().cpu().numpy()  # (bs,P,D)
        gt_last = patch_batch[:, -1]                                                 # (bs,P,D)

        # pool over patches -> (bs,D)
        pred_pool = pred_last.mean(axis=1)
        gt_pool = gt_last.mean(axis=1)

        for i in range(bs):
            t = t0 + i + H  # the time index being predicted (t+H)
            t_list.append(t)
            gt_xy.append([gt_pool[i, d0], gt_pool[i, d1]])
            pr_xy.append([pred_pool[i, d0], pred_pool[i, d1]])

    gt_xy = np.asarray(gt_xy)
    pr_xy = np.asarray(pr_xy)
    t_list = np.asarray(t_list)

    # Plot time evolution for dim d0 and d1
    plt.figure()
    plt.plot(t_list, gt_xy[:, 0], label=f"GT dim {d0}")
    plt.plot(t_list, pr_xy[:, 0], label=f"Pred dim {d0}")
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.title(f"Trajectory {traj}: pooled feature dim {d0} over time (teacher-forced)")
    plt.legend()
    p0 = out_dir / f"traj{traj}_dim{d0}_timeseries.png"
    plt.tight_layout()
    plt.savefig(p0, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t_list, gt_xy[:, 1], label=f"GT dim {d1}")
    plt.plot(t_list, pr_xy[:, 1], label=f"Pred dim {d1}")
    plt.xlabel("time index")
    plt.ylabel("value")
    plt.title(f"Trajectory {traj}: pooled feature dim {d1} over time (teacher-forced)")
    plt.legend()
    p1 = out_dir / f"traj{traj}_dim{d1}_timeseries.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=200)
    plt.close()

    # Also plot 2D phase (dim d0 vs d1)
    plt.figure()
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], label="GT")
    plt.plot(pr_xy[:, 0], pr_xy[:, 1], label="Pred")
    plt.xlabel(f"dim {d0}")
    plt.ylabel(f"dim {d1}")
    plt.title(f"Trajectory {traj}: pooled 2D evolution (dims {d0},{d1})")
    plt.legend()
    p2 = out_dir / f"traj{traj}_dims{d0}_{d1}_phase.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=200)
    plt.close()

    print(f"[saved] {p0}")
    print(f"[saved] {p1}")
    print(f"[saved] {p2}")


if __name__ == "__main__":
    main()