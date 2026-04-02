#!/usr/bin/env python3
"""
train_unicycle_dinowm_style.py

Closest-to-DINO-WM implementation for your patch-token HDF5:
- imports ViTPredictor (vit.py) and VWorldModel (visual_world_model.py)
- uses concat_dim=1 and num_action_repeat like their code
- trains ONLY predictor + action encoder (encoder/decoder are dummy here because you already stored patchtokens)

IMPORTANT:
- This script bypasses their image encoder because your dataset already contains patchtokens.
- We create a tiny "PatchTokenEncoder" that just returns your patchtokens as if they came from DINO.
"""

from __future__ import annotations
import os
import random
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ---- import their code (you uploaded these files)
from DINO_WM_vit import ViTPredictor
from DINO_WM_visual_world_model import VWorldModel

import glob
from dataclasses import dataclass
from collections import OrderedDict



def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def prune_old_snapshots(out_dir: Path, keep_last: int):
    snaps = sorted(out_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(snaps) <= keep_last:
        return
    for p in snaps[:-keep_last]:
        try:
            p.unlink()
        except Exception as e:
            print(f"[warn] failed to delete {p}: {e}")

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # x: (d, d)
    d = x.size(0)
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()



def vc_loss(
    x: torch.Tensor,
    sigma_min: float = 1.0,
    eps: float = 1e-4,
    var_weight: float = 1.0,
    cov_weight: float = 1.0,
) -> torch.Tensor:
    """
    Variance-Covariance regularization (VICReg without invariance term).
    x: (N, d) features
    """
    # center
    x = x - x.mean(dim=0, keepdim=True)

    # variance term: enforce std >= sigma_min
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)  # (d,)
    var_term = torch.mean(torch.relu(sigma_min - std) ** 2)

    # covariance term: penalize off-diagonal covariance
    N, d = x.shape
    cov = (x.T @ x) / max(N, 1)  # (d, d)
    cov_term = off_diagonal(cov).pow(2).sum() / (d * (d - 1) + 1e-12)

    return var_weight * var_term + cov_weight * cov_term
    
    

class PatchBisimEncoder(nn.Module):
    """
    Per-patch MLP: maps DINO patch token dim D -> dw.
    Operates on last dimension only; shared across patches and timesteps.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., in_dim)
        return self.net(z)
    
    

# -------------------------
# Multi-file dataset (streaming)
# -------------------------

class MultiH5PatchSegments(Dataset):
    """
    Random segment sampler over multiple HDF5 shards.

    Key goals:
      - never load entire files
      - keep only a small cache of open h5 handles PER WORKER
      - validate that all shards match (T may vary but must be >= H+1; P,D must match)
    """

    def __init__(
        self,
        h5_paths: list[str],
        H: int,
        traj_len_trunc: int | None = None,
        stride: int = 1,
        handle_cache_size: int = 2,
        max_tries: int = 50,
    ):
        self.paths = [str(p) for p in h5_paths]
        if not self.paths:
            raise ValueError("No HDF5 paths provided.")

        self.H = int(H)
        self.stride = int(stride)
        self.traj_len_trunc = traj_len_trunc  # optionally cap T used for sampling
        self.handle_cache_size = int(handle_cache_size)
        self.max_tries = int(max_tries)

        # Per-file metadata (read once at init)
        self.meta = []
        P0 = D0 = None

        for p in self.paths:
            with h5py.File(p, "r") as f:
                if "patchtokens" not in f or "controls" not in f:
                    raise KeyError(f"{p} missing required keys. Need 'patchtokens' and 'controls'.")
                N, T, P, D = f["patchtokens"].shape
                cN, cT, cA = f["controls"].shape
                if cA != 2 or cN != N or cT != T:
                    raise ValueError(f"{p} controls shape {f['controls'].shape} incompatible with patchtokens {f['patchtokens'].shape}")

                if P0 is None:
                    P0, D0 = P, D
                else:
                    if P != P0 or D != D0:
                        raise ValueError(f"Patch shape mismatch: {p} has (P,D)=({P},{D}) but expected ({P0},{D0})")

                # effective T for sampling (optionally truncated)
                Teff = min(T, traj_len_trunc) if traj_len_trunc is not None else T

                if Teff < (self.H + 1) * self.stride:
                    # shard too short; skip it
                    continue

                self.meta.append({"path": p, "N": N, "T": T, "Teff": Teff})

        if not self.meta:
            raise RuntimeError("No usable HDF5 shards: all were too short or invalid.")

        self.P = P0
        self.D = D0

        # sampling weights proportional to number of trajectories (N)
        self.weights = np.array([m["N"] for m in self.meta], dtype=np.float64)
        self.weights /= self.weights.sum()

        # Per-worker state: opened files cache (lazy)
        self._handles = None  # created per worker process

    def _get_handle(self, path: str) -> h5py.File:
        """
        LRU cache of open handles per worker.
        """
        if self._handles is None:
            self._handles = OrderedDict()

        if path in self._handles:
            f = self._handles.pop(path)
            self._handles[path] = f
            return f

        # evict if needed
        while len(self._handles) >= self.handle_cache_size:
            _, old = self._handles.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass

        f = h5py.File(path, "r")
        self._handles[path] = f
        return f

    def __len__(self):
        # This is a "random sampler" dataset. Length is arbitrary.
        # Make it large so DataLoader has enough indices.
        return 10_000_000

    def __getitem__(self, idx):
        # idx is not used as a deterministic index; we sample randomly.
        # This is fine with shuffle=True (or even shuffle=False).
        for _ in range(self.max_tries):
            m = self.meta[np.random.choice(len(self.meta), p=self.weights)]
            f = self._get_handle(m["path"])

            traj_idx = random.randrange(m["N"])
            Teff = m["Teff"]

            max_start = Teff - (self.H + 1) * self.stride
            if max_start < 0:
                continue
            s = random.randint(0, max_start)
            t_idx = np.arange(s, s + (self.H + 1) * self.stride, self.stride)

            patch = f["patchtokens"][traj_idx, t_idx, :, :].astype(np.float32)  # (H+1,P,D)
            ctrl = f["controls"][traj_idx, t_idx[:-1], :].astype(np.float32)    # (H,2)

            return {
                "patch": torch.from_numpy(patch),
                "ctrl": torch.from_numpy(ctrl),
            }

        # If repeated failures, raise (should be rare)
        raise RuntimeError("Failed to sample a valid segment after many tries.")

    def __del__(self):
        if self._handles is not None:
            for _, f in self._handles.items():
                try:
                    f.close()
                except Exception:
                    pass


# -------------------------
# Minimal adapters for VWorldModel
# -------------------------

class PatchTokenEncoder(nn.Module):
    def __init__(self, emb_dim: int, patch_size: int = 14, name: str = "dino"):
        super().__init__()
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.latent_ndim = 2
        self.name = name
        
    def forward(self, x):
        return x


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


class PatchTokenWorldModel(nn.Module):
    """
    DINO-WM-style training but with:
      - learned patch encoder h_eta: D -> dw
      - predictor trained to predict w_{t+1} from w_t + action conditioning
      - VC regularizer on w (anti-collapse)
      - bisimilarity loss: ||w - w'|| ≈ gamma ||T(w) - T(w')||
    """
    def __init__(
        self,
        core: VWorldModel,
        bisim_encoder: nn.Module,
        dw: int,
        vc_weight: float = 0.1,
        sigma_min: float = 1.0,
        vc_var_weight: float = 1.0,
        vc_cov_weight: float = 1.0,
        bisim_weight: float = 0.1,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.core = core
        self.bisim = bisim_encoder
        self.dw = int(dw)

        self.vc_weight = float(vc_weight)
        self.sigma_min = float(sigma_min)
        self.vc_var_weight = float(vc_var_weight)
        self.vc_cov_weight = float(vc_cov_weight)

        self.bisim_weight = float(bisim_weight)
        self.gamma = float(gamma)

    @staticmethod
    def _pool_state(x: torch.Tensor) -> torch.Tensor:
        """
        Pool patch tokens into a single state vector.
        x: (B,H,P,D) -> (B,H,D)
        """
        return x.mean(dim=2)

    def forward(self, z_visual: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # z_visual: (B,H+1,P,D) ; act: (B,H,2)
        B, Tp1, P, D = z_visual.shape
        H = Tp1 - 1

        # ---- encode to bisim latent w: (B,H+1,P,dw)
        w_all = self.bisim(z_visual)

        # ---- prediction (DINO-WM style) in w-space
        act_emb = self.core.encode_act(act)  # (B,H,action_emb_dim)

        w_src = w_all[:, :H]          # (B,H,P,dw)
        w_tgt = w_all[:, 1:H+1]       # (B,H,P,dw)

        if self.core.concat_dim != 1:
            raise RuntimeError("Set concat_dim=1 for closest-to-implementation conditioning.")

        act_tiled = act_emb.unsqueeze(2).expand(B, H, P, act_emb.shape[-1])
        act_rep = act_tiled.repeat(1, 1, 1, self.core.num_action_repeat)

        x_src = torch.cat([w_src, act_rep], dim=-1)   # (B,H,P,dw+...)
        x_tgt = torch.cat([w_tgt, act_rep], dim=-1)

        x_pred = self.core.predict(x_src)             # (B,H,P,dw+...)
        w_pred_next = x_pred[..., : self.dw]          # (B,H,P,dw)

        # ---- prediction loss on state part only
        pred_loss = self.core.emb_criterion(
            w_pred_next,
            x_tgt[..., : self.dw].detach()
        )

        # ---- VC loss on w (flatten across B*(H+1)*P)
        w_flat = w_all.reshape(-1, self.dw)
        reg_vc = vc_loss(
            w_flat,
            sigma_min=self.sigma_min,
            var_weight=self.vc_var_weight,
            cov_weight=self.vc_cov_weight,
        )

        # ---- bisimilarity loss (pair within batch via permutation)
        # Pool patch tokens to state vectors
        s_now = self._pool_state(w_src)        # (B,H,dw)
        s_next = self._pool_state(w_pred_next) # (B,H,dw)

        # Pairing: permute batch indices
        perm = torch.randperm(B, device=s_now.device)
        s_now_p = s_now[perm]                  # (B,H,dw)
        s_next_p = s_next[perm]                # (B,H,dw)

        # Distances per (b,t)
        d_now = torch.linalg.norm(s_now - s_now_p, dim=-1)     # (B,H)
        d_next = torch.linalg.norm(s_next - s_next_p, dim=-1)  # (B,H)

        bisim = ((d_now - self.gamma * d_next) ** 2).mean()

        return pred_loss + self.vc_weight * reg_vc + self.bisim_weight * bisim


# -------------------------
# Training
# -------------------------

@dataclass
class Cfg:
    root_glob: str
    out: str
    H: int = 3
    stride: int = 1
    traj_trunc: int | None = None

    batch: int = 8
    epochs: int = 50
    steps_per_epoch: int = 2000  # since dataset is "infinite"
    lr: float = 5e-5
    wd: float = 1e-4
    workers: int = 2
    seed: int = 0

    # DINO-WM-ish predictor knobs
    action_emb_dim: int = 10
    num_action_repeat: int = 7
    depth: int = 6
    heads: int = 16
    mlp_dim: int = 2048
    dim_head: int = 64

    # HDF5 handle cache per worker
    handle_cache_size: int = 2
    
    dw: int = 64
    bisim_hidden: int = 512
    vc_weight: float = 0.1
    sigma_min: float = 1.0
    vc_var_weight: float = 1.0
    vc_cov_weight: float = 1.0
    
    bisim_weight: float = 0.1
    gamma: float = 0.99


def main():

    global_step = 0
    loss_window = []  # list of recent losses for moving average
    best_ma = float("inf")

    h5_root =  "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset/run_*/dataset.h5"
    save_path = '/storage/scratch1/6/rjiang77/Autoencoder/3_2_1/'
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_glob", type=str, default = h5_root,
                    help='Glob for shards, e.g. "/path/to/2_14_dataset/run_*/dataset.h5"')
    ap.add_argument("--out", type=str, default = save_path)

    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--traj_trunc", type=int, default=0, help="0 means no truncation; else cap T used")

    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--action_emb_dim", type=int, default=10)
    ap.add_argument("--num_action_repeat", type=int, default=7)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--mlp_dim", type=int, default=2048)
    ap.add_argument("--dim_head", type=int, default=64)

    ap.add_argument("--handle_cache_size", type=int, default=2)
    ap.add_argument("--ckpt_latest_every", type=int, default=1000)
    ap.add_argument("--ckpt_snapshot_every", type=int, default=10000)
    ap.add_argument("--ckpt_keep_last", type=int, default=3)
    ap.add_argument("--best_ma_window", type=int, default=200)  # moving-average window in steps
    
    ap.add_argument("--dw", type=int, default=64, help="bisim latent dimension")
    ap.add_argument("--bisim_hidden", type=int, default=512)
    ap.add_argument("--vc_weight", type=float, default=0.1)
    ap.add_argument("--sigma_min", type=float, default=1.0)
    ap.add_argument("--vc_var_weight", type=float, default=1.0)
    ap.add_argument("--vc_cov_weight", type=float, default=1.0)
    ap.add_argument("--bisim_weight", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.99)

    args = ap.parse_args()
    


    cfg = Cfg(
        root_glob=args.root_glob,
        out=args.out,
        H=args.H,
        stride=args.stride,
        traj_trunc=(None if args.traj_trunc == 0 else args.traj_trunc),
        batch=args.batch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        wd=args.wd,
        workers=args.workers,
        seed=args.seed,
        action_emb_dim=args.action_emb_dim,
        num_action_repeat=args.num_action_repeat,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        handle_cache_size=args.handle_cache_size,
        dw=args.dw,
        bisim_hidden=args.bisim_hidden,
        vc_weight=args.vc_weight,
        sigma_min=args.sigma_min,
        vc_var_weight=args.vc_var_weight,
        vc_cov_weight=args.vc_cov_weight,
        bisim_weight=args.bisim_weight,
        gamma=args.gamma,
    )

    set_seed(cfg.seed)

    paths = sorted(glob.glob(cfg.root_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {cfg.root_glob}")
    print(f"[shards] found {len(paths)} files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    ds = MultiH5PatchSegments(
        h5_paths=paths,
        H=cfg.H,
        traj_len_trunc=cfg.traj_trunc,
        stride=cfg.stride,
        handle_cache_size=cfg.handle_cache_size,
    )
    print(f"[shape] P={ds.P}, D={ds.D}")

    dl = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=2 if cfg.workers > 0 else None,
    )

    # Build DINO-WM modules
    encoder = PatchTokenEncoder(emb_dim=ds.D, patch_size=14, name="dino")
    proprio_encoder = MLPSeqEncoder(in_chans=1, emb_dim=1)  # unused (proprio_dim=0)
    action_encoder = MLPSeqEncoder(in_chans=2, emb_dim=cfg.action_emb_dim)

     # ---- learned bisim encoder (D -> dw)
    bisim = PatchBisimEncoder(in_dim=ds.D, out_dim=cfg.dw, hidden=cfg.bisim_hidden).to(device)

    # predictor now operates on dw (not ds.D)
    predictor_dim = cfg.dw + cfg.action_emb_dim * cfg.num_action_repeat

    predictor = ViTPredictor(
        num_patches=ds.P,
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
        encoder=encoder,                      # still dummy
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=None,
        predictor=predictor,
        proprio_dim=0,
        action_dim=cfg.action_emb_dim,
        concat_dim=1,
        num_action_repeat=cfg.num_action_repeat,
        num_proprio_repeat=0,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    ).to(device)

    model = PatchTokenWorldModel(
        core=core,
        bisim_encoder=bisim,
        dw=cfg.dw,
        vc_weight=cfg.vc_weight,
        sigma_min=cfg.sigma_min,
        vc_var_weight=cfg.vc_var_weight,
        vc_cov_weight=cfg.vc_cov_weight,
        bisim_weight=cfg.bisim_weight,
        gamma=cfg.gamma,
    ).to(device)
    
    
    
    opt = optim.AdamW(
        list(core.predictor.parameters()) +
        list(core.action_encoder.parameters()) +
        list(bisim.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )
    
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    latest_path = out_dir / "ckpt_latest.pt"
    start_epoch = 1
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu")
        core.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        best_ma = float(ckpt.get("ma_loss", float("inf")))  # not perfect; best is from ckpt_best anyway
        print(f"[resume] loaded ckpt_latest.pt at step {global_step}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        loss_window = ckpt.get("loss_window", [])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        if "bisim" in ckpt:
            bisim.load_state_dict(ckpt["bisim"])
            
    best_path = out_dir / "ckpt_best.pt"
    if best_path.exists():
        ckptb = torch.load(best_path, map_location="cpu")
        best_ma = float(ckptb.get("ma_loss", best_ma))
        print(f"[resume] best_ma from ckpt_best.pt: {best_ma:.6f}")

    it = iter(dl)
    best = float("inf")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        losses = []

        for step in range(cfg.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            patch = batch["patch"].to(device, non_blocking=True)  # (B,H+1,P,D)
            ctrl = batch["ctrl"].to(device, non_blocking=True)    # (B,H,2)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(patch, ctrl)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_val = float(loss.item())
            losses.append(loss_val)
            global_step += 1
            loss_window.append(loss_val)
            if len(loss_window) > args.best_ma_window:
                loss_window.pop(0)
            ma = float(np.mean(loss_window))

            if (step + 1) % 100 == 0:
                print(f"epoch {epoch:03d} step {step+1:04d}/{cfg.steps_per_epoch} loss {np.mean(losses[-100:]):.6f}")
                
            if (global_step % args.ckpt_latest_every) == 0:
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model": core.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else vars(args),
                    "P": ds.P,
                    "D": ds.D,
                    "num_shards": len(paths),
                    "ma_loss": ma,
                    "scaler": scaler.state_dict(),
                    "loss_window": loss_window,
                    "bisim": bisim.state_dict(),
                }
                atomic_torch_save(ckpt, out_dir / "ckpt_latest.pt")
                print(f"[ckpt] saved latest at step {global_step} (ma={ma:.6f})")
                
                
            if ma < best_ma:
                best_ma = ma
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model": core.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else vars(args),
                    "P": ds.P,
                    "D": ds.D,
                    "num_shards": len(paths),
                    "ma_loss": ma,
                    "scaler": scaler.state_dict(),
                    "loss_window": loss_window,
                    "bisim": bisim.state_dict(),
                }
                atomic_torch_save(ckpt, out_dir / "ckpt_best.pt")
                print(f"[ckpt] NEW BEST ma={best_ma:.6f} at step {global_step}")
                
                
            if (global_step % args.ckpt_snapshot_every) == 0:
                snap_path = out_dir / f"ckpt_step_{global_step:09d}.pt"
                ckpt = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "model": core.state_dict(),
                    "opt": opt.state_dict(),
                    "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else vars(args),
                    "P": ds.P,
                    "D": ds.D,
                    "num_shards": len(paths),
                    "ma_loss": ma,
                    "scaler": scaler.state_dict(),
                    "loss_window": loss_window,
                    "bisim": bisim.state_dict(),
                }
                atomic_torch_save(ckpt, snap_path)
                prune_old_snapshots(out_dir, keep_last=args.ckpt_keep_last)
                print(f"[ckpt] snapshot saved {snap_path.name}")

        mean_loss = float(np.mean(losses))
        print(f"[epoch {epoch:03d}] mean loss: {mean_loss:.6f}")


    print("done.")


if __name__ == "__main__":
    main()