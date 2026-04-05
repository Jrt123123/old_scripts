#!/usr/bin/env python3
"""
train_unicycle_decoder_vqvae.py

Train a (patchtokens -> image) decoder using DINO-WM's VQVAE module.

Assumptions:
- Your HDF5 shards contain:
    patchtokens : (N, T, P, D)
    positions  : (N, T, 2)
    orientations : (N, T)
- Images are NOT stored. We regenerate the teardrop mask on-the-fly from (pos, theta).
- Uses VQVAE from DINO-WM-style `vqvae.py` (the file you uploaded).
  That file imports `distributed_fn`; we stub it out for single-process training.

Notes:
- VQVAE expects input: (b, t, num_patches, emb_dim)
  We train with t=1 (single frame) to keep it simple.
- VQVAE outputs an image at ~256x256 (because patches become 16x16 then upsample x16).
  We resize logits to your desired img_dim (e.g., 224) before computing loss.
- Strongly recommended: project DINO dim D (e.g., 768) down to a smaller emb_dim (e.g., 64/128)
  before decoding, otherwise the conv stacks are huge.
"""

from __future__ import annotations
import os
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict

import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.draw import polygon
from skimage.transform import resize

# -------------------------
# Stub out distributed_fn for vqvae.py (single-process)
# -------------------------
import types, sys
dist_stub = types.SimpleNamespace(
    all_reduce=lambda x: x,     # no-op
    is_distributed=lambda: False,
    get_world_size=lambda: 1,
    get_rank=lambda: 0,
)
sys.modules["distributed_fn"] = dist_stub

# -------------------------
# Import your uploaded VQVAE implementation
# (Make sure vqvae.py is in the same folder as this script, or adjust path.)
# -------------------------
from experiments.dino_wm.models.vqvae import VQVAE  # expects forward(input) -> (dec_logits, diff)


# -------------------------
# Your teardrop renderer (same as your generator)
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
    return img.astype(np.float32)  # (H,W) in [0,1]


# -------------------------
# Streaming dataset: sample single frames from multiple shards
# -------------------------
class MultiH5PatchFrames(Dataset):
    """
    Random frame sampler over multiple HDF5 shards.

    Returns:
      patch: (P,D) float32
      img  : (1,H,W) float32 in [0,1]
    """

    def __init__(
        self,
        h5_paths: list[str],
        *,
        img_dim: int = 224,
        radius: int = 10,
        traj_len_trunc: int | None = None,
        handle_cache_size: int = 2,
        max_tries: int = 50,
    ):
        self._handles = None
        self.paths = [str(p) for p in h5_paths]
        if not self.paths:
            raise ValueError("No HDF5 paths provided.")

        self.img_dim = int(img_dim)
        self.radius = int(radius)
        self.traj_len_trunc = traj_len_trunc
        self.handle_cache_size = int(handle_cache_size)
        self.max_tries = int(max_tries)

        self.meta = []
        P0 = D0 = None

        for p in self.paths:
            with h5py.File(p, "r") as f:
                for k in ("patchtokens", "positions", "orientations"):
                    if k not in f:
                        raise KeyError(f"{p} missing required key '{k}'")

                N, T, P, D = f["patchtokens"].shape
                pN, pT, _ = f["positions"].shape
                oN, oT = f["orientations"].shape
                if pN != N or pT != T or oN != N or oT != T:
                    raise ValueError(f"{p} positions/orientations shape mismatch vs patchtokens.")

                if P0 is None:
                    P0, D0 = P, D
                else:
                    if P != P0 or D != D0:
                        raise ValueError(f"Patch shape mismatch: {p} has (P,D)=({P},{D}) expected ({P0},{D0})")

                Teff = min(T, traj_len_trunc) if traj_len_trunc is not None else T
                if Teff < 1:
                    continue

                self.meta.append({"path": p, "N": N, "T": T, "Teff": Teff})

        if not self.meta:
            raise RuntimeError("No usable HDF5 shards.")

        self.P = P0
        self.D = D0

        # weights proportional to number of trajectories
        self.weights = np.array([m["N"] for m in self.meta], dtype=np.float64)
        self.weights /= self.weights.sum()

        self._handles = None  # per worker

    def _get_handle(self, path: str) -> h5py.File:
        if self._handles is None:
            self._handles = OrderedDict()

        if path in self._handles:
            f = self._handles.pop(path)
            self._handles[path] = f
            return f

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
        # "infinite" random sampler
        return 10_000_000

    def __getitem__(self, idx):
        for _ in range(self.max_tries):
            m = self.meta[np.random.choice(len(self.meta), p=self.weights)]
            f = self._get_handle(m["path"])
    
            traj_idx = random.randrange(m["N"])
            t = random.randrange(m["Teff"])
    
            # DEBUG: show progress occasionally (avoid spamming)
            if (idx % 10000) == 0:
                print(f"[debug][getitem] path={m['path']} traj={traj_idx} t={t}", flush=True)
    
            # ---- HDF5 read
            if (idx % 10000) == 0:
                print("[debug][getitem] reading patch/pos/theta...", flush=True)
    
            patch = f["patchtokens"][traj_idx, t, :, :].astype(np.float32)
            pos = f["positions"][traj_idx, t, :].astype(np.float32)
            theta = float(f["orientations"][traj_idx, t])
    
            if (idx % 10000) == 0:
                print("[debug][getitem] rendering mask...", flush=True)
    
            img = generate_teardrop_mask(
                center=(float(pos[0]), float(pos[1])),
                radius=self.radius,
                theta=theta,
                image_size=self.img_dim,
            )
    
            if (idx % 10000) == 0:
                print("[debug][getitem] done.", flush=True)
    
            img = img[None, ...]
            return {"patch": torch.from_numpy(patch), "img": torch.from_numpy(img)}
            
            
    
    def __del__(self):
        if hasattr(self, "_handles") and self._handles is not None:
            for _, f in self._handles.items():
                try: f.close()
                except: pass


# -------------------------
# Helper
# -------------------------
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# -------------------------
# Train
# -------------------------
@dataclass
class Cfg:
    root_glob: str
    out: str

    img_dim: int = 224
    radius: int = 10

    # decoder bits
    emb_dim: int = 128      # projection dim before VQVAE
    vq_quantize: bool = False

    # training
    batch: int = 32
    epochs: int = 50
    steps_per_epoch: int = 2000
    lr: float = 2e-4
    wd: float = 1e-4
    workers: int = 1
    seed: int = 0

    # data
    traj_trunc: int | None = None
    handle_cache_size: int = 2

    # logging / ckpt
    ckpt_every: int = 2000


def main():

    h5_root =  "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset/run_*/dataset.h5"
    save_path = '/storage/scratch1/6/rjiang77/Autoencoder/2_28_1/'
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_glob", type=str, default = h5_root,
                    help='Glob for shards, e.g. "/path/to/run_*/dataset.h5"')
    ap.add_argument("--out", type=str, default = save_path)

    ap.add_argument("--img_dim", type=int, default=224)
    ap.add_argument("--radius", type=int, default=10)

    ap.add_argument("--emb_dim", type=int, default=128,
                    help="Project DINO patch dim D to this before decoding.")
    ap.add_argument("--vq_quantize", action="store_true",
                    help="Enable quantization (requires real distributed_fn; leave off for single GPU).")

    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--steps_per_epoch", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--traj_trunc", type=int, default=0, help="0 means no truncation; else cap T used")
    ap.add_argument("--handle_cache_size", type=int, default=2)

    ap.add_argument("--ckpt_every", type=int, default=2000)

    args = ap.parse_args()

    cfg = Cfg(
        root_glob=args.root_glob,
        out=args.out,
        img_dim=args.img_dim,
        radius=args.radius,
        emb_dim=args.emb_dim,
        vq_quantize=bool(args.vq_quantize),
        batch=args.batch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        lr=args.lr,
        wd=args.wd,
        workers=args.workers,
        seed=args.seed,
        traj_trunc=(None if args.traj_trunc == 0 else args.traj_trunc),
        handle_cache_size=args.handle_cache_size,
        ckpt_every=args.ckpt_every,
    )

    set_seed(cfg.seed)

    paths = sorted(glob.glob(cfg.root_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {cfg.root_glob}")
    print(f"[shards] found {len(paths)} files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)

    ds = MultiH5PatchFrames(
        h5_paths=paths,
        img_dim=cfg.img_dim,
        radius=cfg.radius,
        traj_len_trunc=cfg.traj_trunc,
        handle_cache_size=cfg.handle_cache_size,
    )
    print(f"[shape] P={ds.P}, D={ds.D}  (img_dim={cfg.img_dim})")

    dl = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=2 if cfg.workers > 0 else None,
        timeout=60
    )

    # Project DINO dim D -> emb_dim for the decoder (recommended)
    proj = nn.Linear(ds.D, cfg.emb_dim).to(device)

    # VQVAE decoder: output 1 channel (mask), input emb_dim matches projection dim
    dec = VQVAE(
        in_channel=1,
        emb_dim=cfg.emb_dim,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        quantize=cfg.vq_quantize,   # keep False for single GPU unless you implement dist_fn properly
        n_embed=512,
    ).to(device)

    params = list(proj.parameters()) + list(dec.parameters())
    opt = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "ckpt_latest.pt"

    global_step = 0
    start_epoch = 1
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu")
        proj.load_state_dict(ckpt["proj"])
        dec.load_state_dict(ckpt["dec"])
        opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "scaler" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[resume] loaded {latest_path} at step {global_step}")

    it = iter(dl)
    
    print("[debug] about to fetch first batch...")
    batch0 = next(it)
    print("[debug] got first batch:",
          {k: tuple(v.shape) for k, v in batch0.items()})


    for epoch in range(start_epoch, cfg.epochs + 1):
        proj.train()
        dec.train()

        losses = []
        for step in range(cfg.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            patch = batch["patch"].to(device, non_blocking=True)  # (B,P,D)
            img = batch["img"].to(device, non_blocking=True)      # (B,1,H,W)
            
            if step == 0 and epoch == start_epoch:
                print("[debug] entering first forward pass", flush=True)

            # Make input shape VQVAE expects: (b,t,P,emb_dim), using t=1
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x = proj(patch)              # (B,P,emb_dim)
                x = x.unsqueeze(1)           # (B,1,P,emb_dim)

                logits, diff = dec(x)        # logits: (B*1,1,H',W')
                # Reshape back to (B,1,1,H',W') then squeeze t=1
                logits = logits.view(patch.size(0), 1, 1, logits.size(-2), logits.size(-1)).squeeze(1)  # (B,1,H',W')

                # Resize to supervision resolution
                logits = F.interpolate(logits, size=(cfg.img_dim, cfg.img_dim), mode="bilinear", align_corners=False)

                # Binary mask loss (use logits directly)
                rec_loss = F.binary_cross_entropy_with_logits(logits, img)

                # If quantize is on, diff is a small regularizer term (their VQ commitment loss)
                # diff is 0 when quantize=False in your vqvae.py
                vq_loss = diff.mean()
                loss = rec_loss + vq_loss

            opt.zero_grad(set_to_none=True)
            
            
            if step == 0 and epoch == start_epoch:
                print("[debug] finished forward (computed loss)", flush=True)
                
            scaler.scale(loss).backward()
            
            if step == 0 and epoch == start_epoch:
                print("[debug] finished backward", flush=True)
            scaler.step(opt)
            scaler.update()
            
            if step == 0 and epoch == start_epoch:
                print("[debug] optimizer step done", flush=True)

            global_step += 1
            losses.append(float(loss.item()))

            if (step + 1) % 100 == 0:
                print(f"epoch {epoch:03d} step {step+1:04d}/{cfg.steps_per_epoch} "
                      f"loss {np.mean(losses[-100:]):.6f} (rec {float(rec_loss.item()):.6f})")

            if (global_step % cfg.ckpt_every) == 0:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "proj": proj.state_dict(),
                    "dec": dec.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg": cfg.__dict__,
                    "P": ds.P,
                    "D": ds.D,
                }
                atomic_torch_save(ckpt, latest_path)
                print(f"[ckpt] saved {latest_path} at step {global_step}")

        print(f"[epoch {epoch:03d}] mean loss: {float(np.mean(losses)):.6f}")

    print("done.")


if __name__ == "__main__":
    main()