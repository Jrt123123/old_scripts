#!/usr/bin/env python3
"""
train_patchtokens_decoder_transposedconv.py

DINO-WM-style decoder:
  patchtokens (B,P,D) -> reshape -> (B,D,Hp,Wp) -> ConvTranspose2d stack -> image

This trains the decoder to reconstruct **dino_x** (the exact tensor fed to DINO after
transform+normalize) from the corresponding stored patchtokens.

Assumes your "supervision" shards contain:
  - patchtokens : (N,T,P,D)   (float32)
  - dino_x      : (N,T,3,R,R) (float32 or float16)  [R typically 224]

You already created these shards with make_dino_supervision_h5.py.
"""

from __future__ import annotations
import os, glob, random, argparse
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# If your cluster filesystem has HDF5 locking issues:
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# -------------------------
# Dataset: random frame sampler across many shards
# -------------------------
class MultiH5PatchToDinoX(Dataset):
    """
    Returns one random frame per sample:
      patch: (P,D) float32
      x    : (3,R,R) float32  (dino_x supervision)
    """

    def __init__(
        self,
        h5_paths: list[str],
        *,
        handle_cache_size: int = 2,
        traj_len_trunc: int | None = None,
        max_tries: int = 50,
    ):
        self.paths = [str(p) for p in h5_paths]
        if not self.paths:
            raise ValueError("No HDF5 paths provided.")

        self.handle_cache_size = int(handle_cache_size)
        self.traj_len_trunc = traj_len_trunc
        self.max_tries = int(max_tries)

        self.meta = []
        P0 = D0 = R0 = None

        for p in self.paths:
            with h5py.File(p, "r") as f:
                if "patchtokens" not in f or "dino_x" not in f:
                    raise KeyError(f"{p} must contain 'patchtokens' and 'dino_x'.")

                N, T, P, D = f["patchtokens"].shape
                xN, xT, C, R, R2 = f["dino_x"].shape
                if xN != N or xT != T or C != 3 or R != R2:
                    raise ValueError(f"{p} dino_x shape {f['dino_x'].shape} incompatible with patchtokens {f['patchtokens'].shape}")

                if P0 is None:
                    P0, D0, R0 = P, D, R
                else:
                    if P != P0 or D != D0 or R != R0:
                        raise ValueError(f"Shape mismatch in {p}: got (P,D,R)=({P},{D},{R}) expected ({P0},{D0},{R0})")

                Teff = min(T, traj_len_trunc) if traj_len_trunc is not None else T
                if Teff < 1:
                    continue

                self.meta.append({"path": p, "N": N, "T": T, "Teff": Teff})

        if not self.meta:
            raise RuntimeError("No usable HDF5 shards.")

        self.P = P0
        self.D = D0
        self.R = R0

        self.weights = np.array([m["N"] for m in self.meta], dtype=np.float64)
        self.weights /= self.weights.sum()

        self._handles = None  # per-worker lazy

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

            traj = random.randrange(m["N"])
            t = random.randrange(m["Teff"])

            patch = f["patchtokens"][traj, t, :, :].astype(np.float32)  # (P,D)
            x = f["dino_x"][traj, t, :, :, :]                           # (3,R,R), may be f16
            x = np.asarray(x, dtype=np.float32)

            return {
                "patch": torch.from_numpy(patch),  # (P,D)
                "x": torch.from_numpy(x),          # (3,R,R)
            }

        raise RuntimeError("Failed to sample a valid item after many tries.")

    def __del__(self):
        if self._handles is not None:
            for _, f in self._handles.items():
                try:
                    f.close()
                except Exception:
                    pass


# -------------------------
# DINO-WM-style decoder: reshape tokens -> transposed conv
# -------------------------
class PatchTokensTransposedConvDecoder(nn.Module):
    """
    Input:
      patchtokens: (B,P,D)
    Steps:
      - reshape to (B,D,Hp,Wp) where Hp*Wp=P and (Hp==Wp==sqrt(P) in common cases)
      - ConvTranspose2d stack to upsample to ~256x256
      - final Conv2d to 3 channels
      - interpolate to target R if needed (e.g., 224)

    This matches the *spirit* of DINO-WM: direct spatial decoding from patch tokens.
    """

    def __init__(
        self,
        *,
        D: int,
        P: int,
        out_res: int = 224,
        base_ch: int = 256,
        out_channels: int = 3,
    ):
        super().__init__()
        self.D = int(D)
        self.P = int(P)
        self.out_res = int(out_res)
        self.out_channels = int(out_channels)

        s = int(round(self.P ** 0.5))
        if s * s != self.P:
            raise ValueError(f"P must be a perfect square for grid reshape. Got P={P}.")
        self.Hp = s
        self.Wp = s

        # 16x16 (for P=256) -> 32 -> 64 -> 128 -> 256
        # then interpolate to 224 (or whatever) to match supervision.
        self.net = nn.Sequential(
            # reduce channels a bit first (optional but helpful)
            nn.Conv2d(self.D, base_ch, kernel_size=1, stride=1, padding=0),

            nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=4, stride=2, padding=1),  # x2
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch // 2, base_ch // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch // 4, base_ch // 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_ch // 8, base_ch // 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_ch // 16, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, patchtokens: torch.Tensor) -> torch.Tensor:
        """
        patchtokens: (B,P,D)
        returns: (B,3,out_res,out_res) in the same space as dino_x (normalized)
        """
        B, P, D = patchtokens.shape
        assert P == self.P and D == self.D, f"Expected (P,D)=({self.P},{self.D}) got ({P},{D})"

        # (B,P,D) -> (B,D,P) -> (B,D,Hp,Wp)
        x = patchtokens.transpose(1, 2).contiguous().view(B, D, self.Hp, self.Wp)
        y = self.net(x)  # (B,3,256,256) when P=256

        if y.shape[-1] != self.out_res:
            y = F.interpolate(y, size=(self.out_res, self.out_res), mode="bilinear", align_corners=False)
        return y


# -------------------------
# Utils
# -------------------------
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


# -------------------------
# Train
# -------------------------
@dataclass
class Cfg:
    in_glob: str
    out: str
    batch: int = 64
    epochs: int = 50
    steps_per_epoch: int = 2000
    lr: float = 2e-4
    wd: float = 1e-4
    workers: int = 2
    seed: int = 0
    traj_trunc: int | None = None
    handle_cache_size: int = 2
    ckpt_every: int = 2000
    print_every: int = 50


def main():
    h5_root = "/storage/scratch1/6/rjiang77/Image_patch_dataset/run_*_with_dino_x.h5"
    out_path = "/storage/scratch1/6/rjiang77/unicycle_dino_wm_decoder/3_1_1"
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", type=str, default = h5_root,
                    help='e.g. "/storage/.../Image_patch_dataset/run_*_with_dino_x.h5"')
    ap.add_argument("--out", type=str, default = out_path,
                    help="output directory for ckpt_latest.pt")

    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj_trunc", type=int, default=0)
    ap.add_argument("--handle_cache_size", type=int, default=2)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--print_every", type=int, default=50)

    args = ap.parse_args()
    cfg = Cfg(
        in_glob=args.in_glob,
        out=args.out,
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
        print_every=args.print_every,
    )

    set_seed(cfg.seed)

    paths = sorted(glob.glob(cfg.in_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched glob: {cfg.in_glob}")
    print(f"[shards] found {len(paths)} files", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, flush=True)

    ds = MultiH5PatchToDinoX(
        h5_paths=paths,
        handle_cache_size=cfg.handle_cache_size,
        traj_len_trunc=cfg.traj_trunc,
    )
    print(f"[shape] P={ds.P}, D={ds.D}, R={ds.R}", flush=True)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=(cfg.workers > 0),
        prefetch_factor=2 if cfg.workers > 0 else None,
        timeout=60,
    )

    # Direct decode from patch tokens (no pooling)
    dec = PatchTokensTransposedConvDecoder(
        D=ds.D,
        P=ds.P,
        out_res=ds.R,
        base_ch=256,        # you can increase (e.g., 512) if you truly have plenty of GPU
        out_channels=3,
    ).to(device)

    opt = optim.AdamW(dec.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "ckpt_latest.pt"

    global_step = 0
    start_epoch = 1
    if latest_path.exists():
        ckpt = torch.load(latest_path, map_location="cpu")
        dec.load_state_dict(ckpt["dec"])
        opt.load_state_dict(ckpt["opt"])
        global_step = int(ckpt.get("global_step", 0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "scaler" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler"])
        print(f"[resume] loaded {latest_path} at step {global_step}", flush=True)

    it = iter(dl)

    # Since dino_x is already normalized (ImageNet mean/std), use L1 or MSE on that tensor.
    # L1 often looks nicer; MSE often trains a bit smoother.
    loss_fn = nn.L1Loss()

    for epoch in range(start_epoch, cfg.epochs + 1):
        dec.train()
        losses = []

        for step in range(cfg.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            patch = batch["patch"].to(device, non_blocking=True)  # (B,P,D)
            x_gt = batch["x"].to(device, non_blocking=True)       # (B,3,R,R)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                x_pred = dec(patch)  # (B,3,R,R)
                loss = loss_fn(x_pred, x_gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            losses.append(float(loss.item()))

            if (step + 1) % cfg.print_every == 0:
                print(f"epoch {epoch:03d} step {step+1:04d}/{cfg.steps_per_epoch} "
                      f"loss {np.mean(losses[-cfg.print_every:]):.6f}", flush=True)

            if (global_step % cfg.ckpt_every) == 0:
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "dec": dec.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "P": ds.P,
                    "D": ds.D,
                    "R": ds.R,
                    "cfg": cfg.__dict__,
                }
                atomic_torch_save(ckpt, latest_path)
                print(f"[ckpt] saved {latest_path} at step {global_step}", flush=True)

        print(f"[epoch {epoch:03d}] mean loss: {float(np.mean(losses)):.6f}", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()