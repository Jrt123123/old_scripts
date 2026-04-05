import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import gc
import argparse
from skimage.draw import polygon
from skimage.transform import resize
import h5py
import random
import time

# Your DINO wrapper (with extract_cls_and_patches enabled)
from experiments.dino_wm.models.DINO_vit_patch import DINOv2_ViT as Dinov2_VIT

from experiments.unicycle.data.unicycle_trajectory_generator_bounded_angle import unicycle_single_int_trajectory_generator


image_size = 224
radius = 10


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

    # Work at higher resolution
    high_res_size = image_size * upscale_factor
    high_cx, high_cy = cx * upscale_factor, cy * upscale_factor
    high_radius = radius * upscale_factor

    # Define arc (back of teardrop) at high resolution
    arc_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, arc_resolution)
    arc = np.stack(
        [high_radius * np.cos(arc_angles), high_radius * np.sin(arc_angles)], axis=1
    )

    # Define tip (nose), scaled for sharpness
    tip = np.array([[tip_scale * high_radius, 0.0]])

    # Combine arc and tip to form teardrop
    shape = np.vstack([arc[::-1], tip])

    # Apply rotation
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotated = shape @ R.T
    translated = rotated + np.array([high_cx, high_cy])

    # Create high-resolution image and rasterize polygon
    high_img = np.ones((high_res_size, high_res_size), dtype=np.float32)
    rr, cc = polygon(translated[:, 1], translated[:, 0], shape=high_img.shape)
    high_img[rr, cc] = 0.0

    # Downsample to target resolution (natural anti-aliasing)
    img = resize(
        high_img, (image_size, image_size), anti_aliasing=True, preserve_range=True
    )
    return img.astype(np.float32)


################################################################################
# ---------------------------  DATASET GENERATION  --------------------------- #
################################################################################


def save_dataset_hdf5_batched(
    output_path: Path | str,
    num_trajectories: int = 10_000,
    traj_len: int = 15,
    *,
    img_dim: int = 224,
    radius: int = 10,
    dt: float = 0.1,
    min_step: float = 3,
    max_step: float = 15.0,
    max_angle_step_deg: float = 30.0,
    chunk_size: int = 256,
    dino_model_name: str = "dinov2_vits14",
    dino_resize_size: int = 224,   # IMPORTANT: patch count depends on this (and patch size)
    dino_batch_size: int = 128,
    layer: int = 10,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    h5_file = output_path / "dataset.h5"
    if h5_file.exists():
        raise FileExistsError(f"HDF5 dataset already exists: {h5_file}")

    # ---------------------------------------------------------------- Models
    dinov2 = Dinov2_VIT(model_name=dino_model_name, resize_size=dino_resize_size)

    # ---------------------------------------------------------------- Bounds
    # (kept as in your code)
    xbound = (35, 102)
    ybound = (35, 90)

    # ---------------------------------------------------------------- Infer token dims once (P, D)
    # Use a single dummy image to infer patch count and embedding dim.
    dummy_img = generate_teardrop_mask(
        center=(100, 100), radius=radius, theta=0.0, image_size=img_dim
    )
    dummy_cls, dummy_patches = dinov2.extract_cls_and_patches([dummy_img], batch_size=1)
    feat_dim = int(dummy_cls.shape[-1])            # D
    n_patches = int(dummy_patches.shape[1])        # P
    print(f"[DINO] model={dino_model_name}, resize={dino_resize_size} -> D={feat_dim}, P={n_patches}")

    # Patch tokens are huge. Chunking per-trajectory is usually safer.
    patch_chunks = (1, traj_len, n_patches, feat_dim)

    # ----------------------------------------------------------------  HDF5
    with h5py.File(h5_file, "w") as f:
        pos_ds = f.create_dataset(
            "positions",
            shape=(0, traj_len, 2),
            maxshape=(None, traj_len, 2),
            dtype="f4",
            compression="lzf",
            chunks=(min(chunk_size, 256), traj_len, 2),
        )
        ori_ds = f.create_dataset(
            "orientations",
            shape=(0, traj_len),
            maxshape=(None, traj_len),
            dtype="f4",
            compression="lzf",
            chunks=(min(chunk_size, 256), traj_len),
        )
        ctrl_ds = f.create_dataset(
            "controls",
            shape=(0, traj_len, 2),
            maxshape=(None, traj_len, 2),
            dtype="f4",
            compression="lzf",
            chunks=(min(chunk_size, 256), traj_len, 2),
        )

        # CLS tokens
        dino_ds = f.create_dataset(
            "dinovecs",
            shape=(0, traj_len, feat_dim),
            maxshape=(None, traj_len, feat_dim),
            dtype="f4",
            compression="lzf",
            chunks=(min(chunk_size, 256), traj_len, feat_dim),
        )

        # Patch tokens
        patch_ds = f.create_dataset(
            "patchtokens",
            shape=(0, traj_len, n_patches, feat_dim),
            maxshape=(None, traj_len, n_patches, feat_dim),
            dtype="f4",
            compression="lzf",
            chunks=patch_chunks,
        )

        # ----------------------------------------------------- Generation loop
        n = 0
        # pbar = tqdm(total=num_trajectories, desc="Generating trajectories")
        while n < num_trajectories:
            start = (random.randint(*xbound), random.randint(*ybound))

            raw = unicycle_single_int_trajectory_generator(
                start,
                xbound,
                ybound,
                num_u=traj_len - 1,
                dt=dt,
                min_step=min_step,
                max_step=max_step,
                max_angle_step_deg=max_angle_step_deg,
            )
            if raw.shape[0] != traj_len - 1:
                #print("saw shape length", raw.shape[0], "expected", traj_len - 1)
                continue

            # ----------------------------------------------- Parse trajectory
            pos = np.zeros((traj_len, 2), dtype=np.float32)
            ori = np.zeros((traj_len,), dtype=np.float32)
            ctrl = np.zeros((traj_len, 2), dtype=np.float32)

            for t in range(traj_len - 1):
                (p_t, p_tp1, (θ_t, θ_tp1), u_t) = raw[t]
                pos[t] = p_t
                ori[t] = θ_t
                ctrl[t] = u_t
                if t == traj_len - 2:
                    pos[t + 1] = p_tp1
                    ori[t + 1] = θ_tp1

            # --------------------------------------- Render → DINOv2 tokens
            imgs = [
                generate_teardrop_mask(
                    center=tuple(p),
                    radius=radius,
                    theta=float(θ),
                    image_size=img_dim,
                )
                for p, θ in zip(pos, ori)
            ]

            # ONE forward for both CLS + patches
            cls_tokens_t, patch_tokens_t = dinov2.extract_cls_and_patches(
                imgs, batch_size=dino_batch_size, layer = layer
            )  # cls: (T,D), patches: (T,P,D)

            cls_tokens = cls_tokens_t.numpy().astype(np.float32)
            patch_tokens = patch_tokens_t.numpy().astype(np.float32)

            # Optional sanity (can comment out for speed)
            # if cls_tokens.shape != (traj_len, feat_dim):
            #     print("CLS shape mismatch:", cls_tokens.shape)
            #     continue
            # if patch_tokens.shape != (traj_len, n_patches, feat_dim):
            #     print("Patch shape mismatch:", patch_tokens.shape)
            #     continue

            # Append
            for ds, data in (
                (pos_ds, pos),
                (ori_ds, ori),
                (ctrl_ds, ctrl),
                (dino_ds, cls_tokens),
                (patch_ds, patch_tokens),
            ):
                ds.resize(n + 1, axis=0)
                ds[n] = data

            n += 1
            # pbar.update(1)

        # pbar.close()

    print(f"✓ Saved {num_trajectories} trajectories → {h5_file}")
    print(f"  - positions     : (N, T, 2)")
    print(f"  - orientations  : (N, T)")
    print(f"  - controls      : (N, T, 2)")
    print(f"  - dinovecs      : (N, T, D)      with D={feat_dim}")
    print(f"  - patchtokens   : (N, T, P, D)   with P={n_patches}, D={feat_dim}")


################################################################################
# ----------------------------------  CLI  ---------------------------------- #
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unicycle-DINOv2 dataset (CLS + patch tokens)")
    parser.add_argument("--output", type=Path, default="garbage", help="Output directory")
    parser.add_argument("--num_trajectories", "-n", type=int, default=2)
    parser.add_argument("--traj_len", type=int, default=12)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--img_dim", type=int, default=224)

    # DINO config
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--dino_resize_size", type=int, default=224)
    parser.add_argument("--dino_batch_size", type=int, default=128)
    parser.add_argument("--patch_token_layer",type=int,default = 0)

    args = parser.parse_args()

    save_dataset_hdf5_batched(
        output_path=args.output,
        num_trajectories=args.num_trajectories,
        traj_len=args.traj_len,
        img_dim=args.img_dim,
        radius=args.radius,
        dino_model_name=args.dino_model_name,
        dino_resize_size=args.dino_resize_size,
        dino_batch_size=args.dino_batch_size,
        layer = args.patch_token_layer,
        
    )