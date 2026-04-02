#!/usr/bin/env python3
"""
make_dino_supervision_h5.py

Create a new dataset.h5 containing:
  - dino_x     : the exact tensor fed to DINO (after transform+normalize)
  - patchtokens: copied from existing dataset.h5
Optionally also copy:
  - positions, orientations, controls

This is designed for decoder training: patchtokens -> image (supervised by dino_x).

Assumptions:
- Input HDF5 contains at least: patchtokens (N,T,P,D), positions (N,T,2), orientations (N,T)
  (controls optional but common)
- You have your DINO wrapper class available (the one you pasted), and it matches the old patchtokens.

Usage example:
  python -u make_dino_supervision_h5.py \
    --in_glob "/storage/.../run_*/dataset.h5" \
    --out_dir "/storage/.../decoder_supervision_h5" \
    --model_name dinov2_vitb14 \
    --resize_size 224 \
    --batch_frames 128
"""

from __future__ import annotations
import os
import glob
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# ---- import your wrapper (must match the patchtokens generation preprocessing!)
# Adjust import path/module name to where you saved the class.
from DINO_vit_patch import DINOv2_ViT as DINOv2_ViT

# If your cluster filesystem has HDF5 locking issues, keep this:
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def main():
    h5_root =  "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset/run_*/dataset.h5"
    out_path = "/storage/scratch1/6/rjiang77/Image_patch_dataset" 
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_glob", type=str, default = h5_root,
                    help='Glob of input shards, e.g. "/path/run_*/dataset.h5"')
    ap.add_argument("--out_dir", type=str, default = out_path,
                    help="Output directory to write new shards")
    ap.add_argument("--suffix", type=str, default="with_dino_x",
                    help="Name suffix for output shards")

    ap.add_argument("--model_name", type=str, default="dinov2_vitb14")
    ap.add_argument("--resize_size", type=int, default=224,
                    help="Must match how patchtokens were generated (DINO input size).")

    ap.add_argument("--batch_frames", type=int, default=128,
                    help="How many frames (images) to transform at once when building dino_x.")
    ap.add_argument("--dtype", type=str, default="f4", choices=["f4", "f2"],
                    help="dtype for dino_x storage: f4=float32, f2=float16 (storage doesn't matter -> use f4).")

    ap.add_argument("--copy_states", action="store_true",
                    help="Also copy positions/orientations/controls into new file.")

    args = ap.parse_args()

    in_paths = sorted(glob.glob(args.in_glob))
    if not in_paths:
        raise FileNotFoundError(f"No files matched: {args.in_glob}")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Build DINO wrapper (transform is inside; model is loaded but we do NOT run it here)
    # We only need transform to compute dino_x. (But wrapper loads model anyway.)
    dino = DINOv2_ViT(model_name=args.model_name, resize_size=args.resize_size)

    # Loop through shards
    for in_path in in_paths:
        in_path = Path(in_path)
        shard_name = in_path.parent.name  # e.g. run_11
        out_path = out_dir / f"{shard_name}_{args.suffix}.h5"
        out_path = out_dir / f"{shard_name}_{args.suffix}.h5"
        if out_path.exists():
            print(f"[skip] exists: {out_path}", flush=True)
            continue

        with h5py.File(in_path, "r") as fin:
            if "patchtokens" not in fin:
                raise KeyError(f"{in_path} missing 'patchtokens'")

            patch = fin["patchtokens"]
            N, T, P, D = patch.shape

            # Sanity: require positions/orientations to regenerate the same masks as before,
            # BUT: you asked to save "image seen by DINO" and copy patchtokens.
            # If your old dataset did NOT store images, you must regenerate them somewhere.
            # In your current pipeline, you regenerate from (positions, orientations).
            # So we require these to exist.
            if "positions" not in fin or "orientations" not in fin:
                raise KeyError(f"{in_path} missing 'positions'/'orientations' needed to regenerate images.")

            pos = fin["positions"]        # (N,T,2)
            ori = fin["orientations"]     # (N,T)

            ctrl = fin["controls"] if ("controls" in fin) else None
            
            
            traj_i = 0
            pos0 = pos[traj_i]                         # use the already-defined pos/ori handles
            ori0 = ori[traj_i]
            pt0  = patch[traj_i].astype(np.float32)
            
            imgs0 = render_masks_from_states(
                positions=pos0,
                orientations=ori0,
                image_size=224,                          # EXACT same size as you will store in dino_x
            )
            
            _, pt_new0 = dino.extract_cls_and_patches(imgs0, batch_size=64)
            pt_new0 = pt_new0.numpy().astype(np.float32)
            
            max_abs = float(np.max(np.abs(pt_new0 - pt0)))
            print(f"[preflight] max_abs diff = {max_abs:.6e}", flush=True)
            
            if max_abs > 1e-5:
                raise RuntimeError("Preflight check failed: regenerated patchtokens don't match stored ones.")
        
            
            
            

            # Create output
            with h5py.File(out_path, "w") as fout:
                # ---- copy patchtokens (exact)
                patch_ds = fout.create_dataset(
                    "patchtokens",
                    shape=(N, T, P, D),
                    dtype="f4",
                    compression="lzf",
                    chunks=(1, T, P, D),
                )

                # ---- store "image seen by DINO": normalized float tensor (T,3,R,R)
                R = int(args.resize_size)
                x_ds = fout.create_dataset(
                    "dino_x",
                    shape=(N, T, 3, R, R),
                    dtype=args.dtype,
                    compression="lzf",
                    chunks=(1, T, 3, R, R),
                )

                # ---- optional state copy
                if args.copy_states:
                    fout.create_dataset(
                        "positions",
                        data=pos.astype(np.float32),
                        dtype="f4",
                        compression="lzf",
                        chunks=(1, T, 2),
                    )
                    fout.create_dataset(
                        "orientations",
                        data=ori.astype(np.float32),
                        dtype="f4",
                        compression="lzf",
                        chunks=(1, T),
                    )
                    if ctrl is not None:
                        fout.create_dataset(
                            "controls",
                            data=ctrl.astype(np.float32),
                            dtype="f4",
                            compression="lzf",
                            chunks=(1, T, 2),
                        )

                # ---- metadata
                fout.attrs["source_file"] = str(in_path)
                fout.attrs["dino_model_name"] = args.model_name
                fout.attrs["dino_resize_size"] = R
                fout.attrs["dino_preprocess"] = (
                    "ensure_rgb->Resize(int(R*256/224),bicubic)->CenterCrop(R)->ToTensor()->ImageNetNormalize"
                )

                # ---- copy patchtokens efficiently
                # If the dataset is huge, copying all at once may be too much RAM.
                # We'll stream by trajectory.
                print(f"[write] {out_path}  (N={N}, T={T}, P={P}, D={D})")
                
                
                
                # after you load one shard and before the big loop
                

                for i in range(N):
                    # 1) copy patchtokens trajectory
                    patch_ds[i] = patch[i].astype(np.float32)

                    # 2) regenerate images and compute dino_x using the SAME transform
                    #    Your old generator created teardrop masks from (pos, theta).
                    #    If you want EXACT match to old patchtokens, you MUST regenerate masks
                    #    with the SAME function and same params used originally (radius, tip_scale, etc.).
                    #    Here we assume you will import and call the same renderer you used.
                    #
                    #    IMPORTANT: This script is only doing transform-to-dino_x.
                    #    You still need to provide 'imgs' (np arrays / PIL) for the transform.
                    #
                    #    So we call a user-provided renderer hook below.
                    imgs = render_masks_from_states(
                        positions=pos[i], orientations=ori[i],
                        image_size=R,
                    )

                    # Transform -> tensor batch x: (T,3,R,R), already normalized
                    # (This is literally what your wrapper feeds to the DINO model.)
                    x = dino._to_tensor_batch(imgs).cpu().numpy()

                    if args.dtype == "f2":
                        x_ds[i] = x.astype(np.float16)
                    else:
                        x_ds[i] = x.astype(np.float32)
                        
                    if (i + 1) % 200 == 0:
                        print(f"{shard_name}: {i+1}/{N}", flush=True)

                    

                

        print(f"✓ wrote {out_path}")


from skimage.draw import polygon
from skimage.transform import resize
import numpy as np

RADIUS = 10          # set to the exact radius you used when making patchtokens
TIP_SCALE = 3.0
UPSCALE = 4
ARC_RES = 200

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
    arc = np.stack([high_radius * np.cos(arc_angles), high_radius * np.sin(arc_angles)], axis=1)

    tip = np.array([[tip_scale * high_radius, 0.0]])
    shape = np.vstack([arc[::-1], tip])

    Rm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    rotated = shape @ Rm.T
    translated = rotated + np.array([high_cx, high_cy])

    high_img = np.ones((high_res_size, high_res_size), dtype=np.float32)
    rr, cc = polygon(translated[:, 1], translated[:, 0], shape=high_img.shape)
    high_img[rr, cc] = 0.0

    img = resize(high_img, (image_size, image_size), anti_aliasing=True, preserve_range=True)
    return img.astype(np.float32)

def render_masks_from_states(*, positions: np.ndarray, orientations: np.ndarray, image_size: int):
    imgs = [
        generate_teardrop_mask(
            center=(float(p[0]), float(p[1])),
            radius=RADIUS,
            theta=float(th),
            image_size=image_size,
            arc_resolution=ARC_RES,
            tip_scale=TIP_SCALE,
            upscale_factor=UPSCALE,
        )
        for p, th in zip(positions, orientations)
    ]
    return imgs

if __name__ == "__main__":
    main()