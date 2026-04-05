#!/usr/bin/env python3
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


import h5py
import numpy as np
from pathlib import Path
from skimage.draw import polygon
from skimage.transform import resize

from experiments.dino_wm.models.DINO_vit_patch import DINOv2_ViT as Dinov2_VIT

DATASET = Path("/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset/run_1/dataset.h5")

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

def cosine(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def compare_frame(traj_idx, t_idx, dinov2):
    with h5py.File(DATASET, "r") as f:
        pos = f["positions"][traj_idx, t_idx]
        th = f["orientations"][traj_idx, t_idx]
        stored = f["patchtokens"][traj_idx, t_idx]   # (P,D)

    img = generate_teardrop_mask(
        center=(float(pos[0]), float(pos[1])),
        radius=radius,
        theta=float(th),
        image_size=image_size,
    )

    # IMPORTANT: use default call, no layer argument
    _, patches = dinov2.extract_cls_and_patches([img], batch_size=1)
    current = patches[0].detach().cpu().numpy()

    diff = current - stored
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    cos = cosine(current, stored)

    return {
        "traj": traj_idx,
        "t": t_idx,
        "stored_mean": float(stored.mean()),
        "stored_std": float(stored.std()),
        "current_mean": float(current.mean()),
        "current_std": float(current.std()),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "cosine": cos,
    }

def main():
    dinov2 = Dinov2_VIT(model_name="dinov2_vits14", resize_size=224)

    test_indices = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
    ]

    results = []
    for traj_idx, t_idx in test_indices:
        try:
            r = compare_frame(traj_idx, t_idx, dinov2)
            results.append(r)
            print(r)
        except Exception as e:
            print(f"FAILED on traj={traj_idx}, t={t_idx}: {e}")

    if results:
        print("\n=== aggregate ===")
        print("avg mse   :", np.mean([r["mse"] for r in results]))
        print("avg rmse  :", np.mean([r["rmse"] for r in results]))
        print("avg mae   :", np.mean([r["mae"] for r in results]))
        print("avg cosine:", np.mean([r["cosine"] for r in results]))

if __name__ == "__main__":
    main()