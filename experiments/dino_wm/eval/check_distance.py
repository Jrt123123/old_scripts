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

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.transform import resize

from experiments.dino_wm.models.DINO_vit_patch import DINOv2_ViT as Dinov2_VIT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_teardrop_mask(
    center=(10, 112),
    radius=10,
    theta=0.0,
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


def angle_wrap_scalar(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


@torch.no_grad()
def state_to_patchtokens(
    dinov2: Dinov2_VIT,
    x: float,
    y: float,
    th: float,
    image_size: int,
    radius: int,
    batch_size: int = 1,
    layer: int | None = None,
) -> torch.Tensor:
    img = generate_teardrop_mask(
        center=(float(x), float(y)),
        radius=radius,
        theta=float(th),
        image_size=image_size,
    )
    if layer is None:
        _, patches = dinov2.extract_cls_and_patches([img], batch_size=batch_size)
    else:
        _, patches = dinov2.extract_cls_and_patches([img], batch_size=batch_size, layer=layer)
    return patches[0].detach().cpu()


def patch_distance(a: torch.Tensor, b: torch.Tensor, mode: str = "mse") -> float:
    diff = a - b
    if mode == "mse":
        return float((diff.pow(2)).mean().item())
    if mode == "rmse":
        return float(torch.sqrt((diff.pow(2)).mean()).item())
    if mode == "fro":
        return float(diff.reshape(-1).norm().item())
    if mode == "cosine":
        aa = a.reshape(-1)
        bb = b.reshape(-1)
        denom = aa.norm() * bb.norm() + 1e-12
        return float(1.0 - torch.dot(aa, bb).item() / denom.item())
    raise ValueError(f"Unknown distance mode: {mode}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def binned_curve_plot(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    num_bins: int = 20,
) -> None:
    if len(x) < num_bins:
        return

    edges = np.linspace(x.min(), x.max(), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(num_bins, np.nan, dtype=np.float64)
    stds = np.full(num_bins, np.nan, dtype=np.float64)

    for i in range(num_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1] if i < num_bins - 1 else x <= edges[i + 1])
        if np.any(mask):
            means[i] = y[mask].mean()
            stds[i] = y[mask].std()

    valid = ~np.isnan(means)

    plt.figure(figsize=(6.5, 5.0))
    plt.errorbar(centers[valid], means[valid], yerr=stds[valid], marker="o", capsize=3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def sample_same_angle_pairs(
    dinov2: Dinov2_ViT,
    num_pairs: int,
    image_size: int,
    radius: int,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    theta_low: float,
    theta_high: float,
    layer: int | None,
    dist_mode: str,
    delta_xy: float,
) -> list[dict]:
    rows = []
    for k in range(num_pairs):
        th = np.random.uniform(theta_low, theta_high)

        #x1 = np.random.uniform(x_low, x_high)
        #y1 = np.random.uniform(y_low, y_high)
        #x2 = np.random.uniform(x_low, x_high)
        #y2 = np.random.uniform(y_low, y_high)
        
        
        # sample base point
        x1 = np.random.uniform(x_low, x_high)
        y1 = np.random.uniform(y_low, y_high)
        
        # small perturbation
        dx = np.random.normal(scale=delta_xy)
        dy = np.random.normal(scale=delta_xy)
        
        x2 = np.clip(x1 + dx, x_low, x_high)
        y2 = np.clip(y1 + dy, y_low, y_high)

        z1 = state_to_patchtokens(dinov2, x1, y1, th, image_size, radius, layer=layer)
        z2 = state_to_patchtokens(dinov2, x2, y2, th, image_size, radius, layer=layer)

        pixel_dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        token_dist = patch_distance(z1, z2, mode=dist_mode)

        rows.append({
            "pair_id": k,
            "theta": th,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "pixel_distance": pixel_dist,
            "token_distance": token_dist,
        })
    return rows


def sample_same_location_pairs(
    dinov2: Dinov2_ViT,
    num_pairs: int,
    image_size: int,
    radius: int,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    theta_low: float,
    theta_high: float,
    layer: int | None,
    dist_mode: str,
) -> list[dict]:
    rows = []
    for k in range(num_pairs):
        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)

        th1 = np.random.uniform(theta_low, theta_high)
        th2 = np.random.uniform(theta_low, theta_high)

        z1 = state_to_patchtokens(dinov2, x, y, th1, image_size, radius, layer=layer)
        z2 = state_to_patchtokens(dinov2, x, y, th2, image_size, radius, layer=layer)

        dtheta = abs(angle_wrap_scalar(th1 - th2))
        token_dist = patch_distance(z1, z2, mode=dist_mode)

        rows.append({
            "pair_id": k,
            "x": x,
            "y": y,
            "th1": th1,
            "th2": th2,
            "angle_distance": dtheta,
            "token_distance": token_dist,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_pairs", type=int, default=5000)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--dino_model_name", type=str, default="dinov2_vits14")
    parser.add_argument("--dino_resize_size", type=int, default=224)

    parser.add_argument("--x_low", type=float, default=35.0)
    parser.add_argument("--x_high", type=float, default=102.0)
    parser.add_argument("--y_low", type=float, default=35.0)
    parser.add_argument("--y_high", type=float, default=102.0)

    parser.add_argument("--theta_low", type=float, default=-math.pi)
    parser.add_argument("--theta_high", type=float, default=math.pi)

    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--dist_mode", type=str, default="fro",
                        choices=["mse", "rmse", "fro", "cosine"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="patchtoken_vs_pixel_plots")
    parser.add_argument("--delta_xy", type=float, default=2.0)

    args = parser.parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dinov2 = Dinov2_VIT(
        model_name=args.dino_model_name,
        resize_size=args.dino_resize_size,
    )

    # 1) same angle, varying location
    rows_same_angle = sample_same_angle_pairs(
        dinov2=dinov2,
        num_pairs=args.num_pairs,
        image_size=args.image_size,
        radius=args.radius,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        layer=args.layer,
        dist_mode=args.dist_mode,
        delta_xy = args.delta_xy,
    )
    write_csv(rows_same_angle, save_dir / "same_angle_pairs.csv")

    x_loc = np.array([r["pixel_distance"] for r in rows_same_angle], dtype=np.float64)
    y_loc = np.array([r["token_distance"] for r in rows_same_angle], dtype=np.float64)

    scatter_plot(
        x_loc, y_loc,
        xlabel="pixel-space distance in (x,y)",
        ylabel=f"patch-token distance ({args.dist_mode})",
        title="Same angle: patch-token distance vs location distance",
        out_path=save_dir / "same_angle_scatter.png",
    )
    binned_curve_plot(
        x_loc, y_loc,
        xlabel="pixel-space distance in (x,y)",
        ylabel=f"patch-token distance ({args.dist_mode})",
        title="Same angle: binned mean±std",
        out_path=save_dir / "same_angle_binned.png",
    )

    # 2) same location, varying angle
    rows_same_loc = sample_same_location_pairs(
        dinov2=dinov2,
        num_pairs=args.num_pairs,
        image_size=args.image_size,
        radius=args.radius,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
        theta_low=args.theta_low,
        theta_high=args.theta_high,
        layer=args.layer,
        dist_mode=args.dist_mode,
    )
    write_csv(rows_same_loc, save_dir / "same_location_pairs.csv")

    x_ang = np.array([r["angle_distance"] for r in rows_same_loc], dtype=np.float64)
    y_ang = np.array([r["token_distance"] for r in rows_same_loc], dtype=np.float64)

    scatter_plot(
        x_ang, y_ang,
        xlabel="wrapped angle difference |Δθ| (rad)",
        ylabel=f"patch-token distance ({args.dist_mode})",
        title="Same location: patch-token distance vs angle difference",
        out_path=save_dir / "same_location_scatter.png",
    )
    binned_curve_plot(
        x_ang, y_ang,
        xlabel="wrapped angle difference |Δθ| (rad)",
        ylabel=f"patch-token distance ({args.dist_mode})",
        title="Same location: binned mean±std",
        out_path=save_dir / "same_location_binned.png",
    )

    print("[saved]", save_dir / "same_angle_pairs.csv")
    print("[saved]", save_dir / "same_angle_scatter.png")
    print("[saved]", save_dir / "same_angle_binned.png")
    print("[saved]", save_dir / "same_location_pairs.csv")
    print("[saved]", save_dir / "same_location_scatter.png")
    print("[saved]", save_dir / "same_location_binned.png")


if __name__ == "__main__":
    main()