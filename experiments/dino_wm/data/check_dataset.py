#!/usr/bin/env python3
"""
check_control_stats.py

Compute statistics of control inputs (v, w) from the sliced dataset
that the model actually sees.

It reports:
- min / max
- mean / std
- percentiles
for both train and valid splits.

Usage:
    python check_control_stats.py \
        --data_root /storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset \
        --n_rollout 10000 \
        --split_ratio 0.8 \
        --num_hist 3 \
        --num_pred 1 \
        --frameskip 1 \
        --split train
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.dino_wm.data.unicycle_dataloader import load_unicycle_slice_train_val


def summarize_controls(all_act: torch.Tensor, name: str) -> None:
    """
    all_act: shape (N, 2), columns are [v, w]
    """
    if all_act.ndim != 2 or all_act.shape[1] != 2:
        raise ValueError(f"Expected control tensor of shape (N, 2), got {tuple(all_act.shape)}")

    v = all_act[:, 0].cpu().numpy()
    w = all_act[:, 1].cpu().numpy()

    def print_stats(x: np.ndarray, label: str) -> None:
        print(f"\n[{name}] {label}")
        print(f"  count      : {x.shape[0]}")
        print(f"  min        : {x.min():.8f}")
        print(f"  max        : {x.max():.8f}")
        print(f"  mean       : {x.mean():.8f}")
        print(f"  std        : {x.std():.8f}")
        print(f"  abs max    : {np.abs(x).max():.8f}")
        print(f"  p1         : {np.percentile(x, 1):.8f}")
        print(f"  p5         : {np.percentile(x, 5):.8f}")
        print(f"  p25        : {np.percentile(x, 25):.8f}")
        print(f"  p50        : {np.percentile(x, 50):.8f}")
        print(f"  p75        : {np.percentile(x, 75):.8f}")
        print(f"  p95        : {np.percentile(x, 95):.8f}")
        print(f"  p99        : {np.percentile(x, 99):.8f}")

    print_stats(v, "v")
    print_stats(w, "w")


def collect_controls_from_sliced_dataset(dataset, batch_size: int = 256, num_workers: int = 4) -> torch.Tensor:
    """
    Collect controls from a sliced dataset returned by load_unicycle_slice_train_val.
    Each item is assumed to be:
        obs, act, state
    or
        obs, act, state, meta
    where act has shape (L, 2).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    acts = []

    for batch in loader:
        if len(batch) == 3:
            obs, act, state = batch
        elif len(batch) == 4:
            obs, act, state, meta = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        # act shape: (B, L, 2)
        act = act.reshape(-1, act.shape[-1])   # -> (B*L, 2)
        acts.append(act)

    if not acts:
        raise RuntimeError("No controls collected from dataset.")

    return torch.cat(acts, dim=0)


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--data_root", type=str, default = "/storage/scratch1/6/rjiang77/DINOv2_patch_224/2_21_dataset")             #768d dataset
    parser.add_argument("--data_root", type=str, default = "/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset")    #384d dataset
    
    parser.add_argument("--n_rollout", type=int, default=2000)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--num_hist", type=int, default=3)
    parser.add_argument("--num_pred", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "both"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    datasets, traj_dset = load_unicycle_slice_train_val(
        data_root=args.data_root,
        n_rollout=args.n_rollout,
        split_ratio=args.split_ratio,
        num_hist=args.num_hist,
        num_pred=args.num_pred,
        frameskip=args.frameskip,
    )

    print("===== Dataset info =====")
    print(f"data_root    : {args.data_root}")
    print(f"n_rollout    : {args.n_rollout}")
    print(f"split_ratio  : {args.split_ratio}")
    print(f"num_hist     : {args.num_hist}")
    print(f"num_pred     : {args.num_pred}")
    print(f"frameskip    : {args.frameskip}")
    print(f"train slices : {len(datasets['train'])}")
    print(f"valid slices : {len(datasets['valid'])}")

    if args.split in ("train", "both"):
        all_act_train = collect_controls_from_sliced_dataset(
            datasets["train"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        summarize_controls(all_act_train, "train")

    if args.split in ("valid", "both"):
        all_act_valid = collect_controls_from_sliced_dataset(
            datasets["valid"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        summarize_controls(all_act_valid, "valid")


if __name__ == "__main__":
    main()