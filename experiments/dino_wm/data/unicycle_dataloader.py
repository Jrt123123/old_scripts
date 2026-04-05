# unicycle_dset.py

import torch
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from experiments.dino_wm.data.traj_dset import TrajDataset, get_train_val_sliced
import numpy as np


class UnicycleDataset(TrajDataset):
    """
    Multi-file HDF5 trajectory dataset compatible with DINO-WM TrajSlicerDataset.

    Expected per file:
      patchtokens  (N, T, P=256, D=768)
      positions    (N, T, 2)
      orientations (N, T)
      controls     (N, T, 2)
    """

    def __init__(self, data_root: str | Path, n_rollout: Optional[int] = None):
        self.data_root = Path(data_root)
        self.files = sorted(self.data_root.glob("run_*/dataset.h5"))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No run_*.h5 found under: {self.data_root}")

        # Build global index: global traj idx -> (file_id, traj_id)
        self.index_map: list[tuple[int, int]] = []
        self.file_sizes: list[int] = []

        # Read only metadata here (safe)
        for file_id, path in enumerate(self.files):
            with h5py.File(path, "r") as f:
                N = int(f["patchtokens"].shape[0])
                self.file_sizes.append(N)
                self.index_map.extend([(file_id, traj_id) for traj_id in range(N)])



        # randomly shuffle trajectories
        rng = np.random.default_rng(42)
        rng.shuffle(self.index_map)

        # Optionally cap number of trajectories used
        if n_rollout is not None:
            n_rollout = int(n_rollout)
            self.index_map = self.index_map[:n_rollout]
    
    

        self.total_traj = len(self.index_map)

        # Infer dims from first file (metadata read)
        with h5py.File(self.files[0], "r") as f:
            example = f["patchtokens"]
            self.traj_len = int(example.shape[1])
            self.num_patches = int(example.shape[2])  # should be 256
            self.token_dim = int(example.shape[3])    # should be 768

        # DINO-WM expects these attributes
        self.action_dim = 2
        self.state_dim = 3
        self.proprio_dim = 0

        # Per-worker file handle cache (lazy)
        self._h5_cache: Dict[int, h5py.File] = {}

        print(f"[UnicycleDataset] files={len(self.files)} total_traj={self.total_traj} "
              f"T={self.traj_len} P={self.num_patches} D={self.token_dim}")

    def __len__(self) -> int:
        return self.total_traj

    def get_seq_length(self, idx: int) -> int:
        return self.traj_len

    def _get_h5(self, file_id: int) -> h5py.File:
        # Lazy-open per worker/process; cache handle for speed
        if file_id not in self._h5_cache:
            self._h5_cache[file_id] = h5py.File(self.files[file_id], "r")
        return self._h5_cache[file_id]

    def get_all_actions(self) -> torch.Tensor:
        # Concatenate all actions across all trajectories you included.
        # Shape: (total_traj * T, 2)
        acts = []
        for global_idx in range(self.total_traj):
            file_id, traj_id = self.index_map[global_idx]
            h5 = self._get_h5(file_id)
            a = torch.as_tensor(h5["controls"][traj_id], dtype=torch.float32)  # (T,2)
            acts.append(a)
        return torch.cat(acts, dim=0).reshape(-1, self.action_dim)

    def get_frames(self, idx: int, frames) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        file_id, traj_id = self.index_map[idx]
        h5 = self._get_h5(file_id)

        patch = torch.as_tensor(h5["patchtokens"][traj_id, frames], dtype=torch.float32)  # (L,P,D)
        pos = torch.as_tensor(h5["positions"][traj_id, frames], dtype=torch.float32)     # (L,2)
        ori = torch.as_tensor(h5["orientations"][traj_id, frames], dtype=torch.float32).unsqueeze(-1)  # (L,1)
        proprio = torch.cat([pos, ori], dim=-1)                                          # (L,3)
        act = torch.as_tensor(h5["controls"][traj_id, frames], dtype=torch.float32)      # (L,2)

        state = proprio  # same here
        obs = {"visual": patch, "proprio": proprio}
        meta = {"traj_idx": int(idx), "file_id": int(file_id), "local_traj_id": int(traj_id)}

        return obs, act, state, meta

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))


def load_unicycle_slice_train_val(
    data_root: str | Path,
    *,
    n_rollout: int = 10000,
    split_ratio: float = 0.8,
    num_hist: int = 3,
    num_pred: int = 3,
    frameskip: int = 0,
):
    """
    Mirrors point_maze_dset.load_point_maze_slice_train_val().
    Returns:
      datasets["train"/"valid"] = sliced datasets (TrajSlicerDataset)
      traj_dset["train"/"valid"] = raw trajectory datasets
    """
    dset = UnicycleDataset(data_root=data_root, n_rollout=n_rollout)

    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )

    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset