import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

def concat_h5_datasets(root_dir: str, output_path: str, traj_len:int):
    root = Path(root_dir)
    output_path = Path(output_path)
    # run_paths = sorted(root.glob("run_*/dataset.h5"))
    # only combine sorted 1-10
    run_paths = [root / f"run_{i}" / "dataset.h5" for i in range(0, 300,1) if (root / f"run_{i}" / "dataset.h5").exists()]

    if not run_paths:
        raise FileNotFoundError(f"No dataset.h5 files found under {root}")

    print(f"Found {len(run_paths)} dataset files.")
    keys = None
    total_samples = 0
    per_file_counts = []

    # Step 1: Scan all files to determine total size and check compatibility
    for path in run_paths:
        with h5py.File(path, "r") as f:
            if keys is None:
                keys = list(f.keys())
                # store original shapes and dtypes, and compute truncated shape (truncate time axis to traj_len)
                shape_map = {k: f[k].shape[1:] for k in keys}
                dtype_map = {k: f[k].dtype for k in keys}
                desired_shape_map = {}
                for k in keys:
                    orig = shape_map[k]
                    if len(orig) >= 1:
                        time_dim = orig[0]
                        trunc_time = min(time_dim, traj_len)
                        desired_shape_map[k] = (trunc_time,) + orig[1:]
                    else:
                        desired_shape_map[k] = orig
            else:
                for k in keys:
                    # ensure non-time dimensions match
                    if f[k].shape[2:] != shape_map[k][1:]:
                        raise ValueError(f"Shape mismatch in {path} for key '{k}' (non-time dims differ)")
                    # dtype check
                    if f[k].dtype != dtype_map[k]:
                        raise ValueError(f"Dtype mismatch in {path} for key '{k}'")
            num_samples = f[keys[0]].shape[0]
            total_samples += num_samples
            per_file_counts.append(num_samples)

    print(f"Total samples after concat: {total_samples}")

    # Step 2: Create output file and datasets (use truncated time dims)
    with h5py.File(output_path, "w") as fout:
        out_datasets = {}
        for k in keys:
            out_shape = (total_samples,) + desired_shape_map[k]
            out_datasets[k] = fout.create_dataset(
                k, shape=out_shape, dtype=dtype_map[k], compression="lzf"
            )

        # Step 3: Copy data (truncating time axis)
        current_index = 0
        for path, count in zip(run_paths, per_file_counts):
            with h5py.File(path, "r") as f:
                for k in keys:
                    desired = desired_shape_map[k]
                    if len(desired) >= 1:
                        # slice first desired[0] timesteps along axis=1
                        out_datasets[k][current_index:current_index+count] = f[k][:, :desired[0], ...]
                    else:
                        out_datasets[k][current_index:current_index+count] = f[k][:]
            current_index += count
            print(f"Appended {path.name} → total: {current_index}")

    print(f"\n✓ Combined dataset saved to: {output_path}")
    
    
    
    
def concat_h5_datasets_new(root_dir: str, output_path: str, traj_len: int):
    root = Path(root_dir)
    output_path = Path(output_path)
    run_paths = [root / f"run_{i}" / "dataset.h5" for i in range(0, 300, 1) if (root / f"run_{i}" / "dataset.h5").exists()]

    if not run_paths:
        raise FileNotFoundError(f"No dataset.h5 files found under {root}")

    print(f"Found {len(run_paths)} dataset files.")
    
    valid_run_paths = []
    keys = None
    total_samples = 0
    per_file_counts = []

    # Step 1: Scan all files with error handling
    for path in run_paths:
        try:
            with h5py.File(path, "r") as f:
                # Basic check: is it really an HDF5 with data?
                if not f.keys():
                    print(f"⚠️ Skipping {path}: HDF5 file has no datasets.")
                    continue

                if keys is None:
                    keys = list(f.keys())
                    shape_map = {k: f[k].shape[1:] for k in keys}
                    dtype_map = {k: f[k].dtype for k in keys}
                    desired_shape_map = {}
                    for k in keys:
                        orig = shape_map[k]
                        if len(orig) >= 1:
                            time_dim = orig[0]
                            trunc_time = min(time_dim, traj_len)
                            desired_shape_map[k] = (trunc_time,) + orig[1:]
                        else:
                            desired_shape_map[k] = orig
                else:
                    # Validate compatibility
                    for k in keys:
                        if k not in f:
                            raise ValueError(f"Key '{k}' missing in {path}")
                        if f[k].shape[2:] != shape_map[k][1:]:
                            raise ValueError(f"Shape mismatch in {path} for key '{k}' (non-time dims differ)")
                        if f[k].dtype != dtype_map[k]:
                            raise ValueError(f"Dtype mismatch in {path} for key '{k}'")

                num_samples = f[keys[0]].shape[0]
                total_samples += num_samples
                per_file_counts.append(num_samples)
                valid_run_paths.append(path)

        except OSError as e:
            print(f"❌ Skipping corrupted or invalid file: {path}\n    Error: {e}")
            continue
        except Exception as e:
            print(f"❌ Unexpected error reading {path}: {e}")
            continue

    if not valid_run_paths:
        raise RuntimeError("No valid HDF5 files found after filtering.")

    print(f"✅ Valid files: {len(valid_run_paths)} | Total samples: {total_samples}")

    # Step 2: Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as fout:
        out_datasets = {}
        for k in keys:
            out_shape = (total_samples,) + desired_shape_map[k]
            out_datasets[k] = fout.create_dataset(
                k, shape=out_shape, dtype=dtype_map[k], compression="lzf"
            )

        # Step 3: Copy data
        current_index = 0
        for path, count in zip(valid_run_paths, per_file_counts):
            try:
                with h5py.File(path, "r") as f:
                    for k in keys:
                        desired = desired_shape_map[k]
                        if len(desired) >= 1:
                            out_datasets[k][current_index:current_index+count] = f[k][:, :desired[0], ...]
                        else:
                            out_datasets[k][current_index:current_index+count] = f[k][:]
                current_index += count
                print(f"Appended {path} → total: {current_index}")
            except Exception as e:
                print(f"⚠️ Failed to copy data from {path}, skipping: {e}")
                # Note: This would leave a gap in the output. To avoid that,
                # you'd need to recompute counts. But usually, if scan passed, copy should work.

    print(f"\n✓ Combined dataset saved to: {output_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # time.sleep(43200)  # wait for 12 hours to ensure all runs are complete
    
    root_dir = "/storage/scratch1/6/rjiang77/DINOv2_CLS_672_image/2_14_dataset"
    output_path = "/storage/scratch1/6/rjiang77/DINOv2_CLS_672_image/2_14_dataset/All/combined_dataset.h5"
    traj_len = 12  # or 15 depending on your runs

    concat_h5_datasets_new(root_dir, output_path,traj_len)
