#!/usr/bin/env python3
import os
import subprocess
import sys

# Slurm array index
i = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

ckpt_dir = f"/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_18_{i}"
ckpt_path = f"{ckpt_dir}/model_0020.pt"

out_dir = f"cem_out_{i}"

cmd = [
    "python", "test_CEM_all_layer.py",
    "--ckpt", ckpt_path,
    "--layer", str(i),
    "--out_dir", out_dir,
]

print(f"[Job {i}] Running:", " ".join(cmd))
sys.stdout.flush()

subprocess.check_call(cmd)