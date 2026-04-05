import os
import subprocess
import sys

# Slurm array index
i = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))



out = f"/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_18_{i}"
data_root = f"/storage/scratch1/6/rjiang77/DINOv2_patch_224_layers/layer_{i}"

cmd = [
    "python", "-m", "experiments.dino_wm.train.train_384",
    "--ckpt_dir", out,
    "--data_root",data_root,
]

print(f"[Job {i}] Running:", " ".join(cmd))
sys.stdout.flush()

subprocess.check_call(cmd)