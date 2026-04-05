import os
import subprocess  # <-- missing import
import sys

# Get array index (defaults to 0 if not under Slurm)
i = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Define output directory for this job
out = f"/storage/scratch1/6/rjiang77/DINOv2_CLS_672_image/2_20_dataset/run_{i}"
os.makedirs(out, exist_ok=True)

# Build command
cmd = [
    "python", "-m", "experiments.unicycle.data.generate_unicycle_dataset_manual_s",
    "--output", out,
    "--num_trajectories", "2000",
    "--traj_len", "12",
    "--radius", "20",
    "--img_dim", "672"
]

# Optional: print for logging
print(f"[Job {i}] Running:", " ".join(cmd))
sys.stdout.flush()

# Execute
subprocess.check_call(cmd)