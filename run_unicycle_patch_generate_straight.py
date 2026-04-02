import os
import subprocess
import sys

# Slurm array index
i = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# Loop over layers
for layer in range(1, 11):

    out = f"/storage/scratch1/6/rjiang77/DINOv2_patch_224_layers/layer_{layer}/run_{i}"
    os.makedirs(out, exist_ok=True)

    cmd = [
        "python", "generate_unicycle_patch_dataset_straight.py",
        "--output", out,
        "--num_trajectories", "2000",
        "--traj_len", "12",
        "--radius", "10",
        "--img_dim", "224",
        "--patch_token_layer", str(layer)
    ]

    print(f"[Job {i}] Layer {layer} Running:", " ".join(cmd))
    sys.stdout.flush()

    subprocess.check_call(cmd)