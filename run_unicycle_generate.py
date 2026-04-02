import subprocess
import time


# time.sleep(4000)  # wait for 2 seconds to
for i in range(0, 20):
    out = f"/storage/scratch1/6/rjiang77/dataset/unicycle_closely_spaced_384/run_{i}" 
    cmd = [
        "python", "generate_unicycle_dataset.py",
        "--output",out,
        "--num_trajectories", "10000",
        "--traj_len", "12",
        "--radius", "15",
        "--img_dim", "384"
    ]
    print("Calling:", " ".join(cmd))
    subprocess.check_call(cmd)

