import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import gc
import os
import argparse
from skimage.draw import polygon
from skimage.transform import resize
import h5py
from experiments.franca.models.Franca_vit import franca_vit as Dinov2_VIT


#from unicycle_trajectory_generator_bounded_angle import unicycle_single_int_trajectory_generator
#from unicycle_straight_line_generator import unicycle_single_int_trajectory_generator
from experiments.unicycle.data.unicycle_manual_s_shape_trajectory import unicycle_single_int_trajectory_generator
#from unicycle_trajectory_vertical_stricp import unicycle_single_int_trajectory_generator
import random
import time




image_size = 224
radius = 10
#r=10 for 224, 15 for 384
# dinov2_vit = Dinov2_VIT(model_name='dinov2_vits14', img_dim=image_size)









def generate_teardrop_mask(center=(10, 112), radius=10, theta=0, image_size=224, arc_resolution=200, tip_scale=3.0, upscale_factor=4):
    cx, cy = center

    # Work at higher resolution
    high_res_size = image_size * upscale_factor
    high_cx, high_cy = cx * upscale_factor, cy * upscale_factor
    high_radius = radius * upscale_factor

    # Define arc (back of teardrop) at high resolution
    arc_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, arc_resolution)
    arc = np.stack([
        high_radius * np.cos(arc_angles),
        high_radius * np.sin(arc_angles)
    ], axis=1)

    # Define tip (nose), scaled for sharpness
    tip = np.array([[tip_scale * high_radius, 0.0]])

    # Combine arc and tip to form teardrop
    shape = np.vstack([arc[::-1], tip])

    # Apply rotation
    R = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta),  np.cos(theta)]
    ])
    rotated = shape @ R.T
    translated = rotated + np.array([high_cx, high_cy])

    # Create high-resolution image and rasterize polygon
    high_img = np.ones((high_res_size, high_res_size), dtype=np.float32)
    rr, cc = polygon(translated[:,1], translated[:,0], shape=high_img.shape)
    high_img[rr, cc] = 0.0

    # Downsample to target resolution (this creates natural anti-aliasing)
    img = resize(high_img, (image_size, image_size), anti_aliasing=True, preserve_range=True)
    
    return img.astype(np.float32)



################################################################################
# ---------------------------  DATASET GENERATION  --------------------------- #
################################################################################

def save_dataset_hdf5_batched(
    output_path: Path | str,
    num_trajectories: int = 10_000,
    traj_len: int = 15,
    *,
    img_dim: int = 224,
    radius: int = 10,
    dt: float = 0.1,
    min_step: float = 3.0,
    max_step: float = 15.0,
    max_angle_step_deg: float = 30.0,
    chunk_size: int = 256,
) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    h5_file = output_path / "dataset.h5"
    if h5_file.exists():
       raise FileExistsError(f"HDF5 dataset already exists: {h5_file}")

    # ---------------------------------------------------------------- Models
    dinov2 = Dinov2_VIT()
    # model_path = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    # model_name = "dinov3_vitb16"
    # dinov2 = DinoV3(model_name, model_path)

    feat_dim = 768  # vits14 outputs 384‑D

    # ---------------------------------------------------------------- Bounds
    pad = radius * 2  # ensure full teardrop stays in‑frame
    xbound = (45,147)
    ybound = (45,147)
    
    #(35,102)(35,90) for 224, (45,147)(45,147) for 384

    # ----------------------------------------------------------------  HDF5
    with h5py.File(h5_file, "w") as f:
        pos_ds = f.create_dataset("positions", shape=(0, traj_len, 2), maxshape=(None, traj_len, 2), dtype="f4",
                                  compression="lzf", chunks=(chunk_size, traj_len, 2))
        ori_ds = f.create_dataset("orientations", shape=(0, traj_len), maxshape=(None, traj_len), dtype="f4",
                                  compression="lzf", chunks=(chunk_size, traj_len))
        ctrl_ds = f.create_dataset("controls", shape=(0, traj_len, 2), maxshape=(None, traj_len, 2), dtype="f4",
                                   compression="lzf", chunks=(chunk_size, traj_len, 2))
        dino_ds = f.create_dataset("dinovecs", shape=(0, traj_len, feat_dim), maxshape=(None, traj_len, feat_dim),
                                   dtype="f4", compression="lzf", chunks=(chunk_size, traj_len, feat_dim))

        # ----------------------------------------------------- Generation loop
        #pbar = tqdm(total=num_trajectories, desc="Generating trajectories")
        n = 0
        while n < num_trajectories:
            # Random start inside bounds
            start = (random.randint(*xbound), random.randint(*ybound))
            # print("Generating trajectory", n, "from start", start)
            raw = unicycle_single_int_trajectory_generator(
                start, xbound, ybound, num_u=traj_len - 1, dt=dt,
                min_step=min_step, max_step=max_step, max_angle_step_deg=max_angle_step_deg)
            if raw.shape[0] != traj_len - 1:
                #print("saw shape length", raw.shape[0], "expected", traj_len - 1)
                continue  # try again
            # print("trajectory:", raw)

            # ----------------------------------------------- Parse trajectory
            pos = np.zeros((traj_len, 2), dtype=np.float32)
            ori = np.zeros((traj_len,), dtype=np.float32)
            ctrl = np.zeros((traj_len, 2), dtype=np.float32)
            for t in range(traj_len - 1):
                (p_t, p_tp1, (θ_t, θ_tp1), u_t) = raw[t]
                pos[t] = p_t
                ori[t] = θ_t
                ctrl[t] = u_t
                if t == traj_len - 2:
                    pos[t + 1] = p_tp1
                    ori[t + 1] = θ_tp1
            # --------------------------------------- Render → DINOv2 embed

            # start_time = time.time()
            imgs = [generate_teardrop_mask(center=tuple(p), radius=radius, theta=θ, image_size=img_dim)
                    for p, θ in zip(pos, ori)]
            # print(f"Image rendering: {time.time() - start_time:.2f}s")
            # print("pos and ori")
            # print(pos)
            # print(ori)
            # print()

            # print("length of imgs:", len(imgs))

            # dinos = dinov2.get_dinovecs_from_image_list(imgs)  # → (traj_len, 384)
            dinos = dinov2.extract_features(imgs, batch_size=128)  # → (traj_len, 768)
            # check_if_dino_same(imgs,pos,ori,dinos)


            # Append
            for ds, data in ((pos_ds, pos), (ori_ds, ori), (ctrl_ds, ctrl), (dino_ds, dinos)):
                ds.resize(n + 1, axis=0)
                ds[n] = data
            n += 1
            #pbar.update(1)
        #pbar.close()
    print(f"✓ Saved {num_trajectories} trajectories → {h5_file}")

################################################################################
# ----------------------------------  CLI  ---------------------------------- #
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate unicycle‑DINOv2 dataset")
    parser.add_argument("--output", type=Path, default="garbage", help="Output directory")
    parser.add_argument("--num_trajectories", "-n", type=int, default=2)
    parser.add_argument("--traj_len", type=int, default=12)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--img_dim", type=int, default=224)
    args = parser.parse_args()

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark     = False
    # torch.use_deterministic_algorithms(True)

    save_dataset_hdf5_batched(
        output_path=args.output,
        num_trajectories=args.num_trajectories,
        traj_len=args.traj_len,
        img_dim=args.img_dim,
        radius=args.radius,
    )
