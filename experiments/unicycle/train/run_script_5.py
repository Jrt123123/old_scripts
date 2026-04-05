import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import argparse


import os
import random

import time






# ---------------------- Dataset Loader ----------------------
class UnicycleHDF5Dataset(torch.utils.data.Dataset):
    """
    Loads the entire HDF5 file into memory (RAM) once at init.
    Much faster than lazy h5py indexing for small/medium datasets.
    """
    def __init__(self, h5_path,CLS_sliced_dim, traj_len=10):
        self.h5_path = str(h5_path)
        self.traj_len = traj_len

        with h5py.File(self.h5_path, "r") as f:
            dino_full = f["dinovecs"][:]   # (N, T, D)
            dino_sliced = dino_full[:, :, :CLS_sliced_dim]  # take only the first CLS_sliced_dim dimensions
            # Load into numpy arrays, then into torch tensors
            self.dino = torch.from_numpy(dino_sliced).float()  # (N, T, 768)
            self.ctrl = torch.from_numpy(f["controls"][:, :-1]).float()  # (N, T-1, 3)
            self.pos  = torch.from_numpy(f["positions"][:]).float()
            self.ori  = torch.from_numpy(f["orientations"][:]).float()

        self._file = None
        self.slice_size = CLS_sliced_dim

    def __len__(self):
        return self.dino.shape[0]

    def __getitem__(self, idx):
        return {
            "states": self.dino[idx],    # (T, 768)
            "controls": self.ctrl[idx],  # (T-1, 3)
        }
        
    #def __getitem__(self, idx):
    #    with h5py.File(self.h5_path, "r") as f:
    #        states = torch.from_numpy(f["dinovecs"][idx, :, :self.slice_size]).float()
    #        controls = torch.from_numpy(f["controls"][idx, :-1]).float()
    #    return {"states": states, "controls": controls}

    # ---------- tidy-up ----------
    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            # avoid noisy destructor exceptions on interpreter shutdown
            pass


# ---------------------- Models ----------------------
# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()

#         # Encoder: deeper + LayerNorm
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),

#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),

#             nn.Linear(128, latent_dim)
#         )

#         # Decoder: symmetric to encoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),

#             nn.Linear(128, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),

#             nn.Linear(256, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),

#             nn.Linear(512, input_dim)
#         )

#     def forward(self, z_hat):
#         z = self.encoder(z_hat)
#         x_hat = self.decoder(z)
#         return z, x_hat




class LatentDynamics(nn.Module):
    def __init__(self, state_dim=768, control_dim=2):
        super().__init__()
        in_dim = 2 * state_dim + 2 * control_dim  # (z_{t-1}, z_t, u_{t-1}, u_t)

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),

            nn.Linear(1024, state_dim),
        )

    def forward(self, z_tm1, z_t, u_tm1, u_t):
        # z_tm1, z_t: (B, D), u_tm1, u_t: (B, U)
        x = torch.cat([z_tm1, z_t, u_tm1, u_t], dim=-1)
        return self.fc(x)

    


# ---------- before the training loop ----------
DETACH_SCHEDULE = [
    (0,  1),   # epochs   0– 4 : detach EVERY step  (stable one-step learning)
    (5,  2),   # epochs   5–14 : detach every 2 steps
    (15, 4),   # epochs  15–29 : detach every 4 steps
    (30, None) # epochs ≥30   : NO detach  (full BPTT)
]
# helper
def detach_period(epoch):
    for start, period in reversed(DETACH_SCHEDULE):
        if epoch >= start:
            return period         # None  ⇒  no detaching at all
    



# ---------------------- Training ----------------------
def train(
    h5_root_path: str,
    save_path: str,
    traj_len: int = 10,
    latent_dim: int = 20,
    control_dim: int = 2,
    batch_size: int = 128,
    num_epochs: int = 50,
    device: str = None,
    lr: float = 1e-3,
    latent_dyn_weight: float = 0.1,
    obs_dyn_weight: float = 0.1,
    slice_ratio: int = 1,
    CLS_dim: int = 768
):
    

    

    sliced_CLS_dim = CLS_dim // slice_ratio
    print("Using CLS dimension:", CLS_dim)


    # ae = Autoencoder(input_dim=sliced_CLS_dim, latent_dim=latent_dim).to(device)

    # checkpoint = torch.load("/storage/home/hcoda1/6/rjiang77/autoencoder/11_14_1.pth", map_location=device)
    # ae.load_state_dict(checkpoint["model_state_dict"])

    # Freeze everything
    # for param in ae.parameters():
        # param.requires_grad = False

    # ae.eval()  # important: disable dropout/LN updates



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # dyn = LatentDynamics(latent_dim=latent_dim, control_dim=control_dim).to(device)
    dyn = LatentDynamics(state_dim=sliced_CLS_dim, control_dim=control_dim).to(device)

    optimizer = optim.Adam(list(dyn.parameters()), lr=1e-3)
    mse = nn.MSELoss()





    N_T = 1
    batch_count = 0

    # all_folders = sorted(glob(os.path.join(h5_root_path, "run_*/dataset.h5")))
    # h5_path = "combined_dataset/combined_dataset.h5"
    h5_path = "half_dataset.h5"
    dataset = UnicycleHDF5Dataset(h5_path,CLS_sliced_dim=sliced_CLS_dim,  traj_len=traj_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,         # try 8–16 if CPU has cores
        pin_memory=True,       # keeps tensors in page-locked memory for faster transfer
        prefetch_factor=4,     # each worker preloads batches ahead
        persistent_workers=True
    )


    for epoch in range(num_epochs):
        # ae.train()
        dyn.train()
        total_loss = 0


        if epoch <10:
            N_T = 1
        elif epoch <20:
            N_T = 2
        elif epoch <30:
            N_T = 3
        elif epoch <40:
            N_T = 4
        elif epoch <50:
            N_T = 5
        elif epoch <60:
            N_T = 6
        elif epoch <70:
            N_T = 7
        elif epoch <80:
            N_T = 8
        elif epoch <90:
            N_T = 9
        elif epoch <100:
            N_T = 10
        elif epoch <110:
            N_T = 11
        else:
            N_T = 11



        # Randomly pick one run folder
        
        
        #print(f"[Epoch {epoch}] Loading from {selected_path}")


        # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


        print("epoch: ",epoch)
        for batch in loader:
            batch_count +=1
            
            if batch_count % 200 ==0:
                print("Processing batch:", batch_count)
                start_time = time.time()

            
            states = batch['states'].to(device)     # (B, T, 384)
            ctrls  = batch['controls'].to(device)   # (B, T-1, 3)
            ctrls = ctrls[:, :, :control_dim]       # now (B, T, control_dim)

            B, T1, _ = states.shape
            T = T1 - 1
            N_T = min(N_T, T)  # don't exceed available controls/states

            # Encode all states to latent
            # with torch.no_grad():
                # z_all = ae.encoder(states.view(-1, sliced_CLS_dim)).view(B, T+1, latent_dim)

            z_all = states

            
            # Latent rollout
            # -------- 2-step rollout: (z_{t-1}, z_t, u_t) -> z_{t+1} --------
            latent_dyn_loss = 0.0
            period = detach_period(epoch)

            # seed with ground-truth first two states
            z_preds = [z_all[:, 0], z_all[:, 1]]  # z0, z1

            # We can only start predicting from t=1 (need z_{t-1} and z_t)
            # Predict z2..z_{N_T} (so number of predicted transitions here is N_T-1)
            steps_2 = max(N_T - 1, 1)
            steps_2 = min(steps_2, T - 1)  # because we access ctrls[:, t] and z_all[:, t+1]

            for t in range(1, steps_2 + 1):
                z_tm1 = z_preds[-2]
                z_t   = z_preds[-1]

                if period is not None and (t % period == 0):
                    z_tm1 = z_tm1.detach()
                    z_t   = z_t.detach()

                # use control u_t to predict z_{t+1}
                u_tm1 = ctrls[:, t-1]   # u_{t-1}
                u_t   = ctrls[:, t]     # u_t
                z_next = dyn(z_tm1, z_t, u_tm1, u_t)

                z_preds.append(z_next)
                latent_dyn_loss += mse(z_next, z_all[:, t + 1])  # match GT z_{t+1}

            z_preds = torch.stack(z_preds, dim=1)  # shape: (B, 2 + steps_2, D)
            latent_dyn_loss /= steps_2



            loss = (
                    latent_dyn_weight * latent_dyn_loss 
            )


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_count % 200 == 0:
                z_norms = torch.norm(z_all, dim=2)
                average_z_norm = z_norms.mean()
                

                #})
                print("epoch:" + str(epoch) +" batch:" + str(batch_count))

                print(latent_dyn_loss)

                print(loss)
                print("-----")
                # end_time = time.time()
                # print(f"Rest of computations time: {end_time - start_time:.4f} seconds")


        if epoch % 40 == 0:
            save_dir = save_path
            torch.save({
                "latent_dynamics_state_dict": dyn.state_dict(),
            }, f"{save_dir}{epoch}.pth")




# ---------------------- CLI ----------------------
if __name__ == "__main__":
    print("384 image, basic traj, no swd")
    # import time
    # time.sleep(39600)
    h5_root_path=""
    traj_len=12
    latent_dim=768
    batch_size=512
    num_epochs=600
    latent_dyn_weight=3.0
    obs_dyn_weight=1.0
    control_dim = 2
    slice_ratio = 1
    CLS_dim = 768

    save_path = '/storage/scratch1/6/rjiang77/Autoencoder/2_15_5/'


    train(
        h5_root_path=h5_root_path,
        save_path = save_path,
        traj_len=traj_len,
        latent_dim=latent_dim,
        control_dim=control_dim,
        batch_size=batch_size,
        num_epochs=num_epochs,
        latent_dyn_weight=latent_dyn_weight,
        obs_dyn_weight=obs_dyn_weight,
        slice_ratio=slice_ratio,
        CLS_dim = CLS_dim
    )