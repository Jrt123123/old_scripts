import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from glob import glob
import os
import random

import time




def time_contrastive_loss_l2_control(z_all, u_all, alpha=1.0, distance_penalty=0.5, num_traj_sample=256):
    """
    L2-based Time-Contrastive Loss where margin scales with control magnitude.
    Args:
        z_all: (B, T, D)
        u_all: (B, T-1, U)
    """
    B, T, D = z_all.shape
    num_traj = min(num_traj_sample, B)
    sampled_idx = torch.randperm(B, device=z_all.device)[:num_traj]

    losses = []
    for b in sampled_idx:
        z_traj = z_all[b]           # (T, D)
        u_traj = u_all[b]           # (T-1, U)

        # pairwise distances
        dist = torch.cdist(z_traj, z_traj, p=2)  # (T, T)

        # compute accumulated control magnitudes
        ctrl_mag = torch.norm(u_traj, dim=1)     # (T-1,)
        cum_ctrl = torch.cumsum(ctrl_mag, dim=0)
        cum_ctrl = torch.cat([torch.zeros(1, device=z_all.device), cum_ctrl])  # length T

        # build margin matrix
        m_ij = alpha * (cum_ctrl[None, :] - cum_ctrl[:, None]).abs() / cum_ctrl[-1].clamp(min=1e-6)

        near_mask = (torch.abs(torch.arange(T, device=z_all.device)[:, None]
                              - torch.arange(T, device=z_all.device)[None, :]) <= 1).float()
        far_mask = 1 - near_mask

        close_loss = (dist * near_mask).mean()
        far_loss = (torch.relu(m_ij - dist) * far_mask).mean()

        losses.append(close_loss + distance_penalty * far_loss)

    return torch.stack(losses).mean()


def sliced_wasserstein_loss(z, prior=None, num_projections=256, device=None):
    """
    Compute Sliced Wasserstein Distance (SWD) between latent codes z and prior samples.

    Args:
        z:       (N, D) latent samples (e.g. flattened z_all)
        prior:   (N, D) prior samples; if None, sample from N(0, I)
        num_projections: number of random directions for slicing
    """
    if device is None:
        device = z.device
    N, D = z.shape

    # sample prior if not given
    if prior is None:
        # prior = torch.randn_like(z, device=device)
        prior = torch.empty_like(z).uniform_(-1, 1)

    # random projection directions
    dirs = torch.randn(num_projections, D, device=device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)

    # project both distributions
    proj_z = z @ dirs.T            # (N, K)
    proj_p = prior @ dirs.T        # (N, K)

    # sort along each projection
    proj_z_sorted, _ = proj_z.sort(dim=0)
    proj_p_sorted, _ = proj_p.sort(dim=0)

    # 1D Wasserstein distances averaged over all projections
    swd = (proj_z_sorted - proj_p_sorted).abs().mean()
    return swd

# def time_contrastive_loss_l2_control(z_all, u_all, alpha=1.0, distance_penalty=0.5, num_traj_sample=256):
#     """
#     Vectorized, batched Time-Contrastive Loss.
#     z_all: (B, T, D)
#     u_all: (B, T-1, U)
#     """
#     B, T, D = z_all.shape
#     num_traj = min(num_traj_sample, B)
#     idx = torch.randperm(B, device=z_all.device)[:num_traj]

#     # gather sampled trajectories
#     z_batch = z_all[idx]       # (M, T, D) where M=num_traj
#     u_batch = u_all[idx]       # (M, T-1, U)

#     # pairwise distances per trajectory -> (M, T, T)
#     dist = torch.cdist(z_batch, z_batch, p=2)

#     # accumulated control magnitudes per trajectory
#     ctrl_mag = torch.norm(u_batch, dim=2)           # (M, T-1)
#     cum_ctrl = torch.cumsum(ctrl_mag, dim=1)       # (M, T-1)
#     zeros = torch.zeros((num_traj, 1), device=z_all.device, dtype=cum_ctrl.dtype)
#     cum_ctrl = torch.cat([zeros, cum_ctrl], dim=1)  # (M, T)

#     total = cum_ctrl[:, -1].clamp(min=1e-6)         # (M,)
#     # margin matrix per trajectory -> (M, T, T)
#     m_ij = alpha * (cum_ctrl[:, None, :] - cum_ctrl[:, :, None]).abs() / total[:, None, None]

#     # masks (T, T) broadcast to (M, T, T)
#     ar = torch.arange(T, device=z_all.device)
#     near_mask = (torch.abs(ar[:, None] - ar[None, :]) <= 1).float()   # (T, T)
#     far_mask = 1.0 - near_mask

#     close_loss = (dist * near_mask[None, :, :]).mean(dim=(1, 2))   # (M,)
#     far_loss = (torch.relu(m_ij - dist) * far_mask[None, :, :]).mean(dim=(1, 2))  # (M,)

#     loss_per_traj = close_loss + distance_penalty * far_loss
#     return loss_per_traj.mean()



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

    # ---------- tidy-up ----------
    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            # avoid noisy destructor exceptions on interpreter shutdown
            pass


# ---------------------- Models ----------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        # Encoder: deeper + LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, latent_dim)
        )

        # Decoder: symmetric to encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, input_dim)
        )

    def forward(self, z_hat):
        z = self.encoder(z_hat)
        x_hat = self.decoder(z)
        return z, x_hat



class LatentDynamics(nn.Module):
    def __init__(self, latent_dim=20, control_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + control_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z, u):
        return self.fc(torch.cat([z, u], dim=-1))
    

def random_batch_from_dataset(dataset, batch_size, device=None):
    """Return one random batch (dict with 'states' and 'controls')."""
    n = len(dataset)
    idx = torch.randperm(n)[:batch_size]
    states = []
    ctrls = []
    for i in idx:
        item = dataset[int(i)]            # calls __getitem__
        states.append(item["states"])
        ctrls.append(item["controls"])
    states = torch.stack(states, dim=0)
    ctrls  = torch.stack(ctrls, dim=0)
    if device is not None:
        states = states.to(device)
        ctrls  = ctrls.to(device)
    return {"states": states, "controls": ctrls}
    


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



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    ae = Autoencoder(input_dim=sliced_CLS_dim, latent_dim=latent_dim).to(device)
    dyn = LatentDynamics(latent_dim=latent_dim, control_dim=control_dim).to(device)

    optimizer = optim.Adam(list(ae.parameters()) + list(dyn.parameters()), lr=1e-3)
    mse = nn.MSELoss()



    wandb.init(project="Unicycle_franca_in_Latent", entity="rjiang77-georgia-institute-of-technology")
    run = wandb.init(
        entity="rjiang77-georgia-institute-of-technology",
        project="Unicycle_franca_in_Latent",
        config={
            "learning_rate": 1e-3,
            "epochs": num_epochs,
            "batch size": batch_size,
            'latent dimension': latent_dim,
            'latent dynamic weight': latent_dyn_weight,
            'observed dynamic weight': obs_dyn_weight
        },
    )
    wandb.watch(ae, log="all", log_freq=100)



    N_T = 1
    batch_count = 0

    # all_folders = sorted(glob(os.path.join(h5_root_path, "run_*/dataset.h5")))
    h5_path = "combined_dataset/combined_dataset.h5"
    dataset = UnicycleHDF5Dataset(h5_path,CLS_sliced_dim=sliced_CLS_dim,  traj_len=traj_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,         # try 8–16 if CPU has cores
        pin_memory=True,       # keeps tensors in page-locked memory for faster transfer
        prefetch_factor=4,     # each worker preloads batches ahead
        persistent_workers=True
    )

    # f_set_path = "unicycle_upward_35_102_fake/All/combined_dataset.h5"
    # fff_set = UnicycleHDF5Dataset(f_set_path,CLS_sliced_dim=sliced_CLS_dim, traj_len=traj_len)
    # f_loader = DataLoader(
    #     fff_set,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=8,         # try 8–16 if CPU has cores
    #     pin_memory=True,       # keeps tensors in page-locked memory for faster transfer
    #     prefetch_factor=4,     # each worker preloads batches ahead
    #     persistent_workers=True
    # )

    for epoch in range(num_epochs):
        ae.train()
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


        # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


        
        for batch in loader:
            batch_count +=1
            
            if batch_count % 200 ==0:
                print("Processing batch:", batch_count)
                start_time = time.time()

            
            states = batch['states'].to(device)     # (B, T, 384)
            ctrls  = batch['controls'].to(device)   # (B, T-1, 3)
            B, T1, _ = states.shape
            T = T1 - 1

            # Encode all states to latent
            z_all = ae.encoder(states.view(-1, sliced_CLS_dim)).view(B, T+1, latent_dim)

            
            # Latent rollout
            z_preds = [z_all[:, 0]]
            latent_dyn_loss = 0.0
            period = detach_period(epoch)      # pick rule for this epoch

            for t in range(N_T):
                z_prev = z_preds[-1]
                if period is not None and (t % period == 0):
                    z_prev = z_prev.detach()   # truncate gradient every <period> steps

                z_next = dyn(z_prev, ctrls[:, t])
                z_preds.append(z_next)

                latent_dyn_loss += mse(z_next, z_all[:, t+1])


            z_preds = torch.stack(z_preds, dim=1)  # (B, T+1, latent_dim)
            latent_dyn_loss /= N_T


            #reconstruction loss
            x_flat = states.view(-1, sliced_CLS_dim)                         # (B*T, 384)
            z_flat, x_recon  = ae(x_flat)                          # (B*T, 384), (B*T, z_dim)
            rec_loss = mse(x_recon, x_flat)



            # Decode and compute reconstruction loss
            x_hat = ae.decoder(z_preds[:, 1:].reshape(-1, latent_dim))       # (B * N_T, 384)
            x_target = states[:, 1:N_T+1].reshape(-1, sliced_CLS_dim)                         # (B * N_T, 384)
            obs_dyn_loss = mse(x_hat, x_target)

            #variance loss 
            # prevent z norm sparse cheating
            # z_all: (B, T+1, z_dim) stack of all latent encodings
            # We reshape to (B*(T+1), z_dim) so we measure variance over all 
            # examples and timesteps at once:
            # var_penalty_weight = 0.05
            # z_flat = z_all.reshape(-1, latent_dim)          # (B*(T+1), z_dim)

            # var_per_dim = torch.var(z_flat,   dim=0)   # (z_dim,)
            # eps         = 1e-5
            # # If a var is near zero, 1/var blows up; we take the mean
            # var_penalty = torch.mean(1.0 / (var_per_dim + eps))



            # # PCA penalize unbalanced coordinate
            # pca_weight = 0.05
            # all_z = z_flat
            # cov = torch.cov(all_z.T)
            # eigvals = torch.linalg.eigvalsh(cov)
            # eigvals = eigvals[eigvals > 0]  # Keep only positive eigenvalues
            # eigvals = eigvals.clamp_min(0.0)
            # pca_penalty = (eigvals[-1] - eigvals[0])**2

            if batch_count % 200 ==0:
                end_time = time.time()
                print(f"Batch processing time: {end_time - start_time:.4f} seconds")

            #time contrastive loss
            tcl_weight = 0.05

            if batch_count % 200 ==0:
                print("Computing TCL...")
                #measure time
                start_time = time.time()
            tcl_loss_1 = time_contrastive_loss_l2_control(z_all, ctrls, alpha=1.0, distance_penalty=0.5, num_traj_sample=128)
            
            # if batch_count % 200 ==0:
            #     end_time = time.time()
            #     print(f"TCL computation time: {end_time - start_time:.4f} seconds")

            
            # use fake data to augment TCL
            f_batch = random_batch_from_dataset(fff_set, batch_size, device)
            f_states = f_batch['states'].to(device)     # (B, T, 384)
            f_ctrls  = f_batch['controls'].to(device)   # (B, T-1, 3)
            # sample 64 trajectories
            f_sample_states = f_states[:64]
            f_sample_ctrls  = f_ctrls[:64]
            z_state = ae.encoder(f_sample_states)

            B_f, T1_f, _ = f_states.shape
            T_f = T1_f - 1

            if epoch >= 5:
                # if batch_count % 200 ==0:
                #     print("Computing TCL on fake data...")
                #     #measure time
                #     start_time = time.time()
                tcl_loss_2 = time_contrastive_loss_l2_control(
                    z_all=z_state,
                    u_all=f_sample_ctrls,
                    alpha=25.0,
                    distance_penalty=1.5,
                    num_traj_sample=64
                )
                # if batch_count % 200 ==0:
                #     end_time = time.time()
                #     print(f"TCL on fake data computation time: {end_time - start_time:.4f} seconds")
            else:
                tcl_loss_2 = torch.tensor(0.0).to(device)
            tcl_loss = (tcl_loss_1 + tcl_loss_2) / 2

            # if batch_count % 200 ==0:
            #     print("the rest time:")
            #     start_time = time.time()

            # ----- SWD loss -----
            swd_weight = 0.01     # tune between 0.001–0.05
            z_flat = z_all.reshape(-1, latent_dim)
            swd_loss = sliced_wasserstein_loss(z_flat)




            loss = (rec_loss 
                    + latent_dyn_weight * latent_dyn_loss 
                    + obs_dyn_weight * obs_dyn_loss 
                    # + var_penalty_weight * var_penalty
                    # + pca_weight * pca_penalty
                    + tcl_weight * tcl_loss
                    + swd_weight * swd_loss
            )


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_count % 200 == 0:
                z_norms = torch.norm(z_all, dim=2)
                average_z_norm = z_norms.mean()
                
                wandb.log({
                    "reconstruction_loss": rec_loss.item(),
                    "latent_dynamic_loss": latent_dyn_loss.item(),
                    "observed_dynamic_loss": obs_dyn_loss.item(),
                    "average_z_norm": average_z_norm,
                    # "low_var_penalty": var_penalty.item(),
                    "tcl_loss": tcl_loss.item(),

                })
                print("epoch:" + str(epoch) +" batch:" + str(batch_count))
                print(rec_loss)
                print(latent_dyn_loss)
                print(obs_dyn_loss)
                print(tcl_loss)
                print(loss)
                print("-----")
                # end_time = time.time()
                # print(f"Rest of computations time: {end_time - start_time:.4f} seconds")


        save_dir = save_path
        torch.save({
            "model_state_dict": ae.state_dict(),
            "latent_dynamics_state_dict": dyn.state_dict(),
        }, f"{save_dir}{epoch}.pth")

        torch.save(ae.state_dict(), "autoencoder.pth")
        torch.save(dyn.state_dict(), "latent_dynamics.pth")


# ---------------------- CLI ----------------------
if __name__ == "__main__":
    import time
    # time.sleep(39600)
    h5_root_path=""
    traj_len=12
    latent_dim=3
    batch_size=3072
    num_epochs=600
    latent_dyn_weight=3.0
    obs_dyn_weight=1.0
    control_dim = 2
    slice_ratio = 1
    CLS_dim = 768

    save_path = './autoencoder/11_3_1/'


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
