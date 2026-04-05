#!/usr/bin/env python3
"""
mean_rse.py  –  evaluate mean Relative-Step-Error over *all* 10-step trajectories.

Usage
-----
python mean_rse.py \
       --model  autoencoder/unicycle/6_25_2/30.pth \
       --h5     unicycle/run_10/dataset.h5        \
       [--batch 256] [--eps 1e-8]
"""

import argparse, h5py, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.utils as nn_utils

# ------------------------------------------------------------------ models
# class Autoencoder(torch.nn.Module):
#     def __init__(self, input_dim, latent_dim):
#         super().__init__()
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(input_dim,512), torch.nn.LayerNorm(512), torch.nn.ReLU(),
#             torch.nn.Linear(512,256),       torch.nn.LayerNorm(256), torch.nn.ReLU(),
#             torch.nn.Linear(256,128),       torch.nn.LayerNorm(128), torch.nn.ReLU(),
#             torch.nn.Linear(128, latent_dim)
#         )
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim,128), torch.nn.LayerNorm(128), torch.nn.ReLU(),
#             torch.nn.Linear(128,256),        torch.nn.LayerNorm(256), torch.nn.ReLU(),
#             torch.nn.Linear(256,512),        torch.nn.LayerNorm(512), torch.nn.ReLU(),
#             torch.nn.Linear(512, input_dim)
#         )
#     def forward(self,x):                       # not needed here
#         z = self.encoder(x); return z, self.decoder(z)

# class LatentDynamics(torch.nn.Module):
#     def __init__(self, latent_dim, control_dim):
#         super().__init__()
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim+control_dim,64), torch.nn.ReLU(),
#             torch.nn.Linear(64,256), torch.nn.LayerNorm(256), torch.nn.ReLU(),
#             torch.nn.Linear(256,64), torch.nn.LayerNorm(64),  torch.nn.ReLU(),
#             torch.nn.Linear(64, latent_dim)
#         )
#     def forward(self,z,u): return self.fc(torch.cat([z,u],dim=-1))




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
            nn.GELU(),
            nn.Linear(64, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, z, u):
        return self.fc(torch.cat([z, u], dim=-1))




# ------------------------------------------------------------------ rollout
@torch.no_grad()
def free_rollout(ae,dyn,x0,ctrls):
    """
    x0 : (B,384)  first state           ctrls : (B,9,2)
    returns z_pred : (B,10,lat_dim)
    """
    zt  = ae.encoder(x0)                      # (B,z)
    outs=[zt]
    for t in range(ctrls.size(1)):            # 9 steps
        zt = dyn(zt, ctrls[:,t])              # (B,z)
        outs.append(zt)
    return torch.stack(outs,dim=1)            # (B,10,z)

# ------------------------------------------------------------------ main
def main(args):
    CLS_sliced_dim = args.CLS_dim // args.CLS_slice_ratio
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ae  = Autoencoder(CLS_sliced_dim,args.latent_dim).to(device)
    dyn = LatentDynamics(args.latent_dim, args.control_dim).to(device)

    ckpt = torch.load(args.model,map_location=device)
    ae.load_state_dict(ckpt["model_state_dict"])
    dyn.load_state_dict(ckpt["latent_dynamics_state_dict"])
    ae.eval(); dyn.eval()

    # ------------- iterate over the entire file (or files) -------------
    #h5_paths = [Path(p) for p in args.h5]
    h5_paths = [Path(args.h5)]
    rse_list = []
    final_err_list = []
    inf_norm_error = []
    first_step_inf_err_list = []

    all_z_pred_list = []
    all_z_nom_list = []

    for h5_path in h5_paths:
        with h5py.File(h5_path,'r') as f:
            X = torch.tensor(f["dinovecs"][:20000],  dtype=torch.float32)   # (N,10,384)
            X = X[:20000,:args.traj_length,:CLS_sliced_dim]                                 # slice if needed
            U = torch.tensor(f["controls"][:20000,:args.traj_length,:],  dtype=torch.float32)   # (N, 9, 2)
            pos = torch.tensor(f["positions"][:20000,:args.traj_length,:], dtype=torch.float32)   # (N, 10, 2)

            # print range of u with 2% 98% quantiles
            u_flat = U.view(-1, args.control_dim)  # (N*9, 2)
            u_p02 = torch.quantile(u_flat, 0.02, dim=0).numpy()
            u_p98 = torch.quantile(u_flat, 0.98, dim=0).numpy()

            N  = X.size(0)
            loader_idx = list(range(0,N,args.batch))
            for beg in tqdm(loader_idx, desc=f"{h5_path.name}"):
                end   = min(beg+args.batch, N)
                x0    = X[beg:end,0].to(device)
                ctrls = U[beg:end].to(device)

                z_flat = X[beg:end].reshape(-1, CLS_sliced_dim).to(device).float()
                z_nom  = ae.encoder(z_flat).view(end - beg, args.traj_length, args.latent_dim)
                z_pred= free_rollout(ae,dyn,x0,ctrls)                  # (B,10,z)

                all_z_pred_list.append(z_pred.detach().cpu())
                all_z_nom_list.append(z_nom.detach().cpu())

                # ---- per-step RSE  ----------------------------------
                # align lengths defensively
                T_pred, T_nom = z_pred.size(1), z_nom.size(1)
                T_common = min(T_pred, T_nom)          # typically 9 or 10

                z_pred_t = z_pred[:, 1:T_common]       # drop initial state
                z_nom_t  = z_nom[:, 1:T_common]

                step_err = (z_pred_t - z_nom_t).norm(dim=2)                # (B, T_common-1)
                step_nom = (z_nom[:, 1:T_common] - z_nom[:, :T_common-1]).norm(dim=2)  # (B, T_common-1)
                rse      = step_err / (step_nom + args.eps)


                if T_pred != T_nom:
                    print(f"Warning: pred len {T_pred}, nom len {T_nom}  (file {h5_path}, idx {beg})")



                rse_list.append(rse)

                final_err = (z_pred[:, -1] - z_nom[:, -1]).norm(dim=1).detach().cpu().numpy()
                nominal_lengths = step_nom.sum(dim=1).detach().cpu().numpy()
                final_err_list.append(final_err / (nominal_lengths + args.eps))


                # 1) compute absolute error tensor
                err = torch.abs(z_pred_t - z_nom_t)                  # shape (B, T, z_dim)

                # 2) infinity norm over the latent dimension
                #    err_inf[b, t] = max_i |z_pred[b,t,i] - z_nom[b,t,i]|
                err_inf = err.max(dim=2).values                 # shape (B, T)

                # 3) flatten over batches and time
                flat_err_inf = err_inf.detach().cpu().numpy().ravel()
                inf_norm_error.extend(flat_err_inf.tolist())
                first_step_inf_err = err_inf[:, 0].detach().cpu().numpy()  # first step only
                first_step_inf_err_list.extend(first_step_inf_err.tolist())

    print()
    w_max_empirical = np.percentile(inf_norm_error, 95)  # 99th percentile
    print(f"Empirical 95%-ile ∞-norm error: {w_max_empirical:.4f}")

    print(f"average first-step ∞-norm error:{ np.mean(first_step_inf_err_list):.4f}")
    print(f"median first-step ∞-norm error: { np.median(first_step_inf_err_list):.4f}")
    print(f"95%-ile first-step ∞-norm error: {np.percentile(first_step_inf_err_list, 95):.4f}")




    all_rse = torch.cat(rse_list,0)      # (N,9)
    final_err_list_np = np.concatenate(final_err_list)
    
    print(f"\nEvaluated {all_rse.numel()} steps over {all_rse.size(0)} trajectories")
    print(f"mean RSE per step : {torch.mean(all_rse,0).detach().cpu().numpy().round(3)}")
    print(f"overall mean RSE  : {all_rse.mean().item():.3f}")
    print(f"median RSE       : {torch.median(all_rse).item():.3f}")
    print(f"95-percentile RSE : {torch.quantile(all_rse,0.95).item():.3f}")

    print(f"\nMean normalized final L2 error: {final_err_list_np.mean():.3f}")
    print(f"95-percentile normalized final L2 error: {np.percentile(final_err_list_np, 95):.3f}")





    all_rse_np = all_rse.detach().cpu().numpy().ravel()

    # final_err_list_np is already a 1D NumPy array

    #print range of latent dimension
    all_z_pred = torch.cat(all_z_pred_list, dim=0)  # (total_trajs, T, latent_dim)
    all_z_nom = torch.cat(all_z_nom_list, dim=0)    # (total_trajs, T, latent_dim)
    # Flatten to get all latent values across trajectories and time
    z_pred_flat = all_z_pred.view(-1, args.latent_dim)  # (total_trajs*T, latent_dim)
    z_nom_flat = all_z_nom.view(-1, args.latent_dim)    # (total_trajs*T, latent_dim)

    # Combine predicted and nominal for overall statistics
    z_all = torch.cat([z_pred_flat, z_nom_flat], dim=0)  # (2*total_trajs*T, latent_dim)

    print(f"Latent dimension statistics over {z_all.size(0)} samples:")
    print("-" * 60)

    for i in range(args.latent_dim):
        z_dim = z_all[:, i]
        min_val = z_dim.min().item()
        max_val = z_dim.max().item()
        mean_val = z_dim.mean().item()
        std_val = z_dim.std().item()
        p02 = torch.quantile(z_dim, 0.02).item()
        p98 = torch.quantile(z_dim, 0.98).item()
        
        print(f"Dim {i:2d}: [{min_val:8.4f}, {max_val:8.4f}] "
            f"μ={mean_val:7.4f} σ={std_val:6.4f} "
            f"[2%:{p02:7.4f}, 98%:{p98:7.4f}]")

    # Also print separate statistics for predicted vs nominal
    print("\nPREDICTED vs NOMINAL comparison:")
    print("-" * 60)
    for i in range(args.latent_dim):
        pred_range = z_pred_flat[:, i].max() - z_pred_flat[:, i].min()
        nom_range = z_nom_flat[:, i].max() - z_nom_flat[:, i].min()
        print(f"Dim {i:2d}: Pred range={pred_range:.4f}, Nom range={nom_range:.4f}")
    



    # —– plot RSE distribution —–
    plt.figure(figsize=(6,4))
    plt.hist(all_rse_np, bins=50,range=(0.0, 1.25), alpha=0.7)
    plt.xlabel("Relative Step Error")
    plt.ylabel("Count")
    plt.title("Histogram of All RSE Values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # —– plot normalized final‐error distribution —–
    plt.figure(figsize=(6,4))
    plt.hist(final_err_list_np, bins=50,range=(0.0, 0.1), alpha=0.7)
    plt.xlabel("Normalized Final L2 Error")
    plt.ylabel("Count")
    plt.title("Histogram of Normalized Final Errors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        

# ------------------------------------------------------------------ CLI
if __name__ == "__main__":

    model = "/storage/home/hcoda1/6/rjiang77/autoencoder/12_22_2/460.pth"
    h5dataset_path = "/storage/scratch1/6/rjiang77/dataset/unicycle_up_12_21/All/combined_dataset.h5"
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=model, help="*.pth with model+dyn state_dict")
    ap.add_argument('--h5',    default=h5dataset_path, help="one or more dataset.h5 files")
    ap.add_argument('--latent_dim',  type=int, default=3)
    ap.add_argument('--control_dim', type=int, default=2)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--eps',   type=float, default=1e-8, help="denominator epsilon")
    ap.add_argument('--CLS_dim', type=int, default=768, help="DINOv2 CLS dim")
    ap.add_argument('--CLS_slice_ratio', type=int, default=1, help="If >1, use only the first 1/slice_ratio of the CLS vector")
    ap.add_argument('--traj_length', type=int, default=11, help="Number of state timepoints to use (T+1). e.g. 14 → uses first 14 states, ignores the last step")
    args = ap.parse_args()
    main(args)
