"""
train_world_model.py

Skeleton training script for a DINO-WM style world model.
Only structure + comments.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# --------------------------------------------------------
# 1. Imports
# --------------------------------------------------------


from visual_world_model import VWorldModel
from ControlEncoderSingle import ControlEncoderSingle
# or: from ControlEncoderMLP import ControlEncoderMLP
from Predictor import ViTPredictor



# --------------------------------------------------------
# 2. Configuration
# --------------------------------------------------------

# define hyperparameters
# - batch size
# - learning rate
# - number of epochs
# - history window
# - number of patches
# - embedding dimension
# - transformer depth / heads


# --------------------------------------------------------
# 3. Device setup
# --------------------------------------------------------

# Pick device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Refusing to run on CPU.")
device = torch.device("cuda")
print(f"[device] {device}")
print(f"[cuda] name: {torch.cuda.get_device_name(0)}")

# Basic info
print(f"[device] {device}")
if use_cuda:
    print(f"[cuda] name: {torch.cuda.get_device_name(0)}")
    print(f"[cuda] capability: {torch.cuda.get_device_capability(0)}")
    print(f"[cuda] torch.cuda.device_count(): {torch.cuda.device_count()}")


    
def train(args):
    # --------------------------------------------------------
    # 4. Load dataset
    # --------------------------------------------------------
    

    
    # adjust this import to match where you placed the file
    # e.g., if you put it under datasets/unicycle_dset.py, use:
    # from datasets.unicycle_dset import load_unicycle_slice_train_val
    from unicycle_dataloader import load_unicycle_slice_train_val
    
    data_root = "/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset"
    
    # Must match your model config
    num_hist = 2
    num_pred = 1
    frameskip = 1
    
    # How many trajectories to use (cap). If you want "all", set a very large number
    n_rollout = 10_000
    split_ratio = 0.8
    
    datasets, traj_dset = load_unicycle_slice_train_val(
        data_root=data_root,
        n_rollout=n_rollout,
        split_ratio=split_ratio,
        num_hist=num_hist,
        num_pred=num_pred,
        frameskip=frameskip,
    )
    
    full_train_dataset = datasets["train"]
    
    batch_size = 16
    num_workers = 4
    
    dataloader_train = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    dataloader_val = DataLoader(
        datasets["valid"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # quick sanity check (optional)
    obs, act, state = next(iter(dataloader_train))
    print("[batch] visual:", obs["visual"].shape)   # (B, L, 256, 768)
    print("[batch] proprio:", obs["proprio"].shape) # (B, L, 3)
    print("[batch] act:", act.shape)                # (B, L, 2)
    print("[batch] state:", state.shape)            # (B, L, 3)
    
    
    
    
    
    
    
    
    
    # --------------------------------------------------------
    # 5. Build model
    # --------------------------------------------------------
    
    # --------------------------------------------------------
    # Hyperparameters (should match dataset + config)
    # --------------------------------------------------------
    
    image_size = 224
    num_hist = 2
    num_pred = 1
    
    num_patches = 256
    token_dim = 384
    
    action_dim = 2
    action_emb_dim = 10
    
    num_action_repeat = 7
    num_proprio_repeat = 7
    
    
    # --------------------------------------------------------
    # Dummy encoder (because dataset already has patch tokens)
    # --------------------------------------------------------
    
    class PatchTokenEncoder(nn.Module):
    
        def __init__(self, emb_dim=768):
            super().__init__()
            self.emb_dim = emb_dim
            self.name = "dummy_encoder"
            self.patch_size = 16
    
        def forward(self, x):
            # x already contains patch tokens
            return x
    
    
    encoder = PatchTokenEncoder(token_dim)
    
    
    # --------------------------------------------------------
    # Proprio encoder (not used)
    # --------------------------------------------------------
    
    class DummyProprio(nn.Module):
    
        def forward(self, x):
            return x
    
    
    proprio_encoder = DummyProprio()
    
    
    # --------------------------------------------------------
    # Action encoder
    # --------------------------------------------------------
    
    action_encoder = ControlEncoderSingle(
        action_dim=action_dim,
        emb_dim=action_emb_dim
    )
    
    # alternatively
    # action_encoder = ControlEncoderMLP(action_dim, action_emb_dim)
    
    
    # --------------------------------------------------------
    # Predictor (Transformer)
    # --------------------------------------------------------
    
    predictor_dim = token_dim + action_emb_dim * num_action_repeat
    
    predictor = ViTPredictor(
        num_patches=num_patches,
        num_frames=num_hist,
        dim=predictor_dim,
        depth=6,
        heads=16,
        mlp_dim=2048
    )
    
    
    # --------------------------------------------------------
    # Decoder (not used for patch-token training)
    # --------------------------------------------------------
    
    decoder = None
    
    
    # --------------------------------------------------------
    # Build world model
    # --------------------------------------------------------
    
    model = VWorldModel(
        image_size=image_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        proprio_encoder=proprio_encoder,
        action_encoder=action_encoder,
        decoder=decoder,
        predictor=predictor,
        proprio_dim=0,
        action_dim=action_emb_dim,
        concat_dim=1,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=num_proprio_repeat,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
    )
    
    model = model.to(device)
    
    
    
    
    
    
    
    # --------------------------------------------------------
    # 6. Optimizer and loss
    # --------------------------------------------------------
    

    
    
    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    
    pred_learning_rate = 1e-4
    pred_weight_decay = 1e-4
    act_learning_rate = 1e-4
    act_weight_decay = 1e-4
    
    predictor_optimizer = optim.AdamW(
        model.predictor.parameters(),
        lr=pred_learning_rate,
        weight_decay=pred_weight_decay,
    )
    
    action_encoder_optimizer = optim.AdamW(
        model.action_encoder.parameters(),
        lr=act_learning_rate,
        weight_decay=act_weight_decay,
    )
    
    
    
    # --------------------------------------------------------
    # Resume from checkpoint if exists
    # --------------------------------------------------------
    
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    
    ckpt_files = list(ckpt_dir.glob("model_*.pt"))
    
    if ckpt_files:
        latest_ckpt = max(
            ckpt_files,
            key=lambda p: int(p.stem.split("_")[1])
        )
    
        print(f"[checkpoint] Loading {latest_ckpt}")
    
        checkpoint = torch.load(latest_ckpt, map_location=device)
    
        model.load_state_dict(checkpoint["model"])
    
        if "predictor_opt" in checkpoint:
            predictor_optimizer.load_state_dict(checkpoint["predictor_opt"])
    
        if "action_opt" in checkpoint:
            action_encoder_optimizer.load_state_dict(checkpoint["action_opt"])
    
        start_epoch = checkpoint["epoch"] + 1
    
        print(f"[checkpoint] Resuming from epoch {start_epoch}")
    
    else:
        print("[checkpoint] No checkpoint found. Starting fresh.")
    
    
    
    
    # --------------------------------------------------------
    # 7. Training loop  (train.py-style)
    #   Assumption: each batch from dataloader is (obs, act, state)
    #     - obs is a dict (e.g., obs["visual"], obs["proprio"] if present)
    #     - act is a tensor (B, T, action_dim) or whatever your dataset provides
    #     - state is unused here (kept for parity with original DINO-WM)
    # --------------------------------------------------------
    

    
    num_epochs = 100
    grad_clip_norm = 1.0  # set None to disable
    log_every = 50        # batches
    subset_size = 50000   # number of windows per epoch, tune this

    for epoch in range(start_epoch, num_epochs):
        model.train()
        
        n_total = len(full_train_dataset)
        n_use = min(subset_size, n_total)
    
        perm = torch.randperm(n_total)[:n_use]
        epoch_train_dataset = Subset(full_train_dataset, perm.tolist())
    
        dataloader_train = DataLoader(
            epoch_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        
    
        running_loss = 0.0
    
        for i, data in enumerate(tqdm(dataloader_train, desc=f"Epoch {epoch} Train")):
            obs, act, state = data  # matches train.py batch format :contentReference[oaicite:0]{index=0}
    
            # ---- move batch to device (obs is a dict)
            for k in obs.keys():
                obs[k] = obs[k].to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)
            # state is optional; keep it on CPU unless you need it
            # state = state.to(device, non_blocking=True)
    
    
            # ---- zero grads (same pattern as train.py) :contentReference[oaicite:1]{index=1}
            # If you have separate optimizers like in Part 6:
            if model.train_predictor:
                predictor_optimizer.zero_grad(set_to_none=True)
                action_encoder_optimizer.zero_grad(set_to_none=True)
    
            # If you kept an encoder_optimizer/decoder_optimizer, you'd follow train.py:
            # encoder_optimizer.zero_grad(set_to_none=True)
            # if has_decoder: decoder_optimizer.zero_grad(set_to_none=True)
            
            # ---- forward
            z_out, visual_out, visual_reconstructed, loss, loss_components = model(obs, act)
    
            # ---- backward
            loss.backward()
    
            # ---- optional grad clipping
            if grad_clip_norm is not None and grad_clip_norm > 0:
                # clip everything that is being trained
                params_to_clip = []
                if model.train_predictor:
                    params_to_clip += list(model.predictor.parameters())
                    params_to_clip += list(model.action_encoder.parameters())
                    if hasattr(model, "proprio_encoder") and model.proprio_encoder is not None:
                        params_to_clip += list(model.proprio_encoder.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
    
            # ---- step (same gating idea as train.py) :contentReference[oaicite:2]{index=2}
            if model.train_predictor:
                predictor_optimizer.step()
                action_encoder_optimizer.step()
    
            # ---- (optional) scheduler step
            # if scheduler is not None:
            #     scheduler.step()
    
            # ---- bookkeeping / logging
            running_loss += float(loss.detach().item())
    
            if (i + 1) % log_every == 0:
                avg = running_loss / log_every
                running_loss = 0.0
    
                # loss_components in their code is a dict; print a compact view
                # (yours is returned from VWorldModel.forward) :contentReference[oaicite:3]{index=3}
                comp_str = ""
                if isinstance(loss_components, dict):
                    # print a couple common keys if present
                    keys = list(loss_components.keys())
                    show_keys = keys[:4]
                    comp_str = " | " + " ".join(
                        [f"{k}:{float(loss_components[k]):.4g}" for k in show_keys]
                    )
    
                print(f"[epoch {epoch:03d} | batch {i:05d}] loss={avg:.6f}{comp_str}")
                
                
        save_every = 1
        # ---- end of epoch (optional) checkpoint
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        if (epoch + 1) % save_every == 0:
         torch.save(
             {
                 "epoch": epoch,
                 "model": model.state_dict(),
                 "predictor_opt": predictor_optimizer.state_dict(),
                 "action_opt": action_encoder_optimizer.state_dict(),
             },
             ckpt_dir / f"model_{epoch:04d}.pt",
         )
    
    

# --------------------------------------------------------
# 10. Entry point
# --------------------------------------------------------


import argparse

def main():
    parser = argparse.ArgumentParser()

    # ---- data
    parser.add_argument("--data_root", type=str,
                        default="/storage/scratch1/6/rjiang77/DINO_v2_unicycle_patch_384d/3_6_dataset")
    parser.add_argument("--n_rollout", type=int, default=10000)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--frameskip", type=int, default=1)

    # ---- training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1)

    # ---- model windowing
    parser.add_argument("--num_hist", type=int, default=2)
    parser.add_argument("--num_pred", type=int, default=1)

    # ---- optimizer
    parser.add_argument("--pred_lr", type=float, default=1e-4)
    parser.add_argument("--pred_wd", type=float, default=1e-4)
    parser.add_argument("--act_lr", type=float, default=1e-4)
    parser.add_argument("--act_wd", type=float, default=1e-4)
    parser.add_argument("--ckpt_dir", type=str, default="/storage/scratch1/6/rjiang77/My_DINO_WM_RESULTS/Checkpoints/3_14_3/")

    args = parser.parse_args()

    # ---- call your existing code blocks, but using args instead of hardcoded values
    # You can either:
    #   (A) move blocks 3-9 into functions and call them here (cleanest), or
    #   (B) keep your script as-is and just ensure these variables are set from args.

    # Minimal approach (B): set the variables your script already uses
    global num_hist, num_pred, frameskip, n_rollout, split_ratio
    global batch_size, num_workers, num_epochs, grad_clip_norm, save_every
    global pred_learning_rate, pred_weight_decay, act_learning_rate, act_weight_decay
    global data_root

    data_root = args.data_root
    n_rollout = args.n_rollout
    split_ratio = args.split_ratio
    frameskip = args.frameskip

    num_hist = args.num_hist
    num_pred = args.num_pred

    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    grad_clip_norm = args.grad_clip
    save_every = args.save_every

    pred_learning_rate = args.pred_lr
    pred_weight_decay = args.pred_wd
    act_learning_rate = args.act_lr
    act_weight_decay = args.act_wd

    # ---- If you refactor into functions later, this becomes:
    train(args)

    print("[main] starting training with:")
    print(vars(args))

    # IMPORTANT:
    # If your script currently runs training at import time (top-level for-loops),
    # you must wrap that training loop into a function and call it here.
    # Otherwise it will start before args are applied.


if __name__ == "__main__":
    main()
    
    
    