# action_encoder.py
import torch
import torch.nn as nn


class ControlEncoderMLP(nn.Module):
    """
    Two-layer MLP action encoder.
    """

    def __init__(self, action_dim, emb_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.GELU(),
            nn.Linear(32, emb_dim)
        )

    def forward(self, a):
        return self.net(a)