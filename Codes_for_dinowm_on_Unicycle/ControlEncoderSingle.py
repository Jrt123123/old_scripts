# action_encoder.py
import torch
import torch.nn as nn


class ControlEncoderSingle(nn.Module):
    """
    Simple action encoder:
        (B, T, action_dim) -> (B, T, emb_dim)
    """

    def __init__(self, action_dim, emb_dim=10):
        super().__init__()
        self.fc = nn.Linear(action_dim, emb_dim)

    def forward(self, a):
        return self.fc(a)