import torch
from torch import nn
from einops import rearrange
import numpy as np



def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    return torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,TP,TP)



class ViTPredictor(nn.Module):
    def __init__(self, num_patches, num_frames, dim, depth, heads, mlp_dim,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            num_patches=num_patches,
            num_frames=num_frames,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):  # x: (B, T*P, dim)
        n = x.shape[1]
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)
        return self.transformer(x)

        
    
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)



class Attention(nn.Module):
    def __init__(self, dim, num_patches, num_frames, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        # causal mask buffer (moves with the module device)
        bias = generate_mask_matrix(num_patches, num_frames)  # (1,1,TP,TP)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        B, T, C = x.size()  # here T == num_frames * num_patches
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.heads) for t in qkv]

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_patches, num_frames, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, num_patches=num_patches, num_frames=num_frames, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ])
            for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
        
        

        