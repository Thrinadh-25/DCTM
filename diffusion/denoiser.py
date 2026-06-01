"""Transformer-based denoiser for tabular diffusion.

Input:  x_t  [B, n_features], t [B] (long)
Output: predicted noise eps [B, n_features]
"""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        n_features: int,
        model_dim: int = 256,
        time_embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.model_dim = model_dim

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        # Each feature becomes its own token — enables real cross-feature attention
        self.feature_proj = nn.Linear(1, model_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, n_features)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x_t: (B, F) — treat each feature as a separate token
        h = self.feature_proj(x_t.unsqueeze(-1))  # (B, F, D)
        te = self.time_mlp(t).unsqueeze(1)          # (B, 1, D) broadcast over F
        h = h + te                                   # (B, F, D)
        h = self.encoder(h)                          # (B, F, D) — real cross-feature attention
        h = h.mean(dim=1)                            # (B, D) — pool over features
        return self.output_proj(h)                   # (B, F)
