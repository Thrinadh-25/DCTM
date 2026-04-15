from __future__ import annotations

import torch
from torch import nn

from .base_model import DeepIDSBase


class _GRUHead(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, F, 1) — treat each feature as a timestep
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


class RNNIDS(DeepIDSBase):
    name = "rnn"

    def _build_network(self) -> nn.Module:
        hidden = self.kwargs.get("hidden_size", 128)
        num_layers = self.kwargs.get("num_layers", 2)
        dropout = self.kwargs.get("dropout", 0.3)
        return _GRUHead(self.input_dim, hidden, num_layers, dropout, self.num_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, F, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return x
