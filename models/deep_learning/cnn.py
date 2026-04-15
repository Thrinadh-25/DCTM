from __future__ import annotations

import torch
from torch import nn

from .base_model import DeepIDSBase


class CNNIDS(DeepIDSBase):
    name = "cnn"

    def _build_network(self) -> nn.Module:
        channels = self.kwargs.get("channels", [32, 64])
        k = self.kwargs.get("kernel_size", 3)
        dropout = self.kwargs.get("dropout", 0.3)

        layers = []
        in_ch = 1
        for c in channels:
            layers += [nn.Conv1d(in_ch, c, kernel_size=k, padding=k // 2), nn.ReLU()]
            in_ch = c
        layers.append(nn.AdaptiveMaxPool1d(1))
        conv = nn.Sequential(*layers)

        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_classes),
        )
        return nn.Sequential(conv, head)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F) -> (B, 1, F) for 1D convolution
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
