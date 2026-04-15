from __future__ import annotations

import torch
from torch import nn

from .base_model import DeepIDSBase


class MLPIDS(DeepIDSBase):
    name = "mlp"

    def _build_network(self) -> nn.Module:
        hidden = self.kwargs.get("hidden", [256, 128, 64])
        dropout = self.kwargs.get("dropout", 0.3)
        layers = []
        prev = self.input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, self.num_classes))
        return nn.Sequential(*layers)
