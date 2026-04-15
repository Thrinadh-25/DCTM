from __future__ import annotations

import torch
from torch import nn

from .base_model import DeepIDSBase


class _CNNBiLSTMNet(nn.Module):
    def __init__(self, input_dim, cnn_channels, lstm_hidden, dropout, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 1, F)
        z = self.conv(x)          # (B, C, F)
        z = z.transpose(1, 2)     # (B, F, C) — feature-axis becomes time
        out, _ = self.lstm(z)
        last = out[:, -1, :]
        return self.head(last)


class CNNBiLSTMIDS(DeepIDSBase):
    name = "cnn_bilstm"

    def _build_network(self) -> nn.Module:
        cnn_channels = self.kwargs.get("cnn_channels", 32)
        lstm_hidden = self.kwargs.get("lstm_hidden", 64)
        dropout = self.kwargs.get("dropout", 0.3)
        return _CNNBiLSTMNet(self.input_dim, cnn_channels, lstm_hidden, dropout, self.num_classes)

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x
