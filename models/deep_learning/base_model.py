"""Shared PyTorch training utilities for deep IDS models.

All deep models:
  - respect `utils.device.get_device()` → CUDA if available, else CPU (with
    a clear warning printed to the user)
  - use an early-stopping train loop with a held-out validation split
  - expose a uniform .train / .predict / .predict_proba / .save / .load API
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.device import get_device, is_cuda
from utils.logger import get_logger

_log = get_logger("deep_base")


class DeepIDSBase:
    name = "deep_base"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 30,
        patience: int = 10,
        **kwargs,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.kwargs = kwargs
        self.device = get_device()
        self.model: nn.Module = self._build_network().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    # ---- subclasses override ----
    def _build_network(self) -> nn.Module:
        raise NotImplementedError

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """Optional reshape for 1D-CNN / RNN models."""
        return x

    # ---- shared training loop ----
    def train(self, X: np.ndarray, y: np.ndarray, val_frac: float = 0.1):
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)

        n = len(X_t)
        n_val = max(1, int(n * val_frac))
        perm = torch.randperm(n)
        v_idx, t_idx = perm[:n_val], perm[n_val:]
        X_tr, y_tr = X_t[t_idx], y_t[t_idx]
        X_val, y_val = X_t[v_idx], y_t[v_idx]

        train_ds = TensorDataset(X_tr, y_tr)
        loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=is_cuda(self.device),
            drop_last=False,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
        patience_left = self.patience
        _log.info(f"[{self.name}] training on {self.device} — epochs={self.epochs}, n_train={len(X_tr)}")

        epoch_bar = tqdm(range(1, self.epochs + 1), desc=f"  {self.name}", unit="ep", ncols=90, leave=True)
        for ep in epoch_bar:
            self.model.train()
            total = 0.0
            n_samples = 0
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                xb = self._reshape_input(xb)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
                n_samples += xb.size(0)
            train_loss = total / max(1, n_samples)
            val_loss = self._eval_loss(X_val, y_val)

            improved = val_loss < best_val - 1e-5
            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                patience=patience_left,
                best="✓" if improved else "",
            )
            _log.info(f"[{self.name}] epoch {ep:03d}  train={train_loss:.4f}  val={val_loss:.4f}")
            if improved:
                best_val = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    _log.info(f"[{self.name}] early-stop at epoch {ep}")
                    epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", status="EARLY STOP")
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.model.eval()
        bs = self.batch_size
        total = 0.0
        n_samples = 0
        for i in range(0, len(X), bs):
            xb = X[i : i + bs].to(self.device)
            yb = y[i : i + bs].to(self.device)
            xb = self._reshape_input(xb)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            total += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        return total / max(1, n_samples)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32)
        bs = self.batch_size
        outs = []
        for i in range(0, len(X_t), bs):
            xb = X_t[i : i + bs].to(self.device)
            xb = self._reshape_input(xb)
            logits = self.model(xb)
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "kwargs": self.kwargs,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "DeepIDSBase":
        ckpt = torch.load(path, map_location="cpu")
        inst = cls(
            input_dim=ckpt["input_dim"],
            num_classes=ckpt["num_classes"],
            **ckpt.get("kwargs", {}),
        )
        inst.model.load_state_dict(ckpt["state_dict"])
        inst.model.to(inst.device)
        return inst
