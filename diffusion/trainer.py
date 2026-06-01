"""Diffusion training loop — one denoiser per attack class.

For each non-benign class with at least `min_samples`, we train a separate
TransformerDenoiser on samples of that class. Class-specific models enable
cleaner conditional generation later.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.device import get_device, is_cuda
from utils.logger import get_logger

from .denoiser import TransformerDenoiser
from .forward_process import q_sample
from .noise_schedule import build_schedule

_log = get_logger("diffusion_trainer")


def _train_one_class(
    X_cls: np.ndarray,
    cls_id: int,
    cfg_d: dict,
    device,
    out_path: str,
) -> None:
    n_features = X_cls.shape[1]
    schedule = build_schedule(
        T=cfg_d["timesteps"],
        kind=cfg_d["beta_schedule"],
        beta_start=cfg_d["beta_start"],
        beta_end=cfg_d["beta_end"],
        s=cfg_d["cosine_s"],
    ).to(device)

    model = TransformerDenoiser(
        n_features=n_features,
        model_dim=cfg_d["model_dim"],
        time_embed_dim=cfg_d["time_embed_dim"],
        nhead=cfg_d["nhead"],
        num_layers=cfg_d["num_layers"],
        dim_feedforward=cfg_d["dim_feedforward"],
        dropout=cfg_d["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_d["lr"],
        weight_decay=cfg_d["weight_decay"],
    )
    epochs = cfg_d["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    X_t = torch.as_tensor(X_cls, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=cfg_d["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=is_cuda(device),
        drop_last=False,
    )

    _log.info(f"Training diffusion for class {cls_id}: n={len(X_t)}, T={schedule.T}, device={device}")
    model.train()
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"  Class {cls_id}", unit="ep", ncols=90, leave=True)
    for ep in epoch_bar:
        t0 = time.time()
        total = 0.0
        count = 0
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            t = torch.randint(0, schedule.T, (xb.size(0),), device=device, dtype=torch.long)
            x_t, noise = q_sample(xb, t, schedule.sqrt_alpha_cumprod, schedule.sqrt_one_minus_alpha_cumprod)
            pred = model(x_t, t)
            loss = torch.nn.functional.mse_loss(pred, noise)
            optimizer.zero_grad()
            loss.backward()
            if cfg_d.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_d["grad_clip"])
            optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        scheduler.step()
        avg = total / max(1, count)
        elapsed = time.time() - t0
        epoch_bar.set_postfix(loss=f"{avg:.5f}", s=f"{elapsed:.1f}")
        _log.info(f"  class={cls_id} epoch {ep:03d}/{epochs}  loss={avg:.5f}  elapsed={elapsed:.1f}s")

    ckpt = {
        "state_dict": model.state_dict(),
        "class_id": cls_id,
        "n_features": n_features,
        "diffusion_cfg": cfg_d,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out_path)
    _log.info(f"  saved -> {out_path}")


def train_diffusion(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    feature_set: str = "dctm",
    min_samples_per_class: int = 100,
    include_benign: bool = False,
) -> dict:
    device = get_device()
    cfg_d = cfg["diffusion"]
    models_dir = cfg["paths"]["models"]
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    ckpts = {}
    classes = sorted(np.unique(y_train).tolist())
    attack_classes = [c for c in classes if include_benign or c != 0]
    print(f"\n[Diffusion] Training {len(attack_classes)} class model(s) on {device}")
    for c in tqdm(attack_classes, desc="Diffusion classes", unit="class", ncols=90):
        mask = y_train == c
        X_cls = X_train[mask]
        if len(X_cls) < min_samples_per_class:
            _log.warning(f"Class {c}: only {len(X_cls)} samples — skipping diffusion training")
            continue
        out_path = os.path.join(models_dir, f"diffusion_{feature_set}_class{c}.pt")
        _train_one_class(X_cls, c, cfg_d, device, out_path)
        ckpts[int(c)] = out_path
    return ckpts


def load_diffusion(ckpt_path: str, device=None) -> tuple[TransformerDenoiser, dict]:
    if device is None:
        device = get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_d = ckpt["diffusion_cfg"]
    n_features = ckpt["n_features"]
    model = TransformerDenoiser(
        n_features=n_features,
        model_dim=cfg_d["model_dim"],
        time_embed_dim=cfg_d["time_embed_dim"],
        nhead=cfg_d["nhead"],
        num_layers=cfg_d["num_layers"],
        dim_feedforward=cfg_d["dim_feedforward"],
        dropout=cfg_d["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
