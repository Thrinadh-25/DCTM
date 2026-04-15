"""Generate adversarial samples by partial noising + constrained denoising.

Procedure (per class):
  1. Take real malicious samples from the test split of that class
  2. Apply FORWARD diffusion up to t = T * partial_t_fraction  (not pure noise)
  3. Run REVERSE denoising from that t down to 0
  4. After every reverse step: clamp IMMUTABLE columns back to original values
  5. Clip to [0,1] (data is MinMax-scaled) and save as parquet

The immutable-feature clamp at every step is the key constraint that keeps
network-layer fields valid (ports, protocol flags) rather than just enforcing
them at the end.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.device import get_device
from utils.logger import get_logger

from .forward_process import q_sample
from .noise_schedule import build_schedule
from .reverse_process import p_sample_loop
from .trainer import load_diffusion

_log = get_logger("adv_gen")


def _build_constrain_fn(x0_orig: torch.Tensor, immutable_idx: list[int]):
    idx = torch.tensor(immutable_idx, dtype=torch.long, device=x0_orig.device)

    def _fn(x: torch.Tensor, t: int) -> torch.Tensor:
        if len(idx) == 0:
            return x
        x = x.clone()
        x[:, idx] = x0_orig[:, idx]
        return x

    return _fn


def _log_drift(orig: np.ndarray, adv: np.ndarray, feature_names: list[str], top_n: int = 5) -> None:
    mean_diff = np.abs(orig.mean(0) - adv.mean(0))
    std_diff = np.abs(orig.std(0) - adv.std(0))
    order = np.argsort(-mean_diff)[:top_n]
    for i in order:
        _log.info(
            f"    drift {feature_names[i]}: "
            f"Δmean={mean_diff[i]:.4f}, Δstd={std_diff[i]:.4f}"
        )


def generate_adversarial_samples(
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    immutable_cols: list[str],
    diffusion_checkpoints: dict[int, str],
    cfg: dict,
    feature_set: str = "dctm",
    dataset_tag: str = "cicids2017",
    per_class_count: int | None = None,
) -> pd.DataFrame:
    """Generate adversarial samples for each class with a trained diffusion ckpt."""
    device = get_device()
    cfg_d = cfg["diffusion"]
    out_dir = cfg["paths"]["adversarial"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    immutable_idx = [feature_names.index(c) for c in immutable_cols if c in feature_names]
    _log.info(f"Immutable indices: {immutable_idx} ({immutable_cols})")

    per_class = per_class_count or cfg_d.get("adv_samples_per_class", 5000)
    partial_frac = cfg_d.get("partial_t_fraction", 0.5)

    all_adv = []
    all_labels = []

    for cls_id, ckpt_path in diffusion_checkpoints.items():
        cls_id = int(cls_id)
        mask = y_test == cls_id
        X_cls = X_test[mask]
        if len(X_cls) == 0:
            _log.warning(f"No test samples for class {cls_id}, skipping")
            continue

        # Sample up to per_class rows (with replacement if needed)
        n_take = min(per_class, len(X_cls))
        if n_take < per_class:
            rng = np.random.default_rng(42)
            extra_idx = rng.choice(len(X_cls), size=per_class - n_take, replace=True)
            X_seed = np.concatenate([X_cls[:n_take], X_cls[extra_idx]], axis=0)
        else:
            rng = np.random.default_rng(42 + cls_id)
            idx = rng.choice(len(X_cls), size=per_class, replace=False)
            X_seed = X_cls[idx]

        model, ckpt = load_diffusion(ckpt_path, device=device)
        schedule = build_schedule(
            T=cfg_d["timesteps"],
            kind=cfg_d["beta_schedule"],
            beta_start=cfg_d["beta_start"],
            beta_end=cfg_d["beta_end"],
            s=cfg_d["cosine_s"],
        ).to(device)

        start_t = max(1, int(schedule.T * partial_frac) - 1)
        _log.info(
            f"Class {cls_id}: generating {len(X_seed)} adv samples "
            f"(partial t={start_t}/{schedule.T}, ckpt={os.path.basename(ckpt_path)})"
        )

        X_seed_t = torch.as_tensor(X_seed, dtype=torch.float32, device=device)
        t_batch = torch.full((X_seed_t.size(0),), start_t, device=device, dtype=torch.long)
        x_t, _ = q_sample(
            X_seed_t, t_batch,
            schedule.sqrt_alpha_cumprod,
            schedule.sqrt_one_minus_alpha_cumprod,
        )

        constrain_fn = _build_constrain_fn(X_seed_t, immutable_idx)
        # Denoise in reasonable-size mini-batches to control memory
        bs = cfg_d.get("batch_size", 256)
        outs = []
        for i in range(0, len(x_t), bs):
            x_chunk = x_t[i : i + bs]
            constrain_chunk = _build_constrain_fn(X_seed_t[i : i + bs], immutable_idx)
            out = p_sample_loop(
                model,
                shape=tuple(x_chunk.shape),
                schedule=schedule,
                start_t=start_t,
                x_init=x_chunk,
                constrain_fn=constrain_chunk,
                device=device,
            )
            outs.append(out)
        adv = torch.cat(outs, dim=0).cpu().numpy()
        adv = np.clip(adv, 0.0, 1.0).astype(np.float32)

        _log_drift(X_seed, adv, feature_names)
        all_adv.append(adv)
        all_labels.append(np.full(len(adv), cls_id, dtype=np.int32))

    if not all_adv:
        _log.warning("No adversarial samples generated — returning empty DataFrame")
        return pd.DataFrame(columns=feature_names + ["y_multiclass"])

    X_adv = np.concatenate(all_adv, axis=0)
    y_adv = np.concatenate(all_labels, axis=0)
    df = pd.DataFrame(X_adv, columns=feature_names)
    df["y_multiclass"] = y_adv

    out_path = os.path.join(out_dir, f"adv_samples_{dataset_tag}_{feature_set}.parquet")
    df.to_parquet(out_path, index=False)
    _log.info(f"Saved {len(df)} adversarial samples -> {out_path}")
    return df
