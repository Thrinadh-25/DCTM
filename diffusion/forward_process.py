"""Forward diffusion: q_sample — add Gaussian noise at timestep t."""
from __future__ import annotations

import torch


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Gather a[t] and reshape to broadcast over `shape`."""
    out = a.gather(0, t)
    while out.dim() < len(shape):
        out = out.unsqueeze(-1)
    return out


def q_sample(
    x_0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alpha_cumprod: torch.Tensor,
    sqrt_one_minus_alpha_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps"""
    if noise is None:
        noise = torch.randn_like(x_0)
    sa = _extract(sqrt_alpha_cumprod, t, x_0.shape)
    sm = _extract(sqrt_one_minus_alpha_cumprod, t, x_0.shape)
    x_t = sa * x_0 + sm * noise
    return x_t, noise
