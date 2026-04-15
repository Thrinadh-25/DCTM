"""Reverse diffusion: p_sample (single step) and p_sample_loop (full chain).

Supports an optional `constrain_fn` that is called on x after every step so we
can clamp immutable features back to their originals (critical for producing
adversarial samples whose network-layer fields remain semantically valid).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .forward_process import _extract


@torch.no_grad()
def p_sample(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule,
) -> torch.Tensor:
    """One reverse step: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred) + sigma*z"""
    betas_t = _extract(schedule.betas, t, x_t.shape)
    sqrt_one_minus = _extract(schedule.sqrt_one_minus_alpha_cumprod, t, x_t.shape)
    sqrt_recip_alphas = _extract(torch.sqrt(1.0 / schedule.alphas), t, x_t.shape)

    eps_pred = model(x_t, t)
    mean = sqrt_recip_alphas * (x_t - betas_t / sqrt_one_minus * eps_pred)

    # Use variance from schedule
    var = _extract(schedule.posterior_variance, t, x_t.shape)
    noise = torch.randn_like(x_t)
    nonzero_mask = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
    return mean + nonzero_mask * torch.sqrt(var) * noise


@torch.no_grad()
def p_sample_loop(
    model,
    shape,
    schedule,
    start_t: Optional[int] = None,
    x_init: Optional[torch.Tensor] = None,
    constrain_fn: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Run reverse chain from start_t (default T-1) down to 0.

    If x_init is provided, start from it (used for partial noising).
    constrain_fn(x, t) is invoked after every step — good place to clamp
    immutable features.
    """
    if device is None:
        device = next(model.parameters()).device
    if x_init is None:
        x = torch.randn(shape, device=device)
    else:
        x = x_init.to(device)
    T = schedule.T
    t0 = (start_t if start_t is not None else T - 1)
    for i in range(t0, -1, -1):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule)
        if constrain_fn is not None:
            x = constrain_fn(x, i)
    return x
