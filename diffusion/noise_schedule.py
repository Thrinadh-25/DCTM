"""Linear and cosine beta schedules and derived tensors.

All tensors are float32 and device-aware. DiffusionSchedule bundles them.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    # Nichol & Dhariwal 2021
    steps = T + 1
    x = torch.linspace(0, T, steps, dtype=torch.float64)
    alpha_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    return torch.clamp(betas.float(), 1e-5, 0.999)


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor
    sqrt_alpha_cumprod: torch.Tensor
    sqrt_one_minus_alpha_cumprod: torch.Tensor
    posterior_variance: torch.Tensor
    T: int

    def to(self, device) -> "DiffusionSchedule":
        return DiffusionSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_cumprod=self.alpha_cumprod.to(device),
            sqrt_alpha_cumprod=self.sqrt_alpha_cumprod.to(device),
            sqrt_one_minus_alpha_cumprod=self.sqrt_one_minus_alpha_cumprod.to(device),
            posterior_variance=self.posterior_variance.to(device),
            T=self.T,
        )


def build_schedule(T: int, kind: str = "linear", **kwargs) -> DiffusionSchedule:
    if kind == "linear":
        betas = linear_beta_schedule(T, kwargs.get("beta_start", 1e-4), kwargs.get("beta_end", 0.02))
    elif kind == "cosine":
        betas = cosine_beta_schedule(T, kwargs.get("s", 0.008))
    else:
        raise ValueError(f"Unknown beta schedule: {kind}")

    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), alpha_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)

    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_cumprod=alpha_cumprod,
        sqrt_alpha_cumprod=sqrt_alpha_cumprod,
        sqrt_one_minus_alpha_cumprod=sqrt_one_minus_alpha_cumprod,
        posterior_variance=posterior_variance,
        T=T,
    )
