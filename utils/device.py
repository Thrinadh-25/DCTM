"""CUDA device helper with explicit fallback reporting.

The user requirement: if CUDA fails, prompt a clear message and proceed on CPU.
Every deep-learning module calls `get_device()` to decide where tensors live.
"""
from __future__ import annotations

from typing import Optional

from .logger import get_logger

_log = get_logger("device")
_CACHED_DEVICE = None
_CACHED_INFO_PRINTED = False


def _probe_cuda() -> tuple[bool, str]:
    """Returns (cuda_usable, message). Runs a tiny smoke test on GPU.

    A machine may report torch.cuda.is_available() == True but still fail on
    the first kernel launch (driver mismatch, missing CUDA runtime, etc.). We
    catch that here and degrade gracefully.
    """
    try:
        import torch
    except Exception as e:
        return False, f"PyTorch import failed: {e}"

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() returned False"

    try:
        dev = torch.device("cuda")
        x = torch.ones(8, 8, device=dev)
        _ = (x @ x).sum().item()
        name = torch.cuda.get_device_name(0)
        return True, f"CUDA OK — device=0 ({name})"
    except Exception as e:
        return False, f"CUDA smoke test failed: {e}"


def get_device(prefer_cuda: bool = True) -> "object":
    """Returns a torch.device, caching the decision across the run.

    Emits a clear log line the first time the decision is made so the user
    can see whether the run is actually using CUDA.
    """
    global _CACHED_DEVICE, _CACHED_INFO_PRINTED
    import torch

    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE

    if not prefer_cuda:
        _CACHED_DEVICE = torch.device("cpu")
        _log.info("Device selected: CPU (prefer_cuda=False)")
        _CACHED_INFO_PRINTED = True
        return _CACHED_DEVICE

    ok, msg = _probe_cuda()
    if ok:
        _CACHED_DEVICE = torch.device("cuda")
        _log.info(f"Device selected: CUDA — {msg}")
    else:
        _CACHED_DEVICE = torch.device("cpu")
        # This is the explicit "CUDA failed" prompt the user asked for.
        print("=" * 60)
        print("[DCTM] CUDA FAILED — falling back to CPU")
        print(f"[DCTM] Reason: {msg}")
        print("[DCTM] Training will NOT use CUDA.")
        print("=" * 60)
        _log.warning(f"CUDA unavailable — using CPU. Reason: {msg}")
    _CACHED_INFO_PRINTED = True
    return _CACHED_DEVICE


def log_device_status() -> None:
    """Call once at program start to surface the device decision early."""
    dev = get_device()
    _log.info(f"Active compute device: {dev}")


def is_cuda(device: Optional[object] = None) -> bool:
    import torch

    if device is None:
        device = get_device()
    return isinstance(device, torch.device) and device.type == "cuda"
