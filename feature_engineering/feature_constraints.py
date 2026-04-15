"""Per-dataset immutable / mutable feature sets.

Immutable columns are clamped back to their original value after every reverse
diffusion step so adversarial samples stay semantically valid at the
network-protocol level.
"""
from __future__ import annotations

CICIDS2017_IMMUTABLE = [
    "Destination Port",
    "URG Flag Count",
    "CWE Flag Count",
]

CICIDS2018_IMMUTABLE = [
    "Dst Port",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
]


def get_constraints(dataset: str, selected_features: list[str]) -> tuple[list[str], list[str]]:
    """Returns (immutable_cols, mutable_cols) intersected with selected_features."""
    if dataset == "cicids2017":
        imm_all = CICIDS2017_IMMUTABLE
    elif dataset == "cicids2018":
        imm_all = CICIDS2018_IMMUTABLE
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    immutable = [c for c in imm_all if c in selected_features]
    mutable = [c for c in selected_features if c not in immutable]
    return immutable, mutable
