"""XGBoost IDS classifier with optional CUDA GPU acceleration.

If a CUDA-capable device is available and xgboost >= 2.0, we use the
`device="cuda"` parameter; otherwise we keep tree_method=hist on CPU.
"""
from __future__ import annotations

from xgboost import XGBClassifier

from utils.device import is_cuda
from utils.logger import get_logger

from ._base import ClassicalBase

_log = get_logger("xgb")


class XGBoostIDS(ClassicalBase):
    name = "xgboost"

    def _build(self, **params):
        params = dict(params)
        if is_cuda():
            params.setdefault("tree_method", "hist")
            params["device"] = "cuda"
            _log.info("XGBoost using CUDA (device=cuda)")
        else:
            params.setdefault("tree_method", "hist")
            _log.info("XGBoost using CPU")
        try:
            return XGBClassifier(**params)
        except TypeError:
            # older xgboost — drop the "device" arg
            params.pop("device", None)
            return XGBClassifier(**params)
