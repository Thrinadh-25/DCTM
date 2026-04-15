"""CSV → cleaned parquet loader for CICIDS2017 and CICIDS2018.

Both datasets use slightly different column conventions. We normalize:
  - Strip whitespace from column names
  - Drop NaN / Inf rows
  - Drop duplicates
  - Encode Label → binary (0 benign, 1 attack) and multiclass integer

Cached parquet lives in data/processed/{dataset}.parquet along with
{dataset}_label_map.json for multiclass label → integer mapping.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.io import ensure_dirs, save_json
from utils.logger import get_logger

_log = get_logger("data_loader")

BENIGN_LABELS = {"BENIGN", "Benign", "benign"}


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def _drop_bad_rows(df: pd.DataFrame, drop_nan: bool, drop_inf: bool, drop_dupes: bool):
    n0 = len(df)
    if drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
    if drop_nan:
        df = df.dropna()
    if drop_dupes:
        df = df.drop_duplicates()
    _log.info(f"  cleaned rows: {n0} -> {len(df)} (dropped {n0 - len(df)})")
    return df


def _read_csv_safe(path: str, max_rows: int = 0) -> pd.DataFrame:
    kw = {"low_memory": False, "on_bad_lines": "skip", "encoding_errors": "ignore"}
    if max_rows and max_rows > 0:
        kw["nrows"] = max_rows
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kw)


def _load_folder(folder: str, max_rows_per_file: int = 0) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    frames = []
    for f in files:
        _log.info(f"Reading {os.path.basename(f)} ...")
        df = _read_csv_safe(f, max_rows=max_rows_per_file)
        df = _clean_columns(df)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True, sort=False)
    _log.info(f"Concatenated: shape={df.shape}")
    return df


def _encode_labels(df: pd.DataFrame, label_col: str = "Label") -> tuple[pd.DataFrame, dict]:
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found. Columns: {list(df.columns)[:10]}...")
    df[label_col] = df[label_col].astype(str).str.strip()

    df["y_binary"] = (~df[label_col].isin(BENIGN_LABELS)).astype(np.int8)

    classes = sorted(df[label_col].unique().tolist())
    label_map = {c: i for i, c in enumerate(classes)}
    df["y_multiclass"] = df[label_col].map(label_map).astype(np.int32)
    return df, label_map


def load_dataset(
    dataset: str,
    cfg: dict,
    force: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Return cleaned DataFrame + label_map, caching to parquet.

    dataset: "cicids2017" or "cicids2018"
    """
    assert dataset in ("cicids2017", "cicids2018"), f"Unknown dataset: {dataset}"
    processed_dir = cfg["paths"]["processed"]
    ensure_dirs(processed_dir)

    parquet_path = os.path.join(processed_dir, f"{dataset}.parquet")
    label_map_path = os.path.join(processed_dir, f"{dataset}_label_map.json")

    if (not force) and os.path.exists(parquet_path) and os.path.exists(label_map_path):
        _log.info(f"Loading cached {parquet_path}")
        df = pd.read_parquet(parquet_path)
        import json as _json
        with open(label_map_path) as f:
            label_map = _json.load(f)
        return df, label_map

    folder = cfg["paths"]["datasets"][dataset]
    pp = cfg["preprocessing"]
    _log.info(f"Loading {dataset} from {folder}")
    df = _load_folder(folder, max_rows_per_file=pp.get("max_rows_per_file", 0))
    df = _drop_bad_rows(df, pp.get("drop_nan", True), pp.get("drop_inf", True), pp.get("drop_duplicates", True))
    df, label_map = _encode_labels(df, "Label")

    # Coerce numerics — anything non-numeric besides label columns gets dropped
    keep_cols = ["Label", "y_binary", "y_multiclass"]
    for c in df.columns:
        if c in keep_cols:
            continue
        if df[c].dtype == object:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.isna().mean() > 0.5:
                _log.warning(f"  dropping non-numeric column: {c}")
                df = df.drop(columns=[c])
            else:
                df[c] = coerced
    # Second clean pass in case coercion introduced NaNs
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    _log.info(f"Final shape for {dataset}: {df.shape}")

    df.to_parquet(parquet_path, index=False)
    save_json(label_map, label_map_path)
    _log.info(f"Saved -> {parquet_path} and {label_map_path}")
    return df, label_map


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = all numeric columns excluding label bookkeeping."""
    skip = {"Label", "y_binary", "y_multiclass"}
    return [c for c in df.columns if c not in skip]
