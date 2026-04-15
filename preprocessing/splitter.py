"""Stratified 80/20 split, persisted to parquet for reproducibility."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from utils.logger import get_logger

_log = get_logger("splitter")


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "y_multiclass",
    test_size: float = 0.2,
    random_state: int = 42,
    out_dir: str = "data/splits",
    tag: str = "cicids2017",
    cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_path = os.path.join(out_dir, f"{tag}_train.parquet")
    test_path = os.path.join(out_dir, f"{tag}_test.parquet")

    if cache and os.path.exists(train_path) and os.path.exists(test_path):
        _log.info(f"Loading cached splits from {out_dir} ({tag})")
        return pd.read_parquet(train_path), pd.read_parquet(test_path)

    # Stratified on multiclass label if possible, else binary
    strat_col = label_col if label_col in df.columns else "y_binary"
    # Drop classes with only 1 sample (can't stratify)
    vc = df[strat_col].value_counts()
    too_small = vc[vc < 2].index.tolist()
    if too_small:
        _log.warning(f"Dropping {len(too_small)} classes with <2 samples: {too_small}")
        df = df[~df[strat_col].isin(too_small)].reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(sss.split(np.zeros(len(df)), df[strat_col].values))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    test_df = df.iloc[idx_test].reset_index(drop=True)
    _log.info(f"Split {tag}: train={len(train_df)}, test={len(test_df)}")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    _log.info(f"Saved splits -> {train_path}, {test_path}")
    return train_df, test_df
