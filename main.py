"""DCTM – End-to-end CLI orchestrator.

Example usage:
  python main.py --phase all
  python main.py --phase preprocess --dataset cicids2017
  python main.py --phase features
  python main.py --phase train --feature-set both
  python main.py --phase diffusion
  python main.py --phase attack
  python main.py --phase evaluate
  python main.py --phase retrain

Pipeline (matches research paper):
  CICIDS2017 → training + main evasion evaluation
  CICIDS2018 → external validation (trained models tested on 2018 adv samples)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.device import log_device_status
from utils.io import ensure_dirs, load_config, load_json, save_json
from utils.logger import get_logger
from utils.seed import set_seed

_log = get_logger("main")

FEATURE_SETS = ("baseline", "dctm")   # 10-feat vs 20-feat


# ----------------------------- helpers -----------------------------

def _select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        _log.warning(f"Missing features (will drop): {missing}")
    kept = [f for f in features if f in df.columns]
    return df[kept]


def _get_feature_list(cfg: dict, feature_set: str) -> list[str]:
    path_map = {
        "baseline": os.path.join(cfg["paths"]["configs"], "features_baseline.json"),
        "dctm": os.path.join(cfg["paths"]["configs"], "features_dctm.json"),
    }
    data = load_json(path_map[feature_set])
    return data["top_k"]


# ----------------------------- phases -----------------------------

def phase_preprocess(cfg: dict, datasets=("cicids2017", "cicids2018")):
    from preprocessing.data_loader import load_dataset
    from preprocessing.splitter import stratified_split

    ensure_dirs(cfg["paths"]["processed"], cfg["paths"]["splits"])
    for ds in datasets:
        _log.info(f"=== Preprocess: {ds} ===")
        df, _ = load_dataset(ds, cfg)
        stratified_split(
            df,
            label_col="y_multiclass",
            test_size=cfg["preprocessing"]["test_size"],
            random_state=cfg["preprocessing"]["random_state"],
            out_dir=cfg["paths"]["splits"],
            tag=ds,
        )


def phase_features(cfg: dict, dataset: str = "cicids2017"):
    from feature_engineering.hybrid_selector import select_hybrid_features
    from feature_engineering.mutual_information import compute_mi
    from feature_engineering.shap_analysis import compute_shap
    from preprocessing.data_loader import get_feature_columns
    from preprocessing.normalizer import Normalizer

    splits_dir = cfg["paths"]["splits"]
    train_df = pd.read_parquet(os.path.join(splits_dir, f"{dataset}_train.parquet"))
    feats_all = get_feature_columns(train_df)
    X = train_df[feats_all].values.astype(np.float32)
    y = train_df["y_multiclass"].values.astype(np.int64)

    # Normalize on train for MI/SHAP (fit_transform)
    norm = Normalizer()
    X = norm.fit_transform(X, columns=feats_all)
    norm.save(cfg["paths"]["scaler"])

    fe = cfg["feature_engineering"]
    compute_mi(
        X, y, feats_all,
        top_k=fe["baseline_top_k"],
        sample_size=fe["mi_sample_size"],
        random_state=cfg["preprocessing"]["random_state"],
        out_json=os.path.join(cfg["paths"]["configs"], "features_baseline.json"),
        plot_path=os.path.join(cfg["paths"]["visualization"], "mi_importance.png"),
    )
    compute_shap(
        X, y, feats_all,
        sample_size=fe["shap_sample_size"],
        lgbm_params=fe["lgbm"],
        out_json=os.path.join(cfg["paths"]["configs"], "shap_scores.json"),
        plot_path=os.path.join(cfg["paths"]["visualization"], "shap_importance.png"),
        random_state=cfg["preprocessing"]["random_state"],
    )
    select_hybrid_features(
        mi_path=os.path.join(cfg["paths"]["configs"], "features_baseline.json"),
        shap_path=os.path.join(cfg["paths"]["configs"], "shap_scores.json"),
        top_k=fe["dctm_top_k"],
        out_json=os.path.join(cfg["paths"]["configs"], "features_dctm.json"),
        plot_path=os.path.join(cfg["paths"]["visualization"], "feature_importance_comparison.png"),
    )


def _load_split_scaled(cfg: dict, dataset: str, feature_set: str, fit_scaler: bool = False):
    """Load train/test splits, apply selected features, scale. Returns X_tr, y_tr, X_te, y_te, features."""
    from preprocessing.data_loader import get_feature_columns
    from preprocessing.normalizer import Normalizer

    splits_dir = cfg["paths"]["splits"]
    tr = pd.read_parquet(os.path.join(splits_dir, f"{dataset}_train.parquet"))
    te = pd.read_parquet(os.path.join(splits_dir, f"{dataset}_test.parquet"))

    features = _get_feature_list(cfg, feature_set)
    features = [f for f in features if f in tr.columns]

    X_tr = tr[features].values.astype(np.float32)
    y_tr = tr["y_multiclass"].values.astype(np.int64)
    X_te = te[features].values.astype(np.float32)
    y_te = te["y_multiclass"].values.astype(np.int64)

    scaler_path = os.path.join(
        cfg["paths"]["processed"], f"scaler_{dataset}_{feature_set}.pkl"
    )
    if fit_scaler or not os.path.exists(scaler_path):
        norm = Normalizer()
        X_tr = norm.fit_transform(X_tr, columns=features)
        X_te = norm.transform(X_te)
        norm.save(scaler_path)
    else:
        norm = Normalizer.load(scaler_path)
        X_tr = norm.transform(X_tr)
        X_te = norm.transform(X_te)
    return X_tr, y_tr, X_te, y_te, features


def phase_train(cfg: dict, dataset: str = "cicids2017", feature_set: str = "both"):
    from models.trainer import train_all_models
    from preprocessing.smote_handler import apply_smote

    sets = FEATURE_SETS if feature_set == "both" else (feature_set,)
    all_results = {}
    for fs in sets:
        _log.info(f"=== Train: {dataset} / feature_set={fs} ===")
        X_tr, y_tr, X_te, y_te, feats = _load_split_scaled(cfg, dataset, fs, fit_scaler=True)

        X_tr, y_tr = apply_smote(
            X_tr, y_tr,
            k_neighbors=cfg["smote"]["k_neighbors"],
            random_state=cfg["smote"]["random_state"],
            min_samples=cfg["smote"]["min_samples"],
            plot_path=os.path.join(cfg["paths"]["visualization"], f"class_distribution_{fs}.png"),
        )

        num_classes = int(max(y_tr.max(), y_te.max())) + 1
        results = train_all_models(
            X_tr, y_tr, X_te, y_te,
            num_classes=num_classes,
            cfg=cfg,
            feature_set=fs,
        )
        all_results[fs] = results

        # Save baseline eval CSV for the report generator
        Path(cfg["paths"]["results"]).mkdir(parents=True, exist_ok=True)
        rows = []
        for name, r in results.items():
            if "metrics" in r:
                rows.append({"model": name, "feature_set": fs, **r["metrics"]})
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(cfg["paths"]["results"], f"baseline_{fs}.csv"),
                index=False,
            )

    save_json(
        {k: {m: r.get("metrics", {}) for m, r in v.items()} for k, v in all_results.items()},
        os.path.join(cfg["paths"]["results"], "train_results.json"),
    )
    return all_results


def phase_diffusion(cfg: dict, dataset: str = "cicids2017", feature_set: str = "dctm"):
    from diffusion import train_diffusion

    _log.info(f"=== Diffusion train: {dataset} / {feature_set} ===")
    X_tr, y_tr, _, _, _ = _load_split_scaled(cfg, dataset, feature_set, fit_scaler=False)
    ckpts = train_diffusion(X_tr, y_tr, cfg, feature_set=feature_set)
    save_json(
        {str(k): v for k, v in ckpts.items()},
        os.path.join(cfg["paths"]["models"], f"diffusion_index_{feature_set}.json"),
    )
    return ckpts


def phase_attack(cfg: dict, dataset: str = "cicids2017", feature_set: str = "dctm"):
    from diffusion import generate_adversarial_samples
    from feature_engineering.feature_constraints import get_constraints

    _log.info(f"=== Adversarial generation: {dataset} / {feature_set} ===")
    ckpt_index = os.path.join(cfg["paths"]["models"], f"diffusion_index_{feature_set}.json")
    if not os.path.exists(ckpt_index):
        raise FileNotFoundError(f"No diffusion index at {ckpt_index}. Run phase diffusion first.")
    ckpts = {int(k): v for k, v in load_json(ckpt_index).items()}

    _, _, X_te, y_te, feats = _load_split_scaled(cfg, dataset, feature_set, fit_scaler=False)
    immutable, _ = get_constraints(dataset, feats)
    df_adv = generate_adversarial_samples(
        X_test=X_te, y_test=y_te,
        feature_names=feats,
        immutable_cols=immutable,
        diffusion_checkpoints=ckpts,
        cfg=cfg,
        feature_set=feature_set,
        dataset_tag=dataset,
    )
    return df_adv


def phase_evaluate(cfg: dict, dataset: str = "cicids2017", feature_set: str = "dctm"):
    """Evaluate evasion on the dataset used for training.
    Also re-runs against CICIDS2018 if adv samples exist for it."""
    from evaluation.evasion_evaluator import evaluate_evasion

    _log.info(f"=== Evasion eval: {dataset} / {feature_set} ===")
    _, _, X_te, y_te, feats = _load_split_scaled(cfg, dataset, feature_set, fit_scaler=False)

    adv_path = os.path.join(
        cfg["paths"]["adversarial"], f"adv_samples_{dataset}_{feature_set}.parquet"
    )
    if not os.path.exists(adv_path):
        raise FileNotFoundError(f"Missing adv samples: {adv_path}. Run phase attack first.")
    df_adv = pd.read_parquet(adv_path)
    X_adv = df_adv[feats].values.astype(np.float32)
    y_adv = df_adv["y_multiclass"].values.astype(np.int64)

    out_csv = os.path.join(cfg["paths"]["results"], f"evasion_{feature_set}.csv")
    return evaluate_evasion(
        X_test_clean=X_te, y_test_clean=y_te,
        X_adv=X_adv, y_adv=y_adv,
        cfg=cfg, feature_set=feature_set,
        out_csv=out_csv,
    )


def phase_retrain(cfg: dict, dataset: str = "cicids2017", feature_set: str = "dctm"):
    from evaluation.retraining import adversarial_retrain
    from preprocessing.smote_handler import apply_smote

    _log.info(f"=== Adversarial retrain: {dataset} / {feature_set} ===")
    X_tr, y_tr, X_te, y_te, feats = _load_split_scaled(cfg, dataset, feature_set, fit_scaler=False)
    X_tr, y_tr = apply_smote(
        X_tr, y_tr,
        k_neighbors=cfg["smote"]["k_neighbors"],
        random_state=cfg["smote"]["random_state"],
        min_samples=cfg["smote"]["min_samples"],
        plot_path=None,
    )

    adv_path = os.path.join(
        cfg["paths"]["adversarial"], f"adv_samples_{dataset}_{feature_set}.parquet"
    )
    df_adv = pd.read_parquet(adv_path)
    X_adv = df_adv[feats].values.astype(np.float32)
    y_adv = df_adv["y_multiclass"].values.astype(np.int64)
    num_classes = int(max(y_tr.max(), y_te.max(), y_adv.max())) + 1

    out_csv = os.path.join(cfg["paths"]["results"], f"retrained_{feature_set}.csv")
    return adversarial_retrain(
        X_tr, y_tr, X_adv, y_adv, X_te, y_te,
        num_classes=num_classes,
        cfg=cfg, feature_set=feature_set,
        out_csv=out_csv,
    )


def phase_external_validation(cfg: dict, feature_set: str = "dctm"):
    """CICIDS2018 external validation: test models trained on 2017 against
    adversarial samples generated from 2018 data."""
    from evaluation.evasion_evaluator import evaluate_evasion
    from feature_engineering.feature_constraints import get_constraints

    _log.info("=== External validation: CICIDS2018 ===")
    # Re-use the 2017 feature list, but the 2018 columns are different.
    # We map by matching names; missing columns for the 2018 feature set stay 0.
    feats_2017 = _get_feature_list(cfg, feature_set)

    splits_dir = cfg["paths"]["splits"]
    te = pd.read_parquet(os.path.join(splits_dir, "cicids2018_test.parquet"))
    common = [f for f in feats_2017 if f in te.columns]
    _log.info(f"Features shared with 2018: {len(common)}/{len(feats_2017)}: {common}")
    if len(common) < 3:
        _log.warning("Too few shared features for external validation; skipping")
        return None

    # Column-align to 2017 feature order; fill missing with 0
    X_te = np.zeros((len(te), len(feats_2017)), dtype=np.float32)
    for i, f in enumerate(feats_2017):
        if f in te.columns:
            X_te[:, i] = te[f].values.astype(np.float32)
    y_te = te["y_multiclass"].values.astype(np.int64)

    # Scale using the 2017 scaler (critical for cross-dataset eval)
    from preprocessing.normalizer import Normalizer
    scaler_path = os.path.join(cfg["paths"]["processed"], f"scaler_cicids2017_{feature_set}.pkl")
    norm = Normalizer.load(scaler_path)
    X_te = norm.transform(X_te)

    # Check: adversarial samples from 2018 if present
    adv_2018 = os.path.join(cfg["paths"]["adversarial"], f"adv_samples_cicids2018_{feature_set}.parquet")
    if os.path.exists(adv_2018):
        df_adv = pd.read_parquet(adv_2018)
        X_adv = np.zeros((len(df_adv), len(feats_2017)), dtype=np.float32)
        for i, f in enumerate(feats_2017):
            if f in df_adv.columns:
                X_adv[:, i] = df_adv[f].values.astype(np.float32)
        y_adv = df_adv["y_multiclass"].values.astype(np.int64)
    else:
        _log.warning(f"No 2018 adv samples at {adv_2018}; evaluating clean 2018 only")
        X_adv = X_te
        y_adv = y_te

    out_csv = os.path.join(cfg["paths"]["results"], f"external_eval_2018_{feature_set}.csv")
    return evaluate_evasion(
        X_test_clean=X_te, y_test_clean=y_te,
        X_adv=X_adv, y_adv=y_adv,
        cfg=cfg, feature_set=feature_set,
        out_csv=out_csv,
    )


def phase_report(cfg: dict, feature_set: str = "dctm"):
    from evaluation.report_generator import write_final_report

    results_dir = cfg["paths"]["results"]
    write_final_report(
        baseline_eval_csv=os.path.join(results_dir, f"evasion_{feature_set}.csv"),
        retrained_eval_csv=os.path.join(results_dir, f"retrained_{feature_set}.csv"),
        cfg=cfg,
        out_path=os.path.join("evaluation", "final_report.txt"),
    )


# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="DCTM CLI")
    parser.add_argument(
        "--phase",
        choices=[
            "all", "preprocess", "features", "train", "diffusion",
            "attack", "evaluate", "retrain", "external", "report",
        ],
        default="all",
    )
    parser.add_argument("--dataset", default="cicids2017", choices=["cicids2017", "cicids2018"])
    parser.add_argument("--feature-set", default="dctm", choices=["baseline", "dctm", "both"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    ensure_dirs(
        cfg["paths"]["processed"], cfg["paths"]["splits"], cfg["paths"]["models"],
        cfg["paths"]["adversarial"], cfg["paths"]["results"], cfg["paths"]["visualization"],
        cfg["paths"]["configs"],
    )
    log_device_status()

    p = args.phase
    if p in ("all", "preprocess"):
        phase_preprocess(cfg)
    if p in ("all", "features"):
        phase_features(cfg, dataset=args.dataset)
    if p in ("all", "train"):
        phase_train(cfg, dataset=args.dataset, feature_set=args.feature_set)
    if p in ("all", "diffusion"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_diffusion(cfg, dataset=args.dataset, feature_set=fs_main)
    if p in ("all", "attack"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_attack(cfg, dataset=args.dataset, feature_set=fs_main)
    if p in ("all", "evaluate"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_evaluate(cfg, dataset=args.dataset, feature_set=fs_main)
    if p in ("all", "retrain"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_retrain(cfg, dataset=args.dataset, feature_set=fs_main)
    if p in ("all", "external"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_external_validation(cfg, feature_set=fs_main)
    if p in ("all", "report"):
        fs_main = "dctm" if args.feature_set == "both" else args.feature_set
        phase_report(cfg, feature_set=fs_main)


if __name__ == "__main__":
    main()
