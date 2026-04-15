# DCTM – Diffusion-Based Cyber Threat Modelling
## Full End-to-End Implementation Plan

---

## Overview

DCTM replaces the GAN-based adversarial generator in DEMGAN with a **Transformer-based Tabular Diffusion Model**, enabling more stable, realistic adversarial traffic generation for Intrusion Detection System (IDS) evaluation and hardening.

**Research contributions reproduced and extended:**

| Contribution | Baseline (DEMGAN) | DCTM |
|---|---|---|
| Adversarial generator | GAN | Tabular Diffusion (Transformer denoiser) |
| Feature selection | 10 features (MI only) | ~20 features (MI + SHAP hybrid) |
| Class imbalance | None | SMOTE on train only |
| Evasion reference | 97.42% ER | Target: ≥ DEMGAN |

---

## Dataset Confirmed

| Dataset | Location | Size |
|---|---|---|
| CICIDS2017 | `datasets/cicids 2017/` | 8 CSV files (~885 MB) |
| CICIDS2018 | `datasets/cicids2018csv/` | 3 CSV files (~1.1 GB) |

---

## Project Directory Structure

```
c:\Users\Thrinadh reddy\DCTM\
│
├── data/
│   ├── raw/                     ← symlink/reference to datasets/
│   ├── processed/               ← .parquet cache after cleaning
│   ├── splits/                  ← train/test splits (parquet)
│   ├── scaler.pkl               ← fitted MinMaxScaler
│   ├── models/                  ← saved .pkl / .pt model files
│   └── adversarial/             ← generated adversarial samples
│
├── preprocessing/
│   ├── __init__.py
│   ├── data_loader.py           ← CSV → parquet, clean, encode labels
│   ├── normalizer.py            ← MinMaxScaler fit/transform
│   ├── splitter.py              ← stratified 80/20 split
│   └── smote_handler.py         ← SMOTE on train only
│
├── feature_engineering/
│   ├── __init__.py
│   ├── mutual_information.py    ← MI ranking → features_baseline.json
│   ├── shap_analysis.py         ← LightGBM + SHAP → shap_importance.png
│   ├── hybrid_selector.py       ← MI + SHAP → features_dctm.json
│   └── feature_constraints.py  ← IMMUTABLE_COLS, MUTABLE_COLS
│
├── models/
│   ├── __init__.py
│   ├── trainer.py               ← unified training loop for all models
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── decision_tree.py
│   │   ├── naive_bayes.py
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── svm_model.py
│   └── deep_learning/
│       ├── __init__.py
│       ├── base_model.py        ← shared PyTorch training utils
│       ├── mlp.py
│       ├── cnn.py
│       ├── rnn.py
│       └── cnn_bilstm.py
│
├── diffusion/
│   ├── __init__.py
│   ├── noise_schedule.py        ← linear + cosine beta schedule
│   ├── forward_process.py       ← q_sample (add noise)
│   ├── denoiser.py              ← Transformer denoising network
│   ├── reverse_process.py       ← p_sample, p_sample_loop
│   ├── trainer.py               ← diffusion training loop
│   └── adversarial_generator.py ← partial noising + constrained generation
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py               ← accuracy, F1, ROC-AUC, evasion rate
│   ├── evasion_evaluator.py     ← clean vs adversarial evaluation
│   ├── retraining.py            ← augment + retrain + compare
│   ├── report_generator.py      ← final summary table + file
│   └── results/
│       ├── baseline_results.csv
│       ├── evasion_results.csv
│       └── retrained_results.csv
│
├── visualization/               ← all plots saved here
│   ├── feature_importance_comparison.png
│   ├── shap_importance.png
│   ├── class_distribution.png
│   ├── confusion_matrices/      ← per model
│   ├── evasion_comparison.png
│   └── retraining_improvement.png
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                ← logging → logs/run.log
│   └── seed.py                  ← seed everything (42)
│
├── configs/
│   ├── config.yaml              ← all hyperparameters
│   ├── features_baseline.json   ← top-10 MI features
│   └── features_dctm.json       ← top-20 hybrid features
│
├── logs/
│   └── run.log
│
└── main.py                      ← CLI: --phase [all|preprocess|features|train|diffusion|attack|retrain]
```

---

## Proposed Changes — Phase by Phase

---

### Phase 1 — Project Scaffold + Data Pipeline

#### [NEW] `configs/config.yaml`
Central hyperparameter store:
- Dataset paths (CICIDS2017 + 2018)
- `train_test_split: 0.8`, `random_seed: 42`
- SMOTE: `k_neighbors: 5`
- Diffusion: `timesteps: 1000`, `beta_schedule: linear`, `epochs: 100`, `batch_size: 256`, `lr: 1e-4`
- All model hyperparams (DT max_depth, RF n_estimators, etc.)
- Output paths

#### [NEW] `utils/logger.py`
- Python `logging` wrapper
- File handler → `logs/run.log`
- Stream handler → console
- Single `get_logger(name)` factory

#### [NEW] `utils/seed.py`
- `set_seed(42)` — sets `random`, `numpy`, `torch`, `torch.cuda` seeds

#### [NEW] `preprocessing/data_loader.py`
- Loads all CSVs from configured path(s), concatenates
- Strips whitespace from column names
- Drops NaN, Inf, duplicate rows
- Encodes `Label` → binary (0=BENIGN, 1=attack) + multiclass integer encoding
- Converts to `.parquet` → caches in `data/processed/`
- Skips re-processing if `.parquet` already exists

#### [NEW] `preprocessing/normalizer.py`
- `fit_transform(X_train)` → sklearn `MinMaxScaler`
- `transform(X_test)` using fitted scaler
- Saves scaler to `data/scaler.pkl`

#### [NEW] `preprocessing/splitter.py`
- `StratifiedShuffleSplit(test_size=0.2, random_state=42)`
- Saves split indices/dataframes to `data/splits/`

#### [NEW] `main.py`
- `argparse` CLI: `--phase`, `--dataset`, `--config`
- Orchestrates all phases via `if args.phase in ('all', 'preprocess'): ...`

---

### Phase 2 — Feature Engineering

#### [NEW] `feature_engineering/mutual_information.py`
- `sklearn.feature_selection.mutual_info_classif` on training features
- Rank all features, select top-10 → `configs/features_baseline.json`
- Plot: `visualization/mi_importance.png`

#### [NEW] `feature_engineering/shap_analysis.py`
- Train `LightGBM` classifier (fast proxy) on training data
- `shap.TreeExplainer` → mean absolute SHAP values per feature
- Save ranked list → `configs/shap_scores.json`
- Plot: `visualization/shap_importance.png`

#### [NEW] `feature_engineering/hybrid_selector.py`
- Normalize MI and SHAP scores to [0,1]
- `hybrid_score = 0.5 * MI_norm + 0.5 * SHAP_norm`
- Select top-20 → `configs/features_dctm.json`
- Plot: `visualization/feature_importance_comparison.png` (3-panel: MI vs SHAP vs Hybrid)

#### [NEW] `feature_engineering/feature_constraints.py`
- Hardcoded per-dataset lists:
  - CICIDS2017 `IMMUTABLE_COLS` = `['Destination Port', 'URG Flag Count', 'CWE Flag Count']`
  - CICIDS2018 `IMMUTABLE_COLS` = `['Dst Port', 'PSH Flag Cnt', 'ACK Flag Cnt']`
- `MUTABLE_COLS` = selected 20 minus immutables
- Exported as module-level constants used by adversarial generator

---

### Phase 3 — Class Imbalance Handling

#### [NEW] `preprocessing/smote_handler.py`
- `imblearn.over_sampling.SMOTE(k_neighbors=5, random_state=42)`
- Applied **only** to training split
- Logs class distribution before/after
- Plot: `visualization/class_distribution.png` (grouped bar)
- Returns `X_train_bal, y_train_bal`

---

### Phase 4 — IDS Baseline Training (9 Models)

All models share a consistent interface:
```python
model.train(X, y)
model.predict(X) -> np.ndarray
model.predict_proba(X) -> np.ndarray
model.save(path)
model.load(path)
```

#### [NEW] `models/classical/decision_tree.py`
`DecisionTreeClassifier(max_depth=10, random_state=42)`

#### [NEW] `models/classical/naive_bayes.py`
`GaussianNB()`

#### [NEW] `models/classical/logistic_regression.py`
`LogisticRegression(max_iter=1000, C=1.0, random_state=42)`

#### [NEW] `models/classical/random_forest.py`
`RandomForestClassifier(n_estimators=100, random_state=42)`

#### [NEW] `models/classical/xgboost_model.py`
`XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42)`

#### [NEW] `models/classical/svm_model.py`
`SVC(kernel='rbf', probability=True, random_state=42)`

#### [NEW] `models/deep_learning/base_model.py`
- Shared PyTorch training loop
- Early stopping (patience=10) by validation loss
- Saves best checkpoint

#### [NEW] `models/deep_learning/mlp.py`
- 3 hidden layers: `[256, 128, 64]`, ReLU, Dropout(0.3)
- Final Softmax output

#### [NEW] `models/deep_learning/cnn.py`
- Reshape input `(batch, 1, n_features)` → 1D CNN
- `Conv1d(1, 32, k=3)` → `Conv1d(32, 64, k=3)` → MaxPool → FC → Softmax

#### [NEW] `models/deep_learning/rnn.py`
- 2-layer GRU, `hidden_size=128`, batch-first
- FC output layer

#### [NEW] `models/deep_learning/cnn_bilstm.py`
- `Conv1d` → `BiLSTM(hidden=64, bidirectional=True)` → FC → Softmax

#### [NEW] `models/trainer.py`
- Loops over all 9 models
- Trains on:
  - **Baseline**: SMOTE-balanced 10-feature data
  - **DCTM**: SMOTE-balanced 20-feature data
- Saves each to `data/models/{model_name}_{feature_set}.{pkl|pt}`

#### [NEW] `evaluation/metrics.py`
- `compute_metrics(y_true, y_pred, y_proba)` → dict {Accuracy, Precision, Recall, F1, ROC-AUC}
- `plot_confusion_matrix(y_true, y_pred, model_name)` → `visualization/confusion_matrices/`
- `compute_evasion_rate(y_true, y_pred)` → `ER = misclassified_benign / total_malicious`

---

### Phase 5 — Diffusion Model (DCTM Core)

#### [NEW] `diffusion/noise_schedule.py`
```python
linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02)
  → betas, alphas, alpha_cumprod, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod

cosine_beta_schedule(T, s=0.008)
  → same tensors via cosine formulation
```

#### [NEW] `diffusion/forward_process.py`
```python
q_sample(x_0, t, sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod, noise=None)
  → x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
```

#### [NEW] `diffusion/denoiser.py` — **Core Architecture**
```
Input: x_t [B, n_features] + t [B] (integer timestep)
  ↓
Sinusoidal timestep embedding → dim=128
  ↓
Linear projection: n_features → model_dim=256
  ↓
[x_t_proj + t_emb] → 4× TransformerEncoderLayer
    (nhead=4, dim_feedforward=512, dropout=0.1, norm_first=True)
  ↓
Linear(256 → n_features)  ← predicts noise ε
```

#### [NEW] `diffusion/reverse_process.py`
```python
p_sample(model, x_t, t, betas, ...)   → x_{t-1}  (single denoising step)
p_sample_loop(model, shape, T, ...)   → x_0        (full T→0 chain)
```

#### [NEW] `diffusion/trainer.py`
- Training loop: sample `t ~ U(0,T)`, compute `x_t`, predict `ε`, MSE loss
- One model per attack class (conditional generation)
- AdamW optimizer, cosine LR schedule
- Checkpoints: `data/models/diffusion_{attack_class}.pt`

---

### Phase 6 — Adversarial Sample Generation

#### [NEW] `diffusion/adversarial_generator.py`
- Load real malicious test samples
- **Partial forward diffusion** to `t = T//2` (not pure noise — preserves attack semantics)
- Run reverse denoising chain from `t=T//2 → 0`
- **After every denoising step**: restore `IMMUTABLE_COLS` to original values
- Validate: log per-feature statistics (mean, std) of original vs adversarial
- Save: `data/adversarial/adv_samples_{dataset}.parquet`

> [!IMPORTANT]
> The immutable-feature clamp after **every** reverse step (not just at the end) is the critical constraint that maintains network-layer realism.

---

### Phase 7 — Evasion Testing

#### [NEW] `evaluation/evasion_evaluator.py`
For each trained IDS model:
1. Evaluate on **clean test data** → baseline metrics
2. Evaluate on **adversarial samples** → attack metrics
3. Compute:
   - `ER = samples_predicted_benign / total_malicious`
   - Accuracy drop, F1 drop
4. Save: `evaluation/results/evasion_results.csv`

#### Plot: `visualization/evasion_comparison.png`
- Grouped bar chart: ER per model
- Reference line at **97.42%** (DEMGAN reported baseline)

---

### Phase 8 — Adversarial Retraining + Final Report

#### [NEW] `evaluation/retraining.py`
- Augment: `X_train_aug = X_train_smote + adv_samples`
- Retrain each IDS model on augmented data
- Re-evaluate on adversarial test set
- Compute: accuracy improvement, F1 improvement
- Save: `evaluation/results/retrained_results.csv`

#### Plot: `visualization/retraining_improvement.png`
- Before/after grouped bar chart per model

#### [NEW] `evaluation/report_generator.py`
```
=== DCTM Final Report ===

Best F1 on clean data:      [Model Name] — F1: X.XX
Most vulnerable (ER):       [Model Name] — ER: X.XX%
Largest retrain improvement:[Model Name] — ΔF1: +X.XX

DEMGAN reference ER:        97.42%
DCTM achieved ER:           X.XX%

Feature sets:
  Baseline (10 feat):  Avg F1 = X.XX
  DCTM (20 feat):      Avg F1 = X.XX
```
Saved to: `evaluation/final_report.txt`

---

## Execution Order in `main.py`

```
--phase all
  1. preprocess   → DataLoader, Normalizer, Splitter
  2. features     → MI, SHAP, Hybrid, Constraints
  3. smote        → SMOTE handler (train only)
  4. train        → All 9 IDS models (10-feat + 20-feat)
  5. diffusion    → Train diffusion model per attack class
  6. attack       → Generate adversarial samples
  7. evaluate     → Evasion testing
  8. retrain      → Adversarial retraining + final report
```

---

## Tech Stack

| Component | Library |
|---|---|
| Data | `pandas`, `numpy`, `pyarrow` |
| ML models | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep models | `PyTorch` |
| SMOTE | `imbalanced-learn` |
| SHAP | `shap` |
| Visualization | `matplotlib`, `seaborn` |
| Config | `PyYAML` |
| Logging | `logging` (stdlib) |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Partial noising (t=T//2) for adversarial gen | Preserves attack semantics; pure noise loses class fidelity |
| One diffusion model per attack class | Enables class-conditional generation without complex conditioning |
| MinMaxScaler fit on train only | Prevents data leakage |
| SMOTE on train only | Prevents oversampling test distribution |
| Immutable clamp after **every** reverse step | Ensures hard constraint satisfaction throughout denoising |
| One model per file with shared interface | Enables the training loop to be model-agnostic |

---

## Verification Plan

### Automated Checks
- `python main.py --phase preprocess` → processed parquet exists, no NaN/Inf
- `python main.py --phase features` → JSON feature lists exist, plot generated
- `python main.py --phase train` → all model files in `data/models/`, CSV results populated
- `python main.py --phase diffusion` → checkpoint `.pt` files exist
- `python main.py --phase all` → `evaluation/final_report.txt` generated

### Sanity Checks
- Class distribution before/after SMOTE logged
- Per-feature stats of adversarial vs original logged (drift check)
- Evasion rate reference: DEMGAN's 97.42% shown as benchmark

---

## Implementation Order (Execution Sequence)

| Step | Files | Depends On |
|---|---|---|
| 1 | `utils/`, `configs/config.yaml` | Nothing |
| 2 | `preprocessing/data_loader.py` | Step 1 |
| 3 | `preprocessing/normalizer.py`, `splitter.py` | Step 2 |
| 4 | `preprocessing/smote_handler.py` | Step 3 |
| 5 | `feature_engineering/` (all 4 files) | Step 3 |
| 6 | `models/classical/` (6 files) | Step 4, 5 |
| 7 | `models/deep_learning/` (5 files) | Step 4, 5 |
| 8 | `models/trainer.py` + `evaluation/metrics.py` | Steps 6, 7 |
| 9 | `diffusion/` (5 files) | Step 5 |
| 10 | `diffusion/adversarial_generator.py` | Steps 8, 9 |
| 11 | `evaluation/evasion_evaluator.py` | Steps 8, 10 |
| 12 | `evaluation/retraining.py` + `report_generator.py` | Step 11 |
| 13 | `main.py` | All |
