# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DCTM (Diffusion-Based Cyber Threat Modelling)** is a research implementation that replaces GAN-based adversarial generators with a Transformer-based Tabular Diffusion Model for Intrusion Detection System (IDS) evaluation and hardening. It extends the DEMGAN baseline using a diffusion approach to generate adversarial network traffic samples that evade detection.

## Running the Pipeline

```bash
# Full end-to-end pipeline
python main.py --phase all

# Individual phases (must run in order)
python main.py --phase preprocess
python main.py --phase features
python main.py --phase train --feature-set both
python main.py --phase diffusion
python main.py --phase attack
python main.py --phase evaluate
python main.py --phase retrain
python main.py --phase external
python main.py --phase report
```

**CLI options:**
- `--phase`: `all | preprocess | features | train | diffusion | attack | evaluate | retrain | external | report`
- `--dataset`: `cicids2017 | cicids2018` (default: `cicids2017`)
- `--feature-set`: `baseline | dctm | both` (default: `dctm`)
- `--config`: path to config file (default: `configs/config.yaml`)

## Environment Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # Windows PowerShell
# or: source .venv/Scripts/activate  # bash/WSL
pip install -r requirements.txt

# Optional GPU support (CUDA 12.1):
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

Raw CSVs go in `datasets/cicids2017/` and `datasets/cicids2018/`. All intermediate artifacts are written to `data/` (parquet caches, model checkpoints, adversarial samples). There is no automated test suite.

## Architecture

### Phase-Based Execution (`main.py`)

`main.py` is the sole entry point — an argparse CLI dispatcher with one function per phase (`phase_preprocess()`, `phase_features()`, etc.). It loads `configs/config.yaml`, fixes `seed=42`, and routes execution. All phases are independent once prerequisites exist (results cached as parquet/pickle).

### Data Flow

```
Raw CSVs (datasets/)
  → preprocess → data/processed/*.parquet + data/splits/
  → feature engineering → features_baseline.json (10 feats) + features_dctm.json (20 feats)
  → SMOTE balancing (train only) → balanced splits
  → train 10 IDS models → data/models/{name}_{feature_set}.{pkl|pt}
  → diffusion training (per attack class) → data/models/diffusion_dctm_class{i}.pt
  → adversarial generation → data/adversarial/adv_samples_*.parquet
  → evasion evaluation → evaluation/results/evasion_*.csv
  → adversarial retraining → evaluation/results/retrained_*.csv
  → report → evaluation/final_report.txt
```

### Feature Engineering (`feature_engineering/`)

Two complementary selection methods are combined:
- **MI Ranking:** Mutual Information on top-100k samples → top-10 features (baseline)
- **SHAP Ranking:** LightGBM proxy trained on top-50k samples → SHAP explainer → feature scores
- **Hybrid:** `0.5*MI + 0.5*SHAP` normalized → top-20 features (DCTM)

Constraints (`constraints.py`) mark "immutable" columns (e.g., Destination Port, Flag Counts) — network-layer features that cannot realistically be altered by an attacker.

### IDS Models (`models/`)

Ten models share a unified interface: `.train(X, y)`, `.predict(X)`, `.predict_proba(X)`, `.save(path)`, `.load(path)`.

- **Classical** (`models/classical/`): Decision Tree, Naive Bayes, Logistic Regression, Random Forest, XGBoost, SVM — each wraps sklearn with pickle serialization.
- **Deep Learning** (`models/deep_learning/`): MLP, CNN, RNN, CNN-BiLSTM — each wraps PyTorch with `.pt` serialization.
- `models/trainer.py` iterates all 10 models and trains each on the requested feature set.

### Diffusion Model (`diffusion/`)

One `TransformerDenoiser` is trained **per attack class** (not conditioned — separate models):

- **Architecture** (`denoiser.py`): sinusoidal timestep embedding (128D) → SiLU → linear projection to `model_dim=256` → 4× `TransformerEncoderLayer` (nhead=4, dim_ff=512, dropout=0.1) → output: predicted noise vector
- **Schedule** (`noise_schedule.py`): linear or cosine beta schedule, T=1000 timesteps
- **Forward process** (`forward_process.py`): `q_sample()` — adds noise to data
- **Reverse process** (`reverse_process.py`): `p_sample()` / `p_sample_loop()` — iterative denoising
- **Adversarial generation** (`adversarial_generator.py`): partial forward diffusion to `t=T//2` (preserves attack semantics), then reverse denoising; **immutable features are clamped after every reverse step** (not just at the end)

### Evaluation (`evaluation/`)

- `evasion_evaluator.py`: compares each model's performance on clean vs. adversarial test data; computes evasion rate (ER) against DEMGAN's 97.42% baseline
- `retraining.py`: augments train set with adversarial samples and retrains all models
- `report_generator.py`: summarizes best F1, most vulnerable model, retraining gains

## Key Configuration (`configs/config.yaml`)

All hyperparameters are centralized here. Key tunable knobs:
- `diffusion.T`: timesteps (default 1000)
- `diffusion.model_dim`: denoiser width (default 256)
- `diffusion.batch_size`: lower if OOM on GPU
- `feature_engineering.shap_sample_size` / `mi_sample_size`: reduce if memory is tight
- `classical.svm.max_train_samples`: reduce if SVM training hangs (default 50k)
- `preprocessing.test_size`: train/test split ratio (default 0.2)

## Common Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: No CSV files` | Place datasets in `datasets/cicids2017/` or `datasets/cicids2018/` |
| `No diffusion index` during attack phase | Run `--phase diffusion` first |
| OOM during SHAP/LightGBM | Reduce `shap_sample_size` in config.yaml |
| SVM training hangs | Reduce `classical.svm.max_train_samples` in config.yaml |
| Logistic regression slow on CPU | Already uses `saga` solver + `max_train_samples=300k`; lower further if needed |
| Out of GPU memory in diffusion | Lower `diffusion.batch_size` in config.yaml |
| CUDA not available | Falls back to CPU automatically; GPU wheel install is optional |
| Want to retrain a single model | Delete its `.pkl`/`.pt` in `data/models/` — trainer auto-skips existing checkpoints |

## GPU Reality Check

scikit-learn classifiers (DT, NB, LR, RF, SVM) are **CPU-only by design** — `CUDA available` in the log means PyTorch found the GPU, but sklearn cannot use it. Only **XGBoost** (via `device=cuda`) and the four **deep models** (PyTorch) actually run on GPU. To keep classical training under ~10 minutes on CPU, the SMOTE cap is set to `max_samples_per_class: 50000` in `config.yaml`.

## Future Scope (post-MVP improvements, not in current abstract)

These are documented here as deliberate exclusions from the current paper scope. Pursue after the prototype review is approved.

| Idea | Pillar it would extend | Why deferred |
|------|------------------------|--------------|
| **Classifier-guided reverse diffusion** — inject IDS gradient into each reverse step (PGD-style) | New 4th pillar on guided generation | Requires abstract revision |
| **SHAP-on-the-IDS targeted perturbation** — bias diffusion noise toward features the DT/LR depend on | Refines P3 | Extra mechanism, new claim |
| **Borda-count feature ranking** (MI + SHAP + permutation importance) | Extends P3 | Two-source hybrid is sufficient for v1 |
| **Cross-dataset transferability** — train diffusion on 2017, attack 2018-trained IDS | New evaluation claim | Currently external phase only validates same-dataset |
| **Multi-class evasion** — attack-class → other-attack-class misclassification (not just → benign) | New evaluation claim | Binary evasion is the abstract baseline |
| **DDIM sampling at inference** (50 steps vs 1000) | Refines P1 (faster) | Optimization, not a new contribution |
| **Conditional diffusion** (single class-embedded model vs one-per-class) | Refines P1 | Architectural change; current per-class training is simpler to defend |
| **Soft constraint loss during training** (in addition to hard inference clamp) | Refines P1 (higher fidelity) | Add when chasing higher ER margins |
