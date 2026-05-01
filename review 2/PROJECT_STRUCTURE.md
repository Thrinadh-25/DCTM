# DCTM — Project Directory Structure & Navigation Guide

> **Quick tip:** Every phase writes its output to `data/` or `evaluation/`. If a phase fails, check that folder for what's missing.

---

## Root Level

```
DCTM/
├── main.py                   ← SINGLE entry point for the entire pipeline
├── configs/config.yaml       ← ALL hyperparameters live here
├── requirements.txt          ← pip dependencies
├── RUNNING_INSTRUCTIONS.md   ← how to set up and run the project
├── implementation_plan.md    ← original design decisions and plan
│
├── preprocessing/            ← Phase 1: raw CSV → clean parquet
├── feature_engineering/      ← Phase 2: pick best 10/20 features
├── models/                   ← Phase 3: train 10 IDS models
├── diffusion/                ← Phase 4: train diffusion + generate adversarial samples
├── evaluation/               ← Phase 5+: measure evasion, retrain, final report
├── utils/                    ← shared helpers (logging, seeding, device, file I/O)
│
├── datasets/                 ← raw input CSVs (you place them here)
├── data/                     ← generated artifacts (parquet caches, model checkpoints)
├── review 2/                 ← review materials, PPTs, documents
└── Dctm/                     ← (ignore — empty legacy folder)
```

---

## main.py

**The only file you need to run.** Contains one function per phase:

| Function | What it does |
|---|---|
| `phase_preprocess()` | Loads CSVs, cleans, splits train/test |
| `phase_features()` | Runs MI + SHAP, saves feature lists |
| `phase_train()` | Trains all 10 IDS models |
| `phase_diffusion()` | Trains one diffusion model per attack class |
| `phase_attack()` | Generates adversarial samples using diffusion |
| `phase_evaluate()` | Measures evasion rate vs. DEMGAN baseline |
| `phase_retrain()` | Retrains IDS models on adversarial-augmented data |
| `phase_external()` | Cross-dataset validation (2017 ↔ 2018) |
| `phase_report()` | Writes `evaluation/final_report.txt` |

```bash
python main.py --phase all          # run everything
python main.py --phase diffusion    # run just one phase
```

---

## configs/config.yaml

Central control panel. Key knobs:

| Key | Default | What to change it for |
|---|---|---|
| `diffusion.T` | 1000 | Fewer steps = faster but lower quality |
| `diffusion.model_dim` | 256 | Bigger = more capacity, more memory |
| `diffusion.batch_size` | — | Lower if GPU runs out of memory |
| `classical.svm.max_train_samples` | 50000 | Lower if SVM hangs |
| `feature_engineering.shap_sample_size` | 50000 | Lower if memory error during SHAP |
| `preprocessing.test_size` | 0.2 | Train/test split ratio |

---

## preprocessing/

**What it does:** Reads raw CSVs → standardizes column names → removes NaN/Inf → splits into train/test → saves as parquet.

| File | Purpose |
|---|---|
| `data_loader.py` | Reads all CSVs from `datasets/`, merges them into one DataFrame |
| `normalizer.py` | StandardScaler — zero mean, unit variance |
| `splitter.py` | Stratified train/test split (preserves class ratios) |
| `smote_handler.py` | SMOTE oversampling on train set only (caps at 50k/class to avoid OOM) |

**Output:** `data/processed/*.parquet` and `data/splits/`

---

## feature_engineering/

**What it does:** Selects the most useful features from ~80 raw CICFlowMeter columns.

| File | Purpose |
|---|---|
| `mutual_information.py` | Computes MI(feature; label) — picks top 10 → `features_baseline.json` |
| `shap_analysis.py` | Trains a LightGBM proxy → SHAP values → feature importance scores |
| `hybrid_selector.py` | Combines: `0.5×MI + 0.5×SHAP` normalized → top 20 → `features_dctm.json` |
| `feature_constraints.py` | Lists "immutable" features (e.g. Destination Port, Flag Counts) that must not be altered |

**Two feature sets produced:**
- `features_baseline.json` → 10 features (MI only) — used by baseline models
- `features_dctm.json` → 20 features (MI + SHAP hybrid) — used by DCTM models

---

## models/

**What it does:** Trains 10 IDS classifiers. All share the same interface: `.train()`, `.predict()`, `.predict_proba()`, `.save()`, `.load()`.

### models/classical/  (CPU-based, sklearn)

| File | Model | Notes |
|---|---|---|
| `decision_tree.py` | Decision Tree | Fast, interpretable |
| `naive_bayes.py` | Gaussian Naive Bayes | Very fast, low accuracy |
| `logistic_regression.py` | Logistic Regression | `saga` solver, capped at 300k samples |
| `random_forest.py` | Random Forest | Best classical performer |
| `xgboost_model.py` | XGBoost | Uses GPU if available (`device=cuda`) |
| `svm_model.py` | SVM (RBF kernel) | Slowest — capped at 50k samples |
| `_base.py` | Base class | Shared pickle save/load logic |

### models/deep_learning/  (GPU via PyTorch)

| File | Model | Architecture |
|---|---|---|
| `mlp.py` | MLP | 3-layer fully connected |
| `cnn.py` | CNN | 1D convolutions over feature vector |
| `rnn.py` | RNN | Bidirectional LSTM |
| `cnn_bilstm.py` | CNN-BiLSTM | CNN feature extraction → BiLSTM |
| `base_model.py` | Base class | Shared PyTorch train loop, `.pt` save/load |

### models/trainer.py

Iterates all 10 models, trains on the requested feature set, auto-skips if checkpoint already exists in `data/models/`.

**Output:** `data/models/{model_name}_{feature_set}.pkl` or `.pt`

---

## diffusion/

**The core contribution of DCTM.** Replaces GAN with a Transformer-based denoising diffusion model.

| File | Purpose |
|---|---|
| `noise_schedule.py` | Defines beta schedule (linear or cosine), precomputes α, ᾱ for T=1000 steps |
| `forward_process.py` | `q_sample(x0, t)` — adds noise to clean data at timestep t |
| `denoiser.py` | `TransformerDenoiser` — the neural network that predicts noise; sinusoidal time embedding → 4× Transformer encoder layers |
| `reverse_process.py` | `p_sample_loop()` — iterative denoising from pure noise back to data |
| `trainer.py` | Trains one `TransformerDenoiser` per attack class, saves to `data/models/diffusion_dctm_class{i}.pt` |
| `adversarial_generator.py` | Adversarial generation logic: partial forward diffusion to t=T//2 (preserves attack semantics) → reverse denoise → clamp immutable features after every step |

**Key design decision:** One separate diffusion model per attack class (not one conditioned model). Simpler to train and defend.

**Output:** `data/adversarial/adv_samples_*.parquet`

---

## evaluation/

**What it does:** Measures how well adversarial samples fool each IDS model.

| File | Purpose |
|---|---|
| `metrics.py` | Shared metric helpers: accuracy, F1, precision, recall |
| `evasion_evaluator.py` | Runs each model on clean vs. adversarial test data; computes **Evasion Rate (ER)**; compares against DEMGAN baseline (97.42%) |
| `retraining.py` | Augments training data with adversarial samples → retrains all 10 models → measures recovery |
| `report_generator.py` | Writes `evaluation/final_report.txt`: best F1, most vulnerable model, retraining gains |

**Output:** `evaluation/results/evasion_*.csv`, `evaluation/results/retrained_*.csv`, `evaluation/final_report.txt`

---

## utils/

Shared helpers imported everywhere.

| File | Purpose |
|---|---|
| `logger.py` | Configures Python `logging` — all phase logs go through this |
| `seed.py` | `set_seed(42)` — ensures reproducibility across numpy, torch, random |
| `device.py` | Returns `torch.device('cuda')` if available, else `'cpu'` |
| `io.py` | Wrappers for saving/loading parquet files and JSON feature lists |

---

## datasets/  (you provide these)

Raw CICFlowMeter CSV files. Never modified by the code.

```
datasets/
├── cicids 2017/              ← 8 daily CSV files (Mon–Fri)
├── cicids 2018 full/         ← full 2018 dataset CSVs
├── cicids2018csv/            ← subset of 2018 CSVs
└── cicids2018/               ← pre-converted parquet files (generated)
```

---

## data/  (generated — not in git)

All intermediate artifacts created during the pipeline run.

```
data/
├── processed/                ← cleaned parquet after preprocessing
├── splits/                   ← train/test splits as parquet
├── models/
│   ├── decision_tree_dctm.pkl
│   ├── random_forest_baseline.pkl
│   ├── mlp_dctm.pt
│   ├── diffusion_dctm_class0.pt   ← one per attack class
│   └── ...
└── adversarial/
    └── adv_samples_class{i}.parquet
```

> To retrain a single model, delete its `.pkl`/`.pt` file — the trainer auto-skips existing checkpoints.

---

## review 2/  (this folder)

Materials for Review 2 presentation.

```
review 2/
├── PROJECT_STRUCTURE.md      ← this file
├── ppt main/
│   ├── review 2 c6 dctm.pptx ← main presentation file
│   ├── arch(main).pdf        ← architecture diagram
│   ├── DCTM abs.pdf          ← project abstract
│   ├── TSP_CMC_64833 (1).pdf ← DEMGAN base paper
│   └── DIAGRAMS.md           ← diagram content descriptions
└── mics/
    ├── CSE-C-06.pptx         ← college PPT template
    ├── MiniProj-PPT-Instructions.pptx
    ├── ARCHITECTURE.md       ← architecture writeup
    ├── IMPLEMENTATION.md     ← implementation details
    ├── PRESENTATION_SCRIPT.md
    ├── SLIDE_CONTENT_GUIDE.md
    ├── SLIDE_NOTES.md
    └── CLAUDE_DESIGN_PPT.md
```

---

## Pipeline Phase Order (must run in sequence)

```
preprocess → features → train → diffusion → attack → evaluate → retrain → external → report
```

Each phase reads from the previous phase's output in `data/`. Skip a phase only if its output already exists.
