# DCTM — System Architecture

**DCTM: Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance**

This document describes the architecture of the DCTM framework — how the components fit together, what each model does, and how data flows from raw network traffic to evaluation results.

---

## 1. High-Level Block Diagram

```
                  ┌────────────────────────────────────────────────────────┐
                  │                    RAW NETWORK TRAFFIC                  │
                  │      CICIDS2017  (8 CSVs)     CICIDS2018  (10 CSVs)     │
                  └────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 1 — DATA PREPROCESSING                                  │
            │   • Drop NaN / Inf / duplicates                                 │
            │   • Encode 15 attack classes → integer labels                   │
            │   • Stratified 80/20 train-test split                           │
            │   • MinMax-scale to [0, 1]                                      │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 2 — FEATURE ENGINEERING (HYBRID MI + SHAP)               │
            │   • Mutual Information      → ranks features by label info     │
            │   • LightGBM proxy + SHAP   → ranks features by impact         │
            │   • Hybrid score = 0.5·MI_norm + 0.5·SHAP_norm                  │
            │   • baseline = top-10 (MI only) → reproduces DEMGAN's set       │
            │   • dctm     = top-20 (hybrid) → DCTM's expanded set            │
            │   • Constraints: mark immutable cols (Dst Port, flags, etc.)    │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 3 — CLASS BALANCING (SMOTE)                              │
            │   • Apply on TRAIN split only (test stays untouched)            │
            │   • k_neighbors = 5,  cap = 50,000 / class                      │
            │   • Skip classes with < 6 native samples                        │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 4 — TRAIN 10 IDS MODELS                                  │
            │                                                                 │
            │   Classical (CPU)             Deep Learning (GPU)               │
            │   ─────────────────           ──────────────────                │
            │   • Decision Tree             • MLP                             │
            │   • Naive Bayes               • CNN (1D)                        │
            │   • Logistic Regression       • RNN (GRU)                       │
            │   • Random Forest             • CNN-BiLSTM                      │
            │   • XGBoost (GPU-capable)                                        │
            │   • SVM (RBF)                                                   │
            │                                                                 │
            │   All share a uniform .train / .predict / .predict_proba API   │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 5 — TRAIN DIFFUSION MODELS (CORE INNOVATION)             │
            │                                                                 │
            │   For each non-benign attack class:                             │
            │      Train one Transformer-based denoiser ε_θ(x_t, t)           │
            │   ─────────────────────────────────────────────────             │
            │   Architecture:                                                  │
            │     Input  x_t ∈ ℝ^{20}  +  timestep t ∈ {0..999}                │
            │     Sinusoidal time embed → MLP → 256-d                         │
            │     Linear project x_t   → 256-d  +  add time embed             │
            │     4× TransformerEncoderLayer (heads=4, FFN=512)               │
            │     Linear project → 20-d predicted noise ε                     │
            │   Loss:  MSE(ε_pred,  ε_true)                                    │
            │   Schedule:  linear β ∈ [1e-4, 0.02], T = 1000 steps             │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 6 — ADVERSARIAL SAMPLE GENERATION                        │
            │                                                                 │
            │   For each malicious test sample x_0:                           │
            │     1.  q_sample(x_0, t = T/2)  →  partially-noised x_t         │
            │     2.  for τ in {T/2, …, 1}:                                   │
            │            x_{τ-1} = p_sample(x_τ, ε_θ)                          │
            │            x_{τ-1}[immutable_cols] = x_0[immutable_cols]        │
            │     3.  clip to [0, 1]   →  adversarial x_adv                   │
            │                                                                 │
            │   Partial noising at t = T/2 keeps the attack identity;         │
            │   per-step clamp keeps protocol fields valid.                   │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 7 — EVASION EVALUATION                                   │
            │   For each of 10 IDS:                                           │
            │     • baseline metrics on clean test                            │
            │     • metrics on adversarial samples                            │
            │     • Evasion Rate = #(malicious → predicted benign) / #malicious│
            │     • F1, Precision, Recall (macro)                             │
            │   Compare against DEMGAN's reported 97.42% reference            │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 8 — ADVERSARIAL RETRAINING (DEFENSE LOOP)                │
            │   X_train_aug = X_train_smote ∪ X_adversarial                   │
            │   Retrain all 10 IDS on augmented data                          │
            │   Re-evaluate on adversarial test → robustness gain             │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 9 — EXTERNAL VALIDATION (CICIDS2018)                     │
            │   2017-trained IDS  ←  attacked by  →  2018 adversarial samples │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 10 — REPORT GENERATION                                   │
            │   Summary table: ER, F1, P, R per model × dataset × phase       │
            │   Comparison vs DEMGAN baseline                                  │
            └───────────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities

| Directory | Responsibility | Key files |
|-----------|----------------|-----------|
| `preprocessing/` | Load raw CSVs, clean, label-encode, split, scale | `data_loader.py`, `cleaner.py`, `splitter.py`, `normalizer.py`, `smote_balancer.py` |
| `feature_engineering/` | MI ranking, SHAP ranking, hybrid selection, immutable-column metadata | `mutual_information.py`, `shap_analysis.py`, `hybrid_selector.py`, `constraints.py` |
| `models/classical/` | 6 sklearn-based IDS classifiers | `decision_tree.py`, `naive_bayes.py`, `logistic_regression.py`, `random_forest.py`, `xgboost_model.py`, `svm_model.py` |
| `models/deep_learning/` | 4 PyTorch IDS classifiers + shared base | `mlp.py`, `cnn.py`, `rnn.py`, `cnn_bilstm.py`, `base_model.py` |
| `models/trainer.py` | Unified train/save/load loop for all 10 models | (single file) |
| `diffusion/` | Tabular Transformer Diffusion Model | `denoiser.py`, `noise_schedule.py`, `forward_process.py`, `reverse_process.py`, `trainer.py`, `adversarial_generator.py` |
| `evaluation/` | Metrics, evasion testing, retraining loop, report generation | `metrics.py`, `evasion_evaluator.py`, `retraining.py`, `report_generator.py` |
| `utils/` | Logger, device detection, config loading, seed management | `logger.py`, `device.py`, `io.py`, `seed.py` |
| `main.py` | CLI orchestrator that dispatches phases | (single file) |
| `configs/config.yaml` | All hyperparameters and paths | (single file) |

---

## 3. The Three Core Architectures

### 3.1 IDS Classifier — Unified Interface

Every one of the 10 IDS models implements the same interface:

```python
model.train(X, y)            # fit on training data
model.predict(X) → y_pred    # hard labels
model.predict_proba(X) → p   # class probabilities (n_samples × n_classes)
model.save(path)             # pickle (.pkl) or torch (.pt)
model.load(path)             # classmethod
```

This uniformity is what makes the trainer loop, evaluator, and retraining loop completely model-agnostic. Adding an 11th model only requires implementing this interface.

| Model | Library | Why included | GPU? |
|-------|---------|--------------|------|
| Decision Tree | sklearn | Linear classifier — DEMGAN's stated weakness target | ✗ |
| Naive Bayes | sklearn | Probabilistic baseline | ✗ |
| Logistic Regression | sklearn (saga) | Linear classifier — DEMGAN's stated weakness target | ✗ |
| Random Forest | sklearn | Strong tree ensemble baseline | ✗ |
| XGBoost | xgboost | SOTA tree boosting; harder target than DEMGAN tested | ✓ (CUDA) |
| SVM | sklearn (RBF) | Non-linear margin classifier | ✗ |
| MLP | PyTorch | Dense feedforward baseline | ✓ |
| CNN (1-D) | PyTorch | Convolution over feature axis | ✓ |
| RNN (GRU) | PyTorch | Sequence over features | ✓ |
| CNN-BiLSTM | PyTorch | Hybrid conv + bidirectional sequence | ✓ |

### 3.2 Transformer Diffusion Denoiser (Core Innovation)

The denoising network ε_θ(x_t, t) — a single Transformer block — is the novelty over DEMGAN's WGAN.

```
                        x_t  ∈ ℝ^{B×F}          t  ∈ ℕ^{B}
                            │                       │
                            │                       ▼
                            │             SinusoidalTimeEmbedding (128-d)
                            │                       │
                            │                       ▼
                            │                  Linear → SiLU → Linear  (256-d)
                            ▼                       │
                       Linear (F→256)               │
                            │                       │
                            └─────────  +  ─────────┘
                                        │
                                        ▼
                                 unsqueeze(1)            ← (B, 1, 256) single-token
                                        │
                                        ▼
                          ┌─── TransformerEncoderLayer ───┐
                          │   nhead=4, dim_ff=512          │  ×4
                          │   pre-norm + GELU              │
                          └────────────────────────────────┘
                                        │
                                        ▼
                                 squeeze(1)
                                        │
                                        ▼
                                 Linear (256 → F)
                                        │
                                        ▼
                                ε_pred  ∈ ℝ^{B×F}
```

**Why a Transformer over an MLP for tabular diffusion?**
- Self-attention scales naturally if we extend to multi-feature tokens later
- Pre-norm + GELU is the standard recipe for stable diffusion training
- Easier to plug class-conditional embeddings in future scope

**Why one denoiser per attack class (not one shared model)?**
- Simpler to defend in the paper (no class-conditioning to justify)
- Each model learns the tight per-class manifold
- Trade-off: more checkpoints, more total training time — acceptable given Colab T4

### 3.3 Adversarial Generation — Partial Noising + Constrained Reverse

```
       Real malicious sample x_0  (e.g., a DDoS flow, MinMax-scaled)
                  │
                  ▼
        ┌─────────────────────────────┐
        │   FORWARD DIFFUSION          │
        │   q(x_t | x_0) = √α_t · x_0  │
        │                + √(1-α_t)·ε  │
        │   t = T / 2                  │   ← partial noising preserves identity
        └─────────────────────────────┘
                  │
                  ▼
                x_{T/2}     ← noisy but still attack-shaped
                  │
                  ▼
        ┌─────────────────────────────┐
        │   REVERSE DIFFUSION (LOOP)   │
        │   for τ in {T/2, …, 1}:      │
        │     ε_θ(x_τ, τ)              │
        │     x_{τ-1} ← p_sample(...)  │
        │     x_{τ-1}[immutable] ← x_0[immutable]   ← per-step clamp
        └─────────────────────────────┘
                  │
                  ▼
              clip [0, 1]  →  x_adv
```

**Three design choices that earn the paper's claims:**

1. **Partial noising at t = T/2 (not T)** — pure noise would discard attack semantics; T/2 is a calibrated balance between "stays an attack" and "looks novel enough to evade".
2. **Per-step immutable clamp** — protocol fields (Dst Port, ACK Flag count, etc.) are set back to the original after every reverse step, not just at the end. This forces the trajectory through the constrained subspace.
3. **MinMax range [0, 1]** — the diffusion operates on already-scaled features, so a final clip enforces feasibility cheaply.

---

## 4. Data Flow Summary

```
   datasets/                                      data/
   ├── cicids 2017/*.csv  ─┐                     ├── processed/*.parquet
   └── cicids2018/*.parquet│ ──preprocess──▶     ├── splits/{train,test}_*.parquet
                           │                     └── scaler.pkl
                           ├── features──▶    configs/features_{baseline,dctm}.json
                           ├── train ────▶        data/models/{name}_{set}.{pkl|pt}
                           ├── diffusion ─▶       data/models/diffusion_dctm_class{i}.pt
                           ├── attack ────▶       data/adversarial/adv_*_dctm.parquet
                           ├── evaluate ──▶       evaluation/results/evasion_*.csv
                           ├── retrain ───▶       evaluation/results/retrained_*.csv
                           └── report ────▶       evaluation/final_report.txt
```

---

## 5. How the Three Abstract Pillars Map to Code

| Abstract pillar | Implemented in | Defends what claim |
|-----------------|----------------|--------------------|
| **P1** — Diffusion-driven generation | `diffusion/denoiser.py`, `diffusion/trainer.py`, `diffusion/adversarial_generator.py` | Stable training (no GAN min-max), high-fidelity samples |
| **P2** — SMOTE before diffusion | `preprocessing/smote_balancer.py` (called in train phase) | Diffusion sees balanced classes, captures rare attacks |
| **P3** — 20-feature hybrid selection | `feature_engineering/hybrid_selector.py` (combines MI + SHAP) | Most potent feature subset for evasion |
| **Eval pillar** — F1/P/R extension | `evaluation/metrics.py` (compute_metrics returns all four) | Comprehensive benchmark, not just ER |

---

## 6. End-to-End Command Flow

```bash
python main.py --phase preprocess         # CSVs → processed parquet + splits
python main.py --phase features           # MI + SHAP → features_*.json
python main.py --phase train --feature-set both    # 10 IDS × 2 feature sets = 20 models
python main.py --phase diffusion          # 1 denoiser per attack class
python main.py --phase attack             # generate adversarial samples
python main.py --phase evaluate           # ER + F1/P/R per model
python main.py --phase retrain            # adversarial retraining defense
python main.py --phase external           # cross-validate on CICIDS2018
python main.py --phase report             # final summary report
```

The CLI is the single entry point — every block in the diagram corresponds to exactly one `--phase` argument.
