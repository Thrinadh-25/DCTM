# DCTM — Implementation Document

**Project:** DCTM — Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance
**Team:** Kasturi Vishnu Vardhan (245523733160), Kukunuru Thrinadh Reddy (245523733163), Neha Sri Tirunagari (245523733187)
**Guide:** Ms. Duddeda Aishwarya, Asst. Professor
**Stage:** Prototype Review

---

## 1. Problem Statement

Machine-learning Intrusion Detection Systems (IDS) are vulnerable to **adversarial traffic** — synthetic packets crafted to fool the classifier into labelling malicious flows as benign. The state-of-the-art baseline, **DEMGAN** (Xu et al., 2025), uses Wasserstein GANs with multiple generators to achieve a **97.42% evasion rate** on CICIDS2017. DEMGAN has three open weaknesses, which DCTM directly targets.

| DEMGAN limitation | DCTM solution |
|-------------------|---------------|
| Restricted to **10 features** (Mutual Information only) | Expanded to **20 features** via hybrid MI + SHAP |
| **GAN instability** — fails on linear classifiers (DT, LR) | Replaced with **Transformer Diffusion** (no min-max game) |
| Severe **class imbalance** ignored | **SMOTE** rebalancing before diffusion |
| Reports only Evasion Rate | Reports **F1, Precision, Recall** alongside ER |

---

## 2. Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Classical ML | scikit-learn (1.4+), XGBoost (2.0+) |
| Deep Learning | PyTorch (2.1+) |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Feature analysis | SHAP, LightGBM (proxy model) |
| Data | pandas, numpy, pyarrow (parquet) |
| Visualization | matplotlib, seaborn |
| Config / utilities | PyYAML, tqdm |
| Compute | NVIDIA T4 (Google Colab) — CUDA 12.1 |

---

## 3. Datasets

| Dataset | Source | Role | Size after cleaning |
|---------|--------|------|---------------------|
| **CICIDS 2017** | Canadian Institute for Cybersecurity (UNB) | Primary training + main evasion benchmark | ~2.8 M flows, 15 classes (1 benign + 14 attack types) |
| **CICIDS 2018** | Canadian Institute for Cybersecurity (UNB) | External validation (cross-dataset robustness) | ~2.4 M flows |

Attack classes include: DoS Hulk, DoS GoldenEye, DDoS, PortScan, Bot, Web Attacks (XSS, SQL Injection, Brute Force), Heartbleed, Infiltration, FTP-Patator, SSH-Patator.

---

## 4. Pipeline — 10 Phases

| # | Phase | CLI | What happens | Output |
|---|-------|-----|--------------|--------|
| 1 | Preprocess | `--phase preprocess` | Drop NaN/Inf, encode 15 classes, stratified 80/20 split, MinMax-scale | `data/processed/`, `data/splits/`, `data/scaler.pkl` |
| 2 | Feature Engineering | `--phase features` | MI (top-100k) + LightGBM proxy → SHAP → hybrid score → top-20 | `configs/features_baseline.json` (top-10), `configs/features_dctm.json` (top-20) |
| 3 | Class Balancing | (in train) | SMOTE on training split (k=5), cap 50k/class | balanced X_train, y_train |
| 4 | Train 10 IDS | `--phase train --feature-set both` | Train each of 10 models on baseline (10-feat) + DCTM (20-feat) feature sets | `data/models/{name}_{set}.{pkl,pt}` |
| 5 | Train Diffusion | `--phase diffusion` | One Transformer denoiser per attack class (~14 classes) | `data/models/diffusion_dctm_class{i}.pt` |
| 6 | Adversarial Attack | `--phase attack` | Partial noising → constrained reverse → adversarial samples | `data/adversarial/adv_samples_*.parquet` |
| 7 | Evaluate | `--phase evaluate` | ER + F1/P/R/Acc per IDS on clean vs adversarial test | `evaluation/results/evasion_*.csv` |
| 8 | Retrain (Defense) | `--phase retrain` | Augment train with adversarial → retrain → re-evaluate | `evaluation/results/retrained_*.csv` |
| 9 | External Validation | `--phase external` | 2017-trained IDS attacked by 2018 adversarial samples | per-model 2018 ER table |
| 10 | Report | `--phase report` | Final summary table + DEMGAN comparison | `evaluation/final_report.txt` |

Run all at once: `python main.py --phase all`

---

## 5. Key Algorithms

### 5.1 Hybrid Feature Selection (Pillar 3)

```
1. Take top-100k stratified samples
2. score_MI[i]   = MutualInformation(X[:,i], y)        for each feature i
3. Train LightGBM proxy on top-50k samples
4. score_SHAP[i] = mean(|SHAP_i|)                       for each feature i
5. Normalize both scores to [0, 1]
6. score_hybrid[i] = 0.5 · score_MI_norm[i] + 0.5 · score_SHAP_norm[i]
7. Take top-20 features ranked by score_hybrid
8. Mark immutable subset:
   {Destination Port, *Flag Count*, Protocol, ACK Flag, etc.}
   These cannot be perturbed by the diffusion model.
```

### 5.2 SMOTE Balancing (Pillar 2)

```
For each class c:
  if count(c) < 6: skip                # not enough neighbours for k=5
  if count(c) > 50000: subsample to 50000
  generate synthetic samples until count(c) = max_count
```

Applied **only on training split** — test set keeps natural imbalance for fair evaluation.

### 5.3 Diffusion Forward Process (q_sample)

For timestep t ∈ {0, …, T-1}:

```
α_t      = 1 − β_t                       (linear β schedule, 1e-4 → 0.02)
ᾱ_t      = ∏(α_1 ... α_t)
ε        ~ N(0, I)
x_t      = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε
```

### 5.4 Diffusion Reverse Process (p_sample, one step)

```
ε_θ_pred = TransformerDenoiser(x_t, t)
μ_θ      = (1/√α_t) · (x_t − (β_t / √(1 − ᾱ_t)) · ε_θ_pred)
σ_θ²     = β_t                            (fixed variance choice)
z        ~ N(0, I)   if t > 0  else  z = 0
x_{t-1}  = μ_θ + √σ_θ² · z
```

### 5.5 Constrained Adversarial Generation (Pillar 1)

```python
def generate_adversarial(x_0, denoiser, immutable_idx):
    T_partial = T // 2                              # partial noising
    x_t = q_sample(x_0, t=T_partial)                # forward to T/2
    for tau in range(T_partial, 0, -1):
        x_t = p_sample(x_t, tau, denoiser)          # reverse one step
        x_t[:, immutable_idx] = x_0[:, immutable_idx]   # CLAMP per step
    return clip(x_t, 0.0, 1.0)
```

The per-step clamp is the key novelty — it forces the entire denoising trajectory through the feasible (protocol-valid) subspace, not just the endpoint.

### 5.6 Evasion Rate (Pillar 4 metric)

```
ER = #(malicious samples predicted as benign) / #(total malicious samples)
```

Evaluated on the **adversarial** version of the test set. Higher ER means the IDS was successfully fooled.

---

## 6. Hyperparameters (configs/config.yaml)

### 6.1 Diffusion Model
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Timesteps T | 1000 | Standard DDPM choice |
| β schedule | linear (1e-4 → 0.02) | Stable for tabular |
| Model dim | 256 | Sufficient capacity for 20-d input |
| Num layers | 4 | Transformer encoder blocks |
| Heads | 4 | Multi-head attention |
| FFN dim | 512 | 2× model dim — standard |
| Dropout | 0.1 | Light regularization |
| Optimizer | AdamW + Cosine LR | Stable for diffusion |
| LR | 1e-4 | Conservative |
| Epochs | 100 | Per attack class |
| Batch size | 256 | Fits on T4 |
| Partial-noising t | T / 2 = 500 | Identity-preserving |

### 6.2 IDS Models (key knobs)
| Model | Setting |
|-------|---------|
| Decision Tree | `max_depth=25, min_samples_leaf=10, class_weight='balanced'` |
| Naive Bayes | GaussianNB (intentionally simple baseline) |
| Logistic Regression | `solver='saga', class_weight='balanced', max_train_samples=300k` |
| Random Forest | `n_estimators=200, max_depth=25, n_jobs=-1, class_weight='balanced_subsample'` |
| XGBoost | `n_estimators=200, max_depth=8, lr=0.1, device='cuda', tree_method='hist'` |
| SVM | `kernel='rbf', max_train_samples=50k` |
| Deep models | Adam, lr=1e-3, batch=512, epochs=30, early-stop patience=10 |

### 6.3 SMOTE
| Parameter | Value |
|-----------|-------|
| `k_neighbors` | 5 |
| `min_samples` (skip threshold) | 6 |
| `max_samples_per_class` | **50,000** (was 200k — reduced for CPU classifier speed) |

---

## 7. Repository Layout

```
DCTM/
├── main.py                       ← CLI orchestrator (entry point)
├── configs/config.yaml           ← All hyperparameters
├── preprocessing/                ← Phase 1: clean, split, scale, SMOTE
├── feature_engineering/          ← Phase 2: MI + SHAP + hybrid + constraints
├── models/
│   ├── classical/                ← 6 sklearn models
│   ├── deep_learning/            ← 4 PyTorch models
│   └── trainer.py                ← Phase 4: unified train loop (resume-aware)
├── diffusion/                    ← Phase 5–6: denoiser + adversarial generator
├── evaluation/                   ← Phase 7–10: metrics, retraining, report
├── utils/                        ← logger, device, io, seed
├── datasets/                     ← raw CSVs (gitignored)
├── data/                         ← all generated artifacts (gitignored)
├── visualization/                ← plots
├── logs/run.log                  ← execution log
├── requirements.txt
├── ARCHITECTURE.md               ← system design (this folder)
├── IMPLEMENTATION.md             ← this file
└── CLAUDE.md                     ← AI assistant guide
```

---

## 8. Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Preprocessing module | ✅ Complete | NaN/Inf drop, label encode, MinMax, stratified split |
| Feature engineering (MI + SHAP + hybrid) | ✅ Complete | Outputs `features_baseline.json` and `features_dctm.json` |
| SMOTE balancer | ✅ Complete | Cap configurable, skips rare classes safely |
| 6 classical IDS models | ✅ Complete + improved | Tuned hyperparams added; resume-aware loading |
| 4 deep IDS models (PyTorch) | ✅ Complete | Adam + early stopping, GPU-accelerated |
| Transformer denoiser | ✅ Complete | 4-layer encoder, sinusoidal time embedding |
| Noise schedule (linear/cosine β) | ✅ Complete | Both supported; default linear |
| Forward process (q_sample) | ✅ Complete | Vectorized over batch |
| Reverse process (p_sample, p_sample_loop) | ✅ Complete | Per-step constraint hook |
| Adversarial generator | ✅ Complete | Partial noising + per-step immutable clamp |
| Evasion-rate evaluator | ✅ Complete | ER + F1 + Precision + Recall + ROC-AUC |
| Confusion matrix plotting | ✅ Complete | Per-model PNG in `visualization/` |
| Adversarial retraining loop | ✅ Complete | Augments training, retrains all 10 |
| External validation | ✅ Complete | 2017-trained IDS vs 2018 adversarial |
| Report generator | ✅ Complete | Final summary text file |
| **End-to-end pipeline run** | 🟡 Partial | Stopped during classical training; restart with new config |
| Diffusion training run | ⏳ Pending | Awaits classifier completion |
| Adversarial generation | ⏳ Pending | Awaits diffusion |
| Final results | ⏳ Pending | Target run on T4 in next session |

---

## 9. Demonstration Plan for Review

A 5–10 minute walkthrough that proves the prototype works end to end.

| Step | What to show | Command / Artifact |
|------|--------------|---------------------|
| 1 | Project layout & abstract → architecture mapping | `ARCHITECTURE.md` (this folder) |
| 2 | Preprocessed data + class distribution before/after SMOTE | `visualization/class_distribution_dctm.png` |
| 3 | Feature ranking output (MI vs SHAP vs hybrid top-20) | `configs/features_dctm.json` |
| 4 | Trained IDS metrics table (10 models × 2 feature sets) | `data/models/` listing + `logs/run.log` |
| 5 | Transformer denoiser architecture diagram | `ARCHITECTURE.md` §3.2 |
| 6 | Sample adversarial generation (a real DDoS flow → its adversarial version) | print original vs `data/adversarial/adv_samples_cicids2017_dctm.parquet` |
| 7 | Evasion rate table — DCTM vs DEMGAN reference | `evaluation/results/evasion_dctm.csv` |
| 8 | F1 / Precision / Recall comparison (the abstract's eval-extension claim) | same CSV — extra columns |
| 9 | Retraining defense results — robustness gain | `evaluation/results/retrained_dctm.csv` |
| 10 | Final report | `evaluation/final_report.txt` |

---

## 10. Risks and Mitigations

| Risk | Mitigation already in place |
|------|------------------------------|
| Colab T4 disconnect during long runs | Trainer skips already-saved models on restart (resume-aware) |
| sklearn classifiers too slow on 3M-row SMOTE output | Cap reduced to 50k/class; LR uses `saga` + subsample to 300k |
| Diffusion mode collapse on rare attack classes | Per-class denoiser (no shared collapse vector); SMOTE-balanced input |
| Adversarial samples violating protocol fields | Per-step immutable clamp, not just endpoint clamp |
| ROC-AUC undefined for rare classes | `zero_division=0`, multiclass-OvR averaging, NaN-safe |
| GPU OOM during diffusion | `batch_size` configurable in `config.yaml` |

---

## 11. What's Next After This Review

The future-scope ideas (not part of this prototype, deferred to next phase):

1. **Classifier-guided reverse diffusion** — inject IDS gradient at each step (PGD analog)
2. **DDIM sampling** — 50-step inference instead of 1000-step (20× faster)
3. **Conditional diffusion** — single class-embedded model instead of one-per-class
4. **Cross-dataset transferability** experiment
5. **Multi-class evasion** — attack-class → other-attack-class misclassification

These are documented under "Future Scope" in `CLAUDE.md` for traceability. The current abstract and prototype intentionally stop at the 4 pillars (P1–P3 + extended evaluation) to keep scope crisp for the review.

---

## 12. Reference

Xu, D., Lv, Y., Wang, M., Zheng, B., Zhao, J., & Yu, J. (2025). *DEMGAN: A Machine Learning-Based Intrusion Detection System Evasion Scheme*. **Computers, Materials & Continua**, 84(1). DOI: 10.32604/cmc.2025.064833

Datasets: CICIDS2017, CICIDS2018 — Canadian Institute for Cybersecurity, University of New Brunswick.
