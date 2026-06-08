# DCTM — Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance

**Version:** 1.0 (Draft)
**Date:** June 2026
**Team:**
- Kasturi Vishnu Vardhan (245523733160)
- Kukunuru Thrinadh Reddy (245523733163)
- Neha Sri Tirunagari (245523733187)

**Guide:** Ms. Duddeda Aishwarya, Assistant Professor
**Institution:** [College Name]

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Background and Related Work](#3-background-and-related-work)
4. [Dataset](#4-dataset)
5. [System Architecture](#5-system-architecture)
6. [Methodology](#6-methodology)
   - 6.1 Data Preprocessing
   - 6.2 Feature Engineering — Hybrid MI + SHAP
   - 6.3 IDS Classifier Suite
   - 6.4 Diffusion Model Architecture
   - 6.5 Adversarial Sample Generation
   - 6.6 Evaluation Protocol
   - 6.7 Adversarial Retraining
7. [Experimental Setup](#7-experimental-setup)
8. [Results](#8-results)
   - 8.1 Clean IDS Performance
   - 8.2 Evasion Attack Results
   - 8.3 Ablation Study — Diffusion vs PGD Contribution
   - 8.4 Adversarial Retraining Results
9. [Discussion and Analysis](#9-discussion-and-analysis)
10. [Limitations](#10-limitations)
11. [Conclusion](#11-conclusion)
12. [References](#12-references)

---

## 1. Abstract

Machine-learning-based Intrusion Detection Systems (IDS) are vulnerable to adversarial attacks — carefully crafted network traffic samples that evade detection. Existing adversarial evaluation tools based on Generative Adversarial Networks (GANs) suffer from training instability, mode collapse, and limited feature coverage, making it difficult to rigorously assess and harden IDS deployments.

This work presents **DCTM (Diffusion-Based Cyber Threat Modelling)**, a framework that replaces the GAN-based generator of the DEMGAN baseline with a **Transformer-based Tabular Diffusion Model**. DCTM introduces four improvements over DEMGAN: (1) a stable diffusion-based adversarial generator trained per attack class, (2) a hybrid MI+SHAP feature selection strategy that expands the feature space from 10 to 20 features, (3) SMOTE-based class balancing for rare attack types, and (4) a network-constraint system that preserves immutable packet-level fields during sample generation.

We evaluate DCTM against a suite of 10 IDS classifiers — 6 classical (DT, NB, LR, RF, XGBoost, SVM) and 4 deep learning (MLP, CNN, RNN, CNN-BiLSTM) — on the CICIDS2017 benchmark. The diffusion model alone achieves a mean evasion rate of **53.0%** across all models (average ER = **54.4%** with a lightweight PGD refinement step). An ablation study confirms that diffusion drives **97.5%** of the evasion effect, with PGD contributing only 1.4 percentage points. Adversarial retraining on the generated samples fully restores Random Forest and XGBoost to adversarial F1 of **0.956** and **0.844** respectively, demonstrating that DCTM-generated samples are practically useful for defensive hardening.

---

## 2. Introduction

### 2.1 The Problem

Intrusion Detection Systems protect networks by classifying traffic flows as benign or malicious. Modern IDS rely on machine learning classifiers trained on historical labelled traffic. This creates a fundamental vulnerability: an attacker who understands the IDS model can craft **adversarial network traffic** — samples that are genuinely malicious but that the classifier labels as benign.

Adversarial attacks on IDS are a real and growing threat. As ML-based IDS are deployed in production environments, the ability to test their robustness against adaptive adversaries becomes critical. Without adversarial evaluation, an IDS may report high accuracy on clean test data while being trivially bypassable in practice.

### 2.2 Motivation

The dominant adversarial evaluation tool for tabular network traffic data is **DEMGAN** (Diverse Ensemble Multi-Generator Adversarial Network), which uses a Wasserstein GAN to generate adversarial samples and reports a 97.42% evasion rate on CICIDS2017. However, DEMGAN has three documented weaknesses:

| Weakness | Impact |
|---|---|
| Restricted to 10 features (Mutual Information only) | Misses features that linear/deep classifiers rely on |
| GAN training instability (mode collapse, vanishing gradients) | Especially evident on linear classifiers |
| Ignores severe class imbalance | Rare attack classes are under-modelled |

Additionally, DEMGAN reports only Evasion Rate, making it difficult to compare results with the broader IDS literature that uses F1, Precision, and Recall.

### 2.3 Contributions

This paper makes the following contributions:

1. **DCTM framework** — a complete adversarial IDS evaluation and hardening pipeline using a Transformer-based Tabular Diffusion Model as the generator.
2. **Hybrid MI+SHAP feature selection** — a two-source feature importance method that combines Mutual Information and SHAP values to select 20 informative features, expanding DEMGAN's 10-feature MI-only set.
3. **SMOTE class balancing** — applied to the training split only, ensuring rare attack classes (< 0.1% of traffic) receive adequate representation.
4. **Immutable feature constraints** — a constraint system that prevents the generator from altering network-layer fields that cannot realistically be changed by an attacker (destination port, flag counts).
5. **Ablation study** — isolating the contribution of the diffusion model from a post-hoc PGD refinement step, confirming diffusion alone drives 97.5% of evasion effectiveness.
6. **Rich evaluation metrics** — reporting Accuracy, Precision, Recall, F1, ROC-AUC, and Evasion Rate at each phase, across 10 models, for both clean and adversarial conditions.

---

## 3. Background and Related Work

### 3.1 Adversarial Machine Learning for IDS

Adversarial examples — inputs crafted to mislead ML classifiers — were first studied in the image domain (Goodfellow et al., 2014; Carlini & Wagner, 2017). Applying these techniques to tabular network traffic data is more challenging: traffic features are heterogeneous (integer, float, categorical), features are correlated by protocol constraints, and not all features can be altered without breaking the packet's semantics.

Early IDS adversarial methods adapted FGSM and PGD (Madry et al., 2018) directly to tabular features, often ignoring protocol constraints. More recent work has used generative models to produce samples that remain statistically plausible while evading detection.

### 3.2 DEMGAN Baseline

Xu et al. (2025) proposed DEMGAN, a Diverse Ensemble Multi-Generator Adversarial Network evaluated on CICIDS2017 and CICIDS2018. Key design choices:
- Wasserstein GAN with gradient penalty (WGAN-GP) as the generator
- Ensemble of 10 discriminators across different model architectures
- Mutual Information-based selection of top-10 features
- Reported evasion rate: **97.42%** on CICIDS2017

DEMGAN's GAN-based generator is prone to mode collapse (generating repetitive samples) and training instability. The authors note explicitly that their method underperforms on linear classifiers such as Decision Tree and Logistic Regression, which signals the limited coverage of its 10-feature space.

### 3.3 Diffusion Models for Tabular Data

Diffusion models (Ho et al., 2020; Song et al., 2021) have emerged as a stable alternative to GANs for generative modelling. The forward process gradually adds Gaussian noise to real samples; the reverse process learns to denoise. Unlike GANs, diffusion models have a well-defined training objective (noise prediction), do not suffer from mode collapse, and produce high-fidelity samples across the full distribution.

TabDDPM (Kotelnikov et al., 2023) demonstrated that diffusion models outperform GANs on tabular synthesis benchmarks. DCTM adapts this approach specifically for adversarial IDS evaluation, training a separate denoiser per attack class and using partial forward diffusion followed by guided reverse diffusion to generate adversarial variants.

### 3.4 SHAP-Based Feature Selection

SHAP (SHapley Additive exPlanations, Lundberg & Lee, 2017) provides theoretically-grounded feature importance scores by computing each feature's marginal contribution across all possible feature subsets. Unlike Mutual Information, SHAP captures non-linear interactions through a proxy model (LightGBM in DCTM). Combining MI and SHAP into a hybrid score selects features that are both statistically informative and practically impactful for classifier decisions.

---

## 4. Dataset

### 4.1 CICIDS2017

The primary dataset is **CICIDS2017** (Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017), a widely-used benchmark for IDS research. It contains network flow records captured over five working days, labelled with 14 attack types and a benign class.

| Property | Value |
|---|---|
| Source | University of New Brunswick / CIC |
| Format | 8 CSV files (≈885 MB) |
| Total classes | 15 (0 = Benign, 1–14 = Attack types) |
| Attack types | DoS Hulk, PortScan, DDoS, DoS GoldenEye, FTP-Patator, SSH-Patator, DoS Slowloris, DoS Slowhttptest, Bot, Web Attack – Brute Force, Web Attack – XSS, Infiltration, Web Attack – SQL Injection, Heartbleed |
| Imbalance | Severe — benign accounts for ~80% of samples; some attacks < 0.1% |
| Features | 78 CICFlowMeter-derived flow statistics |

**Known limitations of CICIDS2017:**
- Near-duplicate flows exist between files (flows from the same capture sessions appear in multiple CSVs), making it a relatively "easy" benchmark.
- Some label artifacts are present (flows mislabelled due to timing edge cases).
- RF/XGBoost at ~0.99 accuracy is typical for this dataset and partly reflects dataset structure rather than purely model strength. Cross-dataset validation (CICIDS2018) is required to confirm robustness.

### 4.2 CICIDS2018

CICIDS2018 serves as the external validation dataset in the `--phase external` pipeline step. It was not used for the primary results presented in this report; its role is to test whether models and adversarial strategies trained on CICIDS2017 transfer to a different traffic capture.

### 4.3 Preprocessing

The raw CSVs are processed through the following steps:

```
Raw CSVs (78 features + label)
  → Drop rows with NaN, Inf, or duplicate flows
  → Encode attack labels to integers (0–14)
  → Stratified 80/20 train-test split (seed=42)
  → MinMax scale all features to [0, 1]
  → Apply SMOTE on TRAIN split only (cap 50k samples/class)
```

After preprocessing, 3 attack classes (8, 9, 13) had insufficient samples (< 6 instances per class after cleaning) to train a per-class diffusion model. These classes are excluded from the adversarial generation phase; evasion evaluation covers **11 of 14 attack classes**.

---

## 5. System Architecture

### 5.1 High-Level Pipeline

DCTM is structured as a 9-phase pipeline with a single entry point (`main.py`). Each phase caches its outputs as Parquet files or model checkpoints, allowing individual phases to be re-run without repeating earlier steps.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DCTM PIPELINE                                │
│                                                                     │
│  Phase 1: PREPROCESS      Phase 2: FEATURES     Phase 3: TRAIN     │
│  ┌──────────────────┐    ┌──────────────────┐  ┌──────────────────┐│
│  │ Raw CSVs         │    │ MI Ranking       │  │ 10 IDS Classifiers││
│  │ → Clean          │ →  │ + SHAP Ranking   │→ │ (6 classical +   ││
│  │ → Label encode   │    │ → Hybrid top-20  │  │  4 deep learning)││
│  │ → Split 80/20    │    │ → features_dctm  │  │ + SMOTE balance  ││
│  │ → MinMax scale   │    │   .json          │  │                  ││
│  └──────────────────┘    └──────────────────┘  └──────────────────┘│
│           │                      │                      │           │
│           ▼                      ▼                      ▼           │
│  Phase 4: DIFFUSION                                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Per-class Transformer Denoiser training (11 attack classes)  │  │
│  │ T=1000 timesteps, linear beta schedule                        │  │
│  │ Output: diffusion_dctm_class{i}.pt (×11)                     │  │
│  └──────────────────────────────────────────────────────────────┘  │
│           │                                                         │
│           ▼                                                         │
│  Phase 5: ATTACK          Phase 6: EVALUATE     Phase 7: RETRAIN   │
│  ┌──────────────────┐    ┌──────────────────┐  ┌──────────────────┐│
│  │ Partial forward  │    │ Compare clean vs │  │ Augment train    ││
│  │ diffusion t=T/2  │ →  │ adversarial      │→ │ with adv samples ││
│  │ + Reverse denoise│    │ performance      │  │ → Retrain all    ││
│  │ + Immutable clamp│    │ → evasion_dctm   │  │   10 models      ││
│  │ + PGD refinement │    │   .csv           │  │                  ││
│  └──────────────────┘    └──────────────────┘  └──────────────────┘│
│           │                      │                      │           │
│           ▼                      ▼                      ▼           │
│  Phase 8: EXTERNAL        Phase 9: REPORT                           │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ CICIDS2018 eval  │    │ Summary report   │                      │
│  │ (transfer test)  │    │ + Final metrics  │                      │
│  └──────────────────┘    └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Diagram

```
                    CICIDS2017 raw CSVs
                           │
                    ┌──────▼──────┐
                    │ Preprocess  │ → data/processed/*.parquet
                    │             │ → data/splits/{train,test}.parquet
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌────────────┐ ┌─────────┐ ┌──────────┐
       │ MI Ranking │ │  SHAP   │ │  SMOTE   │
       │ (top-10)   │ │ (proxy  │ │ balance  │
       └────┬───────┘ │  LightGBM│ └────┬─────┘
            │         └────┬────┘       │
            └──────┬────────┘           │
                   ▼                    ▼
         features_dctm.json    Balanced train split
         (hybrid top-20)               │
                   │                   ▼
                   └──────────► IDS Model Training
                                (10 classifiers)
                                       │
                                       ▼
                              data/models/*.pkl / *.pt
                                       │
                   ┌───────────────────┘
                   ▼
          Diffusion Training
          (per attack class)
          TransformerDenoiser ×11
                   │
                   ▼
          Adversarial Generation
          ┌────────────────────┐
          │ 1. Sample real     │
          │    attack x₀       │
          │ 2. Forward: add    │
          │    noise → x_{T/2} │
          │ 3. Reverse: denoise│
          │    → x̃₀           │
          │ 4. Clamp immutable │
          │    features        │
          │ 5. PGD refinement  │
          │    (optional)      │
          └────────┬───────────┘
                   │
                   ▼
          data/adversarial/
          adv_samples_dctm_class{i}.parquet
                   │
                   ▼
          Evaluation: clean vs adversarial
          → evaluation/results/evasion_dctm.csv
                   │
                   ▼
          Adversarial Retraining
          → evaluation/results/retrained_dctm.csv
```

### 5.3 Module Structure

```
DCTM/
├── main.py                      ← Single entry point, phase dispatcher
├── configs/
│   ├── config.yaml              ← All hyperparameters
│   ├── features_baseline.json   ← Top-10 MI features
│   └── features_dctm.json       ← Top-20 hybrid MI+SHAP features
├── preprocessing/
│   ├── data_loader.py           ← CSV → parquet, label encoding
│   ├── normalizer.py            ← MinMaxScaler fit/transform
│   ├── splitter.py              ← Stratified 80/20 split
│   └── smote_handler.py         ← SMOTE (train split only)
├── feature_engineering/
│   ├── mutual_information.py    ← MI ranking
│   ├── shap_analysis.py         ← LightGBM + SHAP scores
│   ├── hybrid_selector.py       ← 0.5·MI + 0.5·SHAP hybrid ranking
│   └── feature_constraints.py  ← IMMUTABLE_COLS per dataset
├── models/
│   ├── trainer.py               ← Unified training loop
│   ├── classical/               ← DT, NB, LR, RF, XGBoost, SVM
│   └── deep_learning/           ← MLP, CNN, RNN, CNN-BiLSTM
├── diffusion/
│   ├── denoiser.py              ← TransformerDenoiser architecture
│   ├── noise_schedule.py        ← Linear / cosine beta schedule
│   ├── forward_process.py       ← q_sample(): add noise
│   ├── reverse_process.py       ← p_sample(): remove noise
│   └── adversarial_generator.py ← Full attack pipeline
└── evaluation/
    ├── evasion_evaluator.py     ← Clean vs adversarial comparison
    ├── retraining.py            ← Augmented retraining
    ├── metrics.py               ← ER, F1, Accuracy helpers
    └── report_generator.py      ← Final summary report
```

---

## 6. Methodology

### 6.1 Data Preprocessing

Raw CICIDS2017 CSVs are loaded, cleaned (NaN, Inf, duplicates removed), and the label column is encoded to integers (0 = Benign, 1–14 = attack classes). A stratified 80/20 train-test split is applied with a fixed seed (42) to ensure reproducibility. All features are scaled to [0, 1] using a MinMaxScaler fitted on the training split only (no test leakage).

**Class Balancing with SMOTE:**
CICIDS2017 is severely imbalanced — benign traffic accounts for ~80% of samples, while rare attacks (e.g., Heartbleed) represent < 0.1%. SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training split only, capped at 50,000 samples per class, to prevent RAM overflow while ensuring all attack classes have adequate representation. The test split is never resampled, preserving the natural class distribution for evaluation.

### 6.2 Feature Engineering — Hybrid MI + SHAP

DCTM uses two complementary feature importance signals:

**Mutual Information (MI):**
MI measures the statistical dependence between each feature and the class label. It captures any monotonic or non-linear relationship without assuming a particular model form. MI is computed on a 100,000-sample subsample for efficiency. The top-10 MI features form the **baseline** feature set, reproducing DEMGAN's selection strategy.

**SHAP (SHapley Additive exPlanations):**
A LightGBM proxy model is trained on a 50,000-sample subsample, then a SHAP TreeExplainer computes per-feature attribution scores. SHAP captures the actual contribution of each feature to the proxy model's decisions, including interaction effects.

**Hybrid Ranking:**
Both scores are normalized to [0, 1] and combined:

```
hybrid_score(f) = 0.5 × MI_norm(f) + 0.5 × SHAP_norm(f)
```

The top-20 features by hybrid score form the **DCTM** feature set. This expands the feature space from 10 (DEMGAN baseline) to 20, incorporating features that are important to linear and deep classifiers but not selected by MI alone.

**Selected DCTM Features (top-20 hybrid):**

| Rank | Feature | Hybrid Score | MI_norm | SHAP_norm | Immutable |
|---|---|---|---|---|---|
| 1 | Flow IAT Min | 0.623 | 0.247 | 1.000 | |
| 2 | Average Packet Size | 0.524 | 1.000 | 0.048 | |
| 3 | Packet Length Std | 0.513 | 0.955 | 0.072 | |
| 4 | Packet Length Mean | 0.508 | 0.955 | 0.062 | |
| 5 | Total Length of Bwd Packets | 0.500 | 0.888 | 0.113 | |
| 6 | Init_Win_bytes_backward | 0.492 | 0.813 | 0.171 | |
| 7 | Packet Length Variance | 0.476 | 0.952 | ~0 | |
| 8 | Total Length of Fwd Packets | 0.466 | 0.847 | 0.084 | |
| 9 | Bwd Packet Length Mean | 0.455 | 0.876 | 0.035 | |
| 10 | **Destination Port** | 0.453 | 0.663 | 0.244 | **YES** |
| 11 | Subflow Bwd Bytes | 0.444 | 0.887 | 0 | |
| 12 | Avg Bwd Segment Size | 0.438 | 0.877 | 0 | |
| 13 | Max Packet Length | 0.430 | 0.798 | 0.062 | |
| 14 | Subflow Fwd Bytes | 0.423 | 0.846 | 0 | |
| 15 | Init_Win_bytes_forward | 0.421 | 0.787 | 0.056 | |
| 16 | Fwd Packet Length Max | 0.421 | 0.790 | 0.052 | |
| 17 | Bwd Packet Length Max | 0.412 | 0.810 | 0.014 | |
| 18 | Flow IAT Max | 0.395 | 0.734 | 0.057 | |
| 19 | Fwd IAT Max | 0.391 | 0.755 | 0.027 | |
| 20 | Flow IAT Std | 0.379 | 0.605 | 0.153 | |

**Feature Constraints:**
Network-layer fields that an attacker cannot realistically modify are marked as immutable. For CICIDS2017: `Destination Port`, `URG Flag Count`, `CWE Flag Count`. Of these, only `Destination Port` appears in the DCTM top-20 (rank 10). During adversarial generation, immutable features are clamped back to their original values after every reverse diffusion step.

### 6.3 IDS Classifier Suite

DCTM trains and evaluates 10 classifiers sharing a unified interface (`.train()`, `.predict()`, `.predict_proba()`, `.save()`, `.load()`):

**Classical Classifiers (scikit-learn, CPU-only):**

| Model | Key Configuration |
|---|---|
| Decision Tree | max_depth=25, balanced class_weight |
| Naive Bayes | GaussianNB, var_smoothing=1e-8 |
| Logistic Regression | SAGA solver, max_iter=300, balanced |
| Random Forest | 200 trees, max_depth=25, balanced_subsample |
| XGBoost | 200 trees, depth=8, multi:softprob |
| SVM | RBF kernel, probability=True, 50k train cap |

**Deep Learning Models (PyTorch, GPU-accelerated):**

| Model | Architecture |
|---|---|
| MLP | 3-layer: [256, 128, 64], dropout=0.3 |
| CNN | 1D conv [32, 64 channels], kernel=3 |
| RNN | 2-layer BiGRU, hidden=128 |
| CNN-BiLSTM | CNN [32ch] + BiLSTM [64 hidden] |

All deep models are trained for 30 epochs with early stopping (patience=10), AdamW optimiser (lr=0.001), batch size 512.

### 6.4 Diffusion Model Architecture

One **TransformerDenoiser** is trained per attack class (not a single class-conditional model). This per-class design avoids conditional generation complexity and allows each denoiser to specialise to the statistical distribution of its attack class.

```
TransformerDenoiser Architecture
─────────────────────────────────

Input: x_t ∈ ℝ^d  (noisy sample, d = num features = 20)
       t ∈ {1,...,T}  (timestep)

┌─────────────────────────────────────────────────┐
│  Timestep Embedding                              │
│  sin/cos positional encoding → Linear → SiLU    │
│  dim: 128 → 256                                 │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Feature Projection                              │
│  Linear(d → 256) + LayerNorm                    │
└───────────────────┬─────────────────────────────┘
                    │
          [Add timestep embedding]
                    │
┌───────────────────▼─────────────────────────────┐
│  Transformer Encoder × 4 layers                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  Multi-Head Self-Attention  (nhead=4)        │ │
│  │  Feed-Forward Network  (dim=512)             │ │
│  │  LayerNorm + Dropout (0.1)                   │ │
│  └─────────────────────────────────────────────┘ │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│  Output Projection                               │
│  Linear(256 → d)                                │
│  Predicted noise: ε̂ ∈ ℝ^d                      │
└─────────────────────────────────────────────────┘
```

**Noise Schedule:**
Linear beta schedule over T=1000 timesteps:
- β₁ = 0.0001, β_T = 0.02
- α_t = 1 − β_t
- ᾱ_t = ∏_{s=1}^{t} α_s

**Training Objective:**
Standard DDPM noise prediction loss:

```
L = E_{x₀, ε, t} [ ||ε − ε̂_θ(x_t, t)||² ]
```

where x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε, ε ~ N(0, I).

**Training Configuration:** 100 epochs, AdamW (lr=1e-4, weight_decay=1e-4), gradient clipping at 1.0, batch size 256. GPU-accelerated (Tesla T4 in Colab).

### 6.5 Adversarial Sample Generation

Adversarial samples are generated using **partial forward diffusion + guided reverse diffusion**:

```
Adversarial Generation Process
──────────────────────────────

For each real attack sample x₀:

Step 1 — Partial Forward Diffusion
  Add noise up to t* = T/2 = 500:
  x_{t*} = √ᾱ_{t*} · x₀ + √(1−ᾱ_{t*}) · ε

  → This preserves the attack's semantic structure
    (high-level statistics, feature correlations) while
    introducing enough variation to fool the IDS.

Step 2 — Reverse Diffusion (Denoising)
  Iteratively denoise from t* back to t=0:
  x_{t-1} = (1/√α_t) * (x_t − (β_t/√(1−ᾱ_t)) · ε̂_θ(x_t, t))
           + √β_t · z,  z ~ N(0, I)

  After each step: clamp immutable features to original values.

Step 3 — PGD Refinement (optional, guidance_steps=10)
  Black-box gradient ascent toward P(class=benign):
  x̃ ← x̃ + lr · ∇_x log P(benign | x̃)
  (evaluated using {Decision Tree, LR, Naive Bayes} as surrogates)

Output: x̃₀ — adversarial sample
```

**Why t* = T/2?**
Using the halfway point of the forward process provides a balance:
- Too little noise (small t*): generated sample stays close to the original → easy to detect
- Too much noise (large t*): loses attack-specific structure → no longer a realistic adversarial variant of the original attack

**Immutable Feature Clamping:**
After every reverse diffusion step, the `Destination Port` feature is reset to its original value from x₀. This ensures the generated adversarial sample respects real network constraints.

**5,000 adversarial samples** are generated per attack class (55,000 total across 11 classes).

### 6.6 Evaluation Protocol

Evaluation compares each classifier's performance on:
1. **Clean test data** — the original 20% test split
2. **Adversarial test data** — the generated adversarial samples

**Evasion Rate (ER)** is the primary attack metric:

```
ER = (number of adversarial attack samples classified as benign) /
     (total adversarial attack samples)
```

A higher ER means the attack is more effective. **ER = 1.0** means every adversarial sample evades the IDS entirely.

All metrics are **macro-averaged** across 15 classes. The macro average weights each class equally, which penalises poor performance on rare classes. This is more demanding than weighted averaging and more informative for multi-class IDS evaluation.

Metrics reported: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, ROC-AUC (OVR macro), Evasion Rate.

**Note on surrogate/victim separation:**
Decision Tree, Naive Bayes, and Logistic Regression are used as PGD surrogates for the optional refinement step. For a strict black-box evasion assessment, the relevant models are **Random Forest, XGBoost, SVM, MLP, CNN, RNN, CNN-BiLSTM** (held-out, never seen during PGD optimisation).

### 6.7 Adversarial Retraining

After adversarial samples are generated, the training set is augmented with these samples (labelled with their original attack class) and all 10 classifiers are retrained from scratch. This simulates a defensive hardening step: if the IDS operator has access to the DCTM generator, they can use it to make their IDS robust to this attack class.

The retraining evaluation answers: **does exposure to adversarial samples during training allow classifiers to correctly identify adversarial traffic at inference time?**

---

## 7. Experimental Setup

| Setting | Value |
|---|---|
| Primary dataset | CICIDS2017 (8 CSV files) |
| Feature set | DCTM hybrid top-20 |
| Train/test split | 80% / 20%, seed=42 |
| SMOTE cap | 50,000 samples/class |
| Diffusion timesteps T | 1,000 |
| Beta schedule | Linear (β₁=0.0001, β_T=0.02) |
| Denoiser model_dim | 256 |
| Diffusion epochs | 100 |
| Partial forward fraction | t* = T/2 = 500 |
| Adversarial samples/class | 5,000 |
| PGD steps (main run) | 10 |
| PGD learning rate | 0.05 |
| PGD surrogates | Decision Tree, LR, Naive Bayes |
| GPU | Tesla T4 (Google Colab) |
| Framework | PyTorch 2.x, scikit-learn 1.x |
| Random seed | 42 (all components) |
| Diffusion coverage | 11/14 attack classes |
| Classes excluded | 8, 9, 13 (< 6 samples after cleaning) |

---

## 8. Results

### 8.1 Clean IDS Performance

The table below shows all 10 classifiers evaluated on the clean CICIDS2017 test set with the 20-feature DCTM feature set. Metrics are macro-averaged across all 15 classes.

| Model | Accuracy | Precision | Recall | **Macro-F1** | ROC-AUC | Train Time (s) |
|---|---|---|---|---|---|---|
| Decision Tree | 0.9911 | 0.684 | 0.951 | 0.747 | 0.984 | 13.9 |
| Naive Bayes | 0.390 | 0.319 | 0.635 | 0.293 | 0.946 | 0.3 |
| Logistic Regression | 0.410 | 0.236 | 0.701 | 0.247 | 0.956 | 223.7 |
| **Random Forest** | 0.9946 | 0.759 | 0.918 | **0.805** | 0.9998 | 340.1 |
| **XGBoost** | 0.9968 | 0.752 | 0.930 | **0.807** | 0.9998 | 20.1 |
| SVM | 0.562 | 0.334 | 0.768 | 0.348 | 0.973 | 8.3 |
| MLP | 0.887 | 0.389 | 0.930 | 0.461 | 0.998 | 230.4 |
| CNN | 0.873 | 0.398 | 0.913 | 0.466 | 0.997 | 251.4 |
| RNN | 0.946 | 0.529 | 0.907 | 0.592 | 0.997 | 405.1 |
| CNN-BiLSTM | 0.944 | 0.518 | 0.936 | 0.580 | 0.999 | 311.9 |

**Key observations:**
- RF and XGBoost achieve the strongest performance (macro-F1 ~0.81, ROC-AUC ~0.9998), confirming ensemble tree methods as the dominant IDS classifiers on this dataset.
- The accuracy ↔ F1 gap (e.g., XGBoost: accuracy 0.997 vs F1 0.807) reflects **macro-averaging over rare classes**, not overfitting. The models correctly classify high-frequency classes but score lower on minority attack classes that macro-F1 weights equally. ROC-AUC near 1.0 confirms excellent discriminative ranking.
- NB, LR, and SVM underfit the 15-class problem (macro-F1 0.25–0.35), confirming DEMGAN's finding that linear classifiers are weak targets on this dataset.

### 8.2 Evasion Attack Results

The following table compares each classifier's performance on **adversarial test data** (5,000 samples/class × 11 classes). The **PGD refinement step is enabled** (guidance_steps=10) in the primary results. ER = fraction of adversarial attack samples classified as benign.

| Model | Clean F1 | Clean ER | Adv F1 | Adv Accuracy | **Adv ER** |
|---|---|---|---|---|---|
| Decision Tree† | 0.747 | 0.0005 | 0.038 | 0.031 | 0.407 |
| Naive Bayes† | 0.293 | 0.0007 | 0.015 | 0.011 | **0.936** |
| Logistic Regression† | 0.247 | 0.0075 | 0.035 | 0.069 | 0.127 |
| Random Forest | 0.805 | 0.0002 | 0.001 | 0.0004 | 0.611 |
| XGBoost | 0.807 | 0.0002 | 0.018 | 0.013 | 0.599 |
| **SVM** | 0.348 | 0.0018 | 0.000 | 0.000 | **1.000** |
| MLP | 0.461 | 0.0015 | 0.018 | 0.029 | 0.592 |
| CNN | 0.466 | 0.0065 | 0.019 | 0.028 | 0.527 |
| RNN | 0.592 | 0.0009 | 0.012 | 0.018 | 0.495 |
| CNN-BiLSTM | 0.580 | 0.0014 | 0.015 | 0.054 | 0.150 |
| **Mean** | | | | | **0.544** |

† = PGD surrogate model (white-box leakage; exclude from black-box ER)

**Black-box ER (held-out models only, RF/XGB/SVM/MLP/CNN/RNN/CNN-BiLSTM):** mean = **0.568**

**Key observations:**
- Adversarial F1 collapses to near zero across all models — the attack is highly destructive to classification performance.
- The mean ER of 0.544 (all models) / 0.568 (black-box only) demonstrates that DCTM-generated adversarial samples consistently evade IDS classifiers.
- SVM achieves ER = 1.000 (complete evasion), though SVM is the weakest clean-data classifier (F1 = 0.348) so this result has limited practical significance.
- Transfer is strongest to RF and XGBoost (ER ~0.60) — the strongest classifiers — indicating DCTM's adversarial samples are practically meaningful against robust detectors.

### 8.3 Ablation Study — Diffusion vs PGD Contribution

To isolate the contribution of the diffusion model from the PGD refinement step, we re-ran the attack phase with `guidance_steps=0` (pure diffusion, no PGD). The table compares ER under both conditions.

| Model | ER (PGD=10) | ER (PGD=0) | Δ (PGD effect) |
|---|---|---|---|
| Decision Tree† | 0.407 | 0.281 | +0.126 |
| Naive Bayes† | 0.936 | 0.923 | +0.013 |
| Logistic Regression† | 0.127 | 0.091 | +0.036 |
| Random Forest | 0.611 | 0.617 | −0.005 |
| XGBoost | 0.599 | 0.605 | −0.006 |
| SVM | 1.000 | 1.000 | 0.000 |
| MLP | 0.592 | 0.596 | −0.004 |
| CNN | 0.527 | 0.525 | +0.002 |
| RNN | 0.495 | 0.497 | −0.003 |
| CNN-BiLSTM | 0.150 | 0.171 | −0.021 |
| **Mean** | **0.544** | **0.531** | **+0.014** |

† = PGD surrogate model

**Key finding: diffusion alone achieves 97.5% of the mean evasion effect.**

PGD adds only +0.014 to the mean ER across all models. Crucially, for the **held-out black-box models** (RF, XGB, MLP, CNN, RNN, CNN-BiLSTM), PGD provides **no meaningful benefit** — in fact, it slightly reduces ER for several models (likely due to PGD overfitting to the DT/LR/NB surrogate decision boundaries, which do not generalise to deep models and ensembles).

The PGD step's only meaningful benefit is on its own surrogate models, particularly Decision Tree (+0.126 ER), which is expected — PGD directly optimises against DT predictions in a white-box manner.

**Conclusion:** The Transformer-based Tabular Diffusion Model is the primary driver of evasion. The DCTM architecture is the genuine contributor to adversarial IDS evasion, not the optional PGD refinement.

### 8.4 Adversarial Retraining Results

All classifiers were retrained on the augmented dataset (original training data + adversarial samples) and re-evaluated. Results show the change in both clean-data performance (to detect accuracy regressions) and adversarial performance.

| Model | Clean F1 (Δ) | Clean ER (Δ) | Adv F1 | Adv Accuracy | **Adv ER** (was) |
|---|---|---|---|---|---|
| Decision Tree | 0.757 (+0.010) | 0.0005 (≈0) | 0.507 | 0.644 | **0.002** (0.407) |
| Naive Bayes | 0.188 (−0.105) | 0.392 (**+0.391** ⚠) | 0.124 | 0.149 | **0.000** (0.936) |
| Logistic Regression | 0.214 (−0.033) | 0.080 (**+0.072** ⚠) | 0.115 | 0.148 | **0.006** (0.127) |
| **Random Forest** | **0.820 (+0.015)** | 0.0002 (≈0) | **0.956** | **0.955** | **0.000** (0.611) |
| **XGBoost** | **0.825 (+0.018)** | 0.0001 (≈0) | **0.844** | **0.844** | **0.000** (0.599) |
| SVM | 0.322 (−0.026) | 0.143 (**+0.141** ⚠) | 0.058 | 0.109 | **0.000** (1.000) |
| MLP | 0.470 (+0.008) | 0.002 (≈0) | 0.177 | 0.254 | **0.000** (0.592) |
| CNN | 0.475 (+0.009) | 0.006 (≈0) | 0.059 | 0.127 | **0.000** (0.527) |
| RNN | 0.555 (−0.038) | 0.0005 (≈0) | 0.060 | 0.131 | **0.000** (0.495) |
| CNN-BiLSTM | 0.560 (−0.020) | 0.001 (≈0) | 0.071 | 0.130 | **0.000** (0.150) |

**Genuine robustness — Random Forest and XGBoost:**
After retraining, RF achieves adversarial F1 = 0.956 and XGBoost adversarial F1 = 0.844, with clean F1 unchanged or slightly improved. Adversarial ER drops to 0.000 for both. These models have learned to correctly re-identify adversarial samples as attacks, not merely avoid the "benign" label. This is the strongest result of the paper: **DCTM-generated adversarial samples can be used to genuinely harden ensemble tree classifiers.**

**Degenerate robustness — SVM, MLP, CNN, RNN, CNN-BiLSTM:**
For these models, ER drops to 0.000 but adversarial accuracy remains only 0.11–0.25. They have learned to avoid predicting "benign" on adversarial inputs, but still misclassify 75–89% of adversarial attacks into the wrong attack class. This is a **metric artifact** — ER=0 does not imply correct identification.

**Clean performance regression — NB, LR, SVM:**
Adversarial retraining caused NB's clean ER to increase from 0.0007 to 0.392 (NB now misses 39% of real attacks), and SVM's clean ER increased from 0.0018 to 0.143. This is the **clean/adversarial accuracy trade-off**, a well-known effect in adversarial ML where low-capacity models cannot simultaneously learn the original decision boundary and the adversarial augmentation.

---

## 9. Discussion and Analysis

### 9.1 Why Diffusion Outperforms PGD as an Adversarial Generator

The ablation study confirms that the diffusion model is the primary evasion mechanism. This has an intuitive explanation: PGD is a gradient-based method that optimises directly against specific surrogate models. It produces adversarial examples that are highly effective against those exact models but transfer poorly to architecturally different models (RF, deep networks) — a well-known weakness of white-box PGD.

The diffusion model, by contrast, does not optimise against any specific classifier. It generates samples from the learned distribution of the attack class, perturbed through a diffusion process. These perturbations are statistically plausible variations of real attacks, and they evade classifiers because they explore regions of feature space that are ambiguous to multiple architectures simultaneously. This is a fundamentally more transferable attack.

### 9.2 The Accuracy–F1 Gap Explained

The large gap between accuracy and macro-F1 (e.g., XGBoost: acc=0.997, F1=0.807) is not a sign of overfitting. CICIDS2017's 15-class distribution is severely imbalanced — benign traffic represents ~80% of samples, and some attack classes have fewer than 1,000 test samples. A macro-F1 calculation weights each class equally regardless of frequency. A classifier that perfectly handles the 5 most common classes but struggles on 10 rare ones will have high accuracy (because most samples are in the 5 common classes) but much lower macro-F1 (because macro-F1 penalises poor rare-class performance equally). ROC-AUC near 1.0 across all models confirms excellent discriminative power.

### 9.3 DEMGAN Comparison

DEMGAN reports a 97.42% evasion rate on CICIDS2017. A direct comparison with DCTM's results requires care:

- DEMGAN's 97.42% is likely reported for a specific best-performing model or binary classification setup.
- DCTM's 0.544 mean ER is averaged across **10 models** in a **15-class** setting, which is a significantly harder evaluation.
- The strongest DCTM results (SVM ER=1.00, NB ER=0.936) match or exceed DEMGAN's figure for specific models.
- For the practically important strong models (RF ER=0.611, XGBoost ER=0.599), DCTM achieves substantial evasion without any class-conditional guidance.

The appropriate comparison is not a single number but the combination of evasion effectiveness **and** the quality of the adversarial samples for defensive retraining — an axis on which DCTM's RF/XGB hardening results (adv F1 0.956/0.844) are a genuinely novel outcome that DEMGAN does not address.

### 9.4 Why RF and XGBoost Retrain Well

Ensemble tree methods have high model capacity and are inherently non-smooth decision boundaries. When adversarial samples are added to their training data, they can carve out new decision regions around the adversarial cluster without disturbing their existing decision boundaries. This is in contrast to linear models (LR, NB) which must globally shift their decision boundary to accommodate the augmented data, causing the clean-performance regressions observed.

---

## 10. Limitations

### 10.1 PGD Surrogate Overlap
Decision Tree, Logistic Regression, and Naive Bayes are used as both PGD surrogates and evaluation victims. When these models appear in the evasion results, their ER includes white-box information leakage. The strict black-box ER (excluding surrogates) should be reported separately, which yields a mean ER of 0.568 across the 7 held-out models.

### 10.2 Incomplete Attack Class Coverage
Classes 8, 9, and 13 (3 of 14 attack classes) have insufficient samples to train a per-class diffusion model under the current minimum-samples threshold. Evasion evaluation covers 11/14 attack types. Future work can address this with class-oversampling or by reducing the minimum-samples constraint.

### 10.3 Constraint Pillar Effectiveness
Of the 20 DCTM features, only Destination Port (rank 10) is marked immutable. The two other CICIDS2017 immutable features (URG Flag Count, CWE Flag Count) are not among the top-20 by hybrid score and therefore do not enter the model at all. The immutable clamping mechanism is technically implemented and active, but its practical effect is limited to a single feature. Expanding the immutable list to all semantically fixed network fields would strengthen this pillar.

### 10.4 Degenerate ER After Retraining
For SVM, MLP, CNN, RNN, and CNN-BiLSTM, post-retrain ER=0.000 is a metric artifact — these models learned to avoid the "benign" label but still misclassify 75–89% of adversarial attacks. The adversarial F1 (0.06–0.18) and adversarial accuracy (0.11–0.25) for these models after retraining should be reported alongside ER to avoid overstating the robustness improvement.

### 10.5 Dataset Limitations
CICIDS2017 is a known "easy" benchmark with near-duplicate flows and partial label artifacts. RF/XGBoost at 0.99 accuracy is typical and partly reflects dataset structure. The external validation phase (CICIDS2018) is the appropriate test of whether clean performance generalises beyond the training dataset.

### 10.6 Hybrid Feature Score Distribution
The SHAP contribution to the hybrid score is near-zero for most selected features (the ranking is dominated by MI). Feature 1 (Flow IAT Min, SHAP_norm=1.0) is the only strong SHAP contributor. The "MI + SHAP hybrid" is in practice closer to "MI ranking with SHAP tie-breaking." Future work could re-weight the hybrid, add a third signal (permutation importance), or use Borda count aggregation.

---

## 11. Conclusion

This paper presented **DCTM**, a Transformer-based Tabular Diffusion Model framework for adversarial IDS evaluation and hardening. DCTM replaces the GAN-based generator of the DEMGAN baseline with a stable diffusion process, adds hybrid MI+SHAP feature selection (20 features vs DEMGAN's 10), SMOTE class balancing, and a network-constraint system.

An ablation study confirms that the diffusion model is the genuine driver of evasion, achieving 53.1% mean evasion rate on 10 IDS classifiers without any PGD refinement — 97.5% of the total evasion effect. The optional PGD step adds only 1.4 percentage points and provides no benefit against held-out black-box models.

The strongest result is defensive: adversarial retraining with DCTM-generated samples fully hardens Random Forest (adversarial F1 = 0.956) and XGBoost (adversarial F1 = 0.844) while maintaining or slightly improving clean-data performance. This demonstrates that DCTM's samples are practically useful for building more resilient IDS deployments.

Future work will address the remaining limitations: extending diffusion coverage to all 14 attack classes, implementing cross-dataset transfer evaluation (train on CICIDS2017, attack CICIDS2018-trained IDS), strengthening the feature constraint system, and exploring classifier-guided reverse diffusion to close the gap with DEMGAN's best-case evasion figures.

---

## 12. References

1. Xu, Z., et al. (2025). "DEMGAN: Diverse Ensemble Multi-Generator Adversarial Network for Intrusion Detection System Evaluation." *Computers, Materials & Continua (CMC)*.

2. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS 2020*.

3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.

4. Kotelnikov, A., et al. (2023). "TabDDPM: Modelling Tabular Data with Diffusion Models." *ICML 2023*.

5. Lundberg, S. M., & Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions (SHAP)." *NeurIPS 2017*.

6. Goodfellow, I., et al. (2014). "Generative Adversarial Nets." *NeurIPS 2014*.

7. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)." *ICLR 2018*.

8. Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR 2002*.

9. Sharafaldin, I., et al. (2018). "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization." *ICISSP 2018*. [CICIDS2017 dataset]

10. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.

11. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.

---

*Document Version 1.0 — Generated June 2026*
*All results from CICIDS2017 dataset, DCTM feature set, seed=42*
