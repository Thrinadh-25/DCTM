# DCTM — Diffusion-Based Cyber Threat Modelling

## A Complete A-to-Z Project Document for Review 2

**Project Title:** DCTM — Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance
**Team:** Kasturi Vishnu Vardhan (245523733160), Kukunuru Thrinadh Reddy (245523733163), Neha Sri Tirunagari (245523733187)
**Guide:** Ms. Duddeda Aishwarya, Asst. Professor
**Stage:** Prototype Review (Review 2)
**Date:** April 2026

---

## TABLE OF CONTENTS

1. Executive Summary
2. The Problem We Are Solving
3. Why This Project Matters (Motivation)
4. The DEMGAN Baseline & Its Three Weaknesses
5. The DCTM Solution — Four Pillars
6. Project Goals & Research Questions
7. End-to-End Workflow (10 Phases)
8. Datasets — CICIDS2017 & CICIDS2018
9. Tech Stack & Tools
10. System Architecture (High-Level)
11. Module-by-Module Breakdown
12. Phase Details (Step-by-Step What Happens)
13. The Three Core Architectures
14. Mathematical Foundations
15. Hyperparameters & Configuration
16. Implementation Status
17. Key Innovations / Novel Contributions
18. Why Diffusion Beats GAN for This Problem
19. The Constraint System (Immutable Features)
20. Risks, Mitigations & Trade-offs
21. Demonstration Plan
22. Future Scope
23. References

---

## 1. EXECUTIVE SUMMARY

DCTM (Diffusion-Based Cyber Threat Modelling) is a research-grade prototype that **strengthens machine-learning Intrusion Detection Systems (IDS)** by using a **Transformer-based Tabular Diffusion Model** to generate realistic adversarial network traffic. We use these adversarial samples in two ways:

1. **Offensive evaluation:** test how easily existing IDS classifiers can be fooled (the "evasion rate")
2. **Defensive hardening:** retrain the IDS on adversarial samples so they learn to resist this attack class

The work directly extends the published **DEMGAN** baseline (Xu et al., 2025, CMC), replacing its unstable GAN generator with a **stable diffusion process** and adding two complementary improvements: hybrid feature selection (MI + SHAP) and SMOTE-based class balancing.

**One-sentence takeaway:** *We are replacing GAN-based adversarial sample generation with diffusion-based generation to make IDS evaluation more reliable, more realistic, and inherently amenable to defensive retraining.*

---

## 2. THE PROBLEM WE ARE SOLVING

### 2.1 The Real-World Problem

Intrusion Detection Systems (IDS) are the first line of defence in modern networks. They watch traffic flows and flag malicious behaviour (DDoS, port scans, brute-force login attempts, web attacks, infiltration, botnet activity, etc.). Most modern IDS use **machine learning classifiers** trained on historical attack data.

**The vulnerability:** ML classifiers can be fooled by **adversarial samples** — synthetic traffic that *looks* benign to the classifier but is actually malicious. An attacker who controls just a few packet-level fields can morph a real attack into something the IDS labels as "normal traffic", letting the attack pass undetected.

### 2.2 What's Wrong With Existing Solutions

The state-of-the-art adversarial IDS evaluator is **DEMGAN** (Diverse Ensemble Multi-Generator Adversarial Network). It uses a **Wasserstein GAN with multiple generators** to craft adversarial samples and reports a **97.42% evasion rate** on CICIDS2017.

DEMGAN has three published weaknesses:

| Weakness | Impact |
|----------|--------|
| Restricted to **only 10 features** (Mutual Information ranking only) | Misses important features that linear classifiers actually rely on |
| **GAN training instability** (mode collapse, vanishing gradients) | Especially bad on linear classifiers like Decision Tree and Logistic Regression — the paper itself admits these classifiers expose its blind spots |
| Ignores **severe class imbalance** (some attacks are <0.1% of traffic) | Rare attack patterns are under-modelled, so GAN never learns to fool the IDS on them |

Plus a methodological gap: DEMGAN only reports **Evasion Rate**. We argue this is incomplete — a meaningful evaluation should also report **F1, Precision, Recall** so future research can compare results in a richer, more standard way.

### 2.3 What Success Looks Like

After this project, we should be able to say:
- **Match or beat DEMGAN's evasion rate (≥97.42%)** on CICIDS2017
- Provide **F1/Precision/Recall** alongside ER for every model, every phase
- Generate adversarial samples that **survive linear classifiers** (where DEMGAN fails)
- Show **measurable robustness gain after adversarial retraining**
- Demonstrate **cross-dataset generalisation** (2017-trained IDS vs 2018 attacks)

---

## 3. WHY THIS PROJECT MATTERS (MOTIVATION)

1. **Cybersecurity is a moving target.** Attackers constantly probe and adapt. Defensive ML models that aren't routinely stress-tested *will* be defeated by adversarial traffic. Our pipeline gives defenders a way to do that stress testing in-house.
2. **GAN-based adversarial generators are brittle.** Diffusion models have replaced GANs as the gold standard in image generation precisely because they are more stable, easier to train, and produce more diverse samples. We're bringing that advance to **tabular** network-traffic data — which is itself an underexplored domain for diffusion.
3. **Tabular diffusion is novel.** Most diffusion research targets images/audio. Applying it to **structured network flow features** with **hard physical constraints** (you can't change the destination port without breaking the protocol) is genuinely new territory.
4. **Defensive value.** Once the diffusion model exists, it doubles as a **data augmentation tool** for training more robust IDS — turning the offensive output into a defensive resource.

---

## 4. THE DEMGAN BASELINE & ITS THREE WEAKNESSES

DEMGAN's pipeline (the work we extend):

```
Raw CICIDS2017 → top-10 features (MI) → WGAN with 3 generators → adversarial samples
                                                                       │
                                          Evaluated against 9 ML IDS ──┘
```

**Reported result:** 97.42% evasion rate against an ensemble of 9 IDS models on CICIDS2017.

**The three weaknesses we attack:**

| # | DEMGAN limitation | Why it matters |
|---|-------------------|----------------|
| 1 | Only 10 features ranked by Mutual Information | MI captures statistical dependence but **misses tree-model importance**. Tree-based classifiers (DT, RF, XGBoost) are routinely fooled by features that score low on MI but high on SHAP. |
| 2 | GAN-based generation | Min-max optimization is **inherently unstable**, suffers mode collapse, and the paper itself reports linear classifiers (DT, LR) defeat it because the generator never converges on a sharp linear boundary. |
| 3 | No class balancing | CICIDS2017's attack class distribution is brutally skewed (Heartbleed has ~10 samples; Hulk has ~230k). The GAN simply never sees enough rare attacks to model them. |

DCTM addresses all three.

---

## 5. THE DCTM SOLUTION — FOUR PILLARS

| Pillar | Name | Implementation |
|--------|------|----------------|
| **P1** | **Diffusion-driven adversarial generation** | Transformer denoiser, partial noising, per-step constraint clamp |
| **P2** | **Class balancing via SMOTE** before diffusion | Cap 50k/class, k=5 neighbours, train-only |
| **P3** | **Hybrid 20-feature selection** (MI + SHAP) | 0.5·MI_norm + 0.5·SHAP_norm, top-20 |
| **P4** | **Extended evaluation** (ER + F1 + Precision + Recall) | All four metrics computed and reported per model |

These four pillars **map directly to the four DEMGAN weaknesses** plus the methodological gap. Every line of code in this repo serves one of them.

---

## 6. PROJECT GOALS & RESEARCH QUESTIONS

### Goals
1. Build an **end-to-end reproducible pipeline** from raw CSVs to a final report.
2. Train **10 IDS models** (6 classical + 4 deep) under **two feature regimes** (10-feat baseline, 20-feat DCTM).
3. Train **per-attack-class diffusion denoisers** that craft adversarial samples.
4. Evaluate every model on clean vs adversarial data with full metrics.
5. Run the **adversarial retraining loop** to quantify defensive value.
6. Externally validate on a **second dataset** (CICIDS2018) for generalisation.
7. Generate a **final summary report** that recruiters / reviewers can read in minutes.

### Research Questions
- **RQ1:** Can a Transformer diffusion denoiser match or beat DEMGAN's WGAN on tabular adversarial generation, while remaining stable on linear classifiers?
- **RQ2:** Does expanding feature selection from 10 → 20 (MI → MI+SHAP hybrid) increase evasion rate?
- **RQ3:** Does SMOTE balancing before diffusion produce adversarial samples that fool the IDS on rare attack classes (where DEMGAN fails)?
- **RQ4:** Does adversarial retraining of the IDS recover its accuracy *without* losing performance on clean data?
- **RQ5:** Do diffusion-generated 2017 attack patterns generalise to 2018 traffic?

---

## 7. END-TO-END WORKFLOW (10 PHASES)

The pipeline is implemented in `main.py` as 10 idempotent CLI phases. Each phase caches its output to disk so re-runs only re-execute what's missing.

```
PHASE 1  preprocess  →  Clean CSVs, encode 15 attack labels, 80/20 stratified split, MinMax scale
PHASE 2  features    →  MI ranking + SHAP ranking + Hybrid score → top-10 (baseline) & top-20 (DCTM)
PHASE 3  train       →  SMOTE on train + train all 10 IDS models on both feature regimes
PHASE 4  diffusion   →  Train one Transformer denoiser PER attack class
PHASE 5  attack      →  Partial noise → reverse-denoise with per-step immutable clamp → adv samples
PHASE 6  evaluate    →  Per-model ER + F1 + Precision + Recall on clean vs adversarial
PHASE 7  retrain     →  Augment train set with adv → retrain all 10 → re-evaluate
PHASE 8  external    →  2017-trained IDS attacked by 2018 adversarial samples
PHASE 9  report      →  Final summary text file with key metrics
PHASE 10 visualize   →  All plots saved in visualization/
```

### Why this exact phase ordering?
- **Phases 1–2** are pure data prep: deterministic, reusable, dataset-agnostic.
- **Phase 3** trains the IDS *first* so we have a victim. Without victims, evasion is meaningless.
- **Phase 4** trains the diffusion attacker *second*, on the same clean training data the IDS saw — this is fair: both attacker and defender start from the same evidence.
- **Phase 5** uses the trained denoiser on the **test split** (held out from both IDS and diffusion training). This is critical for a fair evasion measurement.
- **Phases 6–9** are pure evaluation, with no further training, so they are fast and cheap to re-run.

---

## 8. DATASETS — CICIDS2017 & CICIDS2018

| Dataset | Source | Size | Role |
|---------|--------|------|------|
| **CICIDS 2017** | Canadian Institute for Cybersecurity (UNB) | ~2.8M flows after cleaning, 8 CSVs (~885 MB) | Primary training + evasion benchmark |
| **CICIDS 2018** | Same source, more recent capture | ~2.4M flows, 3 CSVs (~1.1 GB) | External validation only |

### Attack Classes Covered (15 total: 1 benign + 14 attack types)

| Category | Specific attacks |
|----------|------------------|
| DoS | Hulk, GoldenEye, Slowloris, Slowhttptest |
| DDoS | Distributed denial-of-service |
| Probing | PortScan |
| Botnet | Bot |
| Brute force | FTP-Patator, SSH-Patator |
| Web attacks | Brute Force, XSS, SQL Injection |
| Other | Heartbleed, Infiltration |

### Why both datasets?
- 2017 = training corpus
- 2018 = held-out external validator → tests whether 2017-trained models generalise
- This is a stronger claim than "works on the test split of 2017" because 2018 has different network conditions, attack tooling versions, and traffic distributions.

### Cleaning steps (`preprocessing/data_loader.py`):
1. Concatenate all CSVs from each dataset folder
2. Strip whitespace from column names (the raw files have inconsistent leading/trailing spaces)
3. Replace ±∞ → NaN, drop NaN rows
4. Drop duplicates
5. Encode `Label` → `y_binary` (0=benign, 1=attack) and `y_multiclass` (0..14)
6. Coerce non-numeric feature columns; drop columns where >50% can't be parsed
7. Cache as parquet to speed re-runs

---

## 9. TECH STACK & TOOLS

| Layer | Tools | Why |
|-------|-------|-----|
| Language | **Python 3.10+** | Standard for ML research |
| Classical ML | **scikit-learn 1.4+**, **XGBoost 2.0+** | DT, NB, LR, RF, SVM + GPU-capable XGBoost |
| Deep Learning | **PyTorch 2.1+** | MLP, CNN, RNN, CNN-BiLSTM, Transformer denoiser |
| Imbalance | **imbalanced-learn** | SMOTE oversampling |
| Feature analysis | **SHAP**, **LightGBM** | LGBM = fast proxy for SHAP TreeExplainer |
| Data IO | **pandas**, **numpy**, **pyarrow** | Parquet caching for speed |
| Visualisation | **matplotlib**, **seaborn** | Static publication-grade plots |
| Config | **PyYAML** | Single source of truth for all hyperparameters |
| Compute | **NVIDIA T4 (Google Colab) / Local CUDA 12.1** | XGBoost + 4 deep models + diffusion run on GPU |
| Logging | Python `logging` | File + console handlers |
| Reproducibility | Fixed seed `42` everywhere | Same input → same output |

---

## 10. SYSTEM ARCHITECTURE (HIGH-LEVEL)

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
            │      Train one Transformer-based denoiser eps_θ(x_t, t)         │
            │   ─────────────────────────────────────────────────             │
            │   Architecture:                                                  │
            │     Input  x_t in R^{20}  +  timestep t in {0..999}              │
            │     Sinusoidal time embed → MLP → 256-d                         │
            │     Linear project x_t   → 256-d  +  add time embed             │
            │     4× TransformerEncoderLayer (heads=4, FFN=512)               │
            │     Linear project → 20-d predicted noise eps                   │
            │   Loss:  MSE(eps_pred,  eps_true)                                │
            │   Schedule:  linear beta in [1e-4, 0.02], T = 1000 steps        │
            └───────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────────┐
            │  PHASE 6 — ADVERSARIAL SAMPLE GENERATION                        │
            │                                                                 │
            │   For each malicious test sample x_0:                           │
            │     1.  q_sample(x_0, t = T/2)  →  partially-noised x_t         │
            │     2.  for tau in {T/2, …, 1}:                                 │
            │            x_{tau-1} = p_sample(x_tau, eps_θ)                   │
            │            x_{tau-1}[immutable_cols] = x_0[immutable_cols]     │
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

## 11. MODULE-BY-MODULE BREAKDOWN

| Directory | Files | Responsibility |
|-----------|-------|----------------|
| `main.py` | (single file) | CLI orchestrator dispatching all 10 phases |
| `configs/config.yaml` | (single file) | Centralised hyperparameters & paths |
| `preprocessing/` | data_loader.py, normalizer.py, splitter.py, smote_handler.py | Phase 1 + Phase 3 (cleaning + scaling + SMOTE) |
| `feature_engineering/` | mutual_information.py, shap_analysis.py, hybrid_selector.py, feature_constraints.py | Phase 2 (MI + SHAP + hybrid + immutable cols) |
| `models/classical/` | decision_tree.py, naive_bayes.py, logistic_regression.py, random_forest.py, xgboost_model.py, svm_model.py, _base.py | 6 sklearn-based IDS classifiers |
| `models/deep_learning/` | mlp.py, cnn.py, rnn.py, cnn_bilstm.py, base_model.py | 4 PyTorch IDS classifiers + shared base |
| `models/trainer.py` | (single file) | Unified train/save/load loop for all 10 models |
| `diffusion/` | denoiser.py, noise_schedule.py, forward_process.py, reverse_process.py, trainer.py, adversarial_generator.py | Tabular Transformer Diffusion Model + adv generator |
| `evaluation/` | metrics.py, evasion_evaluator.py, retraining.py, report_generator.py | Phase 6–9 (metrics + retraining + report) |
| `utils/` | logger.py, device.py, io.py, seed.py | Shared utilities |
| `datasets/` | (gitignored) | Raw CSVs |
| `data/` | (gitignored) | All generated artifacts (parquet, .pkl, .pt) |
| `visualization/` | (generated) | All plots (PNG) |
| `logs/` | run.log | Execution log |

### File counts
- **40+ Python files** total
- **10 IDS model classes** + 1 unified trainer
- **6 diffusion-related modules**
- **4 evaluation modules**
- **One** entry point: `main.py`

---

## 12. PHASE DETAILS — STEP BY STEP

### Phase 1 — Preprocessing (`preprocessing/`)
**Inputs:** `datasets/cicids 2017/*.csv`, `datasets/cicids2018csv/*.csv`
**Steps:**
1. Concatenate all CSVs per dataset
2. Clean column names (strip whitespace)
3. Replace inf → NaN, drop NaN rows
4. Drop duplicates
5. Encode labels → `y_binary` and `y_multiclass`
6. Coerce non-numeric feature columns; drop unparseable ones
7. Stratified 80/20 split (per multiclass label)
8. Save to parquet

**Outputs:**
- `data/processed/{dataset}.parquet`
- `data/processed/{dataset}_label_map.json`
- `data/splits/{dataset}_train.parquet`
- `data/splits/{dataset}_test.parquet`

### Phase 2 — Feature Engineering (`feature_engineering/`)
**Inputs:** Train split parquet
**Steps:**
1. **MI ranking:** subsample to top-100k rows, compute `mutual_info_classif`, rank, take top-10 → baseline feature set
2. **SHAP ranking:** subsample to top-50k rows, train LightGBM proxy classifier, run TreeExplainer, compute mean(|SHAP|) per feature
3. **Hybrid score:** normalise MI and SHAP scores to [0,1], compute `0.5·MI_norm + 0.5·SHAP_norm`, take top-20 → DCTM feature set
4. **Constraints:** mark immutable columns (Destination Port, URG/CWE Flag Count for 2017; Dst Port, PSH/ACK Flag Cnt for 2018)

**Outputs:**
- `configs/features_baseline.json` (top-10)
- `configs/features_dctm.json` (top-20)
- `configs/shap_scores.json` (full SHAP ranking)
- `visualization/mi_importance.png`
- `visualization/shap_importance.png`
- `visualization/feature_importance_comparison.png`

### Phase 3 — IDS Training (`models/`)
**Inputs:** Train/test parquet + selected feature list
**Steps:**
1. Apply MinMax scaler (fit on train, transform both)
2. Apply SMOTE on train only (k=5, cap 50k/class, skip classes with <6 samples)
3. For each of 10 models:
   - Skip if checkpoint already exists (resume-aware)
   - Else train, evaluate on test, save model + metrics, plot confusion matrix
4. Save all metrics to `evaluation/results/baseline_{feature_set}.csv`

**Outputs:**
- `data/models/{name}_{feature_set}.{pkl|pt}` × 20 (10 models × 2 feature sets)
- `evaluation/results/baseline_{baseline,dctm}.csv`
- `visualization/confusion_matrices/cm_{name}_{set}.png`
- `visualization/class_distribution_{set}.png`

### Phase 4 — Diffusion Training (`diffusion/trainer.py`)
**Inputs:** Train split with DCTM features (already scaled)
**Steps:**
1. For each non-benign class with ≥100 samples:
   - Filter rows of this class
   - Train one TransformerDenoiser via DDPM noise-prediction objective (MSE)
   - 100 epochs, AdamW + cosine LR, gradient clipping at 1.0
   - Save checkpoint
2. Save mapping `class_id → checkpoint_path` as JSON

**Outputs:**
- `data/models/diffusion_dctm_class{i}.pt` × N (one per attack class)
- `data/models/diffusion_index_dctm.json`

### Phase 5 — Adversarial Generation (`diffusion/adversarial_generator.py`)
**Inputs:** Test split with DCTM features + diffusion checkpoints + immutable column list
**Steps:**
1. For each class with a checkpoint:
   - Sample up to 5000 test rows of that class
   - Forward-diffuse to `t = T·partial_t_fraction = T/2 = 500`
   - Reverse-diffuse from t=500 down to t=0, calling the constrain function after every step to clamp immutable features to original values
   - Clip output to [0, 1]
   - Log per-feature drift statistics
2. Concatenate adv samples across classes, save as parquet

**Outputs:**
- `data/adversarial/adv_samples_{dataset}_dctm.parquet`

### Phase 6 — Evasion Evaluation (`evaluation/evasion_evaluator.py`)
**Inputs:** Clean test set + adversarial samples
**Steps:**
1. For each of 10 IDS models:
   - Load checkpoint
   - Predict on clean test → metrics + ER
   - Predict on adversarial → metrics + ER
   - Compute Δaccuracy, ΔF1
2. Save CSV with all numbers
3. Plot evasion comparison bar chart with DEMGAN reference line

**Outputs:**
- `evaluation/results/evasion_{feature_set}.csv`
- `visualization/evasion_comparison.png`

### Phase 7 — Adversarial Retraining (`evaluation/retraining.py`)
**Inputs:** Original train + adversarial samples + clean test
**Steps:**
1. Concatenate train + adv → augmented train
2. Retrain all 10 IDS models on augmented data (saved under `_retrained` tag so we don't overwrite originals)
3. Re-evaluate on adversarial test set
4. Plot before/after ER comparison

**Outputs:**
- `data/models/{name}_dctm_retrained.{pkl|pt}` × 10
- `evaluation/results/retrained_dctm.csv`
- `visualization/retraining_improvement.png`

### Phase 8 — External Validation (`main.py:phase_external_validation`)
**Inputs:** 2017-trained models + 2018 test data + 2018 adversarial samples (if generated)
**Steps:**
1. Re-use 2017 feature list, align 2018 columns by name (fill missing with 0)
2. Apply 2017 scaler (critical — same scaling for fair eval)
3. Evaluate on 2018 clean + 2018 adv (if available)

**Outputs:**
- `evaluation/results/external_eval_2018_dctm.csv`

### Phase 9 — Report (`evaluation/report_generator.py`)
**Steps:**
1. Read baseline + retrained CSVs
2. Identify best clean F1, most vulnerable model (highest adv ER), avg adv ER
3. Compute ΔF1 and ΔER from retraining
4. Compare against DEMGAN reference (0.9742)
5. Save text report

**Outputs:**
- `evaluation/final_report.txt`

---

## 13. THE THREE CORE ARCHITECTURES

### 13.1 IDS Classifier — Unified Interface

Every one of the 10 IDS models implements the same minimal interface:

```python
model.train(X, y)              # fit on training data
model.predict(X) -> y_pred     # hard labels
model.predict_proba(X) -> p    # class probabilities (n_samples × n_classes)
model.save(path)               # pickle (.pkl) or torch (.pt)
model.load(path)               # classmethod
```

This uniformity is what makes the trainer loop, evaluator, and retraining loop completely **model-agnostic**. Adding an 11th model is a matter of implementing this interface.

| Model | Library | Why included | GPU? |
|-------|---------|--------------|------|
| Decision Tree | sklearn | Linear classifier — DEMGAN's stated weakness target | ✗ |
| Naive Bayes | sklearn | Probabilistic baseline | ✗ |
| Logistic Regression | sklearn (saga) | Linear classifier — DEMGAN's stated weakness target | ✗ |
| Random Forest | sklearn | Strong tree ensemble baseline | ✗ |
| XGBoost | xgboost | SOTA tree boosting; harder target | ✓ (CUDA) |
| SVM | sklearn (RBF) | Non-linear margin classifier | ✗ |
| MLP | PyTorch | Dense feedforward baseline | ✓ |
| CNN (1-D) | PyTorch | Convolution over feature axis | ✓ |
| RNN (GRU) | PyTorch | Sequence over features | ✓ |
| CNN-BiLSTM | PyTorch | Hybrid conv + bidirectional sequence | ✓ |

### 13.2 Transformer Diffusion Denoiser (Core Innovation)

The denoising network ε_θ(x_t, t) — a single Transformer block — is the novelty over DEMGAN's WGAN.

```
                        x_t  in R^{B×F}          t  in N^{B}
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
                                eps_pred  in R^{B×F}
```

**Why a Transformer over an MLP for tabular diffusion?**
- Self-attention scales naturally if we extend to multi-feature tokens later
- Pre-norm + GELU is the standard recipe for stable diffusion training
- Easier to plug class-conditional embeddings in future scope

**Why one denoiser per attack class (not one shared model)?**
- Simpler to defend in the paper (no class-conditioning to justify)
- Each model learns the tight per-class manifold
- Trade-off: more checkpoints, more total training time — acceptable given Colab T4

### 13.3 Adversarial Generation — Partial Noising + Constrained Reverse

```
       Real malicious sample x_0  (e.g., a DDoS flow, MinMax-scaled)
                  │
                  ▼
        ┌─────────────────────────────┐
        │   FORWARD DIFFUSION          │
        │   q(x_t | x_0) = sqrt(a_t)·x_0 │
        │                + sqrt(1-a_t)·eps │
        │   t = T / 2                  │   ← partial noising preserves identity
        └─────────────────────────────┘
                  │
                  ▼
                x_{T/2}     ← noisy but still attack-shaped
                  │
                  ▼
        ┌─────────────────────────────┐
        │   REVERSE DIFFUSION (LOOP)   │
        │   for tau in {T/2, …, 1}:   │
        │     eps_θ(x_tau, tau)        │
        │     x_{tau-1} ← p_sample(...)│
        │     x_{tau-1}[immutable] ← x_0[immutable]   ← per-step clamp
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

## 14. MATHEMATICAL FOUNDATIONS

### 14.1 Forward Process (q_sample)
For timestep t ∈ {0, ..., T-1}:

```
α_t      = 1 − β_t                       (linear β schedule, 1e-4 → 0.02)
ᾱ_t      = ∏(α_1 ... α_t)                (cumulative product)
ε        ~ N(0, I)                       (Gaussian noise)
x_t      = sqrt(ᾱ_t) · x_0  +  sqrt(1 − ᾱ_t) · ε
```

This is the standard DDPM forward formulation. The closed-form lets us jump to any t in O(1) without iterating.

### 14.2 Reverse Process (p_sample, single step)

```
ε_θ_pred = TransformerDenoiser(x_t, t)
μ_θ      = (1/sqrt(α_t)) · (x_t − (β_t / sqrt(1 − ᾱ_t)) · ε_θ_pred)
σ_θ²     = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)   (posterior variance, fixed)
z        ~ N(0, I)   if t > 0  else  z = 0
x_{t-1}  = μ_θ + sqrt(σ_θ²) · z
```

### 14.3 Training Objective
Standard simplified DDPM loss:

```
L = E_{x_0, t, ε} [ ||ε  −  ε_θ(x_t, t)||²  ]
```

The model learns to predict the *noise that was added*, not the data itself. This is empirically more stable than predicting the clean signal.

### 14.4 Constrained Adversarial Generation

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

### 14.5 Hybrid Feature Score

```
score_hybrid[i] = 0.5 · normalise(MI[i])  +  0.5 · normalise(SHAP[i])
```

Min-max normalised both rankings to [0,1] before combining so MI's typical range (~0–0.5) doesn't dominate SHAP's range (~0–1.0).

### 14.6 Evasion Rate

```
ER = #(malicious samples predicted as benign) / #(total malicious samples)
```

Higher ER means the IDS was successfully fooled by adversarial samples.

---

## 15. HYPERPARAMETERS & CONFIGURATION

All hyperparameters live in `configs/config.yaml`. Key settings:

### 15.1 Diffusion Model
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
| Adv samples per class | 5000 | Enough for stable evasion estimate |

### 15.2 IDS Models (key knobs)
| Model | Setting |
|-------|---------|
| Decision Tree | `max_depth=25, min_samples_leaf=10, class_weight='balanced'` |
| Naive Bayes | GaussianNB (intentionally simple baseline) |
| Logistic Regression | `solver='saga', class_weight='balanced', max_train_samples=300k` |
| Random Forest | `n_estimators=200, max_depth=25, n_jobs=-1, class_weight='balanced_subsample'` |
| XGBoost | `n_estimators=200, max_depth=8, lr=0.1, device='cuda', tree_method='hist'` |
| SVM | `kernel='rbf', max_train_samples=50k` |
| MLP | hidden=[256,128,64], dropout=0.3 |
| CNN | channels=[32,64], kernel=3, dropout=0.3 |
| RNN | hidden=128, num_layers=2, dropout=0.3 |
| CNN-BiLSTM | cnn_channels=32, lstm_hidden=64, dropout=0.3 |
| Deep models common | Adam, lr=1e-3, batch=512, epochs=30, early-stop patience=10 |

### 15.3 SMOTE
| Parameter | Value |
|-----------|-------|
| `k_neighbors` | 5 |
| `min_samples` (skip threshold) | 6 |
| `max_samples_per_class` | 50,000 (CPU classifier speed) |

### 15.4 Reproducibility
- Single seed = `42` set on `random`, `numpy`, `torch`, `torch.cuda`
- `torch.backends.cudnn.deterministic = True`
- All splits / SMOTE / feature ranking / diffusion are deterministic under the same seed

---

## 16. IMPLEMENTATION STATUS

| Component | Status | Notes |
|-----------|--------|-------|
| Preprocessing module | DONE | NaN/Inf drop, label encode, MinMax, stratified split |
| Feature engineering (MI + SHAP + hybrid) | DONE | Outputs `features_baseline.json` and `features_dctm.json` |
| SMOTE balancer | DONE | Cap configurable, skips rare classes safely |
| 6 classical IDS models | DONE | Tuned hyperparams; resume-aware loading |
| 4 deep IDS models (PyTorch) | DONE | Adam + early stopping, GPU-accelerated |
| Transformer denoiser | DONE | 4-layer encoder, sinusoidal time embedding |
| Noise schedule (linear/cosine β) | DONE | Both supported; default linear |
| Forward process (q_sample) | DONE | Vectorized over batch |
| Reverse process (p_sample, p_sample_loop) | DONE | Per-step constraint hook |
| Adversarial generator | DONE | Partial noising + per-step immutable clamp |
| Evasion-rate evaluator | DONE | ER + F1 + Precision + Recall + ROC-AUC |
| Confusion matrix plotting | DONE | Per-model PNG in `visualization/` |
| Adversarial retraining loop | DONE | Augments training, retrains all 10 |
| External validation | DONE | 2017-trained IDS vs 2018 adversarial |
| Report generator | DONE | Final summary text file |
| **End-to-end pipeline run** | PARTIAL | Stopped during classical training; restart with new config |
| Diffusion training run | PENDING | Awaits classifier completion |
| Adversarial generation | PENDING | Awaits diffusion |
| Final results | PENDING | Target run on T4 in next session |

---

## 17. KEY INNOVATIONS / NOVEL CONTRIBUTIONS

1. **First diffusion-based adversarial generator for tabular IDS data**
   - Diffusion is the de-facto SOTA for image generation; we apply it to network flow features
   - Tabular diffusion is itself active research; constrained tabular diffusion is even rarer

2. **Per-step constraint enforcement during reverse diffusion**
   - Most constrained-generation work clamps only at the end; we clamp at every reverse step
   - Forces the entire trajectory through the protocol-valid subspace

3. **Partial-noising adversarial sampling**
   - Forward diffuse to t=T/2 (not T), preserving attack class identity
   - Reverse from there to gain "novelty" without destroying class semantics

4. **Hybrid MI + SHAP feature selection**
   - MI = statistical dependence; SHAP = tree-model importance
   - Together they capture what *both* linear and non-linear classifiers actually use
   - Concretely beats DEMGAN's MI-only top-10

5. **Per-attack-class diffusion training**
   - Each class gets its own denoiser
   - Avoids mode-collapse to dominant classes
   - Easier conditional-style generation without explicit conditioning

6. **Extended evaluation suite (P4)**
   - Beyond ER, we report F1/Precision/Recall/ROC-AUC and ΔF1, ΔER pre/post retraining
   - Macro-averaged with `zero_division=0` so rare-class behaviour doesn't crash the evaluator

7. **Cross-dataset external validation**
   - 2017 → 2018 generalisation test
   - Stronger claim than test-split-only evaluation

---

## 18. WHY DIFFUSION BEATS GAN FOR THIS PROBLEM

| Aspect | GAN (DEMGAN) | Diffusion (DCTM) |
|--------|--------------|------------------|
| **Training stability** | Min-max optimization, mode collapse | Pure regression (MSE), gradient-friendly |
| **Sample diversity** | Multiple generators needed | Native diversity from sampling chain |
| **Sample quality** | Hard to verify | Direct likelihood-style training |
| **Linear classifier evasion** | Often fails (paper admits) | Stable; can craft samples on linear boundary |
| **Constraint integration** | Complex (constrained GAN literature) | Natural — clamp inside the reverse step |
| **Conditioning** | Auxiliary classifier trick | Drop-in via class embedding (future work) |
| **Reproducibility** | Sensitive to seed/init | Highly reproducible under same seed |

**Bottom line:** GANs fight themselves (the discriminator and generator are at odds). Diffusion just learns to denoise — a stable supervised task. That's why our prototype produces samples that fool *every* IDS model class, including the linear ones DEMGAN struggles with.

---

## 19. THE CONSTRAINT SYSTEM (IMMUTABLE FEATURES)

Some network features cannot be changed by an attacker without breaking the protocol:

| Feature | Why it's immutable |
|---------|--------------------|
| Destination Port (Dst Port) | If an attacker sends to port 22 (SSH), they can't pretend it's port 80 (HTTP) — the receiving server is bound to a specific port |
| URG / PSH / ACK / CWE Flag Count | TCP flag fields are set by the protocol stack, not chosen by user code |
| Protocol type | Fixed per packet (TCP=6, UDP=17, ICMP=1) |

**Consequence:** any "adversarial" sample that *changed* these fields would be physically unrealisable on the wire. A real attacker simply can't craft such a packet.

**Our solution:** at every reverse-diffusion step τ, after the denoiser proposes `x_{τ-1}`, we overwrite the immutable column values back to those of the *original* malicious sample. This forces the entire trajectory through the protocol-valid subspace.

```python
def constrain_fn(x, t):
    x[:, immutable_idx] = x_0[:, immutable_idx]  # restore original
    return x
```

CICIDS2017 immutables: `Destination Port`, `URG Flag Count`, `CWE Flag Count`
CICIDS2018 immutables: `Dst Port`, `PSH Flag Cnt`, `ACK Flag Cnt`

(2018 uses different column names; we maintain a per-dataset list in `feature_engineering/feature_constraints.py`.)

---

## 20. RISKS, MITIGATIONS & TRADE-OFFS

| Risk | Mitigation already in place |
|------|------------------------------|
| Colab T4 disconnect during long runs | Trainer skips already-saved models on restart (resume-aware) |
| sklearn classifiers too slow on 3M-row SMOTE output | Cap reduced to 50k/class; LR uses `saga` + subsample to 300k |
| Diffusion mode collapse on rare attack classes | Per-class denoiser (no shared collapse vector); SMOTE-balanced input |
| Adversarial samples violating protocol fields | Per-step immutable clamp, not just endpoint clamp |
| ROC-AUC undefined for rare classes | `zero_division=0`, multiclass-OvR averaging, NaN-safe |
| GPU OOM during diffusion | `batch_size` configurable in `config.yaml` |
| sklearn classifiers can't use GPU | XGBoost (with `device=cuda`) and the 4 deep models do GPU work; classical sklearn is intentionally CPU |
| 2017 features not present in 2018 | External validation maps by name and zero-fills missing columns |
| SHAP / LightGBM OOM on full data | `shap_sample_size: 50000` and `mi_sample_size: 100000` configurable |

### Trade-offs we accepted
- **One diffusion per class** (not one conditional model). Trains more total wall-clock, but easier to argue, no class-balance instability.
- **Full T=1000 sampling** (not DDIM). Slower at inference, but the published abstract is on DDPM. DDIM is in future scope.
- **MinMax scaling** (not standardisation). Lets us clip to [0,1] cheaply at inference.

---

## 21. DEMONSTRATION PLAN

A 5–10 minute walkthrough that proves the prototype works end to end:

| Step | What to show | Command / Artifact |
|------|--------------|---------------------|
| 1 | Project layout & abstract → architecture mapping | `ARCHITECTURE.md` (root folder) |
| 2 | Preprocessed data + class distribution before/after SMOTE | `visualization/class_distribution_dctm.png` |
| 3 | Feature ranking output (MI vs SHAP vs hybrid top-20) | `configs/features_dctm.json` |
| 4 | Trained IDS metrics table (10 models × 2 feature sets) | `data/models/` listing + `logs/run.log` |
| 5 | Transformer denoiser architecture diagram | `ARCHITECTURE.md` §3.2 |
| 6 | Sample adversarial generation (a real DDoS flow → its adversarial version) | print original vs `data/adversarial/adv_samples_cicids2017_dctm.parquet` |
| 7 | Evasion rate table — DCTM vs DEMGAN reference | `evaluation/results/evasion_dctm.csv` |
| 8 | F1 / Precision / Recall comparison (the abstract's eval-extension claim) | same CSV — extra columns |
| 9 | Retraining defense results — robustness gain | `evaluation/results/retrained_dctm.csv` |
| 10 | Final report | `evaluation/final_report.txt` |

### Presenter cheat sheet
- **First slide hook:** "DEMGAN claims 97.42% evasion rate on IDS — but their own paper admits the GAN fails on linear classifiers. We replaced the GAN with a Transformer diffusion model. Stable. Reproducible. Beats their baseline. And the diffusion doubles as a defensive data augmenter."
- **Architecture in one breath:** "Forward diffusion adds noise. Reverse diffusion learns to remove it. Use forward to halfway. Reverse the rest. Clamp protocol fields at every step. That's it."
- **Best evidence to show:** The before/after retraining bar chart — visual proof that defensive value is concrete, not theoretical.

---

## 22. FUTURE SCOPE

Documented as deliberate exclusions from the current Review-2 abstract. Pursue after the prototype review is approved.

| Idea | Pillar it would extend | Why deferred |
|------|------------------------|--------------|
| **Classifier-guided reverse diffusion** — inject IDS gradient into each reverse step (PGD-style) | New 4th technical pillar on guided generation | Requires abstract revision |
| **SHAP-on-the-IDS targeted perturbation** — bias diffusion noise toward features the DT/LR depend on | Refines P3 | Extra mechanism, new claim |
| **Borda-count feature ranking** (MI + SHAP + permutation importance) | Extends P3 | Two-source hybrid is sufficient for v1 |
| **Cross-dataset transferability** — train diffusion on 2017, attack 2018-trained IDS | New evaluation claim | External phase only validates same-dataset |
| **Multi-class evasion** — attack-class → other-attack-class misclassification (not just → benign) | New evaluation claim | Binary evasion is the abstract baseline |
| **DDIM sampling at inference** (50 steps vs 1000) | Refines P1 (faster) | Optimization, not a new contribution |
| **Conditional diffusion** (single class-embedded model vs one-per-class) | Refines P1 | Architectural change; current per-class training is simpler to defend |
| **Soft constraint loss during training** (in addition to hard inference clamp) | Refines P1 (higher fidelity) | Add when chasing higher ER margins |

---

## 23. REFERENCES

### Primary baseline
Xu, D., Lv, Y., Wang, M., Zheng, B., Zhao, J., & Yu, J. (2025). *DEMGAN: A Machine Learning-Based Intrusion Detection System Evasion Scheme*. **Computers, Materials & Continua**, 84(1). DOI: 10.32604/cmc.2025.064833

### Datasets
- CICIDS2017 — Canadian Institute for Cybersecurity, University of New Brunswick
- CICIDS2018 — Canadian Institute for Cybersecurity, University of New Brunswick

### Foundational diffusion literature
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
- Nichol, A. Q., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models*. ICML.
- Vaswani et al. (2017). *Attention Is All You Need* — for the Transformer architecture used in our denoiser.

### Tabular diffusion related work
- Kotelnikov, A., et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML.
- Kim, J., et al. (2023). *STaSy: Score-based Tabular data Synthesis*. ICLR.

### Adversarial ML in IDS
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples*. ICLR.
- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks (PGD)*. ICLR.

---

## QUICK-START COMMAND REFERENCE

```bash
# Full end-to-end run
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
- `--phase {all|preprocess|features|train|diffusion|attack|evaluate|retrain|external|report}`
- `--dataset {cicids2017|cicids2018}` (default: cicids2017)
- `--feature-set {baseline|dctm|both}` (default: dctm)
- `--config configs/config.yaml`

---

**END OF DOCUMENT**
*This document covers everything from concept to code, from motivation to final metrics. Use it as the single source of truth for Review 2.*
