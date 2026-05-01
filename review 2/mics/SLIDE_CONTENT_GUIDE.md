# DCTM — Slide Content Guide
# CSE-C-06 | Review 2

---

## GLOBAL RULES
- Header every slide: `CSE-C-06          DCTM: Diffusion-Based Cyber Threat Modelling`
- Footer every slide: page number (right side)
- Font: Times New Roman or Arial, minimum size 20
- Abstract: paragraph format, no bullets
- All other slides: bullet points, one liner each, max 5 points
- Single-side layout only

---

## SLIDE 1 — TITLE SLIDE

```
KESHAV MEMORIAL ENGINEERING COLLEGE
Department of Computer Science & Engineering (CSE-C)

Mini Project Presentation on

DCTM: Diffusion-Based Cyber Threat Modelling
for Improved Intrusion Detection Performance

Kasturi Vishnu Vardhan     — 245523733160
Kukunuru Thrinadh Reddy    — 245523733163
Neha Sri Tirunagari        — 245523733187

Guide: Ms. Duddeda Aishwarya (Asst. Professor)
```

---

## SLIDE 2 — INDEX

```
1. Abstract
2. Introduction
3. Problem Statement
4. Workflow
5. Architecture
6. Methodology
7. Dataset
8. References
```

---

## SLIDE 3 — ABSTRACT

> No bullets. Two short paragraphs. Keywords at end.

ML-based Intrusion Detection Systems are highly vulnerable to adversarial traffic. DEMGAN achieves 97.42% evasion rate using Wasserstein GANs but is limited by a 10-feature set, GAN instability, and class imbalance.

DCTM replaces the GAN with a Transformer-based Diffusion Model, integrates SMOTE for class balancing, and expands features to 20 using hybrid MI + SHAP selection. Evaluation extends to F1, Precision, and Recall on CICIDS2017 and CICIDS2018.

**Keywords:** Adversarial Examples, IDS, Tabular Diffusion, Transformer, SMOTE, Feature Selection, Evasion Attack

---

## SLIDE 4 — INTRODUCTION

- ML-based IDS classifies network traffic as Benign or Malicious.
- Adversarial examples are crafted inputs that fool the classifier into wrong predictions.
- An evasion attack makes malicious traffic appear benign to the IDS.
- DEMGAN is the current state-of-the-art adversarial generator using Wasserstein GAN.
- DCTM improves on DEMGAN using Diffusion Models — stable, consistent, and comprehensive.

**Diagram:** IDS evasion concept block (Attacker → Adversarial flow → IDS → "Benign" ✗)

---

## SLIDE 5 — PROBLEM STATEMENT

- DEMGAN uses only 10 features (Mutual Information only) — limits adversarial coverage.
- GAN training is unstable — fails on linear classifiers (DT ER: 61.15%, LR ER: 48.94%).
- CICIDS2017 is severely imbalanced — Heartbleed has only 11 samples, SQL Injection has 21.
- DEMGAN reports Evasion Rate only — no F1, Precision, or Recall.
- DCTM proposes four pillars: Diffusion (P1), SMOTE (P2), Hybrid Features (P3), Extended Metrics (P4).

---

## SLIDE 6 — WORKFLOW

- Raw CSVs → clean NaN/Inf → MinMax normalise → 80/20 train-test split.
- MI + SHAP hybrid ranking selects top 10 (baseline) and top 20 (DCTM) features.
- SMOTE balances train split (cap 50k/class) → train 10 IDS models (20 checkpoints total).
- Transformer Denoiser trained per attack class → generates adversarial samples at t=T/2.
- Adversarial samples evaluated on all 10 IDS → ER, F1, Precision, Recall computed.

**Diagram:** ARCH_OVERVIEW diagram (full pipeline, one page)

---

## SLIDE 7 — ARCHITECTURE

- 10 IDS classifiers: Decision Tree, Naive Bayes, Logistic Regression, Random Forest, XGBoost, SVM, MLP, CNN, RNN, CNN-BiLSTM.
- Each model trained on both 10-feature and 20-feature sets — 20 checkpoints total.
- Transformer Denoiser: one model per attack class, predicts noise ε given noisy input x_t and timestep t.
- Architecture: Linear projection → 4× Transformer Encoder Layers → output noise vector.
- Diffusion internals are under active study — detailed explanation in Review 3.

---

## SLIDE 8 — METHODOLOGY

- Feature selection: 0.5×MI_norm + 0.5×SHAP_norm → top 20 hybrid features.
- Forward diffusion adds noise to attack sample x₀ up to timestep t = T/2 = 500.
- Reverse denoising runs from T/2 back to 0 — immutable features clamped every step.
- Immutable features (cannot be changed): Destination Port, URG Flag Count, CWE Flag Count.
- SMOTE applied on train split only — test kept naturally imbalanced for fair evaluation.

---

## SLIDE 9 — DATASET

- CICIDS2017: 8 CSV files, ~885 MB, ~2.83M flows, 15 classes (primary benchmark).
- CICIDS2018: 3 CSV files, ~1.1 GB, ~2.4M flows (external generalisation test only).
- Preprocessing: remove NaN/Inf, MinMax normalise, stratified 80/20 split.
- Class imbalance: Benign = 2.27M, Heartbleed = 11 — SMOTE fixes this on train data.
- Test split is never touched by SMOTE or any augmentation.

**Diagram:** Before vs After SMOTE bar chart (class distribution)

---

## SLIDE 10 — REFERENCES

```
[1] Xu et al., "DEMGAN," CMC Vol.84, 2025. DOI: 10.32604/cmc.2025.064833
[2] Ho et al., "DDPM," NeurIPS, 2020.
[3] Vaswani et al., "Attention Is All You Need," NeurIPS, 2017.
[4] Arjovsky et al., "Wasserstein GAN," ICML, 2017.
[5] Lundberg & Lee, "SHAP," NeurIPS, 2017.
[6] Chawla et al., "SMOTE," JAIR, 2002.
[7] Sharafaldin et al., "CICIDS2017 & 2018 Datasets," ICISSP, 2018.
```

---

## SLIDE 11 — THANK YOU

```
         Thank You

      Any Questions?
```

---

## DIAGRAM SUMMARY — ONLY 3 SLIDES NEED A DIAGRAM

| Slide | Diagram |
|---|---|
| Introduction | Simple block: Attacker → IDS → "Benign" (evasion concept) |
| Workflow | ARCH_OVERVIEW.md diagram (full pipeline, one page) |
| Dataset | Grouped bar chart: class counts before vs after SMOTE |
