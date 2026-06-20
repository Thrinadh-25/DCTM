# DCTM Documentation — Diagram & Figure Guide

This file tells you exactly **which figures to insert where**, what each should contain, and where the source file lives (Drive or local).

---

## Fig. 3.1 — Overall System Architecture of DCTM

**Location in doc:** Chapter 3, Section 3.3 (right after the first paragraph)  
**Source:** Draw this diagram yourself (no existing file). See construction below.

### What to draw:
A top-to-bottom flowchart with 6 boxes connected by arrows:

```
[Raw CICIDS2017 CSVs]
        ↓
[Module 1: Data Preprocessing]
  • Merge 8 daily CSVs
  • Remove duplicates, clean infinities
  • Binary label mapping (0=Benign, 1=Attack)
  • 80/20 stratified train-test split
        ↓
[Module 2: Adaptive Feature Selection]
  • Mutual Information scoring (100k subsample)
  • SHAP via LightGBM proxy (50k subsample)
  • Score = 0.5×MI_norm + 0.5×SHAP_norm
  • Top 20 features → DCTM feature set
  • SMOTE balancing on training split
        ↓
[Module 3: IDS Model Training (×10 models)]
  • Classical: DT, NB, LR, RF, XGBoost, SVM
  • Deep Learning: MLP, CNN, RNN, CNN-BiLSTM
        ↓
[Module 4: Diffusion Model Training (per attack class)]
  • Transformer Denoiser (4×TransformerEncoderLayer)
  • Linear noise schedule, T=1000 timesteps
  • MSE noise prediction loss
        ↓
[Module 5: Adversarial Sample Generation]
  • Partial forward diffusion → t_mid = 500
  • Full reverse denoising → t = 0
  • Immutable feature clamping after each step
        ↓
[Module 6: IDS Evaluation & Adversarial Retraining]
  • Measure Evasion Rate (ER) on adversarial samples
  • Augment training set with adversarial samples
  • Retrain all 10 IDS models
  • Re-evaluate adversarial robustness
```

**Recommended tool:** PowerPoint SmartArt or draw.io  
**Suggested size:** Full page width, portrait orientation

---

## Fig. 3.2 — Diffusion Forward and Reverse Process

**Location in doc:** Chapter 3, Section 3.4.4 (after the TransformerDenoiser paragraph)  
**Source:** Draw this diagram (standard DDPM illustration).

### What to draw:
Two horizontal rows of circles connected by arrows:

**Top row (Forward process — q):**
```
x₀ →[+ε₁]→ x₁ →[+ε₂]→ x₂ → ... → x₅₀₀ → ... → x₁₀₀₀ (pure noise)
```
Label above: **"Forward Process q(x_t | x_{t-1}): Gradual noise addition"**

**Bottom row (Reverse process — p_θ):**
```
x₁₀₀₀ →[Denoiser]→ ... → x₅₀₀ ←[Partial forward starts here] ... → x₀ (adversarial sample)
```
Label below: **"Reverse Process p_θ(x_{t-1} | x_t): Learned denoising (TransformerDenoiser)"**

**Annotation:** Mark t_mid = T/2 = 500 with a vertical dashed line and label "Adversarial generation starts here — partial forward from real malicious sample x₀"

**Note:** Immutable features are clamped after each reverse step (add a small annotation box on the reverse row).

---

## Fig. 4.1 — Class Distribution Before and After SMOTE

**Location in doc:** Chapter 4, Section 4.4 (after the SMOTE paragraph)  
**Source files (Google Drive → DCTM/visualization/):**
- `class_distribution_baseline.png` — Before SMOTE
- `class_distribution_dctm.png` — After SMOTE

### How to insert:
Place both images side by side (two-column layout in Word):
- Left image: `class_distribution_baseline.png` — caption "Before SMOTE"
- Right image: `class_distribution_dctm.png` — caption "After SMOTE"
- Combined caption below: **Fig. 4.1: Class Distribution Before and After SMOTE (CICIDS2017)**

**In Word:** Insert → Table (1 row × 2 columns, no borders) → insert one image per cell → add combined caption below the table.

---

## Fig. 4.2 — Mutual Information Feature Ranking

**Location in doc:** Chapter 4, Section 4.3 (after the feature selection paragraph)  
**Source file (local):**  
`mini project/cleaned/feature_selection/mi_feature_ranking.png`

### How to insert:
- Insert the PNG directly
- Caption below: **Fig. 4.2: Mutual Information Feature Ranking — Top 20 Features**
- Size: Scale to ~80% of page width, centered

---

## Fig. 5.1 — Adversarial Evasion Rate per IDS Model

**Location in doc:** Chapter 5, Section 5.2 (after Table 5.2 and the following paragraph)  
**Source file (Google Drive → DCTM/visualization/):**  
`evasion_comparison.png`

### What the chart shows:
- Horizontal bar chart of evasion rate for each of the 10 IDS models
- A horizontal reference line at 0.9742 (DEMGAN baseline)
- Models on Y-axis, Evasion Rate (0.0 to 1.0) on X-axis

### How to insert:
- Insert the PNG directly
- Caption below: **Fig. 5.1: Adversarial Evasion Rate per IDS Model — DEMGAN reference shown at 0.9742**
- Size: Full text width

---

## Fig. 5.2 — Retraining Impact: Evasion Rate Before vs. After

**Location in doc:** Chapter 5, Section 5.3 (after Table 5.3 and the following paragraph)  
**Source file (Google Drive → DCTM/visualization/):**  
`retraining_improvement.png`

### What the chart shows:
- Grouped bar chart: for each model, two bars — "Before Retraining" (darker) and "After Retraining" (lighter)
- Y-axis: Evasion Rate (0.0 to 1.0)
- X-axis: 10 IDS model names

### How to insert:
- Insert the PNG directly
- Caption below: **Fig. 5.2: Impact of Adversarial Retraining on Evasion Rate (Before vs. After) — all 10 models**
- Size: Full text width

---

## Figs. 5.3–5.6 — Confusion Matrices (Post-Retraining)

**Location in doc:** Chapter 5, Section 5.3 (after the paragraph referencing Figures 5.3–5.6)  
**Source files (Google Drive → DCTM/visualization/confusion_matrices/):**

| Figure | File | Caption |
|--------|------|---------|
| Fig. 5.3 | `cm_random_forest_dctm_retrained.png` | Confusion Matrix — Random Forest (Post-Retraining) |
| Fig. 5.4 | `cm_xgboost_dctm_retrained.png` | Confusion Matrix — XGBoost (Post-Retraining) |
| Fig. 5.5 | `cm_decision_tree_dctm_retrained.png` | Confusion Matrix — Decision Tree (Post-Retraining) |
| Fig. 5.6 | `cm_cnn_bilstm_dctm_retrained.png` | Confusion Matrix — CNN-BiLSTM (Post-Retraining) |

### How to insert:
- 2×2 grid (Word table with no borders, 2 rows × 2 columns)
- Each cell: one confusion matrix image + its individual caption
- Combined note: Figs. 5.3–5.6 show the four strongest post-retraining models

---

## All Available Drive Files (for reference)

**Drive path: DCTM/visualization/**
- `evasion_comparison.png` → Use for Fig. 5.1
- `retraining_improvement.png` → Use for Fig. 5.2
- `class_distribution_baseline.png` → Use for Fig. 4.1 (left)
- `class_distribution_dctm.png` → Use for Fig. 4.1 (right)

**Drive path: DCTM/visualization/confusion_matrices/**
- `cm_decision_tree_dctm_retrained.png` → Fig. 5.5
- `cm_naive_bayes_dctm_retrained.png` → (optional, can add as appendix)
- `cm_logistic_regression_dctm_retrained.png` → (optional)
- `cm_random_forest_dctm_retrained.png` → Fig. 5.3
- `cm_xgboost_dctm_retrained.png` → Fig. 5.4
- `cm_svm_dctm_retrained.png` → (optional)
- `cm_mlp_dctm_retrained.png` → (optional)
- `cm_cnn_dctm_retrained.png` → (optional)
- `cm_rnn_dctm_retrained.png` → (optional)
- `cm_cnn_bilstm_dctm_retrained.png` → Fig. 5.6

**Local file: mini project/cleaned/feature_selection/**
- `mi_feature_ranking.png` → Fig. 4.2

---

## Diagrams You Need to Draw (no existing file)
1. **Fig. 3.1** — System architecture flowchart (draw in PowerPoint or draw.io)
2. **Fig. 3.2** — DDPM forward/reverse process illustration (draw in PowerPoint or draw.io)

Both are standard diagrams. See descriptions above for exact content. Estimated time: 20–30 minutes each.

---

## Quick Checklist

- [ ] Fig. 3.1: Draw system architecture → insert in Chapter 3.3
- [ ] Fig. 3.2: Draw DDPM diagram → insert in Chapter 3.4.4
- [ ] Fig. 4.1: Download 2 PNGs from Drive → insert side-by-side in Chapter 4.4
- [ ] Fig. 4.2: Copy mi_feature_ranking.png from local folder → insert in Chapter 4.3
- [ ] Fig. 5.1: Download evasion_comparison.png from Drive → insert in Chapter 5.2
- [ ] Fig. 5.2: Download retraining_improvement.png from Drive → insert in Chapter 5.3
- [ ] Figs. 5.3–5.6: Download 4 confusion matrix PNGs from Drive → insert 2×2 grid in Chapter 5.3
