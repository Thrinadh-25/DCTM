# DCTM — Full Results & Methodology Analysis

**Generated:** 2026-06-02
**Run analyzed:** `dataset=cicids2017`, `feature_set=dctm`
**Data sources:** Google Drive (`evaluation/results/*.csv`, `configs/features_*.json`, `diffusion_index_dctm.json`, confusion-matrix PNGs) + local repo (`configs/config.yaml`, `diffusion/`, `evaluation/`).

> **TL;DR.** The clean detectors (RF/XGBoost) are genuinely strong; the attack is genuinely
> destructive; adversarial retraining genuinely fixes the tree models. **But three things
> inflate the story and need fixing before this is defensible:** (1) the "diffusion" attack
> actually ends with a **PGD step that optimizes samples toward *benign* using DT/LR/NB as
> surrogates** — so the evasion is a *diffusion + transfer-PGD* result, not pure diffusion;
> (2) for SVM/MLP/CNN/RNN/CNN-BiLSTM the post-retrain `ER=0` is **degenerate** (they stop
> saying "benign" but still misclassify 75–89% of attacks); (3) the **constraint pillar is
> effectively inactive** — only 1 of the 20 DCTM features is immutable.

---

## 1. Pipeline & configuration (what actually ran)

| Stage | Setting (from `config.yaml`) |
|---|---|
| Datasets | CICIDS2017 (8 CSVs) + CICIDS2018 (subset). This run = **CICIDS2017**. |
| Classes | **15 total** (0 = benign, 1–14 = attacks). Confirmed by ROC-AUC warnings (`15 columns`). |
| Train balancing | SMOTE, capped `max_samples_per_class = 50000` |
| Test split | 20% (`test_size = 0.2`, `seed = 42`) |
| Feature sets | baseline = top-10 MI; **dctm = top-20 hybrid (0.5·MI + 0.5·SHAP)** |
| Diffusion | Transformer denoiser, `T=1000`, linear β, `model_dim=256`, 4 layers, `epochs=100`, per-class |
| Attack | partial forward to `t=T/2` → reverse denoise (immutable clamp each step) → **PGD refine `guidance_steps=10`, `lr=0.05`** |
| Adv samples | `adv_samples_per_class = 5000` |
| Metrics | accuracy + **macro** precision/recall/F1, OVR macro ROC-AUC; `ER = #(malicious→benign)/#malicious` |

**Diffusion coverage (`diffusion_index_dctm.json`):** trained for attack classes
`{1,2,3,4,5,6,7,10,11,12,14}` — **11 of 14 attack classes. Classes 8, 9, 13 are missing**
(too few samples to train a per-class denoiser). So evasion is measured over 11/14 attack types only.

---

## 2. Features actually in use

### DCTM — top-20 hybrid (`features_dctm.json`)
Ranked by `0.5·MI_norm + 0.5·SHAP_norm`:

| # | Feature | hybrid | MI_norm | SHAP_norm | Immutable? |
|---|---|---|---|---|---|
| 1 | Flow IAT Min | 0.623 | 0.247 | **1.000** | |
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

**Observations**
- The hybrid is **MI-dominated**. Except for `Flow IAT Min` (SHAP=1.0, MI=0.25), SHAP_norm is near-zero for almost every selected feature, so `0.5·SHAP` contributes little beyond rank-1. The "hybrid" is, in practice, ≈ MI ranking + one SHAP feature.
- **Only `Destination Port` (rank 10) is immutable.** The other CICIDS2017 immutables — `URG Flag Count`, `CWE Flag Count` — rank far down the MI list and are **not selected**, so they never enter the model or the constraint.

### Baseline — top-10 MI (`features_baseline.json`)
`Average Packet Size, Packet Length Mean, Packet Length Std, Packet Length Variance,
Total Length of Bwd Packets, Subflow Bwd Bytes, Avg Bwd Segment Size, Bwd Packet Length Mean,
Total Length of Fwd Packets, Subflow Fwd Bytes` — **0 immutable features**. All are
packet-size / byte-volume statistics (heavily correlated with each other).

---

## 3. Clean training results (`baseline_dctm.csv`)

Macro-averaged, on the 15-class clean test set.

| Model | Accuracy | Precision | Recall | **Macro-F1** | ROC-AUC | Train (s) |
|---|---|---|---|---|---|---|
| decision_tree | 0.9911 | 0.684 | 0.951 | 0.747 | 0.984 | 13.9 |
| naive_bayes | 0.390 | 0.319 | 0.635 | 0.293 | 0.946 | 0.3 |
| logistic_regression | 0.410 | 0.236 | 0.701 | 0.247 | 0.956 | 223.7 |
| random_forest | 0.9946 | 0.759 | 0.918 | 0.805 | 0.9998 | 340.1 |
| **xgboost** | **0.9968** | 0.752 | 0.930 | **0.807** | 0.9998 | 20.1 |
| svm | 0.562 | 0.334 | 0.768 | 0.348 | 0.973 | 8.3 |
| mlp | 0.887 | 0.389 | 0.930 | 0.461 | 0.998 | 230.4 |
| cnn | 0.873 | 0.398 | 0.913 | 0.466 | 0.997 | 251.4 |
| rnn | 0.946 | 0.529 | 0.907 | 0.592 | 0.997 | 405.1 |
| cnn_bilstm | 0.944 | 0.518 | 0.936 | 0.580 | 0.999 | 311.9 |

**Reading it**
- **Big accuracy↔macro-F1 gap** (e.g. XGB acc 0.997 vs F1 0.807). This is the **rare-class effect of macro-averaging**, *not* overfitting: the models nail the high-frequency classes (benign + large attacks) but score poorly on small classes, which macro-F1 weights equally. ROC-AUC ≈ 1.0 confirms ranking is excellent.
- **NB, LR, SVM are underfit** for 15-class (F1 0.25–0.35). SVM is capped at 50k training rows; LR/NB are simply too weak for this many classes. Their downstream attack numbers are therefore the least informative.
- **Caveat on the 0.99s:** CICIDS2017 is a known "easy"/partially-leaky dataset (near-duplicate flows, label artifacts). RF/XGB at 0.99 accuracy is *typical* for this dataset and partly reflects dataset easiness — not proof of a robust detector. Cross-dataset validation (the `external` phase on CICIDS2018) is the real test.

---

## 4. Attack results — clean vs adversarial (`evasion_dctm.csv`)

Pre-retraining. `ER` = fraction of malicious samples pushed to **benign**.

| Model | clean F1 | clean ER | **adv F1** | **adv acc** | **adv ER** | Role |
|---|---|---|---|---|---|---|
| decision_tree | 0.747 | 0.0005 | 0.038 | 0.031 | **0.407** | PGD surrogate |
| naive_bayes | 0.293 | 0.0007 | 0.015 | 0.011 | **0.936** | PGD surrogate |
| logistic_regression | 0.247 | 0.0075 | 0.035 | 0.069 | **0.127** | PGD surrogate |
| random_forest | 0.805 | 0.0002 | 0.001 | 0.0004 | **0.611** | held-out |
| xgboost | 0.807 | 0.0002 | 0.018 | 0.013 | **0.599** | held-out |
| **svm** | 0.348 | 0.0018 | 0.000 | 0.000 | **1.000** | held-out |
| mlp | 0.461 | 0.0015 | 0.018 | 0.029 | **0.592** | held-out |
| cnn | 0.466 | 0.0065 | 0.019 | 0.028 | **0.527** | held-out |
| rnn | 0.592 | 0.0009 | 0.012 | 0.018 | **0.494** | held-out |
| cnn_bilstm | 0.580 | 0.0014 | 0.015 | 0.054 | **0.150** | held-out |
| **Average** | | | | | **0.5441** | |

**Two effects are tangled together here:**

1. **Adversarial *accuracy* collapses to ≈0 everywhere** (RF 0.04%, XGB 1.3%, DT 3%). The attack is brutally effective at *breaking* classification. But the **ER (→benign) is only 0.40–0.61 for the strong models**, meaning **40–60% of attacks evade to benign and the rest scatter into the *wrong attack class***. So the headline metric (ER) *under*-reports total disruption, while "evasion" overstates how much is true benign-evasion vs generic confusion.

2. **This is a transfer attack, and transfer is uneven.** Surrogates = {DT, LR, NB}. ER ranges from **0.127 (LR)** to **1.000 (SVM)**. The `avg = 0.5441` masks a 0.13–1.0 spread. Notably the attack transfers *strongly* to held-out RF/XGB/MLP (~0.6) and *weakly* to LR (0.13) and CNN-BiLSTM (0.15).

---

## 5. Adversarial retraining results (`retrained_dctm.csv`)

Models retrained with adversarial samples added, then re-evaluated.

| Model | clean F1 (Δ vs base) | clean ER (Δ) | adv F1 | **adv acc** | **adv ER** (was) |
|---|---|---|---|---|---|
| decision_tree | 0.757 (+0.010) | 0.0005 (≈0) | 0.507 | 0.644 | **0.002** (0.407) |
| naive_bayes | 0.188 (−0.105) | **0.392 (+0.391)** ⚠ | 0.124 | 0.149 | **0.000** (0.936) |
| logistic_regression | 0.214 (−0.033) | **0.080 (+0.072)** ⚠ | 0.115 | 0.148 | **0.006** (0.127) |
| random_forest | 0.820 (+0.015) | 0.0002 | **0.956** | **0.955** | **0.000** (0.611) |
| xgboost | 0.825 (+0.018) | 0.0001 | **0.844** | **0.844** | **0.000** (0.599) |
| svm | 0.322 (−0.026) | **0.143 (+0.141)** ⚠ | 0.058 | 0.109 | **0.000** (1.000) |
| mlp | 0.470 (+0.008) | 0.002 | 0.177 | 0.254 | **0.000** (0.592) |
| cnn | 0.475 (+0.009) | 0.006 | 0.059 | 0.127 | **0.000** (0.527) |
| rnn | 0.555 (−0.038) | 0.0005 | 0.060 | 0.131 | **0.000** (0.494) |
| cnn_bilstm | 0.560 (−0.020) | 0.001 | 0.071 | 0.130 | **0.000** (0.150) |

**This table is the heart of the "true vs overfitting" question:**

- **Genuine robustness — RF & XGBoost.** `adv_ER → 0` **and** `adv_acc 0.84–0.96`, `adv_f1 0.84–0.96`, with clean F1 *unchanged/slightly up*. They learned to correctly re-identify adversarial samples to their true class. **This is real and is your strongest defensive result.**

- **Degenerate "robustness" — SVM, MLP, CNN, RNN, CNN-BiLSTM.** `adv_ER → 0` but **`adv_acc` only 0.11–0.25**. They stopped predicting *benign* on adversarial inputs, but still misclassify **75–89%** of attacks (into wrong attack classes). **`ER=0` here is a metric artifact** — the models learned "don't say benign," not "identify the attack." Reporting only ER overstates this as a fixed vulnerability.

- **Retraining *damaged* the weak models on clean data** ⚠:
  - **naive_bayes clean ER 0.0007 → 0.392** (and clean F1 dropped). Retraining made NB miss ~39% of *real* attacks.
  - **svm clean ER 0.0018 → 0.143**, **logistic_regression 0.0075 → 0.080**. Both got worse at clean detection.
  This is the classic **clean/robust trade-off and a sign of adversarial overfitting** in the low-capacity models.

---

## 6. Are the results "true" or overfitting? — verdict

| Claim | Verdict | Basis |
|---|---|---|
| Clean RF/XGB detectors are strong | **True** (with dataset caveat) | Test-set macro-F1 0.81, ROC-AUC 0.9998; but CICIDS2017 is an easy/leaky dataset → confirm on CICIDS2018. |
| acc↔F1 gap = overfitting | **No** | It's macro-averaging over rare classes; report per-class F1 to show this. |
| The diffusion attack evades IDS | **Partly / unattributed** | A **PGD step toward benign** runs after diffusion (`guidance_steps=10`). Evasion cannot be credited to diffusion without a `guidance_steps=0` ablation. |
| Adversarial retraining hardens the models | **True for RF/XGB; degenerate for SVM+deep** | RF/XGB get adv_acc 0.84–0.96; others get ER=0 but adv_acc 0.11–0.25. |
| Retraining is "free" | **No** | NB/LR/SVM **clean** ER rose (NB to 0.39); robustness bought at the cost of clean detection. |
| `ER=0` after retrain = robust | **Overstated** | For 5/10 models it means "avoids benign label," not "detects attack." |
| Robustness will generalize | **Untested → likely overfit** | Retraining used samples optimized against {DT,LR,NB}+this diffusion config. No cross-attack/cross-config evaluation. |

**Bottom line:** the *clean detection* and the *RF/XGB hardening* are real. The *attack’s
attribution to diffusion*, the *deep-model "robustness,"* and the *generalization of
retraining* are **not yet established** and partly artifactual.

---

## 7. Methodology issues found (ranked by impact)

1. **PGD contaminates the diffusion claim (highest).** `_pgd_refine()` in `diffusion/adversarial_generator.py` runs 10 finite-difference gradient-ascent steps toward `P(benign)` using `{decision_tree, logistic_regression, naive_bayes}`. You cannot claim "diffusion evades IDS" while a transfer-PGD attack is doing unknown share of the work.
2. **Surrogates are also evaluated as victims.** DT/LR/NB are both attack surrogates *and* reported IDS models → white-box leakage into the average. The honest "black-box" number is RF/XGB/deep only.
3. **The constraint pillar (P3) is effectively inactive.** Only 1/20 DCTM features (Destination Port) is immutable; 0/10 baseline. The flag-count immutables aren't even selected. The per-step clamp constrains almost nothing.
4. **Degenerate ER after retrain.** ER=0 with adv_acc≈0.12 is reported as robustness. Always report adv_acc + adv_f1 next to ER.
5. **Clean-performance regression after retraining** (NB/LR/SVM) is not surfaced in the final report.
6. **Incomplete attack coverage.** Classes 8, 9, 13 have no diffusion model; evasion covers 11/14 attacks.
7. **DEMGAN 97.42% comparison is apples-to-oranges.** Your 0.5441 is a 10-model *average* over a 15-class task; DEMGAN's figure is (almost certainly) a best/binary number. Best-vs-best, your SVM = 1.0 already exceeds it — but that's a weak, underfit model, so it's not a flattering comparison either.
8. **Finite-difference PGD on tree/NB surrogates is mostly zero-gradient** (piecewise-constant `predict_proba`), so the refinement signal is dominated by **LR**. Effectively an LR-guided attack with DT/NB contributing little.
9. **SVM evaluated on a 20k subsample** (`max_eval_samples`) while others use the full test set → ER not measured on identical data.
10. **Hybrid ≈ MI.** SHAP_norm is ~0 for nearly all chosen features; the SHAP half barely changes the ranking. The "two-source hybrid" novelty is thin as-is.

---

## 8. Recommended changes (concrete, ordered)

**To make the attack claim defensible**
1. **Ablation:** run `guidance_steps ∈ {0, 5, 10, 20}` and report ER for each. `0` = pure diffusion → this isolates the actual contribution of the diffusion model. *This is the single most important experiment.*
2. **Strict black-box protocol:** optimize PGD only on held-out surrogates and report transfer to a disjoint victim set. Never average surrogate + victim ERs together.
3. Switch deep-model attack to **true gradient PGD** (they're differentiable) instead of finite-difference on trees, or drop trees from the surrogate ensemble and state that PGD is LR-guided.

**To make the defense claim defensible**
4. **Report `adv_acc` and `adv_f1` beside ER** everywhere, plus **per-class F1**, to expose the degenerate ER=0 cases.
5. **Report clean-performance deltas after retraining** (NB/LR/SVM regressions) — don't hide the trade-off.
6. **Cross-attack generalization test:** retrain on attack-config A, evaluate robustness on config B (different `partial_t`, different surrogate). If ER stays ~0, robustness is real; if it jumps, it was adversarial overfitting.

**To make the contributions real**
7. **Fix the constraint pillar:** force-include the immutable features into the DCTM set, or expand the immutable list to all semantically-fixed fields (ports, protocol, flag counts) and re-rank. Then show the clamp actually changes generated samples.
8. **Complete coverage:** lower `min_samples` / oversample so classes 8, 9, 13 get diffusion models (11/14 → 14/14).
9. **Strengthen the hybrid:** either add a third signal (permutation importance / Borda) or re-weight so SHAP actually moves the ranking; otherwise call it "MI + SHAP tie-break," not a hybrid.

**To address dataset easiness / overfitting**
10. **Cross-dataset eval:** train on CICIDS2017, attack/evaluate on CICIDS2018 (the `external` phase) — the real test that 0.99 clean accuracy isn't dataset leakage.
11. Consider **flow de-duplication** on CICIDS2017 before splitting.

**Optimization (optional)**
12. **DDIM (≈50 steps)** to cut generation time vs 1000-step ancestral sampling; report fidelity via the existing `_log_drift` Δmean/Δstd so you can argue the samples remain realistic.

---

## 9. Data inventory (Google Drive)

| Artifact | File | Notes |
|---|---|---|
| Clean metrics | `evaluation/results/baseline_dctm.csv` | §3 |
| Evasion (pre-retrain) | `evaluation/results/evasion_dctm.csv` | §4 |
| Retrained eval | `evaluation/results/retrained_dctm.csv` | §5 |
| Feature ranking (hybrid) | `configs/features_dctm.json` | §2 |
| Feature ranking (MI) | `configs/features_baseline.json` | §2 |
| Diffusion index | `diffusion_index_dctm.json` | classes 1-7,10,11,12,14 |
| Diffusion checkpoints | `diffusion_dctm_class*.pt` (×11) | ~8.8 MB each |
| IDS checkpoints | `{model}_dctm.pt/.pkl`, `{model}_dctm_retrained.*` | RF retrained = 494 MB |
| Confusion matrices | `cm_{model}_dctm[_retrained].png` (×20) | per-model, clean + retrained |
| Evasion bar chart | `evasion_comparison.png` | vs DEMGAN 0.9742 line |
| Notebook | `Dctm.ipynb` | Colab driver |

> Note: the per-model confusion-matrix PNGs were located but are not machine-readable as text.
> To visually confirm §5's degeneracy claim, open `cm_svm_dctm_retrained.png` and
> `cm_mlp_dctm_retrained.png` — you should see adversarial rows piling into a **single wrong
> attack column** (not the benign column). Compare with `cm_xgboost_dctm_retrained.png`, which
> should show a clean diagonal. I can pull and embed these images if you want them in the report.
