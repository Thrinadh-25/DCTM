# SLIDE NOTES — DCTM Review 2
# Quick glance cue cards | P1 = Neha | P2 = Thrinadh

---

## SLIDE 1 — TITLE | P1: Neha

- Good morning, we are Team 06, CSE-C
- Project: DCTM — Diffusion-Based Cyber Threat Modelling
- I'm Neha, teammate is Thrinadh
- Guide: Ms. Duddeda Aishwarya

---

## SLIDE 2 — INDEX | P1: Neha

- Quick walkthrough of slides
- Abstract → Introduction → Problem → Workflow → Architecture → Methodology → Dataset → References

---

## SLIDE 3 — ABSTRACT | P1: Neha

- IDS = system that detects network attacks
- DEMGAN (existing paper) = 97.42% evasion rate but 3 problems
- 3 problems: only 10 features, GAN unstable, ignores class imbalance
- We built DCTM → fixes all 3
- DCTM uses: Diffusion model + SMOTE + 20 features
- Tested on 2 datasets, measured with F1 + Precision + Recall too

---

## SLIDE 4 — INTRODUCTION | P1: Neha

- Networks face attacks: DDoS, PortScan, SQL Injection etc.
- IDS = ML model that labels traffic as Safe or Attack
- Problem → attacker can craft fake "safe-looking" traffic → IDS gets fooled
- This is called an evasion attack
- We need tools to test how easily IDS gets fooled
- DEMGAN does this with GAN → we do it better with Diffusion
- Our tool works both ways → attack IDS AND strengthen it

---

## SLIDE 5 — PROBLEM STATEMENT | P2: Thrinadh

**Gaps (what's wrong with DEMGAN):**
- Only 10 features used → too limited
- GAN is unstable → bad on Decision Tree and Logistic Regression
- Rare attacks ignored → Heartbleed has only 11 samples
- Only evasion rate measured → no F1, Precision, Recall

**Our fix:**
- Replace GAN → Diffusion model (stable)
- 10 → 20 features using MI + SHAP
- SMOTE → fixes class imbalance
- Add F1, Precision, Recall to evaluation

---

## SLIDE 6 — WORKFLOW | P2: Thrinadh

*(point to diagram while speaking)*

- Start: raw CSV files (CICIDS2017 dataset)
- Step 1: clean data → remove bad rows → normalize to 0-1
- Step 2: pick top 20 features using MI + SHAP combined
- Step 3: SMOTE → balance rare attack classes → train 10 IDS models
- Step 4: train Diffusion model per attack class
- Step 5: generate adversarial samples → test on IDS → measure evasion rate
- End: retrain IDS with adversarial data → IDS becomes stronger

---

## SLIDE 7 — ARCHITECTURE | P2: Thrinadh

- Two main parts: 10 IDS classifiers + Diffusion model (Transformer Denoiser)
- IDS models: DT, NB, LR, RF, XGB, SVM → classical / MLP, CNN, RNN, CNN-BiLSTM → deep learning
- Diffusion model: one model per attack class
- Takes noisy sample + timestep → predicts noise → removes it → gets adversarial sample
- We are still studying diffusion internals deeply → will cover in Review 3

---

## SLIDE 8 — METHODOLOGY | P2: Thrinadh

**5 modules:**
- Preprocessing → clean, normalize, split 80/20
- Feature selection → top 20 via MI + SHAP, protocol fields frozen (port, flags)
- Diffusion training → one denoiser per class, MSE loss
- Adversarial generation → add noise halfway → denoise back → clamp immutable features
- Evaluation + Retraining → measure ER, F1 → retrain → IDS improves

---

## SLIDE 9 — DATASET | P2: Thrinadh

**CICIDS2017:**
- 2.83 million samples, 79 features, 14 attack types
- Main training + evaluation dataset
- Same dataset used in DEMGAN

**CICIDS2018:**
- 2M+ samples, 4 attack types
- Only for generalisation testing → never used in training

---

## SLIDE 10 — REFERENCES | P1: Neha

- DEMGAN 2025 → base paper we are improving
- Wasserstein GAN 2017 → what DEMGAN was built on
- SMOTE 2002 → for class balancing
- CICIDS dataset paper → dataset we used

---

## SLIDE 11 — THANK YOU | P1: Neha

- Thank you for your time
- Open to questions and suggestions
