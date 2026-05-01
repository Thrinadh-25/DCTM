# DCTM — Presentation Script
# 2 Speakers | Review 2 | CSE-C-06

P1 = Neha Sri
P2 = Thrinadh Reddy

---

## SLIDE 1 — TITLE

**P1 speaks:**

Good morning, respected faculty and panel members.
We are Team 06 from CSE-C.
Our mini project is DCTM — Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance.
I am Neha Sri, and my teammate is Thrinadh Reddy.
Our guide is Ms. Duddeda Aishwarya, Assistant Professor.

---

## SLIDE 2 — INDEX

**P1 speaks:**

Here is our presentation index.
We will be covering Abstract, Introduction, Problem Statement, Workflow, Architecture, Methodology, Dataset, and References.

---

## SLIDE 3 — ABSTRACT

**P1 speaks:**

ML-based Intrusion Detection Systems are highly vulnerable to adversarial traffic.
The existing framework DEMGAN uses Wasserstein GANs and achieves 97.42% evasion rate on CICIDS2017 — but it has three limitations: only 10 features are used, GAN training is unstable, and class imbalance is ignored.

We propose DCTM — which replaces the GAN with a Transformer-based Diffusion Model, adds SMOTE for class balancing, and expands features to 20 using a hybrid MI and SHAP approach.
We evaluate on both CICIDS2017 and CICIDS2018, with metrics including F1, Precision, and Recall — not just evasion rate.

---

## SLIDE 4 — INTRODUCTION

**P1 speaks:**

Modern networks face constant threats like DDoS, PortScan, SQL Injection, and Botnets.
ML-based IDS systems classify network traffic as benign or malicious — and they perform well on standard datasets.

But the problem is — ML models are not robust to adversarial inputs.
An adversarial traffic sample is a malicious flow that is crafted to look benign to the IDS — this is called an evasion attack.

To test how robust an IDS is, we need tools that generate these adversarial samples.
DEMGAN is the current state-of-the-art tool for this — it uses a Wasserstein GAN.
DCTM improves on DEMGAN using Diffusion Models, which are more stable and consistent.
And DCTM is dual-use — it both attacks the IDS and helps defend it through retraining.

---

## SLIDE 5 — PROBLEM STATEMENT

**P2 speaks:**

Now let me walk you through the problem statement.

The existing system DEMGAN has these gaps —
First, it uses only 10 features — 69 or more informative features are completely ignored.
Second, GAN training is unstable — it suffers from mode collapse, giving low-quality samples.
Third, class imbalance in the dataset is ignored — rare attacks like Heartbleed have only 11 samples.
And fourth, only evasion rate is reported — there is no F1, Precision, or Recall.

Our solution addresses all four —
We replace the GAN with a Tabular Diffusion Model for stable generation.
We expand features to 20 using MI and SHAP.
We apply SMOTE to fix class imbalance.
And we extend evaluation to include F1, Precision, and Recall.

---

## SLIDE 6 — WORKFLOW

**P2 speaks:**

This diagram shows our end-to-end pipeline.

Starting from raw CSV data — we clean it, normalize it, and do an 80-20 train-test split.
Then we do feature selection using MI and SHAP to get our top 20 features.
SMOTE is applied on the train split to balance rare attack classes.
We then train 10 IDS classifiers — both classical and deep learning models.
In parallel, we train one Transformer Denoiser per attack class.
The denoiser generates adversarial samples by adding partial noise and then denoising.
These adversarial samples are evaluated on all 10 IDS models — giving us evasion rate, F1, Precision, and Recall.
Finally, we retrain the IDS with adversarial data to measure the defense improvement.

---

## SLIDE 7 — ARCHITECTURE

**P2 speaks:**

For the architecture — our system has two main components.

First, the 10 IDS classifiers — these include Decision Tree, Naive Bayes, Logistic Regression, Random Forest, XGBoost, SVM on the classical side, and MLP, CNN, RNN, and CNN-BiLSTM on the deep learning side.

Second, the Transformer Denoiser — this is the core of DCTM.
It takes a noisy attack sample and a timestep as input, and predicts the noise.
One denoiser is trained per attack class — so one for DDoS, one for PortScan, and so on.
We are still studying the deeper internals of the diffusion model and will cover that in detail in Review 3.

---

## SLIDE 8 — METHODOLOGY

**P2 speaks:**

Our methodology has five modules.

Data Preprocessing — clean, normalize, split.
Adaptive Feature Selection — top 20 features using MI and SHAP. Protocol fields like IP, flags, and port are frozen and never changed.
Diffusion Training — train one Transformer Denoiser per attack class using noise prediction loss.
Adversarial Sample Generation — apply partial noise up to halfway, then denoise back — clamping immutable features at every step.
IDS Evaluation and Retraining — feed adversarial samples into 7 IDS classifiers, measure evasion rate, then retrain and measure robustness improvement.

---

## SLIDE 9 — DATASET

**P2 speaks:**

We use two standard benchmark datasets.

CICIDS2017 — from the Canadian Institute for Cybersecurity, UNB.
It has around 2.83 million samples after cleaning, 79 features, and 14 attack types including DDoS, PortScan, Bot, and XSS.
This is our primary training and evaluation dataset — same one used in DEMGAN.

CICIDS2018 — also from CIC, UNB.
We use a partial selection — around 2 million samples with DoSHulk, Bot, SlowHTTPTest, and Benign classes.
This is used only for external generalisation testing — it is never touched during training.

---

## SLIDE 10 — REFERENCES

**P1 speaks:**

Our key references are —
DEMGAN, 2025 — this is our base paper.
Wasserstein GAN, 2017 — the foundation for DEMGAN.
SMOTE, 2002 — for class balancing.
And the CICIDS dataset paper, 2023 — for dataset details.

---

## SLIDE 11 — THANK YOU

**P1 speaks:**

That concludes our presentation.
Thank you for your time.
We are open to questions and suggestions.

---

## QUICK SPLIT SUMMARY

| Slide | Speaker |
|---|---|
| Title | P1 — Neha Sri |
| Index | P1 |
| Abstract | P1 |
| Introduction | P1 |
| Problem Statement | P2 — Thrinadh Reddy |
| Workflow | P2 |
| Architecture | P2 |
| Methodology | P2 |
| Dataset | P2 |
| References | P1 |
| Thank You | P1 |
