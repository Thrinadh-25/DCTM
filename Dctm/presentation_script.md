# DCTM — Presentation Script
### Mini Project Final Review | Batch C6 | CSE-C

---

## Slide 1 — Title

"Good morning / afternoon. We are presenting our mini project titled **DCTM — Diffusion-Based Cyber Threat Modelling for Improved Intrusion Detection Performance**. Our team consists of Kasturi Vishnu Vardhan, Kukunuru Thrinadh Reddy, and Neha Sri Tirunagari, guided by Ms. Duddeda Aishwarya."

---

## Slide 2 — Index

"Here is the outline of our presentation. We'll cover the abstract, introduction, literature survey, problem statement, methodology, dataset, UML diagrams, implementation, results, and conclusion."

---

## Slide 3 — Abstract

"In brief, our project addresses a core problem in cybersecurity — ML-based Intrusion Detection Systems can be fooled by adversarial network traffic. The existing framework, DEMGAN, had three major limitations: it used only 10 features, GAN training was unstable, and class imbalance in the dataset was ignored.

We propose DCTM — which replaces the GAN with a Transformer-based Diffusion Model, expands feature selection to 20 using a hybrid MI and SHAP approach, and applies SMOTE to handle class imbalance. We evaluate on both CICIDS2017 and CICIDS2018 datasets using evasion rate, F1-score, precision, and recall."

---

## Slide 4 — Introduction

"Network intrusions are increasing in frequency and sophistication. ML-based IDS systems detect attacks by analysing network traffic patterns — but they are vulnerable to adversarial examples, which are crafted inputs designed to fool the classifier.

GANs have been used to generate such adversarial examples, but they suffer from training instability and mode collapse. Diffusion models offer a more stable alternative. We also incorporate SMOTE for class imbalance and an expanded 20-feature selection to improve adversarial sample quality."

---

## Slide 5 — Literature Survey

"We reviewed two key works.

First, **Wasserstein GAN by Arjovsky et al., 2017** — this replaced the standard GAN loss with Earth Mover distance, used a critic instead of a discriminator, and solved mode collapse and training instability. This was an important foundation.

Second, **DEMGAN by Xu et al., 2025** — our base paper. It used multi-generator Transformer-based generators with an RNN-based discriminator and achieved 97.42% evasion rate on CICIDS2017. However, it was limited to 10 features and showed poor generalization against linear classifiers. Our work directly extends and improves on this."

---

## Slide 6 — Problem Statement

"The gaps in DEMGAN are clear. It used only 10 out of 79 available features, so a lot of informative signal was lost. GAN training was unstable — mode collapse meant adversarial samples lacked diversity. Class imbalance in the dataset was completely ignored, so rare attack types were underrepresented. And the evaluation was limited only to evasion rate.

Our solution: we replace GAN with a Tabular Diffusion Model, expand features to 20 using MI and SHAP, apply SMOTE for class balance, and extend evaluation to F1, Precision, and Recall."

---

## Slide 7 — Methodology

"Our pipeline has five modules.

First, **data preprocessing** — we clean, encode, and normalize the traffic data.

Second, **adaptive feature selection** — we use a hybrid of Mutual Information and SHAP to rank and select the top 20 features. Immutable features like Destination Port and URG Flag are frozen and never altered.

Third, **diffusion model training** — we train a separate Transformer-based DDPM for each attack class, with T=1000 timesteps.

Fourth, **adversarial sample generation** — we apply partial forward diffusion to t=500, then full reverse denoising to generate realistic adversarial traffic.

Fifth, **evaluation and retraining** — we measure evasion rate, F1, precision, and recall on all 10 IDS models, then retrain them on adversarial samples to measure robustness improvement."

---

## Slide 8 — Dataset

"We use two datasets.

**CICIDS2017** — from the Canadian Institute for Cybersecurity. It has approximately 2.83 million samples, 79 features, and 14 attack categories including DoS, DDoS, PortScan, Bot, SQL Injection, and XSS. This is our primary evaluation dataset, same as DEMGAN.

**CICIDS2018** — also from CIC, with 2 million+ samples and similar features. We use this for external validation — to test whether our approach generalizes beyond the training dataset."

---

## Slide 9 — Architecture Diagram

"This is the system architecture of DCTM. The pipeline flows from raw CSV data through preprocessing, feature engineering, IDS model training, diffusion model training, adversarial generation, evasion evaluation, adversarial retraining, and finally report generation. Each stage feeds into the next — all phases are modular and can be run independently once prerequisites are met."

---

## Slide 10 — Activity Diagram

"The activity diagram shows the step-by-step flow of execution — from loading the dataset, cleaning and encoding it, applying SMOTE, selecting features, training the IDS models, training the diffusion model per class, generating adversarial samples, evaluating, and retraining. This represents the actual runtime flow of our main.py pipeline."

---

## Slide 11 — Data Flow Diagram

"The data flow diagram shows how data moves through the system. Raw network sensor data comes in as CSV files, gets processed into cleaned parquets, passes through feature selection and both IDS and diffusion trainers in parallel, then feeds into the adversarial generator, which produces synthetic evasive traffic, which is then evaluated and reported."

---

## Slide 12 — Use Case Diagram

"The use case diagram shows three actors interacting with our system — the Security Researcher who runs the full pipeline and analyses results, the Red Team Attacker who generates adversarial traffic to test IDS robustness, and the Network Administrator who uses the retrained models to harden their IDS deployment."

---

## Slide 13 — Implementation

"For implementation, we used Python 3.10 as the core language. PyTorch for all deep learning models including the diffusion model. Scikit-learn for classical models. XGBoost and LightGBM for gradient boosting and SHAP feature importance. Imbalanced-learn for SMOTE. Pandas and NumPy for data handling.

Development was done in VS Code, training on Google Colab and local GPU. Hardware includes AMD Ryzen for classical models, Apple M4 with NPU for accelerated ML, and NVIDIA CUDA for PyTorch deep models and diffusion training."

---

## Slide 14 — Results

"Our results are shown in two tables.

The first table shows the **evasion attack** — how well our adversarial samples fool the original IDS models. SVM shows 100% evasion rate, Naive Bayes 93.6%, Random Forest 61.1%, XGBoost 59.9%. This confirms that our diffusion-generated samples successfully evade most classifiers.

The second table shows performance after **adversarial retraining** — the evasion rate drops to near zero for all models, meaning retraining on adversarial samples significantly hardens the IDS. This is the key contribution of our work."

---

## Slide 15 — Conclusion & Future Scope

"To conclude — DCTM successfully replaces GAN with a stable Transformer-based Diffusion Model. We expanded feature selection from 10 to 20, handled class imbalance with SMOTE, and evaluated comprehensively across 10 IDS models. Adversarial retraining significantly reduces evasion rates.

For future scope — we plan classifier-guided diffusion to inject IDS gradients into the reverse process, DDIM sampling to reduce generation time from 1000 to 50 steps, and eventually deploying DCTM as an automated IDS hardening framework."

---

## Slide 16 — References

"Our key references are DEMGAN 2025 which is our base paper, Wasserstein GAN 2017, the CSE-CIC-IDS2018 feature selection paper from 2023, and the original SMOTE paper from 2002."

---

## Slide 17 — Thank You

"That concludes our presentation. We are open to questions and suggestions. Thank you."

---

## Common Q&A to Prepare

**Q: Why diffusion over GAN?**
GANs suffer from training instability and mode collapse — they can collapse to generating only a few types of samples. Diffusion models have a stable, mathematically grounded training process using a fixed noise schedule and are known to produce higher fidelity samples.

**Q: What is evasion rate?**
It is the fraction of adversarial (malicious) samples that the IDS misclassifies as benign. A 100% evasion rate means the IDS was completely fooled.

**Q: Why did you use 20 features?**
DEMGAN used only 10. We ran MI + SHAP hybrid ranking on all 79 features and selected the top 20 — this captures more informative signal while keeping the feature space manageable.

**Q: What is SMOTE?**
Synthetic Minority Over-sampling Technique. It generates synthetic samples for minority classes by interpolating between existing samples, so rare attack types are not underrepresented.

**Q: What are immutable features?**
Features that cannot realistically be changed by an attacker without breaking the network protocol — like Destination Port, URG Flag, and CWE Flag. We freeze these so our adversarial samples remain realistic.

**Q: Why one diffusion model per class?**
Training separate models per attack class ensures each model specializes in the distribution of that specific attack type, rather than trying to learn all classes at once.

**Q: What is T=1000?**
The number of timesteps in the diffusion process — noise is added over 1000 steps during training, and removed over 1000 steps during generation.

**Q: Why partial forward diffusion to t=500?**
We don't want to fully corrupt the original attack sample. By diffusing only halfway, we preserve the attack-class semantics while still introducing enough noise for the reverse process to create a diverse, evasive variant.
