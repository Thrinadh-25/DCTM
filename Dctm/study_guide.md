# DCTM — Complete Study Guide
### Everything A to Z | Mini Project Final Review

---

## 1. What is an Intrusion Detection System (IDS)?

An IDS monitors network traffic and flags suspicious activity. It is trained on labelled network traffic data — packets labelled either "benign" (normal) or one of many attack types (DoS, DDoS, PortScan, etc.).

**How it works:**
- Features are extracted from each network flow (packet size, duration, flag counts, etc.)
- A classifier (Decision Tree, SVM, Neural Network, etc.) is trained on these features
- At runtime, it predicts whether incoming traffic is an attack or benign

**The vulnerability:** If an attacker crafts traffic that looks benign to the classifier but is actually an attack — that is an adversarial example, and it causes an evasion.

---

## 2. What is Adversarial Machine Learning?

Adversarial ML is the study of attacks on ML models. An adversarial example is an input that has been slightly modified to fool a classifier.

**In IDS context:**
- The attacker wants the IDS to classify their malicious traffic as benign (label 0)
- They perturb the traffic features just enough to cross the decision boundary
- The traffic still performs the attack but evades detection

**Evasion Rate (ER):** The fraction of adversarial (malicious) samples that the IDS misclassifies as benign.
- ER = 1.0 → IDS is completely fooled
- ER = 0.0 → IDS detects everything even after perturbation

---

## 3. CICIDS2017 Dataset

**Full name:** Canadian Institute for Cybersecurity Intrusion Detection Evaluation Dataset 2017

**Who made it:** University of New Brunswick (UNB) + CIC

**What it contains:**
- ~2.83 million network flow samples after cleaning
- 79 features per sample (packet length stats, flag counts, flow duration, etc.)
- 14 attack types: DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS Slowhttptest, DDoS, PortScan, FTP-Patator, SSH-Patator, Bot, Web Attack (XSS, SQL Injection, Brute Force), Heartbleed, Infiltration
- Plus benign (normal) traffic

**Why it's used:** It's the standard benchmark dataset for IDS research. DEMGAN (our base paper) used it, so we use it to compare directly.

**Class imbalance:** The dataset is heavily imbalanced — benign traffic dominates, and rare attacks like Infiltration and Heartbleed have very few samples. This is why SMOTE is needed.

---

## 4. CICIDS2018 Dataset

**Full name:** CSE-CIC-IDS2018

**Who made it:** Communications Security Establishment (CSE) + CIC + UNB

**What it contains:**
- 2 million+ network flow samples
- Similar 79 features
- Attack types include DoS Hulk, Bot, SlowHTTPTest + Benign

**Why we use it:** External validation — we train on 2017 and test generalizability on 2018. If evasion works on 2018-trained models too, our approach generalizes across datasets.

---

## 5. What is GAN (Generative Adversarial Network)?

A GAN has two networks that compete:
- **Generator (G):** tries to produce fake data that looks real
- **Discriminator (D):** tries to distinguish real from fake

They train together — G gets better at fooling D, and D gets better at detecting fakes.

**In adversarial IDS context:** G generates fake network traffic samples. D tries to tell them apart from real attack traffic. Eventually G produces realistic adversarial traffic.

**Problems with GAN:**
- **Training instability:** G and D can oscillate and never converge
- **Mode collapse:** G collapses to generating only one or a few types of output, ignoring diversity
- **Gradient vanishing:** When D is too good, G gets no learning signal

---

## 6. Wasserstein GAN (WGAN) — Arjovsky et al., 2017

WGAN fixes GAN instability by changing the loss function.

**Key changes:**
- Uses **Earth Mover (EM) distance** (also called Wasserstein-1 distance) instead of Jensen-Shannon divergence
- Replaces "Discriminator" with a "**Critic**" that outputs a score instead of a probability
- Enforces **Lipschitz constraint** via weight clipping (keeps the critic from being too extreme)

**Why it matters:**
- The EM distance is smoother and always provides a gradient
- Training is more stable — no sudden collapses
- The loss correlates directly with generation quality (lower loss = better samples)

**Limitation:** Weight clipping is a crude way to enforce Lipschitz. Can be improved with gradient penalty (WGAN-GP).

---

## 7. DEMGAN — Xu et al., 2025 (Base Paper)

**Full name:** Dual-Evasion Multi-Generator Adversarial Network

**What it does:**
- Uses **multiple generators** (each specializing in a different attack type) — increases diversity
- Uses a **Transformer-based generator** architecture
- Discriminator is **RNN-based** with a **distortion rate** in the loss function (penalizes samples that are too distorted from the original)
- Trained on CICIDS2017

**Results:**
- Achieved **97.42% evasion rate** — the baseline we are trying to beat or match
- Struggled against **linear classifiers** (Logistic Regression, Naive Bayes) — these were not well-fooled

**Limitations (what we fix):**
1. Used only **10 features** — 69 potentially informative features ignored
2. **GAN instability** — training can fail or collapse
3. Ignored **class imbalance** — rare attacks had no representation
4. Evaluation was only **evasion rate** — no F1, Precision, Recall

---

## 8. What is a Diffusion Model (DDPM)?

**DDPM = Denoising Diffusion Probabilistic Model** — Ho et al., 2020

A diffusion model generates data by learning to reverse a noise process.

**Two processes:**

**Forward process (adding noise):**
- Take a real data sample x₀ (e.g., a real attack flow)
- Gradually add Gaussian noise over T steps
- At step T, the sample is pure Gaussian noise
- This is fixed — no learning required. Controlled by a **noise schedule** (beta values)

**Reverse process (removing noise — this is what we learn):**
- Start from pure noise at step T
- Learn a neural network that predicts and removes noise step by step
- After T steps, we get a realistic generated sample

**Training:** The network is trained to predict the noise that was added at each step. Loss = mean squared error between predicted noise and actual noise.

**Why it's better than GAN:**
- Training is stable — there is no adversarial game, just supervised noise prediction
- No mode collapse — the stochastic sampling process naturally produces diverse outputs
- Higher fidelity samples in many domains

**The cost:** Slower inference — generation requires T reverse steps (T=1000 in our case)

---

## 9. Our Diffusion Model — TransformerDenoiser

We use a **Transformer-based denoiser** (not a U-Net like image diffusion models, because we're working with tabular data not images).

**Architecture:**
1. Input: noisy sample x_t (a vector of 20 features) + timestep t
2. Timestep t is encoded using **sinusoidal embedding** (128-dimensional) — same idea as positional encodings in NLP Transformers
3. The embedding is projected to model_dim=256
4. Passed through **4 Transformer Encoder layers** (nhead=4, feedforward dim=512, dropout=0.1)
5. Output: predicted noise vector (same dimension as input features)

**Why Transformer for tabular data?**
- Self-attention can model interactions between features (e.g., how packet size relates to flag count)
- More expressive than simple MLP for capturing feature dependencies

**Noise schedule:** We use a **linear beta schedule** — beta increases linearly from a small value (0.0001) to a larger value (0.02) over T=1000 steps. This controls how much noise is added at each step.

---

## 10. Adversarial Generation — How We Create Evasive Samples

Instead of generating from pure noise (which might not resemble attack traffic), we use **partial forward diffusion**:

**Step 1 — Partial forward:** Take a real attack sample, add noise only up to t=500 (halfway). The sample is noisy but still has the attack's structure.

**Step 2 — Full reverse:** Run the reverse denoising process from t=500 back to t=0. The denoiser creates a new variant that is similar to the original attack but different enough to fool the IDS.

**Why t=500?** Going all the way to T=1000 would destroy all attack semantics. Stopping at 500 preserves the attack class identity while introducing enough variation for evasion.

**Immutable feature clamping:** After every reverse step, we clamp immutable features (Destination Port, URG Flag, CWE Flag for CICIDS2017) back to their original values. This ensures the generated traffic remains realistic — you can't change the destination port mid-flight in a real network.

---

## 11. Feature Engineering — Mutual Information (MI)

**What it is:** MI measures how much knowing one variable reduces uncertainty about another.

**In our context:** MI(feature, label) — how much does knowing feature X help predict whether traffic is an attack or benign?

**How we use it:**
- Compute MI between each of the 79 features and the traffic label
- Rank features by MI score
- Top 10 by MI = **baseline feature set** (same as DEMGAN)
- MI contributes 50% weight to the hybrid ranking

**Why it works:** Features with high MI are statistically predictive of the class label. Simple, fast, and dataset-agnostic.

---

## 12. Feature Engineering — SHAP

**SHAP = SHapley Additive exPlanations**

**What it is:** A game theory approach to explain the contribution of each feature to a model's prediction.

**How it works:**
- Train a proxy model (we use LightGBM — fast and accurate)
- For each sample, SHAP computes how much each feature pushed the prediction higher or lower
- Average the absolute SHAP values across all samples → feature importance score

**In our context:**
- We train LightGBM on top 50,000 samples
- Compute SHAP values → rank all 79 features by mean absolute SHAP
- SHAP contributes 50% weight to the hybrid ranking

**Why SHAP over just accuracy:** SHAP tells you *why* the model made a decision — which features actually matter. MI is statistical, SHAP is model-based. Together they cover both perspectives.

**Hybrid formula:** Score = 0.5 × (normalized MI) + 0.5 × (normalized SHAP) → select top 20

---

## 13. Immutable Features

Some network features cannot be changed by an attacker without breaking the network protocol or the attack itself.

**CICIDS2017 immutable features:**
- **Destination Port** — you can't change where you're sending traffic
- **URG Flag Count** — controlled by TCP protocol
- **CWE Flag Count** — controlled by TCP protocol

**Why this matters:** If we perturb these during adversarial generation, the resulting traffic would be physically impossible or protocol-violating. So we freeze them — after every reverse diffusion step, we restore these to their original values.

---

## 14. SMOTE — Synthetic Minority Over-sampling Technique

**Problem:** CICIDS2017 is heavily imbalanced. Benign traffic = millions of samples. Some attack types (Heartbleed, Infiltration) = only tens of samples. A classifier trained on this will be biased toward benign.

**What SMOTE does:**
- For each minority class sample, find its K nearest neighbours in feature space
- Create new synthetic samples by interpolating between the sample and a randomly chosen neighbour
- `synthetic = sample + λ × (neighbour - sample)` where λ is random between 0 and 1

**In our pipeline:**
- Applied to training data only (never to test data — that would leak information)
- We cap at 50,000 samples per class — avoids over-generating and slowing down training
- After SMOTE, all classes are roughly balanced → better classifier training, better adversarial generation for rare attack types

---

## 15. The 10 IDS Models

We train and evaluate on 10 classifiers:

**Classical (CPU-based, Scikit-learn):**
| Model | How it works |
|-------|-------------|
| Decision Tree (DT) | Splits features at thresholds to partition data into classes. Interpretable but can overfit. |
| Naive Bayes (NB) | Applies Bayes' theorem assuming features are independent. Fast, works well on simple distributions. |
| Logistic Regression (LR) | Learns a linear boundary between classes. Simple and fast but limited to linear separation. |
| Random Forest (RF) | Ensemble of many decision trees — majority vote. More robust than a single DT. |
| XGBoost | Gradient boosted trees — trains trees sequentially, each correcting the previous one's errors. |
| SVM | Finds the maximum-margin hyperplane between classes. Can use kernels for nonlinear boundaries. |

**Deep Learning (PyTorch, can use GPU):**
| Model | How it works |
|-------|-------------|
| MLP | Multi-Layer Perceptron — fully connected layers with nonlinear activations. Standard neural net. |
| CNN | Convolutional Neural Network — applies filters to detect local patterns in feature sequences. |
| RNN | Recurrent Neural Network — processes features sequentially, capturing temporal dependencies. |
| CNN-BiLSTM | CNN for local features + Bidirectional LSTM for sequential context in both directions. |

---

## 16. Evaluation Metrics

**Accuracy:** (TP + TN) / Total — overall correctness. Misleading on imbalanced data.

**Precision:** TP / (TP + FP) — of all samples predicted as attack, how many actually were?

**Recall:** TP / (TP + FN) — of all actual attacks, how many did we detect?

**F1-Score:** 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean. Good for imbalanced classes.

**Evasion Rate (ER):** Of all adversarial (malicious) samples, what fraction was classified as benign?
- ER = misclassified-as-benign / total-adversarial-samples
- High ER = our adversarial samples successfully fool the IDS

**Clean Accuracy vs Adversarial Accuracy:**
- Clean Acc = model accuracy on original (unperturbed) test data
- Adv Acc = model accuracy on adversarial samples

---

## 17. Noise Schedule

Controls how much noise is added at each diffusion timestep.

**Linear schedule:** beta_t increases linearly from β_min=0.0001 to β_max=0.02 over T=1000 steps.

**alpha_t = 1 - beta_t**

**alpha_bar_t = product of all alpha values from 1 to t**

The forward process: x_t = sqrt(alpha_bar_t) × x₀ + sqrt(1 - alpha_bar_t) × ε, where ε ~ N(0, I)

This means at t=0 the sample is clean, and at t=T it's pure noise. The model learns to reverse this.

**Cosine schedule** (alternative, more gradual): noise is added more smoothly — useful when the linear schedule degrades quality too quickly early on.

---

## 18. Results Interpretation

**Table 1 — Evasion Attack:**

| Model | Clean Acc | Adv Acc | Adv ER |
|-------|-----------|---------|--------|
| SVM | 0.562 | 0.000 | 1.000 |
| Naive Bayes | 0.390 | 0.011 | 0.936 |
| Random Forest | 0.995 | 0.000 | 0.611 |

- SVM and NB are most vulnerable (ER near 1.0)
- DT and CNN-BiLSTM are most robust (lower ER ~0.15-0.4)
- Clean accuracy stays high — adversarial perturbation is subtle and doesn't break the attack

**Table 2 — After Adversarial Retraining:**

| Model | Clean Acc | Adv Acc | Adv ER |
|-------|-----------|---------|--------|
| Decision Tree | 0.991 | 0.644 | 0.002 |
| SVM | 0.686 | 0.109 | 0.000 |

- After retraining on adversarial samples, ER drops to near 0 for all models
- This is **adversarial training** — exposing the model to adversarial examples makes it robust
- Clean accuracy is largely preserved or even improved

**Key takeaway:** Our diffusion model generates effective adversarial samples (high ER), and adversarial retraining successfully hardens the IDS (ER → 0). This is the complete red-team → blue-team cycle.

---

## 19. Pipeline Phases (main.py)

| Phase | What happens |
|-------|-------------|
| preprocess | Load CSVs, clean (drop NaN/inf), encode labels, normalize, split 80/20 train/test, save as parquet |
| features | Compute MI + SHAP hybrid scores, select top 10 (baseline) and top 20 (DCTM) features, save JSON |
| train | Apply SMOTE, train all 10 IDS models, save checkpoints (.pkl for classical, .pt for deep) |
| diffusion | Train one TransformerDenoiser per attack class, save diffusion model checkpoints |
| attack | Load real attack samples, apply partial forward diffusion to t=500, reverse denoise, clamp immutables, save adversarial parquets |
| evaluate | Run all 10 IDS models on adversarial samples, compute ER, F1, Precision, Recall |
| retrain | Augment training set with adversarial samples, retrain all 10 models |
| external | Repeat evaluate/retrain on CICIDS2018 for generalization testing |
| report | Summarize best F1, most vulnerable model, retraining gains into final_report.txt |

---

## 20. Likely Viva Questions & Answers

**Q: What is the main contribution of DCTM?**
Replacing GAN with a stable Transformer-based Diffusion Model for adversarial IDS evasion, combined with expanded feature selection (10 → 20) and SMOTE class balancing.

**Q: Why is GAN bad for this problem?**
GAN training is adversarial — G and D can fail to converge. Mode collapse causes G to generate only a few types of samples. Diffusion is supervised (noise prediction), so it's stable and produces diverse samples.

**Q: What is the difference between the forward and reverse process in diffusion?**
Forward = gradually add Gaussian noise to data over T steps (fixed, no learning). Reverse = neural network predicts and removes noise step by step to generate new samples (this is what's learned).

**Q: Why do you train one diffusion model per attack class?**
Each attack class has a distinct feature distribution. A per-class model specializes in that distribution and generates more realistic, class-specific adversarial samples.

**Q: What does MI measure?**
Mutual Information measures statistical dependence between a feature and the class label. High MI = the feature carries a lot of information about whether traffic is an attack or benign.

**Q: What does SHAP measure?**
SHAP measures each feature's contribution to a trained model's prediction. It uses Shapley values from game theory to fairly distribute credit among features.

**Q: Why hybrid MI + SHAP and not just one of them?**
MI is statistical and model-agnostic. SHAP is model-based. Together they cover both perspectives — a feature ranked high by both is genuinely important. Using only one could miss features that the other would catch.

**Q: What is SMOTE?**
Synthetic Minority Over-sampling Technique. It creates synthetic samples for underrepresented classes by interpolating between existing samples of that class. Ensures rare attacks are well-represented during training.

**Q: What are immutable features and why do you clamp them?**
Features that cannot physically be changed in real network traffic (Destination Port, URG Flag, CWE Flag). We clamp them after every reverse diffusion step to keep adversarial samples realistic and protocol-compliant.

**Q: Why partial forward diffusion to t=500?**
Full diffusion (t=1000) destroys all information — the sample becomes pure noise with no attack identity. t=500 preserves the attack structure while still introducing enough variation for the reverse process to create a diverse evasive variant.

**Q: What does evasion rate mean?**
Fraction of adversarial (malicious) samples misclassified as benign by the IDS. ER=1.0 means the IDS was completely fooled. ER=0 means it detected everything.

**Q: Which model was most vulnerable?**
SVM with 100% evasion rate, followed by Naive Bayes at 93.6%.

**Q: What happened after adversarial retraining?**
Evasion rate dropped to near 0 for all models — the IDS became robust against the same adversarial samples it was retrained on.

**Q: What is T=1000?**
The number of timesteps in the diffusion process. More steps = more gradual noise addition = higher fidelity but slower generation.

**Q: What does the Transformer in TransformerDenoiser do?**
It takes the noisy feature vector and timestep embedding as input, uses self-attention across features to capture inter-feature dependencies, and outputs the predicted noise to be removed.

**Q: How is your work different from DEMGAN?**
(1) Diffusion vs GAN — more stable, no mode collapse. (2) 20 features vs 10 — more informative. (3) SMOTE — handles class imbalance DEMGAN ignored. (4) Evaluation extended to F1, Precision, Recall. (5) 10 IDS models vs DEMGAN's subset.

**Q: What is sinusoidal embedding?**
A way to encode the timestep t as a fixed-dimensional vector using sine and cosine functions at different frequencies. Same concept as positional encoding in the original Transformer paper. It allows the model to distinguish between different noise levels.

**Q: Why LightGBM as the SHAP proxy?**
LightGBM is fast, accurate on tabular data, and natively supports SHAP computation. It's used only to rank features — not as an IDS model itself.

**Q: What is the CICIDS2018 used for?**
External validation — we test if the adversarial samples generated from CICIDS2017-trained diffusion models also fool CICIDS2018-trained IDS models. This tests cross-dataset generalizability.
