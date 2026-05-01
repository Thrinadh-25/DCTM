# DCTM — Two Core Diagrams
# Team 06 | CSE-C-06

---

# DIAGRAM 1 — DATA PIPELINE WORKFLOW
# "From raw CSV to trained models — step by step"

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAW INPUT DATA                                  │
│                                                                     │
│   CICIDS2017 (8 CSV files, ~885 MB)                                 │
│   CICIDS2018 (3 CSV files, ~1.1 GB)                                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 1 — DATA CLEANING                          │
│                                                                     │
│   • Remove rows with NaN (not a number) values                      │
│   • Remove rows with Infinity values                                │
│   • Strip whitespace from column names                              │
│   • Encode attack class labels as integers                          │
│                                                                     │
│   Input:  ~3M+ raw rows (noisy)                                     │
│   Output: ~2.83M clean rows (CICIDS2017)                            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 2 — NORMALISATION                          │
│                                                                     │
│   Min-Max Scaling applied to every feature column:                  │
│                                                                     │
│               x* = (x − x_min) / (x_max − x_min)                   │
│                                                                     │
│   All feature values now live in range [0, 1]                       │
│   This is required for diffusion model clipping later               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 3 — TRAIN / TEST SPLIT                     │
│                                                                     │
│   Stratified split:  80% Train  |  20% Test                         │
│                                                                     │
│   ┌──────────────────────┐    ┌─────────────────────────┐           │
│   │     TRAIN SET        │    │       TEST SET           │           │
│   │  ~2.26M rows         │    │   ~566k rows             │           │
│   │  used for training   │    │   NEVER touched during   │           │
│   │  and SMOTE           │    │   SMOTE or training      │           │
│   └──────────┬───────────┘    └─────────────────────────┘           │
└──────────────┼──────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 4 — SMOTE BALANCING                        │
│             (applied on TRAIN SPLIT ONLY)                           │
│                                                                     │
│   Problem: classes are severely imbalanced                          │
│     Benign      = 2,271,319  ←── too dominant                       │
│     DDoS        =   128,025                                         │
│     Bot         =     1,956                                         │
│     Heartbleed  =        11  ←── too rare to learn                  │
│                                                                     │
│   SMOTE fix: synthetically generate minority class samples          │
│     k_neighbors = 5                                                 │
│     cap = 50,000 samples per class                                  │
│     skip classes with fewer than 6 native samples                   │
│                                                                     │
│   After SMOTE: each usable class has up to 50,000 samples           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 5 — FEATURE SELECTION                      │
│                                                                     │
│   All 78 CICIDS features → select the most important ones           │
│                                                                     │
│   ┌─────────────────────┐    ┌──────────────────────────┐           │
│   │   MI RANKING        │    │   SHAP RANKING           │           │
│   │                     │    │                          │           │
│   │ Mutual Information  │    │ Train LightGBM proxy     │           │
│   │ on 100k subsampled  │    │ on 50k rows              │           │
│   │ rows                │    │ → compute SHAP values    │           │
│   │                     │    │ → mean(|SHAP|) per feat  │           │
│   │ Measures: how much  │    │                          │           │
│   │ each feature tells  │    │ Measures: how much each  │           │
│   │ us about the label  │    │ feature affects the      │           │
│   │                     │    │ model's prediction       │           │
│   └──────────┬──────────┘    └────────────┬─────────────┘           │
│              │                            │                         │
│              └─────────────┬──────────────┘                         │
│                            │                                        │
│                            ▼                                        │
│          Hybrid Score = 0.5 × MI_norm + 0.5 × SHAP_norm            │
│                            │                                        │
│                 ┌──────────┴──────────┐                             │
│                 ▼                     ▼                             │
│           Top 10 features       Top 20 features                     │
│           (baseline set)        (DCTM set)                          │
│         features_baseline.json  features_dctm.json                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 6 — IDS MODEL TRAINING                     │
│                                                                     │
│   Train all 10 IDS classifiers on SMOTE-balanced train data         │
│   Each model trained TWICE: once on 10 features, once on 20         │
│   Total: 20 model checkpoints saved to  data/models/                │
│                                                                     │
│   Classical (CPU):                 Deep Learning (GPU):             │
│   • Decision Tree                  • MLP [256 → 128 → 64]           │
│   • Naive Bayes                    • CNN (1D, channels 32/64)       │
│   • Logistic Regression            • RNN (GRU, hidden=128)          │
│   • Random Forest (200 trees)      • CNN-BiLSTM                     │
│   • XGBoost (200 trees, CUDA)                                       │
│   • SVM (RBF kernel, 50k cap)                                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 7 — DIFFUSION MODEL TRAINING               │
│                                                                     │
│   One Transformer Denoiser trained per attack class                 │
│   (separate model for DDoS, DoS, PortScan, Bot, etc.)               │
│                                                                     │
│   Training loop per class:                                          │
│     For each batch of class-c samples:                              │
│       1. Sample random timestep t from 0 to 999                     │
│       2. Add noise: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε              │
│       3. Denoiser predicts noise: ε_pred = model(x_t, t)            │
│       4. Loss = MSE(ε_pred, ε)                                      │
│       5. Backprop → AdamW update                                    │
│                                                                     │
│   Hyperparameters:                                                  │
│     T = 1000  |  epochs = 100  |  batch = 256  |  lr = 1e-4        │
│                                                                     │
│   Output: one .pt checkpoint per class → data/models/diffusion_*.pt │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 8 — ADVERSARIAL SAMPLE GENERATION          │
│                                                                     │
│   For each attack class:                                            │
│     1. Take real attack samples from test set                        │
│     2. Forward diffuse to t = T/2 = 500  (add partial noise)        │
│     3. Reverse denoise back to t = 0                                │
│        → At EVERY reverse step: clamp immutable features back       │
│          to original values (Destination Port, Flag Counts)         │
│     4. Clip all values to [0, 1]                                    │
│                                                                     │
│   Output: adversarial samples saved to data/adversarial/*.parquet   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 9 — EVALUATION                             │
│                                                                     │
│   Feed adversarial samples into all 10 trained IDS models           │
│                                                                     │
│   Metrics computed:                                                 │
│     • Evasion Rate (ER) = adversarial samples predicted as Benign   │
│                           ─────────────────────────────────────     │
│                             total adversarial malicious samples     │
│                                                                     │
│     • F1 Score, Precision, Recall, ROC-AUC                          │
│                                                                     │
│   Compare ER against DEMGAN baseline: 97.42%                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP 10 — ADVERSARIAL RETRAINING                │
│                                                                     │
│   Augment original train set with generated adversarial samples     │
│   Retrain all 10 IDS models on augmented data                       │
│   Re-evaluate: ER should DROP (IDS now recognises the attacks)      │
│   F1 should RISE (IDS is more robust after retraining)              │
│                                                                     │
│   This proves DCTM's dual use:                                      │
│     Attack role  → generates evasive adversarial traffic            │
│     Defense role → hardens IDS through adversarial retraining       │
└─────────────────────────────────────────────────────────────────────┘
```

---
---

# DIAGRAM 2 — GENERAL PERSPECTIVE (SYSTEM OVERVIEW)
# "What is DCTM doing — the big picture, not the training"

```
╔═════════════════════════════════════════════════════════════════════╗
║              THE DCTM SYSTEM — WHO DOES WHAT                       ║
╚═════════════════════════════════════════════════════════════════════╝


  ┌──────────────────────────────────────────────────────────────┐
  │                  REAL WORLD NETWORK TRAFFIC                  │
  │                                                              │
  │   Benign flows  (normal user activity)                       │
  │   Attack flows  (DDoS, PortScan, Bot, XSS, SQL injection...) │
  │                                                              │
  │   Source: CICIDS2017 dataset — 2.83 million labelled flows   │
  └────────────────────────┬─────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
  ┌─────────────────────┐       ┌─────────────────────────────┐
  │   THE IDS MODELS    │       │   THE DIFFUSION GENERATOR   │
  │   (10 classifiers)  │       │   (Transformer Denoiser)    │
  │                     │       │                             │
  │  These are the      │       │  This is DCTM's core.       │
  │  DEFENDERS /        │       │  It learns what attack      │
  │  VICTIMS            │       │  traffic looks like, then   │
  │                     │       │  generates modified copies  │
  │  They learn to      │       │  that look BENIGN to the    │
  │  classify traffic   │       │  IDS models.                │
  │  as Attack/Benign   │       │                             │
  │                     │       │  One model per attack class. │
  │  Classical:         │       │  Not one shared model.       │
  │  • Decision Tree    │       │                             │
  │  • Naive Bayes      │       │  Architecture:              │
  │  • Logistic Reg     │       │  Transformer Encoder ×4     │
  │  • Random Forest    │       │  with sinusoidal time embed  │
  │  • XGBoost          │       │                             │
  │  • SVM              │       │  Trained with:              │
  │                     │       │  MSE loss on noise prediction│
  │  Deep Learning:     │       │  (DDPM objective)            │
  │  • MLP              │       │                             │
  │  • CNN (1D)         │       │  NOT a GAN — no discriminator│
  │  • RNN (GRU)        │       │  NO adversarial game         │
  │  • CNN-BiLSTM       │       │  Stable, reproducible        │
  └─────────┬───────────┘       └──────────────┬──────────────┘
            │                                  │
            │                                  │
            │    ┌─────────────────────────────┘
            │    │   Diffusion generates adversarial samples
            │    │   (attack traffic modified to look benign)
            │    │
            ▼    ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              THE EVASION TEST                                │
  │                                                              │
  │   Adversarial samples (from Diffusion) are fed into          │
  │   the trained IDS models (10 classifiers)                    │
  │                                                              │
  │   Expected result:  IDS predicts → "BENIGN"                  │
  │   Reality:          the sample is a MALICIOUS attack flow    │
  │                                                              │
  │   Evasion Rate = how often the IDS is fooled                 │
  │   Target:  beat DEMGAN's 97.42% average ER                   │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │              THE DEFENSE LOOP (Retraining)                   │
  │                                                              │
  │   The same adversarial samples that fooled the IDS           │
  │   are now used to RETRAIN the IDS                            │
  │                                                              │
  │   IDS learns:  "these are actually attacks, not benign"      │
  │                                                              │
  │   After retraining:                                          │
  │     ER drops  ↓  (IDS gets harder to fool)                   │
  │     F1 rises  ↑  (IDS gets more accurate)                    │
  │                                                              │
  │   DCTM is both attacker AND defender                         │
  └──────────────────────────────────────────────────────────────┘


╔═════════════════════════════════════════════════════════════════════╗
║         HOW THE MODELS RELATE TO EACH OTHER                        ║
╚═════════════════════════════════════════════════════════════════════╝

  DEMGAN (old approach)                DCTM (our approach)
  ─────────────────────                ───────────────────
  WGAN with 3 generators               Transformer Denoiser
        │                                      │
        │  min-max adversarial game            │  MSE noise prediction
        │  unstable training                   │  stable training
        │  mode collapse                       │  no mode collapse
        │  10 features (MI only)               │  20 features (MI + SHAP)
        │  no class balancing                  │  SMOTE balanced
        │  fails on linear IDS                 │  works on all 10 IDS
        ▼                                      ▼
  Adversarial samples                  Adversarial samples
  (lower quality on DT, LR)            (consistent quality across all)


╔═════════════════════════════════════════════════════════════════════╗
║         THE FOUR PILLARS — WHAT EACH MODEL/COMPONENT DOES          ║
╚═════════════════════════════════════════════════════════════════════╝

  P1 — TRANSFORMER DIFFUSION MODEL
       What it is:  A neural network that learns to predict noise
       What it does: Generates adversarial attack samples
       Why better:  Stable training, no GAN game, consistent output
       Key detail:  One model per attack class (e.g. one for DDoS,
                    one for PortScan, one for Bot, etc.)

  P2 — SMOTE (Synthetic Minority Over-sampling Technique)
       What it is:  A data augmentation method
       What it does: Creates synthetic samples for rare attack classes
       Why needed:  Without it, Bot (1956 samples) and Heartbleed
                    (11 samples) are never properly learned
       Key detail:  Applied on TRAIN data only. Test kept as-is.

  P3 — HYBRID FEATURE SELECTION (MI + SHAP)
       What it is:  A feature ranking system combining two methods
       What it does: Picks the 20 most important features out of 78
       Why better:  MI alone misses classifier-specific dependencies.
                    SHAP fills that gap. Together = better coverage.
       Key detail:  10 features = baseline. 20 features = DCTM set.

  P4 — EXTENDED EVALUATION METRICS
       What it is:  A broader measurement framework
       What it does: Reports F1, Precision, Recall alongside ER
       Why needed:  Evasion Rate alone tells only part of the story.
                    A model with 99% ER but 10% F1 is useless.
       Key detail:  Both clean-data and adversarial metrics reported.


╔═════════════════════════════════════════════════════════════════════╗
║         ONE-LINE SUMMARY                                           ║
╚═════════════════════════════════════════════════════════════════════╝

  DCTM uses a Diffusion Model to generate realistic adversarial
  network traffic that evades 10 ML-based IDS classifiers — and
  then uses that same adversarial data to make those classifiers
  stronger through retraining.
```
