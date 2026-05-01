# DCTM — Full Architecture Overview (One Page)

```
                                    DCTM SYSTEM ARCHITECTURE
                         From Raw Data  →  Adversarial Evaluation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ┌──────────────┐     ┌────────────────────┐     ┌──────────────────────────────────┐
 │  RAW DATA    │     │   PREPROCESSING    │     │       FEATURE SELECTION          │
 │              │     │                    │     │                                  │
 │ CICIDS2017   │────▶│ • Remove NaN/Inf   │────▶│  MI Ranking + SHAP Ranking       │
 │ CICIDS2018   │     │ • MinMax Normalize │     │  Hybrid = 0.5·MI + 0.5·SHAP     │
 │              │     │ • 80/20 Split      │     │  Top 10 (baseline) │ Top 20 (DCTM)│
 └──────────────┘     └────────────────────┘     └────────────────┬─────────────────┘
                                                                   │
                             ┌─────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────────────────────────┐
         │                    TRAIN SPLIT                                │
         │                                                               │
         │    SMOTE Balancing          →      Balanced Train Data        │
         │    (cap 50k / class)               (all attack classes equal) │
         └────────────┬──────────────────────────────┬──────────────────┘
                      │                              │
                      ▼                              ▼
         ┌────────────────────────┐    ┌─────────────────────────────────┐
         │   IDS MODEL TRAINING   │    │   DIFFUSION MODEL TRAINING      │
         │                        │    │                                 │
         │  10 Classifiers        │    │  1 Transformer Denoiser         │
         │  × 2 feature sets      │    │  per attack class               │
         │  = 20 checkpoints      │    │                                 │
         │                        │    │  Learns: how attack traffic     │
         │  DT · NB · LR · RF     │    │  is distributed (noise → data)  │
         │  XGB · SVM             │    │                                 │
         │  MLP · CNN · RNN       │    │  Loss = MSE(ε_pred, ε_true)     │
         │  CNN-BiLSTM            │    │  Stable — no adversarial game   │
         └────────────────────────┘    └───────────────┬─────────────────┘
                      │                                │
                      │                                ▼
                      │               ┌─────────────────────────────────┐
                      │               │   ADVERSARIAL GENERATION        │
                      │               │                                 │
                      │               │  Real attack sample x₀          │
                      │               │      │                          │
                      │               │      ▼  Forward to t = T/2      │
                      │               │  x_{T/2}  (partial noise)       │
                      │               │      │                          │
                      │               │      ▼  Reverse denoise to t=0  │
                      │               │  x_adv  (clamp immutable feats) │
                      │               └───────────────┬─────────────────┘
                      │                               │
                      ▼                               ▼
         ┌────────────────────────────────────────────────────────────────┐
         │                     EVALUATION                                 │
         │                                                                │
         │   Adversarial samples  ──▶  10 IDS Models  ──▶  Predictions   │
         │                                                                │
         │   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
         │   │ Evasion Rate │   │   F1 Score   │   │ Precision/Recall │  │
         │   │              │   │              │   │                  │  │
         │   │ % samples    │   │ Balance of   │   │ Attack detection │  │
         │   │ predicted as │   │ precision &  │   │ accuracy of IDS  │  │
         │   │ benign       │   │ recall       │   │ after evasion    │  │
         │   │              │   │              │   │                  │  │
         │   │ Target:      │   │              │   │                  │  │
         │   │ > 97.42%     │   │              │   │                  │  │
         │   └──────────────┘   └──────────────┘   └──────────────────┘  │
         └────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
         ┌────────────────────────────────────────────────────────────────┐
         │                  ADVERSARIAL RETRAINING                        │
         │                                                                │
         │   Train set  +  Adversarial samples  →  Retrain 10 IDS        │
         │                                                                │
         │   Result:  ER ↓ (harder to fool)   F1 ↑ (more robust IDS)    │
         └────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
