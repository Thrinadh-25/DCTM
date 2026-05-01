# DCTM — Review 2 Slide Content Guide
# Strictly follows MiniProj-PPT-Instructions.pptx rules

---

## GLOBAL RULES (from instructions — apply to EVERY slide)

1. **Header (top of every slide):** `CSE-C-06          DCTM: Diffusion-Based Cyber Threat Modelling`
2. **Footer (bottom of every slide):** Page number (right-aligned)
3. **Font:** Times New Roman or Arial — uniform throughout — minimum size 20
4. **Section name at top of every content slide** (matches exactly the slide title below)
5. **If a section overflows to a second slide:** title becomes `Section Name (Contd.)`
6. **Abstract:** One paragraph, max two paragraphs. Keywords at end. NO bullets, NO point-wise content.
7. **Introduction onwards:** Bullet points only. ONE LINER per bullet. No large paragraphs.
8. **Single-side layout only** — no two-column splits anywhere.
9. **All points are spoken — slides stay brief and visual.**

---

## SLIDE 1 — TITLE SLIDE

### Section name: (none — this is the title slide)

### Content:

```
[College Logo — top left]

KESHAV MEMORIAL ENGINEERING COLLEGE
A unit of Keshav Memorial Technical Educational Society (KMTES)
(Approved by AICTE, New Delhi & Affiliated to Osmania University, Hyderabad)

DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
(CSE — C)

Mini Project Presentation on

DCTM: Diffusion-Based Cyber Threat Modelling
for Improved Intrusion Detection Performance

by

Kasturi Vishnu Vardhan     — 245523733160
Kukunuru Thrinadh Reddy    — 245523733163
Neha Sri Tirunagari        — 245523733187

Guide:
Ms. Duddeda Aishwarya
(Assistant Professor)
```

### Diagram: NONE

---

## SLIDE 2 — INDEX

### Section name: `Index`

### Content (list format — tab-separated, as per instructions template):

```
1. Abstract          —  Overview of the project, base paper problem, and keywords
2. Introduction      —  Basic idea and scope of DCTM
3. Problem Statement —  Gaps in existing system and proposed solution
4. Workflow          —  End-to-end pipeline of the DCTM framework
5. Architecture      —  Transformer Denoiser and IDS model structure
6. Methodology       —  Feature selection, diffusion process, adversarial generation
7. Dataset           —  CICIDS2017 and CICIDS2018 description
8. References        —  Base papers and citations used
```

### Diagram: NONE

---

## SLIDE 3 — ABSTRACT

### Section name: `Abstract`

### RULE: No bullets. Write as paragraph(s). Max two paragraphs. Keywords at end.

### Content:

**Paragraph 1:**
Machine learning-based Network Intrusion Detection Systems (IDS) remain highly vulnerable to adversarial traffic. While frameworks like DEMGAN utilize multi-generator Wasserstein GANs to achieve an average evasion rate of 97.42% on the CICIDS2017 dataset, they face three critical limitations: a restrictive feature set limited to 10 features via Mutual Information, GAN-based instability that fails to generalize against linear classifiers like Decision Trees, and severe dataset class imbalance.

**Paragraph 2:**
In this paper, we propose DCTM — a Tabular Diffusion-based Adversarial Framework. We replace the unstable GAN with a Transformer-based Denoising Diffusion Model that learns the attack data distribution through iterative denoising. We integrate SMOTE to handle class imbalance and expand the feature set to 20 attributes through a hybrid analysis of Mutual Information and SHAP values. Evaluations on CICIDS2017 and CICIDS2018 extend metrics beyond evasion rate to include F1-score, Precision, and Recall.

**Keywords:**
Adversarial Examples, Network Intrusion Detection System, Tabular Diffusion, Transformer, Feature Selection, SMOTE, Class Imbalance, Evasion Attack

### Diagram: NONE

---

## SLIDE 4 — INTRODUCTION

### Section name: `1. Introduction`

### RULE: Bullet points. One liner each. No paragraphs.

### Content (bullets):

- ML-based Intrusion Detection Systems (IDS) classify network traffic as Benign or Malicious.
- Modern IDS uses models such as Decision Tree, Random Forest, MLP, CNN, and RNN.
- ML models are vulnerable to adversarial inputs — subtle modifications that cause misclassification.
- An adversarial traffic sample is a malicious flow crafted to be predicted as "Benign" by the IDS.
- This is known as an evasion attack — the IDS is bypassed without triggering any alert.
- DEMGAN (CMC, 2025) is the state-of-the-art adversarial generator using Wasserstein GAN.
- DEMGAN achieves 97.42% average evasion rate on CICIDS2017 but has known limitations.
- DCTM replaces the GAN with a Transformer-based Diffusion Model for stable, high-quality generation.
- Diffusion models learn data distribution through iterative denoising — no adversarial game.
- DCTM serves dual purpose: evasion attack tool and adversarial retraining defense tool.

### Diagram:

**Diagram 1 — IDS Evasion Concept (simple block diagram, single column, centered):**

Draw a top-to-bottom or left-to-right block diagram:

```
[Attacker]
    |
    |  Crafts adversarial traffic using DCTM
    v
[Adversarial Network Flow]  ←── looks like benign traffic
    |
    v
[IDS Classifier]
(DT / RF / MLP / CNN / RNN ...)
    |
    v
Predicted: "BENIGN"   ✗  ← evasion success (IDS fooled)

Reality: the flow is MALICIOUS
```

- Keep it clean and centered on the slide.
- Use a red label for "Evasion Success" to highlight the threat.
- One sentence caption below: "DCTM generates adversarial samples that evade ML-based IDS."

---

## SLIDE 5 — PROBLEM STATEMENT

### Section name: `2. Problem Statement`

### RULE: Bullet points. One liner each.

### Content (bullets):

- DEMGAN uses only 10 features selected by Mutual Information — limits adversarial effectiveness.
- Mutual Information ignores feature interactions and classifier-specific dependencies.
- GAN training is a min-max game — prone to mode collapse and unstable convergence.
- DEMGAN fails on linear classifiers: DT evasion rate = 61.15%, Logistic Regression = 48.94% (WGAN baseline).
- CICIDS2017 is severely imbalanced: Benign = 2,271,319 samples, Heartbleed = 11 samples.
- DEMGAN ignores class imbalance — rare attack types are under-modelled.
- DEMGAN reports only Evasion Rate — no F1, Precision, or Recall.
- DCTM proposes four pillars to fix each limitation:
  - P1: Replace GAN with Transformer Diffusion Model
  - P2: Apply SMOTE for class balancing
  - P3: Hybrid MI + SHAP feature selection (top 20)
  - P4: Extend evaluation to F1, Precision, Recall, ROC-AUC

### Diagram:

**Diagram 2 — DEMGAN Crack → DCTM Pillar Map (single column, centered table or arrow map):**

Draw as a 2-column table (acceptable since it is a structured mapping, not a layout split):

```
     DEMGAN LIMITATION              →       DCTM SOLUTION
─────────────────────────────────────────────────────────────────
10 features (MI only)               →  P3: Hybrid MI+SHAP → 20 features
GAN instability / mode collapse     →  P1: Transformer Diffusion (MSE loss)
Class imbalance ignored             →  P2: SMOTE (cap 50k/class, k=5)
Evasion Rate metric only            →  P4: F1 + Precision + Recall + ROC-AUC
```

- Left column: red/orange text (problems)
- Right column: green/cyan text (solutions)
- Title above table: "Each Pillar Fixes One Specific Gap"

---

## SLIDE 6 — WORKFLOW

### Section name: `3. Workflow`

### RULE: Bullet points. One liner each. Diagram is the main visual.

### Content (bullets — brief, to support the diagram):

- Phase 1: Load raw CSVs → clean NaN/Infinity → MinMax normalise → 80-20 split.
- Phase 2: Compute MI and SHAP scores → hybrid ranking → select top 10 and top 20 features.
- Phase 3: Apply SMOTE on train split → train 10 IDS models on both feature sets (20 checkpoints).
- Phase 4: Train one Transformer Denoiser per attack class (100 epochs, AdamW, lr=1e-4).
- Phase 5: Partial forward diffusion to t=T/2 → reverse denoise → clamp immutable features every step.
- Phase 6: Feed adversarial samples to all 10 IDS → compute ER, F1, Precision, Recall.
- Phase 7: Augment train set with adversarial samples → retrain IDS → measure defense gain.
- Phase 8: Apply 2017-trained models to CICIDS2018 → external generalisation test.
- Phase 9: Generate final report with best ER, most vulnerable model, and retraining improvement.
- Entire pipeline runs with: `python main.py --phase all`

### Diagram:

**Diagram 3 — 10-Phase Pipeline Flowchart (vertical, single column, centered):**

Draw a vertical flowchart. Each box = one phase. Arrows connect top to bottom.
Use color bands to group phases:

```
┌─────────────────────────────────┐
│  Phase 1: Preprocess            │  ← Blue (Data Prep)
│  Clean → Normalize → Split      │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 2: Feature Engineering   │  ← Blue
│  MI + SHAP → top-10 & top-20    │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 3: IDS Training          │  ← Purple
│  SMOTE → 10 models × 2 sets     │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 4: Diffusion Training    │  ← Cyan
│  1 Denoiser per attack class    │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 5: Adversarial Attack    │  ← Cyan
│  t=T/2 noise → reverse → clamp  │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 6: Evasion Evaluation    │  ← Orange
│  ER + F1 + Precision + Recall   │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 7: Adversarial Retraining│  ← Orange
│  Augment train → retrain IDS    │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 8: External Validation   │  ← Green
│  2017 models → CICIDS2018 data  │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│  Phase 9: Report Generation     │  ← Green
│  final_report.txt               │
└─────────────────────────────────┘
```

- Caption below: `python main.py --phase all` runs all phases end-to-end.

---

## SLIDE 7 — ARCHITECTURE

### Section name: `4. Architecture`

### RULE: Bullet points. One liner each. Diagram is the main visual.

### Content (bullets):

- Transformer Denoiser is trained per attack class — one model per class, no shared conditioning.
- Input: noisy feature vector x_t (20 features) + timestep t (0 to 999).
- Timestep t is encoded via Sinusoidal Embedding (128-dim) → MLP → 256-dim vector.
- Feature vector is projected from 20-dim to 256-dim via Linear layer.
- Time embedding and feature projection are added element-wise.
- Fused vector passes through 4 Transformer Encoder Layers (pre-norm, GELU, nhead=4, FFN=512).
- Output is projected back to 20-dim — this is the predicted noise ε_θ(x_t, t).
- 10 IDS classifiers share a unified interface: train, predict, predict_proba, save, load.
- Classical models (CPU): Decision Tree, Naive Bayes, Logistic Regression, Random Forest, XGBoost, SVM.
- Deep models (GPU): MLP [256→128→64], CNN (1D), GRU-RNN, CNN-BiLSTM.

### Diagram:

**Diagram 4 — Transformer Denoiser Architecture (vertical flow, centered, single column):**

```
         INPUTS
    ┌─────────┐    ┌──────────┐
    │  x_t    │    │    t     │
    │(B × 20) │    │(timestep)│
    └────┬────┘    └────┬─────┘
         │              │
         │              ▼
         │   SinusoidalTimeEmbedding(128)
         │              │
         │              ▼
         │   MLP: Linear→SiLU→Linear (256-dim)
         │              │
         ▼              │
   Linear(20→256) ──(+)─┘   ← element-wise ADD
         │
         ▼
   unsqueeze → (B, 1, 256)
         │
    ┌────▼─────────────────────────┐
    │   TransformerEncoderLayer    │
    │   nhead=4 | FFN=512 | drop=0.1│  × 4
    │   pre-norm | GELU            │
    └────┬─────────────────────────┘
         │
   squeeze → (B, 256)
         │
         ▼
   Linear(256→20)
         │
         ▼
    ε_pred (B × 20)
    (predicted noise)
```

- Render with clean rounded boxes and arrows.
- Label the ×4 repetition on the Transformer block.
- Use cyan color for input/output. Dark navy for transformer block.
- Small parameter box beside diagram:
  ```
  model_dim  = 256
  nhead      = 4
  num_layers = 4
  dim_ff     = 512
  dropout    = 0.1
  params     ≈ 1.2M
  ```

---

## SLIDE 8 — METHODOLOGY

### Section name: `5. Methodology`

### RULE: Bullet points. One liner each. Diagrams support each sub-section.

### Content (bullets):

**Feature Engineering (P3):**
- MI ranking: compute mutual information between each feature and label on 100k samples.
- SHAP ranking: train LightGBM proxy on 50k samples → compute mean absolute SHAP values.
- Hybrid score: normalize both → 0.5 × MI_norm + 0.5 × SHAP_norm → select top 20.
- Immutable features (cannot be perturbed): Destination Port, URG Flag Count, CWE Flag Count.

**Diffusion Training (P1):**
- Forward process: x_t = √(ᾱ_t) · x_0 + √(1−ᾱ_t) · ε, where ε ~ N(0,I).
- Beta schedule: linear, β_start=1e-4, β_end=0.02, T=1000 steps.
- Training loss: MSE between true noise ε and predicted noise ε_θ(x_t, t).
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4, gradient clip=1.0, epochs=100.

**Adversarial Generation:**
- Step 1: Apply forward diffusion to real attack sample x_0 up to t=T/2 (partial noising).
- Step 2: Run reverse denoising chain from T/2 to 0 — clamp immutable features at every step.
- Step 3: Clip final output to [0,1] — ensures feasibility.
- Starting at T/2 (not T=1000) preserves attack-class identity in the sample.
- Per-step clamping keeps the full trajectory inside the protocol-valid feature space.

**Class Balancing (P2):**
- SMOTE applied on train split only — test split is kept naturally imbalanced.
- Parameters: k_neighbors=5, max 50,000 samples per class, skip classes with <6 samples.

### Diagrams:

**Diagram 5 — Feature Engineering Pipeline (top-down, single column, centered):**

```
All 78 CICIDS2017 Features
         │
    ┌────┴─────┐
    ▼           ▼
MI Ranking    SHAP Ranking
(100k rows)   (LightGBM, 50k rows)
MI score      mean(|SHAP value|)
    │              │
    ▼              ▼
MinMax         MinMax
Normalize      Normalize
    │              │
    └──────┬────────┘
           ▼
  Hybrid = 0.5·MI + 0.5·SHAP
           │
      Sort descending
           │
    ┌──────┴────────┐
    ▼               ▼
  Top 10         Top 20
(baseline)      (DCTM set)
```

---

**Diagram 6 — Adversarial Generation Timeline (horizontal arc, single column, centered):**

```
t = 0 ──────────────► t = 500 ──────────────► t = 0
  │                      │                       │
x_0                   x_{T/2}                 x_adv
(real attack)          (half-noised)          (adversarial)
                          │
         Forward diffusion │  Reverse denoising
         (add noise)       │  (clamp every step)
                           │
                    immutable features
                    clamped back at
                    every reverse step
```

Caption: "Partial noising at T/2 preserves attack identity. Per-step clamp ensures protocol validity."

---

## SLIDE 9 — DATASET

### Section name: `6. Dataset`

### RULE: Bullet points. One liner each.

### Content (bullets):

**CICIDS2017 (Primary):**
- Source: Canadian Institute for Cybersecurity, University of New Brunswick.
- Size: 8 CSV files, approximately 885 MB raw.
- Flows after cleaning: approximately 2.83 million network flow records.
- Classes: 15 total — 1 Benign + 14 attack types (DDoS, DoS, PortScan, Bot, XSS, SQL Injection, etc.).
- Role: Main training dataset and evasion evaluation benchmark.
- Split: 80% train / 20% test — stratified by class.

**CICIDS2018 (External Validation):**
- Source: Same — CIC, University of New Brunswick.
- Size: 3 CSV files, approximately 1.1 GB raw, ~2.4 million flows.
- Classes used: Benign, Bot, DoSHulk, SlowHTTPTest.
- Role: Held-out generalisation test — not used in any training phase.
- Feature mapping: matched by column name; missing columns zero-filled.

**Preprocessing (both datasets):**
- Remove rows with NaN or Infinity values.
- Strip leading/trailing whitespace from column names.
- Apply Min-Max normalisation → all features scaled to [0, 1].
- SMOTE on train split only: k=5, cap=50,000 samples/class, skip if class has <6 samples.

### Diagram:

**Diagram 7 — Class Distribution Before vs After SMOTE (horizontal bar chart, centered):**

Draw a grouped bar chart — two bars per class:
- Bar 1 (red): original sample count
- Bar 2 (green): after SMOTE

```
Class              Before SMOTE        After SMOTE
───────────────────────────────────────────────────
Benign             ████████ 2,271,319  ████ 50,000
DoS                █████ 251,712       ████ 50,000
DDoS               ████ 128,025        ████ 50,000
PortScan           ████ 158,804        ████ 50,000
SSH-Patator        █ 5,897             ████ 50,000
FTP-Patator        █ 7,935             ████ 50,000
Bot                . 1,956             ████ 50,000
XSS                . 625               ████ 50,000
Heartbleed         . 11                SKIPPED (<6 min threshold)
SQL Injection      . 21                SKIPPED (<6 min threshold)
```

Caption: "SMOTE applied on train split only. Test split kept naturally imbalanced for fair evaluation."

---

## SLIDE 10 — REFERENCES

### Section name: `7. References`

### RULE: Numbered citation list. Follow standard reference format.

### Content:

```
[1] Dawei Xu, Yue Lv, Min Wang, Baokun Zheng, Jian Zhao, Jiaxuan Yu,
    "DEMGAN: A Machine Learning-Based Intrusion Detection System Evasion Scheme,"
    Computers, Materials & Continua, Vol. 84, No. 1, pp. 1731–1746, 2025.
    DOI: 10.32604/cmc.2025.064833

[2] Jonathan Ho, Ajay Jain, Pieter Abbeel,
    "Denoising Diffusion Probabilistic Models,"
    NeurIPS, 2020. arXiv:2006.11239

[3] Ashish Vaswani et al.,
    "Attention Is All You Need,"
    NeurIPS, 2017.

[4] Martin Arjovsky, Soumith Chintala, Leon Bottou,
    "Wasserstein GAN,"
    ICML, 2017. arXiv:1701.07875

[5] Scott M. Lundberg, Su-In Lee,
    "A Unified Approach to Interpreting Model Predictions (SHAP),"
    NeurIPS, 2017.

[6] Nitesh Chawla et al.,
    "SMOTE: Synthetic Minority Over-sampling Technique,"
    JAIR, 2002.

[7] Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani,
    "Toward Generating a New Intrusion Detection Dataset,"
    ICISSP, 2018. [CICIDS2017 & CICIDS2018 datasets]
```

### Diagram: NONE

---

## SLIDE 11 — END SLIDE

### Section name: (none)

### Content (centered, large text):

```
Thank You

Any Questions?
```

Optional sub-line:
```
Team 06 | CSE-C | Keshav Memorial Engineering College
Contact: thrinadh.code@gmail.com
```

### Diagram: NONE

---

# COMPLETE SLIDE ORDER SUMMARY

| Slide No. | Section Name | Type | Has Diagram |
|---|---|---|---|
| 1 | Title Slide | Title | No |
| 2 | Index | List | No |
| 3 | Abstract | Paragraph (no bullets) | No |
| 4 | 1. Introduction | Bullets (one-liner each) | Yes — Diagram 1 |
| 5 | 2. Problem Statement | Bullets (one-liner each) | Yes — Diagram 2 |
| 6 | 3. Workflow | Bullets (one-liner each) | Yes — Diagram 3 |
| 7 | 4. Architecture | Bullets (one-liner each) | Yes — Diagram 4 |
| 8 | 5. Methodology | Bullets (one-liner each) | Yes — Diagrams 5 & 6 |
| 9 | 6. Dataset | Bullets (one-liner each) | Yes — Diagram 7 |
| 10 | 7. References | Numbered list | No |
| 11 | Thank You | End slide | No |

---

# DIAGRAM SUMMARY

| # | Slide | Diagram | Style |
|---|---|---|---|
| 1 | Introduction | IDS Evasion Concept — attacker → IDS → predicted benign | Block flow diagram |
| 2 | Problem Statement | DEMGAN limitation → DCTM solution mapping | 2-column mapping table |
| 3 | Workflow | 10-phase pipeline | Vertical flowchart, color-coded |
| 4 | Architecture | Transformer Denoiser layer-by-layer | Vertical component diagram |
| 5 | Methodology | Feature engineering fork — MI + SHAP → hybrid | Fork flow diagram |
| 6 | Methodology | Adversarial generation timeline — t=0 → T/2 → 0 | Horizontal arc diagram |
| 7 | Dataset | Class distribution before vs after SMOTE | Grouped horizontal bar chart |

---

# STRICT RULES — CHECKLIST BEFORE FINALISING PPT

- [ ] Every slide has `CSE-C-06` and project title in header
- [ ] Every slide has page number in footer
- [ ] Font is Times New Roman or Arial, minimum size 20, uniform throughout
- [ ] Abstract slide has ZERO bullets — paragraph text only
- [ ] All slides from Introduction onwards use bullet points — one liner per bullet
- [ ] No slide uses a two-column or split layout — single side only
- [ ] Index slide is slide 2 (immediately after title slide)
- [ ] End slide says "Thank You / Any Questions"
- [ ] Section names exactly match: Introduction, Problem Statement, Workflow, Architecture, Methodology, Dataset, References
- [ ] If any section needs a second slide, title it `Section Name (Contd.)`
- [ ] All numbers are exact: 97.42% ER, 2,271,319 Benign samples, T=1000, t=T/2=500
- [ ] Immutable features: CICIDS2017 → Destination Port, URG Flag Count, CWE Flag Count
- [ ] SMOTE is train-only — never applied to test split
- [ ] File saved as: `CSE-C-06.pptx`
