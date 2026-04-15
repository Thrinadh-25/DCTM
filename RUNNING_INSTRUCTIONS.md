# DCTM — Running Instructions

End-to-end guide for running the DCTM (Diffusion-Based Cyber Threat Modelling) pipeline.

---

## 0. Prerequisites

- **Python** 3.10+ (3.11 recommended)
- **pip** / **venv**
- Optional: **NVIDIA GPU + CUDA 12.1** (if missing, the code automatically falls back to CPU and will print a clear `CUDA FAILED — falling back to CPU` banner)
- **~5 GB free disk** for datasets + processed parquet + models

---

## 1. Download the datasets

Datasets are **not** included in the repo (they are in `.gitignore` because of their size).

### 1a. Download from Google Drive

> **Google Drive link:**
>
> https://drive.google.com/drive/folders/1jtk9pSWJqXlOVBWt3S3fBFee1D1Irse7?usp=sharing

The archive should contain two folders:

| Folder           | Files                                     | Approx. size | Role                                                                               |
| ---------------- | ----------------------------------------- | ------------ | ---------------------------------------------------------------------------------- |
| `cicids 2017/`   | 8 CSVs (Monday ... Friday)                | ~885 MB      | **Core dataset** — training IDS + diffusion, main evasion evaluation               |
| `cicids2018csv/` | 3 CSVs (Wednesday, Thursday x2, Feb 2018) | ~1.1 GB      | **External validation** — tests generalization of 2017-trained models on 2018 data |

### Note :

Download these exact folders from drive and place them in the datasets folder as shown below ( if not make sure to change paths in config.yaml file)

### 1b. Place them here

Extract so the final layout is:

```
DCTM/
└── datasets/
    ├── cicids 2017/                              ← note the SPACE in the folder name
    │   ├── Monday-WorkingHours.pcap_ISCX.csv
    │   ├── Tuesday-WorkingHours.pcap_ISCX.csv
    │   ├── Wednesday-workingHours.pcap_ISCX.csv
    │   ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    │   ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    │   ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    │   ├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
    │   └── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    │
    └── cicids2018csv/
        ├── Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
        ├── Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv
        └── Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv
```

If you placed them elsewhere, update the paths in `configs/config.yaml` under `paths.datasets`.

---

## 2. Install dependencies

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or Git Bash / WSL:
source .venv/Scripts/activate

pip install -r requirements.txt
```

### 2a. (Optional but recommended) Install PyTorch with CUDA

The default `torch` in `requirements.txt` is CPU-only. For GPU:

```bash
# CUDA 12.1 example — check https://pytorch.org for the wheel matching your CUDA version
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA isn't usable you'll see `CUDA FAILED — falling back to CPU` the first time you run `main.py` — everything still works, just slower.

---

## 3. Run the pipeline

All commands run from the repo root (`DCTM/`).

### Fast path — run everything

```bash
python main.py --phase all
```

This executes every phase in order on CICIDS2017 by default.

### Ordered path — run each phase manually

Run these **in order**. Each phase caches its output so re-runs are fast.

| #   | Command                                                | What it does                                                                    | Outputs                                                                                                     |
| --- | ------------------------------------------------------ | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 1   | `python main.py --phase preprocess`                    | Loads both datasets, cleans NaN/Inf, encodes labels, stratified 80/20 split     | `data/processed/*.parquet`, `data/splits/*.parquet`                                                         |
| 2   | `python main.py --phase features --dataset cicids2017` | Computes MI + SHAP rankings, builds hybrid top-20 feature list                  | `configs/features_baseline.json` (top-10), `configs/features_dctm.json` (top-20), plots in `visualization/` |
| 3   | `python main.py --phase train --feature-set both`      | SMOTE + trains all 10 IDS models on both 10-feat and 20-feat sets               | `data/models/*.pkl`, `data/models/*.pt`, `evaluation/results/baseline_*.csv`                                |
| 4   | `python main.py --phase diffusion --feature-set dctm`  | Trains one Transformer diffusion denoiser per attack class                      | `data/models/diffusion_dctm_class*.pt`, `data/models/diffusion_index_dctm.json`                             |
| 5   | `python main.py --phase attack --feature-set dctm`     | Partial noising + constrained reverse denoising → generates adversarial samples | `data/adversarial/adv_samples_cicids2017_dctm.parquet`                                                      |
| 6   | `python main.py --phase evaluate --feature-set dctm`   | Evaluates every IDS model on clean vs adversarial data (computes evasion rate)  | `evaluation/results/evasion_dctm.csv`, `visualization/evasion_comparison.png`                               |
| 7   | `python main.py --phase retrain --feature-set dctm`    | Augments training set with adv samples, retrains, re-evaluates                  | `evaluation/results/retrained_dctm.csv`, `visualization/retraining_improvement.png`                         |
| 8   | `python main.py --phase external --feature-set dctm`   | External validation — tests 2017-trained models on CICIDS2018                   | `evaluation/results/external_eval_2018_dctm.csv`                                                            |
| 9   | `python main.py --phase report --feature-set dctm`     | Writes the final summary report                                                 | `evaluation/final_report.txt`                                                                               |

---

## 4. Expected runtimes (rough)

On a modest GPU (e.g., RTX 3060) + 16 GB RAM:

| Phase                                  | CPU          | GPU          |
| -------------------------------------- | ------------ | ------------ |
| preprocess                             | 2–4 min      | 2–4 min      |
| features                               | 5–10 min     | 5–10 min     |
| train (all 10 models)                  | 30–60 min    | 10–20 min    |
| diffusion (per class × N classes)      | hours        | 20–60 min    |
| attack                                 | 5–15 min     | 1–5 min      |
| evaluate / retrain / external / report | < 5 min each | < 5 min each |

**Tip:** For a faster smoke test, edit `configs/config.yaml`:

- `preprocessing.max_rows_per_file: 50000` (subsamples each CSV)
- `diffusion.epochs: 5`
- `deep_learning.epochs: 3`

---

## 5. Verifying a successful run

After `phase all`, check:

- [ ] `data/processed/cicids2017.parquet` and `cicids2018.parquet` exist
- [ ] `configs/features_baseline.json` and `configs/features_dctm.json` exist
- [ ] `data/models/` contains `.pkl` files for classical models and `.pt` files for deep/diffusion models
- [ ] `data/adversarial/adv_samples_cicids2017_dctm.parquet` exists
- [ ] `evaluation/results/evasion_dctm.csv` is populated
- [ ] `evaluation/final_report.txt` exists and prints the key metrics
- [ ] `logs/run.log` has no ERROR-level messages

---

## 6. CLI reference

```bash
python main.py --phase {all|preprocess|features|train|diffusion|attack|evaluate|retrain|external|report}
               --dataset {cicids2017|cicids2018}          # default: cicids2017
               --feature-set {baseline|dctm|both}         # default: dctm
               --config configs/config.yaml               # default shown
```

---

## 7. Troubleshooting

| Symptom                                           | Fix                                                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `CUDA FAILED — falling back to CPU`               | Expected if you don't have CUDA / correct PyTorch wheel. Training still works, just slower. Install GPU wheel per §2a. |
| `FileNotFoundError: No CSV files in datasets/...` | Dataset folders not placed correctly. See §1b.                                                                         |
| SHAP / LightGBM OOM on large data                 | Reduce `feature_engineering.shap_sample_size` and `mi_sample_size` in `configs/config.yaml`.                           |
| SVM training hangs                                | Reduce `classical.svm.max_train_samples` in `configs/config.yaml` (default 50 000).                                    |
| Out of GPU memory during diffusion                | Lower `diffusion.batch_size` in `configs/config.yaml`.                                                                 |
| `No diffusion index at ...` when running `attack` | Run `--phase diffusion` first.                                                                                         |

---

## 8. Reproducibility

Seed is fixed at `42` (see `configs/config.yaml → project.seed` and `utils/seed.py`). Splits, SMOTE, feature ranking, and diffusion are all deterministic under the same seed. Cached parquet / JSON files short-circuit re-runs — delete them to force recomputation.
