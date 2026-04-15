"""Final summary report — text file with key metrics."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

_log = get_logger("report")


def write_final_report(
    baseline_eval_csv: str,
    retrained_eval_csv: str | None,
    cfg: dict,
    out_path: str = "evaluation/final_report.txt",
    per_set_summary: dict | None = None,
) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append(" DCTM — Final Report ")
    lines.append("=" * 60)

    ref = cfg["evaluation"].get("demgan_reference_er", 0.9742)

    if os.path.exists(baseline_eval_csv):
        df = pd.read_csv(baseline_eval_csv)
        if len(df):
            best_f1_row = df.loc[df["clean_f1"].idxmax()]
            worst_er_row = df.loc[df["adv_er"].idxmax()]
            avg_er = df["adv_er"].mean()
            lines.append("")
            lines.append(f"Best clean F1:          {best_f1_row['model']} — F1={best_f1_row['clean_f1']:.4f}")
            lines.append(f"Most vulnerable (ER):   {worst_er_row['model']} — ER={worst_er_row['adv_er']:.4f}")
            lines.append(f"Avg adversarial ER:     {avg_er:.4f}")
            lines.append(f"DEMGAN reference ER:    {ref:.4f}")

    if retrained_eval_csv and os.path.exists(retrained_eval_csv):
        dr = pd.read_csv(retrained_eval_csv)
        base = pd.read_csv(baseline_eval_csv)
        merged = base.merge(dr, on="model", suffixes=("_base", "_retrain"))
        if len(merged):
            merged["delta_f1"] = merged["clean_f1_retrain"] - merged["clean_f1_base"]
            merged["delta_er"] = merged["adv_er_base"] - merged["adv_er_retrain"]
            best_d = merged.loc[merged["delta_f1"].idxmax()]
            best_er = merged.loc[merged["delta_er"].idxmax()]
            lines.append("")
            lines.append(f"Largest ΔF1 after retrain:  {best_d['model']}  Δ={best_d['delta_f1']:+.4f}")
            lines.append(f"Largest ΔER  after retrain: {best_er['model']}  Δ={best_er['delta_er']:+.4f}")

    if per_set_summary:
        lines.append("")
        lines.append("Per feature-set average F1:")
        for tag, avg in per_set_summary.items():
            lines.append(f"  {tag:<18} Avg F1 = {avg:.4f}")

    lines.append("")
    lines.append("=" * 60)
    txt = "\n".join(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    _log.info(f"Final report -> {out_path}")
    print(txt)
