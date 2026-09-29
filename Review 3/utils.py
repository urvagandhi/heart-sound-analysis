"""
Review 3 — Small utilities
==========================
Plot helpers and a comparison table generator for the final report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# COMPARISON TABLE (versus literature)
# ─────────────────────────────────────────────────────────────

LITERATURE = [
    {"Paper": "Li et al. (2020)",              "Dataset": "PhysioNet 2016",
     "Classes": 2, "Accuracy": 86.8, "F1": "—",   "AUC": "—",  "XAI": "None"},
    {"Paper": "Ren et al. (2021)",             "Dataset": "HSS",
     "Classes": 3, "Accuracy": "—",  "F1": "—",   "AUC": "—",  "XAI": "Attn"},
    {"Paper": "Alrabie & Barnawi (2025)",      "Dataset": "HeartWave",
     "Classes": 4, "Accuracy": 97.3, "F1": "—",   "AUC": "—",  "XAI": "Grad-CAM"},
    {"Paper": "Padhy et al. (2025)",           "Dataset": "PhysioNet 2016",
     "Classes": 5, "Accuracy": 99.15,"F1": "—",   "AUC": 0.99, "XAI": "Grad-CAM"},
]


def build_comparison_table(metrics: Dict, save_path: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = list(LITERATURE) + [{
        "Paper":    "Ours (Review 3)",
        "Dataset":  "PhysioNet 2016",
        "Classes":  2,
        "Accuracy": round(metrics["accuracy"]  * 100, 2),
        "F1":       round(metrics["f1_macro"], 4),
        "AUC":      round(metrics["auc_roc"],  4),
        "XAI":      "Grad-CAM + SHAP + Attn",
    }]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for col in range(len(df.columns)):
        table[(len(df), col)].set_facecolor("#E6F1FB")
        table[(len(df), col)].set_text_props(fontweight="bold")

    plt.title("Comparison with literature", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")
    return df


# ─────────────────────────────────────────────────────────────
# SAMPLE PICKER FOR XAI VISUALISATION
# ─────────────────────────────────────────────────────────────

def pick_explainable_indices(metrics: Dict, n_correct: int = 2,
                             n_wrong: int = 2,
                             rng_seed: int = 42) -> List[int]:
    """Pick a mix of correct and misclassified test samples for XAI plots."""
    labels = metrics["labels"]; preds = metrics["preds"]
    probs  = metrics["probs"]

    correct_mask = preds == labels
    correct_idx  = np.where(correct_mask)[0]
    wrong_idx    = np.where(~correct_mask)[0]

    rng = np.random.default_rng(rng_seed)
    picks: List[int] = []

    if len(correct_idx) > 0:
        # Sort correct by margin (highest confidence first) and sample evenly
        if probs.shape[1] == 2:
            margins = np.abs(probs[correct_idx, 1] - probs[correct_idx, 0])
        else:
            top2 = np.sort(probs[correct_idx], axis=1)
            margins = top2[:, -1] - top2[:, -2]
        order = correct_idx[np.argsort(-margins)]
        picks.extend(order[:n_correct].tolist())

    if len(wrong_idx) > 0:
        picks.extend(rng.choice(wrong_idx,
                                size=min(n_wrong, len(wrong_idx)),
                                replace=False).tolist())

    return picks


# ─────────────────────────────────────────────────────────────
# METRICS SUMMARY PRINTING
# ─────────────────────────────────────────────────────────────

def metrics_to_dict(metrics: Dict) -> Dict[str, float]:
    """Strip out the heavy numpy arrays — keep only scalars for JSON dump."""
    keep = ("accuracy", "balanced_accuracy", "precision_macro",
            "recall_macro", "f1_macro", "auc_roc", "auc_pr")
    return {k: float(metrics[k]) for k in keep if k in metrics}
