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
    {"Paper": "Ren et al. (2022)",             "Dataset": "HSS",
     "Classes": 3, "Accuracy": "— (HSS)", "F1": "—", "AUC": "—",  "XAI": "Frame attention (UAR 51.2)"},
    {"Paper": "Alrabie & Barnawi (2025)",      "Dataset": "HeartWave (seg)",
     "Classes": 4, "Accuracy": 97.3, "F1": "—",   "AUC": "—",  "XAI": "Grad-CAM"},
    {"Paper": "Padhy et al. (2025)",           "Dataset": "PhysioNet 2016 subset (2,400, bal)",
     "Classes": 5, "Accuracy": 99.15,"F1": "—",   "AUC": 0.99, "XAI": "Saliency"},
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
    """Pick a mix of correct and misclassified test samples for XAI plots.
    Selects confident correct cases per class (e.g., confident Normal and
    confident Abnormal) plus misclassified cases (False Positives and False Negatives).
    """
    labels = np.asarray(metrics["labels"])
    preds  = np.asarray(metrics["preds"])
    probs  = np.asarray(metrics["probs"])

    rng = np.random.default_rng(rng_seed)
    picks: List[int] = []

    # 1. Confident correct per class
    unique_classes = np.unique(labels)
    n_per_class = max(1, n_correct // len(unique_classes)) if len(unique_classes) > 0 else 1
    for c in unique_classes:
        c_correct = np.where((labels == c) & (preds == c))[0]
        if len(c_correct) > 0:
            c_conf = probs[c_correct, c]
            order = c_correct[np.argsort(-c_conf)]
            picks.extend(order[:n_per_class].tolist())

    # 2. Misclassified cases (errors)
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) > 0:
        fp_idx = np.where((labels == 0) & (preds == 1))[0]
        fn_idx = np.where((labels == 1) & (preds == 0))[0]
        wrong_picks = []
        if len(fp_idx) > 0 and len(fn_idx) > 0 and n_wrong >= 2:
            wrong_picks.append(int(rng.choice(fp_idx)))
            wrong_picks.append(int(rng.choice(fn_idx)))
            rem = n_wrong - 2
            if rem > 0:
                avail = [i for i in wrong_idx if i not in wrong_picks]
                if avail:
                    wrong_picks.extend(rng.choice(avail, size=min(rem, len(avail)), replace=False).tolist())
        else:
            wrong_picks = rng.choice(wrong_idx, size=min(n_wrong, len(wrong_idx)), replace=False).tolist()
        picks.extend(wrong_picks)

    return picks


# ─────────────────────────────────────────────────────────────
# METRICS SUMMARY PRINTING
# ─────────────────────────────────────────────────────────────

def metrics_to_dict(metrics: Dict) -> Dict[str, float]:
    """Strip out the heavy numpy arrays — keep only scalars for JSON dump."""
    keep = ("accuracy", "balanced_accuracy", "precision_macro",
            "recall_macro", "f1_macro", "auc_roc", "auc_pr",
            "sensitivity", "specificity")
    return {k: float(metrics[k]) for k in keep if k in metrics}
