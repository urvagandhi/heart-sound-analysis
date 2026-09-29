"""
Review 3 — Evaluation
=====================
Test-set metrics + plots:
* Accuracy, balanced accuracy, precision, recall, F1 (per-class + macro)
* AUC-ROC and AUC-PR (binary case)
* Confusion matrix (counts + row-normalised)
* ROC and PR curves
* Per-recording prediction table (saved as CSV)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CLASS_NAMES


@torch.no_grad()
def evaluate_on_test(model: torch.nn.Module,
                     test_loader: DataLoader,
                     device: str,
                     class_names: List[str] = CLASS_NAMES
                    ) -> Dict[str, object]:
    """Return per-sample predictions and aggregate metrics."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_score,
        recall_score, roc_auc_score, average_precision_score,
        classification_report,
    )

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for X, y in tqdm(test_loader, desc="  Test"):
        X = X.to(device, non_blocking=True)
        logits = model(X)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(y.numpy())
        all_probs.append(probs)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs  = np.concatenate(all_probs)

    n_classes = probs.shape[1]

    metrics: Dict[str, object] = {
        "accuracy":          accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision_macro":   precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":      recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro":          f1_score(labels, preds, average="macro", zero_division=0),
        "precision_per_class": precision_score(labels, preds, average=None, zero_division=0).tolist(),
        "recall_per_class":    recall_score(labels, preds, average=None, zero_division=0).tolist(),
        "f1_per_class":        f1_score(labels, preds, average=None, zero_division=0).tolist(),
    }

    # AUC-ROC / AUC-PR
    if n_classes == 2:
        metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1])
        metrics["auc_pr"]  = average_precision_score(labels, probs[:, 1])
    else:
        metrics["auc_roc"] = roc_auc_score(labels, probs, multi_class="ovr",
                                           average="macro")
        # PR-AUC macro across classes
        ap = []
        for c in range(n_classes):
            ap.append(average_precision_score((labels == c).astype(int),
                                              probs[:, c]))
        metrics["auc_pr"] = float(np.mean(ap))

    print("\n" + "=" * 60)
    print(f"  Test set ({len(labels)} samples) — {len(class_names)} classes")
    print("=" * 60)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro precision   : {metrics['precision_macro']:.4f}")
    print(f"  Macro recall      : {metrics['recall_macro']:.4f}")
    print(f"  Macro F1          : {metrics['f1_macro']:.4f}")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR            : {metrics['auc_pr']:.4f}")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=class_names,
                                digits=4, zero_division=0))

    metrics["preds"]  = preds
    metrics["labels"] = labels
    metrics["probs"]  = probs
    return metrics


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics: Dict, class_names: List[str],
                          save_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm      = confusion_matrix(metrics["labels"], metrics["preds"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Confusion matrix (counts)", "Confusion matrix (row-normalised)"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5, cbar=True, square=True)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def plot_roc_pr_curves(metrics: Dict, class_names: List[str],
                       save_path: Path) -> None:
    """ROC and Precision-Recall curves (binary only)."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    import matplotlib.pyplot as plt

    labels = metrics["labels"]; probs = metrics["probs"]
    if probs.shape[1] != 2:
        print("  ROC/PR plots are binary-only; skipping.")
        return

    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    prec, rec, _ = precision_recall_curve(labels, probs[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, color="#1F3A5F", linewidth=2,
                 label=f"AUC = {metrics['auc_roc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", alpha=0.7)
    axes[0].set_title("ROC curve", fontweight="bold")
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(rec, prec, color="#E24B4A", linewidth=2,
                 label=f"AP = {metrics['auc_pr']:.4f}")
    base = float((labels == 1).mean())
    axes[1].axhline(base, ls="--", color="grey", alpha=0.7,
                    label=f"Base rate = {base:.3f}")
    axes[1].set_title("Precision-Recall curve", fontweight="bold")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def save_predictions_csv(metrics: Dict, test_df: pd.DataFrame,
                         class_names: List[str], save_path: Path) -> None:
    out = test_df[["unique_id", "filename", "subset", "label"]].copy()
    out["pred"]       = metrics["preds"]
    out["pred_name"]  = [class_names[p] for p in metrics["preds"]]
    out["true_name"]  = [class_names[t] for t in metrics["labels"]]
    out["correct"]    = out["pred"] == out["label"]
    for i, name in enumerate(class_names):
        out[f"prob_{name}"] = metrics["probs"][:, i]
    out.to_csv(save_path, index=False)
    print(f"  Saved: {save_path}")
