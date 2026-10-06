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


def compute_and_save_extended_metrics(metrics: Dict,
                                      test_df: pd.DataFrame,
                                      train_df: Optional[pd.DataFrame],
                                      output_dir: Path,
                                      n_resamples: int = 2000,
                                      seed: int = 42) -> Dict[str, object]:
    """Compute and save:
    1) output/per_subset_metrics.csv (N, normal, abnormal, accuracy, normal accuracy, abnormal accuracy per subset)
    2) output/test_metrics_ci.json (95% bootstrap intervals, 2000 resamples, seed 42,
       for accuracy, balanced accuracy, macro F1, AUC-ROC and AUC-PR, plus duplicated vs unique accuracy).
    """
    import json
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(metrics["labels"])
    preds  = np.asarray(metrics["preds"])
    probs  = np.asarray(metrics["probs"])

    # 1. Per-subset metrics CSV
    subsets = sorted(test_df["subset"].unique())
    rows = []
    for sub in subsets:
        mask = (test_df["subset"] == sub).values
        sub_labels = labels[mask]
        sub_preds  = preds[mask]
        N = int(len(sub_labels))
        n_norm = int(np.sum(sub_labels == 0))
        n_abn  = int(np.sum(sub_labels == 1))
        acc = float(np.mean(sub_preds == sub_labels)) if N > 0 else 0.0
        norm_acc = float(np.mean(sub_preds[sub_labels == 0] == 0)) if n_norm > 0 else np.nan
        abn_acc  = float(np.mean(sub_preds[sub_labels == 1] == 1)) if n_abn > 0 else np.nan
        rows.append({
            "subset": sub,
            "N": N,
            "normal": n_norm,
            "abnormal": n_abn,
            "accuracy": round(acc * 100, 2),
            "normal_accuracy": round(norm_acc * 100, 2) if not np.isnan(norm_acc) else None,
            "abnormal_accuracy": round(abn_acc * 100, 2) if not np.isnan(abn_acc) else None,
        })

    # Overall / All row
    N_all = len(labels)
    n_norm_all = int(np.sum(labels == 0))
    n_abn_all  = int(np.sum(labels == 1))
    acc_all = float(np.mean(preds == labels))
    norm_acc_all = float(np.mean(preds[labels == 0] == 0)) if n_norm_all > 0 else 0.0
    abn_acc_all  = float(np.mean(preds[labels == 1] == 1)) if n_abn_all > 0 else 0.0
    rows.append({
        "subset": "All",
        "N": N_all,
        "normal": n_norm_all,
        "abnormal": n_abn_all,
        "accuracy": round(acc_all * 100, 2),
        "normal_accuracy": round(norm_acc_all * 100, 2),
        "abnormal_accuracy": round(abn_acc_all * 100, 2),
    })

    df_subset = pd.DataFrame(rows)
    subset_csv_path = output_dir / "per_subset_metrics.csv"
    df_subset.to_csv(subset_csv_path, index=False)
    print(f"  Saved: {subset_csv_path}")

    # 2. Leakage analysis: duplicated vs unique test rows
    leakage_stats = {}
    if train_df is not None:
        train_pairs = set(zip(train_df["filename"], train_df["label"]))
        is_dup = np.array([(fn, lab) in train_pairs for fn, lab in zip(test_df["filename"], test_df["label"])])
        n_dup = int(np.sum(is_dup))
        n_uniq = int(np.sum(~is_dup))
        acc_dup = float(np.mean(preds[is_dup] == labels[is_dup])) if n_dup > 0 else None
        acc_uniq = float(np.mean(preds[~is_dup] == labels[~is_dup])) if n_uniq > 0 else None
        dup_correct = int(np.sum(preds[is_dup] == labels[is_dup])) if n_dup > 0 else 0
        uniq_correct = int(np.sum(preds[~is_dup] == labels[~is_dup])) if n_uniq > 0 else 0
        leakage_stats = {
            "duplicated_in_train": {
                "count": n_dup,
                "correct": dup_correct,
                "accuracy": acc_dup,
                "accuracy_pct": round(acc_dup * 100, 2) if acc_dup is not None else None,
            },
            "unique": {
                "count": n_uniq,
                "correct": uniq_correct,
                "accuracy": acc_uniq,
                "accuracy_pct": round(acc_uniq * 100, 2) if acc_uniq is not None else None,
            }
        }
        print(f"  Duplicated in train: {n_dup} (acc: {leakage_stats['duplicated_in_train']['accuracy_pct']}%)")
        print(f"  Unique test rows: {n_uniq} (acc: {leakage_stats['unique']['accuracy_pct']}%)")

    # 3. Bootstrap intervals (2000 resamples, seed 42)
    n = len(labels)
    rng = np.random.RandomState(seed)
    acc_list, bal_acc_list, f1_list, roc_list, pr_list = [], [], [], [], []

    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        y_true = labels[idx]
        y_pred = preds[idx]
        acc_list.append(accuracy_score(y_true, y_pred))
        bal_acc_list.append(balanced_accuracy_score(y_true, y_pred))
        f1_list.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if len(np.unique(y_true)) > 1:
            roc_list.append(roc_auc_score(y_true, probs[idx, 1]))
            pr_list.append(average_precision_score(y_true, probs[idx, 1]))

    ci_dict = {
        "n_resamples": n_resamples,
        "seed": seed,
        "accuracy": {
            "point": float(metrics["accuracy"]),
            "ci_95": [float(np.percentile(acc_list, 2.5)), float(np.percentile(acc_list, 97.5))],
            "ci_95_pct": [round(float(np.percentile(acc_list, 2.5)) * 100, 2),
                          round(float(np.percentile(acc_list, 97.5)) * 100, 2)],
        },
        "balanced_accuracy": {
            "point": float(metrics["balanced_accuracy"]),
            "ci_95": [float(np.percentile(bal_acc_list, 2.5)), float(np.percentile(bal_acc_list, 97.5))],
            "ci_95_pct": [round(float(np.percentile(bal_acc_list, 2.5)) * 100, 2),
                          round(float(np.percentile(bal_acc_list, 97.5)) * 100, 2)],
        },
        "macro_f1": {
            "point": float(metrics["f1_macro"]),
            "ci_95": [float(np.percentile(f1_list, 2.5)), float(np.percentile(f1_list, 97.5))],
            "ci_95_pct": [round(float(np.percentile(f1_list, 2.5)) * 100, 2),
                          round(float(np.percentile(f1_list, 97.5)) * 100, 2)],
        },
        "auc_roc": {
            "point": float(metrics["auc_roc"]),
            "ci_95": [float(np.percentile(roc_list, 2.5)), float(np.percentile(roc_list, 97.5))],
        },
        "auc_pr": {
            "point": float(metrics["auc_pr"]),
            "ci_95": [float(np.percentile(pr_list, 2.5)), float(np.percentile(pr_list, 97.5))],
        },
        "leakage_analysis": leakage_stats,
    }

    ci_json_path = output_dir / "test_metrics_ci.json"
    with open(ci_json_path, "w", encoding="utf-8") as f:
        json.dump(ci_dict, f, indent=2)
    print(f"  Saved: {ci_json_path}")

    return {"per_subset": rows, "ci": ci_dict}
