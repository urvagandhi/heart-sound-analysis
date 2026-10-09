"""Calibration evaluation, clinical operating point selection, and SQI reliability audit."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def compute_ece(probs: np.ndarray,
                labels: np.ndarray,
                n_bins: int = 15) -> Tuple[float, Dict[str, List[float]]]:
    """Compute Expected Calibration Error (ECE) across equal-width confidence bins.

    Args:
        probs: Array of shape (N, C) containing predicted class probabilities.
        labels: Ground-truth class labels (N,).
        n_bins: Number of equal-width bins (default: 15).

    Returns:
        Tuple of (ece_value, reliability_diagram_dict).
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels)

    bin_centers = []
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences < high) if i < n_bins - 1 else (confidences >= low) & (confidences <= high)
        count = int(np.sum(mask))
        center = float((low + high) / 2.0)
        bin_centers.append(center)
        bin_counts.append(count)

        if count > 0:
            b_acc = float(np.mean(accuracies[mask]))
            b_conf = float(np.mean(confidences[mask]))
            ece += (count / n_total) * abs(b_acc - b_conf)
            bin_accs.append(b_acc)
            bin_confs.append(b_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(center)

    reliability = {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }
    return float(ece), reliability


def select_operating_point(val_probs: np.ndarray,
                           val_labels: np.ndarray,
                           test_probs: np.ndarray,
                           test_labels: np.ndarray,
                           target_sensitivity: float = 0.90) -> Dict[str, float]:
    """Select the clinical threshold on validation set (sensitivity >= target), then evaluate on test.

    The target threshold is chosen as the largest threshold with validation sensitivity >= 0.90.
    At this threshold, test sensitivity, specificity, and balanced accuracy are computed.
    """
    val_pos_probs = val_probs[:, 1]
    test_pos_probs = test_probs[:, 1]

    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh = 0.5
    found = False

    # Find the largest threshold achieving target sensitivity on validation set
    for t in sorted(thresholds, reverse=True):
        val_preds = (val_pos_probs >= t).astype(int)
        cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
        # cm: [[TN, FP], [FN, TP]]
        tp, fn = cm[1, 1], cm[1, 0]
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if sens >= target_sensitivity:
            best_thresh = float(t)
            found = True
            break

    if not found:
        # Fall back to threshold maximizing sensitivity
        best_sens = -1.0
        for t in thresholds:
            val_preds = (val_pos_probs >= t).astype(int)
            cm = confusion_matrix(val_labels, val_preds, labels=[0, 1])
            tp, fn = cm[1, 1], cm[1, 0]
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if sens > best_sens:
                best_sens = sens
                best_thresh = float(t)

    # Evaluate on test set at the selected threshold
    test_preds_at_thresh = (test_pos_probs >= best_thresh).astype(int)
    cm_test = confusion_matrix(test_labels, test_preds_at_thresh, labels=[0, 1])
    tn, fp, fn, tp = cm_test[0, 0], cm_test[0, 1], cm_test[1, 0], cm_test[1, 1]

    test_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    test_ba = float(balanced_accuracy_score(test_labels, test_preds_at_thresh))

    return {
        "selected_threshold": best_thresh,
        "target_sensitivity": target_sensitivity,
        "test_sensitivity": test_sens,
        "test_specificity": test_spec,
        "test_balanced_accuracy": test_ba,
    }


def evaluate_sqi_breakdown(test_df: pd.DataFrame,
                           test_preds: np.ndarray,
                           test_labels: np.ndarray,
                           search_dirs: Sequence[Path]) -> Dict[str, Any]:
    """Find REFERENCE*SQI*.csv and evaluate clean vs noisy performance breakdown."""
    sqi_file: Optional[Path] = None
    for d in search_dirs:
        if d and Path(d).exists():
            for root, _, files in Path(d).walk():
                for f in files:
                    if fnmatch.fnmatch(f.upper(), "*REFERENCE*SQI*.CSV") or fnmatch.fnmatch(f.upper(), "*SQI*.CSV"):
                        sqi_file = root / f
                        break
                if sqi_file:
                    break
        if sqi_file:
            break

    if not sqi_file or not sqi_file.exists():
        print("  [Calibration/SQI] No SQI reference file matching REFERENCE*SQI*.csv found; skipping clean vs noisy breakdown.")
        return {"sqi_status": "skipped", "message": "No SQI file found"}

    print(f"  [Calibration/SQI] Found SQI annotations at {sqi_file}")
    try:
        sqi_df = pd.read_csv(sqi_file, header=None)
        # Expected format: filename, quality_flag (1=clean, 0=noisy)
        sqi_map = dict(zip(sqi_df[0].astype(str), sqi_df[1].astype(int)))
        correct = (test_preds == test_labels).astype(int)

        clean_correct = []
        noisy_correct = []

        for idx, r in enumerate(test_df.itertuples()):
            fn = str(r.filename)
            if fn in sqi_map:
                if sqi_map[fn] == 1:
                    clean_correct.append(correct[idx])
                else:
                    noisy_correct.append(correct[idx])

        return {
            "sqi_status": "evaluated",
            "sqi_file": str(sqi_file),
            "clean_accuracy": float(np.mean(clean_correct)) if clean_correct else None,
            "noisy_accuracy": float(np.mean(noisy_correct)) if noisy_correct else None,
            "n_clean": len(clean_correct),
            "n_noisy": len(noisy_correct),
        }
    except Exception as e:
        print(f"  [Calibration/SQI] Error parsing SQI file: {e}")
        return {"sqi_status": "error", "error": str(e)}


def run_calibration_audit(val_probs: np.ndarray,
                          val_labels: np.ndarray,
                          test_probs: np.ndarray,
                          test_labels: np.ndarray,
                          test_df: pd.DataFrame,
                          raw_dir: Optional[Path] = None,
                          annotations_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute complete calibration and operating point audit."""
    ece, reliability = compute_ece(test_probs, test_labels, n_bins=15)
    operating_point = select_operating_point(val_probs, val_labels, test_probs, test_labels, target_sensitivity=0.90)

    # Per-subset accuracy
    preds = np.argmax(test_probs, axis=1)
    correct = (preds == test_labels).astype(int)
    per_subset_acc: Dict[str, float] = {}
    for sub in sorted(test_df["subset"].unique()):
        mask = (test_df["subset"] == sub).values
        per_subset_acc[sub] = float(np.mean(correct[mask])) if np.any(mask) else 0.0

    # Clean vs noisy SQI analysis
    search_dirs = [p for p in [raw_dir, annotations_dir] if p is not None]
    sqi_result = evaluate_sqi_breakdown(test_df, preds, test_labels, search_dirs)

    return {
        "ece": ece,
        "reliability": reliability,
        "operating_point": operating_point,
        "per_subset_accuracy": per_subset_acc,
        "sqi_analysis": sqi_result,
    }
