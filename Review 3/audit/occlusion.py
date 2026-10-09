"""Frequency band occlusion audit with contiguous random-window control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from audit.common import get_band_row_ranges, to_norm, to_unit
from config import AudioConfig
from model import HeartSoundModel


@torch.no_grad()
def evaluate_occluded_loader(model: HeartSoundModel,
                             loader: DataLoader,
                             device: str,
                             band_rows: Optional[Tuple[int, int]] = None,
                             mode: str = "none") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score a DataLoader under frequency row occlusion.

    Args:
        model: Trained HeartSoundModel.
        loader: Test DataLoader.
        device: Execution device.
        band_rows: Tuple of (row_start, row_end) to occlude or keep.
        mode: Occlusion mode in {'none', 'mask', 'keep_only'}.

    Returns:
        Tuple of (probabilities, predictions, ground_truth_labels).
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for X, y in loader:
        X = X.to(device)
        if mode != "none" and band_rows is not None:
            r0, r1 = band_rows
            u = to_unit(X)
            if mode == "mask":
                u_mod = u.clone()
                u_mod[:, :, r0:r1, :] = 0.0
            elif mode == "keep_only":
                u_mod = torch.zeros_like(u)
                u_mod[:, :, r0:r1, :] = u[:, :, r0:r1, :]
            else:
                raise ValueError(f"Unknown occlusion mode: {mode}")
            X = to_norm(u_mod)

        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y.numpy())

    return np.concatenate(all_probs), np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics_and_deltas(probs: np.ndarray,
                               preds: np.ndarray,
                               labels: np.ndarray,
                               test_df: pd.DataFrame,
                               base_ba: float) -> Dict[str, Any]:
    """Compute AUC, BA, delta BA, sensitivity, specificity, and per-subset BA."""
    ba = float(balanced_accuracy_score(labels, preds))
    auc = float(roc_auc_score(labels, probs[:, 1])) if probs.shape[1] == 2 else 0.0
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    subset_ba: Dict[str, float] = {}
    for sub in sorted(test_df["subset"].unique()):
        mask = (test_df["subset"] == sub).values
        if np.any(mask):
            sub_preds = preds[mask]
            sub_labels = labels[mask]
            if len(np.unique(sub_labels)) > 1:
                subset_ba[sub] = float(balanced_accuracy_score(sub_labels, sub_preds))
            else:
                subset_ba[sub] = float(np.mean(sub_preds == sub_labels))

    return {
        "balanced_accuracy": ba,
        "delta_balanced_accuracy": float(ba - base_ba),
        "auc_roc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "per_subset_ba": subset_ba,
    }


def run_band_occlusion_audit(model: HeartSoundModel,
                             test_loader: DataLoader,
                             test_df: pd.DataFrame,
                             device: str,
                             audio_cfg: Optional[AudioConfig] = None,
                             n_random_controls: int = 20,
                             seed: int = 42) -> Dict[str, Any]:
    """Execute complete band occlusion audit with random-window control."""
    cfg = audio_cfg or AudioConfig()
    bands, row_to_hz = get_band_row_ranges(cfg)

    # 1. Unmasked baseline performance
    unmasked_probs, unmasked_preds, test_labels = evaluate_occluded_loader(
        model, test_loader, device, mode="none"
    )
    base_ba = float(balanced_accuracy_score(test_labels, unmasked_preds))
    base_auc = float(roc_auc_score(test_labels, unmasked_probs[:, 1]))

    out: Dict[str, Any] = {
        "unmasked": {
            "balanced_accuracy": base_ba,
            "auc_roc": base_auc,
        },
        "bands": {},
        "controls": {},
        "reliances": {},
    }

    # 2. Band occlusion evaluation
    for band_name, (r0, r1) in bands.items():
        band_height = r1 - r0
        out["bands"][band_name] = {"rows": [r0, r1], "height": band_height}

        for mode in ("mask", "keep_only"):
            occ_probs, occ_preds, _ = evaluate_occluded_loader(
                model, test_loader, device, band_rows=(r0, r1), mode=mode
            )
            met = compute_metrics_and_deltas(occ_probs, occ_preds, test_labels, test_df, base_ba)
            out["bands"][band_name][mode] = met

        # 3. Random contiguous row window control for this band
        rng = np.random.default_rng(seed)
        max_start = max(1, 224 - band_height)
        random_starts = rng.integers(0, max_start, size=n_random_controls)
        control_deltas: List[float] = []

        for w0 in random_starts:
            w1 = int(w0 + band_height)
            rnd_probs, rnd_preds, _ = evaluate_occluded_loader(
                model, test_loader, device, band_rows=(int(w0), w1), mode="mask"
            )
            rnd_ba = float(balanced_accuracy_score(test_labels, rnd_preds))
            control_deltas.append(float(rnd_ba - base_ba))

        mean_rnd = float(np.mean(control_deltas))
        sd_rnd = float(np.std(control_deltas, ddof=1)) if len(control_deltas) > 1 else 0.0
        threshold = float(mean_rnd - 2.0 * sd_rnd)

        out["controls"][band_name] = {
            "mean_delta_ba": mean_rnd,
            "sd_delta_ba": sd_rnd,
            "reliance_threshold": threshold,
            "n_samples": n_random_controls,
        }

        # Check reliance: delta BA below mean_random - 2*sd_random
        mask_delta = out["bands"][band_name]["mask"]["delta_balanced_accuracy"]
        is_reliant = bool(mask_delta < threshold)
        out["reliances"][band_name] = is_reliant

        print(f"  [Occlusion {band_name}] mask delta BA = {mask_delta:+.4f} | "
              f"Control = {mean_rnd:+.4f} +/- {sd_rnd:.4f} (thresh={threshold:.4f}) | "
              f"Reliance: {is_reliant}")

    return out
