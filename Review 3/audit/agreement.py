"""Explanation method agreement audit: pairwise Spearman correlation and top-20% IoU."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F

from audit.faithfulness import compute_attention_map, compute_gradcam_map
from model import HeartSoundModel


def pool_to_7x7(arr_224: np.ndarray) -> np.ndarray:
    """Downsample a 224x224 attribution map to 7x7 via 2D average pooling."""
    t = torch.from_numpy(arr_224).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(t, (7, 7)).squeeze().numpy()
    return pooled


def top_k_iou(arr1: np.ndarray, arr2: np.ndarray, top_fraction: float = 0.20) -> float:
    """Compute Intersection over Union between the top fraction of cells."""
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    k = max(1, int(round(len(flat1) * top_fraction)))

    top_idx1 = set(np.argsort(-flat1)[:k])
    top_idx2 = set(np.argsort(-flat2)[:k])

    intersection = len(top_idx1.intersection(top_idx2))
    union = len(top_idx1.union(top_idx2))
    return float(intersection / union) if union > 0 else 0.0


def run_agreement_audit(model: HeartSoundModel,
                        test_dataset: Any,
                        shap_sample_indices: Sequence[int],
                        shap_values: Sequence[np.ndarray],
                        device: str) -> Dict[str, Any]:
    """Compute pairwise 7x7 pooled agreement across Grad-CAM, Attention, and SHAP."""
    model.eval()
    n_samples = len(shap_sample_indices)

    has_mha = hasattr(model, "mha")
    pairs = [("Grad-CAM", "SHAP")]
    if has_mha:
        pairs.extend([("Grad-CAM", "Attention"), ("Attention", "SHAP")])

    pair_metrics: Dict[str, Dict[str, List[float]]] = {
        f"{p[0]}_vs_{p[1]}": {"spearman": [], "iou": []} for p in pairs
    }

    print(f"  [Agreement] Evaluating agreement across {n_samples} samples ...")

    for i, idx in enumerate(shap_sample_indices):
        x_norm, _ = test_dataset[int(idx)]
        x_batch = x_norm.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = int(model(x_batch).argmax(dim=1).item())

        # 1. Grad-CAM (7x7)
        cam_224 = compute_gradcam_map(model, x_batch, target_class=pred)
        cam_7x7 = pool_to_7x7(cam_224)

        # 2. SHAP (7x7)
        sv = shap_values[pred][i]  # (3, 224, 224)
        shap_224 = np.mean(np.abs(sv), axis=0)
        shap_7x7 = pool_to_7x7(shap_224)

        maps_dict = {
            "Grad-CAM": cam_7x7,
            "SHAP": shap_7x7,
        }

        # 3. Attention (7x7)
        if has_mha:
            _ = model(x_batch)
            attn_224 = compute_attention_map(model, x_batch)
            attn_7x7 = pool_to_7x7(attn_224) if attn_224 is not None else np.zeros((7, 7))
            maps_dict["Attention"] = attn_7x7

        for m1, m2 in pairs:
            key = f"{m1}_vs_{m2}"
            arr1 = maps_dict[m1]
            arr2 = maps_dict[m2]

            r, _ = stats.spearmanr(arr1.flatten(), arr2.flatten())
            spearman_val = float(r) if not np.isnan(r) else 0.0
            iou_val = top_k_iou(arr1, arr2, top_fraction=0.20)

            pair_metrics[key]["spearman"].append(spearman_val)
            pair_metrics[key]["iou"].append(iou_val)

    summary: Dict[str, Any] = {"n_samples": n_samples, "pairwise": {}}
    for key, data in pair_metrics.items():
        mean_spearman = float(np.mean(data["spearman"]))
        sd_spearman = float(np.std(data["spearman"], ddof=1)) if len(data["spearman"]) > 1 else 0.0
        mean_iou = float(np.mean(data["iou"]))
        sd_iou = float(np.std(data["iou"], ddof=1)) if len(data["iou"]) > 1 else 0.0

        if mean_spearman < 0.30:
            agreement_level = "low"
        elif mean_spearman <= 0.60:
            agreement_level = "moderate"
        else:
            agreement_level = "high"

        summary["pairwise"][key] = {
            "mean_spearman": mean_spearman,
            "sd_spearman": sd_spearman,
            "mean_top20_iou": mean_iou,
            "sd_top20_iou": sd_iou,
            "level": agreement_level,
        }
        print(f"    {key:20s}: Spearman={mean_spearman:+.4f} +/- {sd_spearman:.4f} ({agreement_level}) | IoU={mean_iou:.4f} +/- {sd_iou:.4f}")

    return summary
