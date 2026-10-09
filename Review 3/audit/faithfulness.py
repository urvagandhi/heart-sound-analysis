"""Faithfulness audit: deletion and insertion curves for Grad-CAM, attention, and SHAP."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audit.common import to_norm, to_unit
from model import HeartSoundModel
from xai import GradCAM, compute_shap_values


def compute_gradcam_map(model: HeartSoundModel, x: torch.Tensor, target_class: int) -> np.ndarray:
    """Generate Grad-CAM attribution map (224x224) for single input x (1, 3, 224, 224)."""
    cam_extractor = GradCAM(model, model.backbone_last_conv)
    cam = cam_extractor.generate(x, target_class=target_class)
    cam_extractor.remove()
    return cam  # (224, 224) in [0, 1]


def compute_attention_map(model: HeartSoundModel, x: torch.Tensor) -> Optional[np.ndarray]:
    """Generate upsampled self-attention map (224x224) if MHA is present."""
    if not hasattr(model, "mha") or model.last_attention_weights is None:
        return None
    attn = model.last_attention_weights[0].cpu()  # (Q, K)
    per_key = attn.mean(dim=0)                   # (K,)
    side = int(np.sqrt(per_key.numel()))
    attn_2d = per_key.view(1, 1, side, side)
    attn_up = F.interpolate(attn_2d, size=(224, 224), mode="bilinear", align_corners=False)
    m = attn_up.squeeze().numpy()
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m_norm


def evaluate_perturbation_curves(model: HeartSoundModel,
                                 x_unit: torch.Tensor,
                                 saliency_map: np.ndarray,
                                 intact_class: int,
                                 device: str,
                                 p_steps: Sequence[float] = tuple(np.arange(0, 55, 5) / 100.0),
                                ) -> Tuple[List[float], List[float]]:
    """Compute deletion and insertion curves for a single input given a 224x224 saliency map."""
    H, W = saliency_map.shape
    total_pixels = H * W
    flat_map = saliency_map.flatten()
    order = np.argsort(-flat_map)  # Descending order of importance

    del_scores: List[float] = []
    ins_scores: List[float] = []

    # Flattened unit-space image: (3, H*W)
    c, _, _ = x_unit.shape
    flat_img = x_unit.view(c, total_pixels)

    for p in p_steps:
        k = int(round(p * total_pixels))
        top_indices = order[:k]

        # Deletion: top k pixels set to 0
        img_del = flat_img.clone()
        if k > 0:
            img_del[:, top_indices] = 0.0
        img_del = img_del.view(1, c, H, W)
        x_del_norm = to_norm(img_del).to(device)
        with torch.no_grad():
            prob_del = torch.softmax(model(x_del_norm), dim=1)[0, intact_class].item()
        del_scores.append(float(prob_del))

        # Insertion: starting from all-zero unit image, unmask top k pixels
        img_ins = torch.zeros_like(flat_img)
        if k > 0:
            img_ins[:, top_indices] = flat_img[:, top_indices]
        img_ins = img_ins.view(1, c, H, W)
        x_ins_norm = to_norm(img_ins).to(device)
        with torch.no_grad():
            prob_ins = torch.softmax(model(x_ins_norm), dim=1)[0, intact_class].item()
        ins_scores.append(float(prob_ins))

    return del_scores, ins_scores


def curve_auc(scores: List[float], p_steps: Sequence[float]) -> float:
    """Compute normalized area under curve across perturbation steps."""
    p_arr = np.asarray(p_steps)
    s_arr = np.asarray(scores)
    width = float(p_arr[-1] - p_arr[0])
    if width <= 0:
        return 0.0
    return float(np.trapezoid(s_arr, p_arr) / width)


def run_faithfulness_audit(model: HeartSoundModel,
                           train_loader: DataLoader,
                           test_dataset: Any,
                           test_df: Any,
                           device: str,
                           out_dir: Path,
                           shap_n: int = 128,
                           n_random_maps: int = 5,
                           seed: int = 42) -> Dict[str, Any]:
    """Execute faithfulness deletion and insertion audit comparing to random map baselines."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    p_steps = [float(p / 100.0) for p in range(0, 55, 5)]

    # Select balanced test cohort for SHAP and faithfulness evaluation (shap_n // 2 per class)
    labels_arr = test_df["label"].values.astype(int)
    rng = np.random.default_rng(seed)
    n_per = max(1, shap_n // 2)
    norm_idx = np.where(labels_arr == 0)[0]
    abn_idx = np.where(labels_arr == 1)[0]
    sel_norm = rng.choice(norm_idx, size=min(n_per, len(norm_idx)), replace=False)
    sel_abn = rng.choice(abn_idx, size=min(n_per, len(abn_idx)), replace=False)
    chosen_indices = np.concatenate([sel_norm, sel_abn])

    # 1. Compute SHAP explanations on cohort
    print(f"  [Faithfulness] Computing SHAP for {len(chosen_indices)} balanced samples ...")
    shap_vals, shap_inputs, shap_labels, _ = compute_shap_values(
        model, train_loader, test_dataset, device,
        n_background=32, n_explain=len(chosen_indices), seed=seed,
        test_indices=chosen_indices,
    )
    np.savez(
        out_dir / "shap_values.npz",
        shap_class_0=shap_vals[0],
        shap_class_1=shap_vals[1],
        inputs=shap_inputs.numpy() if isinstance(shap_inputs, torch.Tensor) else np.asarray(shap_inputs),
        labels=shap_labels.numpy() if isinstance(shap_labels, torch.Tensor) else np.asarray(shap_labels),
        test_row_indices=chosen_indices,
    )

    methods = ["Grad-CAM"]
    if hasattr(model, "mha"):
        methods.append("Attention")
    methods.append("SHAP")

    results: Dict[str, Any] = {
        "p_steps": p_steps,
        "n_samples": len(chosen_indices),
        "methods": {},
    }

    # Evaluate each method
    for method in methods:
        del_aucs: List[float] = []
        ins_aucs: List[float] = []
        rnd_del_aucs: List[float] = []
        rnd_ins_aucs: List[float] = []

        del_curve_accum = np.zeros(len(p_steps), dtype=float)
        ins_curve_accum = np.zeros(len(p_steps), dtype=float)
        rnd_del_accum   = np.zeros(len(p_steps), dtype=float)
        rnd_ins_accum   = np.zeros(len(p_steps), dtype=float)

        for i, idx in enumerate(chosen_indices):
            x_norm, y_true = test_dataset[int(idx)]
            x_batch = x_norm.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(x_batch)
                intact_class = int(logits.argmax(dim=1).item())

            x_unit = to_unit(x_norm)  # (3, 224, 224)

            # Generate saliency map
            if method == "Grad-CAM":
                s_map = compute_gradcam_map(model, x_batch, target_class=intact_class)
            elif method == "Attention":
                _ = model(x_batch)
                s_map = compute_attention_map(model, x_batch)
                if s_map is None:
                    s_map = np.zeros((224, 224), dtype=float)
            elif method == "SHAP":
                # Average SHAP values across color channels for predicted class
                sv = shap_vals[intact_class][i]  # (3, 224, 224)
                s_map = np.mean(np.abs(sv), axis=0)
                s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)

            d_scores, i_scores = evaluate_perturbation_curves(
                model, x_unit, s_map, intact_class, device, p_steps=p_steps
            )
            del_aucs.append(curve_auc(d_scores, p_steps))
            ins_aucs.append(curve_auc(i_scores, p_steps))
            del_curve_accum += np.asarray(d_scores)
            ins_curve_accum += np.asarray(i_scores)

            # 5 random map baselines for this sample
            sample_rnd_del: List[float] = []
            sample_rnd_ins: List[float] = []
            for r_seed in range(n_random_maps):
                rnd_map = rng.random((224, 224))
                rd_scores, ri_scores = evaluate_perturbation_curves(
                    model, x_unit, rnd_map, intact_class, device, p_steps=p_steps
                )
                sample_rnd_del.append(curve_auc(rd_scores, p_steps))
                sample_rnd_ins.append(curve_auc(ri_scores, p_steps))
                rnd_del_accum += np.asarray(rd_scores) / n_random_maps
                rnd_ins_accum += np.asarray(ri_scores) / n_random_maps

            rnd_del_aucs.append(float(np.mean(sample_rnd_del)))
            rnd_ins_aucs.append(float(np.mean(sample_rnd_ins)))

        n_s = len(chosen_indices)
        mean_del_curve = (del_curve_accum / n_s).tolist()
        mean_ins_curve = (ins_curve_accum / n_s).tolist()
        mean_rnd_del_curve = (rnd_del_accum / n_s).tolist()
        mean_rnd_ins_curve = (rnd_ins_accum / n_s).tolist()

        # Wilcoxon signed-rank test (paired: deletion AUC vs random baseline)
        stat_w, p_val = stats.wilcoxon(del_aucs, rnd_del_aucs, alternative="less")

        results["methods"][method] = {
            "mean_deletion_auc": float(np.mean(del_aucs)),
            "mean_insertion_auc": float(np.mean(ins_aucs)),
            "mean_random_deletion_auc": float(np.mean(rnd_del_aucs)),
            "mean_random_insertion_auc": float(np.mean(rnd_ins_aucs)),
            "wilcoxon_stat": float(stat_w),
            "wilcoxon_p_value": float(p_val),
            "deletion_curve": mean_del_curve,
            "insertion_curve": mean_ins_curve,
            "random_deletion_curve": mean_rnd_del_curve,
            "random_insertion_curve": mean_rnd_ins_curve,
        }
        print(f"  [{method}] Deletion AUC={np.mean(del_aucs):.4f} vs Random={np.mean(rnd_del_aucs):.4f} (Wilcoxon p={p_val:.2e})")

    return results
