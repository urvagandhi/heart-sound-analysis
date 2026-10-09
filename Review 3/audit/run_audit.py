"""End-to-end population-level explanation audit runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure Review 3 is on path
review3_dir = Path(__file__).resolve().parent.parent
if str(review3_dir) not in sys.path:
    sys.path.insert(0, str(review3_dir))

from audit.agreement import run_agreement_audit
from audit.calibration import run_calibration_audit
from audit.common import get_band_row_ranges, load_model
from audit.enrichment import STATE_NAMES, run_enrichment_audit
from audit.faithfulness import run_faithfulness_audit
from audit.occlusion import run_band_occlusion_audit
from audit.sanity import run_sanity_check
from config import AudioConfig, CLASS_NAMES, Paths, get_device
from data import HeartSoundDataset, get_dataloaders, load_metadata
from evaluate import evaluate_on_test
from xai import (
    plot_shap_frequency_importance,
    plot_shap_sample_overlays,
    visualise_attention,
    visualise_gradcam,
)


def plot_occlusion_figures(occlusion_res: Dict[str, Any], save_path: Path) -> None:
    """Generate grayscale-safe bar chart of frequency band delta BA with hatch patterns."""
    bands = ["B1", "B2", "B3"]
    labels = ["B1 (20-150 Hz)", "B2 (150-500 Hz)", "B3 (500-1000 Hz)"]

    mask_deltas = [occlusion_res["bands"][b]["mask"]["delta_balanced_accuracy"] for b in bands]
    ctrl_means  = [occlusion_res["controls"][b]["mean_delta_ba"] for b in bands]
    ctrl_sds    = [occlusion_res["controls"][b]["sd_delta_ba"] for b in bands]

    x = np.arange(len(bands))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, mask_deltas, width, label="Band Masked",
                    color="#D0D0D0", edgecolor="black", hatch="//", linewidth=1.2)
    rects2 = ax.bar(x + width/2, ctrl_means, width, yerr=ctrl_sds, capsize=5,
                    label="Random Window Control (+/- 1 SD)",
                    color="#F5F5F5", edgecolor="black", hatch="\\\\", linewidth=1.2)

    ax.set_ylabel("Delta Balanced Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Frequency Band Occlusion Impact vs Random Window Controls", fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_faithfulness_figures(faith_res: Dict[str, Any], save_path: Path) -> None:
    """Generate grayscale-safe deletion and insertion curves with distinct line styles."""
    p_steps = np.asarray(faith_res["p_steps"]) * 100.0  # Convert to percent
    methods = list(faith_res["methods"].keys())

    line_styles = {"Grad-CAM": "-", "Attention": "--", "SHAP": "-."}
    markers = {"Grad-CAM": "o", "Attention": "s", "SHAP": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Deletion curves
    for m in methods:
        data = faith_res["methods"][m]
        ax1.plot(p_steps, data["deletion_curve"], label=f"{m} (AUC={data['mean_deletion_auc']:.3f})",
                 linestyle=line_styles.get(m, "-"), marker=markers.get(m, "o"),
                 color="black" if m == "Grad-CAM" else ("#505050" if m == "SHAP" else "#808080"),
                 linewidth=1.5, markersize=4)

    # Plot random baseline for first method
    rnd_del = faith_res["methods"][methods[0]]["random_deletion_curve"]
    ax1.plot(p_steps, rnd_del, label="Random Baseline", linestyle=":", color="#A0A0A0", linewidth=1.8)

    ax1.set_xlabel("Pixels Deleted (%)", fontsize=11)
    ax1.set_ylabel("Predicted Class Probability", fontsize=11)
    ax1.set_title("Deletion Faithfulness Curve", fontsize=12, fontweight="bold")
    ax1.legend(frameon=True, edgecolor="black")
    ax1.grid(linestyle=":", alpha=0.6)

    # Insertion curves
    for m in methods:
        data = faith_res["methods"][m]
        ax2.plot(p_steps, data["insertion_curve"], label=f"{m} (AUC={data['mean_insertion_auc']:.3f})",
                 linestyle=line_styles.get(m, "-"), marker=markers.get(m, "o"),
                 color="black" if m == "Grad-CAM" else ("#505050" if m == "SHAP" else "#808080"),
                 linewidth=1.5, markersize=4)

    rnd_ins = faith_res["methods"][methods[0]]["random_insertion_curve"]
    ax2.plot(p_steps, rnd_ins, label="Random Baseline", linestyle=":", color="#A0A0A0", linewidth=1.8)

    ax2.set_xlabel("Pixels Inserted (%)", fontsize=11)
    ax2.set_ylabel("Predicted Class Probability", fontsize=11)
    ax2.set_title("Insertion Faithfulness Curve", fontsize=12, fontweight="bold")
    ax2.legend(frameon=True, edgecolor="black")
    ax2.grid(linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_enrichment_figures(enrich_res: Dict[str, Any], save_path: Path) -> None:
    """Generate grayscale-safe enrichment bar plot with error bars and reference ratio line."""
    if enrich_res.get("enrichment_status") != "evaluated":
        return

    states = STATE_NAMES
    means = [enrich_res["overall"][s]["mean"] for s in states]
    ci_lows = [enrich_res["overall"][s]["ci_low"] for s in states]
    ci_highs = [enrich_res["overall"][s]["ci_high"] for s in states]

    yerr_lower = [m - l for m, l in zip(means, ci_lows)]
    yerr_upper = [h - m for m, h in zip(means, ci_highs)]
    yerr = [yerr_lower, yerr_upper]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(states))
    ax.bar(x, means, yerr=yerr, capsize=6, width=0.5,
           color="#E8E8E8", edgecolor="black", hatch="///", linewidth=1.2)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5,
               label="Proportional Attention ($E_s = 1.0$)")

    ax.set_xticks(x)
    ax.set_xticklabels(["S1", "Systole", "S2", "Diastole"], fontsize=11)
    ax.set_ylabel("State Enrichment Ratio ($E_s$)", fontsize=11, fontweight="bold")
    ax.set_title("Grad-CAM State Annotation Enrichment Ratio ($E_s$ +/- 95% Bootstrap CI)",
                 fontsize=12, pad=12)
    ax.legend(frameon=True, edgecolor="black")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def pick_four_balanced_cases(metrics: Dict, test_df: pd.DataFrame) -> List[int]:
    """Pick 4 balanced cases: confident normal, confident abnormal, false positive, false negative."""
    labels = np.asarray(metrics["labels"])
    preds  = np.asarray(metrics["preds"])
    probs  = np.asarray(metrics["probs"])

    picks = []

    # 1. Confident Normal (label 0, pred 0, max prob 0)
    norm_corr = np.where((labels == 0) & (preds == 0))[0]
    if len(norm_corr) > 0:
        picks.append(int(norm_corr[np.argmax(probs[norm_corr, 0])]))

    # 2. Confident Abnormal (label 1, pred 1, max prob 1)
    abn_corr = np.where((labels == 1) & (preds == 1))[0]
    if len(abn_corr) > 0:
        picks.append(int(abn_corr[np.argmax(probs[abn_corr, 1])]))

    # 3. False Positive (label 0, pred 1)
    fp = np.where((labels == 0) & (preds == 1))[0]
    if len(fp) > 0:
        picks.append(int(fp[np.argmax(probs[fp, 1])]))

    # 4. False Negative (label 1, pred 0)
    fn = np.where((labels == 1) & (preds == 0))[0]
    if len(fn) > 0:
        picks.append(int(fn[np.argmax(probs[fn, 0])]))

    return picks


def run_audit(tag: str,
              raw_dir: Optional[Path] = None,
              annotations_dir: Optional[Path] = None,
              shap_n: int = 128,
              results_dir: Optional[Path] = None,
              base_dir: Optional[Path] = None) -> None:
    """Execute complete 5-part audit on a trained model checkpoint."""
    repo_root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    res_root = Path(results_dir) if results_dir else repo_root / "results"
    out_dir = res_root / f"audit_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("=" * 70)
    print(f"  Running Explanation & Clinical Audit for Tag: {tag}")
    print(f"  Audit Output Directory: {out_dir}")
    print("=" * 70)

    # 1. Load model and config
    model, model_cfg, config_dict = load_model(tag, results_dir=res_root, base_dir=repo_root, device=device)

    # 2. Load data
    split_type = config_dict.get("split", "dedup")
    seed_val = config_dict.get("seed", 42)
    holdout_val = config_dict.get("holdout", None)

    paths = Paths.auto_detect(
        base_override=str(repo_root),
        tag=tag,
        split=split_type,
        holdout=holdout_val,
        seed=seed_val,
        out_root=res_root,
    )
    df = load_metadata(paths)

    from config import TrainConfig
    train_cfg = TrainConfig(seed=seed_val)
    train_loader, val_loader, test_loader, train_df, val_df, test_df = (
        get_dataloaders(paths, model_cfg, train_cfg, df)
    )
    test_dataset = HeartSoundDataset(test_df, paths, model_cfg, augment=False)
    audio_cfg = AudioConfig()

    # 3. Base Evaluation & Calibration
    print("\n[Audit 1/6] Calibration and operating point audit ...")
    test_metrics = evaluate_on_test(model, test_loader, device, CLASS_NAMES)
    val_metrics  = evaluate_on_test(model, val_loader,  device, CLASS_NAMES)

    calib_res = run_calibration_audit(
        val_probs=np.asarray(val_metrics["probs"]),
        val_labels=np.asarray(val_metrics["labels"]),
        test_probs=np.asarray(test_metrics["probs"]),
        test_labels=np.asarray(test_metrics["labels"]),
        test_df=test_df,
        raw_dir=raw_dir,
        annotations_dir=annotations_dir,
    )
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib_res, f, indent=2)

    # 4. Band Occlusion Audit
    print("\n[Audit 2/6] Frequency band occlusion audit ...")
    occlusion_res = run_band_occlusion_audit(
        model, test_loader, test_df, device, audio_cfg=audio_cfg, n_random_controls=20, seed=42
    )
    with open(out_dir / "occlusion.json", "w", encoding="utf-8") as f:
        json.dump(occlusion_res, f, indent=2)
    plot_occlusion_figures(occlusion_res, save_path=out_dir / "occlusion_bars.png")

    # 5. Faithfulness Deletion and Insertion Curves
    print("\n[Audit 3/6] Faithfulness deletion and insertion curves ...")
    faith_res = run_faithfulness_audit(
        model, train_loader, test_dataset, test_df, device,
        out_dir=out_dir, shap_n=shap_n, n_random_maps=5, seed=42
    )
    with open(out_dir / "faithfulness.json", "w", encoding="utf-8") as f:
        json.dump(faith_res, f, indent=2)
    plot_faithfulness_figures(faith_res, save_path=out_dir / "faithfulness_curves.png")

    # 6. Adebayo Sanity Check (Weight Randomization)
    print("\n[Audit 4/6] Weight randomization sanity check ...")
    sanity_res = run_sanity_check(
        model, test_dataset, test_df, device, n_samples=100, seed=42
    )
    with open(out_dir / "sanity.json", "w", encoding="utf-8") as f:
        json.dump(sanity_res, f, indent=2)

    # 7. Method Agreement (7x7 pooled)
    print("\n[Audit 5/6] Pairwise explanation agreement audit ...")
    # Load cohort from shap_values.npz saved in step 5
    shap_data = np.load(out_dir / "shap_values.npz")
    shap_sample_idx = shap_data["test_row_indices"]
    shap_vals_list = [shap_data["shap_class_0"], shap_data["shap_class_1"]]

    agree_res = run_agreement_audit(
        model, test_dataset, shap_sample_idx, shap_vals_list, device
    )
    with open(out_dir / "agreement.json", "w", encoding="utf-8") as f:
        json.dump(agree_res, f, indent=2)

    # 8. PhysioNet State Enrichment Audit
    print("\n[Audit 6/6] PhysioNet state annotation enrichment audit ...")
    enrich_res = run_enrichment_audit(
        model, test_dataset, test_df, device,
        annotations_dir=annotations_dir, raw_dir=raw_dir, results_dir=res_root
    )
    with open(out_dir / "enrichment.json", "w", encoding="utf-8") as f:
        json.dump(enrich_res, f, indent=2)
    plot_enrichment_figures(enrich_res, save_path=out_dir / "enrichment_bars.png")

    # 9. Regenerate sample overlays for 4 balanced cases
    print("\nRegenerating explanation figures for 4 balanced sample cases ...")
    four_picks = pick_four_balanced_cases(test_metrics, test_df)
    if len(four_picks) > 0:
        visualise_gradcam(model, test_dataset, four_picks, device, audio_cfg,
                          CLASS_NAMES, save_path=out_dir / "gradcam_overlays.png")
        if model_cfg.use_mha:
            visualise_attention(model, test_dataset, four_picks, device, audio_cfg,
                                CLASS_NAMES, save_path=out_dir / "attention_overlays.png")

    print(f"\nAudit complete for {tag}! All JSON summaries and figures saved to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete explanation and clinical evaluation audit.")
    parser.add_argument("--tag", type=str, required=True, help="Run tag (e.g., 'dedup_se_mha_seed42').")
    parser.add_argument("--raw_dir", type=str, default=None, help="Directory containing raw PhysioNet WAV recordings.")
    parser.add_argument("--annotations_dir", type=str, default=None, help="Directory containing StateAns .mat files.")
    parser.add_argument("--shap-n", type=int, default=128, help="Number of balanced samples for SHAP cohort.")
    parser.add_argument("--results_dir", type=str, default=None, help="Path to results root directory.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base repo directory.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    ann_dir = Path(args.annotations_dir) if args.annotations_dir else None
    res_dir = Path(args.results_dir) if args.results_dir else None
    base_dir = Path(args.base_dir) if args.base_dir else None

    run_audit(
        tag=args.tag,
        raw_dir=raw_dir,
        annotations_dir=ann_dir,
        shap_n=args.shap_n,
        results_dir=res_dir,
        base_dir=base_dir,
    )


if __name__ == "__main__":
    main()
