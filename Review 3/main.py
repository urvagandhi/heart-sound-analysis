"""
Review 3 — End-to-end runner
============================
Single entry-point that wires every module together:

    1. Resolve paths (Drive / local / override) and load Phase 1 outputs.
    2. Build train / val / test DataLoaders.
    3. Construct the model (ResNet50 + SE + MHA) and train it.
    4. Evaluate on the test set and emit metrics + plots.
    5. Generate Grad-CAM, attention, and SHAP explanations.
    6. Save a comparison table versus literature.

Usage:
    python main.py                          # auto-detect base dir
    python main.py --base /path/to/RMS      # override base dir
    python main.py --epochs 2 --no-amp      # quick smoke run

Skip individual stages via flags (--skip-train, --skip-xai, --skip-shap).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running both as `python main.py` and `python Review\ 3/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config   import (
    Paths, AudioConfig, ModelConfig, TrainConfig,
    CLASS_NAMES, get_device, set_seed, is_colab, is_drive_mounted,
)
from data     import (
    load_metadata, load_class_weights, get_dataloaders, HeartSoundDataset,
)
from model    import build_model
from train    import train, plot_training_history
from evaluate import (
    evaluate_on_test, plot_confusion_matrix, plot_roc_pr_curves,
    save_predictions_csv, compute_and_save_extended_metrics,
)
from xai      import (
    visualise_gradcam, visualise_attention,
    compute_shap_values, frequency_share, plot_shap_frequency_importance,
    plot_shap_sample_overlays,
)
from utils    import build_comparison_table, pick_explainable_indices, metrics_to_dict
# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heart Sound XAI — Review 3 runner")
    p.add_argument("--base",       type=str, default=None,
                   help="Override base directory (containing physionet_2016/).")
    p.add_argument("--variant",    type=str, default="se_mha",
                   choices=["baseline", "se", "mha", "se_mha", "se_mha_dilated"],
                   help="Model architecture variant.")
    p.add_argument("--split",      type=str, default="legacy",
                   choices=["legacy", "dedup", "lodo"],
                   help="Dataset split protocol.")
    p.add_argument("--holdout",    type=str, default=None,
                   choices=["a", "b", "e", "f"],
                   help="Held-out database for LODO protocol.")
    p.add_argument("--seed",       type=int, default=42,
                   help="Random seed for split, initialization, and training.")
    p.add_argument("--tag",        type=str, default=None,
                   help="Explicit tag for output directory out-root/<tag>/.")
    p.add_argument("--out-root",   type=str, default=None,
                   help="Root directory for outputs (defaults to <base>/results).")
    p.add_argument("--freeze-bn",  dest="freeze_bn", action="store_true", default=True,
                   help="Freeze BatchNorm running stats during warm-up phase (default: True).")
    p.add_argument("--no-freeze-bn", dest="freeze_bn", action="store_false",
                   help="Do not freeze BatchNorm running stats during warm-up phase.")
    p.add_argument("--eval-only",  action="store_true",
                   help="Run evaluation only using specified checkpoint without retraining.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint for --eval-only.")
    p.add_argument("--with-xai",   action="store_true", default=False,
                   help="Enable XAI figure generation (off by default in benchmark runs).")
    p.add_argument("--epochs",     type=int, default=None, help="Total epochs.")
    p.add_argument("--warmup",     type=int, default=None, help="Warm-up epochs.")
    p.add_argument("--batch",      type=int, default=None, help="Batch size.")
    p.add_argument("--no-amp",     action="store_true", help="Disable mixed precision.")
    p.add_argument("--shap-n",     type=int, default=128,
                   help="Number of balanced test samples for SHAP.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; reuse existing checkpoint.")
    p.add_argument("--skip-xai",   action="store_true",
                   help="Skip Grad-CAM / attention plots.")
    p.add_argument("--skip-shap",  action="store_true",
                   help="Skip SHAP.")
    p.add_argument("--dedup",      action="store_true",
                   help="Legacy convenience flag for --split dedup.")
    return p.parse_args()


def _apply_overrides(args: argparse.Namespace, train_cfg: TrainConfig) -> TrainConfig:
    if args.epochs is not None:
        train_cfg.num_epochs = args.epochs
    if args.warmup is not None:
        train_cfg.warmup_epochs = args.warmup
    if args.batch is not None:
        train_cfg.batch_size = args.batch
    if args.seed is not None:
        train_cfg.seed = args.seed
    if args.no_amp:
        train_cfg.use_amp = False
    if args.num_workers is not None:
        train_cfg.num_workers = args.num_workers
    train_cfg.freeze_bn = bool(args.freeze_bn)
    return train_cfg


def _resolve_tag(args: argparse.Namespace) -> str:
    """Format standard tag: dedup_{variant}_seed{S}, lodo_{variant}_hold{X}_seed{S}, or legacy_se_mha."""
    if args.tag:
        return args.tag
    if args.eval_only and args.split == "legacy" and args.variant == "se_mha":
        return "legacy_se_mha"
    if args.split == "dedup" or args.dedup:
        return f"dedup_{args.variant}_seed{args.seed}"
    if args.split == "lodo":
        h = args.holdout or "a"
        return f"lodo_{args.variant}_hold{h}_seed{args.seed}"
    return f"{args.split}_{args.variant}_seed{args.seed}"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    import shutil
    from model import get_variant_param_details

    args = parse_args()
    if args.dedup and args.split == "legacy":
        args.split = "dedup"

    tag = _resolve_tag(args)
    paths = Paths.auto_detect(
        base_override=args.base,
        tag=tag,
        split=args.split,
        holdout=args.holdout,
        seed=args.seed,
        out_root=args.out_root,
    )
    paths.ensure_output_dirs()

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig(variant=args.variant)
    train_cfg = _apply_overrides(args, TrainConfig())

    set_seed(train_cfg.seed)
    device = get_device()

    print("=" * 60)
    print(f"  Heart Sound XAI — Run: {tag}")
    print("=" * 60)
    print(f"  Variant     : {args.variant}")
    print(f"  Split       : {args.split} (holdout={args.holdout}, seed={args.seed})")
    print(f"  Device      : {device}")
    print(f"  Output Dir  : {paths.output_dir}")
    print(paths)

    # 1) Data ---------------------------------------------------
    print("\n[1/4] Loading data ...")
    df = load_metadata(paths)
    print(f"  Recordings: {len(df)} | "
          f"distribution: {df['label'].value_counts().sort_index().to_dict()}")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = (
        get_dataloaders(paths, model_cfg, train_cfg, df)
    )

    class_weights = load_class_weights(paths, train_df, device=device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    # 2) Model --------------------------------------------------
    print("\n[2/4] Building model ...")
    is_eval = args.eval_only or args.skip_train
    model = build_model(model_cfg, device=device, pretrained=not is_eval,
                        freeze_backbone=not is_eval)

    # Persist run configuration JSON
    config_record = {
        "tag": tag,
        "variant": args.variant,
        "split": args.split,
        "holdout": args.holdout,
        "seed": args.seed,
        "args": vars(args),
        "git_commit": get_git_commit(),
        "torch_version": torch.__version__,
        "param_counts": get_variant_param_details(model),
    }
    with open(paths.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    # 3) Training / Checkpoint Loading -------------------------
    ckpt_path = paths.checkpoint_dir / "best_model.pth"

    if args.eval_only:
        ckpt_to_load = Path(args.checkpoint) if args.checkpoint else (paths.base_dir / "Review 3" / "output" / "checkpoints" / "best_model.pth")
        if not ckpt_to_load.exists():
            raise FileNotFoundError(f"--eval-only specified but checkpoint not found at: {ckpt_to_load}")
        print(f"\n[3/4] Evaluation only. Loading checkpoint from {ckpt_to_load} ...")
        raw_ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=True)
        state_dict = raw_ckpt["model_state_dict"] if (isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt) else raw_ckpt
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if ckpt_to_load.resolve() != ckpt_path.resolve():
            shutil.copyfile(ckpt_to_load, ckpt_path)

        hist_src = ckpt_to_load.parent / "history.json"
        hist_dest = paths.checkpoint_dir / "history.json"
        if hist_src.exists() and hist_src.resolve() != hist_dest.resolve():
            shutil.copyfile(hist_src, hist_dest)
        elif not hist_dest.exists():
            with open(hist_dest, "w", encoding="utf-8") as f:
                json.dump({"eval_only": True, "source_checkpoint": str(ckpt_to_load)}, f, indent=2)
    elif not args.skip_train:
        print("\n[3/4] Training ...")
        history = train(model, train_loader, val_loader, class_weights,
                        paths, train_cfg, device)
        if args.with_xai:
            plot_training_history(history, save_path=paths.output_dir / "training_history.png")
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--skip-train passed but no checkpoint at {ckpt_path}")
        raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = raw_ckpt["model_state_dict"] if (isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt) else raw_ckpt
        model.load_state_dict(state_dict, strict=True)
        model.unfreeze_backbone()
        print(f"\n[3/4] Skipped training. Loaded weights from {ckpt_path}.")

    # 4) Evaluation --------------------------------------------
    print("\n[4/4] Evaluating on test set ...")
    metrics = evaluate_on_test(model, test_loader, device, CLASS_NAMES)

    # Persist predictions CSV and scalar metrics JSON
    save_predictions_csv(metrics, test_df, CLASS_NAMES,
                         save_path=paths.output_dir / "test_predictions.csv")
    with open(paths.output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_dict(metrics), f, indent=2)

    compute_and_save_extended_metrics(metrics, test_df, train_df, paths.output_dir)

    # 5) Optional XAI figures ----------------------------------
    if args.with_xai and not args.skip_xai:
        print("\nGenerating XAI visualisations ...")
        plot_confusion_matrix(metrics, CLASS_NAMES,
                              save_path=paths.output_dir / "confusion_matrix.png")
        plot_roc_pr_curves(metrics, CLASS_NAMES,
                           save_path=paths.output_dir / "roc_pr_curves.png")

        test_dataset = HeartSoundDataset(test_df, paths, model_cfg, augment=False)
        picks = pick_explainable_indices(metrics, n_correct=2, n_wrong=2,
                                         rng_seed=train_cfg.seed)
        visualise_gradcam(model, test_dataset, picks, device, audio_cfg,
                          CLASS_NAMES,
                          save_path=paths.output_dir / "gradcam_overlays.png")
        if model_cfg.use_mha:
            visualise_attention(model, test_dataset, picks, device, audio_cfg,
                                CLASS_NAMES,
                                save_path=paths.output_dir / "attention_overlays.png")

        if not args.skip_shap:
            try:
                shap_values, shap_inputs, shap_labels, shap_indices = compute_shap_values(
                    model, train_loader, test_dataset, device,
                    n_background=32, n_explain=min(args.shap_n, 32), seed=42,
                )
                np.savez(
                    paths.output_dir / "shap_values.npz",
                    shap_class_0=shap_values[0],
                    shap_class_1=shap_values[1],
                    inputs=shap_inputs.numpy() if isinstance(shap_inputs, torch.Tensor) else np.asarray(shap_inputs),
                    labels=shap_labels.numpy() if isinstance(shap_labels, torch.Tensor) else np.asarray(shap_labels),
                    test_row_indices=shap_indices,
                )
                frequency_share(shap_values, audio_cfg, cutoff_hz=400.0,
                                output_path=paths.output_dir / "shap_summary.json")
                plot_shap_frequency_importance(
                    shap_values, audio_cfg, CLASS_NAMES,
                    save_path=paths.output_dir / "shap_frequency_importance.png")
                plot_shap_sample_overlays(
                    shap_values, shap_inputs, shap_labels, audio_cfg, CLASS_NAMES,
                    save_path=paths.output_dir / "shap_sample_overlays.png")
            except Exception as e:
                print(f"  SHAP visualisations skipped: {e}")

        build_comparison_table(metrics, save_path=paths.output_dir / "comparison_table.png")

    print(f"\nDone. Run completed for tag '{tag}'. Output saved to: {paths.output_dir}")


if __name__ == "__main__":
    main()
