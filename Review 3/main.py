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
    save_predictions_csv,
)
from xai      import (
    visualise_gradcam, visualise_attention,
    compute_shap_values, plot_shap_frequency_importance,
    plot_shap_sample_overlays,
)
from utils    import build_comparison_table, pick_explainable_indices, metrics_to_dict


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Heart Sound XAI — Review 3 runner")
    p.add_argument("--base",     type=str, default=None,
                   help="Override base directory (containing physionet_2016/).")
    p.add_argument("--epochs",   type=int, default=None, help="Total epochs.")
    p.add_argument("--warmup",   type=int, default=None, help="Warm-up epochs.")
    p.add_argument("--batch",    type=int, default=None, help="Batch size.")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-amp",   action="store_true", help="Disable mixed precision.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training; reuse existing checkpoint.")
    p.add_argument("--skip-xai",   action="store_true",
                   help="Skip Grad-CAM / attention plots.")
    p.add_argument("--skip-shap",  action="store_true",
                   help="Skip SHAP (which can be slow on CPU).")
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def _apply_overrides(args, train_cfg: TrainConfig) -> TrainConfig:
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
    return train_cfg


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1) Config -------------------------------------------------
    paths     = Paths.auto_detect(base_override=args.base)
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = _apply_overrides(args, TrainConfig())

    paths.ensure_output_dirs()
    set_seed(train_cfg.seed)
    device = get_device()

    print("=" * 60)
    print("  Heart Sound XAI — Review 3")
    print("=" * 60)
    print(f"  Environment : Colab={is_colab()}  Drive mounted={is_drive_mounted()}")
    print(f"  Device      : {device}")
    print(paths)

    # 2) Data ---------------------------------------------------
    print("\n[1/5] Loading data ...")
    df = load_metadata(paths)
    print(f"  Recordings: {len(df)} | "
          f"distribution: {df['label'].value_counts().sort_index().to_dict()}")

    train_loader, val_loader, test_loader, train_df, val_df, test_df = (
        get_dataloaders(paths, model_cfg, train_cfg, df)
    )

    class_weights = load_class_weights(paths, train_df, device=device)
    print(f"  Class weights: {class_weights.cpu().numpy()}")

    # Sanity-check one batch
    X_batch, y_batch = next(iter(train_loader))
    print(f"  Batch shape : {tuple(X_batch.shape)} | "
          f"value range [{X_batch.min():.2f}, {X_batch.max():.2f}]")

    # 3) Model --------------------------------------------------
    print("\n[2/5] Building model ...")
    model = build_model(model_cfg, device=device, pretrained=True,
                        freeze_backbone=True)

    # 4) Train --------------------------------------------------
    ckpt_path = paths.checkpoint_dir / "best_model.pth"
    if not args.skip_train:
        print("\n[3/5] Training ...")
        history = train(model, train_loader, val_loader, class_weights,
                        paths, train_cfg, device)
        plot_training_history(history,
                              save_path=paths.output_dir / "training_history.png")
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--skip-train passed but no checkpoint at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                         weights_only=True))
        # Unfreeze so any subsequent fine-tuning is unobstructed
        model.unfreeze_backbone()
        print(f"\n[3/5] Skipping training. Loaded weights from {ckpt_path}.")

    # 5) Evaluate -----------------------------------------------
    print("\n[4/5] Evaluating on test set ...")
    metrics = evaluate_on_test(model, test_loader, device, CLASS_NAMES)
    plot_confusion_matrix(metrics, CLASS_NAMES,
                          save_path=paths.output_dir / "confusion_matrix.png")
    plot_roc_pr_curves(metrics, CLASS_NAMES,
                       save_path=paths.output_dir / "roc_pr_curves.png")
    save_predictions_csv(metrics, test_df, CLASS_NAMES,
                         save_path=paths.output_dir / "test_predictions.csv")

    # Persist scalar metrics
    with open(paths.output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_to_dict(metrics), f, indent=2)

    # 6) XAI ----------------------------------------------------
    print("\n[5/5] Generating explanations ...")
    test_dataset = HeartSoundDataset(test_df, paths, model_cfg, augment=False)
    picks = pick_explainable_indices(metrics, n_correct=2, n_wrong=2,
                                     rng_seed=train_cfg.seed)
    print(f"  Showing samples (correct + wrong): {picks}")

    if not args.skip_xai:
        visualise_gradcam(model, test_dataset, picks, device, audio_cfg,
                          CLASS_NAMES,
                          save_path=paths.output_dir / "gradcam_overlays.png")
        visualise_attention(model, test_dataset, picks, device, audio_cfg,
                            CLASS_NAMES,
                            save_path=paths.output_dir / "attention_overlays.png")

    if not args.skip_shap:
        try:
            shap_values, shap_inputs, shap_labels = compute_shap_values(
                model, train_loader, test_loader, device,
                n_background=32, n_explain=8,
            )
            plot_shap_frequency_importance(
                shap_values, audio_cfg, CLASS_NAMES,
                save_path=paths.output_dir / "shap_frequency_importance.png")
            plot_shap_sample_overlays(
                shap_values, shap_inputs, shap_labels, audio_cfg, CLASS_NAMES,
                save_path=paths.output_dir / "shap_sample_overlays.png")
        except Exception as e:
            print(f"  SHAP failed: {e}. Continuing without SHAP.")

    build_comparison_table(metrics,
                           save_path=paths.output_dir / "comparison_table.png")

    print("\nDone. Artifacts saved to:", paths.output_dir)


if __name__ == "__main__":
    main()
