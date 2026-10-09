"""Common audit utilities: model loading, unit-space conversions, mel-row mapping, and bootstrap CIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from config import AudioConfig, ModelConfig, Paths
from model import HeartSoundModel, build_model

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_model(tag: str,
               results_dir: Optional[Path] = None,
               base_dir: Optional[Path] = None,
               device: str = "cpu") -> Tuple[HeartSoundModel, ModelConfig, Dict]:
    """Load a trained HeartSoundModel from its run tag directory.

    Args:
        tag: Run identifier (e.g. 'dedup_se_mha_seed42' or 'legacy_se_mha').
        results_dir: Path to directory containing run folders.
        base_dir: Base repository directory.
        device: Target execution device.

    Returns:
        Tuple of (model, ModelConfig, config_dict).
    """
    repo_root = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent
    root = Path(results_dir) if results_dir else repo_root / "results"
    run_dir = root / tag

    # Fallback for legacy run if results/legacy_se_mha doesn't exist yet
    if not run_dir.exists() and tag == "legacy_se_mha":
        legacy_dir = repo_root / "Review 3" / "output"
        if legacy_dir.exists():
            run_dir = legacy_dir

    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "best_model.pth"

    variant = "se_mha"
    config_dict: Dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        variant = config_dict.get("variant", config_dict.get("args", {}).get("variant", "se_mha"))

    model_cfg = ModelConfig(variant=variant)
    model = HeartSoundModel(model_cfg, pretrained=False)

    if ckpt_path.exists():
        raw = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = raw["model_state_dict"] if (isinstance(raw, dict) and "model_state_dict" in raw) else raw
        model.load_state_dict(state_dict, strict=True)
        print(f"  Loaded model for tag '{tag}' (variant: {variant}) from {ckpt_path}")
    else:
        print(f"  Warning: Checkpoint not found at {ckpt_path}; returning uninitialized model.")

    model.to(device)
    model.eval()
    return model, model_cfg, config_dict


def to_unit(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalisation back to unit [0, 1] spectrogram space."""
    dev = tensor.device
    mean = IMAGENET_MEAN.to(dev)
    std = IMAGENET_STD.to(dev)
    if tensor.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def to_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Reapply ImageNet normalisation from unit [0, 1] spectrogram space."""
    dev = tensor.device
    mean = IMAGENET_MEAN.to(dev)
    std = IMAGENET_STD.to(dev)
    if tensor.ndim == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)
    return (tensor - mean) / std


def get_band_row_ranges(audio_cfg: Optional[AudioConfig] = None,
                        img_h: int = 224) -> Tuple[Dict[str, Tuple[int, int]], np.ndarray]:
    """Compute exact row slices for frequency bands B1, B2, B3 on resized (224x224) spectrograms.

    Bands:
      B1: [fmin, 150) Hz
      B2: [150, 500) Hz
      B3: [500, fmax] Hz

    Returns:
        Tuple of (band_dict, row_to_hz_array).
    """
    import librosa
    cfg = audio_cfg or AudioConfig()

    assert cfg.fmin == 20, f"Expected fmin=20 Hz matching cached spectrograms, got {cfg.fmin}"
    assert cfg.fmax == 1000, f"Expected fmax=1000 Hz matching cached spectrograms, got {cfg.fmax}"
    print(f"  [Mel-Hz Alignment] Slaney mel scale fmin={cfg.fmin} Hz, fmax={cfg.fmax} Hz, n_mels={cfg.n_mels}")

    # Center frequencies of 128 mel bins (Slaney scale)
    mf = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)

    # Resized row indices 0..223 mapped to Hz
    row_indices = np.arange(img_h)
    mel_indices = row_indices / (img_h - 1) * (cfg.n_mels - 1)
    row_to_hz = np.interp(mel_indices, np.arange(cfg.n_mels), mf)

    # Identify row boundaries
    # B1: [fmin, 150)
    # B2: [150, 500)
    # B3: [500, fmax]
    b1_mask = (row_to_hz >= cfg.fmin) & (row_to_hz < 150.0)
    b2_mask = (row_to_hz >= 150.0) & (row_to_hz < 500.0)
    b3_mask = (row_to_hz >= 500.0) & (row_to_hz <= cfg.fmax + 1e-4)

    b1_start, b1_end = int(np.where(b1_mask)[0][0]), int(np.where(b1_mask)[0][-1]) + 1
    b2_start, b2_end = int(np.where(b2_mask)[0][0]), int(np.where(b2_mask)[0][-1]) + 1
    b3_start, b3_end = int(np.where(b3_mask)[0][0]), int(np.where(b3_mask)[0][-1]) + 1

    bands = {
        "B1": (b1_start, b1_end),   # [fmin, 150) Hz
        "B2": (b2_start, b2_end),   # [150, 500) Hz
        "B3": (b3_start, b3_end),   # [500, fmax] Hz
    }
    return bands, row_to_hz


def bootstrap_ci(data: np.ndarray | Sequence[float],
                 n_resamples: int = 2000,
                 ci: float = 0.95,
                 seed: int = 42,
                 stat_fn: Callable[[np.ndarray], float] = np.mean) -> Tuple[float, float, float]:
    """Compute non-parametric bootstrap confidence interval for a statistic.

    Args:
        data: Array or sequence of scalar measurements.
        n_resamples: Number of bootstrap iterations (default 2000).
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility (default 42).
        stat_fn: Summary statistic function (default np.mean).

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        val = float(stat_fn(arr))
        return val, val, val

    point = float(stat_fn(arr))
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_stats = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, 100.0 * alpha))
    high = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha)))
    return point, low, high
