"""
Review 3 — Configuration
========================
Central place for paths, audio constants, model hyperparams and training
hyperparams. Path detection automatically handles Google Colab (Drive
mounted at /content/drive) and local execution (relative to repo root).

Override any value at runtime via the helper builders:

    from config import Paths, AudioConfig, ModelConfig, TrainConfig
    paths = Paths.auto_detect()
    paths = Paths.auto_detect(base_override="/some/custom/path")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ─────────────────────────────────────────────────────────────

def is_colab() -> bool:
    """Detect Google Colab. Looks for the IPython colab kernel module."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_drive_mounted() -> bool:
    """Check whether /content/drive/MyDrive is currently mounted."""
    return Path("/content/drive/MyDrive").exists()


# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

# Drive layout used by the Phase 1 notebook:
#   /content/drive/MyDrive/CSE/College/RMS/physionet_2016/{raw,spectrograms,...}
DEFAULT_DRIVE_BASE = Path("/content/drive/MyDrive/CSE/College/RMS")

# Local layout assumes Review 3/ sits next to physionet_2016/ at the repo root.
DEFAULT_LOCAL_BASE = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    """All paths derived from a single ``base_dir`` so the pipeline runs in
    either Drive or local mode by swapping one root."""

    base_dir: Path
    data_dir: Path
    raw_dir: Path
    spec_dir: Path
    metadata_csv: Path
    split_indices_json: Path
    class_weights_pt: Path
    output_dir: Path
    checkpoint_dir: Path

    @staticmethod
    def from_base(base_dir: Path, output_subdir: str = "Review 3/output") -> "Paths":
        base = Path(base_dir)
        data = base / "physionet_2016"
        out  = base / output_subdir
        return Paths(
            base_dir           = base,
            data_dir           = data,
            raw_dir            = data / "raw",
            spec_dir           = data / "spectrograms",
            metadata_csv       = data / "metadata.csv",
            split_indices_json = data / "split_indices.json",
            class_weights_pt   = data / "class_weights.pt",
            output_dir         = out,
            checkpoint_dir     = out / "checkpoints",
        )

    @staticmethod
    def auto_detect(base_override: Optional[str] = None) -> "Paths":
        """Pick Drive base on Colab (if mounted), else local base."""
        if base_override:
            base = Path(base_override).expanduser().resolve()
        elif is_colab() and is_drive_mounted():
            base = DEFAULT_DRIVE_BASE
        else:
            base = DEFAULT_LOCAL_BASE
        return Paths.from_base(base)

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def __str__(self) -> str:
        lines = [f"  base_dir           = {self.base_dir}"]
        for k, v in asdict(self).items():
            if k == "base_dir":
                continue
            exists = "[OK]" if Path(v).exists() else "[--]"
            lines.append(f"  {k:18s} = {v}  {exists}")
        return "Paths:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# AUDIO / SPECTROGRAM CONSTANTS (must match Phase 1)
# ─────────────────────────────────────────────────────────────

@dataclass
class AudioConfig:
    sr_target: int  = 4000
    duration: float = 8.0
    n_fft: int      = 512
    hop_length: int = 128
    n_mels: int     = 128
    fmin: int       = 25      # aligned with bandpass low cutoff
    fmax: int       = 1000


# ─────────────────────────────────────────────────────────────
# MODEL CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    num_classes: int    = 2
    num_heads: int      = 8
    mha_dropout: float  = 0.1
    head_dropout: float = 0.4
    se_reduction: int   = 16
    img_size: int       = 224          # ResNet50 input
    feature_dim: int    = 2048         # ResNet50 final channel count


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    batch_size: int        = 32
    # ``num_workers=0`` keeps DataLoader single-process: avoids Colab's
    # ``_MultiProcessingDataLoaderIter.__del__`` AssertionError spam and is
    # plenty fast for cached .npy spectrograms. Bump to 2 only when you have
    # a non-Colab Linux GPU and want extra throughput.
    num_workers: int       = 0         # Local - 2 | Colab - 0
    num_epochs: int        = 40
    warmup_epochs: int     = 10        # frozen backbone phase
    lr_head: float         = 3e-4      # phase A LR (head + SE + MHA)
    lr_full: float         = 3e-5      # phase B LR (all params)
    weight_decay: float    = 1e-4
    grad_clip: float       = 1.0
    early_stop_patience: int = 8
    label_smoothing: float = 0.05
    seed: int              = 42
    use_class_weights: bool = True
    use_amp: bool          = True      # mixed precision when on CUDA


CLASS_NAMES = ["Normal", "Abnormal"]   # index 0 = Normal, 1 = Abnormal


# ─────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Deterministic seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
