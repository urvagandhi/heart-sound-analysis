"""
Review 3 — Data Module
======================
Loads Phase 1 outputs (metadata.csv + cached .npy spectrograms) into PyTorch
Datasets and DataLoaders. Uses split_indices.json when available so the
train/val/test split matches the one recorded in Review 2; otherwise falls
back to a stratified split.

Key design choices
------------------
* SpecAugment is applied in spectrogram (log-mel) space BEFORE ImageNet
  normalisation — zeroing in [0, 1] dB space means "silence", which matches
  the SpecAugment paper.
* Grayscale spectrograms are replicated across 3 channels and normalised
  with ImageNet statistics, matching the pretrained ResNet50 expectation.
* All paths are resolved against ``Paths`` so the module is portable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import Paths, AudioConfig, ModelConfig, TrainConfig, CLASS_NAMES


# ─────────────────────────────────────────────────────────────
# METADATA
# ─────────────────────────────────────────────────────────────

def load_metadata(paths: Paths, verify_spectrograms: bool = True) -> pd.DataFrame:
    """Load the metadata.csv produced by Phase 1.

    Verifies that every referenced .npy spectrogram exists on disk; drops
    rows whose spectrogram is missing (with a warning).
    """
    if not paths.metadata_csv.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {paths.metadata_csv}. "
            "Run the Phase 1 notebook first to generate it."
        )

    df = pd.read_csv(paths.metadata_csv)

    required_cols = {"filename", "label", "subset", "unique_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    df["label"] = df["label"].astype(int)

    if verify_spectrograms:
        spec_exists = df["unique_id"].apply(
            lambda uid: (paths.spec_dir / f"{uid}.npy").exists()
        )
        n_missing = int((~spec_exists).sum())
        if n_missing > 0:
            print(f"  Warning: {n_missing} spectrograms missing — dropped.")
            df = df[spec_exists].reset_index(drop=True)

    return df


def load_class_weights(paths: Paths, df: pd.DataFrame,
                       device: str = "cpu") -> torch.Tensor:
    """Load class weights from class_weights.pt if present, else compute."""
    if paths.class_weights_pt.exists():
        w = torch.load(paths.class_weights_pt, map_location=device,
                       weights_only=True)
        return w.to(device).float()

    counts = df["label"].value_counts().sort_index().values.astype(np.float32)
    w = 1.0 / counts
    w = w / w.sum()
    return torch.tensor(w, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────

class HeartSoundDataset(Dataset):
    """Loads pre-saved .npy log-mel spectrograms and converts them into the
    3-channel 224×224 tensors expected by an ImageNet-pretrained ResNet50.

    Augmentation pipeline (training only):
        spectrogram (128, T)
          → SpecAugment (freq + time masks, in [0, 1] dB space)
          → bilinear resize to (224, 224)
          → replicate to 3 channels
          → ImageNet normalise

    Inference pipeline skips the SpecAugment step.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 paths: Paths,
                 model_cfg: ModelConfig,
                 augment: bool = False,
                 freq_mask_max: int = 18,
                 time_mask_max: int = 25,
                 n_masks: int = 2):
        self.df       = df.reset_index(drop=True)
        self.spec_dir = paths.spec_dir
        self.img_size = model_cfg.img_size
        self.augment  = augment
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.n_masks  = n_masks
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )

    def __len__(self) -> int:
        return len(self.df)

    def _spec_augment(self, spec: np.ndarray) -> np.ndarray:
        """SpecAugment-style time + frequency masking on the (n_mels, T)
        log-mel spectrogram. Values are already in [0, 1] dB space."""
        spec = spec.copy()
        n_mels, n_frames = spec.shape

        for _ in range(self.n_masks):
            # Frequency mask
            f = np.random.randint(1, max(2, self.freq_mask_max))
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = 0.0

            # Time mask
            t = np.random.randint(1, max(2, self.time_mask_max))
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[:, t0:t0 + t] = 0.0

        return spec

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        spec_path = self.spec_dir / f"{row['unique_id']}.npy"
        spec = np.load(spec_path).astype(np.float32)        # (n_mels, T)

        if self.augment:
            spec = self._spec_augment(spec)

        # Bilinear resize to (img_size, img_size)
        spec_t = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)   # (1,1,M,T)
        spec_t = torch.nn.functional.interpolate(
            spec_t, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)                                                # (1,H,W)

        # Replicate to 3 channels and ImageNet-normalise
        spec_t = spec_t.repeat(3, 1, 1)
        spec_t = self.normalize(spec_t)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return spec_t, label


# ─────────────────────────────────────────────────────────────
# SPLITS
# ─────────────────────────────────────────────────────────────

def load_split(paths: Paths,
               df: pd.DataFrame,
               seed: int = 42,
               val_size: float = 0.15,
               test_size: float = 0.15
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resolve a train/val/test split.

    Priority:
      1. ``split_indices.json`` from Phase 1 (preserves Review 2 reproducibility).
      2. Stratified random split with the supplied seed.

    The metadata DataFrame must already be cleaned (missing spectrograms
    dropped) so its row order matches what Phase 1 saved.
    """
    if paths.split_indices_json.exists():
        with open(paths.split_indices_json) as f:
            splits = json.load(f)

        tr_idx = splits["train_indices"]
        va_idx = splits["val_indices"]
        te_idx = splits["test_indices"]

        # Sanity: indices must be valid for the current df
        max_idx = max(max(tr_idx), max(va_idx), max(te_idx))
        if max_idx >= len(df):
            print(f"  Warning: saved indices reference row {max_idx} but df "
                  f"has only {len(df)} rows. Falling back to stratified split.")
        else:
            train_df = df.iloc[tr_idx].reset_index(drop=True)
            val_df   = df.iloc[va_idx].reset_index(drop=True)
            test_df  = df.iloc[te_idx].reset_index(drop=True)
            print(f"  Loaded splits from {paths.split_indices_json.name}")
            return train_df, val_df, test_df

    # Fallback: stratified split
    from sklearn.model_selection import train_test_split
    print(f"  No split file found — computing stratified split (seed={seed}).")
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df["label"], random_state=seed,
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test,
        stratify=temp_df["label"], random_state=seed,
    )
    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame,
                  test_df: pd.DataFrame) -> None:
    print(f"  Train: {len(train_df):>4d} | "
          f"Val: {len(val_df):>4d} | Test: {len(test_df):>4d}")
    for name, dfs in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = dfs["label"].value_counts().sort_index().to_dict()
        pretty = {CLASS_NAMES[k]: v for k, v in dist.items()}
        print(f"  {name:5s} class dist: {pretty}")


# ─────────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────────

def get_dataloaders(paths: Paths,
                    model_cfg: ModelConfig,
                    train_cfg: TrainConfig,
                    df: Optional[pd.DataFrame] = None,
                   ) -> Tuple[DataLoader, DataLoader, DataLoader,
                              pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df is None:
        df = load_metadata(paths)

    train_df, val_df, test_df = load_split(paths, df, seed=train_cfg.seed)
    split_summary(train_df, val_df, test_df)

    train_ds = HeartSoundDataset(train_df, paths, model_cfg, augment=True)
    val_ds   = HeartSoundDataset(val_df,   paths, model_cfg, augment=False)
    test_ds  = HeartSoundDataset(test_df,  paths, model_cfg, augment=False)

    pin = torch.cuda.is_available()
    # ``persistent_workers=True`` keeps the worker processes alive across
    # epochs. Without it, Colab/Jupyter sporadically spams the cell output
    # with harmless `_MultiProcessingDataLoaderIter.__del__` AssertionErrors
    # ("can only test a child process") triggered by GC cycling the workers.
    persistent = train_cfg.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              shuffle=True,  num_workers=train_cfg.num_workers,
                              pin_memory=pin, drop_last=False,
                              persistent_workers=persistent)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=train_cfg.num_workers,
                              pin_memory=pin,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_ds,  batch_size=train_cfg.batch_size,
                              shuffle=False, num_workers=train_cfg.num_workers,
                              pin_memory=pin,
                              persistent_workers=persistent)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
