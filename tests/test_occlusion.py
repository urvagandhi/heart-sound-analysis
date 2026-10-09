"""Unit tests for frequency band occlusion audit using a synthetic model."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from audit.common import get_band_row_ranges, to_norm, to_unit
from audit.occlusion import run_band_occlusion_audit
from config import AudioConfig


class ToyBandModel(torch.nn.Module):
    """Toy model whose predictions depend exclusively on energy in a target frequency band."""

    def __init__(self, target_band_rows: tuple[int, int]) -> None:
        super().__init__()
        self.r0, self.r1 = target_band_rows

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = to_unit(x)
        # Calculate mean energy within the target frequency band
        band_energy = u[:, :, self.r0:self.r1, :].mean(dim=(1, 2, 3))
        logit_1 = (band_energy - 0.25) * 20.0
        logit_0 = -logit_1
        return torch.stack([logit_0, logit_1], dim=1)


def test_occlusion_flags_target_band() -> None:
    """Verify that frequency band occlusion correctly flags only the band the model relies on."""
    audio_cfg = AudioConfig()
    bands, _ = get_band_row_ranges(audio_cfg)
    target_band = "B1"
    target_rows = bands[target_band]

    model = ToyBandModel(target_rows)
    model.eval()

    # Generate synthetic dataset: 10 normal and 10 abnormal samples
    n_samples = 20
    images = torch.full((n_samples, 3, 224, 224), 0.05)
    labels = torch.zeros(n_samples, dtype=torch.long)

    # Abnormal samples (second half): set high energy in B1
    for i in range(10, 20):
        labels[i] = 1
        images[i, :, target_rows[0]:target_rows[1], :] = 0.80

    norm_images = to_norm(images)
    dataset = TensorDataset(norm_images, labels)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    df = pd.DataFrame({
        "filename": [f"rec_{i}" for i in range(n_samples)],
        "label": labels.numpy(),
        "subset": ["training-a"] * 10 + ["training-b"] * 10,
    })

    results = run_band_occlusion_audit(
        model=model,  # type: ignore[arg-type]
        test_loader=loader,
        test_df=df,
        device="cpu",
        audio_cfg=audio_cfg,
        n_random_controls=20,
        seed=42,
    )

    # Assertions
    reliances = results["reliances"]
    assert reliances["B1"] is True, "Target band B1 must be flagged as relied upon"
    assert reliances["B2"] is False, "Non-target band B2 must not be flagged"
    assert reliances["B3"] is False, "Non-target band B3 must not be flagged"
    assert results["unmasked"]["balanced_accuracy"] == 1.0
