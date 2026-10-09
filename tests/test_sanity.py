"""Unit tests for Adebayo model randomization sanity check on a toy CNN."""

import numpy as np
import pytest
import scipy.stats as stats
import torch
import torch.nn as nn

from audit.sanity import reinit_module
from xai import GradCAM


class ToyCNN(nn.Module):
    """Simple 2-layer CNN with global average pooling and linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).flatten(1)
        return self.fc(h)


def test_adebayo_randomization_drops_correlation() -> None:
    """Verify that cascading layer re-initialization drops Grad-CAM rank correlation on a toy CNN."""
    torch.manual_seed(42)
    model = ToyCNN()
    model.eval()

    x = torch.randn(1, 3, 224, 224, requires_grad=True)

    # 1. Baseline Grad-CAM attribution
    cam_engine = GradCAM(model, model.conv2)
    cam_orig = cam_engine.generate(x, target_class=0)
    cam_engine.remove()

    # Self-correlation check
    r_self, _ = stats.spearmanr(cam_orig.flatten(), cam_orig.flatten())
    assert abs(r_self - 1.0) < 1e-5

    # 2. Randomize top layer (fc) and second layer (conv2)
    reinit_module(model.fc)
    reinit_module(model.conv2)

    # 3. Grad-CAM after randomization
    cam_engine = GradCAM(model, model.conv2)
    cam_rand = cam_engine.generate(x, target_class=0)
    cam_engine.remove()

    # Compute Spearman rank correlation
    r_rand, _ = stats.spearmanr(cam_orig.flatten(), cam_rand.flatten())
    if np.isnan(r_rand):
        r_rand = 0.0

    # Randomization should break the attribution map correlation
    assert r_rand < 0.60, f"Expected Spearman correlation < 0.60 after randomization, got {r_rand:.4f}"
