"""Unit tests for explanation faithfulness evaluation on a synthetic model."""

import numpy as np
import pytest
import torch

from audit.common import to_norm, to_unit
from audit.faithfulness import curve_auc, evaluate_perturbation_curves


class ToySaliencyModel(torch.nn.Module):
    """Model whose class 1 probability depends strictly on a 10x10 patch at the top-left."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = to_unit(x)
        # Class 1 depends on activation in top-left 10x10 region
        patch_mean = u[:, :, :10, :10].mean(dim=(1, 2, 3))
        logit_1 = (patch_mean - 0.2) * 10.0
        logit_0 = -logit_1
        return torch.stack([logit_0, logit_1], dim=1)


def test_deletion_auc_lower_than_random_for_faithful_map() -> None:
    """Verify that removing top-scoring features produces lower deletion AUC than a random map."""
    model = ToySaliencyModel()
    model.eval()

    # Input image with high values everywhere
    x_unit = torch.ones((3, 224, 224), dtype=torch.float32)

    # Ground truth / faithful saliency map: high only in the 10x10 active region
    faithful_map = np.zeros((224, 224), dtype=np.float32)
    faithful_map[:10, :10] = 1.0

    # Random saliency map
    rng = np.random.default_rng(42)
    random_map = rng.random((224, 224)).astype(np.float32)

    p_steps = [float(p / 100.0) for p in range(0, 55, 5)]

    del_scores_faithful, _ = evaluate_perturbation_curves(
        model=model,  # type: ignore[arg-type]
        x_unit=x_unit,
        saliency_map=faithful_map,
        intact_class=1,
        device="cpu",
        p_steps=p_steps,
    )
    auc_faithful = curve_auc(del_scores_faithful, p_steps)

    del_scores_random, _ = evaluate_perturbation_curves(
        model=model,  # type: ignore[arg-type]
        x_unit=x_unit,
        saliency_map=random_map,
        intact_class=1,
        device="cpu",
        p_steps=p_steps,
    )
    auc_random = curve_auc(del_scores_random, p_steps)

    # Deletion AUC of faithful map must be substantially lower than random
    assert auc_faithful < auc_random, (
        f"Faithful deletion AUC ({auc_faithful:.4f}) must be less than random ({auc_random:.4f})"
    )
    assert auc_faithful < 0.20, f"Expected faithful deletion AUC < 0.20, got {auc_faithful:.4f}"
