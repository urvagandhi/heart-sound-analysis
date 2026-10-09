"""Unit tests for PhysioNet state annotation enrichment ratio calculation."""

import numpy as np
import pytest

from audit.enrichment import STATE_NAMES, compute_enrichment_ratio


def test_uniform_map_enrichment_equals_one() -> None:
    """Verify that a uniform saliency map yields enrichment ratio E_s = 1.0 for all states."""
    # Uniform CAM map across all 224x224 cells
    uniform_cam = np.ones((224, 224), dtype=np.float32)

    # Hand-made state sequence (8,000 samples at 1000 Hz, or 2,240 samples: 10 samples per column)
    # 4 acoustic states equally partitioned: S1 (1), systole (2), S2 (3), diastole (4)
    samples_per_col = 10
    total_cols = 224
    cols_per_state = total_cols // 4  # 56 columns each

    state_seq = []
    for state_code in range(1, 5):
        state_seq.extend([state_code] * (cols_per_state * samples_per_col))
    state_sequence = np.array(state_seq, dtype=np.int32)

    ratios = compute_enrichment_ratio(uniform_cam, state_sequence)

    for state in STATE_NAMES:
        assert state in ratios
        assert ratios[state] == pytest.approx(1.0, abs=1e-3), (
            f"Expected E_s = 1.0 for uniform map on state '{state}', got {ratios[state]}"
        )


def test_concentrated_map_enrichment() -> None:
    """Verify that concentrating CAM mass in S1 region boosts S1 enrichment above 1.0."""
    cam = np.zeros((224, 224), dtype=np.float32)
    # Put all attribution mass in first 56 columns (S1)
    cam[:, :56] = 1.0

    samples_per_col = 10
    total_cols = 224
    cols_per_state = total_cols // 4

    state_seq = []
    for state_code in range(1, 5):
        state_seq.extend([state_code] * (cols_per_state * samples_per_col))
    state_sequence = np.array(state_seq, dtype=np.int32)

    ratios = compute_enrichment_ratio(cam, state_sequence)

    # S1 covers 25% of time, but holds 100% of CAM mass -> E_s = 1.0 / 0.25 = 4.0
    assert ratios["S1"] == pytest.approx(4.0, abs=1e-2)
    # Other states have 0 CAM mass -> E_s = 0.0
    assert ratios["systole"] == pytest.approx(0.0, abs=1e-2)
    assert ratios["S2"] == pytest.approx(0.0, abs=1e-2)
    assert ratios["diastole"] == pytest.approx(0.0, abs=1e-2)
