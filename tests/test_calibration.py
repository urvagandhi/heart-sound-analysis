"""Unit tests for ECE calibration and clinical operating point threshold selection."""

import numpy as np
import pytest

from audit.calibration import compute_ece, select_operating_point


def test_compute_ece_perfect_and_uncalibrated() -> None:
    """Verify ECE computation on perfectly calibrated and miscalibrated distributions."""
    # Perfectly calibrated dummy predictions: 100 samples with prob 0.8, exactly 80 are class 1
    probs_perfect = np.zeros((100, 2))
    probs_perfect[:, 0] = 0.2
    probs_perfect[:, 1] = 0.8
    labels_perfect = np.array([1] * 80 + [0] * 20)

    ece_perf, _ = compute_ece(probs_perfect, labels_perfect, n_bins=10)
    assert ece_perf == pytest.approx(0.0, abs=1e-3)

    # Completely overconfident wrong predictions: prob 0.99 for class 1, but all true label 0
    probs_wrong = np.zeros((100, 2))
    probs_wrong[:, 0] = 0.01
    probs_wrong[:, 1] = 0.99
    labels_wrong = np.zeros(100, dtype=int)

    ece_wrong, _ = compute_ece(probs_wrong, labels_wrong, n_bins=10)
    assert ece_wrong > 0.90


def test_select_operating_point_targets_sensitivity() -> None:
    """Verify that select_operating_point chooses a threshold achieving >= 0.90 sensitivity."""
    rng = np.random.default_rng(42)

    # Validation set: 50 negatives (probs ~ Beta(2, 6)), 50 positives (probs ~ Beta(6, 2))
    val_neg_probs = rng.beta(2, 6, size=50)
    val_pos_probs = rng.beta(6, 2, size=50)
    val_labels = np.array([0] * 50 + [1] * 50)
    val_p1 = np.concatenate([val_neg_probs, val_pos_probs])
    val_probs = np.column_stack([1.0 - val_p1, val_p1])

    # Test set: similar distribution
    test_neg_probs = rng.beta(2, 6, size=50)
    test_pos_probs = rng.beta(6, 2, size=50)
    test_labels = np.array([0] * 50 + [1] * 50)
    test_p1 = np.concatenate([test_neg_probs, test_pos_probs])
    test_probs = np.column_stack([1.0 - test_p1, test_p1])

    target_sens = 0.90
    op = select_operating_point(val_probs, val_labels, test_probs, test_labels, target_sensitivity=target_sens)

    thresh = op["selected_threshold"]
    assert 0.0 < thresh < 1.0

    # Validation sensitivity at selected threshold must achieve the target (>= 0.90)
    val_tp = int(np.sum((val_pos_probs >= thresh)))
    val_sens = val_tp / 50.0
    assert val_sens >= target_sens, f"Validation sensitivity {val_sens:.2f} must be >= {target_sens}"

    # Returned test metrics must be well-formed
    assert 0.0 <= op["test_sensitivity"] <= 1.0
    assert 0.0 <= op["test_specificity"] <= 1.0
    assert 0.0 <= op["test_balanced_accuracy"] <= 1.0
