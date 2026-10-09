"""Unit tests for make_splits invariants on synthetic metadata tables."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from make_splits import make_dedup_splits, make_lodo_splits


def test_make_splits_dedup_invariants(tmp_path: Path) -> None:
    """Verify that make_dedup_splits enforces completeness, disjointness, and duplicate removal."""
    # Create synthetic dataset with 3,541 rows matching PhysioNet structure
    n_kept = 3240
    n_val = 301

    kept_filenames = [f"rec_{i:04d}" for i in range(n_kept)]
    kept_labels = [i % 2 for i in range(n_kept)]
    subsets = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
    kept_subsets = [subsets[i % len(subsets)] for i in range(n_kept)]

    # 301 validation rows copied from the first 301 kept rows
    val_filenames = kept_filenames[:n_val]
    val_labels = kept_labels[:n_val]
    val_subsets = ["validation"] * n_val

    all_filenames = kept_filenames + val_filenames
    all_labels = kept_labels + val_labels
    all_subsets = kept_subsets + val_subsets

    df_synth = pd.DataFrame({
        "filename": all_filenames,
        "label": all_labels,
        "subset": all_subsets,
        "wav_path": [f"{s}/{f}.wav" for f, s in zip(all_filenames, all_subsets)],
        "unique_id": [f"{s}_{f}" for f, s in zip(all_filenames, all_subsets)],
    })

    seeds = [42, 43]
    df_dedup = make_dedup_splits(df_synth, seeds=seeds, data_dir=tmp_path)

    # 1. Row count assertion
    assert len(df_dedup) == 3240
    assert (df_dedup["subset"] != "validation").all()

    # 2. Check generated split JSONs
    splits_dir = tmp_path / "splits"
    assert splits_dir.exists()

    for s in seeds:
        split_file = splits_dir / f"dedup_seed{s}.json"
        assert split_file.exists()
        with open(split_file, "r") as f:
            data = json.load(f)

        tr = set(data["train_indices"])
        va = set(data["val_indices"])
        te = set(data["test_indices"])

        # Disjointness
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)

        # Completeness
        assert len(tr) + len(va) + len(te) == 3240
        assert tr.union(va).union(te) == set(range(3240))

        # Metadata key present
        assert "meta" in data
        assert data["meta"]["seed"] == s
        assert data["meta"]["mode"] == "dedup"


def test_make_splits_small_strata_fallback(tmp_path: Path) -> None:
    """Verify fallback to label-only stratification when any stratum has < 3 rows."""
    # Synthetic dataset with a tiny stratum (e.g. 1 sample in a subset)
    n_samples = 3240
    filenames = [f"rec_{i:04d}" for i in range(n_samples)]
    labels = [i % 2 for i in range(n_samples)]
    # Make subset 'training-c' have only 2 samples for label 0
    subsets = ["training-a"] * (n_samples - 2) + ["training-c", "training-c"]
    labels[-2] = 0
    labels[-1] = 0

    df_synth = pd.DataFrame({
        "filename": filenames,
        "label": labels,
        "subset": subsets,
        "wav_path": [f"{s}/{f}.wav" for f, s in zip(filenames, subsets)],
        "unique_id": [f"{s}_{f}" for f, s in zip(filenames, subsets)],
    })

    # Must complete successfully with warning printed
    df_dedup = make_dedup_splits(df_synth, seeds=[42], data_dir=tmp_path)
    assert len(df_dedup) == 3240


def test_make_lodo_splits(tmp_path: Path) -> None:
    """Verify that make_lodo_splits creates disjoint patient-held-out splits."""
    n_samples = 600
    subsets = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
    df_synth = pd.DataFrame({
        "filename": [f"f_{i}" for i in range(n_samples)],
        "label": [i % 2 for i in range(n_samples)],
        "subset": [subsets[i % len(subsets)] for i in range(n_samples)],
    })

    make_lodo_splits(df_synth, holdouts=["a", "b"], seed=42, data_dir=tmp_path)

    split_file = tmp_path / "splits" / "lodo_holda_seed42.json"
    assert split_file.exists()
    with open(split_file, "r") as f:
        data = json.load(f)

    tr = set(data["train_indices"])
    va = set(data["val_indices"])
    te = set(data["test_indices"])

    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)
    assert len(tr) + len(va) + len(te) == n_samples

    # Check test set contains all and only training-a samples
    test_subsets = df_synth.iloc[list(te)]["subset"].unique()
    assert list(test_subsets) == ["training-a"]

    # Check training and validation sets do not contain training-a
    train_subsets = df_synth.iloc[list(tr)]["subset"].unique()
    val_subsets = df_synth.iloc[list(va)]["subset"].unique()
    assert "training-a" not in train_subsets
    assert "training-a" not in val_subsets
