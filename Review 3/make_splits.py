"""Split generation for deduplicated benchmark and Leave-One-Database-Out (LODO).

Supports:
  --mode dedup --seeds 42 43 44:
    Writes physionet_2016/metadata_dedup.csv (dropping subset == 'validation', asserting 3,240 rows,
    asserting every dropped row's (filename, label) exists among kept rows).
    Writes physionet_2016/splits/dedup_seed{S}.json (70/15/15 stratified on label+subset jointly;
    falls back to label-only if any stratum has fewer than 3 rows; asserts disjoint and complete).

  --mode lodo --holdouts a b e f --seed 42:
    Writes physionet_2016/splits/lodo_hold{X}_seed42.json where test = all rows of training-X,
    remaining rows -> train 85% / val 15% stratified on label+subset. Databases c and d are always in train.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


def make_dedup_splits(df: pd.DataFrame,
                      seeds: Sequence[int],
                      data_dir: Path) -> pd.DataFrame:
    """Generate deduplicated metadata and 70/15/15 seed-replicated splits."""
    total_rows = len(df)
    val_rows = df[df["subset"] == "validation"]
    df_dedup = df[df["subset"] != "validation"].copy().reset_index(drop=True)

    # Invariants
    assert len(df_dedup) == 3240, f"Expected 3,240 deduplicated rows, got {len(df_dedup)}"
    kept_pairs = set(df_dedup[["filename", "label"]].itertuples(index=False, name=None))
    for r in val_rows.itertuples(index=False):
        assert (r.filename, r.label) in kept_pairs, (
            f"Validation row ({r.filename}, {r.label}) missing from kept rows!"
        )

    out_meta_path = data_dir / "metadata_dedup.csv"
    df_dedup.to_csv(out_meta_path, index=False)
    print(f"Saved deduplicated metadata: {out_meta_path} ({len(df_dedup)} rows)")

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        strata = df_dedup["subset"].astype(str) + "_" + df_dedup["label"].astype(str)
        counts = strata.value_counts()
        if (counts < 3).any():
            print(f"  [Seed {seed}] Warning: stratum has < 3 rows; falling back to label-only stratification.")
            strat_col = df_dedup["label"]
        else:
            strat_col = strata

        # 70% train, 30% temp -> 15% val, 15% test
        train_idx, temp_idx = train_test_split(
            df_dedup.index, test_size=0.30, random_state=seed, stratify=strat_col
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=seed, stratify=strat_col.loc[temp_idx]
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        # Assert disjoint and complete
        assert train_set.isdisjoint(val_set), "Train and Val overlap!"
        assert train_set.isdisjoint(test_set), "Train and Test overlap!"
        assert val_set.isdisjoint(test_set), "Val and Test overlap!"
        assert len(train_set) + len(val_set) + len(test_set) == len(df_dedup), "Partitions incomplete!"

        split_dict = {
            "train_indices": sorted(train_idx.tolist()),
            "val_indices":   sorted(val_idx.tolist()),
            "test_indices":  sorted(test_idx.tolist()),
            "meta": {
                "seed": seed,
                "mode": "dedup",
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "total": len(df_dedup),
            },
        }

        split_file = splits_dir / f"dedup_seed{seed}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)
        print(f"  Saved {split_file.name}: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return df_dedup


def make_lodo_splits(df_dedup: pd.DataFrame,
                     holdouts: Sequence[str],
                     seed: int,
                     data_dir: Path) -> None:
    """Generate leave-one-database-out (LODO) patient-disjoint splits."""
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for h in holdouts:
        holdout_subset = f"training-{h}"
        test_mask = df_dedup["subset"] == holdout_subset
        assert test_mask.any(), f"No rows found for holdout subset '{holdout_subset}'"

        test_idx = df_dedup[test_mask].index.tolist()
        rem_idx = df_dedup[~test_mask].index

        rem_strata = df_dedup.loc[rem_idx, "subset"].astype(str) + "_" + df_dedup.loc[rem_idx, "label"].astype(str)
        counts = rem_strata.value_counts()
        if (counts < 3).any():
            print(f"  [LODO holdout {h}] Warning: stratum has < 3 rows; falling back to label-only stratification.")
            strat_col = df_dedup.loc[rem_idx, "label"]
        else:
            strat_col = rem_strata

        # 85% train, 15% val
        train_idx, val_idx = train_test_split(
            rem_idx, test_size=0.15, random_state=seed, stratify=strat_col
        )

        train_set = set(train_idx)
        val_set = set(val_idx)
        test_set = set(test_idx)

        assert train_set.isdisjoint(val_set), "Train and Val overlap in LODO!"
        assert train_set.isdisjoint(test_set), "Train and Test overlap in LODO!"
        assert val_set.isdisjoint(test_set), "Val and Test overlap in LODO!"
        assert len(train_set) + len(val_set) + len(test_set) == len(df_dedup), "Partitions incomplete in LODO!"

        split_dict = {
            "train_indices": sorted(train_idx.tolist()),
            "val_indices":   sorted(val_idx.tolist()),
            "test_indices":  sorted(test_idx),
            "meta": {
                "seed": seed,
                "mode": "lodo",
                "holdout": h,
                "holdout_subset": holdout_subset,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "total": len(df_dedup),
            },
        }

        split_file = splits_dir / f"lodo_hold{h}_seed{seed}.json"
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split_dict, f, indent=2)
        print(f"  Saved {split_file.name}: Holdout={h} ({len(test_idx)} test), Train={len(train_idx)}, Val={len(val_idx)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deduplicated and LODO splits.")
    parser.add_argument("--mode", type=str, choices=["dedup", "lodo", "all"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--holdouts", type=str, nargs="+", default=["a", "b", "e", "f"])
    parser.add_argument("--seed", type=int, default=42, help="Seed for LODO split.")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_root / "physionet_2016"
    meta_csv = data_dir / "metadata.csv"

    if not meta_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found at {meta_csv}")

    df = pd.read_csv(meta_csv)
    print(f"Loaded {len(df)} rows from {meta_csv}")

    df_dedup = None
    if args.mode in ("dedup", "all"):
        df_dedup = make_dedup_splits(df, seeds=args.seeds, data_dir=data_dir)

    if args.mode in ("lodo", "all"):
        if df_dedup is None:
            dedup_csv = data_dir / "metadata_dedup.csv"
            if dedup_csv.exists():
                df_dedup = pd.read_csv(dedup_csv)
            else:
                df_dedup = df[df["subset"] != "validation"].copy().reset_index(drop=True)
        make_lodo_splits(df_dedup, holdouts=args.holdouts, seed=args.seed, data_dir=data_dir)


if __name__ == "__main__":
    main()
