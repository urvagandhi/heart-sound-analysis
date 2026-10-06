"""Review 2 — Deduplicated Dataset Split Generator
================================================
Reads physionet_2016/metadata.csv, drops subset == "validation" (the 301 duplicate entries),
writes metadata_dedup.csv and split_indices_dedup.json (70/15/15 stratified on label + subset,
seed 42, pooling small groups < 6 rows into one key per label).
Prints partition sizes and class counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def make_dedup_split(base_dir: Path | str | None = None) -> None:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)

    data_dir = base_dir / "physionet_2016"
    meta_csv_path = data_dir / "metadata.csv"
    if not meta_csv_path.exists():
        raise FileNotFoundError(f"Cannot find metadata.csv at {meta_csv_path}")

    print(f"Reading {meta_csv_path} ...")
    df = pd.read_csv(meta_csv_path)
    total_orig = len(df)
    print(f"Original entries: {total_orig}")

    # Drop validation subset duplicates
    df_dedup = df[df["subset"] != "validation"].copy().reset_index(drop=True)
    n_dedup = len(df_dedup)
    print(f"Entries after dropping validation subset: {n_dedup} (dropped {total_orig - n_dedup})")

    # Save metadata_dedup.csv
    out_meta_path = data_dir / "metadata_dedup.csv"
    df_dedup.to_csv(out_meta_path, index=False)
    print(f"Saved: {out_meta_path}")

    # Stratify on label plus subset, pooling groups with < 6 rows
    strat_key = df_dedup["subset"].astype(str) + "_" + df_dedup["label"].astype(str)
    counts = strat_key.value_counts()
    small_keys = set(counts[counts < 6].index)
    if small_keys:
        print(f"Pooling small strata (< 6 rows): {small_keys}")
    strat_key_pooled = strat_key.apply(lambda k: "pooled_" + k.split("_")[-1] if k in small_keys else k)

    # 70/15/15 split: 70% train, 30% temp -> 15% val, 15% test
    train_idx, temp_idx = train_test_split(
        df_dedup.index, test_size=0.30, random_state=42, stratify=strat_key_pooled
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=strat_key_pooled.loc[temp_idx]
    )

    split_dict = {
        "train_indices": sorted(train_idx.tolist()),
        "val_indices":   sorted(val_idx.tolist()),
        "test_indices":  sorted(test_idx.tolist()),
    }

    out_split_path = data_dir / "split_indices_dedup.json"
    with open(out_split_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f)
    print(f"Saved: {out_split_path}")

    # Print partition sizes and class counts
    print("\n" + "=" * 60)
    print("Deduplicated Split Summary (70 / 15 / 15, seed 42)")
    print("=" * 60)
    for name, indices in [("Training", train_idx), ("Validation", val_idx), ("Test", test_idx)]:
        sub_df = df_dedup.loc[indices]
        n_total = len(sub_df)
        n_norm = int((sub_df["label"] == 0).sum())
        n_abn = int((sub_df["label"] == 1).sum())
        pct_norm = (n_norm / n_total * 100) if n_total > 0 else 0
        pct_abn = (n_abn / n_total * 100) if n_total > 0 else 0
        print(f"  {name:12s}: Total {n_total:5d} | Normal {n_norm:5d} ({pct_norm:5.2f}%) | Abnormal {n_abn:5d} ({pct_abn:5.2f}%)")

    all_norm = int((df_dedup["label"] == 0).sum())
    all_abn = int((df_dedup["label"] == 1).sum())
    print("-" * 60)
    print(f"  {'All (Dedup)':12s}: Total {n_dedup:5d} | Normal {all_norm:5d} ({all_norm/n_dedup*100:5.2f}%) | Abnormal {all_abn:5d} ({all_abn/n_dedup*100:5.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deduplicated PhysioNet 2016 split.")
    parser.add_argument("--base", type=str, default=None, help="Base repository directory.")
    args = parser.parse_args()
    make_dedup_split(args.base)
