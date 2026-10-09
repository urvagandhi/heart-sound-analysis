"""Duplicate verification and test leakage quantification.

Compares validation files against their training twins:
  - If WAV files exist in --raw_dir, compares MD5 of decoded sample arrays.
  - If WAVs are missing, compares (filename, label) and marks 'UNVERIFIED by content'.
Computes leakage metrics on legacy test split:
  - Number of legacy test rows with a twin in training set.
  - Model accuracy on twin rows vs unique rows.
Writes results to results/leakage_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def compute_audio_md5(file_path: Path) -> Optional[str]:
    """Compute MD5 checksum of raw audio sample array."""
    try:
        import soundfile as sf
        data, _ = sf.read(file_path)
        return hashlib.md5(data.tobytes()).hexdigest()
    except Exception:
        try:
            import scipy.io.wavfile as wav
            _, data = wav.read(file_path)
            return hashlib.md5(data.tobytes()).hexdigest()
        except Exception:
            return None


def verify_duplicates(raw_dir: Optional[Path],
                      meta_csv_path: Path,
                      legacy_split_path: Path,
                      legacy_preds_path: Path,
                      out_json_path: Path) -> Dict[str, Any]:
    """Verify duplicates and quantify test leakage."""
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(meta_csv_path)

    val_df = df[df["subset"] == "validation"].copy()
    train_pool_df = df[df["subset"] != "validation"].copy()

    # Content verification
    content_verification_status = "UNVERIFIED by content"
    md5_matches = 0
    wav_files_found = 0

    if raw_dir and Path(raw_dir).exists():
        raw_path = Path(raw_dir)
        val_md5s = {}
        for r in val_df.itertuples():
            p = raw_path / r.wav_path
            if p.exists():
                h = compute_audio_md5(p)
                if h is not None:
                    val_md5s[r.filename] = h

        wav_files_found = len(val_md5s)
        if wav_files_found > 0:
            for r in train_pool_df.itertuples():
                if r.filename in val_md5s:
                    p = raw_path / r.wav_path
                    if p.exists():
                        h = compute_audio_md5(p)
                        if h == val_md5s[r.filename]:
                            md5_matches += 1

            if md5_matches == wav_files_found and wav_files_found == len(val_df):
                content_verification_status = "VERIFIED by MD5"
            else:
                content_verification_status = f"PARTIALLY VERIFIED ({md5_matches}/{wav_files_found} matches)"

    print(f"Content verification status: {content_verification_status}")

    # Legacy test leakage check
    with open(legacy_split_path, "r", encoding="utf-8") as f:
        legacy_splits = json.load(f)

    legacy_train_df = df.iloc[legacy_splits["train_indices"]]
    train_twins = set(legacy_train_df[["filename", "label"]].itertuples(index=False, name=None))

    preds_df = pd.read_csv(legacy_preds_path)
    is_twin = preds_df.apply(lambda r: (r["filename"], int(r["label"])) in train_twins, axis=1)

    n_total_test = len(preds_df)
    n_twin_rows = int(is_twin.sum())
    n_unique_rows = n_total_test - n_twin_rows

    acc_twin = float(preds_df[is_twin]["correct"].mean()) if n_twin_rows > 0 else 0.0
    acc_unique = float(preds_df[~is_twin]["correct"].mean()) if n_unique_rows > 0 else 0.0
    acc_diff = float(acc_twin - acc_unique)

    result = {
        "content_verification": content_verification_status,
        "total_legacy_test_rows": n_total_test,
        "legacy_test_twin_rows": n_twin_rows,
        "legacy_test_unique_rows": n_unique_rows,
        "accuracy_twin_rows": acc_twin,
        "accuracy_unique_rows": acc_unique,
        "accuracy_delta": acc_diff,
        "total_validation_rows": len(val_df),
        "total_deduplicated_rows": len(train_pool_df),
    }

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Leakage check saved to {out_json_path}:")
    print(f"  Test rows: {n_total_test} total ({n_twin_rows} twin, {n_unique_rows} unique)")
    print(f"  Accuracy: Twin={acc_twin:.4f} vs Unique={acc_unique:.4f} (Delta={acc_diff:+.4f})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PhysioNet duplicate recordings and quantify leakage.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Path to raw audio WAV directory.")
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--legacy_split", type=str, default=None)
    parser.add_argument("--legacy_preds", type=str, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "physionet_2016"
    meta_path = Path(args.meta_path) if args.meta_path else data_dir / "metadata.csv"
    legacy_split = Path(args.legacy_split) if args.legacy_split else data_dir / "split_indices.json"
    legacy_preds = Path(args.legacy_preds) if args.legacy_preds else repo_root / "Review 3" / "output" / "test_predictions.csv"
    out_path = Path(args.out_path) if args.out_path else repo_root / "results" / "leakage_check.json"
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    verify_duplicates(
        raw_dir=raw_dir,
        meta_csv_path=meta_path,
        legacy_split_path=legacy_split,
        legacy_preds_path=legacy_preds,
        out_json_path=out_path,
    )


if __name__ == "__main__":
    main()
