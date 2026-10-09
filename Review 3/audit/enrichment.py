"""PhysioNet state annotation enrichment audit (S1, systole, S2, diastole)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from audit.common import bootstrap_ci
from audit.faithfulness import compute_gradcam_map
from model import HeartSoundModel

STATE_NAMES = ["S1", "systole", "S2", "diastole"]  # States 1, 2, 3, 4


def inspect_and_parse_mat(mat_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inspect and parse a PhysioNet StateAns .mat file.

    Tries scipy.io.loadmat first, falls back to h5py.
    Returns:
        Tuple of (state_array, format_info_dict).
    """
    mat_path = Path(mat_path)
    info: Dict[str, Any] = {"file": mat_path.name}

    # 1. Try scipy.io.loadmat
    try:
        import scipy.io as sio
        data = sio.loadmat(mat_path)
        info["loader"] = "scipy.io.loadmat"
        # Find candidate state array key
        candidate_keys = [k for k in data.keys() if not k.startswith("__")]
        info["keys"] = candidate_keys

        # Common keys in PhysioNet annotations: 'state_ans', 'StateAns', 'assignedstates'
        chosen_key = None
        for k in ["state_ans", "StateAns", "assignedstates"]:
            if k in data:
                chosen_key = k
                break
        if chosen_key is None and candidate_keys:
            chosen_key = candidate_keys[0]

        if chosen_key:
            arr = np.asarray(data[chosen_key]).squeeze()
            info["chosen_key"] = chosen_key
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
            return arr, info
    except Exception as e_scipy:
        info["scipy_error"] = str(e_scipy)

    # 2. Try h5py fallback (for MATLAB v7.3 files)
    try:
        import h5py
        with h5py.File(mat_path, "r") as f:
            info["loader"] = "h5py"
            keys = list(f.keys())
            info["keys"] = keys
            chosen_key = keys[0]
            arr = np.asarray(f[chosen_key]).squeeze()
            info["chosen_key"] = chosen_key
            info["shape"] = list(arr.shape)
            info["dtype"] = str(arr.dtype)
            return arr, info
    except Exception as e_h5py:
        info["h5py_error"] = str(e_h5py)

    raise ValueError(f"Failed to parse {mat_path} using both scipy.io and h5py: {info}")


def determine_sampling_rate(annotation_files: Sequence[Path],
                            raw_dir: Path,
                            candidate_rates: Sequence[int] = (1000, 2000),
                            n_check: int = 20) -> int:
    """Determine the annotation sampling rate from data, comparing against WAV durations."""
    import soundfile as sf

    files_to_check = annotation_files[:n_check]
    if len(files_to_check) < 1:
        raise ValueError("No annotation files available to check sampling rate.")

    for rate in candidate_rates:
        matches_all = True
        for mat_file in files_to_check:
            # Match recording id to wav
            rec_id = mat_file.stem.replace("_StateAns", "")
            # Look for corresponding wav in raw_dir
            wav_matches = list(raw_dir.glob(f"**/{rec_id}.wav"))
            if not wav_matches:
                matches_all = False
                break

            wav_info = sf.info(wav_matches[0])
            wav_dur = wav_info.duration

            states, _ = inspect_and_parse_mat(mat_file)
            last_idx = len(states)
            est_dur = last_idx / float(rate)

            # Tolerance: within 0.15 seconds of WAV duration
            if abs(est_dur - wav_dur) > 0.15:
                matches_all = False
                break

        if matches_all:
            print(f"  [Enrichment Rate] Confirmed annotation sampling rate: {rate} Hz across {len(files_to_check)} recordings.")
            return rate

    raise RuntimeError(
        "Could not determine annotation sampling rate: neither 1000 Hz nor 2000 Hz matches WAV durations. "
        "Aborting enrichment audit."
    )


def compute_enrichment_ratio(cam_224: np.ndarray,
                             state_sequence_8s: np.ndarray) -> Dict[str, float]:
    """Compute enrichment ratio E_s for states S1, systole, S2, diastole.

    Args:
        cam_224: Grad-CAM map of shape (224, 224) over 8.0 s audio.
        state_sequence_8s: Array of states (1..4) sampled across the 8.0 s.

    Returns:
        Dict mapping state name to E_s.
    """
    total_samples = len(state_sequence_8s)
    if total_samples == 0:
        return {s: 1.0 for s in STATE_NAMES}

    # Time marginal of Grad-CAM: sum over frequency rows
    time_marginal = np.sum(cam_224, axis=0)  # (224,)
    total_cam_mass = float(np.sum(time_marginal))
    if total_cam_mass <= 0:
        total_cam_mass = 1e-8

    n_cols = len(time_marginal)  # 224
    # Map state sequence into 224 time columns
    col_edges = np.linspace(0, total_samples, n_cols + 1).astype(int)

    ratios: Dict[str, float] = {}

    for state_code, state_name in enumerate(STATE_NAMES, start=1):
        # Fraction of 8s labelled with state
        state_mask = (state_sequence_8s == state_code)
        t_frac = float(np.mean(state_mask))

        if t_frac <= 1e-4:
            ratios[state_name] = 1.0
            continue

        # CAM mass inside columns where majority state is state_code
        state_cam_mass = 0.0
        for col in range(n_cols):
            sub_seq = state_sequence_8s[col_edges[col]:col_edges[col + 1]]
            if len(sub_seq) > 0 and np.mean(sub_seq == state_code) >= 0.5:
                state_cam_mass += time_marginal[col]

        cam_frac = float(state_cam_mass / total_cam_mass)
        e_s = float(cam_frac / t_frac)
        ratios[state_name] = e_s

    return ratios


def run_enrichment_audit(model: HeartSoundModel,
                         test_dataset: Any,
                         test_df: pd.DataFrame,
                         device: str,
                         annotations_dir: Optional[Path],
                         raw_dir: Optional[Path],
                         results_dir: Path) -> Dict[str, Any]:
    """Execute complete state annotation enrichment audit."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_dir or not Path(annotations_dir).exists():
        print("  [Enrichment] Annotations directory not provided or missing; skipping enrichment audit.")
        return {"enrichment_status": "skipped", "message": "Annotations directory missing"}

    ann_path = Path(annotations_dir)
    mat_files = list(ann_path.glob("**/*_StateAns.mat"))
    if not mat_files:
        print(f"  [Enrichment] No *_StateAns.mat files found under {ann_path}; skipping.")
        return {"enrichment_status": "skipped", "message": "No StateAns files found"}

    # 1. Inspect first file and save format to results/annotation_format.json
    first_mat = mat_files[0]
    _, format_info = inspect_and_parse_mat(first_mat)
    with open(results_dir / "annotation_format.json", "w", encoding="utf-8") as f:
        json.dump(format_info, f, indent=2)
    print(f"  Saved annotation format finding to {results_dir / 'annotation_format.json'}")

    # 2. Determine sampling rate
    if not raw_dir or not Path(raw_dir).exists():
        print("  [Enrichment] raw_dir not provided; cannot verify WAV duration for sampling rate.")
        return {"enrichment_status": "skipped", "message": "raw_dir required to determine rate"}

    rate = determine_sampling_rate(mat_files, Path(raw_dir))

    # Index annotation files by recording ID
    ann_map: Dict[str, Path] = {}
    for mf in mat_files:
        rec_id = mf.stem.replace("_StateAns", "")
        ann_map[rec_id] = mf

    # Evaluate on test set
    model.eval()
    samples_per_state: Dict[str, List[float]] = {s: [] for s in STATE_NAMES}
    by_class: Dict[str, Dict[str, List[float]]] = {
        "Normal": {s: [] for s in STATE_NAMES},
        "Abnormal": {s: [] for s in STATE_NAMES},
    }
    by_correct: Dict[str, Dict[str, List[float]]] = {
        "Correct": {s: [] for s in STATE_NAMES},
        "Incorrect": {s: [] for s in STATE_NAMES},
    }

    n_usable = 0
    total_test = len(test_df)

    for idx, r in enumerate(test_df.itertuples()):
        fn = str(r.filename)
        if fn not in ann_map:
            continue

        try:
            states_all, _ = inspect_and_parse_mat(ann_map[fn])
            # First 8 seconds of annotation
            n_8s = int(round(8.0 * rate))
            states_8s = states_all[:n_8s]
            if len(states_8s) < n_8s // 2:
                continue

            x_norm, y_true = test_dataset[idx]
            x_batch = x_norm.unsqueeze(0).to(device)

            with torch.no_grad():
                pred = int(model(x_batch).argmax(dim=1).item())

            cam_224 = compute_gradcam_map(model, x_batch, target_class=pred)
            ratios = compute_enrichment_ratio(cam_224, states_8s)

            c_name = "Normal" if int(r.label) == 0 else "Abnormal"
            is_corr = "Correct" if pred == int(r.label) else "Incorrect"

            for s in STATE_NAMES:
                val = ratios[s]
                samples_per_state[s].append(val)
                by_class[c_name][s].append(val)
                by_correct[is_corr][s].append(val)

            n_usable += 1
        except Exception as e:
            continue

    coverage = float(n_usable / total_test) if total_test > 0 else 0.0
    print(f"  [Enrichment Coverage] Usable annotated test recordings: {n_usable}/{total_test} ({coverage * 100:.1f}%)")

    # Bootstrap CIs for overall and breakdowns
    overall_ci: Dict[str, Dict[str, float]] = {}
    for s in STATE_NAMES:
        mean_v, low_v, high_v = bootstrap_ci(samples_per_state[s], n_resamples=2000, seed=42)
        overall_ci[s] = {"mean": mean_v, "ci_low": low_v, "ci_high": high_v}

    class_ci: Dict[str, Dict[str, Dict[str, float]]] = {}
    for c_name in ("Normal", "Abnormal"):
        class_ci[c_name] = {}
        for s in STATE_NAMES:
            m, l, h = bootstrap_ci(by_class[c_name][s], n_resamples=2000, seed=42)
            class_ci[c_name][s] = {"mean": m, "ci_low": l, "ci_high": h}

    return {
        "enrichment_status": "evaluated",
        "sampling_rate": rate,
        "n_usable": n_usable,
        "total_test": total_test,
        "coverage": coverage,
        "overall": overall_ci,
        "by_class": class_ci,
    }
