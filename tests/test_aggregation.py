"""Unit tests for aggregate_results.py on mock experimental JSON artifacts."""

import json
from pathlib import Path
import re
import pytest

from aggregate_results import ResultsAggregator


def test_aggregator_end_to_end_mock(tmp_path: Path) -> None:
    """Verify that ResultsAggregator writes valid LaTeX tables and macros with mock results."""
    repo_root = tmp_path
    results_dir = tmp_path / "results"
    gen_dir = tmp_path / "Paper Writing" / "generated"

    results_dir.mkdir(parents=True)
    gen_dir.mkdir(parents=True)

    # 1. Create mock param_counts.json
    param_counts = {
        "baseline": {"total": 23508034, "phase_a_trainable": 2050},
        "se": {"total": 26038466, "phase_a_trainable": 2532482},
        "mha": {"total": 39341506, "phase_a_trainable": 15835522},
        "se_mha": {"total": 41871938, "phase_a_trainable": 18365954},
        "se_mha_dilated": {"total": 41871938, "phase_a_trainable": 18365954},
    }
    with open(results_dir / "param_counts.json", "w", encoding="utf-8") as f:
        json.dump(param_counts, f)

    # 2. Create mock leakage_check.json
    leakage_check = {
        "legacy_test_twin_rows": 70,
        "legacy_test_unique_rows": 462,
        "accuracy_twin_rows": 0.9714,
        "accuracy_unique_rows": 0.8983,
    }
    with open(results_dir / "leakage_check.json", "w", encoding="utf-8") as f:
        json.dump(leakage_check, f)

    # 3. Create mock legacy and dedup runs
    def make_metrics(acc: float, ba: float, f1: float, auc: float) -> dict:
        return {
            "accuracy": acc,
            "balanced_accuracy": ba,
            "f1_macro": f1,
            "auc_roc": auc,
            "sensitivity": ba + 0.02,
            "specificity": ba - 0.02,
        }

    leg_dir = results_dir / "legacy_se_mha"
    leg_dir.mkdir()
    with open(leg_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(make_metrics(0.908, 0.887, 0.892, 0.941), f)

    variants = ["baseline", "se", "mha", "se_mha", "se_mha_dilated"]
    for v in variants:
        for s in (42, 43, 44):
            tag_dir = results_dir / f"dedup_{v}_seed{s}"
            tag_dir.mkdir()
            with open(tag_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(make_metrics(0.880, 0.860, 0.870, 0.920), f)

    # 4. Create mock LODO runs
    for v in ("baseline", "se_mha"):
        for h in ("a", "b", "e", "f"):
            tag_dir = results_dir / f"lodo_{v}_hold{h}_seed42"
            tag_dir.mkdir()
            with open(tag_dir / "test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(make_metrics(0.820, 0.790, 0.800, 0.850), f)

    # 5. Create mock audit directory
    audit_dir = results_dir / "audit_dedup_se_mha_seed42"
    audit_dir.mkdir()
    with open(audit_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({
            "ece": 0.052,
            "operating_point": {
                "selected_threshold": 0.35,
                "test_sensitivity": 0.912,
                "test_specificity": 0.834,
                "test_balanced_accuracy": 0.873,
            },
        }, f)
    with open(audit_dir / "occlusion.json", "w", encoding="utf-8") as f:
        json.dump({
            "reliances": {"B1": True, "B2": False, "B3": False},
        }, f)
    with open(audit_dir / "faithfulness.json", "w", encoding="utf-8") as f:
        json.dump({
            "methods": {
                "Grad-CAM": {"wilcoxon_p_value": 0.001},
                "SHAP": {"wilcoxon_p_value": 0.002},
            },
        }, f)
    with open(audit_dir / "sanity.json", "w", encoding="utf-8") as f:
        json.dump({
            "final_stage_mean_spearman": 0.12,
        }, f)
    with open(audit_dir / "agreement.json", "w", encoding="utf-8") as f:
        json.dump({
            "pairwise": {
                "gradcam_vs_attention": {"level": "low", "mean_spearman": 0.22},
            },
        }, f)
    with open(audit_dir / "enrichment.json", "w", encoding="utf-8") as f:
        json.dump({
            "enrichment_status": "evaluated",
            "coverage": 0.98,
        }, f)

    # Execute aggregator
    aggregator = ResultsAggregator(repo_root=repo_root, results_dir=results_dir, gen_dir=gen_dir)
    aggregator.generate_tables_and_macros()

    # Verify all expected files are generated
    expected_files = [
        "macros.tex", "findings.tex", "tab_leakage.tex", "tab_ablation.tex",
        "tab_lodo.tex", "tab_calib.tex", "tab_occlusion.tex", "tab_faith.tex",
        "tab_enrich.tex", "tab_subset_dedup.tex", "tab_split_dedup.tex",
    ]
    for fn in expected_files:
        p = gen_dir / fn
        assert p.exists(), f"Expected generated file '{fn}' was not created"
        assert p.stat().st_size > 0, f"Generated file '{fn}' is empty"

    # Extract all defined macros from macros.tex and findings.tex
    macros_content = (gen_dir / "macros.tex").read_text(encoding="utf-8")
    findings_content = (gen_dir / "findings.tex").read_text(encoding="utf-8")
    defined_macros = set(re.findall(r"\\(?:providecommand|newcommand|def)\{\\([A-Za-z]+)\}", macros_content + "\n" + findings_content))

    # Verify every \mac... used in generated tables is defined
    macro_usage_pattern = re.compile(r"\\(mac[A-Za-z]+)")
    for fn in expected_files:
        if fn in ("macros.tex", "findings.tex"):
            continue
        content = (gen_dir / fn).read_text(encoding="utf-8")
        used_macros = set(macro_usage_pattern.findall(content))
        undefined = used_macros - defined_macros
        assert not undefined, f"Undefined macros in '{fn}': {undefined}"
