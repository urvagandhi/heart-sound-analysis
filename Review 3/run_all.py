"""Sequential experiment runner across legacy, deduplicated, and LODO benchmarks.

Executes the planned 24 experimental runs:
  1. legacy_se_mha (1 run): Evaluation-only on the existing legacy checkpoint.
  2. dedup x 5 variants x 3 seeds (15 runs):
     Variants: baseline, se, mha, se_mha, se_mha_dilated
     Seeds: 42, 43, 44
  3. lodo x 2 variants x 4 holdouts x seed 42 (8 runs):
     Variants: baseline, se_mha
     Holdouts: a, b, e, f

Features:
  - Resume-safe: skips any run whose test_metrics.json already exists.
  - Logging: writes execution output to out-root/run_all.log.
  - Supports --dry-run to display execution plan and total run count.
  - Supports --only TAGPREFIX to filter execution by tag prefix.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RunSpec:
    """Specification of an individual benchmark run."""
    tag: str
    variant: str
    split: str
    seed: int
    holdout: Optional[str] = None
    eval_only: bool = False
    checkpoint: Optional[str] = None

    def build_cmd(self, main_py: Path, out_root: Path, base_dir: Optional[Path] = None,
                  extra_args: Optional[List[str]] = None) -> List[str]:
        cmd = [
            sys.executable,
            str(main_py),
            "--variant", self.variant,
            "--split", self.split,
            "--seed", str(self.seed),
            "--tag", self.tag,
            "--out-root", str(out_root),
        ]
        if self.holdout:
            cmd.extend(["--holdout", self.holdout])
        if self.eval_only:
            cmd.append("--eval-only")
        if self.checkpoint:
            cmd.extend(["--checkpoint", self.checkpoint])
        if base_dir:
            cmd.extend(["--base", str(base_dir)])
        if extra_args:
            cmd.extend(extra_args)
        return cmd


def build_manifest(legacy_ckpt_path: Path) -> List[RunSpec]:
    """Build the complete 24-run benchmark manifest."""
    manifest: List[RunSpec] = []

    # 1. Legacy evaluation run (1 run)
    manifest.append(RunSpec(
        tag="legacy_se_mha",
        variant="se_mha",
        split="legacy",
        seed=42,
        eval_only=True,
        checkpoint=str(legacy_ckpt_path),
    ))

    # 2. Deduplicated component ablation (15 runs)
    dedup_variants = ["baseline", "se", "mha", "se_mha", "se_mha_dilated"]
    dedup_seeds = [42, 43, 44]
    for var in dedup_variants:
        for s in dedup_seeds:
            manifest.append(RunSpec(
                tag=f"dedup_{var}_seed{s}",
                variant=var,
                split="dedup",
                seed=s,
            ))

    # 3. Leave-one-database-out (LODO) cross-site evaluation (8 runs)
    lodo_variants = ["baseline", "se_mha"]
    lodo_holdouts = ["a", "b", "e", "f"]
    for var in lodo_variants:
        for h in lodo_holdouts:
            manifest.append(RunSpec(
                tag=f"lodo_{var}_hold{h}_seed42",
                variant=var,
                split="lodo",
                seed=42,
                holdout=h,
            ))

    return manifest


def run_all(manifest: List[RunSpec],
            out_root: Path,
            base_dir: Path,
            dry_run: bool = False,
            only_prefix: Optional[str] = None,
            extra_args: Optional[List[str]] = None) -> None:
    """Execute all benchmark runs sequentially with resume checking and logging."""
    main_py = Path(__file__).resolve().parent / "main.py"
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = out_root / "run_all.log"

    # Filter manifest if requested
    filtered_manifest = [
        spec for spec in manifest
        if (only_prefix is None or spec.tag.startswith(only_prefix))
    ]

    print("=" * 70)
    print("  Heart Sound Benchmark Runner (run_all.py)")
    print("=" * 70)
    print(f"  Total planned runs : {len(filtered_manifest)}")
    print(f"  Output root        : {out_root}")
    print(f"  Dry run mode       : {dry_run}")
    if only_prefix:
        print(f"  Filter prefix      : {only_prefix}")
    print("=" * 70)

    for i, spec in enumerate(filtered_manifest, start=1):
        target_dir = out_root / spec.tag
        metrics_file = target_dir / "test_metrics.json"
        status = "[DONE]" if metrics_file.exists() else "[PENDING]"
        print(f"  {i:2d}/{len(filtered_manifest):2d} {status} {spec.tag:32s} (variant={spec.variant}, split={spec.split})")

    if dry_run:
        print("\n[Dry Run] Plan complete. Exiting without execution.")
        return

    # Setup logger
    logger = logging.getLogger("run_all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting run_all with {len(filtered_manifest)} runs. out_root={out_root}")

    for idx, spec in enumerate(filtered_manifest, start=1):
        target_dir = out_root / spec.tag
        metrics_file = target_dir / "test_metrics.json"

        if metrics_file.exists():
            msg = f"[{idx}/{len(filtered_manifest)}] Skipping {spec.tag}: test_metrics.json already exists."
            print(f"\n{msg}")
            logger.info(msg)
            continue

        cmd = spec.build_cmd(main_py, out_root, base_dir, extra_args)
        cmd_str = " ".join(cmd)
        msg_start = f"[{idx}/{len(filtered_manifest)}] Launching: {spec.tag}\nCommand: {cmd_str}"
        print(f"\n{msg_start}")
        logger.info(msg_start)

        # Run process and stream output to log
        try:
            with open(log_file, "a", encoding="utf-8") as f_log:
                proc = subprocess.run(
                    cmd,
                    stdout=f_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True,
                )
            msg_finish = f"[{idx}/{len(filtered_manifest)}] Successfully completed: {spec.tag}"
            print(msg_finish)
            logger.info(msg_finish)
        except subprocess.CalledProcessError as e:
            msg_err = f"[{idx}/{len(filtered_manifest)}] Error running {spec.tag} (exit code {e.returncode}). See {log_file}."
            print(msg_err, file=sys.stderr)
            logger.error(msg_err)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute full benchmark manifest across variants and splits.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and total run count without executing.")
    parser.add_argument("--only", type=str, default=None, help="Filter runs by tag prefix (e.g., 'dedup_se_mha').")
    parser.add_argument("--out-root", type=str, default=None, help="Root directory for results (default: <base>/results).")
    parser.add_argument("--base", type=str, default=None, help="Base repo directory.")
    parser.add_argument("--legacy-ckpt", type=str, default=None, help="Path to legacy checkpoint.")
    args, unknown = parser.parse_known_args()

    repo_root = Path(args.base) if args.base else Path(__file__).resolve().parent.parent
    out_root = Path(args.out_root) if args.out_root else repo_root / "results"
    legacy_ckpt = Path(args.legacy_ckpt) if args.legacy_ckpt else repo_root / "Review 3" / "output" / "checkpoints" / "best_model.pth"

    manifest = build_manifest(legacy_ckpt)
    run_all(
        manifest=manifest,
        out_root=out_root,
        base_dir=repo_root,
        dry_run=args.dry_run,
        only_prefix=args.only,
        extra_args=unknown if unknown else None,
    )


if __name__ == "__main__":
    main()
