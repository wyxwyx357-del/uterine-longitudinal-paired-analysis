#!/usr/bin/env python3
"""Run the frozen dynamic pipeline, then export the longitudinal feature table.

The frozen runtime is invoked unchanged. This wrapper supplies the relocated
runtime on PYTHONPATH and performs only post-run field projection.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "frozen_runtime_20261126"
FROZEN_ENTRY = RUNTIME / "run_final_batch.py"
EXPORTER = ROOT / "export_auto_features.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "frozen_run",
    )
    parser.add_argument(
        "--auto-features",
        type=Path,
        default=ROOT / "outputs" / "auto_features.csv",
    )
    parser.add_argument("--r2-workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-r2", action="store_true")
    parser.add_argument("--skip-a0", action="store_true")
    args = parser.parse_args()

    if not FROZEN_ENTRY.is_file():
        raise FileNotFoundError(FROZEN_ENTRY)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    env = os.environ.copy()
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(RUNTIME)
        if not previous
        else str(RUNTIME) + os.pathsep + previous
    )

    command = [
        sys.executable,
        str(FROZEN_ENTRY),
        "--manifest",
        str(args.manifest.resolve()),
        "--output",
        str(args.work_dir.resolve()),
        "--r2-workers",
        str(args.r2_workers),
    ]
    if args.resume:
        command.append("--resume")
    if args.skip_r2:
        command.append("--skip-r2")
    if args.skip_a0:
        command.append("--skip-a0")

    result = subprocess.run(
        command,
        cwd=RUNTIME,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    final_table = args.work_dir / "FINAL_DYNAMIC_FEATURES.csv"
    if not final_table.is_file():
        raise FileNotFoundError(
            f"Frozen run completed without expected table: {final_table}"
        )

    export = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--input",
            str(final_table.resolve()),
            "--output",
            str(args.auto_features.resolve()),
        ],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return export.returncode


if __name__ == "__main__":
    raise SystemExit(main())
