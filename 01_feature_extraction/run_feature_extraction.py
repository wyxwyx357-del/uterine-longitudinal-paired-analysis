#!/usr/bin/env python3
"""Run the frozen dynamic pipeline and publish the formal feature table.

The frozen runtime is invoked unchanged. This wrapper does not recalculate,
impute, tune or filter dynamic features. It only:
1) runs the locked current workflow;
2) copies FINAL_DYNAMIC_FEATURES.csv to the stable module output path;
3) optionally writes the legacy auto_features.csv compatibility projection.

Pregnancy outcomes are never read here.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "frozen_runtime_20261126"
FROZEN_ENTRY = RUNTIME / "run_final_batch.py"
EXPORTER = ROOT / "export_auto_features.py"
FORMAL_OUTPUT = ROOT / "outputs" / "FINAL_DYNAMIC_FEATURES.csv"
LEGACY_OUTPUT = ROOT / "outputs" / "auto_features.csv"


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, tmp)
    os.replace(tmp, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "outputs" / "frozen_run",
        help="Internal frozen-runtime work directory; use a new directory for a new formal run.",
    )
    parser.add_argument(
        "--final-features",
        type=Path,
        default=FORMAL_OUTPUT,
        help="Stable formal output consumed by longitudinal analysis.",
    )
    parser.add_argument(
        "--compat-auto-features",
        type=Path,
        default=LEGACY_OUTPUT,
        help="Optional legacy projected alias; set to an empty string is not supported.",
    )
    parser.add_argument("--no-compat-auto-features", action="store_true")
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

    result = subprocess.run(command, cwd=RUNTIME, env=env, check=False)
    if result.returncode != 0:
        return result.returncode

    frozen_table = args.work_dir / "FINAL_DYNAMIC_FEATURES.csv"
    if not frozen_table.is_file():
        raise FileNotFoundError(
            f"Frozen run completed without expected table: {frozen_table}"
        )

    # Preserve the exact frozen table as the formal cross-module contract.
    atomic_copy(frozen_table, args.final_features)

    # Compatibility only: projected/renamed subset for older scripts.
    if not args.no_compat_auto_features:
        export = subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--input",
                str(args.final_features.resolve()),
                "--output",
                str(args.compat_auto_features.resolve()),
            ],
            cwd=ROOT,
            env=env,
            check=False,
        )
        if export.returncode != 0:
            return export.returncode

    print(f"formal_features={args.final_features.resolve()}")
    if not args.no_compat_auto_features:
        print(f"compat_auto_features={args.compat_auto_features.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
