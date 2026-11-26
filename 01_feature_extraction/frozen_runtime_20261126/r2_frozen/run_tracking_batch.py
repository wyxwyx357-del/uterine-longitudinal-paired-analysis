#!/usr/bin/env python3
"""Checkpointed parallel runner for frozen tissue-frequency V1."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NATIVE_TABLES = (
    "case_status.csv",
    "optimized_region_frequency.csv",
    "optimized_patient_frequency.csv",
    "signal_origin_evidence.csv",
)
REQUIRED_MANIFEST_COLUMNS = {
    "case_id", "video", "tracking", "nodes", "mm_per_processed_pixel",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def complete_output(path: Path) -> bool:
    if not all((path / table).is_file() for table in NATIVE_TABLES):
        return False
    try:
        _, rows = read_csv(path / "case_status.csv")
    except (OSError, csv.Error):
        return False
    return bool(
        len(rows) == 1
        and rows[0].get("status") == "complete"
    )


def atomic_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    manifest_fields, manifest_rows = read_csv(args.manifest)
    missing_columns = REQUIRED_MANIFEST_COLUMNS - set(manifest_fields)
    if missing_columns:
        raise ValueError(f"Manifest missing columns: {sorted(missing_columns)}")
    if not manifest_rows:
        raise ValueError("Manifest contains no patients")
    case_ids = [row["case_id"].strip() for row in manifest_rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every manifest row must have a non-empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Manifest contains duplicate case_id values")
    for row, case_id in zip(manifest_rows, case_ids):
        row["case_id"] = case_id
    args.run_dir.mkdir(parents=True, exist_ok=True)
    cases_root = args.run_dir / "cases"
    cases_root.mkdir(exist_ok=True)
    status_path = args.run_dir / "resumable_status.json"
    merged_dir = args.run_dir / "merged_results"
    if merged_dir.exists():
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            same_cases = set(previous.get("cases", {})) == {
                row["case_id"] for row in manifest_rows
            }
            same_manifest_rows = True
            for index, row in enumerate(manifest_rows, start=1):
                own_manifest = cases_root / f"case_{index:03d}" / "manifest.csv"
                if not own_manifest.is_file():
                    same_manifest_rows = False
                    break
                own_fields, own_rows = read_csv(own_manifest)
                if own_fields != manifest_fields or own_rows != [row]:
                    same_manifest_rows = False
                    break
            merged_complete = all((merged_dir / table).is_file() for table in NATIVE_TABLES)
            if (
                previous.get("state") in {"complete", "partial_complete"}
                and same_cases and same_manifest_rows and merged_complete
            ):
                print(f"already complete or partial-complete: {merged_dir}", flush=True)
                return 0
        raise FileExistsError(f"Existing merged directory is incomplete or mismatched: {merged_dir}")
    lock = threading.Lock()
    case_order = {row["case_id"]: index for index, row in enumerate(manifest_rows)}
    outputs: dict[str, str] = {}
    records: dict[str, dict] = {}

    for index, row in enumerate(manifest_rows, start=1):
        case_root = cases_root / f"case_{index:03d}"
        case_root.mkdir(exist_ok=True)
        own_manifest = case_root / "manifest.csv"
        if not own_manifest.exists():
            write_csv(own_manifest, manifest_fields, [row])
        else:
            own_fields, own_rows = read_csv(own_manifest)
            if own_fields != manifest_fields or own_rows != [row]:
                raise ValueError(
                    f"Manifest row changed for {row['case_id']}; use a new run directory"
                )
        attempts = sorted(path for path in case_root.glob("attempt_*") if path.is_dir())
        completed = next((path for path in reversed(attempts) if complete_output(path)), None)
        cache_file = case_root / "trajectory_cache" / f"{row['case_id']}.npz"
        if completed is not None and not cache_file.is_file():
            completed = None
        if completed is not None:
            outputs[row["case_id"]] = str(completed)
        records[row["case_id"]] = {
            "index": index,
            "state": "complete" if completed else "pending",
            "output": str(completed) if completed else None,
            "return_code": 0 if completed else None,
        }

    status = {
        "started_or_resumed_at": datetime.now().astimezone().isoformat(),
        "state": "running",
        "workers": args.workers,
        "total_cases": len(manifest_rows),
        "completed_cases": len(outputs),
        "remaining_cases": len(manifest_rows) - len(outputs),
        "cases": records,
    }
    atomic_status(status_path, status)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(args.run_dir / "matplotlib_cache")

    def run_case(index: int, row: dict[str, str]) -> tuple[str, int, str]:
        case_id = row["case_id"]
        case_root = cases_root / f"case_{index:03d}"
        own_manifest = case_root / "manifest.csv"
        attempt_number = len([path for path in case_root.glob("attempt_*") if path.is_dir()]) + 1
        output = case_root / f"attempt_{attempt_number:03d}"
        stdout_path = case_root / f"attempt_{attempt_number:03d}.stdout.log"
        stderr_path = case_root / f"attempt_{attempt_number:03d}.stderr.log"
        with lock:
            records[case_id].update(state="running", output=str(output))
            atomic_status(status_path, status)
        cache_dir = case_root / "trajectory_cache"
        command = [
            sys.executable, str(ROOT / "track_and_cache_frequency.py"),
            "--case-manifest", str(own_manifest), "--out", str(output),
            "--cache-dir", str(cache_dir), "--reuse-cache",
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout_stream, \
                stderr_path.open("w", encoding="utf-8") as stderr_stream:
            result = subprocess.run(
                command, cwd=ROOT, env=env,
                stdout=stdout_stream, stderr=stderr_stream, check=False,
            )
        return case_id, result.returncode, str(output)

    pending = [
        (index, row) for index, row in enumerate(manifest_rows, start=1)
        if row["case_id"] not in outputs
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, index, row) for index, row in pending]
        for future in as_completed(futures):
            case_id, return_code, output = future.result()
            valid = complete_output(Path(output))
            with lock:
                records[case_id].update(
                    state="complete" if valid else "failed",
                    output=output,
                    return_code=return_code,
                )
                if valid:
                    outputs[case_id] = output
                status["completed_cases"] = len(outputs)
                status["remaining_cases"] = len(manifest_rows) - len(outputs)
                atomic_status(status_path, status)

    failed_case_ids = [
        case_id for case_id, record in records.items()
        if record.get("state") != "complete"
    ]
    if not outputs:
        status.update(
            state="failed",
            finished_at=datetime.now().astimezone().isoformat(),
            failed_cases=len(failed_case_ids),
            remaining_cases=0,
        )
        atomic_status(status_path, status)
        return 2

    merged_dir.mkdir()
    failed_rows = [
        {
            "case_id": case_id,
            "state": records[case_id].get("state"),
            "return_code": records[case_id].get("return_code"),
            "output": records[case_id].get("output"),
        }
        for case_id in failed_case_ids
    ]
    write_csv(
        merged_dir / "failed_cases.csv",
        ["case_id", "state", "return_code", "output"],
        failed_rows,
    )

    for table in NATIVE_TABLES:
        fields: list[str] = []
        combined: list[dict[str, str]] = []
        for row in manifest_rows:
            case_id = row["case_id"]
            if case_id not in outputs:
                continue
            own_fields, own_rows = read_csv(Path(outputs[case_id]) / table)
            for field in own_fields:
                if field not in fields:
                    fields.append(field)
            combined.extend(own_rows)
        combined.sort(key=lambda item: case_order.get(item.get("case_id", ""), len(case_order)))
        write_csv(merged_dir / table, fields, combined)

    _, region_rows = read_csv(merged_dir / "optimized_region_frequency.csv")
    confidence_counts: dict[str, int] = {}
    for row in region_rows:
        key = row.get("technical_confidence", "")
        confidence_counts[key] = confidence_counts.get(key, 0) + 1
    cohort_summary = {
        "frozen_rule_version": "R2_TISSUE_FREQUENCY_V1_20261105",
        "clinical_outcome_read": False,
        "requested_patients": len(manifest_rows),
        "patients": len(outputs),
        "failed_patients": len(failed_case_ids),
        "regions": len(region_rows),
        "technical_confidence_counts": confidence_counts,
        "retained_regions": sum(
            row.get("retained_for_patient_model", "").lower() == "true"
            for row in region_rows),
        "frequency_aggregation": "none",
    }
    (merged_dir / "正式汇总规则与计数.json").write_text(
        json.dumps(cohort_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Convenient Chinese aliases; the native English tables remain canonical.
    write_csv(
        merged_dir / "区域级组织位移频率.csv",
        list(read_csv(merged_dir / "optimized_region_frequency.csv")[0]),
        region_rows,
    )
    patient_fields, patient_rows = read_csv(
        merged_dir / "optimized_patient_frequency.csv")
    write_csv(merged_dir / "患者级技术汇总.csv", patient_fields, patient_rows)
    status.update(
        state="complete" if not failed_case_ids else "partial_complete",
        finished_at=datetime.now().astimezone().isoformat(),
        merged_results=str(merged_dir),
        summary_return_code=0,
        completed_cases=len(outputs),
        failed_cases=len(failed_case_ids),
        remaining_cases=0,
        frozen_rule_version="R2_TISSUE_FREQUENCY_V1_20261105",
    )
    atomic_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

