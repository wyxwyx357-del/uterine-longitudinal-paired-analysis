#!/usr/bin/env python3
"""Convert the frozen batch table to the longitudinal-analysis feature contract.

This script only renames/selects fields. It does not recalculate, impute, tune,
or filter any dynamic feature, and it never reads a pregnancy outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "outputs" / "FINAL_DYNAMIC_FEATURES.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "auto_features.csv"

RENAME = {
    "case_id": "exam_id",
    "R2_frequency_cpm": "r2_frequency_cpm",
    "anterior_ER_mean_paired": "anterior_er",
    "posterior_ER_mean_paired": "posterior_er",
    "record_MSE_time_equal": "mse",
    "raw_valid_fraction": "tracking_valid_fraction",
}

PRIMARY_COLUMNS = [
    "exam_id",
    "anterior_amplitude",
    "posterior_amplitude",
    "r2_frequency_cpm",
    "anterior_er",
    "posterior_er",
    "mse",
    "m1_prop_rate_min_per_min",
    "tracking_valid_fraction",
    "video_qc",
    "pose_source",
]

PROVENANCE_COLUMNS = [
    "pipeline_version",
    "a0_status",
    "anterior_amplitude_status",
    "posterior_amplitude_status",
    "mse_status",
    "mse_decomposition_status",
    "m1_status",
    "m1_failure_reason",
    "m1_algorithm_version",
    "R2_status",
    "R2_failure_reason",
    "R2_retained_posterior_region_count",
    "record_MSE_z",
    "record_mean_bias_sq",
    "record_sd_scale_sq",
    "record_time_shape_term",
    "effective_observation_seconds",
    "m1_prop_count_min",
    "m1_prop_count_max",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    source = pd.read_csv(args.input)
    if "case_id" not in source.columns:
        raise ValueError("Frozen table must contain case_id")
    if source["case_id"].isna().any():
        raise ValueError("case_id contains missing values")
    if source["case_id"].astype(str).str.strip().eq("").any():
        raise ValueError("case_id contains blank values")
    if source["case_id"].duplicated().any():
        raise ValueError("case_id must be unique in FINAL_DYNAMIC_FEATURES.csv")

    output = source.rename(columns=RENAME).copy()

    # Keep a stable schema. Empty technical metadata stays missing; it is never
    # synthesized from unrelated status fields.
    for column in PRIMARY_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    keep = PRIMARY_COLUMNS + [
        column for column in PROVENANCE_COLUMNS
        if column in output.columns and column not in PRIMARY_COLUMNS
    ]
    output = output[keep].copy()

    # Numeric coercion preserves failed/missing measurements as NaN.
    numeric = [
        "anterior_amplitude",
        "posterior_amplitude",
        "r2_frequency_cpm",
        "anterior_er",
        "posterior_er",
        "mse",
        "m1_prop_rate_min_per_min",
        "tracking_valid_fraction",
        "R2_retained_posterior_region_count",
        "record_MSE_z",
        "record_mean_bias_sq",
        "record_sd_scale_sq",
        "record_time_shape_term",
        "effective_observation_seconds",
        "m1_prop_count_min",
        "m1_prop_count_max",
    ]
    for column in numeric:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    atomic_csv(output, args.output)
    manifest = {
        "clinical_outcome_read": False,
        "operation": "field projection only; no feature recalculation or imputation",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "rows": int(len(output)),
        "exam_id_unique": bool(output["exam_id"].is_unique),
        "columns": list(output.columns),
        "renamed_fields": RENAME,
    }
    manifest_path = args.output.with_name(args.output.stem + "_export.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"rows={len(output)}")
    print(f"output={args.output.resolve()}")
    print(f"manifest={manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
