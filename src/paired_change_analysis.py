from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "endometrium_thickness_mm",
    "endometrium_volume_ml",
    "vfi",
    "pi",
    "clinical_total_peristalsis_rate_min",
    "anterior_amplitude",
    "posterior_amplitude",
]


def paired_wide(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    required = {"patient_id", "pair_order"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dup = df.duplicated(["patient_id", "pair_order"], keep=False)
    if dup.any():
        raise ValueError("Each patient must have exactly one row for pair_order 1 and 2.")

    present = [c for c in features if c in df.columns]
    if not present:
        raise ValueError("None of the requested features are present.")

    wide = df.pivot(index="patient_id", columns="pair_order", values=present)
    if 1 not in wide.columns.get_level_values(1) or 2 not in wide.columns.get_level_values(1):
        raise ValueError("Both pair_order 1 and 2 are required.")

    rows = pd.DataFrame(index=wide.index)
    for feature in present:
        v1 = pd.to_numeric(wide[(feature, 1)], errors="coerce")
        v2 = pd.to_numeric(wide[(feature, 2)], errors="coerce")
        rows[f"{feature}__exam1"] = v1
        rows[f"{feature}__exam2"] = v2
        rows[f"{feature}__delta"] = v2 - v1

    return rows.reset_index()


def summarize_delta(paired: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records = []
    for feature in features:
        col = f"{feature}__delta"
        if col not in paired.columns:
            continue
        s = pd.to_numeric(paired[col], errors="coerce").dropna()
        if s.empty:
            continue

        records.append(
            {
                "feature": feature,
                "n": int(s.size),
                "median_delta": float(s.median()),
                "q1_delta": float(s.quantile(0.25)),
                "q3_delta": float(s.quantile(0.75)),
                "increase_n": int((s > 0).sum()),
                "decrease_n": int((s < 0).sum()),
                "unchanged_n": int(np.isclose(s, 0.0).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    paired = paired_wide(df, args.features)
    summary = summarize_delta(paired, args.features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_change_case_level.csv", index=False)
    summary.to_csv(args.output_dir / "paired_change_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
