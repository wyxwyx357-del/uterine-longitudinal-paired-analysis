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


def longitudinal_values(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return all repeated examinations in patient-time order."""
    required = {"patient_id", "exam_id", "exam_order"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[["patient_id", "exam_order"]].duplicated().any():
        raise ValueError("Each patient must have only one row per exam_order.")

    present = [c for c in features if c in df.columns]
    if not present:
        raise ValueError("None of the requested features are present.")

    cols = ["patient_id", "exam_id", "exam_order"] + present
    return df[cols].sort_values(["patient_id", "exam_order"]).reset_index(drop=True)


def consecutive_changes(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Calculate every consecutive within-patient change.

    A patient with n examinations contributes n-1 intervals.
    These intervals remain clustered within patient and must not be treated as
    independent patients in downstream inference.
    """
    ordered = longitudinal_values(df, features)
    present = [c for c in features if c in ordered.columns]

    records: list[dict] = []
    for patient_id, g in ordered.groupby("patient_id", sort=False):
        g = g.sort_values("exam_order").reset_index(drop=True)
        if len(g) < 2:
            continue

        for i in range(1, len(g)):
            prev = g.iloc[i - 1]
            curr = g.iloc[i]
            row = {
                "patient_id": patient_id,
                "from_exam_id": prev["exam_id"],
                "to_exam_id": curr["exam_id"],
                "from_exam_order": prev["exam_order"],
                "to_exam_order": curr["exam_order"],
            }
            for feature in present:
                a = pd.to_numeric(pd.Series([prev[feature]]), errors="coerce").iloc[0]
                b = pd.to_numeric(pd.Series([curr[feature]]), errors="coerce").iloc[0]
                row[f"{feature}__from"] = a
                row[f"{feature}__to"] = b
                row[f"{feature}__delta"] = b - a
            records.append(row)

    return pd.DataFrame.from_records(records)


def summarize_delta(changes: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    records = []
    for feature in features:
        col = f"{feature}__delta"
        if col not in changes.columns:
            continue
        s = pd.to_numeric(changes[col], errors="coerce").dropna()
        if s.empty:
            continue

        records.append(
            {
                "feature": feature,
                "interval_n": int(s.size),
                "patient_n": int(
                    changes.loc[s.index, "patient_id"].nunique()
                ),
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
    trajectories = longitudinal_values(df, args.features)
    changes = consecutive_changes(df, args.features)
    summary = summarize_delta(changes, args.features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(
        args.output_dir / "longitudinal_exam_values.csv", index=False
    )
    changes.to_csv(
        args.output_dir / "consecutive_within_patient_changes.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "consecutive_change_summary.csv", index=False
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
