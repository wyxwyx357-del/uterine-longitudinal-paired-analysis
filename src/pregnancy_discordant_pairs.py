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


def discordant_patient_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compare all successful vs failed examinations within each patient.

    If a patient has multiple examinations in either outcome group, all are kept.
    The patient-level contrast uses the median successful examination value minus
    the median failed examination value. No single pair is selected.
    """
    required = {"patient_id", "exam_id", "exam_order", "clinical_pregnancy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.copy()
        g["clinical_pregnancy"] = pd.to_numeric(
            g["clinical_pregnancy"], errors="coerce"
        )
        g = g[g["clinical_pregnancy"].isin([0, 1])]

        if not ({0, 1} <= set(g["clinical_pregnancy"])):
            continue

        failure = g[g["clinical_pregnancy"] == 0]
        success = g[g["clinical_pregnancy"] == 1]

        row = {
            "patient_id": patient_id,
            "failure_exam_n": int(len(failure)),
            "success_exam_n": int(len(success)),
        }

        for feature in features:
            if feature not in df.columns:
                continue
            fail_values = pd.to_numeric(failure[feature], errors="coerce").dropna()
            succ_values = pd.to_numeric(success[feature], errors="coerce").dropna()

            fail_median = fail_values.median() if not fail_values.empty else np.nan
            succ_median = succ_values.median() if not succ_values.empty else np.nan

            row[f"{feature}__failure_median"] = fail_median
            row[f"{feature}__success_median"] = succ_median
            row[f"{feature}__success_minus_failure"] = succ_median - fail_median

        rows.append(row)

    return pd.DataFrame(rows)


def discordant_exam_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Keep every known-outcome examination from discordant patients."""
    x = df.copy()
    x["clinical_pregnancy"] = pd.to_numeric(
        x["clinical_pregnancy"], errors="coerce"
    )
    x = x[x["clinical_pregnancy"].isin([0, 1])]

    ids = []
    for patient_id, g in x.groupby("patient_id"):
        if {0, 1} <= set(g["clinical_pregnancy"]):
            ids.append(patient_id)

    return x[x["patient_id"].isin(ids)].sort_values(
        ["patient_id", "exam_order"]
    ).reset_index(drop=True)


def summarize(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = []
    for feature in features:
        col = f"{feature}__success_minus_failure"
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        out.append(
            {
                "feature": feature,
                "patient_n": int(s.size),
                "median_success_minus_failure": float(s.median()),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
                "success_higher_n": int((s > 0).sum()),
                "success_lower_n": int((s < 0).sum()),
                "same_n": int(np.isclose(s, 0.0).sum()),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = p.parse_args()

    source = pd.read_csv(args.input_csv)
    patient_summary = discordant_patient_summary(source, args.features)
    trajectory = discordant_exam_trajectory(source)
    summary = summarize(patient_summary, args.features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(
        args.output_dir / "discordant_pregnancy_all_exams.csv", index=False
    )
    patient_summary.to_csv(
        args.output_dir / "discordant_pregnancy_patient_summary.csv", index=False
    )
    summary.to_csv(
        args.output_dir / "discordant_pregnancy_feature_summary.csv", index=False
    )

    print(f"discordant_patients={patient_summary['patient_id'].nunique() if not patient_summary.empty else 0}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
