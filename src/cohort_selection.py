from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "patient_id",
    "exam_id",
    "exam_order",
    "video_duration_s",
}


def build_primary_cohort(
    df: pd.DataFrame,
    min_video_s: float = 60.0,
    min_exams_per_patient: int = 2,
) -> pd.DataFrame:
    """Keep all qualifying examinations from patients with repeated eligible exams.

    No examination is selected according to outcome or feature values.
    Patients with 3+ qualifying examinations retain all of them.
    """
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.copy()
    x["video_duration_s"] = pd.to_numeric(x["video_duration_s"], errors="coerce")
    x["exam_order"] = pd.to_numeric(x["exam_order"], errors="coerce")

    if x["exam_id"].duplicated().any():
        dup_ids = x.loc[x["exam_id"].duplicated(keep=False), "exam_id"].tolist()
        raise ValueError(f"exam_id must be globally unique. Duplicates: {dup_ids}")

    if x[["patient_id", "exam_order"]].duplicated().any():
        bad = x.loc[
            x[["patient_id", "exam_order"]].duplicated(keep=False),
            ["patient_id", "exam_id", "exam_order"],
        ]
        raise ValueError(
            "Duplicate patient_id + exam_order rows found. "
            "Resolve chronological order locally before analysis:\n"
            + bad.to_string(index=False)
        )

    x = x[x["video_duration_s"] >= min_video_s].copy()
    x = x[x["exam_order"].notna()].copy()

    counts = x.groupby("patient_id")["exam_id"].nunique()
    eligible_ids = counts[counts >= min_exams_per_patient].index
    out = x[x["patient_id"].isin(eligible_ids)].copy()

    return out.sort_values(["patient_id", "exam_order", "exam_id"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_csv", type=Path)
    p.add_argument("--min-video-s", type=float, default=60.0)
    p.add_argument("--min-exams-per-patient", type=int, default=2)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    cohort = build_primary_cohort(
        df,
        min_video_s=args.min_video_s,
        min_exams_per_patient=args.min_exams_per_patient,
    )

    per_patient = cohort.groupby("patient_id")["exam_id"].nunique()
    print(f"patients={cohort['patient_id'].nunique()}")
    print(f"exams={len(cohort)}")
    print("exams_per_patient:")
    print(per_patient.value_counts().sort_index().to_string())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
