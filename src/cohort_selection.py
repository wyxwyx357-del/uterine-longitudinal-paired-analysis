from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED = {
    "patient_id",
    "exam_id",
    "pair_order",
    "video_duration_s",
}


def build_primary_cohort(df: pd.DataFrame, min_video_s: float = 60.0) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.copy()
    x["video_duration_s"] = pd.to_numeric(x["video_duration_s"], errors="coerce")
    x["pair_order"] = pd.to_numeric(x["pair_order"], errors="coerce")

    # Only explicitly paired examinations are eligible.
    x = x[x["pair_order"].isin([1, 2])]
    x = x[x["video_duration_s"] >= min_video_s]

    counts = x.groupby("patient_id")["pair_order"].nunique()
    eligible_ids = counts[counts == 2].index

    out = x[x["patient_id"].isin(eligible_ids)].copy()

    # Do not silently choose among duplicate pair_order rows.
    dup = out.duplicated(["patient_id", "pair_order"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["patient_id", "exam_id", "pair_order"]]
        raise ValueError(
            "Duplicate patient_id + pair_order rows found. "
            "Resolve the intended pair locally before analysis:\n"
            + bad.to_string(index=False)
        )

    return out.sort_values(["patient_id", "pair_order"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_csv", type=Path)
    p.add_argument("output_csv", type=Path)
    p.add_argument("--min-video-s", type=float, default=60.0)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)
    cohort = build_primary_cohort(df, args.min_video_s)

    print(f"patients={cohort['patient_id'].nunique()}")
    print(f"exams={len(cohort)}")
    print(cohort.groupby("pair_order")["video_duration_s"].describe())

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
