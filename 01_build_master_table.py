from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for name in aliases:
        if name in df.columns:
            return name
    return None


def _rename_known_columns(
    df: pd.DataFrame,
    alias_map: dict[str, list[str]],
    required: set[str] | None = None,
) -> pd.DataFrame:
    required = required or set()
    rename = {}
    missing_required = []

    for canonical, aliases in alias_map.items():
        found = _find_column(df, aliases)
        if found is not None:
            rename[found] = canonical
        elif canonical in required:
            missing_required.append((canonical, aliases))

    if missing_required:
        details = "; ".join(
            f"{canonical}: expected one of {aliases}"
            for canonical, aliases in missing_required
        )
        raise ValueError(f"Missing required source columns: {details}")

    return df.rename(columns=rename)


def _to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _parse_yes_no(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    yes = {"是", "yes", "y", "1", "true", "阳性", "√"}
    no = {"否", "no", "n", "0", "false", "阴性", "×"}
    if text in yes:
        return 1.0
    if text in no:
        return 0.0
    return np.nan


def _build_outcome(df: pd.DataFrame) -> None:
    clinical = df.get("clinical_pregnancy_raw", pd.Series(np.nan, index=df.index))
    biochemical = df.get("biochemical_pregnancy_raw", pd.Series(np.nan, index=df.index))

    clinical_num = clinical.map(_parse_yes_no)
    biochemical_num = biochemical.map(_parse_yes_no)

    outcome = clinical_num.copy()
    source = pd.Series("unknown", index=df.index, dtype="object")
    source[clinical_num.notna()] = "clinical_pregnancy"

    fill_negative = outcome.isna() & (biochemical_num == 0)
    outcome.loc[fill_negative] = 0.0
    source.loc[fill_negative] = "biochemical_negative_fill"

    df["clinical_pregnancy"] = outcome
    df["outcome_source"] = source


def _pseudonymize_patients(df: pd.DataFrame) -> None:
    if df["patient_group_source"].isna().any():
        raise ValueError("patient_group_source contains missing values.")

    first_time = (
        df.groupby("patient_group_source", dropna=False)["exam_datetime"]
        .min()
        .sort_values(kind="stable")
    )
    mapping = {
        source: f"P{i:04d}"
        for i, source in enumerate(first_time.index, start=1)
    }
    df["patient_id"] = df["patient_group_source"].map(mapping)


def _add_time_fields(df: pd.DataFrame) -> None:
    df["exam_datetime"] = pd.to_datetime(df["exam_datetime"], errors="coerce")
    df["transfer_date"] = pd.to_datetime(df.get("transfer_date"), errors="coerce")

    if df["exam_datetime"].isna().any():
        bad = df.loc[df["exam_datetime"].isna(), "exam_id"].tolist()
        raise ValueError(f"Invalid exam_datetime for exam_id: {bad}")

    df.sort_values(["patient_id", "exam_datetime", "exam_id"], inplace=True)
    df["exam_order"] = df.groupby("patient_id").cumcount() + 1

    first_exam = df.groupby("patient_id")["exam_datetime"].transform("min")
    df["days_from_first_exam"] = (
        df["exam_datetime"].dt.normalize() - first_exam.dt.normalize()
    ).dt.days

    df["days_to_transfer"] = (
        df["transfer_date"].dt.normalize() - df["exam_datetime"].dt.normalize()
    ).dt.days


def _add_motility_fields(df: pd.DataFrame) -> None:
    fwd = pd.to_numeric(df.get("clinical_forward_rate_min"), errors="coerce")
    bwd = pd.to_numeric(df.get("clinical_backward_rate_min"), errors="coerce")

    both_known = fwd.notna() & bwd.notna()
    total = pd.Series(np.nan, index=df.index, dtype="float64")
    total.loc[both_known] = fwd.loc[both_known] + bwd.loc[both_known]
    df["clinical_total_peristalsis_rate_min"] = total

    ratio = pd.Series(np.nan, index=df.index, dtype="float64")
    active = both_known & (total > 0)
    ratio.loc[active] = fwd.loc[active] / total.loc[active]
    df["clinical_forward_ratio"] = ratio

    state = pd.Series(np.nan, index=df.index, dtype="object")
    state.loc[both_known & (total == 0)] = "static"
    state.loc[active] = "active"
    df["clinical_motility_state"] = state


def _add_cohort_flags(df: pd.DataFrame) -> None:
    df["video_duration_s"] = pd.to_numeric(df["video_duration_s"], errors="coerce")
    df["video_matched"] = df["video_duration_s"].notna() & (df["video_duration_s"] > 0)
    df["long_video_eligible"] = df["video_duration_s"] >= cfg.DURATION_THRESHOLD

    per_patient = df.groupby("patient_id").agg(
        matched_exam_n=("video_matched", "sum"),
        long_exam_n=("long_video_eligible", "sum"),
        total_exam_n=("exam_id", "nunique"),
    )
    per_patient["in_candidate_set"] = (
        (per_patient["matched_exam_n"] >= 2)
        & (per_patient["long_exam_n"] >= 1)
    )
    per_patient["in_strict_set"] = per_patient["long_exam_n"] >= 2

    df.drop(
        columns=[
            "matched_exam_n",
            "long_exam_n",
            "total_exam_n",
            "in_candidate_set",
            "in_strict_set",
        ],
        errors="ignore",
        inplace=True,
    )

    df.merge(
        per_patient,
        left_on="patient_id",
        right_index=True,
        how="left",
        validate="many_to_one",
        copy=False,
    )

    # DataFrame.merge does not mutate in place; assign via aligned mapping.
    for col in [
        "matched_exam_n",
        "long_exam_n",
        "total_exam_n",
        "in_candidate_set",
        "in_strict_set",
    ]:
        df[col] = df["patient_id"].map(per_patient[col])


def _load_auto_features(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[INFO] Optional auto feature file not found: {path}")
        return None

    auto = pd.read_csv(path)
    auto = _rename_known_columns(
        auto,
        cfg.AUTO_COLUMN_ALIASES,
        required={"exam_id"},
    )

    keep = ["exam_id"] + [
        c for c in cfg.AUTO_COLUMN_ALIASES if c != "exam_id" and c in auto.columns
    ]
    auto = auto[keep].copy()

    if auto["exam_id"].duplicated().any():
        bad = auto.loc[auto["exam_id"].duplicated(keep=False), "exam_id"].tolist()
        raise ValueError(
            "AUTO_FEATURES_CSV must contain at most one row per exam_id. "
            f"Duplicates: {bad[:20]}"
        )

    return auto


def _build_completeness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in {"patient_group_source", "clinical_pregnancy_raw", "biochemical_pregnancy_raw"}:
            continue
        n = len(df)
        available = int(df[col].notna().sum())
        rows.append(
            {
                "column": col,
                "total_n": n,
                "available_n": available,
                "missing_n": n - available,
                "available_fraction": available / n if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_output_dirs()

    if not cfg.CLINICAL_EXCEL.exists():
        raise FileNotFoundError(
            f"Clinical workbook not found: {cfg.CLINICAL_EXCEL}\n"
            "Place the workbook under data_raw/ or edit CLINICAL_EXCEL in config.py."
        )

    clinical = pd.read_excel(
        cfg.CLINICAL_EXCEL,
        sheet_name=cfg.CLINICAL_SHEET,
        engine="openpyxl",
    )
    clinical = _rename_known_columns(
        clinical,
        cfg.CLINICAL_COLUMN_ALIASES,
        required={
            "exam_id",
            "patient_group_source",
            "exam_datetime",
            "video_duration_s",
        },
    )

    if clinical["exam_id"].duplicated().any():
        bad = clinical.loc[
            clinical["exam_id"].duplicated(keep=False),
            "exam_id",
        ].tolist()
        raise ValueError(f"Clinical sheet has duplicate exam_id values: {bad[:20]}")

    numeric_cols = [
        "source_exam_order",
        "endometrium_thickness_mm",
        "clinical_forward_rate_min",
        "clinical_backward_rate_min",
        "endometrium_volume_ml",
        "vi",
        "fi",
        "vfi",
        "sd",
        "pi",
        "ri",
        "video_duration_s",
        "embryo_num",
        "age",
        "bmi",
    ]
    _to_numeric(clinical, numeric_cols)

    _pseudonymize_patients(clinical)
    _add_time_fields(clinical)
    _build_outcome(clinical)
    _add_motility_fields(clinical)
    _add_cohort_flags(clinical)

    auto = _load_auto_features(cfg.AUTO_FEATURES_CSV)
    if auto is not None:
        overlap = [
            c for c in auto.columns
            if c != "exam_id" and c in clinical.columns
        ]
        if overlap:
            clinical = clinical.drop(columns=overlap)
        clinical = clinical.merge(
            auto,
            on="exam_id",
            how="left",
            validate="one_to_one",
        )

    preferred_cols = [
        "patient_id",
        "exam_id",
        "exam_order",
        "source_exam_order",
        "exam_datetime",
        "days_from_first_exam",
        "transfer_date",
        "days_to_transfer",
        "cycle_id",
        "clinical_pregnancy",
        "outcome_source",
        "endometrium_thickness_mm",
        "endometrium_volume_ml",
        "endometrium_type",
        "vi",
        "fi",
        "vfi",
        "sd",
        "pi",
        "ri",
        "clinical_forward_rate_min",
        "clinical_backward_rate_min",
        "clinical_total_peristalsis_rate_min",
        "clinical_forward_ratio",
        "clinical_motility_state",
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
        "video_duration_s",
        "video_matched",
        "long_video_eligible",
        "matched_exam_n",
        "long_exam_n",
        "total_exam_n",
        "in_candidate_set",
        "in_strict_set",
        "treatment_mode",
        "protocol",
        "cycle_type",
        "embryo_info",
        "embryo_num",
        "age",
        "bmi",
        "cycle_link_confidence",
        "cycle_audit_status",
    ]
    master_cols = [c for c in preferred_cols if c in clinical.columns]
    master = clinical[master_cols].copy()

    # Do not export direct source identifiers such as patient_group_source or name.
    master.to_csv(cfg.TABLES / "master_long.csv", index=False)

    completeness = _build_completeness(master)
    completeness.to_csv(cfg.TABLES / "data_completeness.csv", index=False)

    cohort_summary = (
        master.groupby("patient_id", as_index=False)
        .agg(
            total_exam_n=("exam_id", "nunique"),
            matched_exam_n=("video_matched", "sum"),
            long_exam_n=("long_video_eligible", "sum"),
            in_candidate_set=("in_candidate_set", "max"),
            in_strict_set=("in_strict_set", "max"),
        )
        .sort_values(["in_strict_set", "in_candidate_set", "patient_id"], ascending=[False, False, True])
    )
    cohort_summary.to_csv(cfg.TABLES / "cohort_summary.csv", index=False)

    candidate_not_strict = cohort_summary[
        cohort_summary["in_candidate_set"] & ~cohort_summary["in_strict_set"]
    ].copy()
    candidate_not_strict.to_csv(
        cfg.TABLES / "candidate_not_strict.csv",
        index=False,
    )

    strict_patients = int(cohort_summary["in_strict_set"].sum())
    candidate_patients = int(cohort_summary["in_candidate_set"].sum())
    strict_exams = int(
        (master["in_strict_set"] & master["long_video_eligible"]).sum()
    )
    candidate_exams = int(
        (master["in_candidate_set"] & master["video_matched"]).sum()
    )

    print("Master table built successfully.")
    print(f"candidate patients={candidate_patients}, candidate exams={candidate_exams}")
    print(f"strict patients={strict_patients}, strict >=60s exams={strict_exams}")
    print(f"master: {cfg.TABLES / 'master_long.csv'}")
    print(f"cohort summary: {cfg.TABLES / 'cohort_summary.csv'}")


if __name__ == "__main__":
    main()
