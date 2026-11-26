from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config as cfg


def _load_analysis_data() -> pd.DataFrame:
    path = cfg.TABLES / "master_long.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_build_master_table.py first.")

    df = pd.read_csv(path)
    mask = cfg.cohort_mask(df)
    df = df.loc[mask].copy()

    if df.empty:
        raise ValueError(f"No rows available for cohort={cfg.ANALYSIS_COHORT}")

    df["exam_datetime"] = pd.to_datetime(df["exam_datetime"], errors="coerce")
    return df.sort_values(["patient_id", "exam_order", "exam_datetime"])


def _numeric_present(df: pd.DataFrame) -> list[str]:
    wanted = cfg.ALL_CONTINUOUS_COLS + cfg.QC_NUMERIC_COLS
    present = []
    for col in dict.fromkeys(wanted):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            present.append(col)
    return present


def build_adjacent_deltas(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.DataFrame:
    records = []

    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.sort_values(["exam_order", "exam_datetime", "exam_id"]).reset_index(drop=True)
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
                "from_exam_datetime": prev.get("exam_datetime"),
                "to_exam_datetime": curr.get("exam_datetime"),
            }

            if pd.notna(prev.get("exam_datetime")) and pd.notna(curr.get("exam_datetime")):
                row["interval_days"] = (
                    curr["exam_datetime"] - prev["exam_datetime"]
                ).total_seconds() / 86400.0
            else:
                row["interval_days"] = np.nan

            for col in numeric_cols:
                a = prev.get(col, np.nan)
                b = curr.get(col, np.nan)
                row[f"{col}__from"] = a
                row[f"{col}__to"] = b
                row[f"{col}__delta"] = (
                    b - a if pd.notna(a) and pd.notna(b) else np.nan
                )

            records.append(row)

    return pd.DataFrame.from_records(records)


def summarize_deltas(
    changes: pd.DataFrame,
    analysis_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for col in analysis_cols:
        dcol = f"{col}__delta"
        if dcol not in changes.columns:
            continue

        use = changes[["patient_id", dcol]].dropna().copy()
        if use.empty:
            continue

        s = use[dcol]
        unchanged = np.isclose(s, 0.0, atol=cfg.ZERO_TOL, rtol=0.0)
        rows.append(
            {
                "feature": col,
                "interval_n": int(len(use)),
                "patient_n": int(use["patient_id"].nunique()),
                "median_delta": float(s.median()),
                "q1_delta": float(s.quantile(0.25)),
                "q3_delta": float(s.quantile(0.75)),
                "median_abs_delta": float(s.abs().median()),
                "increase_interval_n": int((s > cfg.ZERO_TOL).sum()),
                "decrease_interval_n": int((s < -cfg.ZERO_TOL).sum()),
                "unchanged_interval_n": int(unchanged.sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_patient_median_direction(
    changes: pd.DataFrame,
    analysis_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows = []
    summary_rows = []

    for col in analysis_cols:
        dcol = f"{col}__delta"
        if dcol not in changes.columns:
            continue

        use = changes[["patient_id", dcol]].dropna().copy()
        if use.empty:
            continue

        med = use.groupby("patient_id")[dcol].median()
        for patient_id, value in med.items():
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "feature": col,
                    "patient_median_adjacent_delta": value,
                }
            )

        summary_rows.append(
            {
                "feature": col,
                "patient_n": int(len(med)),
                "patients_positive_median_delta": int((med > cfg.ZERO_TOL).sum()),
                "patients_negative_median_delta": int((med < -cfg.ZERO_TOL).sum()),
                "patients_zero_median_delta": int(
                    np.isclose(
                        med,
                        0.0,
                        atol=cfg.ZERO_TOL,
                        rtol=0.0,
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(patient_rows), pd.DataFrame(summary_rows)


def qc_delta_associations(changes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    auto_features = cfg.AUTO_PRIMARY + cfg.AUTO_EXPLORATORY
    qc_features = list(dict.fromkeys(cfg.QC_NUMERIC_COLS))

    for feature in auto_features:
        fx = f"{feature}__delta"
        if fx not in changes.columns:
            continue

        for qc in qc_features:
            qx = f"{qc}__delta"
            if qx not in changes.columns:
                continue

            sub = changes[["patient_id", fx, qx]].dropna()
            if len(sub) < 3:
                continue
            if sub[fx].nunique() < 2 or sub[qx].nunique() < 2:
                rho = np.nan
            else:
                rho = float(spearmanr(sub[fx], sub[qx]).statistic)

            rows.append(
                {
                    "dynamic_feature": feature,
                    "qc_feature": qc,
                    "interval_n": int(len(sub)),
                    "patient_n": int(sub["patient_id"].nunique()),
                    "spearman_rho_descriptive": rho,
                    "note": "Descriptive only; repeated intervals within a patient are not independent.",
                }
            )

    return pd.DataFrame(rows)


def plot_trajectories(df: pd.DataFrame, features: list[str]) -> None:
    outdir = cfg.FIGURES / f"02_{cfg.ANALYSIS_COHORT}_trajectories"
    outdir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        if feature not in df.columns:
            continue

        use = df[["patient_id", "exam_order", feature]].copy()
        use[feature] = pd.to_numeric(use[feature], errors="coerce")
        use = use.dropna(subset=[feature])
        if use.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for _, g in use.groupby("patient_id"):
            g = g.sort_values("exam_order")
            ax.plot(
                g["exam_order"],
                g[feature],
                marker="o",
                linewidth=1,
                alpha=0.55,
            )

        ax.set_xlabel("Within-patient examination order")
        ax.set_ylabel(feature)
        ax.set_title(
            f"{feature}: within-patient trajectories "
            f"({cfg.ANALYSIS_COHORT} cohort)"
        )
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"{feature}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    cfg.ensure_output_dirs()
    df = _load_analysis_data()
    numeric_cols = _numeric_present(df)

    changes = build_adjacent_deltas(df, numeric_cols)
    changes.to_csv(
        cfg.TABLES / f"adjacent_deltas_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    analysis_cols = [
        c for c in cfg.ALL_CONTINUOUS_COLS if c in df.columns
    ]
    delta_summary = summarize_deltas(changes, analysis_cols)
    delta_summary.to_csv(
        cfg.TABLES / f"delta_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    patient_detail, patient_direction = summarize_patient_median_direction(
        changes,
        analysis_cols,
    )
    patient_detail.to_csv(
        cfg.TABLES / f"patient_median_delta_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )
    patient_direction.to_csv(
        cfg.TABLES / f"patient_direction_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    qc_assoc = qc_delta_associations(changes)
    qc_assoc.to_csv(
        cfg.TABLES / f"delta_qc_associations_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    plot_trajectories(
        df,
        [
            c for c in (
                cfg.PRIMARY_ANALYSIS_COLS
                + cfg.EXPLORATORY_ANALYSIS_COLS
            )
            if c in df.columns
        ],
    )

    print(f"cohort={cfg.ANALYSIS_COHORT}")
    print(f"patients={df['patient_id'].nunique()}, exams={len(df)}")
    print(f"adjacent intervals={len(changes)}")
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
