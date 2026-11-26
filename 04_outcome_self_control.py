from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import config as cfg


def _load_analysis_data() -> pd.DataFrame:
    path = cfg.TABLES / "master_long.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_build_master_table.py first.")

    df = pd.read_csv(path)
    df = df.loc[cfg.cohort_mask(df)].copy()
    df["clinical_pregnancy"] = pd.to_numeric(
        df["clinical_pregnancy"],
        errors="coerce",
    )
    df = df[df["clinical_pregnancy"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("No known-outcome examinations in selected cohort.")

    df["exam_datetime"] = pd.to_datetime(df["exam_datetime"], errors="coerce")
    for col in cfg.ALL_CONTINUOUS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(["patient_id", "exam_order", "exam_datetime"])


def discordant_patients(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("patient_id")["clinical_pregnancy"].nunique()
    ids = counts[counts == 2].index
    return df[df["patient_id"].isin(ids)].copy()


def patient_outcome_summary(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    for patient_id, g in df.groupby("patient_id", sort=False):
        fail = g[g["clinical_pregnancy"] == 0]
        succ = g[g["clinical_pregnancy"] == 1]
        if fail.empty or succ.empty:
            continue

        row = {
            "patient_id": patient_id,
            "failure_exam_n": int(len(fail)),
            "success_exam_n": int(len(succ)),
        }

        for col in features:
            if col not in g.columns:
                continue
            f = pd.to_numeric(fail[col], errors="coerce").dropna()
            s = pd.to_numeric(succ[col], errors="coerce").dropna()

            fmed = f.median() if not f.empty else np.nan
            smed = s.median() if not s.empty else np.nan

            row[f"{col}__failure_median"] = fmed
            row[f"{col}__success_median"] = smed
            row[f"{col}__success_minus_failure"] = smed - fmed

        rows.append(row)

    return pd.DataFrame(rows)


def feature_summary(
    patient_summary: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []

    for col in features:
        dcol = f"{col}__success_minus_failure"
        if dcol not in patient_summary.columns:
            continue

        diff = pd.to_numeric(
            patient_summary[dcol],
            errors="coerce",
        ).dropna()
        if diff.empty:
            continue

        nonzero = diff[~np.isclose(diff, 0.0, atol=cfg.ZERO_TOL, rtol=0.0)]
        if len(nonzero) >= 2:
            try:
                test = wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    method="auto",
                )
                pval = float(test.pvalue)
            except ValueError:
                pval = np.nan
        else:
            pval = np.nan

        rows.append(
            {
                "feature": col,
                "patient_n": int(len(diff)),
                "median_success_minus_failure": float(diff.median()),
                "q1": float(diff.quantile(0.25)),
                "q3": float(diff.quantile(0.75)),
                "success_higher_n": int((diff > cfg.ZERO_TOL).sum()),
                "success_lower_n": int((diff < -cfg.ZERO_TOL).sum()),
                "same_n": int(
                    np.isclose(
                        diff,
                        0.0,
                        atol=cfg.ZERO_TOL,
                        rtol=0.0,
                    ).sum()
                ),
                "wilcoxon_p_descriptive": pval,
                "note": (
                    "Exploratory only; p-value is descriptive and not a primary success criterion."
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_slope(
    patient_summary: pd.DataFrame,
    features: list[str],
) -> None:
    outdir = cfg.FIGURES / f"04_{cfg.ANALYSIS_COHORT}_outcome_slope"
    outdir.mkdir(parents=True, exist_ok=True)

    for col in features:
        fcol = f"{col}__failure_median"
        scol = f"{col}__success_median"
        if fcol not in patient_summary.columns or scol not in patient_summary.columns:
            continue

        use = patient_summary[["patient_id", fcol, scol]].dropna()
        if use.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for _, row in use.iterrows():
            y0 = row[fcol]
            y1 = row[scol]
            if y1 > y0:
                color = "tab:red"
            elif y1 < y0:
                color = "tab:blue"
            else:
                color = "0.5"

            ax.plot(
                [0, 1],
                [y0, y1],
                marker="o",
                linewidth=1.2,
                alpha=0.75,
                color=color,
            )

        ax.set_xticks([0, 1], ["Failure", "Success"])
        ax.set_ylabel(col)
        ax.set_title(f"{col}: within-patient outcome comparison")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"{col}.png", dpi=180)
        plt.close(fig)


def nearest_opposite_outcome_pairs(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Choose one nearest opposite-outcome pair per patient, deterministically.

    This is a sensitivity analysis only. It does not replace the all-cycle
    patient-level median comparison.
    """
    rows = []

    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.sort_values(["exam_datetime", "exam_order", "exam_id"]).reset_index(drop=True)
        candidates = []

        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a = g.iloc[i]
                b = g.iloc[j]
                if a["clinical_pregnancy"] == b["clinical_pregnancy"]:
                    continue

                if pd.notna(a["exam_datetime"]) and pd.notna(b["exam_datetime"]):
                    distance = abs(
                        (b["exam_datetime"] - a["exam_datetime"]).total_seconds()
                    )
                else:
                    distance = abs(float(b["exam_order"]) - float(a["exam_order"]))

                candidates.append((distance, i, j))

        if not candidates:
            continue

        _, i, j = sorted(candidates, key=lambda x: (x[0], x[1], x[2]))[0]
        a = g.iloc[i]
        b = g.iloc[j]

        failure = a if a["clinical_pregnancy"] == 0 else b
        success = a if a["clinical_pregnancy"] == 1 else b

        row = {
            "patient_id": patient_id,
            "failure_exam_id": failure["exam_id"],
            "success_exam_id": success["exam_id"],
            "failure_exam_order": failure["exam_order"],
            "success_exam_order": success["exam_order"],
        }

        for col in features:
            if col not in g.columns:
                continue
            fv = pd.to_numeric(
                pd.Series([failure[col]]),
                errors="coerce",
            ).iloc[0]
            sv = pd.to_numeric(
                pd.Series([success[col]]),
                errors="coerce",
            ).iloc[0]
            row[f"{col}__failure"] = fv
            row[f"{col}__success"] = sv
            row[f"{col}__success_minus_failure"] = (
                sv - fv if pd.notna(fv) and pd.notna(sv) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def sensitivity_feature_summary(
    pairs: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for col in features:
        dcol = f"{col}__success_minus_failure"
        if dcol not in pairs.columns:
            continue
        s = pd.to_numeric(pairs[dcol], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "feature": col,
                "patient_n": int(len(s)),
                "median_success_minus_failure": float(s.median()),
                "q1": float(s.quantile(0.25)),
                "q3": float(s.quantile(0.75)),
                "success_higher_n": int((s > cfg.ZERO_TOL).sum()),
                "success_lower_n": int((s < -cfg.ZERO_TOL).sum()),
                "same_n": int(
                    np.isclose(
                        s,
                        0.0,
                        atol=cfg.ZERO_TOL,
                        rtol=0.0,
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def export_review_workbook(df: pd.DataFrame) -> None:
    cols = [c for c in cfg.REVIEW_COLS if c in df.columns]
    extra = [
        c for c in cfg.PRIMARY_ANALYSIS_COLS
        if c in df.columns and c not in cols
    ]
    review = df[cols + extra].copy()

    review_path = (
        cfg.TABLES
        / f"discordant_case_review_{cfg.ANALYSIS_COHORT}.xlsx"
    )
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        review.to_excel(writer, index=False, sheet_name="all_exams")


def main() -> None:
    cfg.ensure_output_dirs()
    df = _load_analysis_data()
    disc = discordant_patients(df)

    if disc.empty:
        print("No patients with both success and failure outcomes.")
        return

    features = [
        c for c in (
            cfg.PRIMARY_ANALYSIS_COLS
            + cfg.SECONDARY_ANALYSIS_COLS
            + cfg.EXPLORATORY_ANALYSIS_COLS
        )
        if c in disc.columns
    ]

    disc.to_csv(
        cfg.TABLES / f"discordant_all_exams_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    patient_summary = patient_outcome_summary(disc, features)
    patient_summary.to_csv(
        cfg.TABLES / f"outcome_patient_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    summary = feature_summary(patient_summary, features)
    summary.to_csv(
        cfg.TABLES / f"outcome_feature_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    plot_slope(
        patient_summary,
        [c for c in cfg.PRIMARY_ANALYSIS_COLS if c in disc.columns],
    )

    nearest = nearest_opposite_outcome_pairs(disc, features)
    nearest.to_csv(
        cfg.TABLES / f"nearest_opposite_outcome_pairs_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )
    nearest_summary = sensitivity_feature_summary(nearest, features)
    nearest_summary.to_csv(
        cfg.TABLES / f"nearest_opposite_outcome_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    export_review_workbook(disc)

    print(f"cohort={cfg.ANALYSIS_COHORT}")
    print(f"discordant patients={disc['patient_id'].nunique()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
