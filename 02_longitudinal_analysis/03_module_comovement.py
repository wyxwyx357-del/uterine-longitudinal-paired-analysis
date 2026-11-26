from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

import config as cfg


def _load_analysis_data() -> pd.DataFrame:
    path = cfg.TABLES / "master_long.csv"
    if not path.exists():
        raise FileNotFoundError("Run 01_build_master_table.py first.")

    df = pd.read_csv(
        path,
        dtype={"exam_id": str, "patient_id": str},
    )
    df = df.loc[cfg.cohort_mask(df)].copy()
    if df.empty:
        raise ValueError(f"No rows available for cohort={cfg.ANALYSIS_COHORT}")

    df["exam_datetime"] = pd.to_datetime(df["exam_datetime"], errors="coerce")
    for col in cfg.ALL_CONTINUOUS_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "days_to_transfer" in df.columns:
        df["days_to_transfer"] = pd.to_numeric(
            df["days_to_transfer"],
            errors="coerce",
        )
    return df.sort_values(["patient_id", "exam_order"])


def run_rm_corr(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for x, y in cfg.KEY_PAIRS:
        if x not in df.columns or y not in df.columns:
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "status": "missing_column",
                }
            )
            continue

        sub = df[["patient_id", x, y]].dropna().copy()
        counts = sub.groupby("patient_id").size()
        valid_ids = counts[counts >= 2].index
        sub = sub[sub["patient_id"].isin(valid_ids)].copy()

        n_patients = sub["patient_id"].nunique()
        n_exams = len(sub)

        if (
            n_patients < cfg.MIN_RMCORR_PATIENTS
            or sub[x].nunique() < 2
            or sub[y].nunique() < 2
        ):
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "status": "insufficient_variation_or_patients",
                    "n_patients": int(n_patients),
                    "n_exams": int(n_exams),
                }
            )
            continue

        try:
            result = pg.rm_corr(
                data=sub,
                x=x,
                y=y,
                subject="patient_id",
            )
            r = float(result["r"].iloc[0])
            pval = float(result["pval"].iloc[0])
            dof = float(result["dof"].iloc[0])
            ci = result["CI95%"].iloc[0] if "CI95%" in result.columns else np.nan

            rows.append(
                {
                    "x": x,
                    "y": y,
                    "status": "ok",
                    "n_patients": int(n_patients),
                    "n_exams": int(n_exams),
                    "rmcorr_r": r,
                    "rmcorr_dof": dof,
                    "rmcorr_p": pval,
                    "rmcorr_ci95": str(ci),
                    "note": (
                        "Exploratory repeated-measures association; "
                        "interpret with individual trajectories and clinical timing."
                    ),
                }
            )
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "status": f"rmcorr_failed:{type(exc).__name__}",
                    "n_patients": int(n_patients),
                    "n_exams": int(n_exams),
                }
            )

    return pd.DataFrame(rows)


def plot_pair(df: pd.DataFrame, x: str, y: str, r: float | None) -> None:
    sub = df[["patient_id", "exam_order", x, y]].dropna().copy()
    counts = sub.groupby("patient_id").size()
    sub = sub[sub["patient_id"].isin(counts[counts >= 2].index)]
    if sub.empty:
        return

    outdir = cfg.FIGURES / f"03_{cfg.ANALYSIS_COHORT}_rmcorr"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for _, g in sub.groupby("patient_id"):
        g = g.sort_values("exam_order")
        ax.plot(
            g[x],
            g[y],
            marker="o",
            linewidth=1,
            alpha=0.45,
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    title = f"{x} vs {y}: within-patient repeated measures"
    if r is not None and np.isfinite(r):
        title += f"\nrmcorr r={r:.3f}"
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / f"{x}__{y}.png", dpi=180)
    plt.close(fig)


def timing_protocol_context(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for patient_id, g in df.groupby("patient_id"):
        row = {
            "patient_id": patient_id,
            "exam_n": int(len(g)),
        }

        if "days_to_transfer" in g.columns:
            s = pd.to_numeric(g["days_to_transfer"], errors="coerce").dropna()
            row["days_to_transfer_min"] = s.min() if not s.empty else np.nan
            row["days_to_transfer_max"] = s.max() if not s.empty else np.nan
            row["days_to_transfer_range"] = (
                s.max() - s.min() if not s.empty else np.nan
            )

        if "protocol" in g.columns:
            row["protocol_nunique"] = int(g["protocol"].dropna().nunique())
            row["protocol_consistent"] = (
                g["protocol"].dropna().nunique() <= 1
            )

        rows.append(row)

    return pd.DataFrame(rows)


def endometrium_type_transition_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "endometrium_type" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    targets = [
        c for c in [
            "vfi",
            "pi",
            "clinical_total_peristalsis_rate_min",
            "anterior_amplitude",
            "posterior_amplitude",
        ]
        if c in df.columns
    ]

    interval_rows = []
    for patient_id, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("exam_order").reset_index(drop=True)
        for i in range(1, len(g)):
            a = g.iloc[i - 1]
            b = g.iloc[i]
            type_a = a.get("endometrium_type")
            type_b = b.get("endometrium_type")
            if pd.isna(type_a) or pd.isna(type_b):
                continue

            row = {
                "patient_id": patient_id,
                "from_exam_id": a["exam_id"],
                "to_exam_id": b["exam_id"],
                "type_from": type_a,
                "type_to": type_b,
                "type_changed": str(type_a) != str(type_b),
            }

            for col in targets:
                va = pd.to_numeric(pd.Series([a[col]]), errors="coerce").iloc[0]
                vb = pd.to_numeric(pd.Series([b[col]]), errors="coerce").iloc[0]
                row[f"{col}__delta"] = (
                    vb - va if pd.notna(va) and pd.notna(vb) else np.nan
                )
            interval_rows.append(row)

    intervals = pd.DataFrame(interval_rows)
    if intervals.empty:
        return intervals, pd.DataFrame()

    summary_rows = []
    for changed, group in intervals.groupby("type_changed"):
        for col in targets:
            dcol = f"{col}__delta"
            s = group[dcol].dropna()
            if s.empty:
                continue
            summary_rows.append(
                {
                    "type_changed": bool(changed),
                    "feature": col,
                    "interval_n": int(len(s)),
                    "patient_n": int(group.loc[s.index, "patient_id"].nunique()),
                    "median_delta": float(s.median()),
                    "q1_delta": float(s.quantile(0.25)),
                    "q3_delta": float(s.quantile(0.75)),
                }
            )

    return intervals, pd.DataFrame(summary_rows)


def main() -> None:
    cfg.ensure_output_dirs()
    df = _load_analysis_data()

    results = run_rm_corr(df)
    results.to_csv(
        cfg.TABLES / f"rmcorr_results_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    for _, row in results.iterrows():
        if row.get("status") != "ok":
            continue
        plot_pair(
            df,
            row["x"],
            row["y"],
            row.get("rmcorr_r"),
        )

    context = timing_protocol_context(df)
    context.to_csv(
        cfg.TABLES / f"timing_protocol_context_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    transitions, transition_summary = endometrium_type_transition_summary(df)
    transitions.to_csv(
        cfg.TABLES / f"endometrium_type_transitions_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )
    transition_summary.to_csv(
        cfg.TABLES / f"endometrium_type_transition_summary_{cfg.ANALYSIS_COHORT}.csv",
        index=False,
    )

    print(f"cohort={cfg.ANALYSIS_COHORT}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
