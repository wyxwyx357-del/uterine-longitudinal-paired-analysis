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


def discordant_pairs(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    required = {"patient_id", "clinical_pregnancy"}
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

        if len(g) != 2 or set(g["clinical_pregnancy"]) != {0, 1}:
            continue

        failure = g.loc[g["clinical_pregnancy"] == 0].iloc[0]
        success = g.loc[g["clinical_pregnancy"] == 1].iloc[0]

        row = {"patient_id": patient_id}
        for feature in features:
            if feature not in df.columns:
                continue
            a = pd.to_numeric(pd.Series([failure[feature]]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([success[feature]]), errors="coerce").iloc[0]
            row[f"{feature}__failure"] = a
            row[f"{feature}__success"] = b
            row[f"{feature}__success_minus_failure"] = b - a
        rows.append(row)

    return pd.DataFrame(rows)


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
                "n": int(s.size),
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
    paired = discordant_pairs(source, args.features)
    summary = summarize(paired, args.features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "discordant_pregnancy_case_level.csv", index=False)
    summary.to_csv(args.output_dir / "discordant_pregnancy_summary.csv", index=False)

    print(f"discordant_patients={paired['patient_id'].nunique() if not paired.empty else 0}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
