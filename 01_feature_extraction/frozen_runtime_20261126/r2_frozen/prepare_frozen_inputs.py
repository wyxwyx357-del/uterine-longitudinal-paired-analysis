#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ["case_id", "wall", "region"]


def write_input(data: pd.DataFrame, path: Path, include_medium: bool) -> dict[str, int]:
    high = (
        data["retained_for_longitudinal_model"].astype(bool)
        & data["longitudinal_technical_confidence"].eq("high")
    )
    stable_medium = data["validated_stable_medium"].eq(True)
    eligible = high | stable_medium if include_medium else high
    output = data.copy()
    output["optimized_candidate_frequency_cpm"] = output["longitudinal_frequency_cpm"]
    if include_medium:
        output.loc[stable_medium, "optimized_candidate_frequency_cpm"] = output.loc[
            stable_medium, "consensus_frequency_cpm"
        ]
    output["technical_confidence"] = output["longitudinal_technical_confidence"]
    output["retained_for_patient_model"] = eligible
    output.to_csv(path, index=False, encoding="utf-8-sig")
    posterior = eligible & output["wall"].eq("posterior")
    return {
        "eligible_regions": int(eligible.sum()),
        "eligible_posterior_regions": int(posterior.sum()),
        "patients_with_any_eligible_region": int(output.loc[eligible, "case_id"].nunique()),
        "patients_with_posterior_region": int(output.loc[posterior, "case_id"].nunique()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--longitudinal-regions", type=Path, required=True)
    parser.add_argument("--medium-consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    regions = pd.read_csv(
        args.longitudinal_regions,
        dtype={"case_id": str},
    )
    consensus = pd.read_csv(
        args.medium_consensus,
        dtype={"case_id": str},
    )[
        KEYS + ["consensus_frequency_cpm", "validated_stable_medium"]
    ]
    regions["case_id"] = regions["case_id"].astype(str).str.strip()
    consensus["case_id"] = consensus["case_id"].astype(str).str.strip()
    data = regions.merge(consensus, on=KEYS, how="left", validate="one_to_one")
    summary = {
        "clinical_outcome_read": False,
        "main": write_input(data, args.output / "main.csv", True),
        "high_only": write_input(data, args.output / "high_only.csv", False),
    }
    (args.output / "input_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
