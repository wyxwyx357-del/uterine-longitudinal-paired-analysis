#!/usr/bin/env python3
"""Outcome-blind test of robust multi-window consensus for medium confidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import extract_longitudinal_frequency as longitudinal
from validate_reproducibility import icc_absolute_agreement


REGIONS = ("cervical", "middle", "fundal")
WALLS = ("anterior", "posterior")
TOLERANCE_CPM = 0.25
DETAIL_COLUMNS = [
    "case_id", "wall", "region", "full_frequency_cpm",
    "trim_front_10_frequency_cpm", "trim_back_10_frequency_cpm",
    "three_window_range_cpm", "consensus_candidate",
    "consensus_frequency_cpm", "holdout_trim_both_frequency_cpm",
    "holdout_trim_both_retained", "holdout_absolute_difference_cpm",
    "validated_stable_medium",
]


def perturbed_source(source: pd.Series, trim_front: float, trim_back: float) -> pd.Series:
    output = source.copy()
    start = int(longitudinal.finite_float(source.get("selected_start_frame"), -1))
    stop = int(longitudinal.finite_float(source.get("selected_stop_frame_exclusive"), -1))
    if start < 0 or stop <= start:
        return output
    length = stop - start
    output["selected_start_frame"] = start + int(math.floor(length * trim_front))
    output["selected_stop_frame_exclusive"] = stop - int(math.floor(length * trim_back))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--longitudinal-run", type=Path, required=True)
    parser.add_argument("--reproducibility-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    source_regions = pd.read_csv(
        args.source_run / "merged_results" / "optimized_region_frequency.csv",
        dtype={"case_id": str},
    )
    baseline = pd.read_csv(
        args.longitudinal_run / "longitudinal_region_frequency.csv",
        dtype={"case_id": str},
    )
    reproducibility = pd.read_csv(
        args.reproducibility_run / "region_reproducibility_details.csv",
        dtype={"case_id": str},
    )
    source_regions["case_id"] = source_regions["case_id"].astype(str).str.strip()
    baseline["case_id"] = baseline["case_id"].astype(str).str.strip()
    reproducibility["case_id"] = reproducibility["case_id"].astype(str).str.strip()
    medium = baseline.loc[
        baseline["longitudinal_technical_confidence"].eq("medium")
    ].copy()
    cache_paths = longitudinal.cache_map(args.source_run)
    records = []

    for case_order, (case_id, own_medium) in enumerate(
        medium.groupby(medium["case_id"].astype(str), sort=True), start=1
    ):
        tracked, initial, anterior_inward, posterior_inward = longitudinal.load_cache(cache_paths[case_id])
        count = len(initial) // 2
        tangents = np.vstack((
            longitudinal.local_tangents(initial[:count]),
            longitudinal.local_tangents(initial[count:]),
        ))
        inward = np.vstack((anterior_inward, posterior_inward))
        region_map = dict(zip(REGIONS, np.array_split(np.arange(count), 3)))
        for _, base in own_medium.iterrows():
            wall, region = str(base["wall"]), str(base["region"])
            source = source_regions.loc[
                (source_regions["case_id"].astype(str) == case_id)
                & source_regions["wall"].eq(wall)
                & source_regions["region"].eq(region)
            ].iloc[0]
            local = region_map[region]
            columns = local + (0 if wall == "anterior" else count)
            estimates = {}
            for name, front, back in [
                ("trim_front_10", 0.10, 0.0),
                ("trim_back_10", 0.0, 0.10),
            ]:
                result, _ = longitudinal.analyse_region(
                    perturbed_source(source, front, back), tracked, initial,
                    tangents, inward, local, columns,
                )
                estimates[name] = result
            holdout = reproducibility.loc[
                (reproducibility["case_id"].astype(str) == case_id)
                & reproducibility["wall"].eq(wall)
                & reproducibility["region"].eq(region)
            ].iloc[0]
            full_frequency = float(base["longitudinal_frequency_cpm"])
            front_frequency = longitudinal.finite_float(
                estimates["trim_front_10"].get("longitudinal_frequency_cpm")
            )
            back_frequency = longitudinal.finite_float(
                estimates["trim_back_10"].get("longitudinal_frequency_cpm")
            )
            three = np.asarray([full_frequency, front_frequency, back_frequency], float)
            all_finite = bool(np.isfinite(three).all())
            all_retained = bool(
                estimates["trim_front_10"]["retained_for_longitudinal_model"]
                and estimates["trim_back_10"]["retained_for_longitudinal_model"]
            )
            range_cpm = float(np.ptp(three)) if all_finite else math.nan
            consensus_candidate = bool(
                all_finite and all_retained and range_cpm <= TOLERANCE_CPM + 1e-12
            )
            consensus = float(np.median(three)) if consensus_candidate else math.nan
            holdout_frequency = pd.to_numeric(
                holdout["cropped_window_frequency_cpm"], errors="coerce"
            )
            holdout_retained = bool(holdout["cropped_window_retained"])
            holdout_difference = (
                abs(float(holdout_frequency) - consensus)
                if consensus_candidate and np.isfinite(holdout_frequency) else math.nan
            )
            validated = bool(
                consensus_candidate and holdout_retained
                and np.isfinite(holdout_difference)
                and holdout_difference <= TOLERANCE_CPM + 1e-12
            )
            records.append({
                "case_id": case_id,
                "wall": wall,
                "region": region,
                "full_frequency_cpm": full_frequency,
                "trim_front_10_frequency_cpm": front_frequency,
                "trim_back_10_frequency_cpm": back_frequency,
                "three_window_range_cpm": range_cpm,
                "consensus_candidate": consensus_candidate,
                "consensus_frequency_cpm": consensus,
                "holdout_trim_both_frequency_cpm": holdout_frequency,
                "holdout_trim_both_retained": holdout_retained,
                "holdout_absolute_difference_cpm": holdout_difference,
                "validated_stable_medium": validated,
            })
        if case_order % 25 == 0:
            print(f"medium-patient groups {case_order}", flush=True)

    table = pd.DataFrame(records, columns=DETAIL_COLUMNS)
    candidates = table.loc[table["consensus_candidate"]].copy()
    paired_candidates = candidates.loc[
        candidates["consensus_frequency_cpm"].notna()
        & candidates["holdout_trim_both_frequency_cpm"].notna()
    ].copy()
    validated = table.loc[table["validated_stable_medium"]].copy()
    patient_availability = baseline.groupby("case_id").agg(
        has_high=("longitudinal_technical_confidence", lambda s: bool((s == "high").any())),
        has_medium=("longitudinal_technical_confidence", lambda s: bool((s == "medium").any())),
    ).reset_index()
    validated_cases = set(validated["case_id"].astype(str))
    patient_availability["has_validated_stable_medium"] = (
        patient_availability["case_id"].astype(str).isin(validated_cases)
    )
    patient_availability["eligible_high_or_validated_medium"] = (
        patient_availability["has_high"]
        | patient_availability["has_validated_stable_medium"]
    )

    if len(paired_candidates) >= 3:
        icc = icc_absolute_agreement(
            paired_candidates["consensus_frequency_cpm"].to_numpy(float),
            paired_candidates["holdout_trim_both_frequency_cpm"].to_numpy(float),
        )
        holdout_within = float(
            (paired_candidates["holdout_absolute_difference_cpm"] <= TOLERANCE_CPM + 1e-12).mean()
        )
    else:
        icc, holdout_within = math.nan, math.nan
    summary = {
        "clinical_outcome_read": False,
        "medium_regions_tested": int(len(table)),
        "three_window_consensus_candidates": int(len(candidates)),
        "three_window_consensus_candidate_fraction": (
            float(len(candidates) / len(table)) if len(table) else 0.0
        ),
        "consensus_candidates_with_finite_holdout": int(len(paired_candidates)),
        "validated_stable_medium_regions": int(len(validated)),
        "validated_stable_medium_fraction": (
            float(len(validated) / len(table)) if len(table) else 0.0
        ),
        "consensus_vs_independent_both_trim_icc": icc,
        "consensus_vs_independent_both_trim_within_0_25_fraction_among_candidates": holdout_within,
        "validated_stable_medium_patients": int(len(validated_cases)),
        "patients_with_high": int(patient_availability["has_high"].sum()),
        "patients_high_or_validated_stable_medium": int(
            patient_availability["eligible_high_or_validated_medium"].sum()
        ),
        "additional_patients_beyond_high_only": int(
            patient_availability["eligible_high_or_validated_medium"].sum()
            - patient_availability["has_high"].sum()
        ),
        "consensus_definition": "median of full, trim-front-10%, trim-back-10%; all retained and max-min <=0.25 cpm",
        "independent_perturbation": "trim both ends by 10%, not used to form consensus",
    }
    table.to_csv(args.output / "medium_consensus_details.csv", index=False, encoding="utf-8-sig")
    patient_availability.to_csv(
        args.output / "patient_availability_after_consensus.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "medium_consensus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 中可信纵向频率多窗口共识测试

- 测试中可信区域：{summary['medium_regions_tested']}个。
- 完整/去前10%/去后10%三窗口一致候选：{summary['three_window_consensus_candidates']}个（{summary['three_window_consensus_candidate_fraction']:.1%}）。
- 再经独立首尾各裁10%验证：{summary['validated_stable_medium_regions']}个（{summary['validated_stable_medium_fraction']:.1%}）。
- 共识频率与独立扰动ICC：{summary['consensus_vs_independent_both_trim_icc']:.3f}。
- 高可信患者：{summary['patients_with_high']}例；加入验证稳定的中可信后：{summary['patients_high_or_validated_stable_medium']}例，新增{summary['additional_patients_beyond_high_only']}例。

本测试不读取临床结局。只有通过独立扰动验证的中可信区域，才可考虑与高可信区域共同进入主要分析；其余中可信区域仍仅作探索。
"""
    (args.output / "中可信共识测试报告.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
