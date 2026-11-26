#!/usr/bin/env python3
"""Minimum real-data reproducibility checks for longitudinal frequency.

Checks the two analysis choices introduced by the longitudinal definition:
local tangent construction and continuous-window choice.  No outcomes are read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import extract_longitudinal_frequency as longitudinal


REGIONS = ("cervical", "middle", "fundal")
WALLS = ("anterior", "posterior")


def pca_tangents(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, float)
    overall = points[-1] - points[0]
    output = []
    for index in range(len(points)):
        if index == 0:
            local = points[:3]
        elif index == len(points) - 1:
            local = points[-3:]
        else:
            local = points[index - 1:index + 2]
        centered = local - local.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0]
        if direction @ overall < 0:
            direction = -direction
        output.append(direction / np.linalg.norm(direction))
    return np.asarray(output)


def icc_absolute_agreement(values_a: np.ndarray, values_b: np.ndarray) -> float:
    matrix = np.column_stack((values_a, values_b)).astype(float)
    n, k = matrix.shape
    if n < 3:
        return math.nan
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    ms_rows = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_columns = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = (
        ms_rows + (k - 1) * ms_error
        + k * (ms_columns - ms_error) / n
    )
    return float((ms_rows - ms_error) / denominator) if abs(denominator) > 1e-12 else math.nan


def comparison_summary(table: pd.DataFrame, variant: str, stratum: str) -> dict[str, object]:
    own = table if stratum == "all" else table.loc[table["baseline_confidence"] == stratum]
    both = own.loc[
        own["baseline_retained"] & own[f"{variant}_retained"]
        & own["baseline_frequency_cpm"].notna()
        & own[f"{variant}_frequency_cpm"].notna()
    ].copy()
    difference = both[f"{variant}_frequency_cpm"] - both["baseline_frequency_cpm"]
    absolute = difference.abs()
    return {
        "variant": variant,
        "baseline_stratum": stratum,
        "regions_n": int(len(own)),
        "both_retained_frequency_n": int(len(both)),
        "retained_status_agreement": float(
            (own["baseline_retained"] == own[f"{variant}_retained"]).mean()
        ) if len(own) else math.nan,
        "three_level_confidence_agreement": float(
            (own["baseline_confidence"] == own[f"{variant}_confidence"]).mean()
        ) if len(own) else math.nan,
        "baseline_retained_to_excluded_n": int(
            (own["baseline_retained"] & ~own[f"{variant}_retained"]).sum()
        ),
        "baseline_excluded_to_retained_n": int(
            (~own["baseline_retained"] & own[f"{variant}_retained"]).sum()
        ),
        "frequency_icc_absolute_agreement": (
            icc_absolute_agreement(
                both["baseline_frequency_cpm"].to_numpy(),
                both[f"{variant}_frequency_cpm"].to_numpy(),
            ) if len(both) >= 3 else math.nan
        ),
        "frequency_mean_difference_cpm": float(difference.mean()) if len(both) else math.nan,
        "frequency_median_absolute_difference_cpm": float(absolute.median()) if len(both) else math.nan,
        "frequency_p90_absolute_difference_cpm": float(absolute.quantile(0.90)) if len(both) else math.nan,
        "frequency_within_0.25_cpm_fraction": float((absolute <= 0.25 + 1e-12).mean()) if len(both) else math.nan,
        "bland_altman_lower_95_cpm": float(difference.mean() - 1.96 * difference.std(ddof=1)) if len(both) > 1 else math.nan,
        "bland_altman_upper_95_cpm": float(difference.mean() + 1.96 * difference.std(ddof=1)) if len(both) > 1 else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--longitudinal-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    source_regions = pd.read_csv(
        args.source_run / "merged_results" / "optimized_region_frequency.csv"
    )
    baseline = pd.read_csv(
        args.longitudinal_run / "longitudinal_region_frequency.csv"
    )
    cache_paths = longitudinal.cache_map(args.source_run)
    records = []
    cases = sorted(cache_paths)
    for case_order, case_id in enumerate(cases, start=1):
        tracked, initial, anterior_inward, posterior_inward = longitudinal.load_cache(cache_paths[case_id])
        count = len(initial) // 2
        inward = np.vstack((anterior_inward, posterior_inward))
        standard = np.vstack((
            longitudinal.local_tangents(initial[:count]),
            longitudinal.local_tangents(initial[count:]),
        ))
        pca = np.vstack((pca_tangents(initial[:count]), pca_tangents(initial[count:])))
        region_map = dict(zip(REGIONS, np.array_split(np.arange(count), 3)))
        for wall in WALLS:
            offset = 0 if wall == "anterior" else count
            for region in REGIONS:
                key = (case_id, wall, region)
                source = source_regions.loc[
                    (source_regions["case_id"].astype(str) == case_id)
                    & (source_regions["wall"] == wall)
                    & (source_regions["region"] == region)
                ].iloc[0]
                base = baseline.loc[
                    (baseline["case_id"].astype(str) == case_id)
                    & (baseline["wall"] == wall)
                    & (baseline["region"] == region)
                ].iloc[0]
                local = region_map[region]
                columns = local + offset
                axis_result, _ = longitudinal.analyse_region(
                    source, tracked, initial, pca, inward, local, columns
                )
                cropped_source = source.copy()
                start = int(longitudinal.finite_float(source.get("selected_start_frame"), -1))
                stop = int(longitudinal.finite_float(source.get("selected_stop_frame_exclusive"), -1))
                if start >= 0 and stop > start:
                    trim = max(1, int(math.floor((stop - start) * 0.10)))
                    cropped_source["selected_start_frame"] = start + trim
                    cropped_source["selected_stop_frame_exclusive"] = stop - trim
                window_result, _ = longitudinal.analyse_region(
                    cropped_source, tracked, initial, standard, inward, local, columns
                )
                records.append({
                    "case_id": case_id,
                    "wall": wall,
                    "region": region,
                    "baseline_frequency_cpm": pd.to_numeric(base.get("longitudinal_frequency_cpm"), errors="coerce"),
                    "baseline_confidence": str(base["longitudinal_technical_confidence"]),
                    "baseline_retained": bool(base["retained_for_longitudinal_model"]),
                    "pca_axis_frequency_cpm": longitudinal.finite_float(axis_result.get("longitudinal_frequency_cpm")),
                    "pca_axis_confidence": axis_result["longitudinal_technical_confidence"],
                    "pca_axis_retained": bool(axis_result["retained_for_longitudinal_model"]),
                    "cropped_window_frequency_cpm": longitudinal.finite_float(window_result.get("longitudinal_frequency_cpm")),
                    "cropped_window_confidence": window_result["longitudinal_technical_confidence"],
                    "cropped_window_retained": bool(window_result["retained_for_longitudinal_model"]),
                })
        if case_order % 25 == 0 or case_order == len(cases):
            print(f"cases {case_order}/{len(cases)}", flush=True)

    table = pd.DataFrame(records)
    # Convert non-finite float placeholders to missing values for paired summaries.
    for column in ["pca_axis_frequency_cpm", "cropped_window_frequency_cpm"]:
        table.loc[~np.isfinite(table[column]), column] = np.nan
    summaries = []
    for variant in ["pca_axis", "cropped_window"]:
        for stratum in ["all", "high", "medium", "excluded"]:
            summaries.append(comparison_summary(table, variant, stratum))
    summary_table = pd.DataFrame(summaries)
    table.to_csv(args.output / "region_reproducibility_details.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(args.output / "reproducibility_summary.csv", index=False, encoding="utf-8-sig")

    high = summary_table.set_index(["variant", "baseline_stratum"])
    axis = high.loc[("pca_axis", "high")]
    window = high.loc[("cropped_window", "high")]
    report = f"""# 纵向频率最低必要重复性验证

## 设计

- 全部{len(cases)}例、{len(table)}个区域，不读取临床结局。
- 轴扰动：中心差分局部切线改为三点局部PCA切线。
- 窗口扰动：原连续片段首尾各裁剪10%。
- 报告绝对一致ICC、Bland–Altman界限、绝对频率差和技术分级一致率。

## 高可信区域结果

- PCA切线：共同保留{int(axis['both_retained_frequency_n'])}个；ICC={axis['frequency_icc_absolute_agreement']:.3f}；频率差中位数={axis['frequency_median_absolute_difference_cpm']:.3f} cpm；≤0.25 cpm比例={axis['frequency_within_0.25_cpm_fraction']:.1%}；保留状态一致率={axis['retained_status_agreement']:.1%}。
- 裁剪窗口：共同保留{int(window['both_retained_frequency_n'])}个；ICC={window['frequency_icc_absolute_agreement']:.3f}；频率差中位数={window['frequency_median_absolute_difference_cpm']:.3f} cpm；≤0.25 cpm比例={window['frequency_within_0.25_cpm_fraction']:.1%}；保留状态一致率={window['retained_status_agreement']:.1%}。

这些结果衡量分析选择的重复性，不是生理真值，也不替代外部数据验证。
"""
    (args.output / "重复性验证报告.md").write_text(report, encoding="utf-8")
    config = {
        "clinical_outcome_read": False,
        "patients": len(cases),
        "regions": len(table),
        "axis_variant": "three-point local PCA tangent, sign aligned cervix-to-fundus",
        "window_variant": "trim 10 percent of frames from each end",
        "frequency_tolerance_reported_cpm": 0.25,
        "statistics": ["ICC absolute agreement", "Bland-Altman", "absolute difference", "classification agreement"],
    }
    (args.output / "reproducibility_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
