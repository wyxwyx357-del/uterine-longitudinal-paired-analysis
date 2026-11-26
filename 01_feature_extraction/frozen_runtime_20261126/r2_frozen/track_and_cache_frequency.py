#!/usr/bin/env python3
"""Frozen outcome-blind local-tissue displacement frequency pipeline (V1).

The original R2 tracker and primary frequency fit remain unchanged.  This
frozen layer adds fixed-point support, technical usability, paired deformation,
and signal-origin evidence rules validated before full-cohort outcome analysis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "r2_tracking_matplotlib")
)
sys.path.insert(0, str(CODE))

import run_local_motion_frequency as frozen  # noqa: E402
import run_real_context_evidence as base  # noqa: E402


DURATION_CANDIDATES_S = (10.0, 15.0, 20.0)
CYCLE_CANDIDATES = (0.0, 1.0, 1.5)
TREND_IMPROVEMENT_CANDIDATES = (0.0, 0.05, 0.10)
CACHE_FORMAT_VERSION = 1
EVIDENCE_THRESHOLDS = {
    "control_fixed_frequency_score": 0.50,
    "control_periodic_correlation": 0.70,
    "background_to_tissue_range_ratio": 0.50,
    "minimum_point_phase_coherence": 0.50,
    "supported_point_phase_coherence": 0.70,
    "minimum_point_participation_fraction": 2.0 / 3.0,
    "minimum_mean_ncc": 0.80,
    "maximum_p90_forward_backward_px": 0.60,
}


def write_csv(path: Path, rows: list[dict], fallback: list[str]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    fields = fields or fallback
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_cache(
    path: Path,
    tracked: dict,
    initial: np.ndarray,
    anterior_axes: np.ndarray,
    posterior_axes: np.ndarray,
    metadata: dict,
) -> None:
    """Save everything needed to change region/frequency rules without video I/O."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            time_s=tracked["time_s"],
            initial_points=initial,
            anterior_axes=anterior_axes,
            posterior_axes=posterior_axes,
            positions=tracked["positions"],
            raw_positions=tracked["raw_positions"],
            corrected=tracked["corrected"],
            target_valid=tracked["valid"],
            raw_target_valid=tracked["raw_valid"],
            pose_valid=tracked["pose_valid"],
            epoch=tracked["epoch"],
            poses=tracked["poses"],
            target_ncc=tracked["target_ncc"],
            target_fb=tracked["target_fb"],
            point_brightness=tracked["point_brightness"],
            global_brightness=tracked["global_brightness"],
            heldout_raw=tracked["heldout_raw"],
            heldout_corrected=tracked["heldout_corrected"],
            training_inliers=tracked["training_inliers"],
            heldout_count=tracked["heldout_count"],
            cache_format_version=np.asarray([CACHE_FORMAT_VERSION], dtype=np.int32),
            case_id=np.asarray([metadata["case_id"]]),
            video_path=np.asarray([metadata["video_path"]]),
            video_size=np.asarray([metadata["video_size"]], dtype=np.int64),
            video_mtime_ns=np.asarray([metadata["video_mtime_ns"]], dtype=np.int64),
            tracking_sha256=np.asarray([metadata["tracking_sha256"]]),
            nodes_sha256=np.asarray([metadata["nodes_sha256"]]),
        )
    os.replace(temporary, path)


def load_cache(
    path: Path, expected: dict | None = None
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if expected is not None:
            actual = {
                "cache_format_version": int(archive["cache_format_version"][0]),
                "case_id": str(archive["case_id"][0]),
                "video_path": str(archive["video_path"][0]),
                "video_size": int(archive["video_size"][0]),
                "video_mtime_ns": int(archive["video_mtime_ns"][0]),
                "tracking_sha256": str(archive["tracking_sha256"][0]),
                "nodes_sha256": str(archive["nodes_sha256"][0]),
            }
            wanted = {"cache_format_version": CACHE_FORMAT_VERSION, **expected}
            differences = [
                key for key, value in wanted.items() if actual.get(key) != value
            ]
            if differences:
                raise ValueError(
                    "Trajectory cache input mismatch: " + ", ".join(differences)
                )
        tracked = {
            "time_s": archive["time_s"],
            "positions": archive["positions"],
            "raw_positions": archive["raw_positions"],
            "corrected": archive["corrected"],
            "valid": archive["target_valid"],
            "raw_valid": archive["raw_target_valid"],
            "pose_valid": archive["pose_valid"],
            "epoch": archive["epoch"],
            "poses": archive["poses"],
            "target_ncc": archive["target_ncc"],
            "target_fb": archive["target_fb"],
            "point_brightness": archive["point_brightness"],
            "global_brightness": archive["global_brightness"],
            "heldout_raw": archive["heldout_raw"],
            "heldout_corrected": archive["heldout_corrected"],
            "training_inliers": archive["training_inliers"],
            "heldout_count": archive["heldout_count"],
        }
        return (
            tracked,
            archive["initial_points"],
            archive["anterior_axes"],
            archive["posterior_axes"],
        )


def subset_coverage(points: np.ndarray, chosen: tuple[int, ...]) -> float:
    """Euclidean span retained from the first-frame regional point set."""
    def diameter(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        delta = values[:, None, :] - values[None, :, :]
        return float(np.sqrt((delta * delta).sum(axis=2)).max())

    full = diameter(points)
    if full <= 1e-12:
        return 1.0
    return diameter(points[list(chosen)]) / full


def choose_fixed_point_segment(
    tracked: dict,
    initial: np.ndarray,
    columns: np.ndarray,
    min_point_fraction: float,
    min_span_fraction: float,
) -> dict | None:
    """Choose one unchanged point set and one continuous epoch-contained segment."""
    number = len(columns)
    minimum = max(2, int(math.ceil(number * min_point_fraction)))
    best: dict | None = None
    for size in range(minimum, number + 1):
        for local_choice in itertools.combinations(range(number), size):
            coverage = subset_coverage(initial[columns], local_choice)
            if coverage + 1e-12 < min_span_fraction:
                continue
            selected = columns[list(local_choice)]
            valid = (
                tracked["pose_valid"]
                & tracked["valid"][:, selected].all(axis=1)
                & np.isfinite(tracked["corrected"][:, selected]).all(axis=(1, 2))
            )
            for start, stop in base.contiguous_runs(valid, tracked["epoch"]):
                duration = float(
                    tracked["time_s"][stop - 1] - tracked["time_s"][start]
                )
                candidate = {
                    "start": start,
                    "stop": stop,
                    "columns": selected,
                    "local_choice": local_choice,
                    "duration_s": duration,
                    "point_count": size,
                    "point_fraction": size / number,
                    "span_fraction": coverage,
                }
                key = (duration, size, coverage, -start)
                if best is None or key > best["key"]:
                    candidate["key"] = key
                    best = candidate
    if best is not None:
        best.pop("key")
    return best


def periodic_trend_comparison(
    time_s: np.ndarray, signal: np.ndarray, frequency_cpm: float
) -> dict:
    """Compare the frozen periodic model with an explicit quadratic slow trend."""
    tt = np.asarray(time_s, float) - float(np.mean(time_s))
    y = np.asarray(signal, float)
    quadratic = np.column_stack((np.ones(len(tt)), tt, tt * tt))
    phase = 2.0 * np.pi * tt * frequency_cpm / 60.0
    periodic = np.column_stack(
        (np.ones(len(tt)), tt, np.sin(phase), np.cos(phase))
    )
    quadratic_residual = y - quadratic @ np.linalg.lstsq(quadratic, y, rcond=None)[0]
    periodic_residual = y - periodic @ np.linalg.lstsq(periodic, y, rcond=None)[0]
    quadratic_sse = float(quadratic_residual @ quadratic_residual)
    periodic_sse = float(periodic_residual @ periodic_residual)
    improvement = (
        (quadratic_sse - periodic_sse) / quadratic_sse
        if quadratic_sse > 1e-12 else math.nan
    )
    return {
        "quadratic_trend_sse": quadratic_sse,
        "periodic_model_sse": periodic_sse,
        "periodic_over_quadratic_improvement": improvement,
    }


def fit_candidate(time_s: np.ndarray, signal: np.ndarray) -> dict:
    fit = frozen.fit_unified_frequency(time_s, signal)
    if fit.get("status") != "valid":
        return fit
    comparison = periodic_trend_comparison(
        time_s, signal, float(fit["dominant_frequency_cpm"])
    )
    return {**fit, **comparison}


def usability(
    fit: dict,
    duration_s: float,
    min_duration_s: float,
    min_cycles: float,
    min_periodic_improvement: float,
) -> tuple[bool, str]:
    reasons: list[str] = []
    if fit.get("status") != "valid":
        reasons.append(str(fit.get("reason", "candidate_frequency_missing")))
    else:
        if duration_s + 1e-12 < min_duration_s:
            reasons.append("continuous_duration_below_rule")
        if float(fit.get("observed_cycles", 0.0)) + 1e-12 < min_cycles:
            reasons.append("candidate_cycle_coverage_below_rule")
        improvement = fit.get("periodic_over_quadratic_improvement", math.nan)
        if not np.isfinite(improvement) or improvement < min_periodic_improvement:
            reasons.append("periodic_model_not_better_than_quadratic_trend")
    return not reasons, "pass" if not reasons else ";".join(reasons)


def projected_signal(
    tracked: dict,
    initial: np.ndarray,
    axes: np.ndarray,
    columns: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    displacement = tracked["corrected"][start:stop, columns] - initial[columns]
    point_signal = np.einsum("tki,ki->tk", displacement, axes[columns])
    return point_signal.mean(axis=1)


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def fixed_frequency_component(
    time_s: np.ndarray, signal: np.ndarray, frequency_cpm: float
) -> dict:
    """Measure one nominated frequency by joint trend-plus-sinusoid regression."""
    time_s = np.asarray(time_s, float)
    signal = np.asarray(signal, float)
    valid = np.isfinite(time_s) & np.isfinite(signal)
    if valid.sum() < 20:
        return {"score": math.nan, "amplitude": math.nan}
    tt = time_s[valid] - float(np.mean(time_s[valid]))
    y = signal[valid]
    trend = np.column_stack((np.ones(len(tt)), tt))
    trend_fitted = trend @ np.linalg.lstsq(trend, y, rcond=None)[0]
    residual = y - trend_fitted
    energy = float(residual @ residual)
    if energy <= 1e-12:
        return {"score": math.nan, "amplitude": 0.0}
    phase = 2.0 * np.pi * tt * frequency_cpm / 60.0
    waves = np.column_stack((np.sin(phase), np.cos(phase)))
    full_design = np.column_stack((trend, waves))
    full_coefficients = np.linalg.lstsq(full_design, y, rcond=None)[0]
    coefficients = full_coefficients[-2:]
    periodic_valid = waves @ coefficients
    full_residual = y - full_design @ full_coefficients
    score = float(np.clip(1.0 - (full_residual @ full_residual) / energy, 0, 1))
    # Preserve original frame alignment.  Different control channels can have
    # different missing frames; compressing each independently makes a later
    # correlation compare the wrong frames or fail on unequal lengths.
    residual = np.full(len(signal), np.nan, float)
    periodic = np.full(len(signal), np.nan, float)
    residual[valid] = y - trend_fitted
    periodic[valid] = periodic_valid
    return {
        "score": score,
        "amplitude": float(np.linalg.norm(coefficients)),
        "residual": residual,
        "periodic": periodic,
        "valid": valid,
        "coefficient": complex(float(coefficients[0]), float(coefficients[1])),
    }


def background_signal(
    tracked: dict,
    initial: np.ndarray,
    axes: np.ndarray,
    columns: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    values = np.full(stop - start, np.nan, float)
    for output_index, frame in enumerate(range(start, stop)):
        if not tracked["pose_valid"][frame]:
            continue
        inverse = np.linalg.inv(tracked["poses"][frame])
        mapped = initial[columns] @ inverse[:2, :2].T + inverse[:2, 2]
        displacement = mapped - initial[columns]
        values[output_index] = np.einsum(
            "ki,ki->k", displacement, axes[columns]
        ).mean()
    return values


def artifact_evidence(
    tracked: dict,
    initial: np.ndarray,
    axes: np.ndarray,
    selected: dict | None,
    fit: dict,
    usable: bool,
    deformation_supported: bool,
) -> dict:
    """Describe signal origin using controls from the identical point set and segment."""
    if selected is None or fit.get("status") != "valid":
        return {
            "evidence_class": "insufficient_evidence",
            "evidence_reason": "optimized_frequency_or_segment_missing",
        }
    start, stop = selected["start"], selected["stop"]
    columns = selected["columns"]
    time_s = tracked["time_s"][start:stop]
    frequency = float(fit["dominant_frequency_cpm"])
    displacement = tracked["corrected"][start:stop, columns] - initial[columns]
    point_signals = np.einsum("tki,ki->tk", displacement, axes[columns])
    tissue = point_signals.mean(axis=1)
    background = background_signal(
        tracked, initial, axes, columns, start, stop
    )
    local_brightness = np.nanmean(
        tracked["point_brightness"][start:stop, columns], axis=1
    )
    global_brightness = tracked["global_brightness"][start:stop]

    tissue_frequency = fixed_frequency_component(time_s, tissue, frequency)
    background_frequency = fixed_frequency_component(time_s, background, frequency)
    local_brightness_frequency = fixed_frequency_component(
        time_s, local_brightness, frequency
    )
    global_brightness_frequency = fixed_frequency_component(
        time_s, global_brightness, frequency
    )

    def periodic_corr(control: dict) -> float:
        if "periodic" not in tissue_frequency or "periodic" not in control:
            return math.nan
        return safe_correlation(tissue_frequency["periodic"], control["periodic"])

    point_components = [
        fixed_frequency_component(time_s, point_signals[:, index], frequency)
        for index in range(point_signals.shape[1])
    ]
    # Keep participation and phase agreement as separate quantities.  The
    # normalized inverse-participation ratio is 1/n when one of n points owns
    # all same-frequency amplitude, and 1 when all points contribute equally.
    coefficients = np.asarray([
        item.get("coefficient", complex(0.0, 0.0))
        if np.isfinite(abs(item.get("coefficient", complex(math.nan, math.nan))))
        else complex(0.0, 0.0)
        for item in point_components
    ], complex)
    amplitudes = np.abs(coefficients)
    amplitude_sum = float(amplitudes.sum())
    amplitude_square_sum = float(amplitudes @ amplitudes)
    point_participation_fraction = (
        amplitude_sum ** 2 / (len(point_components) * amplitude_square_sum)
        if amplitude_square_sum > 1e-24 else 0.0
    )
    point_phase_coherence = (
        float(abs(coefficients.sum()) / amplitude_sum)
        if amplitude_sum > 1e-12 else math.nan
    )
    tissue_range = float(np.ptp(tissue))
    background_range = float(np.ptp(background[np.isfinite(background)])) if np.isfinite(background).any() else math.nan
    background_ratio = (
        background_range / tissue_range if tissue_range > 1e-12 else math.nan
    )
    ncc = tracked["target_ncc"][start:stop, columns]
    fb = tracked["target_fb"][start:stop, columns]
    mean_ncc = float(np.nanmean(ncc)) if np.isfinite(ncc).any() else math.nan
    p90_fb = float(np.nanpercentile(fb, 90)) if np.isfinite(fb).any() else math.nan
    heldout = tracked["heldout_corrected"][start:stop]
    mean_heldout = (
        float(np.nanmean(heldout)) if np.isfinite(heldout).any() else math.nan
    )

    background_periodic_corr = periodic_corr(background_frequency)
    local_brightness_periodic_corr = periodic_corr(local_brightness_frequency)
    global_brightness_periodic_corr = periodic_corr(global_brightness_frequency)
    thresholds = EVIDENCE_THRESHOLDS
    background_flag = bool(
        np.isfinite(background_ratio)
        and background_ratio >= thresholds["background_to_tissue_range_ratio"]
        and background_frequency.get("score", math.nan)
        >= thresholds["control_fixed_frequency_score"]
        and abs(background_periodic_corr)
        >= thresholds["control_periodic_correlation"]
    )
    brightness_flag = bool(
        global_brightness_frequency.get("score", math.nan)
        >= thresholds["control_fixed_frequency_score"]
        and abs(global_brightness_periodic_corr)
        >= thresholds["control_periodic_correlation"]
    )
    tracking_flag = bool(
        point_participation_fraction < thresholds["minimum_point_participation_fraction"]
        or (np.isfinite(point_phase_coherence)
         and point_phase_coherence < thresholds["minimum_point_phase_coherence"])
        or (np.isfinite(mean_ncc) and mean_ncc < thresholds["minimum_mean_ncc"])
        or (np.isfinite(p90_fb)
            and p90_fb > thresholds["maximum_p90_forward_backward_px"])
    )
    artifact_reasons = []
    if tracking_flag:
        artifact_reasons.append("tracking_or_point_inconsistency")
    if artifact_reasons:
        evidence_class = "artifact_risk"
        evidence_reason = ";".join(artifact_reasons)
    elif background_flag and not deformation_supported:
        # A small periodic registration residual can remain after otherwise
        # successful rigid correction.  Without independent paired-wall
        # deformation, same-frequency background and local translation are not
        # identifiable, so do not retain the value as medium-confidence motion.
        evidence_class = "insufficient_evidence"
        evidence_reason = "background_same_frequency_without_deformation_support"
    elif (
        usable
        and point_participation_fraction >= thresholds["minimum_point_participation_fraction"]
        and np.isfinite(point_phase_coherence)
        and point_phase_coherence >= thresholds["supported_point_phase_coherence"]
    ):
        if deformation_supported:
            evidence_class = "local_tissue_motion_supported"
            evidence_reason = "usable_frequency;coherent_points;paired_deformation_support"
        else:
            evidence_class = "coordinated_tissue_displacement_supported"
            evidence_reason = "usable_frequency;participating_coherent_points;deformation_type_unresolved"
    else:
        evidence_class = "insufficient_evidence"
        evidence_reason = "local_deformation_support_incomplete"
    confounding_reasons = []
    if background_flag:
        confounding_reasons.append("background_same_frequency")
    if brightness_flag:
        confounding_reasons.append("global_brightness_same_frequency")
    return {
        "evidence_class": evidence_class,
        "evidence_reason": evidence_reason,
        "background_same_frequency_flag": background_flag,
        "brightness_same_frequency_flag": brightness_flag,
        "tracking_artifact_flag": tracking_flag,
        "same_frequency_confounding_flag": bool(confounding_reasons),
        "same_frequency_confounding_reason": ";".join(confounding_reasons),
        "tissue_fixed_frequency_score": tissue_frequency.get("score", ""),
        "background_fixed_frequency_score": background_frequency.get("score", ""),
        "background_periodic_correlation": background_periodic_corr,
        "background_to_tissue_range_ratio_same_segment": background_ratio,
        "local_brightness_fixed_frequency_score": local_brightness_frequency.get("score", ""),
        "local_brightness_periodic_correlation": local_brightness_periodic_corr,
        "global_brightness_fixed_frequency_score": global_brightness_frequency.get("score", ""),
        "global_brightness_periodic_correlation": global_brightness_periodic_corr,
        "point_phase_coherence": point_phase_coherence,
        "point_participation_fraction": point_participation_fraction,
        "mean_target_ncc_same_segment": mean_ncc,
        "p90_target_fb_px_same_segment": p90_fb,
        "mean_background_heldout_residual_px_same_segment": mean_heldout,
    }


def paired_deformation_support(
    tracked: dict,
    initial: np.ndarray,
    axes: np.ndarray,
    local_indices: np.ndarray,
    count: int,
    wall: str,
    selected: dict | None,
) -> dict:
    """Use the same selected anatomical pairs and the same continuous segment."""
    if selected is None:
        return {
            "status": "missing",
            "reason": "optimized_fixed_point_segment_missing",
        }
    chosen_local = local_indices[list(selected["local_choice"])]
    anterior = chosen_local
    posterior = chosen_local + count
    current = anterior if wall == "anterior" else posterior
    common_valid = (
        tracked["pose_valid"]
        & tracked["valid"][:, anterior].all(axis=1)
        & tracked["valid"][:, posterior].all(axis=1)
        & np.isfinite(tracked["corrected"][:, anterior]).all(axis=(1, 2))
        & np.isfinite(tracked["corrected"][:, posterior]).all(axis=(1, 2))
    )
    runs = base.contiguous_runs(common_valid, tracked["epoch"])
    if not runs:
        return {"status": "missing", "reason": "no_paired_common_valid_segment"}
    start, stop = max(
        runs,
        key=lambda run: tracked["time_s"][run[1] - 1] - tracked["time_s"][run[0]],
    )
    time_s = tracked["time_s"][start:stop]
    displacement = projected_signal(
        tracked, initial, axes, current, start, stop
    )
    initial_distance = np.linalg.norm(
        initial[posterior] - initial[anterior], axis=1
    )
    current_distance = np.linalg.norm(
        tracked["corrected"][start:stop, posterior]
        - tracked["corrected"][start:stop, anterior],
        axis=2,
    )
    strain = (current_distance / initial_distance[None] - 1.0).mean(axis=1)
    displacement_fit = fit_candidate(time_s, displacement)
    strain_fit = fit_candidate(time_s, strain)
    strain_range = float(np.ptp(strain))
    difference = (
        abs(
            float(displacement_fit["dominant_frequency_cpm"])
            - float(strain_fit["dominant_frequency_cpm"])
        )
        if displacement_fit.get("status") == strain_fit.get("status") == "valid"
        else math.nan
    )
    supported = bool(
        displacement_fit.get("status") == strain_fit.get("status") == "valid"
        and strain_range >= frozen.STRAIN_RANGE_THRESHOLD
        and difference <= frozen.FREQUENCY_TOLERANCE_CPM
    )
    reasons: list[str] = []
    if displacement_fit.get("status") != "valid":
        reasons.append("support_displacement_frequency_missing")
    if strain_fit.get("status") != "valid":
        reasons.append("paired_strain_frequency_missing")
    if strain_range < frozen.STRAIN_RANGE_THRESHOLD:
        reasons.append("strain_range_below_frozen_threshold")
    if np.isfinite(difference) and difference > frozen.FREQUENCY_TOLERANCE_CPM:
        reasons.append("frequency_difference_above_frozen_tolerance")
    return {
        "status": "valid",
        "reason": "pass" if supported else ";".join(reasons),
        "supported": supported,
        "duration_s": float(time_s[-1] - time_s[0]),
        "displacement_frequency_cpm": displacement_fit.get(
            "dominant_frequency_cpm", ""
        ),
        "strain_frequency_cpm": strain_fit.get("dominant_frequency_cpm", ""),
        "strain_range": strain_range,
        "frequency_difference_cpm": difference if np.isfinite(difference) else "",
        "paired_local_point_indices": ";".join(str(i) for i in chosen_local),
    }


def analyse_cache(
    case_id: str,
    cache_path: Path,
    args: argparse.Namespace,
    cache_identity: dict,
) -> list[dict]:
    tracked, initial, anterior_axes, posterior_axes = load_cache(
        cache_path, cache_identity
    )
    axes = np.vstack((anterior_axes, posterior_axes))
    count = len(anterior_axes)
    regions = dict(zip(base.frozen.REGION_NAMES, np.array_split(np.arange(count), 3)))
    rows: list[dict] = []
    for wall, offset in (("anterior", 0), ("posterior", count)):
        for region, local_indices in regions.items():
            columns = local_indices + offset
            arrays = base.region_arrays(tracked, initial, axes, local_indices, columns)
            original_valid = arrays["region_valid"] & np.isfinite(arrays["corrected_mean"])
            original_runs = base.contiguous_runs(original_valid, tracked["epoch"])
            original_segment = None
            if original_runs:
                start, stop = max(
                    original_runs,
                    key=lambda run: tracked["time_s"][run[1] - 1]
                    - tracked["time_s"][run[0]],
                )
                original_duration = float(
                    tracked["time_s"][stop - 1] - tracked["time_s"][start]
                )
                original_fit = fit_candidate(
                    tracked["time_s"][start:stop], arrays["corrected_mean"][start:stop]
                )
                original_segment = (start, stop)
            else:
                original_duration = 0.0
                original_fit = {
                    "status": "missing",
                    "reason": "no_continuous_technical_valid_segment",
                }

            selected = choose_fixed_point_segment(
                tracked,
                initial,
                columns,
                args.min_point_fraction,
                args.min_span_fraction,
            )
            if selected is None:
                optimized_duration = 0.0
                optimized_fit = {
                    "status": "missing",
                    "reason": "no_fixed_point_segment_with_required_coverage",
                }
            else:
                start, stop = selected["start"], selected["stop"]
                optimized_duration = selected["duration_s"]
                optimized_signal = projected_signal(
                    tracked,
                    initial,
                    axes,
                    selected["columns"],
                    start,
                    stop,
                )
                optimized_fit = fit_candidate(
                    tracked["time_s"][start:stop], optimized_signal
                )

            usable, reason = usability(
                optimized_fit,
                optimized_duration,
                args.min_duration_s,
                args.min_cycles,
                args.min_periodic_improvement,
            )
            deformation = paired_deformation_support(
                tracked,
                initial,
                axes,
                local_indices,
                count,
                wall,
                selected,
            )
            evidence = artifact_evidence(
                tracked,
                initial,
                axes,
                selected,
                optimized_fit,
                usable,
                bool(deformation.get("supported", False)),
            )
            row = {
                "case_id": case_id,
                "wall": wall,
                "region": region,
                "actual_video_duration_s": float(tracked["time_s"][-1]),
                "original_candidate_status": original_fit.get("status", "missing"),
                "original_candidate_reason": original_fit.get("reason", ""),
                "original_candidate_frequency_cpm": original_fit.get(
                    "dominant_frequency_cpm", ""
                ),
                "original_segment_duration_s": original_duration,
                "original_observed_cycles": original_fit.get("observed_cycles", ""),
                "original_boundary_hit": original_fit.get("search_boundary", ""),
                "optimized_candidate_status": optimized_fit.get("status", "missing"),
                "optimized_candidate_reason": optimized_fit.get("reason", ""),
                "optimized_candidate_frequency_cpm": optimized_fit.get(
                    "dominant_frequency_cpm", ""
                ),
                "optimized_segment_duration_s": optimized_duration,
                "optimized_observed_cycles": optimized_fit.get("observed_cycles", ""),
                "optimized_boundary_hit": optimized_fit.get("search_boundary", ""),
                "optimized_score": optimized_fit.get("score", ""),
                "quadratic_trend_sse": optimized_fit.get("quadratic_trend_sse", ""),
                "periodic_model_sse": optimized_fit.get("periodic_model_sse", ""),
                "periodic_over_quadratic_improvement": optimized_fit.get(
                    "periodic_over_quadratic_improvement", ""
                ),
                "frequency_usable": usable,
                "frequency_reason": reason,
                "optimized_deformation_support_evaluable": (
                    deformation.get("status") == "valid"
                ),
                "optimized_deformation_supported": deformation.get("supported", False),
                "optimized_deformation_support_reason": deformation.get("reason", ""),
                "optimized_support_common_duration_s": deformation.get("duration_s", ""),
                "optimized_support_displacement_frequency_cpm": deformation.get(
                    "displacement_frequency_cpm", ""
                ),
                "optimized_paired_strain_frequency_cpm": deformation.get(
                    "strain_frequency_cpm", ""
                ),
                "optimized_paired_strain_range": deformation.get("strain_range", ""),
                "optimized_displacement_strain_frequency_difference_cpm": (
                    deformation.get("frequency_difference_cpm", "")
                ),
                **evidence,
                "technical_confidence": (
                    "high"
                    if evidence.get("evidence_class") == "local_tissue_motion_supported"
                    else "medium"
                    if evidence.get("evidence_class") == "coordinated_tissue_displacement_supported"
                    else "excluded"
                ),
                "retained_for_patient_model": evidence.get("evidence_class") in {
                    "local_tissue_motion_supported",
                    "coordinated_tissue_displacement_supported",
                },
                "selected_point_count": selected["point_count"] if selected else 0,
                "original_region_point_count": len(columns),
                "selected_point_fraction": selected["point_fraction"] if selected else "",
                "selected_span_fraction": selected["span_fraction"] if selected else "",
                "selected_local_point_indices": (
                    ";".join(str(i) for i in selected["local_choice"])
                    if selected else ""
                ),
                "selected_start_frame": selected["start"] if selected else "",
                "selected_stop_frame_exclusive": selected["stop"] if selected else "",
                "original_start_frame": original_segment[0] if original_segment else "",
                "original_stop_frame_exclusive": original_segment[1] if original_segment else "",
            }
            rows.append(row)
    return rows


def patient_rows(case_ids: list[str], regions: list[dict], status: dict[str, dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in regions:
        grouped[row["case_id"]].append(row)
    output: list[dict] = []
    for case_id in case_ids:
        own = grouped[case_id]
        original = [row for row in own if row["original_candidate_status"] == "valid"]
        optimized = [row for row in own if row["frequency_usable"]]
        retained = [row for row in own if row.get("retained_for_patient_model")]
        durations = [
            float(row["optimized_segment_duration_s"])
            for row in own
            if row["frequency_usable"]
        ]
        output.append({
            "case_id": case_id,
            "processing_status": status[case_id]["status"],
            "processing_reason": status[case_id]["reason"],
            "original_candidate_region_count": len(original),
            "optimized_usable_region_count": len(optimized),
            "retained_region_count": len(retained),
            "retained_anterior_region_count": sum(
                row["wall"] == "anterior" for row in retained),
            "retained_posterior_region_count": sum(
                row["wall"] == "posterior" for row in retained),
            "optimized_usable_duration_median_s": (
                statistics.median(durations) if durations else ""
            ),
            "optimized_boundary_region_count": sum(
                bool(row["optimized_boundary_hit"])
                for row in own if row["frequency_usable"]
            ),
            "optimized_support_insufficient_region_count": sum(
                not bool(row["frequency_usable"]) for row in own
            ),
            "optimized_deformation_supported_region_count": sum(
                bool(row["optimized_deformation_supported"]) for row in own
            ),
            "local_tissue_motion_supported_region_count": sum(
                row.get("evidence_class") == "local_tissue_motion_supported"
                for row in own
            ),
            "coordinated_tissue_displacement_supported_region_count": sum(
                row.get("evidence_class") == "coordinated_tissue_displacement_supported"
                for row in own
            ),
            "artifact_risk_region_count": sum(
                row.get("evidence_class") == "artifact_risk" for row in own
            ),
            "insufficient_evidence_region_count": sum(
                row.get("evidence_class") == "insufficient_evidence" for row in own
            ),
        })
    return output


def sensitivity_rows(regions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    case_count = len({row["case_id"] for row in regions})
    for duration, cycles, improvement in itertools.product(
        DURATION_CANDIDATES_S,
        CYCLE_CANDIDATES,
        TREND_IMPROVEMENT_CANDIDATES,
    ):
        usable = []
        by_case: dict[str, int] = defaultdict(int)
        for row in regions:
            fit = {
                "status": row["optimized_candidate_status"],
                "reason": row["optimized_candidate_reason"],
                "observed_cycles": row["optimized_observed_cycles"],
                "periodic_over_quadratic_improvement": row[
                    "periodic_over_quadratic_improvement"
                ],
            }
            ok, _ = usability(
                fit,
                float(row["optimized_segment_duration_s"]),
                duration,
                cycles,
                improvement,
            )
            if ok:
                usable.append(row)
                by_case[row["case_id"]] += 1
        rows.append({
            "min_duration_s": duration,
            "min_candidate_cycles": cycles,
            "min_periodic_over_quadratic_improvement": improvement,
            "usable_region_count": len(usable),
            "patients_with_at_least_1_usable_region": sum(v >= 1 for v in by_case.values()),
            "patients_with_at_least_2_usable_regions": sum(v >= 2 for v in by_case.values()),
            "patients_total": case_count,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen V1 outcome-blind R2 local-tissue displacement frequency"
    )
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--min-duration-s", type=float, default=15.0)
    parser.add_argument("--min-cycles", type=float, default=1.0)
    parser.add_argument("--min-periodic-improvement", type=float, default=0.05)
    parser.add_argument("--min-point-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--min-span-fraction", type=float, default=2.0 / 3.0)
    args = parser.parse_args()
    for name in ("min_point_fraction", "min_span_fraction"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.min_duration_s < 0 or args.min_cycles < 0:
        raise ValueError("Duration and cycle thresholds must be non-negative")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)
    cache_dir = args.cache_dir or (args.out / "trajectory_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = frozen.read_case_manifest(args.case_manifest)
    case_ids = [row["case_id"] for row in manifest]
    statuses: dict[str, dict] = {}
    regions: list[dict] = []
    started = time.time()
    for source in manifest:
        case_id = source["case_id"]
        cache_path = cache_dir / f"{case_id}.npz"
        try:
            prepared = frozen.prepare_case(source)
            video_stat = source["video"].stat()
            cache_identity = {
                "case_id": case_id,
                "video_path": str(source["video"].resolve()),
                "video_size": int(video_stat.st_size),
                "video_mtime_ns": int(video_stat.st_mtime_ns),
                "tracking_sha256": sha256(source["tracking"]),
                "nodes_sha256": sha256(source["nodes"]),
            }
            if not (args.reuse_cache and cache_path.is_file()):
                if cache_path.exists():
                    raise FileExistsError(
                        f"Cache exists; use --reuse-cache or a new cache directory: {cache_path}"
                    )
                tracked, initial, anterior_axes, posterior_axes = base.track_real_case(
                    prepared, float(prepared["available_s"])
                )
                save_cache(
                    cache_path,
                    tracked,
                    initial,
                    anterior_axes,
                    posterior_axes,
                    cache_identity,
                )
            own = analyse_cache(case_id, cache_path, args, cache_identity)
            regions.extend(own)
            statuses[case_id] = {"status": "complete", "reason": ""}
            print(case_id, "complete", flush=True)
        except Exception as exc:
            statuses[case_id] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}:{exc}",
            }
            print(case_id, "failed", type(exc).__name__, str(exc), flush=True)

    patients = patient_rows(case_ids, regions, statuses)
    sensitivity = sensitivity_rows(regions)
    status_rows = [
        {"case_id": case_id, **statuses[case_id]} for case_id in case_ids
    ]
    write_csv(args.out / "optimized_region_frequency.csv", regions, ["case_id"])
    write_csv(args.out / "optimized_patient_frequency.csv", patients, ["case_id"])
    evidence_fields = [
        "case_id", "wall", "region", "optimized_candidate_frequency_cpm",
        "frequency_usable", "evidence_class", "evidence_reason",
        "technical_confidence", "retained_for_patient_model",
        "background_same_frequency_flag", "brightness_same_frequency_flag",
        "tracking_artifact_flag", "same_frequency_confounding_flag",
        "same_frequency_confounding_reason", "background_fixed_frequency_score",
        "background_periodic_correlation",
        "background_to_tissue_range_ratio_same_segment",
        "global_brightness_fixed_frequency_score",
        "global_brightness_periodic_correlation", "point_phase_coherence",
        "point_participation_fraction",
        "optimized_deformation_supported", "optimized_paired_strain_range",
        "optimized_displacement_strain_frequency_difference_cpm",
        "mean_target_ncc_same_segment", "p90_target_fb_px_same_segment",
        "mean_background_heldout_residual_px_same_segment",
    ]
    evidence_rows = [
        {field: row.get(field, "") for field in evidence_fields}
        for row in regions
    ]
    write_csv(
        args.out / "signal_origin_evidence.csv", evidence_rows, evidence_fields
    )
    write_csv(args.out / "usability_sensitivity_matrix.csv", sensitivity, ["min_duration_s"])
    write_csv(args.out / "case_status.csv", status_rows, ["case_id", "status", "reason"])

    configuration = {
        "analysis": "frozen V1 outcome-blind local-tissue displacement frequency",
        "frozen_rule_version": "R2_TISSUE_FREQUENCY_V1_20261105",
        "frozen_pipeline_modified": False,
        "clinical_outcome_read": False,
        "input_manifest": str(args.case_manifest.resolve()),
        "input_manifest_sha256": sha256(args.case_manifest),
        "optimizer_sha256": sha256(Path(__file__)),
        "frozen_tracker_sha256": sha256(CODE / "run_real_context_evidence.py"),
        "frozen_frequency_runner_sha256": sha256(
            CODE / "run_local_motion_frequency.py"
        ),
        "trajectory_cache": str(cache_dir.resolve()),
        "primary_candidate_rule": {
            "min_duration_s": args.min_duration_s,
            "min_candidate_cycles": args.min_cycles,
            "min_periodic_over_quadratic_improvement": args.min_periodic_improvement,
            "note": "frozen after prespecified mathematical, image-chain, and seed-stability validation",
        },
        "fixed_point_segment_rule": {
            "min_point_fraction": args.min_point_fraction,
            "minimum_points": 2,
            "min_first_frame_span_fraction": args.min_span_fraction,
            "point_set_constant_within_segment": True,
            "selection_order": "longest duration, then point count, then spatial span",
            "selection_uses_frequency_or_outcome": False,
        },
        "signal_origin_evidence": {
            "classes": [
                "local_tissue_motion_supported",
                "coordinated_tissue_displacement_supported",
                "artifact_risk",
                "insufficient_evidence",
            ],
            "thresholds": EVIDENCE_THRESHOLDS,
            "same_segment_and_point_set": True,
            "uses_clinical_outcome": False,
        },
        "boundary_frequency_excluded": False,
        "missing_frequency_filled_with_zero": False,
        "patient_summary": "counts only; regional frequencies are not aggregated",
        "sensitivity_duration_candidates_s": DURATION_CANDIDATES_S,
        "sensitivity_cycle_candidates": CYCLE_CANDIDATES,
        "sensitivity_trend_improvement_candidates": TREND_IMPROVEMENT_CANDIDATES,
        "elapsed_seconds": time.time() - started,
        "cases_requested": len(case_ids),
        "cases_complete": sum(v["status"] == "complete" for v in statuses.values()),
    }
    (args.out / "optimization_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evidence_summary = {
        "regions_total": len(regions),
        "local_tissue_motion_supported": sum(
            row.get("evidence_class") == "local_tissue_motion_supported"
            for row in regions
        ),
        "coordinated_tissue_displacement_supported": sum(
            row.get("evidence_class") == "coordinated_tissue_displacement_supported"
            for row in regions
        ),
        "artifact_risk": sum(
            row.get("evidence_class") == "artifact_risk" for row in regions
        ),
        "insufficient_evidence": sum(
            row.get("evidence_class") == "insufficient_evidence"
            for row in regions
        ),
        "background_same_frequency_flags": sum(
            bool(row.get("background_same_frequency_flag")) for row in regions
        ),
        "brightness_same_frequency_flags": sum(
            bool(row.get("brightness_same_frequency_flag")) for row in regions
        ),
        "tracking_inconsistency_flags": sum(
            bool(row.get("tracking_artifact_flag")) for row in regions
        ),
        "interpretation": "technical signal-origin evidence; not physiological ground truth",
    }
    (args.out / "signal_origin_summary.json").write_text(
        json.dumps(evidence_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if all(v["status"] == "complete" for v in statuses.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
