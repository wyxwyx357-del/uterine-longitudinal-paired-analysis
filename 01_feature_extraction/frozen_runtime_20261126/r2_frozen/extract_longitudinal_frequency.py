#!/usr/bin/env python3
"""Outcome-blind longitudinal (cervix-to-fundus) tissue-motion frequency.

Reuses background-corrected trajectory caches produced by the tracking stage and the
already selected fixed point sets/continuous segments.  The only primary
signal change is the projection axis: local wall tangent instead of the
paired anterior-posterior inward axis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "r2_longitudinal_matplotlib")
)
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION_NAMES = ("cervical", "middle", "fundal")
WALLS = ("anterior", "posterior")
RULE_VERSION = "R2_LONGITUDINAL_TISSUE_FREQUENCY_V1_1_20261106"
GRID_CPM = np.round(np.arange(0.5, 5.0001, 0.05), 8)
EVIDENCE_THRESHOLDS = {
    "control_fixed_frequency_score": 0.50,
    "control_periodic_correlation": 0.70,
    "background_to_tissue_range_ratio": 0.50,
    "supported_point_phase_coherence": 0.70,
    "minimum_point_participation_fraction": 2.0 / 3.0,
    "minimum_mean_ncc": 0.80,
    "maximum_p90_forward_backward_px": 0.60,
}


def safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 20 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return math.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def fit_unified_frequency(time_s: np.ndarray, signal: np.ndarray) -> dict:
    if len(time_s) < 20:
        return {"status": "missing", "reason": "technical_segment_too_short"}
    tt = np.asarray(time_s, float) - float(np.mean(time_s))
    signal = np.asarray(signal, float)
    trend_x = np.column_stack((np.ones(len(tt)), tt))
    trend = trend_x @ np.linalg.lstsq(trend_x, signal, rcond=None)[0]
    residual = signal - trend
    energy = float(residual @ residual)
    if energy / len(signal) < 1e-12:
        return {"status": "missing", "reason": "technical_nearly_constant"}
    scores = []
    for frequency in GRID_CPM:
        phase = 2.0 * np.pi * tt * frequency / 60.0
        design = np.column_stack((trend_x, np.sin(phase), np.cos(phase)))
        fitted = design @ np.linalg.lstsq(design, signal, rcond=None)[0]
        scores.append(np.clip(1.0 - float(((signal - fitted) ** 2).sum()) / energy, 0.0, 1.0))
    scores = np.asarray(scores)
    index = int(np.argmax(scores))
    frequency = float(GRID_CPM[index])
    return {
        "status": "valid",
        "reason": "unified_fixed_band_dominant_frequency",
        "dominant_frequency_cpm": frequency,
        "period_s": 60.0 / frequency,
        "score": float(scores[index]),
        "search_boundary": bool(index in (0, len(GRID_CPM) - 1)),
        "observed_cycles": float(frequency * (time_s[-1] - time_s[0]) / 60.0),
    }


def periodic_trend_comparison(time_s: np.ndarray, signal: np.ndarray, frequency: float) -> dict:
    tt = np.asarray(time_s, float) - float(np.mean(time_s))
    y = np.asarray(signal, float)
    quadratic = np.column_stack((np.ones(len(tt)), tt, tt * tt))
    phase = 2.0 * np.pi * tt * frequency / 60.0
    periodic = np.column_stack((np.ones(len(tt)), tt, np.sin(phase), np.cos(phase)))
    q_residual = y - quadratic @ np.linalg.lstsq(quadratic, y, rcond=None)[0]
    p_residual = y - periodic @ np.linalg.lstsq(periodic, y, rcond=None)[0]
    q_sse = float(q_residual @ q_residual)
    p_sse = float(p_residual @ p_residual)
    return {
        "quadratic_trend_sse": q_sse,
        "periodic_model_sse": p_sse,
        "periodic_over_quadratic_improvement": (
            (q_sse - p_sse) / q_sse if q_sse > 1e-12 else math.nan
        ),
    }


def fit_candidate(time_s: np.ndarray, signal: np.ndarray) -> dict:
    fit = fit_unified_frequency(time_s, signal)
    if fit.get("status") != "valid":
        return fit
    return {
        **fit,
        **periodic_trend_comparison(time_s, signal, float(fit["dominant_frequency_cpm"])),
    }


def usability(fit: dict, duration_s: float) -> tuple[bool, str]:
    reasons = []
    if fit.get("status") != "valid":
        reasons.append(str(fit.get("reason", "candidate_frequency_missing")))
    else:
        if duration_s + 1e-12 < 15.0:
            reasons.append("continuous_duration_below_rule")
        if float(fit.get("observed_cycles", 0.0)) + 1e-12 < 1.0:
            reasons.append("candidate_cycle_coverage_below_rule")
        improvement = fit.get("periodic_over_quadratic_improvement", math.nan)
        if not np.isfinite(improvement) or improvement < 0.05:
            reasons.append("periodic_model_not_better_than_quadratic_trend")
    return not reasons, "pass" if not reasons else ";".join(reasons)


def fixed_frequency_component(time_s: np.ndarray, signal: np.ndarray, frequency: float) -> dict:
    time_s = np.asarray(time_s, float)
    signal = np.asarray(signal, float)
    valid = np.isfinite(time_s) & np.isfinite(signal)
    if valid.sum() < 20:
        return {"score": math.nan, "amplitude": math.nan}
    tt = time_s[valid] - float(np.mean(time_s[valid]))
    y = signal[valid]
    trend = np.column_stack((np.ones(len(tt)), tt))
    trend_fitted = trend @ np.linalg.lstsq(trend, y, rcond=None)[0]
    residual_valid = y - trend_fitted
    energy = float(residual_valid @ residual_valid)
    if energy <= 1e-12:
        return {"score": math.nan, "amplitude": 0.0}
    phase = 2.0 * np.pi * tt * frequency / 60.0
    waves = np.column_stack((np.sin(phase), np.cos(phase)))
    design = np.column_stack((trend, waves))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    periodic_valid = waves @ coefficients[-2:]
    full_residual = y - design @ coefficients
    residual = np.full(len(signal), np.nan)
    periodic = np.full(len(signal), np.nan)
    residual[valid] = residual_valid
    periodic[valid] = periodic_valid
    return {
        "score": float(np.clip(1.0 - (full_residual @ full_residual) / energy, 0, 1)),
        "amplitude": float(np.linalg.norm(coefficients[-2:])),
        "residual": residual,
        "periodic": periodic,
        "coefficient": complex(float(coefficients[-2]), float(coefficients[-1])),
    }


def load_cache(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        tracked = {
            "time_s": archive["time_s"],
            "corrected": archive["corrected"],
            "valid": archive["target_valid"],
            "pose_valid": archive["pose_valid"],
            "poses": archive["poses"],
            "target_ncc": archive["target_ncc"],
            "target_fb": archive["target_fb"],
            "point_brightness": archive["point_brightness"],
            "global_brightness": archive["global_brightness"],
            "heldout_corrected": archive["heldout_corrected"],
        }
        return tracked, archive["initial_points"], archive["anterior_axes"], archive["posterior_axes"]


def background_signal(
    tracked: dict, initial: np.ndarray, axes: np.ndarray,
    columns: np.ndarray, start: int, stop: int,
) -> np.ndarray:
    values = np.full(stop - start, np.nan)
    for out_index, frame in enumerate(range(start, stop)):
        if not tracked["pose_valid"][frame]:
            continue
        inverse = np.linalg.inv(tracked["poses"][frame])
        mapped = initial[columns] @ inverse[:2, :2].T + inverse[:2, 2]
        values[out_index] = np.einsum(
            "ki,ki->k", mapped - initial[columns], axes[columns]
        ).mean()
    return values


def artifact_evidence(
    tracked: dict, initial: np.ndarray, axes: np.ndarray,
    selected: dict, fit: dict, usable: bool,
) -> dict:
    start, stop = selected["start"], selected["stop"]
    columns = selected["columns"]
    time_s = tracked["time_s"][start:stop]
    frequency = finite_float(fit.get("dominant_frequency_cpm"))
    displacement = tracked["corrected"][start:stop, columns] - initial[columns]
    point_signals = np.einsum("tki,ki->tk", displacement, axes[columns])
    tissue = point_signals.mean(axis=1)
    background = background_signal(tracked, initial, axes, columns, start, stop)
    local_brightness = np.nanmean(tracked["point_brightness"][start:stop, columns], axis=1)
    global_brightness = tracked["global_brightness"][start:stop]
    tissue_component = fixed_frequency_component(time_s, tissue, frequency)
    background_component = fixed_frequency_component(time_s, background, frequency)
    local_component = fixed_frequency_component(time_s, local_brightness, frequency)
    global_component = fixed_frequency_component(time_s, global_brightness, frequency)
    point_components = [
        fixed_frequency_component(time_s, point_signals[:, index], frequency)
        for index in range(point_signals.shape[1])
    ]
    coefficients = np.asarray([
        item.get("coefficient", 0j)
        if np.isfinite(abs(item.get("coefficient", complex(math.nan, math.nan)))) else 0j
        for item in point_components
    ])
    amplitudes = np.abs(coefficients)
    amp_sum = float(amplitudes.sum())
    amp_sq_sum = float(amplitudes @ amplitudes)
    participation = amp_sum ** 2 / (len(amplitudes) * amp_sq_sum) if amp_sq_sum > 1e-24 else 0.0
    coherence = float(abs(coefficients.sum()) / amp_sum) if amp_sum > 1e-12 else math.nan

    def periodic_corr(control: dict) -> float:
        if "periodic" not in tissue_component or "periodic" not in control:
            return math.nan
        return safe_correlation(tissue_component["periodic"], control["periodic"])

    background_corr = periodic_corr(background_component)
    local_corr = periodic_corr(local_component)
    global_corr = periodic_corr(global_component)
    tissue_range = float(np.ptp(tissue))
    finite_background = background[np.isfinite(background)]
    background_range = float(np.ptp(finite_background)) if len(finite_background) else math.nan
    background_ratio = background_range / tissue_range if tissue_range > 1e-12 else math.nan
    threshold = EVIDENCE_THRESHOLDS
    background_flag = bool(
        np.isfinite(background_ratio)
        and background_ratio >= threshold["background_to_tissue_range_ratio"]
        and background_component.get("score", math.nan) >= threshold["control_fixed_frequency_score"]
        and abs(background_corr) >= threshold["control_periodic_correlation"]
    )
    brightness_flag = bool(
        global_component.get("score", math.nan) >= threshold["control_fixed_frequency_score"]
        and abs(global_corr) >= threshold["control_periodic_correlation"]
    )
    ncc = tracked["target_ncc"][start:stop, columns]
    fb = tracked["target_fb"][start:stop, columns]
    mean_ncc = float(np.nanmean(ncc)) if np.isfinite(ncc).any() else math.nan
    p90_fb = float(np.nanpercentile(fb, 90)) if np.isfinite(fb).any() else math.nan
    heldout = tracked["heldout_corrected"][start:stop]
    mean_heldout = float(np.nanmean(heldout)) if np.isfinite(heldout).any() else math.nan
    tracking_flag = bool(
        participation < threshold["minimum_point_participation_fraction"]
        or (np.isfinite(coherence) and coherence < threshold["supported_point_phase_coherence"])
        or (np.isfinite(mean_ncc) and mean_ncc < threshold["minimum_mean_ncc"])
        or (np.isfinite(p90_fb) and p90_fb > threshold["maximum_p90_forward_backward_px"])
    )
    coordinated = bool(
        usable and not tracking_flag
        and participation >= threshold["minimum_point_participation_fraction"]
        and np.isfinite(coherence)
        and coherence >= threshold["supported_point_phase_coherence"]
    )
    if tracking_flag:
        evidence_class, reason = "artifact_risk", "tracking_or_point_inconsistency"
    elif coordinated:
        warnings = []
        if background_flag:
            warnings.append("background_same_frequency_warning")
        if brightness_flag:
            warnings.append("global_brightness_same_frequency_warning")
        reason = "usable_frequency;participating_coherent_points"
        if warnings:
            reason += ";" + ";".join(warnings)
        evidence_class = "coordinated_tissue_displacement_supported"
    else:
        evidence_class, reason = "insufficient_evidence", "longitudinal_motion_support_incomplete"
    return {
        "evidence_class": evidence_class,
        "evidence_reason": reason,
        "background_same_frequency_flag": background_flag,
        "brightness_same_frequency_flag": brightness_flag,
        "tracking_artifact_flag": tracking_flag,
        "same_frequency_confounding_flag": bool(background_flag or brightness_flag),
        "tissue_fixed_frequency_score": tissue_component.get("score", ""),
        "background_fixed_frequency_score": background_component.get("score", ""),
        "background_periodic_correlation": background_corr,
        "background_to_tissue_range_ratio_same_segment": background_ratio,
        "local_brightness_fixed_frequency_score": local_component.get("score", ""),
        "local_brightness_periodic_correlation": local_corr,
        "global_brightness_fixed_frequency_score": global_component.get("score", ""),
        "global_brightness_periodic_correlation": global_corr,
        "point_phase_coherence": coherence,
        "point_participation_fraction": participation,
        "mean_target_ncc_same_segment": mean_ncc,
        "p90_target_fb_px_same_segment": p90_fb,
        "mean_background_heldout_residual_px_same_segment": mean_heldout,
    }


def local_tangents(points: np.ndarray) -> np.ndarray:
    """Fixed first-frame local tangents, oriented by stored cervix->fundus order."""
    points = np.asarray(points, float)
    if len(points) < 3 or points.shape[1] != 2:
        raise ValueError("Each wall needs at least three ordered 2-D points")
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    delta[1:-1] = points[2:] - points[:-2]
    norm = np.linalg.norm(delta, axis=1)
    if not np.isfinite(norm).all() or np.any(norm < 1e-8):
        raise ValueError("Degenerate longitudinal tangent")
    return delta / norm[:, None]


def parse_choice(value: object) -> tuple[int, ...]:
    if pd.isna(value) or str(value).strip() == "":
        return ()
    return tuple(int(item) for item in str(value).split(";") if item != "")


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def synthetic_checks() -> dict[str, object]:
    time_s = np.arange(0.0, 80.0, 1.0 / 25.0)
    points = np.array([[0.0, 0.0], [1.0, 0.2], [2.0, 0.3]], float)
    tangent = local_tangents(points)
    recovered = {}
    for frequency in (1.0, 2.0, 3.0):
        wave = 1.2 * np.sin(2 * np.pi * frequency * time_s / 60.0 + 0.3)
        positions = points[None] + wave[:, None, None] * tangent[None]
        signal = np.einsum(
            "tki,ki->tk", positions - points[None], tangent
        ).mean(axis=1)
        fit = fit_candidate(time_s, signal)
        estimate = finite_float(fit.get("dominant_frequency_cpm"))
        recovered[str(frequency)] = estimate
        if not math.isfinite(estimate) or abs(estimate - frequency) > 0.05 + 1e-12:
            raise AssertionError(f"Known longitudinal frequency failed: {frequency}->{estimate}")

    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    cross_wave = np.sin(2 * np.pi * 2.0 * time_s / 60.0)
    cross_positions = points[None] + cross_wave[:, None, None] * normal[None]
    cross_signal = np.einsum(
        "tki,ki->tk", cross_positions - points[None], tangent
    ).mean(axis=1)
    cross_leakage = float(np.max(np.abs(cross_signal)))
    if cross_leakage > 1e-10:
        raise AssertionError(f"Cross-wall motion leaked into longitudinal signal: {cross_leakage}")

    amplitudes = np.array([1.0, 0.0, 0.0])
    participation = float(
        amplitudes.sum() ** 2 / (len(amplitudes) * (amplitudes @ amplitudes))
    )
    if not participation < 2.0 / 3.0:
        raise AssertionError("Single-point participation check failed")

    # Pure rigid background motion must disappear after perfect correction.
    background_wave = 2.0 * np.sin(2 * np.pi * 2.0 * time_s / 60.0)
    pure_background_corrected = np.repeat(points[None], len(time_s), axis=0)
    pure_background_signal = np.einsum(
        "tki,ki->tk", pure_background_corrected - points[None], tangent
    ).mean(axis=1)
    pure_background_fit = fit_candidate(time_s, pure_background_signal)
    pure_background_usable, _ = usability(
        pure_background_fit, float(time_s[-1] - time_s[0])
    )
    if pure_background_usable:
        raise AssertionError("Perfectly corrected pure background motion was retained")

    # True local motion may coexist at exactly the same frequency as background.
    # Same-frequency background is a warning, not proof that corrected motion is artifact.
    local_wave = 1.2 * np.sin(2 * np.pi * 2.0 * time_s / 60.0 + 0.3)
    local_corrected = points[None] + local_wave[:, None, None] * tangent[None]
    poses = np.repeat(np.eye(3)[None], len(time_s), axis=0)
    poses[:, 0, 2] = -background_wave
    tracked = {
        "time_s": time_s,
        "corrected": local_corrected,
        "valid": np.ones((len(time_s), len(points)), dtype=bool),
        "pose_valid": np.ones(len(time_s), dtype=bool),
        "poses": poses,
        "target_ncc": np.ones((len(time_s), len(points))),
        "target_fb": np.zeros((len(time_s), len(points))),
        "point_brightness": np.ones((len(time_s), len(points))),
        "global_brightness": np.ones(len(time_s)),
        "heldout_corrected": np.zeros(len(time_s)),
    }
    same_frequency_fit = fit_candidate(time_s, local_wave)
    same_frequency_usable, _ = usability(
        same_frequency_fit, float(time_s[-1] - time_s[0])
    )
    same_frequency_evidence = artifact_evidence(
        tracked,
        points,
        tangent,
        {"start": 0, "stop": len(time_s), "columns": np.arange(len(points))},
        same_frequency_fit,
        same_frequency_usable,
    )
    if not same_frequency_evidence["background_same_frequency_flag"]:
        raise AssertionError("Same-frequency background warning was not detected")
    if same_frequency_evidence["evidence_class"] != "coordinated_tissue_displacement_supported":
        raise AssertionError("True local motion was rejected only because background had the same frequency")
    same_frequency_region, _ = analyse_region(
        pd.Series({
            "case_id": "synthetic_same_frequency",
            "wall": "anterior",
            "region": "middle",
            "selected_local_point_indices": "0;1;2",
            "selected_start_frame": 0,
            "selected_stop_frame_exclusive": len(time_s),
            "selected_point_count": len(points),
        }),
        tracked,
        points,
        tangent,
        tangent,
        np.arange(len(points)),
        np.arange(len(points)),
    )
    if not same_frequency_region["retained_for_longitudinal_model"]:
        raise AssertionError("Same-frequency true local motion was not retained")
    if same_frequency_region["longitudinal_technical_confidence"] != "medium":
        raise AssertionError("Same-frequency warning did not downgrade confidence to medium")

    brightness_tracked = dict(tracked)
    brightness_tracked["poses"] = np.repeat(np.eye(3)[None], len(time_s), axis=0)
    brightness_tracked["global_brightness"] = 100.0 + 5.0 * local_wave
    brightness_evidence = artifact_evidence(
        brightness_tracked,
        points,
        tangent,
        {"start": 0, "stop": len(time_s), "columns": np.arange(len(points))},
        same_frequency_fit,
        same_frequency_usable,
    )
    if not brightness_evidence["brightness_same_frequency_flag"]:
        raise AssertionError("Same-frequency brightness warning was not detected")
    if brightness_evidence["evidence_class"] != "coordinated_tissue_displacement_supported":
        raise AssertionError("True local motion was rejected only because brightness had the same frequency")
    return {
        "status": "pass",
        "known_frequency_recovery_cpm": recovered,
        "pure_cross_wall_projection_leakage_px": cross_leakage,
        "pure_background_after_perfect_correction_retained": pure_background_usable,
        "same_frequency_background_warning_detected": bool(
            same_frequency_evidence["background_same_frequency_flag"]
        ),
        "true_local_motion_with_same_frequency_background_class": (
            same_frequency_evidence["evidence_class"]
        ),
        "true_local_motion_with_same_frequency_background_confidence": (
            same_frequency_region["longitudinal_technical_confidence"]
        ),
        "same_frequency_brightness_warning_detected": bool(
            brightness_evidence["brightness_same_frequency_flag"]
        ),
        "true_local_motion_with_same_frequency_brightness_class": (
            brightness_evidence["evidence_class"]
        ),
        "single_moving_point_participation_fraction": participation,
    }


def cache_map(run_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in run_root.glob("cases/case_*/trajectory_cache/*.npz"):
        with np.load(path, allow_pickle=False) as archive:
            case_id = str(archive["case_id"][0])
        if case_id in mapping:
            raise ValueError(f"Duplicate trajectory cache for {case_id}")
        mapping[case_id] = path
    return mapping


def selected_segment(row: pd.Series, columns: np.ndarray) -> dict | None:
    choice = parse_choice(row.get("selected_local_point_indices"))
    start = int(finite_float(row.get("selected_start_frame"), -1))
    stop = int(finite_float(row.get("selected_stop_frame_exclusive"), -1))
    if not choice or start < 0 or stop <= start:
        return None
    if max(choice) >= len(columns):
        raise ValueError("Saved local point choice is outside its region")
    return {
        "start": start,
        "stop": stop,
        "columns": columns[list(choice)],
        "local_choice": choice,
        "duration_s": finite_float(row.get("optimized_segment_duration_s"), 0.0),
    }


def analyse_region(
    row: pd.Series,
    tracked: dict,
    initial: np.ndarray,
    longitudinal_axes: np.ndarray,
    inward_axes: np.ndarray,
    local_indices: np.ndarray,
    columns: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray] | None]:
    selected = selected_segment(row, columns)
    base = {
        "case_id": row["case_id"],
        "wall": row["wall"],
        "region": row["region"],
        "source_selected_point_count": int(finite_float(row.get("selected_point_count"), 0)),
        "source_selected_point_fraction": row.get("selected_point_fraction", ""),
        "source_selected_span_fraction": row.get("selected_span_fraction", ""),
        "source_selected_local_point_indices": row.get("selected_local_point_indices", ""),
        "source_segment_start_frame": row.get("selected_start_frame", ""),
        "source_segment_stop_frame_exclusive": row.get("selected_stop_frame_exclusive", ""),
        "source_inward_frequency_cpm": row.get("optimized_candidate_frequency_cpm", ""),
        "source_inward_technical_confidence": row.get("technical_confidence", ""),
    }
    if selected is None:
        return {
            **base,
            "longitudinal_candidate_status": "missing",
            "longitudinal_candidate_reason": "source_fixed_point_segment_missing",
            "frequency_usable": False,
            "longitudinal_technical_confidence": "excluded",
            "retained_for_longitudinal_model": False,
        }, None

    start, stop = selected["start"], selected["stop"]
    chosen = selected["columns"]
    if stop > len(tracked["time_s"]):
        raise ValueError("Saved segment exceeds trajectory cache")
    time_s = tracked["time_s"][start:stop]
    displacement = tracked["corrected"][start:stop, chosen] - initial[chosen]
    point_signals = np.einsum(
        "tki,ki->tk", displacement, longitudinal_axes[chosen]
    )
    signal = point_signals.mean(axis=1)
    fit = fit_candidate(time_s, signal)
    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
    usable, usability_reason = usability(fit, duration_s)
    evidence = artifact_evidence(
        tracked, initial, longitudinal_axes, selected, fit, usable
    )
    coordinated = evidence.get("evidence_class") == "coordinated_tissue_displacement_supported"
    retained = bool(usable and coordinated)
    cycles = finite_float(fit.get("observed_cycles"), 0.0)
    same_frequency_warning = bool(evidence.get("same_frequency_confounding_flag", False))
    confidence = (
        "high" if retained and cycles >= 2.0 and not same_frequency_warning
        else "medium" if retained
        else "excluded"
    )

    component = {}
    if fit.get("status") == "valid":
        component = fixed_frequency_component(
            time_s, signal, float(fit["dominant_frequency_cpm"])
        )
    coefficient = component.get("coefficient", complex(math.nan, math.nan))
    phase = (
        float(math.atan2(coefficient.imag, coefficient.real))
        if math.isfinite(coefficient.real) and math.isfinite(coefficient.imag)
        else math.nan
    )
    dot = np.sum(
        longitudinal_axes[chosen] * inward_axes[chosen], axis=1
    )
    angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
    effective_min = 60.0 / duration_s if duration_s > 0 else math.nan

    output = {
        **base,
        "longitudinal_candidate_status": fit.get("status", "missing"),
        "longitudinal_candidate_reason": fit.get("reason", ""),
        "longitudinal_frequency_cpm": fit.get("dominant_frequency_cpm", ""),
        "longitudinal_period_s": fit.get("period_s", ""),
        "longitudinal_score": fit.get("score", ""),
        "longitudinal_amplitude_px": component.get("amplitude", ""),
        "longitudinal_phase_rad": phase if math.isfinite(phase) else "",
        "longitudinal_displacement_range_px": float(np.ptp(signal)),
        "segment_duration_s": duration_s,
        "observed_cycles": fit.get("observed_cycles", ""),
        "effective_minimum_one_cycle_frequency_cpm": effective_min,
        "search_boundary": fit.get("search_boundary", ""),
        "periodic_over_quadratic_improvement": fit.get(
            "periodic_over_quadratic_improvement", ""
        ),
        "frequency_usable": bool(usable),
        "frequency_usability_reason": usability_reason,
        "evidence_class": (
            "longitudinal_coordinated_motion_supported" if coordinated
            else evidence.get("evidence_class", "insufficient_evidence")
        ),
        "evidence_reason": evidence.get("evidence_reason", ""),
        "longitudinal_technical_confidence": confidence,
        "retained_for_longitudinal_model": retained,
        "cycle_reliability": (
            "two_or_more_cycles" if cycles >= 2.0 else
            "one_to_less_than_two_cycles" if cycles >= 1.0 else
            "below_one_cycle"
        ),
        "mean_absolute_tangent_inward_dot": float(np.mean(np.abs(dot))),
        "median_tangent_inward_angle_deg": float(np.median(angle)),
        **{key: value for key, value in evidence.items() if key not in {"evidence_class", "evidence_reason"}},
    }
    diagnostic = None
    if component and "residual" in component and "periodic" in component:
        diagnostic = {
            "time_s": time_s - time_s[0],
            "residual": component["residual"],
            "periodic": component["periodic"],
        }
    return output, diagnostic


def patient_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(row)
    output = []
    for case_id, own in grouped.items():
        retained = [row for row in own if row.get("retained_for_longitudinal_model")]
        output.append({
            "case_id": case_id,
            "regions_total": len(own),
            "retained_longitudinal_region_count": len(retained),
            "retained_anterior_count": sum(row["wall"] == "anterior" for row in retained),
            "retained_posterior_count": sum(row["wall"] == "posterior" for row in retained),
            "high_confidence_count": sum(row.get("longitudinal_technical_confidence") == "high" for row in own),
            "medium_confidence_count": sum(row.get("longitudinal_technical_confidence") == "medium" for row in own),
            "two_or_more_cycles_count": sum(row.get("cycle_reliability") == "two_or_more_cycles" for row in retained),
        })
    return pd.DataFrame(output).sort_values("case_id")


def plot_diagnostics(
    output_dir: Path,
    selected_cases: list[str],
    diagnostics: dict[tuple[str, str, str], dict[str, np.ndarray]],
    rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(str(row["case_id"]), str(row["wall"]), str(row["region"])): row for row in rows}
    for case_id in selected_cases:
        figure, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
        for axis, wall, region in zip(
            axes.ravel(),
            ["anterior"] * 3 + ["posterior"] * 3,
            list(REGION_NAMES) * 2,
        ):
            key = (case_id, wall, region)
            diagnostic = diagnostics.get(key)
            row = lookup.get(key, {})
            if diagnostic is None:
                axis.text(0.5, 0.5, "no retained curve", ha="center", va="center")
            else:
                axis.plot(diagnostic["time_s"], diagnostic["residual"], lw=1.0, label="detrended longitudinal displacement")
                axis.plot(diagnostic["time_s"], diagnostic["periodic"], lw=2.0, label="fitted periodic component")
            axis.set_title(
                f"{wall}-{region} | f={row.get('longitudinal_frequency_cpm', '')} cpm | {row.get('longitudinal_technical_confidence', '')}"
            )
            axis.set_xlabel("s")
            axis.set_ylabel("px")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(fontsize=8)
        figure.suptitle(f"Longitudinal cervix-to-fundus motion: {case_id}")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)
        figure.savefig(output_dir / f"{safe}.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Longitudinal tissue-motion frequency from frozen R2 caches")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    checks = synthetic_checks()
    (output / "synthetic_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    region_file = source_run / "merged_results" / "optimized_region_frequency.csv"
    source_regions = pd.read_csv(region_file)
    caches = cache_map(source_run)
    expected_cases = sorted(source_regions["case_id"].astype(str).unique())
    if not expected_cases:
        raise ValueError("Source run contains no patients")
    if set(caches) != set(expected_cases):
        raise ValueError(
            f"Cache/case mismatch: caches={len(caches)}, region cases={len(expected_cases)}"
        )
    validation_count = min(10, len(expected_cases))
    validation_cases = [
        expected_cases[i]
        for i in np.linspace(0, len(expected_cases) - 1, validation_count, dtype=int)
    ]

    rows: list[dict[str, object]] = []
    diagnostics: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for order, case_id in enumerate(expected_cases, start=1):
        tracked, initial, anterior_inward, posterior_inward = load_cache(caches[case_id])
        count = len(initial) // 2
        if len(initial) != 2 * count or count < 6:
            raise ValueError(f"Unexpected wall point count for {case_id}")
        anterior_tangent = local_tangents(initial[:count])
        posterior_tangent = local_tangents(initial[count:])
        tangents = np.vstack((anterior_tangent, posterior_tangent))
        inward = np.vstack((anterior_inward, posterior_inward))
        regions = dict(zip(REGION_NAMES, np.array_split(np.arange(count), 3)))
        own = source_regions.loc[source_regions["case_id"].astype(str) == case_id]
        if len(own) != 6:
            raise ValueError(f"Expected six regions for {case_id}, found {len(own)}")
        for wall in WALLS:
            offset = 0 if wall == "anterior" else count
            for region in REGION_NAMES:
                row = own.loc[(own["wall"] == wall) & (own["region"] == region)]
                if len(row) != 1:
                    raise ValueError(f"Missing/duplicate source region: {case_id} {wall} {region}")
                local_indices = regions[region]
                columns = local_indices + offset
                result, diagnostic = analyse_region(
                    row.iloc[0], tracked, initial, tangents, inward,
                    local_indices, columns,
                )
                rows.append(result)
                if case_id in validation_cases and diagnostic is not None:
                    diagnostics[(case_id, wall, region)] = diagnostic
        if order % 25 == 0 or order == len(expected_cases):
            print(f"cases {order}/{len(expected_cases)}", flush=True)

    region_table = pd.DataFrame(rows).sort_values(["case_id", "wall", "region"])
    patient_table = patient_summary(rows)
    region_table.to_csv(output / "longitudinal_region_frequency.csv", index=False, encoding="utf-8-sig")
    patient_table.to_csv(output / "longitudinal_patient_technical_summary.csv", index=False, encoding="utf-8-sig")
    plot_diagnostics(output / "validation_10_case_plots", validation_cases, diagnostics, rows)

    retained = region_table["retained_for_longitudinal_model"].astype(bool)
    counts = {
        "rule_version": RULE_VERSION,
        "clinical_outcome_read": False,
        "source_run": str(source_run),
        "patients": int(patient_table["case_id"].nunique()),
        "regions": int(len(region_table)),
        "retained_regions": int(retained.sum()),
        "high_confidence_regions": int((region_table["longitudinal_technical_confidence"] == "high").sum()),
        "medium_confidence_regions": int((region_table["longitudinal_technical_confidence"] == "medium").sum()),
        "excluded_regions": int((region_table["longitudinal_technical_confidence"] == "excluded").sum()),
        "patients_with_retained_region": int((patient_table["retained_longitudinal_region_count"] > 0).sum()),
        "patients_without_retained_region": int((patient_table["retained_longitudinal_region_count"] == 0).sum()),
        "projection": "fixed first-frame local wall tangent oriented cervix-to-fundus",
        "frequency_grid_cpm": [0.5, 5.0, 0.05],
        "minimum_duration_s": 15.0,
        "minimum_observed_cycles": 1.0,
        "minimum_periodic_over_quadratic_improvement": 0.05,
        "patient_frequency_aggregation": "none",
        "validation_cases": validation_cases,
    }
    (output / "longitudinal_rules_and_counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# R2纵向局部组织位移频率

- 方向：每面壁第一帧局部切线，统一指向宫颈端→宫底端。
- 基础：复用冻结流程的背景校正二维轨迹、固定点集合和连续片段；没有重新追踪视频。
- 患者：{counts['patients']}例；区域：{counts['regions']}个。
- 保留纵向频率：{counts['retained_regions']}个，其中高可信{counts['high_confidence_regions']}个、中可信{counts['medium_confidence_regions']}个。
- 至少一个保留纵向区域的患者：{counts['patients_with_retained_region']}例；无保留纵向区域：{counts['patients_without_retained_region']}例。
- 当前不做患者级频率聚合，不读取临床结局，不计算AUC。

“高/中可信”只表示技术可靠性：高可信覆盖至少2个周期，中可信覆盖1至不足2个周期；两者均需多点协调且无明确追踪、背景同频或亮度伪影。

本指标定义为“背景校正后、沿宫颈至宫底局部壁切线方向的区域平均组织位移主导频率”，不等同于收缩次数或已经证实的传播方向。相位已经保存，可用于后续独立定义传播特征。
"""
    (output / "纵向频率结果说明.md").write_text(report, encoding="utf-8")
    shutil.copy2(Path(__file__), output / "运行脚本快照.py")
    print(f"complete: {output}", flush=True)


if __name__ == "__main__":
    main()
