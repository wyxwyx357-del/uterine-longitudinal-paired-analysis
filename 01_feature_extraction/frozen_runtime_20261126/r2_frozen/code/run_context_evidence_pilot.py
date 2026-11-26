"""Read-only evidence sidecar for the frozen V1 uterine-motion pipeline.

The script reuses the frozen tracker, rigid registration, regional support,
and frequency functions unchanged.  It adds descriptive evidence channels
for global motion, neighboring wall points, paired-wall distance, brightness,
and tracking quality.  It does not classify real physiology or modify tracks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
CODE = PACKAGE / "code"
sys.path.insert(0, str(CODE))

from ab_block_tracking import patch as tracking_patch  # noqa: E402
from ab_block_tracking import track_step  # noqa: E402
from contrast import local_contrast  # noqa: E402
from rigid_correction import step as rigid_step  # noqa: E402
import validate_regional_v1_1 as frozen  # noqa: E402


RANDOM_SEED = 20261028
FROZEN_FILES = (
    "ab_block_tracking.py",
    "cuda_sparse_lk.py",
    "contrast.py",
    "rigid_correction.py",
    "frequency.py",
    "run_regional_frequency.py",
    "validate_regional_v1_1.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: np.ndarray, axis=None):
    values = np.asarray(values, float)
    if axis is None:
        selected = values[np.isfinite(values)]
        return float(selected.mean()) if len(selected) else math.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return math.nan
    x = left[mask] - left[mask].mean()
    y = right[mask] - right[mask].mean()
    scale = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / scale) if scale > 1e-12 else math.nan


def finite_range(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    selected = values[np.isfinite(values)]
    return float(np.ptp(selected)) if len(selected) else math.nan


def contiguous_runs(mask: np.ndarray, epoch: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, bool)
    epoch = np.asarray(epoch)
    starts, stops = [], []
    active = False
    for index, good in enumerate(mask):
        continued = index > 0 and good and mask[index - 1] and epoch[index] == epoch[index - 1]
        if good and not continued:
            starts.append(index)
            active = True
        if active and (not good or index == len(mask) - 1):
            stops.append(index if good and index == len(mask) - 1 else index - 1)
            active = False
    return [(start, stop + 1) for start, stop in zip(starts, stops)]


def patch_median(frame: np.ndarray, point: np.ndarray) -> float:
    own = tracking_patch(frame, point)
    return float(np.median(own)) if own is not None else math.nan


def target_layout(
    holdout: dict,
    anterior_axes: np.ndarray,
    posterior_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(anterior_axes)
    regions = dict(zip(frozen.REGION_NAMES, np.array_split(np.arange(count), 3)))
    local = regions[holdout["target_region"]]
    columns = local if holdout["target_wall"] == "anterior" else local + count
    all_axes = np.vstack((anterior_axes, posterior_axes))
    return local.astype(int), columns.astype(int), all_axes


def track_with_evidence(
    gray: np.ndarray,
    initial: np.ndarray,
    local_vector: np.ndarray,
    local_envelope: np.ndarray,
    condition: str,
    frequency_cpm: float,
    duration_s: int,
    target_columns: np.ndarray,
) -> dict:
    """Frozen tracking path with additional read-only diagnostic capture."""
    time_s = np.arange(int(round(duration_s * frozen.FPS)) + 1, dtype=float) / frozen.FPS
    n, count = len(time_s), len(initial)
    positions = np.full((n, count, 2), np.nan, np.float32)
    raw_positions = np.full_like(positions, np.nan)
    valid = np.zeros((n, count), bool)
    raw_valid = np.zeros_like(valid)
    corrected = np.full((n, count, 2), np.nan, float)
    pose_valid = np.zeros(n, bool)
    epoch = np.zeros(n, int)
    poses = np.full((n, 3, 3), np.nan, float)
    target_ncc = np.full((n, count), np.nan, float)
    target_fb = np.full((n, count), np.nan, float)
    local_brightness = np.full(n, np.nan, float)
    global_brightness = np.full(n, np.nan, float)
    heldout_raw = np.full(n, np.nan, float)
    heldout_corrected = np.full(n, np.nan, float)
    training_inliers = np.zeros(n, int)
    heldout_count = np.zeros(n, int)

    yy, xx = np.indices(gray.shape, dtype=np.float32)
    fan_mask = gray > 12
    accumulator = np.eye(3)
    current_epoch = 0
    previous = frozen.make_frame(
        gray, xx, yy, local_vector, local_envelope,
        condition, 0.0, frequency_cpm,
    )
    processed_previous = local_contrast(previous)
    positions[0] = raw_positions[0] = initial
    valid[0] = raw_valid[0] = True
    corrected[0] = initial
    pose_valid[0] = True
    poses[0] = accumulator
    local_brightness[0] = finite_mean([
        patch_median(previous, initial[column]) for column in target_columns
    ])
    global_brightness[0] = float(np.median(previous[fan_mask]))

    for index in range(1, n):
        current = frozen.make_frame(
            gray, xx, yy, local_vector, local_envelope,
            condition, float(time_s[index]), frequency_cpm,
        )
        processed = local_contrast(current)
        live = np.flatnonzero(valid[index - 1])
        if len(live):
            found, good, scores, fb = track_step(
                processed_previous, processed, positions[index - 1, live])
            target_ncc[index, live] = scores
            target_fb[index, live] = fb
            kept = live[good]
            positions[index, kept] = found[good]
            valid[index, kept] = True

        raw_live = np.flatnonzero(raw_valid[index - 1])
        if len(raw_live):
            found, good, _, _ = track_step(
                previous, current, raw_positions[index - 1, raw_live])
            kept = raw_live[good]
            raw_positions[index, kept] = found[good]
            raw_valid[index, kept] = True

        cv2.setRNGSeed(RANDOM_SEED)
        matrix, stats = rigid_step(
            previous, current, raw_positions[index - 1, raw_live]
        ) if len(raw_live) >= 3 else (None, {})
        if matrix is None:
            accumulator = np.eye(3)
            current_epoch += 1
        else:
            accumulator = accumulator @ np.linalg.inv(
                np.vstack((matrix, [0.0, 0.0, 1.0])))
            pose_valid[index] = True
            poses[index] = accumulator
            corrected[index, valid[index]] = (
                positions[index, valid[index]] @ accumulator[:2, :2].T
                + accumulator[:2, 2]
            )
        epoch[index] = current_epoch
        heldout_raw[index] = stats.get("heldout_raw_median_px", math.nan)
        heldout_corrected[index] = stats.get("heldout_corrected_median_px", math.nan)
        training_inliers[index] = stats.get("training_inliers", 0)
        heldout_count[index] = stats.get("heldout_count", 0)
        target_points = [
            patch_median(current, positions[index, column])
            for column in target_columns if valid[index, column]
        ]
        local_brightness[index] = finite_mean(target_points)
        global_brightness[index] = float(np.median(current[fan_mask]))
        previous = current
        processed_previous = processed

    return dict(
        time_s=time_s,
        positions=positions,
        raw_positions=raw_positions,
        valid=valid,
        raw_valid=raw_valid,
        corrected=corrected,
        pose_valid=pose_valid,
        epoch=epoch,
        poses=poses,
        target_ncc=target_ncc,
        target_fb=target_fb,
        local_brightness=local_brightness,
        global_brightness=global_brightness,
        heldout_raw=heldout_raw,
        heldout_corrected=heldout_corrected,
        training_inliers=training_inliers,
        heldout_count=heldout_count,
    )


def coordinate_rmse(
    tracked: dict,
    initial: np.ndarray,
    local_vector: np.ndarray,
    target_columns: np.ndarray,
    condition: str,
    frequency_cpm: float,
) -> tuple[float, int]:
    point_vectors = []
    for point in initial:
        x = int(np.clip(round(float(point[0])), 0, local_vector.shape[1] - 1))
        y = int(np.clip(round(float(point[1])), 0, local_vector.shape[0] - 1))
        point_vectors.append(local_vector[y, x])
    point_vectors = np.asarray(point_vectors, float)
    amplitude = frozen.POSITIVE_CONDITIONS.get(condition, 0.0)
    phase = np.sin(2 * np.pi * frequency_cpm * tracked["time_s"] / 60)
    expected = (
        initial[None].astype(float)
        + amplitude * phase[:, None, None] * point_vectors[None]
    )
    mask = (
        tracked["pose_valid"][:, None]
        & tracked["valid"][:, target_columns]
        & np.isfinite(tracked["corrected"][:, target_columns]).all(axis=2)
    )
    delta = tracked["corrected"][:, target_columns] - expected[:, target_columns]
    errors = np.linalg.norm(delta, axis=2)[mask]
    return (
        float(np.sqrt(np.mean(errors * errors))) if len(errors) else math.nan,
        int(len(errors)),
    )


def build_evidence_arrays(
    tracked: dict,
    initial: np.ndarray,
    all_axes: np.ndarray,
    local_indices: np.ndarray,
    target_columns: np.ndarray,
    condition: str,
    frequency_cpm: float,
    local_vector: np.ndarray,
) -> dict:
    corrected_delta = tracked["corrected"] - initial[None]
    raw_delta = tracked["positions"].astype(float) - initial[None]
    corrected_point = np.einsum(
        "nki,ki->nk", corrected_delta[:, target_columns], all_axes[target_columns])
    raw_point = np.einsum(
        "nki,ki->nk", raw_delta[:, target_columns], all_axes[target_columns])
    target_valid = (
        tracked["pose_valid"]
        & tracked["valid"][:, target_columns].all(axis=1)
        & np.isfinite(corrected_point).all(axis=1)
        & np.isfinite(raw_point).all(axis=1)
    )
    corrected_mean = np.where(target_valid, corrected_point.mean(axis=1), np.nan)
    raw_mean = np.where(target_valid, raw_point.mean(axis=1), np.nan)
    neighbor_dispersion = np.where(
        target_valid, np.std(corrected_point, axis=1), np.nan)

    background_point = np.full_like(corrected_point, np.nan, float)
    for frame in np.flatnonzero(tracked["pose_valid"]):
        inverse = np.linalg.inv(tracked["poses"][frame])
        mapped = (
            initial[target_columns] @ inverse[:2, :2].T + inverse[:2, 2]
        )
        displacement = mapped - initial[target_columns]
        background_point[frame] = np.einsum(
            "ki,ki->k", displacement, all_axes[target_columns])
    background_mean = np.where(
        tracked["pose_valid"], np.nanmean(background_point, axis=1), np.nan)

    count = len(initial) // 2
    anterior_columns = local_indices
    posterior_columns = local_indices + count
    pair_valid = (
        tracked["pose_valid"]
        & tracked["valid"][:, anterior_columns].all(axis=1)
        & tracked["valid"][:, posterior_columns].all(axis=1)
        & np.isfinite(tracked["corrected"][:, anterior_columns]).all(axis=(1, 2))
        & np.isfinite(tracked["corrected"][:, posterior_columns]).all(axis=(1, 2))
    )
    initial_distance = np.linalg.norm(
        initial[posterior_columns] - initial[anterior_columns], axis=1)
    current_distance = np.linalg.norm(
        tracked["corrected"][:, posterior_columns]
        - tracked["corrected"][:, anterior_columns], axis=2)
    pair_strain = current_distance / initial_distance[None] - 1.0
    radial_strain = np.where(pair_valid, pair_strain.mean(axis=1), np.nan)

    ncc = tracked["target_ncc"][:, target_columns]
    fb = tracked["target_fb"][:, target_columns]
    mean_ncc = np.array([finite_mean(row) for row in ncc])
    max_fb = np.array([
        float(np.nanmax(row)) if np.isfinite(row).any() else math.nan for row in fb
    ])

    point_vectors = []
    for column in target_columns:
        point = initial[column]
        x = int(np.clip(round(float(point[0])), 0, local_vector.shape[1] - 1))
        y = int(np.clip(round(float(point[1])), 0, local_vector.shape[0] - 1))
        point_vectors.append(local_vector[y, x])
    point_vectors = np.asarray(point_vectors, float)
    projected_truth_scale = np.einsum(
        "ki,ki->k", point_vectors, all_axes[target_columns]).mean()
    amplitude = frozen.POSITIVE_CONDITIONS.get(condition, 0.0)
    phase = np.sin(2 * np.pi * frequency_cpm * tracked["time_s"] / 60)
    truth_mean = amplitude * projected_truth_scale * phase

    evidence_valid = (
        target_valid
        & pair_valid
        & np.isfinite(background_mean)
        & np.isfinite(radial_strain)
        & np.isfinite(tracked["local_brightness"])
        & np.isfinite(tracked["global_brightness"])
    )
    return dict(
        target_valid=target_valid,
        pair_valid=pair_valid,
        evidence_valid=evidence_valid,
        corrected_point=corrected_point,
        raw_point=raw_point,
        corrected_mean=corrected_mean,
        raw_mean=raw_mean,
        background_point=background_point,
        background_mean=background_mean,
        radial_strain=radial_strain,
        neighbor_dispersion=neighbor_dispersion,
        mean_ncc=mean_ncc,
        max_fb=max_fb,
        truth_mean=truth_mean,
    )


def pairwise_neighbor_correlation(values: np.ndarray) -> float:
    correlations = []
    for index in range(values.shape[1] - 1):
        correlations.append(safe_corr(values[:, index], values[:, index + 1]))
    selected = np.asarray(correlations, float)
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if len(selected) else math.nan


def segment_rows(
    holdout: dict,
    condition: str,
    tracked: dict,
    evidence: dict,
    frequency_row: dict,
) -> list[dict]:
    runs = contiguous_runs(evidence["evidence_valid"], tracked["epoch"])
    if not runs:
        return [dict(
            holdout_id=holdout["holdout_id"], condition=condition,
            segment="", primary=False, start_frame="", end_frame="",
            duration_s=0.0, frames=0, expected_class=expected_class(condition),
            frequency_status=frequency_row["status"],
            frequency_cpm=frequency_row["frequency_cpm"],
            diagnostic_best_cpm=frequency_row["diagnostic_best_cpm"],
            target_corrected_range_px="", target_raw_range_px="",
            background_range_px="", radial_strain_range="",
            local_brightness_range="", global_brightness_range="",
            raw_background_corr="", corrected_background_corr="",
            corrected_local_brightness_corr="",
            corrected_global_brightness_corr="",
            corrected_radial_strain_corr="", neighbor_adjacent_corr="",
            mean_neighbor_dispersion_px="", mean_target_ncc="",
            max_target_fb_px="", mean_heldout_residual_px="",
            evidence_valid_fraction=float(evidence["evidence_valid"].mean()),
        )]
    lengths = np.array([stop - start for start, stop in runs])
    primary_index = int(np.argmax(lengths))
    rows = []
    for number, (start, stop) in enumerate(runs):
        own = slice(start, stop)
        corrected = evidence["corrected_mean"][own]
        raw = evidence["raw_mean"][own]
        background = evidence["background_mean"][own]
        local_brightness = tracked["local_brightness"][own]
        global_brightness = tracked["global_brightness"][own]
        radial = evidence["radial_strain"][own]
        rows.append(dict(
            holdout_id=holdout["holdout_id"], condition=condition,
            segment=number, primary=number == primary_index,
            start_frame=start, end_frame=stop - 1,
            duration_s=float(tracked["time_s"][stop - 1] - tracked["time_s"][start]),
            frames=stop - start, expected_class=expected_class(condition),
            frequency_status=frequency_row["status"],
            frequency_cpm=frequency_row["frequency_cpm"],
            diagnostic_best_cpm=frequency_row["diagnostic_best_cpm"],
            target_corrected_range_px=finite_range(corrected),
            target_raw_range_px=finite_range(raw),
            background_range_px=finite_range(background),
            radial_strain_range=finite_range(radial),
            local_brightness_range=finite_range(local_brightness),
            global_brightness_range=finite_range(global_brightness),
            raw_background_corr=safe_corr(raw, background),
            corrected_background_corr=safe_corr(corrected, background),
            corrected_local_brightness_corr=safe_corr(corrected, local_brightness),
            corrected_global_brightness_corr=safe_corr(corrected, global_brightness),
            corrected_radial_strain_corr=safe_corr(corrected, radial),
            neighbor_adjacent_corr=pairwise_neighbor_correlation(
                evidence["corrected_point"][own]),
            mean_neighbor_dispersion_px=finite_mean(
                evidence["neighbor_dispersion"][own]),
            mean_target_ncc=finite_mean(evidence["mean_ncc"][own]),
            max_target_fb_px=float(np.nanmax(evidence["max_fb"][own]))
                if np.isfinite(evidence["max_fb"][own]).any() else math.nan,
            mean_heldout_residual_px=finite_mean(tracked["heldout_corrected"][own]),
            evidence_valid_fraction=float(evidence["evidence_valid"].mean()),
        ))
    return rows


def expected_class(condition: str) -> str:
    return {
        "no_motion": "no_motion",
        "global_translation": "global_motion_artifact",
        "global_brightness": "global_brightness_artifact",
        "local_brightness": "local_brightness_artifact",
        "local_motion_0p3": "true_local_motion",
        "local_motion_1p0": "true_local_motion",
        "mixed_local_0p3": "true_local_motion_with_global_and_brightness",
    }[condition]


def primary_frequency(
    holdout: dict,
    tracked: dict,
    anterior_axes: np.ndarray,
    posterior_axes: np.ndarray,
    duration_s: int,
) -> tuple[dict, list[dict]]:
    rows, _ = frozen.regional_rows(
        holdout["holdout_id"], tracked, anterior_axes, posterior_axes,
        holdout["mm_per_processed_pixel"], duration_s,
    )
    target = frozen.primary_for(
        rows, holdout["target_wall"], holdout["target_region"])
    primary_rows = [row for row in rows if row["primary"]]
    return target, primary_rows


def frame_rows(
    holdout: dict,
    condition: str,
    tracked: dict,
    evidence: dict,
) -> list[dict]:
    rows = []
    for index, seconds in enumerate(tracked["time_s"]):
        rows.append(dict(
            holdout_id=holdout["holdout_id"], condition=condition,
            frame=index, time_s=seconds, epoch=int(tracked["epoch"][index]),
            pose_valid=bool(tracked["pose_valid"][index]),
            target_region_valid=bool(evidence["target_valid"][index]),
            paired_wall_valid=bool(evidence["pair_valid"][index]),
            evidence_valid=bool(evidence["evidence_valid"][index]),
            target_corrected_mean_px=evidence["corrected_mean"][index],
            target_raw_mean_px=evidence["raw_mean"][index],
            background_projected_mean_px=evidence["background_mean"][index],
            radial_strain=evidence["radial_strain"][index],
            local_brightness_median=tracked["local_brightness"][index],
            global_brightness_median=tracked["global_brightness"][index],
            neighbor_dispersion_px=evidence["neighbor_dispersion"][index],
            mean_target_ncc=evidence["mean_ncc"][index],
            max_target_fb_px=evidence["max_fb"][index],
            heldout_raw_median_px=tracked["heldout_raw"][index],
            heldout_corrected_median_px=tracked["heldout_corrected"][index],
            training_inliers=int(tracked["training_inliers"][index]),
            heldout_count=int(tracked["heldout_count"][index]),
            truth_target_mean_px=evidence["truth_mean"][index],
        ))
    return rows


def process_case(
    holdout: dict,
    duration_s: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[str]]:
    cv2.setNumThreads(1)
    gray, initial, anterior_axes, posterior_axes = frozen.load_background(holdout)
    local_indices, target_columns, all_axes = target_layout(
        holdout, anterior_axes, posterior_axes)
    local_vector, local_envelope = frozen.build_fields(
        gray.shape, initial, all_axes, target_columns)
    own_frames, own_segments, own_cases, own_frequency, logs = [], [], [], [], []
    for number, condition in enumerate(frozen.CONDITIONS, start=1):
        started = time.time()
        tracked = track_with_evidence(
            gray, initial, local_vector, local_envelope, condition,
            holdout["true_frequency_cpm"], duration_s, target_columns)
        frequency_row, all_primary = primary_frequency(
            holdout, tracked, anterior_axes, posterior_axes, duration_s)
        evidence = build_evidence_arrays(
            tracked, initial, all_axes, local_indices, target_columns,
            condition, holdout["true_frequency_cpm"], local_vector)
        rows = segment_rows(
            holdout, condition, tracked, evidence, frequency_row)
        primary = next((row for row in rows if row["primary"]), rows[0])
        rmse, samples = coordinate_rmse(
            tracked, initial, local_vector, target_columns,
            condition, holdout["true_frequency_cpm"])
        passed = frequency_row["status"] == "exploratory_repetition_candidate"
        positive = condition in frozen.POSITIVE_CONDITIONS
        agreement = (
            bool(passed and abs(float(frequency_row["frequency_cpm"])
                                - holdout["true_frequency_cpm"]) <= 0.25)
            if positive else ""
        )
        negative_false_pass = (
            sum(row["status"] == "exploratory_repetition_candidate"
                for row in all_primary) if not positive else 0)
        case_row = dict(
            holdout_id=holdout["holdout_id"], condition=condition,
            expected_class=expected_class(condition), duration_s=duration_s,
            target_wall=holdout["target_wall"],
            target_region=holdout["target_region"],
            true_frequency_cpm=holdout["true_frequency_cpm"],
            frequency_status=frequency_row["status"],
            frequency_cpm=frequency_row["frequency_cpm"],
            diagnostic_best_cpm=frequency_row["diagnostic_best_cpm"],
            target_frequency_agreement=agreement,
            negative_region_false_pass=negative_false_pass,
            negative_region_total=6 if not positive else 0,
            coordinate_rmse_px=rmse, coordinate_samples=samples,
            target_valid_fraction=float(evidence["target_valid"].mean()),
            pair_valid_fraction=float(evidence["pair_valid"].mean()),
            evidence_valid_fraction=float(evidence["evidence_valid"].mean()),
            pose_valid_fraction=float(tracked["pose_valid"].mean()),
            primary_segment=primary.get("segment", ""),
            target_corrected_range_px=primary.get("target_corrected_range_px", ""),
            target_raw_range_px=primary.get("target_raw_range_px", ""),
            background_range_px=primary.get("background_range_px", ""),
            radial_strain_range=primary.get("radial_strain_range", ""),
            local_brightness_range=primary.get("local_brightness_range", ""),
            global_brightness_range=primary.get("global_brightness_range", ""),
            raw_background_corr=primary.get("raw_background_corr", ""),
            corrected_background_corr=primary.get("corrected_background_corr", ""),
            corrected_local_brightness_corr=primary.get(
                "corrected_local_brightness_corr", ""),
            corrected_global_brightness_corr=primary.get(
                "corrected_global_brightness_corr", ""),
            corrected_radial_strain_corr=primary.get(
                "corrected_radial_strain_corr", ""),
            neighbor_adjacent_corr=primary.get("neighbor_adjacent_corr", ""),
            mean_neighbor_dispersion_px=primary.get(
                "mean_neighbor_dispersion_px", ""),
            mean_target_ncc=primary.get("mean_target_ncc", ""),
            max_target_fb_px=primary.get("max_target_fb_px", ""),
            mean_heldout_residual_px=primary.get(
                "mean_heldout_residual_px", ""),
        )
        own_frames.extend(frame_rows(holdout, condition, tracked, evidence))
        own_segments.extend(rows)
        own_cases.append(case_row)
        for row in all_primary:
            own_frequency.append(dict(
                holdout_id=holdout["holdout_id"], condition=condition,
                wall=row["wall"], region=row["region"],
                is_target=(row["wall"] == holdout["target_wall"]
                           and row["region"] == holdout["target_region"]),
                status=row["status"], reason=row["reason"],
                frequency_cpm=row["frequency_cpm"],
                diagnostic_best_cpm=row["diagnostic_best_cpm"],
                range_mm=row["range_mm"], duration_valid_s=row["duration_s"],
                support_tier=row["support_tier"],
            ))
        elapsed = time.time() - started
        message = (
            f"{holdout['holdout_id']} {condition} {number}/7 "
            f"completed {elapsed:.1f}s"
        )
        logs.append(message)
        print(message, flush=True)
    return own_frames, own_segments, own_cases, own_frequency, logs


def condition_summary(case_rows: list[dict]) -> list[dict]:
    metrics = (
        "target_corrected_range_px", "target_raw_range_px", "background_range_px",
        "radial_strain_range", "local_brightness_range", "global_brightness_range",
        "raw_background_corr", "corrected_background_corr",
        "corrected_local_brightness_corr", "corrected_global_brightness_corr",
        "corrected_radial_strain_corr", "neighbor_adjacent_corr",
        "mean_neighbor_dispersion_px", "mean_target_ncc", "max_target_fb_px",
        "mean_heldout_residual_px", "coordinate_rmse_px",
        "target_valid_fraction", "evidence_valid_fraction",
    )
    output = []
    for condition in frozen.CONDITIONS:
        rows = [row for row in case_rows if row["condition"] == condition]
        item = dict(
            condition=condition,
            expected_class=expected_class(condition), cases=len(rows),
            target_frequency_agreement=sum(
                row["target_frequency_agreement"] is True for row in rows),
            positive_target_total=sum(
                row["target_frequency_agreement"] != "" for row in rows),
            negative_region_false_pass=sum(
                int(row["negative_region_false_pass"]) for row in rows),
            negative_region_total=sum(
                int(row["negative_region_total"]) for row in rows),
        )
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows], float)
            values = values[np.isfinite(values)]
            item[f"median_{metric}"] = float(np.median(values)) if len(values) else ""
            item[f"min_{metric}"] = float(np.min(values)) if len(values) else ""
            item[f"max_{metric}"] = float(np.max(values)) if len(values) else ""
        output.append(item)
    return output


def plot_metrics(output: Path, summaries: list[dict]) -> None:
    conditions = [row["condition"] for row in summaries]
    labels = [
        "none", "global move", "global bright", "local bright",
        "local 0.3", "local 1.0", "mixed 0.3",
    ]
    specifications = (
        ("median_raw_background_corr", "raw vs background corr"),
        ("median_corrected_local_brightness_corr", "corrected vs local brightness corr"),
        ("median_radial_strain_range", "paired-wall strain range"),
        ("median_target_corrected_range_px", "corrected target range (px)"),
        ("median_coordinate_rmse_px", "coordinate RMSE (px)"),
        ("median_neighbor_adjacent_corr", "adjacent-point corr"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    x = np.arange(len(conditions))
    for axis, (field, title) in zip(axes.ravel(), specifications):
        values = [float(row[field]) if row[field] != "" else math.nan for row in summaries]
        axis.bar(x, values, color="#3b82f6")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Context evidence pilot: median across H02-H04")
    figure.savefig(output / "condition_metric_profiles.png", dpi=160)
    plt.close(figure)


def write_report(
    output: Path,
    case_rows: list[dict],
    summaries: list[dict],
    duration_s: int,
) -> None:
    positives = [
        row for row in case_rows if row["condition"] in frozen.POSITIVE_CONDITIONS]
    positive_agreement = sum(row["target_frequency_agreement"] is True for row in positives)
    negative_rows = [
        row for row in case_rows if row["condition"] in frozen.NEGATIVE_CONDITIONS]
    false_pass = sum(int(row["negative_region_false_pass"]) for row in negative_rows)
    negative_total = sum(int(row["negative_region_total"]) for row in negative_rows)
    summary_by_condition = {row["condition"]: row for row in summaries}
    global_motion = summary_by_condition["global_translation"]
    global_brightness = summary_by_condition["global_brightness"]
    local_brightness = summary_by_condition["local_brightness"]
    local_motion = summary_by_condition["local_motion_1p0"]
    lines = [
        "# 目标—周围组织证据旁路：阶段A小样本结果",
        "",
        f"H02、H03、H04；每例{duration_s}秒；7种已知真值条件。冻结追踪、刚性校正、区域支持和频率规则未修改。",
        "",
        "## 冻结基线复核",
        "",
        f"- 三类已知局部运动目标频率一致：{positive_agreement}/{len(positives)}。",
        f"- 四类阴性条件区域误报：{false_pass}/{negative_total}。本旁路尚未设阈值，因此没有声称减少误报。",
        f"- 所有坐标由冻结流程产生；旁路只读取和比较，不改写轨迹。",
        "",
        "## 关键描述性证据（3例中位数）",
        "",
        f"- 整体平移：目标原始位移与背景姿态相关系数 {float(global_motion['median_raw_background_corr']):.3f}；校正后目标范围 {float(global_motion['median_target_corrected_range_px']):.4f} px。",
        f"- 整体亮度：校正位移与局部亮度相关系数 {float(global_brightness['median_corrected_local_brightness_corr']):.3f}；径向应变范围 {float(global_brightness['median_radial_strain_range']):.5f}。",
        f"- 局部亮度：校正位移与局部亮度相关系数 {float(local_brightness['median_corrected_local_brightness_corr']):.3f}；径向应变范围 {float(local_brightness['median_radial_strain_range']):.5f}。",
        f"- 1.0 px真实局部运动：径向应变范围 {float(local_motion['median_radial_strain_range']):.5f}；相邻点相关系数 {float(local_motion['median_neighbor_adjacent_corr']):.3f}。",
        "",
        "## 解释边界",
        "",
        "- 本轮只检查连续证据能否分离竞争解释，没有根据结果临时设置分类阈值。",
        "- 相关性不是因果证据，也不是单独判定规则；证据冲突时应标为暂不可判断。",
        "- 径向距离变化只有在前后壁候选点具有可接受对应关系时才可解释为探索性形变，不能直接称为生理收缩。",
        "- 相邻点一致只能支持ROI活动或局部形变；没有稳定的时空传播证据时，不报告严格传播方向、速度或CF/FC。",
        "- 本轮是合成已知真值开发试验，不等同于真实患者视频准确率。",
        "",
        "## 文件",
        "",
        "- `frame_evidence.csv`：逐帧证据和有效掩码。",
        "- `segment_evidence.csv`：每个连续共同有效片段的证据。",
        "- `case_summary.csv`：病例×条件汇总。",
        "- `condition_summary.csv`：三例条件中位数及范围。",
        "- `regional_frequency_primary.csv`：冻结规则的六区域主要段结果。",
        "- `evidence_plots/condition_metric_profiles.png`：必要的汇总图。",
        "- `protocol.json` 和 `实际运行日志.txt`：配置、哈希和运行记录。",
    ]
    (output / "结果说明_中文.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Output exists; refusing to overwrite")
    args.out.mkdir(parents=True)
    plots = args.out / "evidence_plots"
    plots.mkdir()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    catalogue = frozen.catalogue(args.videos, args.existing_results)
    resolved = frozen.resolve_locked(protocol, catalogue)
    requested = tuple(part.strip() for part in args.cases.split(",") if part.strip())
    selected = [row for case_id in requested for row in resolved
                if row["holdout_id"] == case_id]
    if tuple(row["holdout_id"] for row in selected) != requested:
        raise ValueError("Requested anonymous cases were not resolved")
    if args.duration <= 0 or args.duration > frozen.MAX_SECONDS:
        raise ValueError(f"Duration must be within 1..{frozen.MAX_SECONDS} seconds")

    recorded_protocol = dict(
        experiment="V1.3 target-surrounding context evidence pilot",
        date="2026-10-28", stage="A_known_truth_descriptive",
        cases=list(requested), duration_s=args.duration,
        conditions=list(frozen.CONDITIONS), synthetic_fps=frozen.FPS,
        classification_thresholds={},
        evidence_mode="descriptive_only_no_outcome_reclassification",
        evidence_channels=[
            "global_pose_projection", "same_wall_neighbor_consistency",
            "paired_wall_distance_strain", "local_and_global_brightness",
            "tracking_and_registration_quality",
        ],
        missingness=(
            "failed frames remain empty; no interpolation; no joining across epoch changes"
        ),
        locked_protocol_sha256=sha256_file(args.protocol),
        frozen_source_hashes={name: sha256_file(CODE / name) for name in FROZEN_FILES},
        sidecar_script_sha256=sha256_file(Path(__file__)),
        privacy="only anonymous holdout IDs are written",
    )
    (args.out / "protocol.json").write_text(
        json.dumps(recorded_protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    all_frames: list[dict] = []
    all_segments: list[dict] = []
    all_cases: list[dict] = []
    all_frequency: list[dict] = []
    log_lines = [
        f"started_local=2026-10-28",
        f"cases={','.join(requested)}",
        f"duration_s={args.duration}",
        "classification_thresholds=none",
    ]
    started = time.time()
    for holdout in selected:
        frames, segments, cases, frequency_rows, logs = process_case(
            holdout, args.duration)
        all_frames.extend(frames)
        all_segments.extend(segments)
        all_cases.extend(cases)
        all_frequency.extend(frequency_rows)
        log_lines.extend(logs)

    summaries = condition_summary(all_cases)
    write_csv(args.out / "frame_evidence.csv", all_frames)
    write_csv(args.out / "segment_evidence.csv", all_segments)
    write_csv(args.out / "case_summary.csv", all_cases)
    write_csv(args.out / "condition_summary.csv", summaries)
    write_csv(args.out / "regional_frequency_primary.csv", all_frequency)
    plot_metrics(plots, summaries)
    write_report(args.out, all_cases, summaries, args.duration)
    elapsed = time.time() - started
    log_lines.extend([
        f"completed_seconds={elapsed:.3f}",
        f"opencv_version={cv2.__version__}",
        f"numpy_version={np.__version__}",
        "frozen_sources_modified=false",
        "historical_results_overwritten=false",
        "patient_identifiers_written=false",
    ])
    (args.out / "实际运行日志.txt").write_text(
        "\n".join(log_lines) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--existing-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default="H02,H03,H04")
    parser.add_argument("--duration", type=int, default=60)
    run(parser.parse_args())
