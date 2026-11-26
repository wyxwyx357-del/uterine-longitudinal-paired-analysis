"""Frozen full-video local-motion frequency runner.

This is a new wrapper around the frozen tracking modules.  It preserves every
technically valid raw dominant frequency and adds the paired-wall deformation
support rule confirmed on the independent U01-U08 holdout experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

import run_real_context_evidence as base


GRID_CPM = np.round(np.arange(0.5, 5.0001, 0.05), 8)
RANDOM_SEED = 20261028
STRAIN_RANGE_THRESHOLD = 0.0045
FREQUENCY_TOLERANCE_CPM = 0.25


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a stable header even when every requested case failed."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def union_fieldnames(rows: list[dict], fallback: list[str]) -> list[str]:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields or fallback


def read_case_manifest(path: Path) -> list[dict]:
    required = {"case_id", "video", "tracking", "nodes"}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "Case manifest is missing columns: " + ", ".join(sorted(missing))
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Case manifest contains no cases")
    case_ids = [row["case_id"].strip() for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("Every manifest row must have a non-empty case_id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Case manifest contains duplicate case_id values")
    base_dir = path.resolve().parent
    resolved = []
    for source, case_id in zip(rows, case_ids):
        row = dict(source)
        row["case_id"] = case_id
        for field in ("video", "tracking", "nodes"):
            value = Path(row[field].strip())
            row[field] = value if value.is_absolute() else base_dir / value
        resolved.append(row)
    return resolved


def prepare_case(source: dict) -> dict:
    case_id = source["case_id"]
    for field in ("video", "tracking", "nodes"):
        if not source[field].is_file():
            raise FileNotFoundError(f"{case_id}: {field} not found: {source[field]}")
    frame_count, fps, width, height = base.video_metadata(source["video"])
    if fps <= 0 or frame_count < 2:
        raise ValueError(f"{case_id}: invalid video timing")
    if (width, height) != (1260, 910):
        raise ValueError(
            f"{case_id}: expected 1260x910 source video, found {width}x{height}"
        )
    with np.load(source["tracking"], allow_pickle=False) as archive:
        tracks = archive["tracks"]
        if tracks.ndim != 3 or tracks.shape[2] != 2:
            raise ValueError(f"{case_id}: tracking array must have shape (T, N, 2)")
        tracking_frames = int(tracks.shape[0])
    if tracking_frames < 1:
        raise ValueError(f"{case_id}: tracking file contains no first-frame seed")
    spacing_text = source.get("mm_per_processed_pixel", "").strip()
    spacing = math.nan
    if spacing_text:
        spacing = float(spacing_text)
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"{case_id}: mm_per_processed_pixel must be positive")
    node_rows = json.loads(source["nodes"].read_text(encoding="utf-8-sig"))
    anterior = [row for row in node_rows if row.get("rail_name") == "anterior_wall"]
    posterior = [row for row in node_rows if row.get("rail_name") == "posterior_wall"]
    if len(anterior) != len(posterior) or len(anterior) < 6:
        raise ValueError(f"{case_id}: unmatched or insufficient anterior/posterior nodes")
    return dict(
        anonymous_id=case_id,
        holdout_id=case_id,
        key=hashlib.sha256(case_id.encode("utf-8")).hexdigest(),
        video=source["video"],
        tracking=source["tracking"],
        nodes=source["nodes"],
        wall_points=len(anterior),
        spacing=spacing,
        frame_count=frame_count,
        tracking_frames=tracking_frames,
        fps=fps,
        available_s=(frame_count - 1) / fps,
        frame_count_match=frame_count == tracking_frames,
    )


def robust_background_view(gray: np.ndarray, fan_mask: np.ndarray) -> np.ndarray:
    """Remove framewise affine brightness drift inside the ultrasound fan."""
    valid = fan_mask & (gray > 12)
    if int(valid.sum()) < 100:
        return gray.copy()
    q25, median, q75 = np.percentile(gray[valid], [25.0, 50.0, 75.0])
    scale = max(float(q75 - q25), 1.0)
    output = np.zeros_like(gray)
    values = 128.0 + 48.0 * (gray.astype(float) - float(median)) / scale
    output[valid] = np.clip(values[valid], 16.0, 240.0).astype(np.uint8)
    return output


def track_normalized_background(row: dict, duration_s: float) -> tuple:
    gray0, initial, anterior_axes, posterior_axes = base.frozen.load_background(row)
    fps = float(row["fps"])
    frame_limit = int(math.floor(duration_s * fps + 1e-9)) + 1
    time_s = np.arange(frame_limit, dtype=float) / fps
    n, count = frame_limit, len(initial)
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
    point_brightness = np.full((n, count), np.nan, float)
    global_brightness = np.full(n, np.nan, float)
    heldout_raw = np.full(n, np.nan, float)
    heldout_corrected = np.full(n, np.nan, float)
    training_inliers = np.zeros(n, int)
    heldout_count = np.zeros(n, int)
    accumulator = np.eye(3)
    current_epoch = 0
    fan_mask = gray0 > 12
    cap = cv2.VideoCapture(str(row["video"]))
    previous = processed_previous = background_previous = None
    try:
        for index in range(n):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"{row['anonymous_id']} missing frame {index}")
            gray = cv2.cvtColor(
                cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
            if gray.shape != (455, 630):
                raise ValueError("Frozen background ROI requires 630x455 processed frames")
            processed = base.local_contrast(gray)
            background = robust_background_view(gray, fan_mask)
            stats = {}
            if index == 0:
                positions[0] = raw_positions[0] = initial
                valid[0] = raw_valid[0] = True
                corrected[0] = initial
                pose_valid[0] = True
                poses[0] = accumulator
            else:
                live = np.flatnonzero(valid[index - 1])
                if len(live):
                    found, good, scores, fb = base.track_step(
                        processed_previous, processed, positions[index - 1, live])
                    target_ncc[index, live] = scores
                    target_fb[index, live] = fb
                    kept = live[good]
                    positions[index, kept] = found[good]
                    valid[index, kept] = True
                raw_live = np.flatnonzero(raw_valid[index - 1])
                if len(raw_live):
                    found, good, _, _ = base.track_step(
                        previous, gray, raw_positions[index - 1, raw_live])
                    kept = raw_live[good]
                    raw_positions[index, kept] = found[good]
                    raw_valid[index, kept] = True
                cv2.setRNGSeed(RANDOM_SEED)
                matrix, stats = base.rigid_step(
                    background_previous, background, raw_positions[index - 1, raw_live]
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
            for column in np.flatnonzero(valid[index]):
                point_brightness[index, column] = base.patch_median(
                    gray, positions[index, column])
            global_brightness[index] = float(np.median(gray[fan_mask]))
            heldout_raw[index] = stats.get("heldout_raw_median_px", math.nan)
            heldout_corrected[index] = stats.get(
                "heldout_corrected_median_px", math.nan)
            training_inliers[index] = stats.get("training_inliers", 0)
            heldout_count[index] = stats.get("heldout_count", 0)
            previous = gray
            processed_previous = processed
            background_previous = background
            if index and index % 300 == 0:
                print(row["anonymous_id"], "B", index, n, flush=True)
    finally:
        cap.release()
    tracked = dict(
        time_s=time_s, positions=positions, raw_positions=raw_positions,
        valid=valid, raw_valid=raw_valid, corrected=corrected,
        pose_valid=pose_valid, epoch=epoch, poses=poses,
        target_ncc=target_ncc, target_fb=target_fb,
        point_brightness=point_brightness,
        global_brightness=global_brightness,
        heldout_raw=heldout_raw, heldout_corrected=heldout_corrected,
        training_inliers=training_inliers, heldout_count=heldout_count,
    )
    return tracked, initial, anterior_axes, posterior_axes


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    good = np.isfinite(a) & np.isfinite(b)
    if int(good.sum()) < 20 or np.std(a[good]) < 1e-9 or np.std(b[good]) < 1e-9:
        return math.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


def fit_unified_frequency(time_s: np.ndarray, signal: np.ndarray) -> dict:
    if len(time_s) < 20:
        return dict(status="missing", reason="technical_segment_too_short")
    tt = time_s - time_s.mean()
    trend_x = np.column_stack((np.ones(len(tt)), tt))
    trend = trend_x @ np.linalg.lstsq(trend_x, signal, rcond=None)[0]
    residual = signal - trend
    energy = float(residual @ residual)
    if energy / len(signal) < 1e-12:
        return dict(status="missing", reason="technical_nearly_constant")
    scores = []
    for frequency in GRID_CPM:
        phase = 2.0 * np.pi * tt * frequency / 60.0
        design = np.column_stack((trend_x, np.sin(phase), np.cos(phase)))
        fitted = design @ np.linalg.lstsq(design, signal, rcond=None)[0]
        scores.append(np.clip(1.0 - float(((signal - fitted) ** 2).sum()) / energy, 0.0, 1.0))
    scores = np.asarray(scores)
    best_index = int(np.argmax(scores))
    best = float(GRID_CPM[best_index])
    period_s = 60.0 / best
    lag = np.interp(time_s + period_s, time_s, residual, left=np.nan, right=np.nan)
    half = np.interp(time_s + period_s / 2.0, time_s, residual, left=np.nan, right=np.nan)
    return dict(
        status="valid", reason="unified_fixed_band_dominant_frequency",
        dominant_frequency_cpm=best, period_s=period_s,
        score=float(scores[best_index]),
        one_period_correlation=correlation(residual, lag),
        half_period_correlation=correlation(residual, half),
        search_boundary=bool(best_index in (0, len(GRID_CPM) - 1)),
        observed_cycles=float(best * (time_s[-1] - time_s[0]) / 60.0),
    )


def analyse_branch(
    case_id: str,
    branch: str,
    tracked: dict,
    initial: np.ndarray,
    anterior_axes: np.ndarray,
    posterior_axes: np.ndarray,
    spacing: float,
) -> list[dict]:
    all_axes = np.vstack((anterior_axes, posterior_axes))
    count = len(anterior_axes)
    regions = dict(zip(base.frozen.REGION_NAMES, np.array_split(np.arange(count), 3)))
    rows = []
    for wall, offset in (("anterior", 0), ("posterior", count)):
        for region, local_indices in regions.items():
            columns = local_indices + offset
            arrays = base.region_arrays(tracked, initial, all_axes, local_indices, columns)
            raw_valid = (
                arrays["region_valid"]
                & np.isfinite(arrays["corrected_mean"])
            )
            raw_runs = base.contiguous_runs(raw_valid, tracked["epoch"])
            if not raw_runs:
                rows.append(dict(
                    case_id=case_id, branch=branch, wall=wall, region=region,
                    status="missing", reason="no_continuous_technical_valid_segment",
                    dominant_frequency_cpm="", actual_video_duration_s=tracked["time_s"][-1],
                    continuous_valid_duration_s=0.0,
                    paired_wall_strain_range="",
                    support_displacement_frequency_status="missing",
                    support_displacement_frequency_cpm="",
                    paired_wall_strain_frequency_cpm="",
                    paired_wall_continuous_valid_duration_s=0.0,
                    support_common_valid_duration_s=0.0,
                    displacement_strain_frequency_difference_cpm="",
                    deformation_supported=False,
                    deformation_support_reason="raw_frequency_missing",
                    mean_heldout_residual_px="",
                    background_to_raw_range_ratio="",
                ))
                continue
            start, stop = max(
                raw_runs,
                key=lambda own: tracked["time_s"][own[1] - 1] - tracked["time_s"][own[0]],
            )
            own = slice(start, stop)
            fit = fit_unified_frequency(
                tracked["time_s"][own], arrays["corrected_mean"][own])
            support_common_valid = (
                raw_valid
                & arrays["pair_valid"]
                & np.isfinite(arrays["radial_strain"])
            )
            support_runs = base.contiguous_runs(
                support_common_valid, tracked["epoch"]
            )
            if support_runs:
                support_start, support_stop = max(
                    support_runs,
                    key=lambda run: (
                        tracked["time_s"][run[1] - 1] - tracked["time_s"][run[0]]
                    ),
                )
                support_own = slice(support_start, support_stop)
                support_displacement_fit = fit_unified_frequency(
                    tracked["time_s"][support_own],
                    arrays["corrected_mean"][support_own],
                )
                strain_fit = fit_unified_frequency(
                    tracked["time_s"][support_own],
                    arrays["radial_strain"][support_own],
                )
                strain_range = float(np.ptp(arrays["radial_strain"][support_own]))
                support_duration = float(
                    tracked["time_s"][support_stop - 1]
                    - tracked["time_s"][support_start]
                )
            else:
                support_displacement_fit = dict(
                    status="missing", reason="no_common_support_valid_segment"
                )
                strain_fit = dict(status="missing", reason="no_paired_wall_valid_segment")
                strain_range = math.nan
                support_duration = 0.0
            frequency_difference = (
                abs(float(support_displacement_fit["dominant_frequency_cpm"])
                    - float(strain_fit["dominant_frequency_cpm"]))
                if support_displacement_fit["status"] == strain_fit["status"] == "valid"
                else math.nan
            )
            deformation_supported = bool(
                fit["status"] == "valid"
                and support_displacement_fit["status"] == strain_fit["status"] == "valid"
                and strain_range >= STRAIN_RANGE_THRESHOLD
                and frequency_difference <= FREQUENCY_TOLERANCE_CPM
            )
            support_reasons = []
            if fit["status"] != "valid":
                support_reasons.append("raw_frequency_missing")
            if support_displacement_fit["status"] != "valid":
                support_reasons.append("support_displacement_frequency_missing")
            if strain_fit["status"] != "valid":
                support_reasons.append("strain_frequency_missing")
            if not np.isfinite(strain_range):
                support_reasons.append("paired_wall_segment_missing")
            elif strain_range < STRAIN_RANGE_THRESHOLD:
                support_reasons.append("strain_range_below_0p0045")
            if (np.isfinite(frequency_difference)
                    and frequency_difference > FREQUENCY_TOLERANCE_CPM):
                support_reasons.append("frequency_difference_above_0p25")
            metrics = base.segment_metrics(case_id, wall, region, tracked, arrays)
            primary = max(metrics, key=lambda item: item["duration_s"]) if metrics else {}
            row = dict(
                case_id=case_id, branch=branch, wall=wall, region=region,
                actual_video_duration_s=float(tracked["time_s"][-1]),
                continuous_valid_duration_s=float(
                    tracked["time_s"][stop - 1] - tracked["time_s"][start]),
                valid_frames=stop - start,
                frequency_range_cpm="0.5-5.0",
                range_mm=(
                    float(np.ptp(arrays["corrected_mean"][own]) * spacing)
                    if math.isfinite(spacing) else ""
                ),
                pose_valid_fraction=float(tracked["pose_valid"].mean()),
                epoch_count=int(tracked["epoch"].max() + 1),
                local_brightness_fractional_range=primary.get(
                    "local_brightness_fractional_range", ""),
                radial_strain_range=primary.get("radial_strain_range", ""),
                paired_wall_strain_range=(
                    strain_range if np.isfinite(strain_range) else ""
                ),
                support_displacement_frequency_status=(
                    support_displacement_fit["status"]
                ),
                support_displacement_frequency_cpm=(
                    support_displacement_fit.get("dominant_frequency_cpm", "")
                ),
                paired_wall_continuous_valid_duration_s=support_duration,
                support_common_valid_duration_s=support_duration,
                paired_wall_strain_frequency_status=strain_fit["status"],
                paired_wall_strain_frequency_cpm=strain_fit.get(
                    "dominant_frequency_cpm", ""),
                displacement_strain_frequency_difference_cpm=(
                    frequency_difference if np.isfinite(frequency_difference) else ""),
                deformation_supported=deformation_supported,
                deformation_support_reason=(
                    "pass" if deformation_supported else ";".join(support_reasons)),
                background_to_raw_range_ratio=primary.get(
                    "background_to_raw_range_ratio", ""),
                corrected_local_brightness_corr=primary.get(
                    "corrected_local_brightness_corr", ""),
                adjacent_point_corr=primary.get("adjacent_point_corr", ""),
                mean_heldout_residual_px=primary.get(
                    "mean_heldout_residual_px", ""),
            )
            row.update(fit)
            row.setdefault("dominant_frequency_cpm", "")
            rows.append(row)
    return rows


def missing_branch_rows(
    case_id: str,
    branch: str,
    duration_s: float,
    reason: str,
) -> list[dict]:
    """Keep the six-row diagnostic shape when an optional branch fails."""
    return [
        dict(
            case_id=case_id, branch=branch, wall=wall, region=region,
            status="missing", reason=reason, dominant_frequency_cpm="",
            actual_video_duration_s=duration_s,
            continuous_valid_duration_s=0.0,
            deformation_supported=False,
            support_displacement_frequency_cpm="",
            mean_heldout_residual_px="",
            background_to_raw_range_ratio="",
        )
        for wall in ("anterior", "posterior")
        for region in base.frozen.REGION_NAMES
    ]


def summarise_patients(region_rows: list[dict]) -> list[dict]:
    patient_rows = []
    for case_id in sorted({row["case_id"] for row in region_rows}):
        for branch in ("A_raw_background", "B_brightness_normalized_background"):
            own = [row for row in region_rows
                   if row["case_id"] == case_id and row["branch"] == branch]
            frequencies = [float(row["dominant_frequency_cpm"]) for row in own
                           if row["status"] == "valid"]
            supported = [float(row["support_displacement_frequency_cpm"]) for row in own
                         if row["status"] == "valid"
                         and row.get("deformation_supported") is True
                         and row.get("support_displacement_frequency_cpm") != ""]
            patient_rows.append(dict(
                case_id=case_id, branch=branch,
                valid_region_count=len(frequencies), total_region_count=6,
                patient_median_dominant_frequency_cpm=(
                    float(np.median(frequencies)) if frequencies else ""),
                deformation_supported_region_count=len(supported),
                patient_median_deformation_supported_frequency_cpm=(
                    float(np.median(supported)) if supported else ""),
                actual_video_duration_s=own[0]["actual_video_duration_s"],
                min_continuous_valid_duration_s=min(
                    float(row["continuous_valid_duration_s"]) for row in own),
                max_continuous_valid_duration_s=max(
                    float(row["continuous_valid_duration_s"]) for row in own),
            ))
    return patient_rows


def run(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise ValueError("Output exists; choose a new directory")
    args.out.mkdir(parents=True)
    manifest_rows = read_case_manifest(args.case_manifest)
    selected = []
    region_rows = []
    case_status_rows = []
    logs = [
        "rule=full_video_fixed_0.5_to_5.0_cpm_all_technical_valid_regions",
        "branch_A=raw_gray_background_registration",
        "branch_B=median_IQR_normalized_background_registration",
        "target_tracking_unchanged=true",
        "final_curve_smoothing=false",
        "deformation_support_comparison=longest_same_common_valid_segment",
        "duration_source=full_video_frame_count",
        "scientific_core_parameters_modified=false",
    ]
    started = time.time()
    for source in manifest_rows:
        case_id = source["case_id"]
        try:
            row = prepare_case(source)
            selected.append(row)
        except Exception as exc:
            case_status_rows.append(dict(
                case_id=case_id, status="failed_input", reason=str(exc),
                actual_video_duration_s="", frame_count_match="",
                formal_A_status="not_run", formal_A_reason="",
                diagnostic_B_status="not_run", diagnostic_B_reason="",
            ))
            logs.append(f"{case_id}_failed_input={type(exc).__name__}:{exc}")
            continue
        duration_s = float(row["available_s"])
        case_started = time.time()
        try:
            print(row["anonymous_id"], "A start", flush=True)
            tracked_a, initial_a, anterior_a, posterior_a = base.track_real_case(
                row, duration_s)
            a_region_rows = analyse_branch(
                row["anonymous_id"], "A_raw_background", tracked_a, initial_a,
                anterior_a, posterior_a, row["spacing"])
        except Exception as exc:
            case_status_rows.append(dict(
                case_id=case_id, status="failed_processing", reason=str(exc),
                actual_video_duration_s=duration_s,
                frame_count_match=row["frame_count_match"],
                formal_A_status="failed", formal_A_reason=str(exc),
                diagnostic_B_status="not_run", diagnostic_B_reason="",
            ))
            logs.append(f"{case_id}_failed_processing={type(exc).__name__}:{exc}")
            continue

        b_status, b_reason = "complete", ""
        try:
            print(row["anonymous_id"], "B start", flush=True)
            tracked_b, initial_b, anterior_b, posterior_b = track_normalized_background(
                row, duration_s)
            b_region_rows = analyse_branch(
                row["anonymous_id"], "B_brightness_normalized_background", tracked_b,
                initial_b, anterior_b, posterior_b, row["spacing"])
        except Exception as exc:
            b_status = "failed_diagnostic"
            b_reason = str(exc)
            b_region_rows = missing_branch_rows(
                case_id, "B_brightness_normalized_background", duration_s,
                f"diagnostic_branch_failed:{type(exc).__name__}:{exc}",
            )
            logs.append(f"{case_id}_diagnostic_B_failed={type(exc).__name__}:{exc}")

        region_rows.extend(a_region_rows)
        region_rows.extend(b_region_rows)
        elapsed = time.time() - case_started
        case_status_rows.append(dict(
            case_id=case_id, status="complete", reason="",
            actual_video_duration_s=duration_s,
            frame_count_match=row["frame_count_match"],
            formal_A_status="complete", formal_A_reason="",
            diagnostic_B_status=b_status, diagnostic_B_reason=b_reason,
        ))
        logs.append(f"{row['anonymous_id']}_completed_seconds={elapsed:.3f}")
        print(row["anonymous_id"], f"completed {elapsed:.1f}s", flush=True)

    patient_rows = summarise_patients(region_rows)

    comparison_rows = []
    for case_id in sorted({row["case_id"] for row in region_rows}):
        for wall in ("anterior", "posterior"):
            for region in base.frozen.REGION_NAMES:
                a = next(row for row in region_rows if row["case_id"] == case_id
                         and row["branch"] == "A_raw_background"
                         and row["wall"] == wall and row["region"] == region)
                b = next(row for row in region_rows if row["case_id"] == case_id
                         and row["branch"] == "B_brightness_normalized_background"
                         and row["wall"] == wall and row["region"] == region)
                comparison_rows.append(dict(
                    case_id=case_id, wall=wall, region=region,
                    A_frequency_cpm=a["dominant_frequency_cpm"],
                    B_frequency_cpm=b["dominant_frequency_cpm"],
                    absolute_frequency_change_cpm=(
                        abs(float(a["dominant_frequency_cpm"])
                            - float(b["dominant_frequency_cpm"]))
                        if a["status"] == b["status"] == "valid" else ""),
                    A_score=a.get("score", ""), B_score=b.get("score", ""),
                    A_heldout_residual_px=a["mean_heldout_residual_px"],
                    B_heldout_residual_px=b["mean_heldout_residual_px"],
                    A_background_to_raw_ratio=a["background_to_raw_range_ratio"],
                    B_background_to_raw_ratio=b["background_to_raw_range_ratio"],
                ))

    region_fields = union_fieldnames(region_rows, [
        "case_id", "branch", "wall", "region", "status", "reason",
        "dominant_frequency_cpm", "deformation_supported",
    ])
    patient_fields = union_fieldnames(patient_rows, [
        "case_id", "branch", "valid_region_count", "total_region_count",
    ])
    comparison_fields = union_fieldnames(comparison_rows, [
        "case_id", "wall", "region", "A_frequency_cpm", "B_frequency_cpm",
    ])
    write_csv_rows(args.out / "case_batch_status.csv", case_status_rows,
                   list(case_status_rows[0]))
    write_csv_rows(args.out / "region_unified_frequency.csv", region_rows, region_fields)
    write_csv_rows(args.out / "patient_unified_frequency.csv", patient_rows, patient_fields)
    write_csv_rows(args.out / "background_normalization_ab.csv", comparison_rows,
                   comparison_fields)
    protocol = dict(
        experiment="batch-ready full-video local-motion frequency with deformation support",
        fixed_frequency_grid_cpm=[0.5, 5.0, 0.05],
        full_available_video=True,
        regional_outputs="anterior/posterior x cervical/middle/fundal",
        patient_summary="median of all technically valid regional dominant frequencies",
        deformation_supported_patient_summary=(
            "median common-segment support displacement frequency among supported regions"
        ),
        frequency_hidden_by_repetition_rules=False,
        raw_frequency_retained_when_deformation_unsupported=True,
        missing_frequency_is_zero=False,
        case_manifest_sha256=base.sha256_file(args.case_manifest),
        branch_roles=dict(
            A_raw_background="frozen baseline; deformation rule validated here",
            B_brightness_normalized_background=(
                "diagnostic A/B branch; deformation rule not independently frozen"
            ),
        ),
        deformation_support_rule=dict(
            comparison_segment=(
                "longest same segment where raw displacement and paired-wall "
                "strain are both valid without crossing epoch"
            ),
            paired_wall_strain_range_at_least=STRAIN_RANGE_THRESHOLD,
            displacement_strain_frequency_difference_cpm_at_most=(
                FREQUENCY_TOLERANCE_CPM),
            validation="independent U01-U08 known-truth holdout",
        ),
        cases=[dict(
            anonymous_id=row["anonymous_id"],
            source_key_sha256=row["key"], fps=row["fps"],
            available_s=row["available_s"], wall_points_per_side=row["wall_points"],
            frame_count_match=row["frame_count_match"],
            mm_per_processed_pixel=(
                row["spacing"] if math.isfinite(row["spacing"]) else None
            ),
        ) for row in selected],
        frozen_source_hashes={name: base.sha256_file(base.CODE / name)
                              for name in base.FROZEN_FILES},
        script_sha256=base.sha256_file(Path(__file__)),
    )
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    logs.extend([
        f"completed_seconds={time.time() - started:.3f}",
        f"region_rows={len(region_rows)}",
        f"patient_rows={len(patient_rows)}",
        f"cases_requested={len(manifest_rows)}",
        f"cases_complete={sum(row['status'] == 'complete' for row in case_status_rows)}",
        f"cases_failed={sum(row['status'] != 'complete' for row in case_status_rows)}",
        f"diagnostic_B_failed={sum(row['diagnostic_B_status'] == 'failed_diagnostic' for row in case_status_rows)}",
        "historical_results_overwritten=false",
        "patient_identifiers_written=false",
    ])
    (args.out / "实际运行日志.txt").write_text(
        "\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps(patient_rows, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(row["status"] == "complete" for row in case_status_rows) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))