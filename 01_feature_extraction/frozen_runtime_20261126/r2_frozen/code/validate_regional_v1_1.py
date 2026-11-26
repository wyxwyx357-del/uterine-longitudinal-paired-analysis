"""Prespecified holdout controls for the V1.1 regional frequency layer.

Real ultrasound first-frame textures and saved first-frame wall seeds are used
only as backgrounds/initial positions.  All subsequent frames are generated
with known geometry or known no-geometry controls.  Production tracking,
rigid correction, support segmentation, and frequency fitting are imported
unchanged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ab_block_tracking import track_step
from contrast import local_contrast
from rigid_correction import step as rigid_step
from run_regional_frequency import (
    REGION_NAMES, _fit_fixed_support_v1_1, _normalise_axes)


FPS = 10
MAX_SECONDS = 75
DURATIONS_S = (30, 45, 60, 75)
SIGMA_PX = 12.0
REGION_ASSIGNMENTS = (
    ("anterior", "cervical"), ("anterior", "middle"),
    ("anterior", "fundal"), ("posterior", "cervical"),
    ("posterior", "middle"), ("posterior", "fundal"),
) * 2
FREQUENCIES_CPM = (2.0, 3.0, 4.0, 3.0, 4.0, 2.0,
                   4.0, 2.0, 3.0, 2.0, 3.0, 4.0)
CONDITIONS = (
    "no_motion", "global_translation", "global_brightness",
    "local_brightness", "local_motion_0p3", "local_motion_1p0",
    "mixed_local_0p3",
)
POSITIVE_CONDITIONS = {
    "local_motion_0p3": 0.3,
    "local_motion_1p0": 1.0,
    "mixed_local_0p3": 0.3,
}
NEGATIVE_CONDITIONS = set(CONDITIONS) - set(POSITIVE_CONDITIONS)
EXCLUDED_NUMERIC_IDS = {210, 211, 217, 257, 345, 762, 779, 793, 809, 1049}
MATCH_FIELDS = (
    "dicom_mp4_match_status", "frame_mapping_status",
    "coordinate_mapping_status", "timing_match_status",
    "calibration_region_status",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_key(case_name: str) -> str:
    return sha256_bytes(case_name.encode("utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_single(folder: Path, pattern: str) -> Path:
    matches = list(folder.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {pattern} under {folder}, found {len(matches)}")
    return matches[0]


def numeric_id(case_name: str) -> int | None:
    match = re.search(r"_(\d+)_\d{8}$", case_name)
    return int(match.group(1)) if match else None


def catalogue(videos: Path, existing: Path) -> list[dict]:
    rows = []
    for folder in existing.iterdir():
        if not folder.is_dir() or numeric_id(folder.name) in EXCLUDED_NUMERIC_IDS:
            continue
        video = videos / f"{folder.name}.mp4"
        if not video.is_file():
            continue
        try:
            tracking = find_single(
                folder / "01_壁追踪", "*_new_patient_tracking_sparse_anchor_tracks.npz")
            nodes = find_single(
                folder / "01_壁追踪", "*_new_patient_tracking_nodes.json")
            calibration = find_single(
                folder / "03_DICOM物理标定", "*_dicom_curvature_v1.npz")
            with np.load(tracking, allow_pickle=False) as z:
                tracks = z["tracks"]
                if tracks.ndim != 3 or tracks.shape[2] != 2:
                    raise ValueError("invalid tracks")
            node_rows = json.loads(nodes.read_text(encoding="utf-8-sig"))
            anterior = sorted(
                (r for r in node_rows if r.get("rail_name") == "anterior_wall"),
                key=lambda r: r["section_order"])
            posterior = sorted(
                (r for r in node_rows if r.get("rail_name") == "posterior_wall"),
                key=lambda r: r["section_order"])
            if len(anterior) != len(posterior) or len(anterior) < 6:
                raise ValueError("unmatched wall points")
            ids = np.array([r["point_id"] for r in anterior + posterior], dtype=int)
            if ids.max() >= tracks.shape[1] or not np.isfinite(tracks[0, ids]).all():
                raise ValueError("invalid first-frame seeds")
            with np.load(calibration, allow_pickle=False) as z:
                if any(str(z[key].item()) != "matched" for key in MATCH_FIELDS):
                    raise ValueError("calibration match failed")
                if not bool(z["physical_curvature_available"].item()):
                    raise ValueError("physical calibration unavailable")
                dx = float(z["dicom_delta_x_mm_per_pixel"].item())
                dy = float(z["dicom_delta_y_mm_per_pixel"].item())
                transform = z["analysis_to_dicom_transform"].astype(float)
            physical_map = np.diag([dx, dy]) @ transform[:2, :2]
            scales = np.linalg.svd(physical_map, compute_uv=False)
            if scales.min() <= 0 or not np.allclose(scales[0], scales[1], rtol=1e-5):
                raise ValueError("anisotropic processed spacing")
            cap = cv2.VideoCapture(str(video))
            ok, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if not ok or (width, height) != (1260, 910):
                raise ValueError("invalid first frame")
            rows.append(dict(
                name=folder.name, key=source_key(folder.name), video=video,
                tracking=tracking, nodes=nodes, calibration=calibration,
                wall_points=len(anterior), spacing=float(scales.mean())))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return rows


def lock_protocol(catalogue_rows: list[dict], output: Path, count: int = 12) -> dict:
    ordered = sorted(
        catalogue_rows,
        key=lambda row: sha256_bytes(("V1.1-holdout-20261027|" + row["key"]).encode()))
    if len(ordered) < count:
        raise ValueError(f"Only {len(ordered)} eligible holdout backgrounds")
    selected = ordered[:count]
    cases = []
    for i, row in enumerate(selected):
        wall, region = REGION_ASSIGNMENTS[i]
        cases.append(dict(
            holdout_id=f"H{i + 1:02d}", source_key_sha256=row["key"],
            wall_points_per_side=row["wall_points"],
            mm_per_processed_pixel=row["spacing"], target_wall=wall,
            target_region=region, true_frequency_cpm=FREQUENCIES_CPM[i]))
    protocol = dict(
        protocol="V1.1 regional known-truth/no-geometry controls and duration sensitivity",
        locked_before_outcome_review=True,
        selection=("SHA256 deterministic ordering among structurally eligible cases; "
                   "excluded current development/test numeric IDs"),
        background_use="real video first frame resized 0.5; no old trajectory/QC reused",
        synthetic_fps=FPS, maximum_seconds=MAX_SECONDS,
        duration_prefixes_s=list(DURATIONS_S), conditions=list(CONDITIONS),
        nominal_local_motion_processed_px={
            key: value for key, value in POSITIVE_CONDITIONS.items()},
        global_translation_processed_px=1.0,
        brightness_modulation_fraction=0.10,
        local_gaussian_sigma_processed_px=SIGMA_PX,
        frequency_agreement_tolerance_cpm=0.25,
        cases=cases)
    output.mkdir(parents=True, exist_ok=False)
    (output / "locked_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    return protocol


def resolve_locked(protocol: dict, catalogue_rows: list[dict]) -> list[dict]:
    lookup = {row["key"]: row for row in catalogue_rows}
    resolved = []
    for locked in protocol["cases"]:
        source = lookup.get(locked["source_key_sha256"])
        if source is None:
            raise ValueError(f"Locked source unavailable: {locked['holdout_id']}")
        resolved.append({**source, **locked})
    return resolved


def load_background(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(row["video"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read {row['holdout_id']}")
    gray = cv2.cvtColor(
        cv2.resize(frame, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY)
    with np.load(row["tracking"], allow_pickle=False) as z:
        tracks = z["tracks"]
    node_rows = json.loads(row["nodes"].read_text(encoding="utf-8-sig"))
    anterior = sorted(
        (r for r in node_rows if r.get("rail_name") == "anterior_wall"),
        key=lambda r: r["section_order"])
    posterior = sorted(
        (r for r in node_rows if r.get("rail_name") == "posterior_wall"),
        key=lambda r: r["section_order"])
    anterior_ids = np.array([r["point_id"] for r in anterior], dtype=int)
    posterior_ids = np.array([r["point_id"] for r in posterior], dtype=int)
    initial = tracks[0, np.r_[anterior_ids, posterior_ids]].astype(np.float32)
    delta = initial[len(anterior_ids):].astype(float) - initial[:len(anterior_ids)]
    anterior_axes = _normalise_axes(delta, len(anterior_ids))
    posterior_axes = -anterior_axes
    return gray, initial, anterior_axes, posterior_axes


def build_fields(shape: tuple[int, int], initial: np.ndarray,
                 axes: np.ndarray, local_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(shape, dtype=np.float32)
    gaussians = []
    for idx in local_indices:
        center = initial[idx]
        gaussians.append(np.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2) /
                                (2 * SIGMA_PX ** 2)))
    weights = np.stack(gaussians)
    envelope = weights.max(axis=0)
    direction = np.einsum("khw,ki->hwi", weights, axes[local_indices])
    norm = np.linalg.norm(direction, axis=2)
    direction /= np.maximum(norm[..., None], 1e-8)
    vector = envelope[..., None] * direction
    return vector.astype(np.float32), envelope.astype(np.float32)


def make_frame(gray: np.ndarray, xx: np.ndarray, yy: np.ndarray,
               local_vector: np.ndarray, local_envelope: np.ndarray,
               condition: str, t: float, frequency_cpm: float) -> np.ndarray:
    phase = np.sin(2 * np.pi * frequency_cpm * t / 60)
    local_amplitude = POSITIVE_CONDITIONS.get(condition, 0.0)
    global_amplitude = 1.0 if condition in {"global_translation", "mixed_local_0p3"} else 0.0
    global_axis = np.array([1.0, 0.35], dtype=np.float32)
    global_axis /= np.linalg.norm(global_axis)
    displacement = local_amplitude * phase * local_vector
    displacement = displacement + global_amplitude * phase * global_axis
    if local_amplitude or global_amplitude:
        result = cv2.remap(
            gray, (xx - displacement[..., 0]).astype(np.float32),
            (yy - displacement[..., 1]).astype(np.float32), cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT)
    else:
        result = gray.copy()
    if condition == "global_brightness":
        multiplier = 1.0 + 0.10 * phase
        result = np.uint8(np.clip(result.astype(float) * multiplier, 0, 255))
    elif condition in {"local_brightness", "mixed_local_0p3"}:
        multiplier = 1.0 + 0.10 * phase * local_envelope
        result = np.uint8(np.clip(result.astype(float) * multiplier, 0, 255))
    return result


def track_condition(gray: np.ndarray, initial: np.ndarray, local_vector: np.ndarray,
                    local_envelope: np.ndarray, condition: str,
                    frequency_cpm: float) -> dict:
    time_s = np.arange(MAX_SECONDS * FPS + 1, dtype=float) / FPS
    n, count = len(time_s), len(initial)
    positions = np.full((n, count, 2), np.nan, np.float32)
    raw = np.full_like(positions, np.nan)
    valid = np.zeros((n, count), bool)
    raw_valid = np.zeros_like(valid)
    corrected = np.full((n, count, 2), np.nan, float)
    pose_valid = np.zeros(n, bool)
    epoch = np.zeros(n, int)
    poses = np.full((n, 3, 3), np.nan, float)
    xx, yy = np.meshgrid(
        np.arange(gray.shape[1], dtype=np.float32),
        np.arange(gray.shape[0], dtype=np.float32))
    accumulator = np.eye(3)
    current_epoch = 0
    previous = make_frame(
        gray, xx, yy, local_vector, local_envelope, condition, 0., frequency_cpm)
    processed_previous = local_contrast(previous)
    positions[0] = raw[0] = initial
    valid[0] = raw_valid[0] = True
    corrected[0] = initial
    pose_valid[0] = True
    poses[0] = accumulator
    for i in range(1, n):
        current = make_frame(
            gray, xx, yy, local_vector, local_envelope, condition,
            time_s[i], frequency_cpm)
        processed = local_contrast(current)
        live = np.flatnonzero(valid[i - 1])
        if len(live):
            found, good, _, _ = track_step(
                processed_previous, processed, positions[i - 1, live])
            kept = live[good]
            positions[i, kept] = found[good]
            valid[i, kept] = True
        raw_live = np.flatnonzero(raw_valid[i - 1])
        if len(raw_live):
            found, good, _, _ = track_step(previous, current, raw[i - 1, raw_live])
            kept = raw_live[good]
            raw[i, kept] = found[good]
            raw_valid[i, kept] = True
        cv2.setRNGSeed(20261027)
        matrix, _ = rigid_step(previous, current, raw[i - 1, raw_live]) \
            if len(raw_live) >= 3 else (None, {})
        if matrix is None:
            accumulator = np.eye(3)
            current_epoch += 1
        else:
            accumulator = accumulator @ np.linalg.inv(
                np.vstack((matrix, [0., 0., 1.])))
            pose_valid[i] = True
            poses[i] = accumulator
            corrected[i, valid[i]] = (
                positions[i, valid[i]] @ accumulator[:2, :2].T + accumulator[:2, 2])
        epoch[i] = current_epoch
        previous = current
        processed_previous = processed
    return dict(time_s=time_s, corrected=corrected, valid=valid,
                pose_valid=pose_valid, epoch=epoch, poses=poses)


def regional_rows(case: str, tracked: dict, anterior_axes: np.ndarray,
                  posterior_axes: np.ndarray, spacing: float,
                  end_seconds: int) -> tuple[list[dict], list[dict]]:
    keep = tracked["time_s"] <= end_seconds + 1e-9
    t = tracked["time_s"][keep]
    corrected = tracked["corrected"][keep]
    valid = tracked["valid"][keep]
    pose_valid = tracked["pose_valid"][keep]
    epoch = tracked["epoch"][keep]
    count = len(anterior_axes)
    region_indices = dict(zip(REGION_NAMES, np.array_split(np.arange(count), 3)))
    rows, curves = [], []
    for wall, columns, axes in (
            ("anterior", np.arange(count), anterior_axes),
            ("posterior", np.arange(count) + count, posterior_axes)):
        projected = np.einsum("nki,ki->nk", corrected[:, columns], axes)
        support = (pose_valid[:, None] & valid[:, columns] &
                   np.isfinite(corrected[:, columns]).all(axis=2))
        for region, local_indices in region_indices.items():
            result_rows, result_curves = _fit_fixed_support_v1_1(
                case, wall, region, t, projected, support, epoch, spacing,
                local_indices, region_indices)
            rows.extend(result_rows)
            curves.extend(result_curves)
    return rows, curves


def primary_for(rows: list[dict], wall: str, region: str) -> dict:
    matches = [r for r in rows if r["wall"] == wall and r["region"] == region
               and r["primary"]]
    if len(matches) != 1:
        raise ValueError(f"Expected one primary row for {wall}/{region}")
    return matches[0]


def result_row(holdout: dict, condition: str, duration: int,
               row: dict, is_target: bool) -> dict:
    frequency = row["frequency_cpm"]
    candidate = row["diagnostic_best_cpm"]
    passed = row["status"] == "exploratory_repetition_candidate"
    true_frequency = holdout["true_frequency_cpm"]
    error = abs(float(frequency) - true_frequency) if passed else ""
    agreement = bool(passed and error <= 0.25) if is_target and condition in POSITIVE_CONDITIONS else ""
    return dict(
        holdout_id=holdout["holdout_id"], condition=condition,
        duration_s=duration, wall=row["wall"], region=row["region"],
        is_injected_target=is_target,
        expected_geometry=("local_periodic_motion" if condition in POSITIVE_CONDITIONS
                           and is_target else "no_target_local_motion"),
        nominal_local_amplitude_processed_px=POSITIVE_CONDITIONS.get(condition, 0.0),
        true_frequency_cpm=true_frequency, status=row["status"],
        reason=row["reason"], frequency_cpm=frequency,
        diagnostic_best_cpm=candidate, absolute_frequency_error_cpm=error,
        target_frequency_agreement=agreement,
        range_mm=row["range_mm"], fitted_amplitude_mm=row["fitted_amplitude_mm"],
        duration_valid_s=row["duration_s"], support_tier=row["support_tier"],
        point_count=row["point_count"], planned_point_count=row["planned_point_count"],
        pose_valid_fraction="", all_wall_points_valid_fraction="")


def aggregate_control(rows: list[dict]) -> list[dict]:
    output = []
    for condition in CONDITIONS:
        subset = [r for r in rows if r["condition"] == condition]
        targets = [r for r in subset if r["is_injected_target"]]
        negative_regions = [r for r in subset if condition in NEGATIVE_CONDITIONS]
        off_target = [r for r in subset if condition in POSITIVE_CONDITIONS
                      and not r["is_injected_target"]]
        errors = [float(r["absolute_frequency_error_cpm"]) for r in targets
                  if r["absolute_frequency_error_cpm"] != ""]
        output.append(dict(
            condition=condition, holdout_cases=len({r["holdout_id"] for r in subset}),
            target_evaluations=len(targets),
            target_rule_pass=sum(r["status"] == "exploratory_repetition_candidate" for r in targets),
            target_frequency_agreement=sum(r["target_frequency_agreement"] is True for r in targets),
            target_frequency_mae_cpm=float(np.mean(errors)) if errors else "",
            negative_region_evaluations=len(negative_regions),
            negative_region_false_pass=sum(
                r["status"] == "exploratory_repetition_candidate" for r in negative_regions),
            off_target_region_evaluations=len(off_target),
            off_target_region_pass=sum(
                r["status"] == "exploratory_repetition_candidate" for r in off_target)))
    return output


def aggregate_duration(rows: list[dict]) -> list[dict]:
    output = []
    for duration in DURATIONS_S:
        for condition in POSITIVE_CONDITIONS:
            subset = [r for r in rows if r["duration_s"] == duration
                      and r["condition"] == condition]
            errors = [float(r["absolute_frequency_error_cpm"]) for r in subset
                      if r["absolute_frequency_error_cpm"] != ""]
            output.append(dict(
                duration_s=duration, condition=condition, evaluations=len(subset),
                rule_pass=sum(r["status"] == "exploratory_repetition_candidate" for r in subset),
                frequency_agreement=sum(r["target_frequency_agreement"] is True for r in subset),
                not_evaluable=sum(r["status"] == "not_evaluable" for r in subset),
                mean_absolute_error_cpm=float(np.mean(errors)) if errors else ""))
    return output


def write_report(output: Path, protocol: dict, controls: list[dict],
                 durations: list[dict], elapsed_s: float) -> None:
    negative_total = sum(r["negative_region_evaluations"] for r in controls)
    false_total = sum(r["negative_region_false_pass"] for r in controls)
    positive = [r for r in controls if r["condition"] in POSITIVE_CONDITIONS]
    target_total = sum(r["target_evaluations"] for r in positive)
    agreement_total = sum(r["target_frequency_agreement"] for r in positive)
    lines = [
        "# V1.1区域频率：未参与本轮开发背景的对照验证",
        "",
        "## 结论摘要",
        "",
        f"- 预先锁定{len(protocol['cases'])}个背景、7种条件；运行耗时{elapsed_s / 60:.1f}分钟。",
        f"- 三种含已知局部运动条件：目标区域频率一致{agreement_total}/{target_total}（容差±0.25次/分，仅为验证统计，不是临床阈值）。",
        f"- 四种无目标局部几何运动条件：区域规则误报{false_total}/{negative_total}。任何误报均按误报计，不因曲线较小而改称通过。",
        "- 30/45/60/75秒结果见duration_aggregate.csv；各时长是同一生成序列的固定前缀，不挑选平稳片段。",
        "",
        "## 理论依据和边界",
        "",
        "已知输入位移提供可计算的运动和频率真值；无几何运动的整体平移、整体明暗及局部明暗用于检验刚性校正与光流亮度恒常假设的失效；固定前缀只改变观察时长，可直接检验冻结规则要求至少两个周期所产生的可辨识边界。区域平均在各点误差不完全同向时可降低随机噪声，但不能消除共同偏差，因此必须同时看阴性对照。",
        "",
        "这不是生理有效性或真实患者运动真值验证。背景来自真实首帧，但后续帧为可控合成；10帧/秒只因目标频率远低于奈奎斯特频率而适合频率试验，不能替代原帧率下的临床采集验证。旧轨迹只提供首帧种子，未复用旧逐帧轨迹、旧QC或旧频率结果。病例名未写入结果，仅保留不可逆来源键。",
        "",
        "## 文件",
        "",
        "- locked_protocol.json：看结果前锁定的方案和匿名背景。",
        "- control_case_results.csv / control_aggregate.csv：真/假运动对照。",
        "- duration_case_results.csv / duration_aggregate.csv：固定前缀时长敏感性。",
        "- target_curves.csv.gz：目标区域主要段曲线；失效帧不填补。",
        "- 实际运行日志.txt：逐病例逐条件运行记录。",
    ]
    (output / "结果说明_中文.md").write_text("\n".join(lines), encoding="utf-8")


def process_holdout(holdout: dict) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """Run one locked background; safe to dispatch in a separate process."""
    cv2.setNumThreads(1)
    control_results, duration_results, curve_rows, log_lines = [], [], [], []
    gray, initial, ant_axes, post_axes = load_background(holdout)
    count = len(ant_axes)
    region_indices = dict(zip(REGION_NAMES, np.array_split(np.arange(count), 3)))
    local = region_indices[holdout["target_region"]]
    if holdout["target_wall"] == "anterior":
        global_indices = local
    else:
        global_indices = local + count
    all_axes = np.vstack((ant_axes, post_axes))
    local_vector, local_envelope = build_fields(
        gray.shape, initial, all_axes, global_indices)
    for condition_index, condition in enumerate(CONDITIONS, start=1):
        condition_start = time.time()
        tracked = track_condition(
            gray, initial, local_vector, local_envelope, condition,
            holdout["true_frequency_cpm"])
        full_rows, full_curves = regional_rows(
            holdout["holdout_id"], tracked, ant_axes, post_axes,
            holdout["mm_per_processed_pixel"], MAX_SECONDS)
        for wall in ("anterior", "posterior"):
            for region in REGION_NAMES:
                primary = primary_for(full_rows, wall, region)
                is_target = (wall == holdout["target_wall"] and
                             region == holdout["target_region"])
                item = result_row(
                    holdout, condition, MAX_SECONDS, primary, is_target)
                item["pose_valid_fraction"] = float(tracked["pose_valid"].mean())
                item["all_wall_points_valid_fraction"] = float(tracked["valid"].all(axis=1).mean())
                control_results.append(item)
                if is_target:
                    matched_curves = [
                        c for c in full_curves
                        if c["wall"] == wall and c["region"] == region
                        and c["segment"] == primary["segment"]
                        and c["support_pattern"] == primary["support_pattern"]]
                    for curve in matched_curves:
                        curve_rows.append(dict(
                            holdout_id=holdout["holdout_id"], condition=condition,
                            duration_s=MAX_SECONDS, frame=curve["frame"],
                            time_s=curve["time_s"], displacement_mm=curve["displacement_mm"],
                            support_tier=curve["support_tier"],
                            point_count=curve["point_count"]))
        if condition in POSITIVE_CONDITIONS:
            for duration in DURATIONS_S:
                if duration == MAX_SECONDS:
                    rows = full_rows
                else:
                    rows, _ = regional_rows(
                        holdout["holdout_id"], tracked, ant_axes, post_axes,
                        holdout["mm_per_processed_pixel"], duration)
                primary = primary_for(
                    rows, holdout["target_wall"], holdout["target_region"])
                duration_results.append(result_row(
                    holdout, condition, duration, primary, True))
        elapsed = time.time() - condition_start
        message = (f"{holdout['holdout_id']} {condition} "
                   f"{condition_index}/7 completed {elapsed:.1f}s")
        log_lines.append(message)
        print(message, flush=True)
    return control_results, duration_results, curve_rows, log_lines


def run(videos: Path, existing: Path, output: Path,
        lock_only: bool, workers: int) -> None:
    catalogued = catalogue(videos, existing)
    if output.exists():
        protocol_path = output / "locked_protocol.json"
        if not protocol_path.is_file():
            raise ValueError("Output exists without locked protocol")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    else:
        protocol = lock_protocol(catalogued, output)
    if lock_only:
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
        return
    if (output / "control_case_results.csv").exists():
        raise ValueError("Results already exist; refusing to overwrite")
    resolved = resolve_locked(protocol, catalogued)
    started = time.time()
    control_results, duration_results, curve_rows, log_lines = [], [], [], []
    if workers < 1:
        raise ValueError("workers must be positive")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for case_control, case_duration, case_curves, case_logs in executor.map(
                process_holdout, resolved):
            control_results.extend(case_control)
            duration_results.extend(case_duration)
            curve_rows.extend(case_curves)
            log_lines.extend(case_logs)
    control_aggregate = aggregate_control(control_results)
    duration_aggregate = aggregate_duration(duration_results)
    write_csv(output / "control_case_results.csv", control_results)
    write_csv(output / "control_aggregate.csv", control_aggregate)
    write_csv(output / "duration_case_results.csv", duration_results)
    write_csv(output / "duration_aggregate.csv", duration_aggregate)
    with gzip.open(output / "target_curves.csv.gz", "wt", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    elapsed_s = time.time() - started
    log_lines.extend([
        f"completed_seconds={elapsed_s:.3f}",
        f"opencv_version={cv2.__version__}", f"numpy_version={np.__version__}",
        "frozen_tracking_rigid_frequency_and_support_rules_imported_unchanged=true",
    ])
    (output / "实际运行日志.txt").write_text("\n".join(log_lines), encoding="utf-8")
    write_report(output, protocol, control_aggregate, duration_aggregate, elapsed_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--existing-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run(args.videos, args.existing_results, args.out, args.lock_only, args.workers)
