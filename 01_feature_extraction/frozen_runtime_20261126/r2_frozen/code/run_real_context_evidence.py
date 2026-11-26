"""Evidence-only Stage-B pilot on deterministically selected unseen videos."""
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
sys.path.insert(0, str(HERE))

from ab_block_tracking import patch as tracking_patch  # noqa: E402
from ab_block_tracking import track_step  # noqa: E402
from contrast import local_contrast  # noqa: E402
from rigid_correction import step as rigid_step  # noqa: E402
import validate_regional_v1_1 as frozen  # noqa: E402
from run_context_evidence_pilot import (  # noqa: E402
    contiguous_runs,
    finite_mean,
    finite_range,
    pairwise_neighbor_correlation,
    safe_corr,
    sha256_file,
    write_csv,
)


SELECTION_SALT = "context-real-20261028|"
RANDOM_SEED = 20261028
FROZEN_FILES = (
    "ab_block_tracking.py", "cuda_sparse_lk.py", "contrast.py", "rigid_correction.py",
    "frequency.py", "run_regional_frequency.py", "validate_regional_v1_1.py",
)


def selection_digest(source_key: str) -> str:
    return hashlib.sha256((SELECTION_SALT + source_key).encode()).hexdigest()


def video_metadata(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def select_unseen_cases(
    videos: Path,
    existing_results: Path,
    locked_protocol: dict,
    count: int,
    duration_s: int,
) -> list[dict]:
    excluded = {row["source_key_sha256"] for row in locked_protocol["cases"]}
    eligible = []
    for row in frozen.catalogue(videos, existing_results):
        if row["key"] in excluded:
            continue
        frame_count, fps, width, height = video_metadata(row["video"])
        with np.load(row["tracking"], allow_pickle=False) as archive:
            tracking_frames = int(archive["tracks"].shape[0])
        available_s = (tracking_frames - 1) / fps if fps > 0 else 0.0
        if (
            frame_count == tracking_frames
            and fps > 0
            and (width, height) == (1260, 910)
            and available_s >= duration_s
        ):
            eligible.append({
                **row,
                "selection_digest": selection_digest(row["key"]),
                "frame_count": frame_count,
                "fps": fps,
                "available_s": available_s,
            })
    eligible.sort(key=lambda row: row["selection_digest"])
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} eligible unseen cases")
    return [dict(row, anonymous_id=f"R{index + 1:02d}")
            for index, row in enumerate(eligible[:count])]


def patch_median(frame: np.ndarray, point: np.ndarray) -> float:
    own = tracking_patch(frame, point)
    return float(np.median(own)) if own is not None else math.nan


def track_real_case(row: dict, duration_s: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    gray0, initial, anterior_axes, posterior_axes = frozen.load_background(row)
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
    previous = processed_previous = None
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
            processed = local_contrast(gray)
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
                        previous, gray, raw_positions[index - 1, raw_live])
                    kept = raw_live[good]
                    raw_positions[index, kept] = found[good]
                    raw_valid[index, kept] = True
                cv2.setRNGSeed(RANDOM_SEED)
                matrix, stats = rigid_step(
                    previous, gray, raw_positions[index - 1, raw_live]
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
                point_brightness[index, column] = patch_median(
                    gray, positions[index, column])
            global_brightness[index] = float(np.median(gray[fan_mask]))
            heldout_raw[index] = stats.get("heldout_raw_median_px", math.nan)
            heldout_corrected[index] = stats.get(
                "heldout_corrected_median_px", math.nan)
            training_inliers[index] = stats.get("training_inliers", 0)
            heldout_count[index] = stats.get("heldout_count", 0)
            previous = gray
            processed_previous = processed
            if index and index % 300 == 0:
                print(row["anonymous_id"], index, n, flush=True)
    finally:
        cap.release()
    return (
        dict(
            time_s=time_s, positions=positions, raw_positions=raw_positions,
            valid=valid, raw_valid=raw_valid, corrected=corrected,
            pose_valid=pose_valid, epoch=epoch, poses=poses,
            target_ncc=target_ncc, target_fb=target_fb,
            point_brightness=point_brightness,
            global_brightness=global_brightness,
            heldout_raw=heldout_raw, heldout_corrected=heldout_corrected,
            training_inliers=training_inliers, heldout_count=heldout_count,
        ),
        initial,
        anterior_axes,
        posterior_axes,
    )


def fractional_range(values: np.ndarray) -> float:
    own = np.asarray(values, float)
    own = own[np.isfinite(own)]
    if not len(own):
        return math.nan
    scale = float(np.median(np.abs(own)))
    return float(np.ptp(own) / scale) if scale > 1e-9 else math.nan


def region_arrays(
    tracked: dict,
    initial: np.ndarray,
    all_axes: np.ndarray,
    local_indices: np.ndarray,
    columns: np.ndarray,
) -> dict:
    corrected_delta = tracked["corrected"] - initial[None]
    raw_delta = tracked["positions"].astype(float) - initial[None]
    corrected_point = np.einsum(
        "nki,ki->nk", corrected_delta[:, columns], all_axes[columns])
    raw_point = np.einsum(
        "nki,ki->nk", raw_delta[:, columns], all_axes[columns])
    region_valid = (
        tracked["pose_valid"]
        & tracked["valid"][:, columns].all(axis=1)
        & np.isfinite(corrected_point).all(axis=1)
        & np.isfinite(raw_point).all(axis=1)
    )
    corrected_mean = np.where(region_valid, corrected_point.mean(axis=1), np.nan)
    raw_mean = np.where(region_valid, raw_point.mean(axis=1), np.nan)
    dispersion = np.where(region_valid, np.std(corrected_point, axis=1), np.nan)

    background_point = np.full_like(corrected_point, np.nan, float)
    for frame in np.flatnonzero(tracked["pose_valid"]):
        inverse = np.linalg.inv(tracked["poses"][frame])
        mapped = initial[columns] @ inverse[:2, :2].T + inverse[:2, 2]
        displacement = mapped - initial[columns]
        background_point[frame] = np.einsum(
            "ki,ki->k", displacement, all_axes[columns])
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
    strain = current_distance / initial_distance[None] - 1.0
    radial_strain = np.where(pair_valid, strain.mean(axis=1), np.nan)

    local_brightness = np.array([
        finite_mean(row) for row in tracked["point_brightness"][:, columns]
    ])
    mean_ncc = np.array([
        finite_mean(row) for row in tracked["target_ncc"][:, columns]
    ])
    max_fb = np.array([
        float(np.nanmax(row)) if np.isfinite(row).any() else math.nan
        for row in tracked["target_fb"][:, columns]
    ])
    evidence_valid = (
        region_valid & pair_valid & np.isfinite(background_mean)
        & np.isfinite(radial_strain) & np.isfinite(local_brightness)
        & np.isfinite(tracked["global_brightness"])
    )
    return dict(
        region_valid=region_valid, pair_valid=pair_valid,
        evidence_valid=evidence_valid, corrected_point=corrected_point,
        corrected_mean=corrected_mean, raw_mean=raw_mean,
        background_mean=background_mean, radial_strain=radial_strain,
        local_brightness=local_brightness, dispersion=dispersion,
        mean_ncc=mean_ncc, max_fb=max_fb,
    )


def segment_metrics(
    case_id: str,
    wall: str,
    region: str,
    tracked: dict,
    arrays: dict,
) -> list[dict]:
    runs = contiguous_runs(arrays["evidence_valid"], tracked["epoch"])
    if not runs:
        return []
    lengths = np.array([stop - start for start, stop in runs])
    primary_index = int(np.argmax(lengths))
    rows = []
    for number, (start, stop) in enumerate(runs):
        own = slice(start, stop)
        corrected = arrays["corrected_mean"][own]
        raw = arrays["raw_mean"][own]
        background = arrays["background_mean"][own]
        brightness = arrays["local_brightness"][own]
        radial = arrays["radial_strain"][own]
        raw_range = finite_range(raw)
        background_range = finite_range(background)
        rows.append(dict(
            case_id=case_id, wall=wall, region=region,
            segment=number, primary=number == primary_index,
            start_frame=start, end_frame=stop - 1,
            duration_s=float(tracked["time_s"][stop - 1] - tracked["time_s"][start]),
            frames=stop - start,
            corrected_range_px=finite_range(corrected),
            raw_range_px=raw_range, background_range_px=background_range,
            background_to_raw_range_ratio=(
                background_range / raw_range if raw_range > 1e-9 else math.nan),
            radial_strain_range=finite_range(radial),
            local_brightness_fractional_range=fractional_range(brightness),
            global_brightness_fractional_range=fractional_range(
                tracked["global_brightness"][own]),
            raw_background_corr=safe_corr(raw, background),
            corrected_background_corr=safe_corr(corrected, background),
            corrected_local_brightness_corr=safe_corr(corrected, brightness),
            corrected_radial_strain_corr=safe_corr(corrected, radial),
            adjacent_point_corr=pairwise_neighbor_correlation(
                arrays["corrected_point"][own]),
            mean_neighbor_dispersion_px=finite_mean(arrays["dispersion"][own]),
            mean_target_ncc=finite_mean(arrays["mean_ncc"][own]),
            max_target_fb_px=(
                float(np.nanmax(arrays["max_fb"][own]))
                if np.isfinite(arrays["max_fb"][own]).any() else math.nan),
            mean_heldout_residual_px=finite_mean(
                tracked["heldout_corrected"][own]),
            evidence_valid_fraction=float(arrays["evidence_valid"].mean()),
        ))
    return rows


def frame_rows(
    case_id: str,
    wall: str,
    region: str,
    tracked: dict,
    arrays: dict,
) -> list[dict]:
    rows = []
    for index, seconds in enumerate(tracked["time_s"]):
        rows.append(dict(
            case_id=case_id, wall=wall, region=region,
            frame=index, time_s=seconds, epoch=int(tracked["epoch"][index]),
            pose_valid=bool(tracked["pose_valid"][index]),
            region_valid=bool(arrays["region_valid"][index]),
            paired_wall_valid=bool(arrays["pair_valid"][index]),
            evidence_valid=bool(arrays["evidence_valid"][index]),
            corrected_mean_px=arrays["corrected_mean"][index],
            raw_mean_px=arrays["raw_mean"][index],
            background_mean_px=arrays["background_mean"][index],
            radial_strain=arrays["radial_strain"][index],
            local_brightness_median=arrays["local_brightness"][index],
            global_brightness_median=tracked["global_brightness"][index],
            neighbor_dispersion_px=arrays["dispersion"][index],
            mean_target_ncc=arrays["mean_ncc"][index],
            max_target_fb_px=arrays["max_fb"][index],
            heldout_corrected_median_px=tracked["heldout_corrected"][index],
        ))
    return rows


def process_case(row: dict, duration_s: int, case_output: Path):
    started = time.time()
    tracked, initial, anterior_axes, posterior_axes = track_real_case(row, duration_s)
    all_axes = np.vstack((anterior_axes, posterior_axes))
    count = len(anterior_axes)
    regions = dict(zip(frozen.REGION_NAMES, np.array_split(np.arange(count), 3)))
    frequency_rows, _ = frozen.regional_rows(
        row["anonymous_id"], tracked, anterior_axes, posterior_axes,
        row["spacing"], duration_s)
    frequency_primary = {
        (wall, region): frozen.primary_for(frequency_rows, wall, region)
        for wall in ("anterior", "posterior") for region in frozen.REGION_NAMES
    }
    frames, segments, summaries, frequencies = [], [], [], []
    for wall, offset in (("anterior", 0), ("posterior", count)):
        for region, local_indices in regions.items():
            columns = local_indices + offset
            arrays = region_arrays(
                tracked, initial, all_axes, local_indices, columns)
            own_segments = segment_metrics(
                row["anonymous_id"], wall, region, tracked, arrays)
            frames.extend(frame_rows(
                row["anonymous_id"], wall, region, tracked, arrays))
            segments.extend(own_segments)
            primary = next((item for item in own_segments if item["primary"]), None)
            freq = frequency_primary[(wall, region)]
            summary = dict(
                case_id=row["anonymous_id"], wall=wall, region=region,
                frequency_status=freq["status"], frequency_reason=freq["reason"],
                frequency_cpm=freq["frequency_cpm"],
                diagnostic_best_cpm=freq["diagnostic_best_cpm"],
                frequency_duration_s=freq["duration_s"],
                frequency_support_tier=freq["support_tier"],
                pose_valid_fraction=float(tracked["pose_valid"].mean()),
                epoch_count=int(tracked["epoch"].max() + 1),
                evidence_segment_count=len(own_segments),
            )
            metric_names = (
                "duration_s", "corrected_range_px", "raw_range_px",
                "background_range_px", "background_to_raw_range_ratio",
                "radial_strain_range", "local_brightness_fractional_range",
                "global_brightness_fractional_range", "raw_background_corr",
                "corrected_background_corr", "corrected_local_brightness_corr",
                "corrected_radial_strain_corr", "adjacent_point_corr",
                "mean_neighbor_dispersion_px", "mean_target_ncc",
                "max_target_fb_px", "mean_heldout_residual_px",
                "evidence_valid_fraction",
            )
            for name in metric_names:
                summary[name] = primary[name] if primary is not None else ""
            summaries.append(summary)
            frequencies.append(dict(
                case_id=row["anonymous_id"], wall=wall, region=region,
                status=freq["status"], reason=freq["reason"],
                frequency_cpm=freq["frequency_cpm"],
                diagnostic_best_cpm=freq["diagnostic_best_cpm"],
                range_mm=freq["range_mm"], duration_s=freq["duration_s"],
                support_tier=freq["support_tier"],
            ))
    case_output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        case_output / "evidence_arrays.npz",
        time_s=tracked["time_s"], corrected_A=tracked["corrected"],
        positions=tracked["positions"], target_valid=tracked["valid"],
        pose_A=tracked["poses"], pose_valid_A=tracked["pose_valid"],
        epoch_A=tracked["epoch"], target_ncc=tracked["target_ncc"],
        target_fb=tracked["target_fb"], point_brightness=tracked["point_brightness"],
        global_brightness=tracked["global_brightness"],
        heldout_corrected=tracked["heldout_corrected"],
        initial_points=initial, anterior_axes=anterior_axes,
        posterior_axes=posterior_axes,
    )
    elapsed = time.time() - started
    return frames, segments, summaries, frequencies, elapsed


def plot_summary(output: Path, rows: list[dict]) -> None:
    labels = [f"{r['case_id']}\n{r['wall'][0].upper()}-{r['region'][0].upper()}" for r in rows]
    fields = (
        ("radial_strain_range", "paired-wall strain range", True),
        ("background_to_raw_range_ratio", "background/raw range", True),
        ("local_brightness_fractional_range", "local brightness fractional range", True),
        ("evidence_valid_fraction", "evidence valid fraction", False),
    )
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    x = np.arange(len(rows))
    colors = ["#2563eb" if r["frequency_status"] != "exploratory_repetition_candidate"
              else "#dc2626" for r in rows]
    for axis, (field, title, log_scale) in zip(axes.ravel(), fields):
        values = [float(row[field]) if row[field] != "" else math.nan for row in rows]
        axis.bar(x, values, color=colors)
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=65, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("symlog", linthresh=1e-4)
    figure.suptitle("Real-video context evidence (red: frozen frequency candidate)")
    figure.savefig(output / "real_region_evidence_summary.png", dpi=160)
    plt.close(figure)


def write_report(output: Path, summaries: list[dict], cases: list[dict]) -> None:
    candidates = [
        row for row in summaries
        if row["frequency_status"] == "exploratory_repetition_candidate"]
    valid = [float(row["evidence_valid_fraction"]) for row in summaries
             if row["evidence_valid_fraction"] != ""]
    strain = [float(row["radial_strain_range"]) for row in summaries
              if row["radial_strain_range"] != ""]
    lines = [
        "# 真实视频目标—周围组织证据旁路：首批3例",
        "",
        "## 范围",
        "",
        "- R01–R03由固定哈希顺序选取，未参与此前短视频、G0217/G0345或H01–H12试验。",
        "- 每例固定前60秒，保持原帧率；不挑选平稳片段。",
        "- 冻结追踪、背景刚性校正、区域支持和频率规则未修改。",
        "- 本轮没有真实组织运动真值，不报告准确率。",
        "",
        "## 数值概览",
        "",
        f"- 六区域共{len(summaries)}项，冻结频率候选{len(candidates)}项。",
        f"- 区域证据有效率范围：{min(valid):.1%}–{max(valid):.1%}。" if valid else "- 无可用区域证据。",
        f"- 主要连续段径向应变范围：{min(strain):.6f}–{max(strain):.6f}。" if strain else "- 无可用径向应变。",
        "",
        "## 解释边界",
        "",
        "- 红色频率候选仅表示冻结重复性规则通过，不等于真实子宫蠕动。",
        "- 径向应变可作为ROI形变证据；没有传播分析时不报告方向、速度或CF/FC。",
        "- 背景共同运动、亮度变化或低有效率存在时，应降级为风险标记或暂不可判断。",
        "- 本轮不把阶段A观察区间直接用作真实视频阈值，也不自动重分类。",
        "- 人工盲法视频复核尚未进行，不能把数值证据当作运动真值。",
        "",
        "## 匿名输入",
        "",
    ]
    for row in cases:
        lines.append(
            f"- {row['anonymous_id']}：原帧率{row['fps']:.3f} fps，可用时长{row['available_s']:.2f}秒，每壁{row['wall_points']}点。")
    (output / "结果说明_中文.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("Output exists; refusing to overwrite")
    args.out.mkdir(parents=True)
    plots = args.out / "evidence_plots"
    plots.mkdir()
    locked = json.loads(args.protocol.read_text(encoding="utf-8"))
    selected = select_unseen_cases(
        args.videos, args.existing_results, locked, args.count, args.duration)
    protocol = dict(
        experiment="V1.3 real-video context evidence pilot",
        date="2026-10-28", stage="B_real_video_interpretability",
        selection=(
            "deterministic SHA256 order after excluding frozen numeric development IDs "
            "and all H01-H12 locked sources; >=60 s; exact video/tracking frame match"
        ),
        duration_s=args.duration, original_frame_rate_preserved=True,
        classification_thresholds={},
        evidence_mode="descriptive_only_no_outcome_reclassification",
        cases=[dict(
            anonymous_id=row["anonymous_id"],
            source_key_sha256=row["key"],
            selection_digest_sha256=row["selection_digest"],
            fps=row["fps"], available_s=row["available_s"],
            wall_points_per_side=row["wall_points"],
            mm_per_processed_pixel=row["spacing"],
            video_sha256=sha256_file(row["video"]),
            tracking_sha256=sha256_file(row["tracking"]),
            nodes_sha256=sha256_file(row["nodes"]),
            calibration_sha256=sha256_file(row["calibration"]),
        ) for row in selected],
        frozen_source_hashes={name: sha256_file(CODE / name) for name in FROZEN_FILES},
        script_sha256=sha256_file(Path(__file__)),
        privacy="no patient names or source paths written",
    )
    (args.out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    all_frames, all_segments, all_summaries, all_frequency = [], [], [], []
    logs = [
        f"started_local=2026-10-28", f"duration_s={args.duration}",
        "classification_thresholds=none",
    ]
    started = time.time()
    for row in selected:
        case_output = args.out / row["anonymous_id"]
        frames, segments, summaries, frequencies, elapsed = process_case(
            row, args.duration, case_output)
        all_frames.extend(frames)
        all_segments.extend(segments)
        all_summaries.extend(summaries)
        all_frequency.extend(frequencies)
        message = f"{row['anonymous_id']} completed {elapsed:.1f}s"
        logs.append(message)
        print(message, flush=True)

    write_csv(args.out / "frame_evidence.csv", all_frames)
    write_csv(args.out / "segment_evidence.csv", all_segments)
    write_csv(args.out / "region_summary.csv", all_summaries)
    write_csv(args.out / "regional_frequency_primary.csv", all_frequency)
    plot_summary(plots, all_summaries)
    write_report(args.out, all_summaries, selected)
    elapsed = time.time() - started
    logs.extend([
        f"completed_seconds={elapsed:.3f}",
        f"opencv_version={cv2.__version__}", f"numpy_version={np.__version__}",
        "frozen_sources_modified=false", "historical_results_overwritten=false",
        "patient_identifiers_written=false", "manual_blinded_review_completed=false",
    ])
    (args.out / "实际运行日志.txt").write_text(
        "\n".join(logs) + "\n", encoding="utf-8")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--existing-results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--duration", type=int, default=60)
    run(parser.parse_args())