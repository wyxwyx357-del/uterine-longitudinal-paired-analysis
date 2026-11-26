"""Regional frequency analysis layered on saved strict V1 A tracks.

The tracking, correction, segmentation, and frequency estimator are not changed.
Points retain the saved cervix-side to fundus-side order.  For eight points,
the three contiguous regions contain 3, 3, and 2 points respectively.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from frequency import fit_frequency, segments
from run_frequency import write_csv


REGION_NAMES = ("cervical", "middle", "fundal")


def _fit_rows(case, signal, t, y, mask, epoch, spacing, extra):
    parts = segments(mask, epoch, np.zeros(len(t), dtype=bool), t)
    primary = max(parts, key=lambda ab: (t[ab[1] - 1] - t[ab[0]], -ab[0])) if parts else None
    rows = []
    curves = []
    if not parts:
        rows.append(dict(
            case=case, signal=signal, segment="", primary=True,
            start_frame="", end_frame="", duration_s=0,
            status="not_evaluable", reason="no_continuous_valid_segment",
            frequency_cpm="", period_s="", range_mm="",
            fitted_amplitude_mm="", diagnostic_best_cpm="", score="",
            lag_correlation="", half_lag_correlation="", **extra))
        return rows, curves

    for sid, (a, b) in enumerate(parts):
        values = y[a:b] - y[a]
        fit, _, _, _, _ = fit_frequency(t[a:b], values)
        amplitude_mm = ""
        if fit["candidate_cpm"] is not None:
            tt = t[a:b] - t[a:b].mean()
            phase = 2 * np.pi * tt * fit["candidate_cpm"] / 60
            design = np.column_stack([
                np.ones(len(tt)), tt, np.sin(phase), np.cos(phase)])
            beta = np.linalg.lstsq(design, values, rcond=None)[0]
            amplitude_mm = float(np.hypot(*beta[-2:]) * spacing)
        passed = fit["status"] == "exploratory_repetition_candidate"
        status = ("not_evaluable" if fit["reason"] ==
                  "continuous_segment_too_short_for_two_cycles_in_search_range"
                  else fit["status"])
        rows.append(dict(
            case=case, signal=signal, segment=sid, primary=(a, b) == primary,
            start_frame=a, end_frame=b - 1,
            duration_s=float(t[b - 1] - t[a]), status=status,
            reason=fit["reason"],
            frequency_cpm=fit["candidate_cpm"] if passed else "",
            period_s=fit["candidate_period_s"] if passed else "",
            range_mm=float(np.ptp(values) * spacing),
            fitted_amplitude_mm=amplitude_mm,
            diagnostic_best_cpm=fit["candidate_cpm"], score=fit["score"],
            lag_correlation=fit["lag_correlation"],
            half_lag_correlation=fit["half_lag_correlation"], **extra))
        for i in range(a, b):
            curves.append(dict(
                case=case, frame=i, time_s=float(t[i]), signal=signal,
                segment=sid, displacement_mm=float(values[i - a] * spacing),
                **extra))
    return rows, curves


def _normalise_axes(axes, expected):
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (expected, 2) or not np.isfinite(axes).all():
        raise ValueError("Invalid wall axes")
    norm = np.linalg.norm(axes, axis=1)
    if np.any(norm <= 0):
        raise ValueError("Zero-length wall axis")
    return axes / norm[:, None]


def _support_tier(member_mask, region, local_regions):
    """Return the prespecified V1.1 tier for one fixed support pattern."""
    count = int(member_mask.sum())
    planned = len(member_mask)
    if region == "whole_wall":
        if count == planned:
            return "full"
        minimum = int(np.ceil(0.75 * planned))
        spatial_coverage = all(member_mask[idx].any() for idx in local_regions.values())
        return "partial_wall_sensitivity" if count >= minimum and spatial_coverage else None
    if count == planned:
        return "full"
    members = np.flatnonzero(member_mask)
    if planned == 3 and count == 2 and np.diff(members).item() == 1:
        return "degraded_2_of_3"
    return None


def _fit_fixed_support_v1_1(case, wall, region, t, projected, support,
                            epoch, spacing, local_indices, local_regions):
    """Fit separate segments whenever the fixed set of valid points changes."""
    rows, curves = [], []
    patterns = np.unique(support[:, local_indices], axis=0)
    for pattern_number, pattern in enumerate(patterns):
        tier = _support_tier(pattern, region, local_regions)
        if tier is None:
            continue
        members = local_indices[np.flatnonzero(pattern)]
        pattern_key = ";".join(str(int(i) + 1) for i in members)
        missing = local_indices[~pattern]
        mask = np.all(support[:, local_indices] == pattern, axis=1)
        signal_px = projected[:, members].mean(axis=1)
        extra = dict(
            wall=wall, region=region, location="", point_id="",
            point_count=len(members), planned_point_count=len(local_indices),
            point_locations=pattern_key,
            missing_point_locations=";".join(str(int(i) + 1) for i in missing),
            support_tier=tier, support_pattern=pattern_number,
            aggregation="fixed_support_equal_weight_mean")
        part_rows, part_curves = _fit_rows(
            case, f"{wall}_{region}", t, signal_px, mask, epoch, spacing, extra)
        if part_rows[0]["segment"] == "":
            continue
        rows.extend(part_rows)
        curves.extend(part_curves)

    if not rows:
        extra = dict(
            wall=wall, region=region, location="", point_id="",
            point_count=0, planned_point_count=len(local_indices),
            point_locations="", missing_point_locations="",
            support_tier="none", support_pattern="",
            aggregation="fixed_support_equal_weight_mean")
        return _fit_rows(
            case, f"{wall}_{region}", t, np.zeros(len(t)),
            np.zeros(len(t), dtype=bool), epoch, spacing, extra)

    rows.sort(key=lambda row: (int(row["start_frame"]), int(row["end_frame"])))
    key_to_segment = {}
    for new_segment, row in enumerate(rows):
        old_key = (row["support_pattern"], row["segment"])
        key_to_segment[old_key] = new_segment
        row["segment"] = new_segment
        row["primary"] = False
    for curve in curves:
        old_key = (curve["support_pattern"], curve["segment"])
        curve["segment"] = key_to_segment[old_key]
    primary_index = max(
        range(len(rows)),
        key=lambda i: (float(rows[i]["duration_s"]), -int(rows[i]["start_frame"])))
    rows[primary_index]["primary"] = True
    return rows, curves


def analyze_regions(path, signed_path, out, case, spacing):
    if out.exists():
        raise ValueError("Output exists; choose a new directory")
    if spacing <= 0 or not np.isfinite(spacing):
        raise ValueError("Invalid calibration")
    with np.load(path, allow_pickle=False) as z:
        t = z["time_s"].astype(float)
        p = z["corrected_A"].astype(float)
        pose_valid = z["pose_valid_A"].astype(bool)
        valid = z["target_valid"].astype(bool)
        epoch = z["epoch_A"].astype(int)
        saved_axis = z["inward_axis"].astype(float)
        target_index = int(z["target_index"])
    with np.load(signed_path, allow_pickle=False) as z:
        anterior_ids = z["anterior_point_ids"].astype(int)
        posterior_ids = z["posterior_point_ids"].astype(int)
        anterior_axes = _normalise_axes(z["anterior_inward_axes"], len(anterior_ids))
        has_posterior_axes = "posterior_inward_axes" in z.files
        posterior_axes = (_normalise_axes(z["posterior_inward_axes"], len(posterior_ids))
                          if has_posterior_axes else -anterior_axes)

    if len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
        raise ValueError("Invalid timestamps")
    if p.shape[:2] != valid.shape or p.shape[0] != len(t):
        raise ValueError("Track and validity shapes do not match")
    if len(anterior_ids) != len(posterior_ids) or len(anterior_ids) < 3:
        raise ValueError("Walls require equal point counts of at least three")
    expected_count = len(anterior_ids) + len(posterior_ids)
    if p.shape[1] < expected_count:
        raise ValueError("Saved V1 A array has fewer points than signed metadata")
    if target_index != 2 or not np.allclose(
            saved_axis / np.linalg.norm(saved_axis), anterior_axes[2], atol=1e-6):
        raise ValueError("Saved V1 A point order does not match signed metadata")
    # Preserve the exact axis already stored with the frozen target result so
    # anterior point 3 remains a byte-for-byte numerical regression check.
    anterior_axes[target_index] = saved_axis / np.linalg.norm(saved_axis)

    out.mkdir(parents=True)
    wall_data = {
        "anterior": (np.arange(len(anterior_ids)), anterior_ids, anterior_axes),
        "posterior": (np.arange(len(posterior_ids)) + len(anterior_ids),
                      posterior_ids, posterior_axes),
    }
    point_rows, point_curves = [], []
    region_rows, region_curves = [], []
    v1_1_rows, v1_1_curves = [], []
    region_indices = dict(zip(REGION_NAMES, np.array_split(np.arange(len(anterior_ids)), 3)))

    for wall, (columns, point_ids, axes) in wall_data.items():
        projected = np.einsum("nki,ki->nk", p[:, columns, :], axes)
        support = (pose_valid[:, None] & valid[:, columns] &
                   np.isfinite(p[:, columns]).all(axis=2))
        for location, (column, point_id) in enumerate(zip(columns, point_ids), start=1):
            mask = pose_valid & valid[:, column] & np.isfinite(p[:, column]).all(axis=1)
            extra = dict(wall=wall, region="point", location=location,
                         point_id=int(point_id), point_count=1,
                         point_locations=str(location), aggregation="none")
            rows, curves = _fit_rows(
                case, f"{wall}_point_{location:02d}", t,
                projected[:, location - 1], mask, epoch, spacing, extra)
            point_rows.extend(rows)
            point_curves.extend(curves)

        scopes = {"whole_wall": np.arange(len(point_ids)), **region_indices}
        for region, local_indices in scopes.items():
            selected_columns = columns[local_indices]
            mask = pose_valid.copy()
            mask &= valid[:, selected_columns].all(axis=1)
            mask &= np.isfinite(p[:, selected_columns]).all(axis=(1, 2))
            # Equal-weight arithmetic mean; no point or time-window selection.
            signal_px = projected[:, local_indices].mean(axis=1)
            locations = ";".join(str(int(i) + 1) for i in local_indices)
            extra = dict(wall=wall, region=region, location="", point_id="",
                         point_count=len(local_indices), point_locations=locations,
                         aggregation="equal_weight_mean")
            rows, curves = _fit_rows(
                case, f"{wall}_{region}", t, signal_px, mask, epoch,
                spacing, extra)
            region_rows.extend(rows)
            region_curves.extend(curves)
            rows, curves = _fit_fixed_support_v1_1(
                case, wall, region, t, projected, support, epoch, spacing,
                local_indices, region_indices)
            v1_1_rows.extend(rows)
            v1_1_curves.extend(curves)

    write_csv(out / "point_segments.csv", point_rows)
    write_csv(out / "point_curves_mm.csv", point_curves)
    write_csv(out / "point_primary_summary.csv",
              [row for row in point_rows if row["primary"]])
    write_csv(out / "region_segments.csv", region_rows)
    write_csv(out / "region_curves_mm.csv", region_curves)
    primary = [row for row in region_rows if row["primary"]]
    write_csv(out / "region_primary_summary.csv", primary)
    write_csv(out / "region_v1_1_segments.csv", v1_1_rows)
    write_csv(out / "region_v1_1_curves_mm.csv", v1_1_curves)
    write_csv(out / "region_v1_1_primary_summary.csv",
              [row for row in v1_1_rows if row["primary"]])
    metadata = dict(
        case=case,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        signed_metadata_sha256=hashlib.sha256(signed_path.read_bytes()).hexdigest(),
        mm_per_processed_pixel=spacing,
        method="strict V1 A tracks plus regional analysis layer",
        frequency_estimator="unchanged frozen estimator in processed-pixel units",
        wall_point_order="saved candidate cervix-side to candidate fundus-side order",
        region_split={key: [int(i) + 1 for i in value]
                      for key, value in region_indices.items()},
        whole_wall_locations=list(range(1, len(anterior_ids) + 1)),
        aggregation="equal-weight arithmetic mean; all selected points must be valid",
        v1_1_support_rule=dict(
            output_prefix="region_v1_1_",
            three_point_region="full 3/3 or fixed adjacent 2/3 degraded segment",
            two_point_region="full 2/2 only",
            whole_wall=("full support, or sensitivity-only fixed support with at least "
                        "75 percent of points and coverage of all three regions"),
            membership_change="break segment; reset origin; never concatenate"),
        posterior_axis_source=("saved posterior_inward_axes" if has_posterior_axes
                               else "negative of paired anterior inward axes"),
        interpretation_boundary=(
            "Diagnostic best CPM is preliminary. Empty frequency_cpm means the "
            "frozen repetition rule did not pass. Candidate wall and endpoint "
            "labels are not independently anatomy-verified; no propagation or "
            "physiological contraction is inferred."),
        ignored_trailing_track_columns=max(0, p.shape[1] - expected_count),
        numpy_version=np.__version__)
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return primary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--signed-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mm-per-pixel", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_regions(
        args.input, args.signed_npz, args.out, args.case, args.mm_per_pixel),
        ensure_ascii=False, indent=2))
