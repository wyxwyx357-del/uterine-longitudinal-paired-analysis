#!/usr/bin/env python3
"""Read-only validation for the generated R2 manifest and its referenced inputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


REQUIRED_COLUMNS = {
    "case_id", "video", "tracking", "nodes", "mm_per_processed_pixel"
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("Manifest contains no patients")

    seen: set[str] = set()
    report_rows: list[dict[str, object]] = []
    for row in source_rows:
        case_id = row["case_id"]
        problems: list[str] = []
        if case_id in seen:
            problems.append("duplicate_case_id")
        seen.add(case_id)

        paths = {key: Path(row[key]) for key in ("video", "tracking", "nodes")}
        for key, path in paths.items():
            if not path.is_absolute():
                problems.append(f"{key}_not_absolute")
            if not path.is_file():
                problems.append(f"{key}_missing")

        frame_count = ""
        fps = ""
        width = ""
        height = ""
        if paths["video"].is_file():
            capture = cv2.VideoCapture(str(paths["video"]))
            try:
                if not capture.isOpened():
                    problems.append("video_open_failed")
                else:
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if (width, height) != (1260, 910):
                        problems.append(
                            f"unsupported_video_geometry:{width}x{height};"
                            "expected_1260x910"
                        )
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        problems.append("video_first_frame_read_failed")
            finally:
                capture.release()

        tracking_keys = ""
        if paths["tracking"].is_file():
            try:
                with np.load(paths["tracking"], allow_pickle=False) as tracking:
                    tracking_keys = "|".join(sorted(tracking.files))
            except Exception as exc:  # report the input issue without altering it
                problems.append(f"tracking_load_failed:{type(exc).__name__}")

        if paths["nodes"].is_file():
            try:
                with paths["nodes"].open("r", encoding="utf-8-sig") as stream:
                    json.load(stream)
            except Exception as exc:  # report the input issue without altering it
                problems.append(f"nodes_load_failed:{type(exc).__name__}")

        report_rows.append({
            "case_id": case_id,
            "status": "ready" if not problems else ";".join(problems),
            "video": str(paths["video"]),
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "tracking": str(paths["tracking"]),
            "tracking_keys": tracking_keys,
            "nodes": str(paths["nodes"]),
        })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)

    ready = sum(row["status"] == "ready" for row in report_rows)
    print(f"total_cases={len(report_rows)}")
    print(f"validated_ready_cases={ready}")
    print(f"validation_failed_cases={len(report_rows) - ready}")
    print(f"report={args.report.resolve()}")
    return 0 if ready == len(report_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

