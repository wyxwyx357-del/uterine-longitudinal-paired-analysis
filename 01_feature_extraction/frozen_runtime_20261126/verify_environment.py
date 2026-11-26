from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


TEXT_SUFFIXES = {".py", ".json", ".csv", ".md", ".txt", ".yml", ".yaml"}


def canonical_bytes(path: Path) -> bytes:
    """Hash repository text canonically as LF, independent of Windows CRLF checkout."""
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return data


def git_blob_sha1(path: Path) -> str:
    data = canonical_bytes(path)
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def main() -> None:
    result = {"python": sys.version.split()[0]}

    import numpy
    import pandas
    import scipy

    result.update(
        numpy=numpy.__version__,
        pandas=pandas.__version__,
        scipy=scipy.__version__,
    )

    try:
        import cv2

        result["opencv"] = cv2.__version__
        result["cuda_devices"] = int(cv2.cuda.getCudaEnabledDeviceCount())
        result["cuda_sparse_pyr_lk_available"] = bool(
            getattr(cv2, "cuda_SparsePyrLKOpticalFlow", None)
            or getattr(cv2.cuda, "SparsePyrLKOpticalFlow_create", None)
        )
    except Exception as exc:
        result["opencv_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from analysis.mse_pregnancy_association.code.measurement.tracking import (
            track_segments,
        )
        from analysis.mse_pregnancy_association.code.record_mse_core import (
            summarize_paired_windows,
        )
        from analysis.mse_pregnancy_association.code.batch_wave_counts import (
            count_map,
        )
        from experiments.step2_salience_sandbox.mechanism_2x2.mechanisms import (
            intervention,
        )

        # Keep imports live so static linters do not erase their validation role.
        _ = (track_segments, summarize_paired_windows, count_map, intervention)
        result["core_imports"] = "ok"
    except Exception as exc:
        result["core_imports"] = f"FAILED: {type(exc).__name__}: {exc}"

    runtime_root = Path(__file__).resolve().parent
    feature_root = runtime_root.parent
    lock = json.loads(
        (runtime_root / "SOURCE_LOCK.json").read_text(encoding="utf-8")
    )

    bad_git = []
    for rel, expected in lock["git_blob_sha1"].items():
        # SOURCE_LOCK git paths are relative to frozen_runtime_20261126 itself.
        path = runtime_root / rel
        if not path.is_file() or git_blob_sha1(path) != expected:
            bad_git.append(rel)

    bad_r2 = []
    for rel, expected in lock["r2_sha256"].items():
        path = runtime_root / "r2_frozen" / rel
        if not path.is_file() or sha256(path) != expected:
            bad_r2.append(rel)

    result["source_lock_git_mismatch"] = bad_git
    result["source_lock_r2_mismatch"] = bad_r2

    orchestration_lock_path = feature_root / "PIPELINE_ORCHESTRATION_LOCK.json"
    bad_orchestration = []
    if orchestration_lock_path.is_file():
        orchestration = json.loads(
            orchestration_lock_path.read_text(encoding="utf-8")
        )
        repo_root = feature_root.parent
        for rel, expected in orchestration.get("git_blob_sha1", {}).items():
            path = repo_root / rel
            if not path.is_file() or git_blob_sha1(path) != expected:
                bad_orchestration.append(rel)
        result["orchestration_lock_present"] = True
    else:
        result["orchestration_lock_present"] = False
        bad_orchestration.append("PIPELINE_ORCHESTRATION_LOCK.json")

    result["orchestration_lock_mismatch"] = bad_orchestration

    r2 = runtime_root / "r2_frozen" / "code" / "verify_cuda_backend.py"
    result["r2_source_present"] = r2.is_file()

    print(json.dumps(result, ensure_ascii=False, indent=2))

    bad = (
        result.get("core_imports") != "ok"
        or result.get("cuda_devices", 0) < 1
        or not result.get("cuda_sparse_pyr_lk_available", False)
        or not r2.is_file()
        or bool(bad_git)
        or bool(bad_r2)
        or bool(bad_orchestration)
    )
    raise SystemExit(2 if bad else 0)


if __name__ == "__main__":
    main()
