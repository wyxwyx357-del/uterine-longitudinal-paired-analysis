"""CUDA-backed sparse pyramidal Lucas-Kanade optical flow."""
import cv2
import numpy as np


_FLOWS = {}


def _create_flow(win_size, max_level, iterations):
    if cv2.cuda.getCudaEnabledDeviceCount() < 1:
        raise RuntimeError(
            "CUDA-enabled OpenCV with the cudaoptflow module is required"
        )
    cv2.cuda.setDevice(0)
    key = (tuple(win_size), int(max_level), int(iterations))
    if key not in _FLOWS:
        sparse_lk = getattr(cv2, "cuda_SparsePyrLKOpticalFlow", None)
        if sparse_lk is not None:
            _FLOWS[key] = sparse_lk.create(key[0], key[1], key[2], False)
        else:
            factory = getattr(cv2.cuda, "SparsePyrLKOpticalFlow_create", None)
            if factory is None:
                raise RuntimeError(
                    "OpenCV was built without the cudaoptflow module"
                )
            _FLOWS[key] = factory(key[0], key[1], key[2], False)
    return _FLOWS[key]


def _upload(array):
    gpu = cv2.cuda_GpuMat()
    gpu.upload(np.ascontiguousarray(array))
    return gpu


def calc_optical_flow_pyr_lk(
    previous, current, points, *, win_size, max_level, iterations=30
):
    """Return CUDA LK results in the same array layout used by the CPU call."""
    point_count = len(points)
    if point_count == 0:
        return (
            np.empty((0, 1, 2), np.float32),
            np.empty((0, 1), np.uint8),
            np.empty((0, 1), np.float32),
        )
    flow = _create_flow(win_size, max_level, iterations)
    previous_gpu = _upload(previous)
    current_gpu = _upload(current)
    points_gpu = _upload(
        np.asarray(points, np.float32).reshape(1, point_count, 2)
    )
    found_gpu, status_gpu, error_gpu = flow.calc(
        previous_gpu, current_gpu, points_gpu, None
    )
    found = found_gpu.download().reshape(point_count, 1, 2)
    status = status_gpu.download().reshape(point_count, 1)
    error = (
        error_gpu.download().reshape(point_count, 1)
        if error_gpu is not None else None
    )
    return found, status, error
