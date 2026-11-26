"""Smoke-test the CUDA sparse LK backend without changing pipeline data."""
import cv2
import numpy as np

from cuda_sparse_lk import calc_optical_flow_pyr_lk


def main():
    rng = np.random.default_rng(20261103)
    previous = rng.integers(0, 256, size=(96, 96), dtype=np.uint8)
    current = np.roll(previous, 1, axis=1)
    points = np.array([[[32.0, 32.0]], [[48.0, 48.0]], [[64.0, 64.0]]], np.float32)
    found, status, _ = calc_optical_flow_pyr_lk(
        previous, current, points, win_size=(21, 21), max_level=3,
        iterations=30,
    )
    if found.shape != points.shape or status.shape != (len(points), 1):
        raise RuntimeError("CUDA LK returned an unexpected result layout")
    print("PASS: CUDA sparse LK executed")


if __name__ == "__main__":
    main()
