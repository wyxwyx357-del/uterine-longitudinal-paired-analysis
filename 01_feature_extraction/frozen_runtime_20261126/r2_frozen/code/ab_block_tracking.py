"""Continuous image-block tracking. Saved tracks are seeds, never filled data."""
import cv2
import numpy as np
from cuda_sparse_lk import calc_optical_flow_pyr_lk

PATCH_RADIUS = 10
MIN_NCC = .75
MAX_FB_PX = .75
MIN_PATCH_STD = 2.


def patch(image, point):
    x, y = point
    h, w = image.shape
    if not np.isfinite(point).all() or min(x, y) < PATCH_RADIUS+1 or x >= w-PATCH_RADIUS-1 or y >= h-PATCH_RADIUS-1:
        return None
    return cv2.getRectSubPix(image, (21, 21), (float(x), float(y)))


def ncc(a, b):
    x, y = a.astype(float).ravel(), b.astype(float).ravel()
    x, y = x-x.mean(), y-y.mean()
    norm = np.linalg.norm(x)*np.linalg.norm(y)
    return float(x @ y/norm) if norm > 1e-10 else -1.


def track_step(previous, current, points):
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    found, ok, _ = calc_optical_flow_pyr_lk(
        previous, current, p, win_size=(21, 21), max_level=3, iterations=30)
    if found is None:
        return np.full((len(p), 2), np.nan), np.zeros(len(p), bool), np.full(len(p), -1.), np.full(len(p), np.inf)
    back, bok, _ = calc_optical_flow_pyr_lk(
        current, previous, found, win_size=(21, 21), max_level=3, iterations=30)
    if back is None:
        return found[:, 0], np.zeros(len(p), bool), np.full(len(p), -1.), np.full(len(p), np.inf)
    fb = np.linalg.norm(back[:, 0]-p[:, 0], axis=1)
    good = ok.ravel().astype(bool) & bok.ravel().astype(bool) & (fb <= MAX_FB_PX)
    scores = np.full(len(p), -1.)
    for j in np.flatnonzero(good):
        a, b = patch(previous, p[j, 0]), patch(current, found[j, 0])
        if a is not None and b is not None and min(a.std(), b.std()) >= MIN_PATCH_STD:
            scores[j] = ncc(a, b)
    return found[:, 0], good & (scores >= MIN_NCC), scores, fb
