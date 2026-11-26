import cv2
import numpy as np
from cuda_sparse_lk import calc_optical_flow_pyr_lk


def rigid(a, b):
    x, y = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-x).T @ (b-y))
    fix = np.eye(2)
    fix[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ fix @ vt
    return np.column_stack((rotation.T, y-x @ rotation))

def step(a, b, wall_points):
    mask = np.uint8((a > 12) & (b > 12))*255
    mask = cv2.erode(mask, np.ones((17, 17), np.uint8))
    mask[:125] = 0
    mask[410:] = 0
    mask[:, :20] = 0
    mask[:, 520:] = 0
    region = np.zeros_like(mask)
    cv2.fillConvexPoly(region, cv2.convexHull(np.rint(wall_points).astype(np.int32)), 255)
    region = cv2.dilate(region, np.ones((31, 31), np.uint8))
    mask[region > 0] = 0
    p = cv2.goodFeaturesToTrack(a, maxCorners=180, qualityLevel=.015, minDistance=8,
                                mask=mask, blockSize=7)
    stats = dict(pose_valid=False, training_inliers=0, heldout_count=0,
                 heldout_raw_median_px=np.nan, heldout_corrected_median_px=np.nan)
    if p is None or len(p) < 18:
        return None, stats
    q, ok, _ = calc_optical_flow_pyr_lk(
        a, b, p, win_size=(25, 25), max_level=3, iterations=30)
    if q is None:
        return None, stats
    back, bok, _ = calc_optical_flow_pyr_lk(
        b, a, q, win_size=(25, 25), max_level=3, iterations=30)
    if back is None:
        return None, stats
    good = (ok.ravel().astype(bool) & bok.ravel().astype(bool) &
            (np.linalg.norm(back[:, 0]-p[:, 0], axis=1) < 1.))
    heldout = np.arange(len(p)) % 3 == 0
    train = good & ~heldout
    test = good & heldout
    p, q = p[:, 0].astype(float), q[:, 0].astype(float)
    if train.sum() < 12:
        return None, stats
    _, inlier = cv2.estimateAffinePartial2D(p[train], q[train], method=cv2.RANSAC,
                                           ransacReprojThreshold=1.5, maxIters=1000)
    if inlier is None or inlier.sum() < 8:
        return None, stats
    select = inlier.ravel().astype(bool)
    matrix = rigid(p[train][select], q[train][select])
    predicted = p[test] @ matrix[:, :2].T + matrix[:, 2]
    stats.update(pose_valid=True, training_inliers=int(select.sum()), heldout_count=int(test.sum()))
    if test.any():
        stats.update(heldout_raw_median_px=float(np.median(np.linalg.norm(q[test]-p[test], axis=1))),
                     heldout_corrected_median_px=float(np.median(np.linalg.norm(q[test]-predicted, axis=1))))
    return matrix, stats
