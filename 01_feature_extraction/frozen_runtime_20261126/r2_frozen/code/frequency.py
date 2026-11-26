import numpy as np
from scipy.signal import find_peaks
GRID_CPM = np.round(np.arange(.5, 5.0001, .05), 8)
MIN_CYCLES = 2.0


def segments(mask: np.ndarray, epoch: np.ndarray, seeded: np.ndarray, time: np.ndarray):
    """Every invalid frame, epoch change, seed, or missing timestamp starts a new part."""
    starts = np.flatnonzero(mask & (~np.r_[False, mask[:-1]] |
                    np.r_[True, np.diff(epoch) != 0] | seeded |
                    np.r_[True, np.diff(time) > 1.5 * np.median(np.diff(time))]))
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False]) + 1
    # A reset can occur without an invalid frame.
    breaks = sorted(set(starts.tolist() + ends.tolist()))
    return [(a, b) for a, b in zip(breaks[:-1], breaks[1:])
            if a < len(mask) and mask[a] and b > a]

def fit_frequency(time: np.ndarray, signal: np.ndarray):
    """Use the project's linear-trend + sine/cosine score on one raw curve."""
    span = float(time[-1] - time[0])
    allowed = GRID_CPM[GRID_CPM * span / 60 >= MIN_CYCLES - 1e-9]
    base = dict(candidate_period_s=None, candidate_cpm=None, score=None,
                lag_correlation=None, half_lag_correlation=None,
                alternative_peak_cpm=None, status='period_unclear', reason='')
    if len(allowed) < 3 or len(time) < 20:
        base['reason'] = 'continuous_segment_too_short_for_two_cycles_in_search_range'
        return base, allowed, np.array([]), None, None
    tt = time - time.mean()
    trend_x = np.column_stack((np.ones(len(tt)), tt))
    trend_beta = np.linalg.lstsq(trend_x, signal, rcond=None)[0]
    trend = trend_x @ trend_beta
    resid = signal - trend
    energy = float(resid @ resid)
    if energy / len(time) < 1e-12:
        base['reason'] = 'nearly_constant_after_linear_trend'
        return base, allowed, np.zeros(len(allowed)), trend, None
    scores = []
    for f in allowed:
        phase = 2 * np.pi * tt * f / 60
        design = np.column_stack((trend_x, np.sin(phase), np.cos(phase)))
        fitted = design @ np.linalg.lstsq(design, signal, rcond=None)[0]
        scores.append(np.clip(1 - float(((signal - fitted) ** 2).sum()) / energy, 0, 1))
    scores = np.asarray(scores)
    k = int(np.argmax(scores))
    best = float(allowed[k])
    period = 60 / best
    phase = 2 * np.pi * tt * best / 60
    design = np.column_stack((trend_x, np.sin(phase), np.cos(phase)))
    fitted = design @ np.linalg.lstsq(design, signal, rcond=None)[0]
    # Cycle-to-cycle similarity is descriptive. It is not a validated test.
    lag = np.interp(time + period, time, resid, left=np.nan, right=np.nan)
    half = np.interp(time + period / 2, time, resid, left=np.nan, right=np.nan)
    def corr(x):
        good = np.isfinite(x)
        if good.sum() < 20 or np.std(resid[good]) < 1e-6 or np.std(x[good]) < 1e-6:
            return None
        return float(np.corrcoef(resid[good], x[good])[0, 1])
    same, opposite = corr(lag), corr(half)
    peaks, _ = find_peaks(scores)
    distant = [j for j in peaks if abs(allowed[j] - best) >= max(.3, .15 * best)]
    alternative = max(distant, key=lambda j: scores[j]) if distant else None
    competing = alternative is not None and scores[alternative] >= .9 * scores[k]
    boundary = k == 0 or k == len(allowed) - 1
    reasons = []
    if boundary:
        reasons.append('search_boundary')
    if competing:
        reasons.append('similar_distant_peak')
    if scores[k] < .2:
        reasons.append('weak_improvement_over_linear_trend')
    if same is None or same < .5 or opposite is None or opposite > -.2:
        reasons.append('repetition_not_supported_by_lag_comparison')
    base.update(candidate_period_s=period, candidate_cpm=best, score=float(scores[k]),
                lag_correlation=same, half_lag_correlation=opposite,
                alternative_peak_cpm=float(allowed[alternative]) if alternative is not None else None,
                status='exploratory_repetition_candidate' if not reasons else 'period_unclear',
                reason=';'.join(reasons) if reasons else 'exploratory_rules_passed_not_validated')
    return base, allowed, scores, trend, fitted
