"""Shared statistical helpers for the IWAC analyses.

Pure NumPy/SciPy functions used by ``topic_prevalence.py`` (trend testing,
bootstrap) and ``keyness_bursts.py`` (multiple-comparison correction). Kept
dependency-light and side-effect-free so they are unit-testable in isolation.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def bh_adjust(pvals) -> np.ndarray:
    """Benjamini–Hochberg step-up adjusted p-values (q-values).

    Returns an array aligned with the input. NaN p-values are ignored
    (returned as NaN) and do not count toward the number of tests m.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return q
    pf = p[finite]
    order = np.argsort(pf, kind="mergesort")
    ranked = pf[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[finite] = out
    return q


def mann_kendall(series: Sequence[float]) -> Tuple[float, float, float]:
    """Mann–Kendall trend test (two-sided) with tie correction.

    Non-parametric test for a monotonic trend in an ordered series. Returns
    ``(S, z, p_two_sided)`` where S is the Kendall statistic (positive =
    increasing), z the normal-approximation score, and p the two-sided
    p-value. Series shorter than 4 points, or all-equal, return
    ``(0.0, 0.0, 1.0)`` — too little signal to reject.
    """
    from scipy import stats

    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return 0.0, 0.0, 1.0

    # S = sum of signs of all pairwise differences (j > i).
    s = 0
    for i in range(n - 1):
        s += int(np.sign(x[i + 1:] - x[i]).sum())

    # Variance with tie correction.
    _, counts = np.unique(x, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return float(s), 0.0, 1.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(s), float(z), float(min(1.0, p))


def weighted_least_squares_slope(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float]
) -> float:
    """Slope of a weighted least-squares line y ~ x (weights = reliability,
    e.g. per-year document counts). Returns NaN if degenerate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(x) < 2 or w.sum() <= 0:
        return float("nan")
    wsum = w.sum()
    xbar = np.sum(w * x) / wsum
    ybar = np.sum(w * y) / wsum
    denom = np.sum(w * (x - xbar) ** 2)
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * (x - xbar) * (y - ybar)) / denom)


def bootstrap_mean_ci(
    values, n_boot: int, seed: int, lo: float = 2.5, hi: float = 97.5
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of a 1-D sample (resample rows
    with replacement). ``values`` may be a matrix (rows = observations); the
    mean is taken over rows. Returns ``(nan, nan)`` when n_boot <= 0.

    For a matrix input this returns the per-column CI as two arrays; for a
    vector it returns two floats.
    """
    arr = np.asarray(values, dtype=float)
    if n_boot <= 0 or arr.shape[0] == 0:
        nan = np.full(arr.shape[1:], np.nan) if arr.ndim > 1 else float("nan")
        return nan, nan
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    means = np.empty((n_boot,) + arr.shape[1:], dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = arr[idx].mean(axis=0)
    return np.percentile(means, lo, axis=0), np.percentile(means, hi, axis=0)


__all__ = [
    "bh_adjust",
    "mann_kendall",
    "weighted_least_squares_slope",
    "bootstrap_mean_ci",
]
