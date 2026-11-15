"""Out-of-sample return prediction tests.

Expanding-window OLS forecasts of next month's excess return, compared with a benchmark by the
Campbell-Thompson out-of-sample R squared and the Clark-West (2007) test for nested models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArr = NDArray[np.float64]


def expanding_forecasts(y: FloatArr, x: FloatArr, min_obs: int) -> FloatArr:
    """Forecast of y[t] from an OLS of y[s] on x[s], s < t. x[t] is known before y[t].

    x has shape (T, k) without a constant (k may be 0, giving the expanding mean of y).
    Returns nan for t < min_obs.
    """
    n = len(y)
    out = np.full(n, np.nan)
    z = np.column_stack([np.ones(n), x]) if x.size else np.ones((n, 1))
    for t in range(min_obs, n):
        beta, *_ = np.linalg.lstsq(z[:t], y[:t], rcond=None)
        out[t] = float(z[t] @ beta)
    return out


@dataclass(frozen=True)
class OosResult:
    r2: float
    clark_west: float
    p_value: float
    n: int


def compare(y: FloatArr, bench: FloatArr, model: FloatArr) -> OosResult:
    """Out-of-sample R squared of model against bench, and the one-sided Clark-West test."""
    ok = np.isfinite(bench) & np.isfinite(model)
    yy, b, m = y[ok], bench[ok], model[ok]
    e_b, e_m = yy - b, yy - m
    r2 = 1.0 - float(e_m @ e_m) / float(e_b @ e_b)
    f = e_b**2 - (e_m**2 - (b - m) ** 2)
    stat = float(f.mean() / (f.std(ddof=1) / np.sqrt(len(f))))
    return OosResult(r2, stat, float(1.0 - norm.cdf(stat)), len(yy))


def placebo_p_value(
    y: FloatArr,
    controls: FloatArr,
    signal: FloatArr,
    min_obs: int,
    actual_r2: float,
    n_shuffles: int,
    seed: int,
) -> float:
    """Share of shuffled-signal runs whose out-of-sample R squared is at least the actual one."""
    rng = np.random.default_rng(seed)
    bench = expanding_forecasts(y, controls, min_obs)
    hits = 0
    for _ in range(n_shuffles):
        fake = rng.permutation(signal)
        model = expanding_forecasts(y, np.column_stack([controls, fake]), min_obs)
        hits += compare(y, bench, model).r2 >= actual_r2
    return hits / n_shuffles
