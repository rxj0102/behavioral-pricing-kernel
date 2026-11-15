"""Market timing from a return forecast, with costs, and the Deflated Sharpe Ratio."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kurtosis, norm, skew

FloatArr = NDArray[np.float64]
EULER = 0.5772156649015329


def timing_returns(
    forecast: FloatArr,
    variance: FloatArr,
    excess: FloatArr,
    risk_aversion: float = 3.0,
    max_weight: float = 1.5,
    cost: float = 0.0005,
) -> tuple[FloatArr, FloatArr]:
    """Monthly excess returns of weight = clip(forecast / (risk_aversion * variance), 0, max).

    forecast[t] and variance[t] are known before excess[t] is earned. Costs are charged on the
    change in weight. Returns the strategy returns and the weights.
    """
    w = np.clip(forecast / (risk_aversion * variance), 0.0, max_weight)
    turnover = np.abs(np.diff(w, prepend=w[0]))
    return w * excess - cost * turnover, w


def sharpe(r: FloatArr, periods: int = 12) -> float:
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))


def deflated_sharpe(r: FloatArr, n_trials: int, trials_sr_std: float) -> tuple[float, float]:
    """Deflated Sharpe Ratio of Bailey and Lopez de Prado (2014) and its per-period hurdle.

    The hurdle is the Sharpe ratio expected from the best of n_trials skill-free strategies whose
    per-period Sharpe ratios have standard deviation trials_sr_std. The result is the probability
    that the true Sharpe ratio is above the hurdle.
    """
    sr = float(r.mean() / r.std(ddof=1))
    if n_trials > 1:
        hurdle = trials_sr_std * (
            (1.0 - EULER) * norm.ppf(1.0 - 1.0 / n_trials)
            + EULER * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    else:
        hurdle = 0.0
    g3, g4 = float(skew(r)), float(kurtosis(r, fisher=False))
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2, 1e-12))
    z = (sr - hurdle) * np.sqrt(len(r) - 1) / denom
    return float(norm.cdf(z)), float(hurdle)
