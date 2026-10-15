"""Rank-dependent probability weighting and the implied SDF kernel.

Prelec (1998) weighting of a probability p in (0, 1):
    w(p) = exp(-beta (-ln p)^alpha),
    w'(p) = alpha beta (-ln p)^(alpha-1) exp(-beta (-ln p)^alpha) / p.
alpha < 1 gives the inverse-S shape (curvature), beta is elevation.
alpha = beta = 1 gives w(p) = p.

Rank-dependent SDF on the log return y (unnormalized):
    m(y) = exp(-gamma y) * w'(S(y)),   S(y) = P(Y > y) under the physical measure.
The first factor is CRRA with relative risk aversion gamma. The second reweights outcomes
by the derivative of the weighting function at their rank. Because w' is large near
S = 0 and S = 1 when alpha < 1, m is U-shaped in the weighting term, which can make m
non-monotone even though the CRRA factor is decreasing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
_EPS = 1e-12


def prelec_weight_derivative(p: FloatArr, alpha: float, beta: float) -> FloatArr:
    """Derivative w'(p) of the Prelec weighting function, with p clipped away from 0 and 1."""
    pc = np.clip(p, _EPS, 1.0 - _EPS)
    neg_log = -np.log(pc)
    out = alpha * beta * neg_log ** (alpha - 1.0) * np.exp(-beta * neg_log**alpha) / pc
    return np.asarray(out, dtype=np.float64)


def survival_from_density(y: FloatArr, f: FloatArr) -> FloatArr:
    """S(y) = P(Y > y) by reverse cumulative trapezoid on a uniform grid."""
    h = float(y[1] - y[0])
    seg = 0.5 * (f[1:] + f[:-1]) * h
    s = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    return np.asarray(s, dtype=np.float64)


def rdu_kernel(y: FloatArr, f: FloatArr, gamma: float, alpha: float, beta: float) -> FloatArr:
    """Unnormalized rank-dependent SDF m(y) = exp(-gamma y) w'(S(y))."""
    s = survival_from_density(y, f)
    return np.asarray(
        np.exp(-gamma * y) * prelec_weight_derivative(s, alpha, beta), dtype=np.float64
    )
