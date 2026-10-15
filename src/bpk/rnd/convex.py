"""Convexity-enforcing fallback: nonnegative probability masses fitted to OTM prices.

Write the risk-neutral law of v = S_T / F as masses m_j >= 0 at nodes v_j. Forward-
normalized undiscounted prices are then
    call(u) = sum_j m_j (v_j - u)^+,   put(u) = sum_j m_j (u - v_j)^+,
which are convex in u by construction (no butterfly arbitrage), with the constraints
sum m_j = 1 and sum m_j v_j = 1 imposed as heavily weighted rows. The fit is weighted
bounded least squares with a second-difference roughness penalty on m (a smoothing
parameter), solved with scipy lsq_linear. This replaces the smoothing spline named in the
project spec, because a spline needs a separate convexity repair step while this
parameterization cannot violate convexity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear

FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]


def fit_convex_masses(
    u: FloatArr,
    price_norm: FloatArr,
    is_call: BoolArr,
    weights: FloatArr,
    smooth: float = 1e-3,
    node_step: float = 0.004,
    u_range: tuple[float, float] = (0.25, 2.5),
) -> tuple[FloatArr, FloatArr]:
    """Return (nodes v_j, masses m_j) of the best convex fit."""
    v = np.arange(u_range[0], u_range[1] + 1e-12, node_step)
    call_rows = np.maximum(v[None, :] - u[:, None], 0.0)
    put_rows = np.maximum(u[:, None] - v[None, :], 0.0)
    a = np.where(is_call[:, None], call_rows, put_rows) * weights[:, None]
    b = price_norm * weights
    n = len(v)
    d2 = np.diff(np.eye(n), 2, axis=0) * smooth
    big = 1e3
    a_full = np.vstack([a, d2, big * np.ones((1, n)), big * v[None, :]])
    b_full = np.concatenate([b, np.zeros(len(d2)), [big, big]])
    sol = lsq_linear(a_full, b_full, bounds=(0.0, np.inf), method="bvls")
    return v, np.asarray(sol.x, dtype=np.float64)


def prices_from_masses(v: FloatArr, m: FloatArr, u: FloatArr, is_call: BoolArr) -> FloatArr:
    """OTM-side forward-normalized prices implied by the masses."""
    call = (m[None, :] * np.maximum(v[None, :] - u[:, None], 0.0)).sum(axis=1)
    put = (m[None, :] * np.maximum(u[:, None] - v[None, :], 0.0)).sum(axis=1)
    return np.asarray(np.where(is_call, call, put), dtype=np.float64)
