"""Black formulas in forward-normalized form and implied total variance.

With k = ln(K/F), total variance w = sigma^2 T, d1 = (-k + w/2)/sqrt(w), d2 = d1 - sqrt(w),
the undiscounted option price divided by the forward is
    call = N(d1) - e^k N(d2),   put = e^k N(-d2) - N(-d1).
Out-of-the-money options are priced with their own formula, which avoids the cancellation
that makes in-the-money prices uninformative about w. Both are increasing in w, so w is
recovered by bisection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArr = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
_W_MIN, _W_MAX = 1e-10, 9.0


def black_norm(k: FloatArr, w: FloatArr, is_call: BoolArr) -> FloatArr:
    """Forward-normalized undiscounted Black price (call where is_call else put)."""
    sw = np.sqrt(np.maximum(w, _W_MIN))
    d1 = (-k + 0.5 * w) / sw
    d2 = d1 - sw
    call = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    put = np.exp(k) * norm.cdf(-d2) - norm.cdf(-d1)
    return np.asarray(np.where(is_call, call, put), dtype=np.float64)


def implied_total_variance(
    price_norm: FloatArr, k: FloatArr, is_call: BoolArr, iters: int = 70
) -> FloatArr:
    """Invert black_norm for w by bisection. Prices outside the attainable range give NaN."""
    price_norm = np.asarray(price_norm, dtype=float)
    lo = np.full_like(price_norm, _W_MIN)
    hi = np.full_like(price_norm, _W_MAX)
    top = black_norm(k, hi, is_call)
    bot = black_norm(k, lo, is_call)
    bad = (price_norm >= top) | (price_norm <= bot) | ~np.isfinite(price_norm)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = black_norm(k, mid, is_call) < price_norm
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    out = 0.5 * (lo + hi)
    return np.asarray(np.where(bad, np.nan, out), dtype=np.float64)
