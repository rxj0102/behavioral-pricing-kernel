"""Raw SVI total-variance smile with no-arbitrage diagnostics.

    w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + s^2)),   b > 0, |rho| < 1, s > 0,
    a + b s sqrt(1 - rho^2) >= 0.
Butterfly condition (Gatheral and Jacquier 2014): with w' and w'' the derivatives in k,
    g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4)(1/w + 1/4) + w'' / 2 >= 0,
which is equivalent to a nonnegative density. Moment (Lee) bound: b (1 + |rho|) <= 2.
Calendar condition between maturities T1 < T2: w(k, T2) >= w(k, T1) for all k.
The fit is weighted least squares in total variance with several starts; if the fitted
slice violates the butterfly condition a penalized refit is tried, and if that fails the
caller falls back to the convex fit in convex.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

FloatArr = NDArray[np.float64]
_B_MAX = 1.0


def svi_w(k: FloatArr, p: FloatArr) -> FloatArr:
    """Raw SVI total variance at log-moneyness k. p = (a, b, rho, m, s)."""
    a, b, rho, m, s = p
    return np.asarray(a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s)), dtype=np.float64)


def butterfly_g(k: FloatArr, p: FloatArr) -> FloatArr:
    """Gatheral g(k); nonnegative everywhere means no butterfly arbitrage."""
    _, b, rho, m, s = p
    r = np.sqrt((k - m) ** 2 + s * s)
    w = svi_w(k, p)
    w1 = b * (rho + (k - m) / r)
    w2 = b * s * s / r**3
    w = np.maximum(w, 1e-12)
    return np.asarray(
        (1.0 - k * w1 / (2.0 * w)) ** 2 - 0.25 * w1 * w1 * (1.0 / w + 0.25) + 0.5 * w2,
        dtype=np.float64,
    )


@dataclass(frozen=True)
class SviFit:
    """Result of a raw SVI fit."""

    params: FloatArr
    rmse_w: float
    butterfly_ok: bool
    min_g: float
    converged: bool


def _residuals(
    p: FloatArr, k: FloatArr, w: FloatArr, wt: FloatArr, kg: FloatArr, lam: float
) -> FloatArr:
    res = (svi_w(k, p) - w) * wt
    if lam > 0:
        res = np.concatenate([res, lam * np.minimum(butterfly_g(kg, p), 0.0)])
    return np.asarray(res, dtype=np.float64)


def _multistart(
    k: FloatArr, w: FloatArr, wt: FloatArr, kg: FloatArr, lam: float
) -> tuple[FloatArr, bool]:
    wmax = float(np.nanmax(w))
    lo = np.array([-wmax, 1e-6, -0.999, float(np.min(k)) - 0.5, 1e-4])
    hi = np.array([wmax * 2.0, _B_MAX, 0.999, float(np.max(k)) + 0.5, 2.0])
    w_atm = float(np.interp(0.0, k, w))
    best_pen, best_x, best_ok = np.inf, lo, False
    for rho0 in (-0.7, -0.3, 0.2):
        for m0 in (-0.05, 0.0, 0.05):
            for s0 in (0.05, 0.25):
                p0 = np.clip(np.array([0.5 * w_atm, 0.1, rho0, m0, s0]), lo + 1e-9, hi - 1e-9)
                r = least_squares(_residuals, p0, bounds=(lo, hi), args=(k, w, wt, kg, lam))
                pen = float(np.sum(r.fun**2))
                if pen < best_pen:
                    best_pen, best_x, best_ok = pen, r.x, bool(r.success)
    return best_x, best_ok


def fit_svi(k: FloatArr, w: FloatArr, wt: FloatArr, check_range: tuple[float, float]) -> SviFit:
    """Weighted least squares SVI fit with butterfly check over check_range.

    First an unpenalized multistart fit; if it violates the butterfly condition (or has
    nonpositive variance) a refit with a penalty on negative g is tried. The returned
    butterfly_ok tells the caller whether to trust the slice or fall back.
    """
    kg = np.linspace(check_range[0], check_range[1], 400)

    def ok(p: FloatArr) -> bool:
        return bool(np.min(butterfly_g(kg, p)) >= -1e-8 and np.min(svi_w(kg, p)) > 0)

    x, success = _multistart(k, w, wt, kg, 0.0)
    if not ok(x):
        x2, success2 = _multistart(k, w, wt, kg, 1e3)
        x, success = x2, success2
    g = butterfly_g(kg, x)
    rmse = float(np.sqrt(np.average((svi_w(k, x) - w) ** 2, weights=wt**2)))
    return SviFit(x, rmse, ok(x), float(np.min(g)), success)
