"""Physical (real-world) density forecasts by filtered historical simulation.

Model for the daily log excess return y_t of the market:
    y_t = mu + e_t,  e_t = sqrt(h_t) z_t,
    h_t = omega + (a + g 1[e_{t-1} < 0]) e_{t-1}^2 + b h_{t-1}        (GJR-GARCH(1,1)).
The parameters are fitted by Gaussian quasi maximum likelihood with variance targeting. The
horizon return is simulated by drawing the standardised residuals z from their own history
(Barone-Adesi, Engle and Mancini 2008), so no distribution is assumed for the shocks.

Nothing here looks ahead: the forecast for date t uses returns up to the day before t, and
parameters estimated on data up to the end of the previous calendar year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArr = NDArray[np.float64]


@dataclass(frozen=True)
class GjrParams:
    omega: float
    a: float
    g: float
    b: float
    mean: float

    @property
    def persistence(self) -> float:
        return self.a + 0.5 * self.g + self.b


def _variance_path(e: FloatArr, omega: float, a: float, g: float, b: float, h0: float) -> FloatArr:
    """h[t] for t = 0..n, where h[n] is the forecast for the day after the sample."""
    n = len(e)
    h = np.empty(n + 1)
    h[0] = h0
    e2 = e * e
    lev = a + g * (e < 0.0)
    for t in range(n):
        h[t + 1] = omega + lev[t] * e2[t] + b * h[t]
    return h


def fit_gjr(y: FloatArr) -> GjrParams:
    """Gaussian QMLE with omega set by variance targeting."""
    mean = float(y.mean())
    e = y - mean
    var = float(e.var())

    def nll(theta: FloatArr) -> float:
        a, g, b = (float(v) for v in theta)
        pers = a + 0.5 * g + b
        if a < 0.0 or g < 0.0 or b < 0.0 or pers >= 0.9995:
            return 1e12
        h = _variance_path(e, var * (1.0 - pers), a, g, b, var)[:-1]
        return float(0.5 * np.sum(np.log(h) + e * e / h))

    best = None
    for start in ((0.03, 0.08, 0.90), (0.01, 0.12, 0.92), (0.05, 0.05, 0.88)):
        sol = minimize(
            nll, np.array(start), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6}
        )
        if best is None or sol.fun < best.fun:
            best = sol
    assert best is not None
    a, g, b = (float(v) for v in best.x)
    return GjrParams(var * (1.0 - a - 0.5 * g - b), a, g, b, mean)


def filter_state(y: FloatArr, p: GjrParams) -> tuple[FloatArr, float]:
    """Standardised residuals of the sample and the variance forecast for the next day."""
    e = y - p.mean
    var0 = p.omega / (1.0 - p.persistence)
    h = _variance_path(e, p.omega, p.a, p.g, p.b, var0)
    return e / np.sqrt(h[:-1]), float(h[-1])


def simulate_horizon(
    z_pool: FloatArr,
    h_next: float,
    p: GjrParams,
    drift: float,
    horizon: int,
    n_paths: int,
    rng: np.random.Generator,
) -> FloatArr:
    """Sorted simulated sums of `horizon` daily log excess returns, with daily mean `drift`."""
    h = np.full(n_paths, h_next)
    total = np.zeros(n_paths)
    draws = rng.integers(0, len(z_pool), size=(horizon, n_paths))
    for t in range(horizon):
        e = np.sqrt(h) * z_pool[draws[t]]
        total += drift + e
        h = p.omega + (p.a + p.g * (e < 0.0)) * e * e + p.b * h
    return np.sort(total)


def variance_forecast(h_next: float, p: GjrParams, horizon: int) -> float:
    """Expected sum of daily variances over the horizon (closed form for GJR-GARCH)."""
    lr = p.omega / (1.0 - p.persistence)
    k = p.persistence
    return float(horizon * lr + (h_next - lr) * (1.0 - k**horizon) / (1.0 - k))
