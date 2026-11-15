"""Stochastic discount factor models and their fit to published risk-neutral distributions.

Each date has a sorted sample x_1 <= ... <= x_N from the physical density of the horizon
return (log return in excess of the forward). A model gives every sample point a weight
m(x_i). The model-implied risk-neutral distribution function at a point k is the share of
total weight on samples at or below k. The loss on a date is the mean squared gap between
that function and the published one at five points.

Models (log weights):
    none     0
    crra     -gamma x
    habit    -(gamma0 + gamma1 d) x          d: log drawdown from the five year maximum
    weight   -gamma x + ln w'(S)             Prelec w with curvature alpha, beta = 1,
                                             S = physical survival probability of the point
    weight_sentiment   as weight with alpha = clip(alpha0 - delta s, 0.2, 1), s: sentiment

Working on the simulated sample avoids any smoothing of the physical density, and the rank
of a sample point gives its survival probability directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from bpk.models.weighting import prelec_weight_derivative

FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]
ALPHA_BOUNDS = (0.2, 1.5)
MODELS = ("none", "crra", "habit", "weight", "weight_sentiment")
N_PARAMS = {"none": 0, "crra": 1, "habit": 2, "weight": 2, "weight_sentiment": 3}
STARTS: dict[str, tuple[tuple[float, ...], ...]] = {
    "crra": ((2.0,), (6.0,)),
    "habit": ((2.0, 0.0), (4.0, 5.0)),
    "weight": ((2.0, 0.8), (4.0, 1.1)),
    "weight_sentiment": ((2.0, 0.8, 0.0), (4.0, 0.9, 0.1)),
}


@dataclass(frozen=True)
class Panel:
    """Everything the models need, one row per date."""

    x: FloatArr  # (D, N) sorted physical samples
    points: FloatArr  # (D, 5) published points, in the same units as x
    probs: FloatArr  # (D, 5) published cumulative probabilities
    drawdown: FloatArr  # (D,)
    sentiment: FloatArr  # (D,) may contain nan

    def __post_init__(self) -> None:
        if not np.all(np.diff(self.x, axis=1) >= 0.0):
            raise ValueError("samples must be sorted along axis 1")

    @property
    def counts(self) -> IntArr:
        """Number of samples at or below each published point."""
        out = np.empty(self.points.shape, dtype=np.int64)
        for d in range(self.x.shape[0]):
            out[d] = np.searchsorted(self.x[d], self.points[d], side="right")
        return out

    def subset(self, mask: NDArray[np.bool_]) -> Panel:
        return Panel(
            self.x[mask],
            self.points[mask],
            self.probs[mask],
            self.drawdown[mask],
            self.sentiment[mask],
        )


def survival_ranks(n: int) -> FloatArr:
    """Physical survival probability of the i-th smallest of n samples."""
    return np.asarray(1.0 - (np.arange(n) + 0.5) / n, dtype=np.float64)


def log_weights(model: str, theta: FloatArr, panel: Panel) -> FloatArr:
    """Unnormalised log SDF at every sample point, shape (D, N)."""
    x = panel.x
    if model == "none":
        return np.zeros(x.shape, dtype=np.float64)
    if model == "crra":
        return np.asarray(-theta[0] * x, dtype=np.float64)
    if model == "habit":
        gamma = theta[0] + theta[1] * panel.drawdown
        return np.asarray(-gamma[:, None] * x, dtype=np.float64)
    s = survival_ranks(x.shape[1])
    if model == "weight":
        lw_rank = np.log(prelec_weight_derivative(s, float(theta[1]), 1.0))
        return np.asarray(-theta[0] * x + lw_rank[None, :], dtype=np.float64)
    if model == "weight_sentiment":
        alpha = np.clip(theta[1] - theta[2] * panel.sentiment, 0.2, 1.0)
        neg_log = -np.log(s)[None, :]
        a = alpha[:, None]
        # ln w'(S) for Prelec with beta = 1, written out so alpha can differ by date
        lw = np.log(a) + (a - 1.0) * np.log(neg_log) - neg_log**a - np.log(s)[None, :]
        return np.asarray(-theta[0] * x + lw, dtype=np.float64)
    raise ValueError(f"unknown model {model}")


def implied_cdf(lw: FloatArr, counts: IntArr) -> FloatArr:
    """Model-implied risk-neutral distribution function at the published points, shape (D, 5)."""
    w = np.exp(lw - lw.max(axis=1, keepdims=True))
    cw = np.cumsum(w, axis=1)
    padded = np.concatenate([np.zeros((cw.shape[0], 1)), cw], axis=1)
    return np.asarray(np.take_along_axis(padded, counts, axis=1) / cw[:, -1:], dtype=np.float64)


def losses(model: str, theta: FloatArr, panel: Panel, counts: IntArr | None = None) -> FloatArr:
    """Per-date mean squared gap between implied and published distribution functions."""
    c = panel.counts if counts is None else counts
    gap = implied_cdf(log_weights(model, theta, panel), c) - panel.probs
    return np.asarray((gap**2).mean(axis=1), dtype=np.float64)


def _in_bounds(model: str, theta: FloatArr) -> bool:
    if (
        model in ("weight", "weight_sentiment")
        and not ALPHA_BOUNDS[0] <= theta[1] <= ALPHA_BOUNDS[1]
    ):
        return False
    return bool(np.all(np.abs(theta) < 50.0))


def fit(model: str, panel: Panel, warm: FloatArr | None = None) -> tuple[FloatArr, float]:
    """Pooled parameters that minimise the mean loss; returns them with that loss."""
    if model == "none":
        return np.zeros(0), float(losses(model, np.zeros(0), panel).mean())
    counts = panel.counts

    def objective(theta: FloatArr) -> float:
        if not _in_bounds(model, theta):
            return 1e6
        return float(losses(model, theta, panel, counts).mean())

    starts = [np.array(s) for s in STARTS[model]]
    if model == "weight_sentiment":
        # the model nests probability weighting (delta = 0), so start from that solution;
        # without it the search can stall where alpha is clipped at 1 and the loss is flat
        nested, _ = fit("weight", panel)
        starts.insert(0, np.array([nested[0], min(float(nested[1]), 0.99), 0.0]))
    if warm is not None:
        starts.insert(0, warm)
    best_x, best_f = starts[0], np.inf
    for start in starts:
        sol = minimize(
            objective, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10}
        )
        if sol.fun < best_f:
            best_x, best_f = np.asarray(sol.x, dtype=np.float64), float(sol.fun)
    return best_x, best_f


def date_curvature(panel: Panel, gamma: FloatArr) -> FloatArr:
    """alpha_t that minimises each date's loss with gamma held fixed (one value per date)."""
    counts = panel.counts
    out = np.empty(panel.x.shape[0])
    for d in range(panel.x.shape[0]):
        one = Panel(
            panel.x[d : d + 1],
            panel.points[d : d + 1],
            panel.probs[d : d + 1],
            panel.drawdown[d : d + 1],
            panel.sentiment[d : d + 1],
        )
        c = counts[d : d + 1]
        g = float(gamma[d])

        def f(alpha: float, one: Panel = one, c: IntArr = c, g: float = g) -> float:
            return float(losses("weight", np.array([g, alpha]), one, c)[0])

        out[d] = float(minimize_scalar(f, bounds=ALPHA_BOUNDS, method="bounded").x)
    return out


@dataclass(frozen=True)
class WalkForward:
    """Out-of-sample losses and the parameters used for each date."""

    loss: dict[str, FloatArr]  # per date, nan before the first out-of-sample year
    theta: dict[str, FloatArr]  # (D, n_params) parameters applied on each date
    by_year: dict[str, dict[int, list[float]]]


def walk_forward(
    panel: Panel,
    years: IntArr,
    first_year: int,
    models: tuple[str, ...] = MODELS,
    progress: Callable[[str], None] | None = None,
) -> WalkForward:
    """Each January, refit on all earlier dates and apply to that year's dates."""
    n = panel.x.shape[0]
    loss = {m: np.full(n, np.nan) for m in models}
    theta = {m: np.full((n, N_PARAMS[m]), np.nan) for m in models}
    by_year: dict[str, dict[int, list[float]]] = {m: {} for m in models}
    warm: dict[str, FloatArr] = {}
    for year in range(first_year, int(np.max(years)) + 1):
        test = years == year
        for m in models:
            usable = (
                np.isfinite(panel.sentiment) if m == "weight_sentiment" else np.ones(n, dtype=bool)
            )
            train = (years < year) & usable
            apply = test & usable
            if not apply.any():
                continue
            est, _ = fit(m, panel.subset(train), warm.get(m))
            warm[m] = est
            by_year[m][year] = [float(v) for v in est]
            loss[m][apply] = losses(m, est, panel.subset(apply))
            theta[m][apply] = est
            if year == first_year:  # dates before the first test year get the first estimate
                theta[m][(years < year) & usable] = est
        if progress is not None:
            progress(f"{year}: " + ", ".join(f"{m} {by_year[m].get(year, [])}" for m in models[1:]))
    return WalkForward(loss, theta, by_year)


def newey_west_t(d: FloatArr, lags: int) -> float:
    """t-statistic of the mean of d with a Newey-West (Bartlett) variance."""
    x = d[np.isfinite(d)]
    n = len(x)
    e = x - x.mean()
    v = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - lag / (lags + 1.0)) * float(e[lag:] @ e[:-lag]) / n
    return float(x.mean() / np.sqrt(v / n))
