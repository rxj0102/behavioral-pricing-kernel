"""Reconstruct a risk-neutral density from published summary statistics.

The Minneapolis Fed publishes, for each date, nine statistics of the option-implied density of
the log return r: mean, standard deviation, skewness, excess kurtosis, three percentiles and
the probabilities of a fall and of a rise of 20 percent or more. A mixture of two normals
(five parameters) is fitted to them by least squares. The mixture is used for two things only:
the forward shift ln E[exp(r)], and pictures of the pricing kernel. Model estimation compares
distribution functions at the published points directly and does not depend on this fit.

The 20 percent thresholds are taken in log-return units (r <= -0.20 and r >= 0.20); a check in
the test suite shows this reading fits the other seven statistics better than ln(0.8), ln(1.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.special import ndtr

FloatArr = NDArray[np.float64]
LARGE_MOVE = 0.20


@dataclass(frozen=True)
class Stats:
    """Published statistics of one density."""

    mu: float
    sd: float
    skew: float
    kurt: float
    p10: float
    p50: float
    p90: float
    pr_dec: float
    pr_inc: float

    def points(self, large_move: float = LARGE_MOVE) -> tuple[FloatArr, FloatArr]:
        """Five (return, cumulative probability) points of the published distribution."""
        x = np.array([-large_move, self.p10, self.p50, self.p90, large_move])
        p = np.array([self.pr_dec, 0.10, 0.50, 0.90, 1.0 - self.pr_inc])
        return x, p


@dataclass(frozen=True)
class Mixture:
    """w N(m1, s1^2) + (1 - w) N(m2, s2^2) for the log return."""

    w: float
    m1: float
    s1: float
    m2: float
    s2: float

    def _parts(self) -> tuple[FloatArr, FloatArr, FloatArr]:
        return (
            np.array([self.w, 1.0 - self.w]),
            np.array([self.m1, self.m2]),
            np.array([self.s1, self.s2]),
        )

    def cdf(self, r: FloatArr) -> FloatArr:
        w, m, s = self._parts()
        z = (np.asarray(r, dtype=np.float64)[..., None] - m) / s
        return np.asarray((ndtr(z) * w).sum(axis=-1), dtype=np.float64)

    def pdf(self, r: FloatArr) -> FloatArr:
        w, m, s = self._parts()
        z = (np.asarray(r, dtype=np.float64)[..., None] - m) / s
        dens = np.exp(-0.5 * z**2) / (s * np.sqrt(2.0 * np.pi))
        return np.asarray((dens * w).sum(axis=-1), dtype=np.float64)

    def moments(self) -> tuple[float, float, float, float]:
        """Mean, standard deviation, skewness and excess kurtosis."""
        w, m, s = self._parts()
        mean = float((w * m).sum())
        d = m - mean
        var = float((w * (s**2 + d**2)).sum())
        m3 = float((w * (d**3 + 3.0 * d * s**2)).sum())
        m4 = float((w * (d**4 + 6.0 * d**2 * s**2 + 3.0 * s**4)).sum())
        sd = float(np.sqrt(var))
        return mean, sd, m3 / sd**3, m4 / var**2 - 3.0

    def forward_shift(self) -> float:
        """ln E[exp(r)]: the log of the risk-neutral mean gross return."""
        w, m, s = self._parts()
        return float(np.log((w * np.exp(m + 0.5 * s**2)).sum()))


def _unpack(theta: FloatArr) -> Mixture:
    w = 1.0 / (1.0 + np.exp(-theta[0]))
    return Mixture(
        float(w), float(theta[1]), float(np.exp(theta[2])), float(theta[3]), float(np.exp(theta[4]))
    )


def residuals(
    mix: Mixture, st: Stats, large_move: float = LARGE_MOVE, tails: bool = True
) -> FloatArr:
    """Scaled gaps between the mixture and the published statistics."""
    mean, sd, skew, kurt = mix.moments()
    x, p = st.points(large_move)
    gap = mix.cdf(x) - p
    if not tails:
        gap = gap[1:4]
    mom = np.array(
        [(mean - st.mu) / st.sd, sd / st.sd - 1.0, (skew - st.skew) / 5.0, (kurt - st.kurt) / 20.0]
    )
    return np.concatenate([mom, gap])


def fit_mixture(st: Stats, large_move: float = LARGE_MOVE, tails: bool = True) -> Mixture:
    """Least-squares fit from a few deterministic starts; returns the best one."""
    best: tuple[float, Mixture] | None = None
    for w0, spread, ratio in ((0.2, 1.5, 2.0), (0.1, 2.5, 2.5), (0.35, 1.0, 1.6), (0.5, 0.5, 1.3)):
        start = np.array(
            [
                np.log(w0 / (1.0 - w0)),
                st.mu - spread * st.sd,
                np.log(st.sd * ratio * 0.7),
                st.mu + 0.3 * spread * st.sd,
                np.log(st.sd * 0.7),
            ]
        )
        sol = least_squares(
            lambda th: residuals(_unpack(th), st, large_move, tails), start, method="lm"
        )
        cost = float(sol.cost)
        if best is None or cost < best[0]:
            best = (cost, _unpack(np.asarray(sol.x, dtype=np.float64)))
    assert best is not None
    return best[1]
