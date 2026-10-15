"""Per-date RND pipeline: slices, arbitrage counts, native and constant-horizon densities.

Constant-horizon interpolation (default 30 days): for the two maturities T1 < T* < T2 that
bracket T*, total variance is interpolated linearly in T at fixed log-moneyness k = ln(K/F),
    w*(k) = w1(k) + (T* - T1)/(T2 - T1) (w2(k) - w1(k)),
on the intersection of the observed k ranges, and ln F* is interpolated linearly in T.
The density is then built from w* as for a native slice. Calendar arbitrage is counted as
the number of grid points where w2 < w1 for adjacent maturities.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bpk.config import RndConfig
from bpk.rnd.density import RndEstimate, rnd_from_w
from bpk.rnd.slice import SliceFit, fit_slice

FloatArr = NDArray[np.float64]


@dataclass
class DateResult:
    """RND outputs for one date."""

    date: pd.Timestamp
    slices: dict[int, SliceFit] = field(default_factory=dict)
    native: dict[int, RndEstimate] = field(default_factory=dict)
    horizon: RndEstimate | None = None
    horizon_forward: float = float("nan")
    calendar_violations: dict[tuple[int, int], int] = field(default_factory=dict)
    horizon_bracket: tuple[int, int] | None = None


def calendar_violations(a: SliceFit, b: SliceFit, tol: float = 1e-10) -> int:
    """Grid points on the common k range where total variance falls from a to b."""
    lo, hi = max(a.k_lo, b.k_lo), min(a.k_hi, b.k_hi)
    if hi <= lo:
        return 0
    kg = np.linspace(lo, hi, 200)
    return int((b.w_func(kg) < a.w_func(kg) - tol).sum())


def interpolate_slices(s1: SliceFit, s2: SliceFit, t_star: float) -> tuple[SliceFit, float]:
    """Total-variance interpolation to maturity t_star between two slices."""
    lam = (t_star - s1.t) / (s2.t - s1.t)
    lo, hi = max(s1.k_lo, s2.k_lo), min(s1.k_hi, s2.k_hi)
    fwd = float(np.exp(np.log(s1.forward) + lam * (np.log(s2.forward) - np.log(s1.forward))))

    def w_func(k: FloatArr) -> FloatArr:
        w1, w2 = s1.w_func(k), s2.w_func(k)
        return np.asarray(w1 + lam * (w2 - w1), dtype=np.float64)

    rate = s1.rate + lam * (s2.rate - s1.rate)
    star = SliceFit(
        round(t_star * 365),
        t_star,
        fwd,
        rate,
        lo,
        hi,
        "interp",
        None,
        s1.n_points + s2.n_points,
        float("nan"),
        0,
        w_func,
    )
    return star, fwd


def rnd_for_date(quotes: pd.DataFrame, horizon_days: int, cfg: RndConfig) -> DateResult:
    """Run the full RND pipeline for the cleaned quotes of a single date."""
    res = DateResult(date=pd.Timestamp(quotes["date"].iloc[0]))
    for dte, q in quotes.groupby("dte", sort=True):
        s = fit_slice(q, cfg)
        if s is not None:
            res.slices[int(str(dte))] = s
    days = sorted(res.slices)
    for d1, d2 in itertools.pairwise(days):
        res.calendar_violations[(d1, d2)] = calendar_violations(res.slices[d1], res.slices[d2])
    for d, s in res.slices.items():
        res.native[d] = rnd_from_w(s.w_func, s.k_lo, s.k_hi, cfg)
    below = [d for d in days if d <= horizon_days]
    above = [d for d in days if d >= horizon_days]
    if below and above:
        d1, d2 = below[-1], above[0]
        if d1 == d2:
            res.horizon, res.horizon_bracket = res.native[d1], (d1, d1)
            res.horizon_forward = res.slices[d1].forward
        else:
            star, fwd = interpolate_slices(res.slices[d1], res.slices[d2], horizon_days / 365.0)
            res.horizon = rnd_from_w(star.w_func, star.k_lo, star.k_hi, cfg)
            res.horizon_forward, res.horizon_bracket = fwd, (d1, d2)
    return res
