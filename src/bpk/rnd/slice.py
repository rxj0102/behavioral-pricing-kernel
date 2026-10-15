"""Fit one maturity slice from cleaned OTM quotes.

Quotes are converted to forward-normalized undiscounted prices p = mid * exp(rT) / F and
then to implied total variance. The weight of a point is the inverse of its bid-ask width
in total variance, capped, so wide deep-OTM quotes count less. Raw SVI is used when it
passes the butterfly check, otherwise the convex fit in convex.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from bpk.config import RndConfig
from bpk.rnd.bs import implied_total_variance
from bpk.rnd.convex import fit_convex_masses, prices_from_masses
from bpk.rnd.svi import SviFit, butterfly_g, fit_svi, svi_w

FloatArr = NDArray[np.float64]


@dataclass
class SliceFit:
    """A fitted smile for one (date, maturity)."""

    dte: int
    t: float
    forward: float
    rate: float
    k_lo: float
    k_hi: float
    method: str  # svi or convex
    svi: SviFit | None
    n_points: int
    rmse_vol: float
    butterfly_violations: int  # grid points with g < 0 in the raw SVI fit (0 if none)
    w_func: Callable[[FloatArr], FloatArr]


def fit_slice(q: pd.DataFrame, cfg: RndConfig) -> SliceFit | None:
    """Fit a slice from the cleaned quotes of one (date, dte). None if too few usable points."""
    dte = int(q["dte"].iloc[0])
    t = dte / 365.0
    fwd = float(q["forward"].iloc[0])
    rate = float(q["rf"].iloc[0])
    scale = np.exp(rate * t) / fwd
    k = np.log(q["strike"].to_numpy(dtype=float) / fwd)
    is_call = (q["cp"] == "C").to_numpy(dtype=bool)
    mid = q["mid"].to_numpy(dtype=float) * scale
    bid = q["bid"].to_numpy(dtype=float) * scale
    ask = q["ask"].to_numpy(dtype=float) * scale
    w = implied_total_variance(mid, k, is_call)
    w_hi = implied_total_variance(ask, k, is_call)
    w_lo = implied_total_variance(bid, k, is_call)
    width = w_hi - w_lo
    good = np.isfinite(w) & np.isfinite(width)
    if int(good.sum()) < cfg.min_points:
        return None
    order = np.argsort(k[good])
    k = k[good][order]
    w = w[good][order]
    width = width[good][order]
    is_call = is_call[good][order]
    mid = mid[good][order]
    bid = bid[good][order]
    ask = ask[good][order]
    wt = 1.0 / np.maximum(width, 0.02 * w)
    wt = wt / float(np.max(wt))
    k_lo, k_hi = float(np.min(k)), float(np.max(k))
    svi = fit_svi(k, w, wt, (k_lo - 0.1, k_hi + 0.1))
    kg = np.linspace(k_lo - 0.1, k_hi + 0.1, 400)
    n_viol = int((butterfly_g(kg, svi.params) < -1e-8).sum())
    if svi.butterfly_ok:
        p = svi.params

        def w_func(kk: FloatArr) -> FloatArr:
            return svi_w(kk, p)

        method, model_w = "svi", svi_w(k, p)
    else:
        pw = 1.0 / np.maximum(ask - bid, 1e-4)
        pw = pw / float(np.max(pw))
        v, m = fit_convex_masses(np.exp(k), mid, is_call, pw, smooth=cfg.convex_smooth)
        dense = np.linspace(k_lo, k_hi, 600)
        pm = prices_from_masses(v, m, np.exp(dense), dense >= 0)
        wd = implied_total_variance(pm, dense, dense >= 0)
        ok = np.isfinite(wd)
        kd, wdd = dense[ok], wd[ok]

        def w_func(kk: FloatArr) -> FloatArr:
            return np.asarray(np.interp(kk, kd, wdd), dtype=np.float64)

        method, model_w = "convex", w_func(k)
    rmse = float(np.sqrt(np.mean((np.sqrt(model_w / t) - np.sqrt(w / t)) ** 2)))
    return SliceFit(dte, t, fwd, rate, k_lo, k_hi, method, svi, len(k), rmse, n_viol, w_func)
