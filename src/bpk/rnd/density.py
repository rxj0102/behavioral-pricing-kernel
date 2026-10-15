"""Risk-neutral density by Breeden-Litzenberger with generalized Pareto tails.

Let v = S_T / F and u = K / F. With forward-normalized undiscounted prices
    call(u) = E(v - u)^+,  put(u) = E(u - v)^+,
the density of v is f_v(u) = call''(u) = put''(u) and the CDF is F_v(u) = put'(u) = 1 + call'(u).
Second differences use the put for u <= 1 and the call for u > 1 (the OTM side), which
avoids cancellation. The density of the log return k = ln v is f_k(k) = f_v(e^k) e^k.

Tail shape: the default rule fixes xi = 0 (exponential tail), chosen a priori. The slope-matching
rule below was tried and scored worse against the exact synthetic density (see
results/results_log.csv, spec ids tail_*); because the synthetic tails are light by construction,
that comparison is NOT evidence about which rule suits real option data.

Tails (Figlewski style, in k): beyond splice points k_L and k_R, where the CDF reaches
tail_alpha from each side (or the edge of the quote range if that comes first), the
density is a generalized Pareto in the outward distance z,
    f(z) = (a / beta) (1 + xi z / beta)^(-1/xi - 1),   a = tail mass at the splice,
with beta = a / f(splice) (matches the density). Under the slope rule xi = -beta s - 1, where
s is the local slope d ln f / dz at the splice, clipped to xi_bounds. The tail
integrates to a exactly, so total mass is 1 up to discretization; the pre-normalization
mass is reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bpk.config import RndConfig
from bpk.rnd.bs import black_norm

FloatArr = NDArray[np.float64]


@dataclass
class RndEstimate:
    """Estimated density of the log return k = ln(S_T / F) on a fixed grid."""

    k: FloatArr
    density: FloatArr
    mass_raw: float
    mean_gross: float  # E[exp(k)], equals 1 for a martingale forward
    k_left: float
    k_right: float
    alpha_left: float
    alpha_right: float
    xi_left: float
    xi_right: float
    n_negative: int
    edge_splice_left: bool
    edge_splice_right: bool


def gpd_density(
    z: FloatArr, mass: float, f0: float, slope: float, cfg: RndConfig
) -> tuple[FloatArr, float]:
    """GPD tail density at outward distances z >= 0. Returns (density, xi).

    beta always matches the density at the splice and the tail mass. xi is either fixed
    (tail_rule fixed, default 0: exponential tail) or matched to the local log-slope
    (tail_rule slope), clipped to xi_bounds. Rule gev uses a GEV tail (gev_density) with
    shape xi_fixed, matched in density and mass at the splice.
    """
    if cfg.tail_rule == "gev":
        return gev_density(z, mass, f0, cfg.xi_fixed), cfg.xi_fixed
    beta = mass / max(f0, 1e-300)
    if cfg.tail_rule == "fixed":
        xi = cfg.xi_fixed
    else:
        xi = float(np.clip(-beta * slope - 1.0, *cfg.xi_bounds))
    if abs(xi) < 1e-6:
        return (mass / beta) * np.exp(-z / beta), xi
    base = np.maximum(1.0 + xi * z / beta, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(base > 0, (mass / beta) * base ** (-1.0 / xi - 1.0), 0.0)
    return np.asarray(dens, dtype=np.float64), xi


def gev_density(z: FloatArr, mass: float, f0: float, xi: float) -> FloatArr:
    """GEV tail density at outward distances z >= 0 with tail mass and density matched at z = 0.

    The tail is the survival of a GEV G(z) = exp(-y^(-1/xi)), y = 1 + xi (z - mu) / s, restricted
    to z >= 0, with shape xi fixed (xi = 0 is Gumbel). With G0 = 1 - mass and
    t = y0^(-1/xi) = -ln G0 at z = 0:
        s = G0 t / (f0 y0),  mu = -s (y0 - 1) / xi,  y0 = t^(-xi)
    (Gumbel: s = G0 t / f0, mu = s ln t). The tail beyond z = 0 then has mass 1 - G0 = mass
    and density f0 at the splice, as the GPD tail.
    """
    g0 = 1.0 - mass
    t = -np.log(g0)
    if abs(xi) < 1e-8:
        s = g0 * t / max(f0, 1e-300)
        mu = s * np.log(t)
        u = np.exp(-(z - mu) / s)
        return np.asarray((1.0 / s) * u * np.exp(-u), dtype=np.float64)
    y0 = t ** (-xi)
    s = g0 * t / (max(f0, 1e-300) * y0)
    mu = -s * (y0 - 1.0) / xi
    y = 1.0 + xi * (z - mu) / s
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(
            y > 0, (1.0 / s) * y ** (-1.0 / xi - 1.0) * np.exp(-(y ** (-1.0 / xi))), 0.0
        )
    return np.asarray(dens, dtype=np.float64)


def _log_slope(k: FloatArr, f: FloatArr, i: int, half: float) -> float:
    """Slope of ln f in k at index i over a +-half window (one-sided at the edges)."""
    a = int(np.searchsorted(k, k[i] - half))
    b = min(int(np.searchsorted(k, k[i] + half)), len(k) - 1)
    a, b = (a, b) if a < i < b else ((i, min(i + 5, len(k) - 1)) if a >= i else (max(i - 5, 0), i))
    if a == b:
        return 0.0
    return float((np.log(max(f[b], 1e-300)) - np.log(max(f[a], 1e-300))) / (k[b] - k[a]))


def rnd_from_w(
    w_func: Callable[[FloatArr], FloatArr], k_lo: float, k_hi: float, cfg: RndConfig
) -> RndEstimate:
    """Density from a total-variance function valid on [k_lo, k_hi]."""
    u = np.arange(np.exp(k_lo), np.exp(k_hi), cfg.u_step)
    k = np.log(u)
    w = np.maximum(w_func(k), 1e-8)
    put = black_norm(k, w, np.zeros_like(k, dtype=bool))
    call = black_norm(k, w, np.ones_like(k, dtype=bool))
    h = cfg.u_step
    f_u = np.zeros_like(u)
    cdf = np.zeros_like(u)
    for arr, side in ((put, u <= 1.0), (call, u > 1.0)):
        d2 = np.zeros_like(u)
        d1 = np.zeros_like(u)
        d2[1:-1] = (arr[2:] - 2 * arr[1:-1] + arr[:-2]) / h**2
        d1[1:-1] = (arr[2:] - arr[:-2]) / (2 * h)
        f_u = np.where(side, d2, f_u)
        cdf = np.where(side, d1 if arr is put else 1.0 + d1, cdf)
    f_k = f_u * u
    n_neg = int((f_k[1:-1] < 0).sum())
    f_k = np.maximum(f_k, 0.0)
    inner = slice(1, len(u) - 1)
    idx = np.arange(len(u))[inner]
    # Left splice: first index where CDF reaches alpha, else the first interior point.
    left_hits = idx[cdf[inner] >= cfg.tail_alpha]
    il = int(left_hits[0]) if len(left_hits) else int(idx[-1])
    right_hits = idx[(1.0 - cdf[inner]) >= cfg.tail_alpha]
    ir = int(right_hits[-1]) if len(right_hits) else int(idx[0])
    edge_l = il == int(idx[0])
    edge_r = ir == int(idx[-1])
    a_l, a_r = float(cdf[il]), float(1.0 - cdf[ir])
    half = cfg.slope_window
    s_l = -_log_slope(k, f_k, il, half)  # outward direction is decreasing k
    s_r = _log_slope(k, f_k, ir, half)
    kg = np.linspace(cfg.k_min, cfg.k_max, cfg.n_k)
    dens = np.interp(kg, k, f_k)
    left = kg < k[il]
    right = kg > k[ir]
    dl, xi_l = gpd_density(k[il] - kg[left], a_l, float(f_k[il]), s_l, cfg)
    dr, xi_r = gpd_density(kg[right] - k[ir], a_r, float(f_k[ir]), s_r, cfg)
    dens[left], dens[right] = dl, dr
    mass = float(np.trapezoid(dens, kg))
    dens = dens / mass
    mean_gross = float(np.trapezoid(np.exp(kg) * dens, kg))
    return RndEstimate(
        kg,
        dens,
        mass,
        mean_gross,
        float(k[il]),
        float(k[ir]),
        a_l,
        a_r,
        xi_l,
        xi_r,
        n_neg,
        edge_l,
        edge_r,
    )
