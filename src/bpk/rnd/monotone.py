"""Monotone total variance in maturity (calendar no-arbitrage) by isotonic regression.

For maturities T_1 < ... < T_n and each log-moneyness k, calendar no-arbitrage requires
w(k, T_1) <= ... <= w(k, T_n). Given fitted values w_j(k), the least-squares nondecreasing
sequence is the isotonic regression, which for equal weights is
    w*_j = max_{i <= j} min_{l >= j} mean(w_i, ..., w_l)   (min-max formula).
It leaves already-monotone inputs unchanged, is idempotent, and changes values as little as
possible in the squared sense. It is applied to every slice of a date before densities and
interpolation, so interpolation between adjacent maturities never sees a calendar violation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from bpk.rnd.slice import SliceFit

FloatArr = NDArray[np.float64]


def isotonic_columns(w: FloatArr) -> FloatArr:
    """Nondecreasing isotonic fit along axis 0 of an (n_maturities, n_k) array."""
    n = w.shape[0]
    csum = np.vstack([np.zeros((1, w.shape[1])), np.cumsum(w, axis=0)])
    out = np.empty_like(w)
    for j in range(n):
        best = np.full(w.shape[1], -np.inf)
        for i in range(j + 1):
            inner = np.full(w.shape[1], np.inf)
            for ell in range(j, n):
                inner = np.minimum(inner, (csum[ell + 1] - csum[i]) / (ell - i + 1))
            best = np.maximum(best, inner)
        out[j] = best
    return out


def enforce_monotone(slices: list[SliceFit]) -> list[SliceFit]:
    """Return slices (sorted by maturity) whose w_func is monotone in maturity at every k."""
    ordered = sorted(slices, key=lambda s: s.t)
    funcs = [s.w_func for s in ordered]

    def make(idx: int) -> Callable[[FloatArr], FloatArr]:
        def w_func(k: FloatArr) -> FloatArr:
            stack = np.vstack([f(k) for f in funcs])
            return np.asarray(isotonic_columns(stack)[idx], dtype=np.float64)

        return w_func

    return [replace(s, w_func=make(i)) for i, s in enumerate(ordered)]
