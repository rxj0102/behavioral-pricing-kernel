"""Heston pricer against Black-Scholes and structural identities."""

import numpy as np
from scipy.stats import norm

from bpk.data.heston import heston_call


def _bs(f: float, k: np.ndarray, t: float, r: float, var: float) -> np.ndarray:
    d1 = (np.log(f / k) + 0.5 * var * t) / np.sqrt(var * t)
    price: np.ndarray = np.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d1 - np.sqrt(var * t)))
    return price


def test_black_scholes_limit() -> None:
    k = np.array([800.0, 950.0, 1000.0, 1050.0, 1200.0])
    h = heston_call(1000.0, k, 0.25, 0.02, 0.04, 3.0, 0.04, 1e-3, 0.0)
    assert np.abs(h - _bs(1000.0, k, 0.25, 0.02, 0.04)).max() < 1e-4


def test_prices_arbitrage_free_in_strike() -> None:
    k = np.arange(600.0, 1500.0, 10.0)
    c = heston_call(1000.0, k, 0.2, 0.02, 0.05, 3.0, 0.04, 0.5, -0.7)
    assert (c > -1e-8).all()
    assert (np.diff(c) < 0).all()  # decreasing in strike
    assert (np.diff(c, 2) > -1e-6).all()  # convex in strike
