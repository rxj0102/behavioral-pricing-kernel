"""RND module tests: analytic lognormal recovery, SVI, convex fallback, tails, interpolation."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import norm

from bpk.config import RndConfig
from bpk.rnd.bs import black_norm, implied_total_variance
from bpk.rnd.convex import fit_convex_masses, prices_from_masses
from bpk.rnd.density import gev_density, gpd_density, rnd_from_w
from bpk.rnd.grade import grade_estimate
from bpk.rnd.pipeline import calendar_violations, interpolate_slices
from bpk.rnd.slice import SliceFit, fit_slice
from bpk.rnd.svi import butterfly_g, fit_svi, svi_w

FloatArr = NDArray[np.float64]
CFG = RndConfig()
R = 0.02


def _quotes(
    dte: int, w_of_k: Callable[[FloatArr], FloatArr], fwd: float = 1000.0, half: float = 0.01
) -> pd.DataFrame:
    """OTM quotes generated from a total-variance smile (exact prices, small spread)."""
    t = dte / 365.0
    strikes = np.arange(round(0.80 * fwd / 5) * 5, 1.20 * fwd, 5.0)
    k = np.log(strikes / fwd)
    is_call = k >= 0
    p = black_norm(k, w_of_k(k), is_call) * fwd * np.exp(-R * t)
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2020-01-02"),
            "dte": dte,
            "strike": strikes,
            "cp": np.where(is_call, "C", "P"),
            "bid": p - half,
            "ask": p + half,
            "mid": p,
            "forward": fwd,
            "rf": R,
        }
    )


def _lognormal_k(k: FloatArr, sigma: float, t: float) -> FloatArr:
    """Density of k = ln(S_T/F) under Black-Scholes: N(-sigma^2 t / 2, sigma^2 t)."""
    s = sigma * np.sqrt(t)
    return np.asarray(norm.pdf(k, loc=-0.5 * s * s, scale=s), dtype=np.float64)


def test_implied_total_variance_roundtrip() -> None:
    k = np.linspace(-0.3, 0.3, 13)
    w = np.full_like(k, 0.04)
    for call in (True, False):
        flag = np.full(k.shape, call)
        back = implied_total_variance(black_norm(k, w, flag), k, flag)
        ok = np.isfinite(back)
        assert ok.sum() >= 9 and np.allclose(back[ok], 0.04, atol=1e-6)


@pytest.mark.parametrize("dte", [30, 90])
def test_rnd_from_flat_smile_recovers_lognormal(dte: int) -> None:
    sigma, t = 0.20, dte / 365.0
    q = _quotes(dte, lambda k: np.full_like(k, sigma**2 * t))
    s = fit_slice(q, CFG)
    assert s is not None
    est = rnd_from_w(s.w_func, s.k_lo, s.k_hi, CFG)
    exact = _lognormal_k(est.k, sigma, t)
    l1 = float(np.trapezoid(np.abs(est.density - exact), est.k))
    assert l1 < 0.01  # stated tolerance, including GPD tails
    assert abs(float(np.trapezoid(est.density, est.k)) - 1) < 1e-9
    assert abs(est.mass_raw - 1) < 5e-3
    assert abs(est.mean_gross - 1) < 5e-3
    assert est.n_negative == 0 and (est.density >= 0).all()


def test_svi_recovers_known_parameters_and_butterfly_ok() -> None:
    true = np.array([0.01, 0.12, -0.5, 0.02, 0.15])
    k = np.linspace(-0.4, 0.4, 41)
    w = svi_w(k, true)
    fit = fit_svi(k, w, np.ones_like(k), (-0.5, 0.5))
    assert fit.butterfly_ok and fit.rmse_w < 1e-6
    assert np.abs(svi_w(k, fit.params) - w).max() < 1e-5


def test_butterfly_check_flags_arbitrageable_smile() -> None:
    bad = np.array(
        [0.0, 0.9, 0.9, 0.0, 0.01]
    )  # very steep wings with tiny curvature at the minimum
    kg = np.linspace(-0.5, 0.5, 400)
    assert np.min(butterfly_g(kg, bad)) < 0
    good = np.array([0.04, 0.05, -0.3, 0.0, 0.2])
    assert np.min(butterfly_g(kg, good)) > 0


def _flat(level: float) -> Callable[[FloatArr], FloatArr]:
    def f(k: FloatArr) -> FloatArr:
        return np.full_like(k, level)

    return f


def test_calendar_violation_is_counted() -> None:
    def mk(dte: int, level: float) -> SliceFit:
        t = dte / 365.0
        return SliceFit(
            dte,
            t,
            1000.0,
            R,
            -0.2,
            0.2,
            "svi",
            None,
            10,
            0.0,
            0,
            _flat(level),
        )

    assert calendar_violations(mk(30, 0.01), mk(60, 0.02)) == 0
    assert calendar_violations(mk(30, 0.02), mk(60, 0.01)) == 200


def test_total_variance_interpolation_exact_for_flat_vol() -> None:
    sigma = 0.2
    s1 = fit_slice(_quotes(21, lambda k: np.full_like(k, sigma**2 * 21 / 365)), CFG)
    s2 = fit_slice(_quotes(45, lambda k: np.full_like(k, sigma**2 * 45 / 365)), CFG)
    assert s1 is not None and s2 is not None
    star, _ = interpolate_slices(s1, s2, 30 / 365.0)
    k = np.linspace(-0.05, 0.05, 11)
    assert np.allclose(star.w_func(k), sigma**2 * 30 / 365, rtol=1e-3)
    est = rnd_from_w(star.w_func, star.k_lo, star.k_hi, CFG)
    l1 = float(np.trapezoid(np.abs(est.density - _lognormal_k(est.k, sigma, 30 / 365.0)), est.k))
    assert l1 < 0.02


def test_convex_fit_is_convex_nonnegative_and_close_to_lognormal() -> None:
    sigma, t = 0.2, 60 / 365.0
    k = np.linspace(-0.25, 0.25, 51)
    is_call = k >= 0
    p = black_norm(k, np.full_like(k, sigma**2 * t), is_call)
    v, m = fit_convex_masses(np.exp(k), p, is_call, np.ones_like(k))
    assert (m >= 0).all() and abs(m.sum() - 1) < 1e-3 and abs((m * v).sum() - 1) < 1e-3
    u = np.linspace(0.7, 1.4, 200)
    c = prices_from_masses(v, m, u, np.ones_like(u, dtype=bool))
    assert (np.diff(c, 2) > -1e-10).all()  # convex in strike by construction
    assert np.abs(prices_from_masses(v, m, np.exp(k), is_call) - p).max() < 5e-4


def test_gpd_tail_matches_mass_and_density_at_splice() -> None:
    z = np.linspace(0, 20, 400001)
    for xi in (0.0, 0.2, -0.1):
        cfg = RndConfig(tail_rule="fixed", xi_fixed=xi)
        d, got = gpd_density(z, 0.03, 0.5, -1.0, cfg)
        assert got == xi
        assert d[0] == pytest.approx(0.5)
        assert float(np.trapezoid(d, z)) == pytest.approx(0.03, rel=2e-3)


def test_slope_rule_matches_requested_slope_within_bounds() -> None:
    z = np.linspace(0, 1e-3, 5)
    cfg = RndConfig(tail_rule="slope")
    f0, mass, slope = 0.5, 0.02, -30.0
    beta = mass / f0
    _, xi = gpd_density(z, mass, f0, slope, cfg)
    assert xi == pytest.approx(np.clip(-beta * slope - 1, *cfg.xi_bounds))


def test_grading_exact_density_has_zero_error_and_perturbation_is_measured() -> None:
    sigma, t = 0.2, 45 / 365.0
    q = _quotes(45, lambda k: np.full_like(k, sigma**2 * t))
    s = fit_slice(q, CFG)
    assert s is not None
    est = rnd_from_w(s.w_func, s.k_lo, s.k_hi, CFG)
    rows = {r["region"]: r for r in grade_estimate(est, est.density)}
    assert all(float(r["l1"]) == 0.0 for r in rows.values())  # type: ignore[arg-type]
    shifted = np.interp(est.k - 0.01, est.k, est.density)
    tot = next(r["l1"] for r in grade_estimate(est, shifted) if r["region"] == "total")
    assert float(tot) > 0.05  # type: ignore[arg-type]  # a 0.01 location shift of a narrow density is clearly visible


@pytest.mark.parametrize("xi", [0.0, 0.15, -0.1])
def test_gev_tail_matches_mass_and_density_at_splice(xi: float) -> None:
    z = np.linspace(0, 30, 600001)
    d = gev_density(z, 0.03, 0.5, xi)
    assert d[0] == pytest.approx(0.5, rel=1e-6)
    assert float(np.trapezoid(d, z)) == pytest.approx(0.03, rel=2e-3)
    assert (d >= 0).all()
    cfg = RndConfig(tail_rule="gev", xi_fixed=xi)
    d2, got = gpd_density(z[:50], 0.03, 0.5, -1.0, cfg)
    assert got == xi and np.allclose(d2, d[:50])
