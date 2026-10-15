"""Behavioral SDF pricing mode of the synthetic generator: internal consistency."""

from datetime import date

import numpy as np
import pytest

from bpk.config import SdfTruthConfig, SyntheticConfig
from bpk.data.heston import heston_call
from bpk.data.synthetic import _Y, SyntheticHestonAdapter

R, Q = 0.02, 0.015


def _adapter(**sdf: float) -> SyntheticHestonAdapter:
    return SyntheticHestonAdapter(SyntheticConfig(sdf=SdfTruthConfig(**sdf)), seed=1)


@pytest.mark.parametrize("dte", [21, 80, 180])
@pytest.mark.parametrize("alpha", [1.0, 0.9, 0.8])
def test_densities_are_consistent(dte: int, alpha: float) -> None:
    ad = _adapter()
    d = ad.densities(0.04, dte, alpha)
    t = dte / 365.0
    assert abs(float(np.trapezoid(d.physical, _Y)) - 1) < 1e-6
    assert abs(float(np.trapezoid(d.risk_neutral, _Y)) - 1) < 1e-8
    assert (d.physical >= 0).all() and (d.risk_neutral >= 0).all()
    # The same SDF prices the bond and the index.
    assert abs(np.trapezoid(d.physical * d.kernel, _Y) * np.exp(R * t) - 1) < 1e-8
    eq = np.trapezoid(d.risk_neutral * np.exp(d.x), _Y)
    assert abs(eq / np.exp((R - Q) * t) - 1) < 1e-8


def test_crra_kernel_is_monotone_and_weighted_kernel_is_not() -> None:
    crra = _adapter().densities(0.04, 45, 1.0)
    body = crra.physical > 1e-3
    assert (np.diff(crra.kernel)[body[:-1]] < 0).all()
    weighted = _adapter().densities(0.04, 45, 0.8)
    body = weighted.physical > 1e-3
    assert (np.diff(weighted.kernel)[body[:-1]] > 0).any()


def test_trivial_sdf_leaves_physical_density_unchanged() -> None:
    ad = _adapter(gamma=0.0, alpha0=1.0, beta=1.0, delta=0.0)
    d = ad.densities(0.04, 45, 1.0)
    assert np.abs(d.risk_neutral - d.physical).max() < 1e-9


def test_density_inversion_matches_closed_form_calls() -> None:
    """Heston-Q density from Fourier inversion reproduces the semi-closed-form call price."""
    c = SyntheticConfig()
    ad = SyntheticHestonAdapter(c, 1)
    t, v0, s0 = 45 / 365.0, 0.04, 1000.0
    f = ad._inv.density(t, v0, c.kappa, c.kappa * c.theta, c.sigma, c.rho, 0.5)
    fwd = s0 * np.exp((R - Q) * t)
    k = np.array([850.0, 950.0, 1000.0, 1050.0, 1150.0])
    ref = heston_call(fwd, k, t, R, v0, c.kappa, c.theta, c.sigma, c.rho)
    for kk, rr in zip(k, ref, strict=True):
        payoff = np.maximum(fwd * np.exp(_Y) - kk, 0.0)
        price = np.exp(-R * t) * np.trapezoid(payoff * f, _Y)
        assert abs(price - rr) < 2e-3


def test_behavioral_call_prices_match_direct_integration_and_are_arbitrage_free() -> None:
    ad = _adapter()
    s0, v, dte, a = 1000.0, 0.04, 45, 0.85
    k = np.arange(600.0, 1500.0, 10.0)
    calls, xi = ad._behavioral_calls(s0, v, dte, a, k)
    d = ad.densities(v, dte, a)
    t = dte / 365.0
    direct = np.array(
        [
            np.exp(-R * t) * np.trapezoid(np.maximum(s0 * np.exp(d.x) - kk, 0) * d.risk_neutral, _Y)
            for kk in k
        ]
    )
    assert np.abs(calls - direct).max() < 5e-3
    assert xi == pytest.approx(d.xi)
    assert (np.diff(calls) < 0).all()
    assert (np.diff(calls, 2) > -1e-4).all()


def test_truth_parameters_are_stored_and_sentiment_drives_alpha() -> None:
    ad = _adapter(gamma=2.5, alpha0=0.85, delta=0.07)
    p = ad.truth_params()
    sdf = p["sdf"]
    assert isinstance(sdf, dict) and sdf["gamma"] == 2.5 and sdf["alpha0"] == 0.85
    data = ad.generate(date(2019, 1, 2), date(2019, 1, 15))
    expect = np.clip(0.85 - 0.07 * data.truth["sentiment"], 0.5, 1.0)
    assert np.allclose(data.truth["alpha_t"], expect)
    assert set(data.sdf_truth.columns) >= {"date", "dte", "xi", "alpha_t", "gamma", "beta"}
    assert (data.sdf_truth["gamma"] == 2.5).all()


def test_heston_q_mode_still_available() -> None:
    cfg = SyntheticConfig(pricing_mode="heston_q")
    data = SyntheticHestonAdapter(cfg, 1).generate(date(2019, 1, 2), date(2019, 1, 4))
    assert data.sdf_truth["xi"].isna().all()
    assert len(data.panel) > 0
