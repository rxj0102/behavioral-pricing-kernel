"""Tests for the real-data study (Amendment 3): reconstruction, physical density, SDF fit,
prediction tests and the timing backtest. Synthetic inputs with known answers throughout."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from bpk.backtest import timing
from bpk.data import real
from bpk.estimation import sdf
from bpk.kernel.epk import empirical_kernel, upward_variation
from bpk.kernel.mixture import Mixture, Stats, fit_mixture, residuals
from bpk.physical import fhs
from bpk.predict import oos


def stats_of(mix: Mixture) -> Stats:
    mean, sd, skew, kurt = mix.moments()
    grid = np.linspace(-2.0, 2.0, 400001)
    cdf = mix.cdf(grid)
    q = [float(grid[np.searchsorted(cdf, p)]) for p in (0.1, 0.5, 0.9)]
    lo, hi = mix.cdf(np.array([-0.2, 0.2]))
    return Stats(mean, sd, skew, kurt, q[0], q[1], q[2], float(lo), float(1.0 - hi))


# ---------------------------------------------------------------- reconstruction
def test_mixture_moments_match_simulation() -> None:
    mix = Mixture(0.25, -0.15, 0.20, 0.04, 0.08)
    rng = np.random.default_rng(0)
    pick = rng.random(2_000_000) < mix.w
    draw = np.where(
        pick, rng.normal(mix.m1, mix.s1, pick.size), rng.normal(mix.m2, mix.s2, pick.size)
    )
    mean, sd, skew, kurt = mix.moments()
    z = (draw - draw.mean()) / draw.std()
    assert mean == pytest.approx(draw.mean(), abs=1e-3)
    assert sd == pytest.approx(draw.std(), rel=2e-3)
    assert skew == pytest.approx((z**3).mean(), abs=0.02)
    assert kurt == pytest.approx((z**4).mean() - 3.0, abs=0.06)
    assert mix.forward_shift() == pytest.approx(np.log(np.exp(draw).mean()), abs=1e-3)


def test_fit_recovers_a_known_mixture() -> None:
    truth = Mixture(0.2, -0.18, 0.22, 0.03, 0.09)
    st = stats_of(truth)
    got = fit_mixture(st)
    assert np.abs(residuals(got, st)).max() < 1e-4
    grid = np.linspace(-0.6, 0.4, 50)
    assert np.abs(got.cdf(grid) - truth.cdf(grid)).max() < 2e-3
    assert got.forward_shift() == pytest.approx(truth.forward_shift(), abs=1e-4)


def test_published_statistics_are_reproduced_closely() -> None:
    mpd = real.load().mpd
    rows = mpd[mpd["market"] == "sp6m"].iloc[::60]
    gaps = []
    for _, row in rows.iterrows():
        st = Stats(*(float(row[c]) for c in real.MPD_COLUMNS))
        gaps.append(np.abs(residuals(fit_mixture(st), st)[4:]).max())
    assert np.median(gaps) < 0.01 and max(gaps) < 0.03


def test_large_move_threshold_is_in_log_units() -> None:
    """Fit without the two tail probabilities, then see which reading of 20 percent they match."""
    mpd = real.load().mpd
    rows = mpd[mpd["market"] == "sp6m"].iloc[::60]
    err_log, err_simple = [], []
    for _, row in rows.iterrows():
        st = Stats(*(float(row[c]) for c in real.MPD_COLUMNS))
        mix = fit_mixture(st, tails=False)
        err_log.append(abs(1.0 - mix.cdf(np.array([0.2]))[0] - st.pr_inc))
        err_simple.append(abs(1.0 - mix.cdf(np.array([np.log(1.2)]))[0] - st.pr_inc))
    assert np.mean(err_log) < 0.5 * np.mean(err_simple)


# ---------------------------------------------------------------- physical density
def simulate_gjr(n: int, p: fhs.GjrParams, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = p.omega / (1.0 - p.persistence)
    out = np.empty(n)
    for t in range(n):
        e = np.sqrt(h) * rng.standard_normal()
        out[t] = p.mean + e
        h = p.omega + (p.a + p.g * (e < 0.0)) * e * e + p.b * h
    return out


def test_gjr_fit_recovers_parameters() -> None:
    truth = fhs.GjrParams(2e-6, 0.03, 0.10, 0.90, 3e-4)
    got = fhs.fit_gjr(simulate_gjr(20000, truth, 1))
    assert got.a == pytest.approx(truth.a, abs=0.015)
    assert got.g == pytest.approx(truth.g, abs=0.03)
    assert got.b == pytest.approx(truth.b, abs=0.02)
    z, h_next = fhs.filter_state(simulate_gjr(20000, truth, 1), got)
    assert z.std() == pytest.approx(1.0, abs=0.02) and h_next > 0.0


def test_simulation_matches_the_closed_form_variance_and_drift() -> None:
    p = fhs.GjrParams(2e-6, 0.03, 0.10, 0.90, 0.0)
    z = np.random.default_rng(2).standard_normal(50000)
    z = (z - z.mean()) / z.std()
    h0 = 3.0 * p.omega / (1.0 - p.persistence)
    x = fhs.simulate_horizon(z, h0, p, 4e-4, 126, 40000, np.random.default_rng(3))
    assert np.all(np.diff(x) >= 0.0)
    assert x.mean() == pytest.approx(126 * 4e-4, abs=3e-3)
    assert x.var() == pytest.approx(fhs.variance_forecast(h0, p, 126), rel=0.06)
    calm = fhs.simulate_horizon(z, 0.3 * h0, p, 4e-4, 126, 40000, np.random.default_rng(3))
    assert calm.std() < x.std()
    again = fhs.simulate_horizon(z, h0, p, 4e-4, 126, 40000, np.random.default_rng(3))
    np.testing.assert_array_equal(x, again)


# ---------------------------------------------------------------- SDF models
def make_panel(
    n_dates: int = 12, n: int = 4000, gamma: float = 3.0, alpha: float = 1.0
) -> sdf.Panel:
    """Physical samples are normal; published points come from the model with known parameters."""
    rng = np.random.default_rng(5)
    xs = np.sort(rng.normal(0.03, 0.12, (n_dates, n)), axis=1)
    dd = np.linspace(0.0, 0.4, n_dates)
    sent = np.linspace(-1.0, 1.0, n_dates)
    base = sdf.Panel(xs, np.zeros((n_dates, 5)), np.zeros((n_dates, 5)), dd, sent)
    lw = sdf.log_weights("weight", np.array([gamma, alpha]), base)
    w = np.exp(lw)
    cw = np.cumsum(w, axis=1) / w.sum(axis=1, keepdims=True)
    idx = (np.array([0.05, 0.10, 0.50, 0.90, 0.97]) * n).astype(int)
    points = (xs[:, idx] + xs[:, idx + 1]) / 2.0
    return sdf.Panel(xs, points, cw[:, idx], dd, sent)


def test_prelec_with_alpha_one_is_crra() -> None:
    panel = make_panel()
    a = sdf.losses("crra", np.array([2.5]), panel)
    b = sdf.losses("weight", np.array([2.5, 1.0]), panel)
    np.testing.assert_allclose(a, b, atol=1e-12)
    c = sdf.losses("weight_sentiment", np.array([2.5, 1.0, 0.0]), panel)
    np.testing.assert_allclose(a, c, atol=1e-12)


def test_crra_tilt_matches_the_normal_closed_form() -> None:
    """exp(-gamma x) applied to N(mu, s^2) gives N(mu - gamma s^2, s^2)."""
    rng = np.random.default_rng(6)
    x = np.sort(rng.normal(0.02, 0.15, (1, 200000)), axis=1)
    k = np.array([[-0.3, -0.1, 0.0, 0.1, 0.3]])
    panel = sdf.Panel(x, k, np.zeros((1, 5)), np.zeros(1), np.zeros(1))
    got = sdf.implied_cdf(sdf.log_weights("crra", np.array([4.0]), panel), panel.counts)
    want = norm.cdf(k, 0.02 - 4.0 * 0.15**2, 0.15)
    np.testing.assert_allclose(got, want, atol=4e-3)


def test_true_parameters_give_zero_loss_and_fit_recovers_them() -> None:
    panel = make_panel(gamma=3.0, alpha=0.7)
    assert float(np.max(sdf.losses("weight", np.array([3.0, 0.7]), panel))) < 1e-20
    theta, loss = sdf.fit("weight", panel)
    assert theta == pytest.approx([3.0, 0.7], abs=2e-2) and loss < 1e-8
    _, crra_loss = sdf.fit("crra", panel)
    assert crra_loss > 100 * loss
    alpha = sdf.date_curvature(panel, np.full(panel.x.shape[0], 3.0))
    assert alpha == pytest.approx(np.full(panel.x.shape[0], 0.7), abs=1e-3)


def test_probability_weighting_fattens_both_tails() -> None:
    panel = make_panel()
    plain = sdf.implied_cdf(sdf.log_weights("weight", np.array([0.0, 1.0]), panel), panel.counts)
    bent = sdf.implied_cdf(sdf.log_weights("weight", np.array([0.0, 0.6]), panel), panel.counts)
    assert np.all(bent[:, 0] > plain[:, 0])  # more weight on the worst outcomes
    assert np.all(bent[:, 4] < plain[:, 4])  # and on the best


def test_walk_forward_only_uses_earlier_years() -> None:
    panel = make_panel(n_dates=12, gamma=3.0)
    years = np.repeat(np.arange(2008, 2014), 2)
    wf = sdf.walk_forward(panel, years, 2011, ("none", "crra"))
    assert np.isnan(wf.loss["crra"][years < 2011]).all()
    assert np.isfinite(wf.loss["crra"][years >= 2011]).all()
    changed = sdf.Panel(
        panel.x,
        panel.points,
        np.where(years[:, None] >= 2012, 0.5, panel.probs),
        panel.drawdown,
        panel.sentiment,
    )
    wf2 = sdf.walk_forward(changed, years, 2011, ("none", "crra"))
    assert wf.by_year["crra"][2011] == wf2.by_year["crra"][2011]
    assert wf.by_year["crra"][2012] == wf2.by_year["crra"][2012]
    assert wf.by_year["crra"][2013] != wf2.by_year["crra"][2013]


def test_newey_west_matches_plain_t_without_lags() -> None:
    d = np.random.default_rng(7).normal(0.2, 1.0, 500)
    plain = d.mean() / (d.std() / np.sqrt(len(d)))
    assert sdf.newey_west_t(d, 0) == pytest.approx(plain, rel=1e-9)
    ar = np.zeros(3000)
    e = np.random.default_rng(8).normal(0, 1, 3000)
    for i in range(1, 3000):
        ar[i] = 0.8 * ar[i - 1] + e[i]
    assert abs(sdf.newey_west_t(ar + 0.5, 26)) < abs(sdf.newey_west_t(ar + 0.5, 0))


# ---------------------------------------------------------------- pricing kernel
def test_kernel_is_monotone_under_crra_and_measured_as_such() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(0.03, 0.12, 20000)
    q = Mixture(0.5, 0.03 - 3.0 * 0.12**2, 0.12, 0.03 - 3.0 * 0.12**2, 0.12)  # CRRA tilt, gamma 3
    _, k = empirical_kernel(x, q, 0.0)
    assert upward_variation(k) < 0.05
    slope = np.polyfit(np.linspace(0, 1, len(k)), np.log(k), 1)[0]
    assert slope < 0
    u_shape = Mixture(0.5, 0.0, 0.20, 0.0, 0.20)  # fatter in both tails
    _, k2 = empirical_kernel(x, u_shape, 0.0)
    assert upward_variation(k2) > 0.5
    assert upward_variation(np.array([3.0, 2.0, 1.0])) == 0.0
    assert upward_variation(np.array([1.0, np.e, 1.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------- prediction and timing
def test_expanding_forecast_has_no_look_ahead() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(0, 1, (200, 1))
    y = 0.5 * x[:, 0] + rng.normal(0, 1, 200)
    f = oos.expanding_forecasts(y, x, 60)
    assert np.isnan(f[:60]).all() and np.isfinite(f[60:]).all()
    y2 = y.copy()
    y2[150:] += 100.0
    f2 = oos.expanding_forecasts(y2, x, 60)
    np.testing.assert_array_equal(f[:151], f2[:151])
    mean = oos.expanding_forecasts(y, np.empty((200, 0)), 60)
    assert mean[100] == pytest.approx(y[:100].mean())


def test_oos_r2_and_clark_west() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, (600, 1))
    y = 0.4 * x[:, 0] + rng.normal(0, 1, 600)
    bench = oos.expanding_forecasts(y, np.empty((600, 0)), 60)
    good = oos.compare(y, bench, oos.expanding_forecasts(y, x, 60))
    assert good.r2 > 0.08 and good.p_value < 0.001 and good.n == 540
    noise = oos.compare(y, bench, oos.expanding_forecasts(y, rng.normal(0, 1, (600, 1)), 60))
    assert noise.r2 < 0.01 and noise.p_value > 0.05
    p = oos.placebo_p_value(y, np.empty((600, 0)), x[:, 0], 60, good.r2, 40, 1)
    assert p == 0.0


def test_timing_weights_costs_and_deflated_sharpe() -> None:
    f = np.array([0.01, -0.01, 0.05, 0.01])
    v = np.full(4, 0.002)
    ex = np.array([0.02, 0.03, -0.01, 0.01])
    r, w = timing.timing_returns(f, v, ex, risk_aversion=3.0, max_weight=1.5, cost=0.001)
    np.testing.assert_allclose(w, [1.5, 0.0, 1.5, 1.5])
    np.testing.assert_allclose(r, [0.03, -0.0015, -0.015 - 0.0015, 0.015])
    rng = np.random.default_rng(12)
    x = rng.normal(0.01, 0.04, 240)
    one, hurdle1 = timing.deflated_sharpe(x, 1, 0.0)
    many, hurdle = timing.deflated_sharpe(x, 40, 0.1)
    sr = x.mean() / x.std(ddof=1)
    assert hurdle1 == 0.0 and 0.2 < hurdle < 0.3
    assert one == pytest.approx(norm.cdf(sr * np.sqrt(239) / np.sqrt(1 + 0.5 * sr**2)), abs=0.01)
    assert many < one
    assert timing.sharpe(x) == pytest.approx(np.sqrt(12) * sr)


# ---------------------------------------------------------------- data snapshot
def test_snapshot_is_complete() -> None:
    data = real.load()
    six = data.mpd[data.mpd["market"] == "sp6m"]
    assert len(six) >= 819 and str(six["date"].min().date()) == "2007-01-12"
    assert six[real.MPD_COLUMNS].notna().all().all()
    assert ((six["p10"] < six["p50"]) & (six["p50"] < six["p90"])).all()
    assert data.market.index.is_monotonic_increasing and data.market.index[0].year == 1926
    assert data.sentiment.index[-1].year >= 2025 and data.vix.min() > 5
