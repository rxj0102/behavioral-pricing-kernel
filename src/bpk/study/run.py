"""Run the real-data study defined in Amendment 3 of docs/hypotheses.md.

    python -m bpk.study.run            full run, writes results/real and appends to the results log
    python -m bpk.study.run --fast     every 8th date and fewer paths, for a quick check (no log)

Every number in the README comes from results/real/metrics.json, written here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from bpk.backtest import timing
from bpk.data import real
from bpk.estimation import sdf
from bpk.kernel.epk import empirical_kernel, upward_variation
from bpk.kernel.mixture import Stats, fit_mixture, residuals
from bpk.physical import fhs
from bpk.predict import oos
from bpk.utils.results_log import count_specs, log_spec

FloatArr = NDArray[np.float64]
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "real"
LOG = ROOT / "results" / "results_log.csv"
SEED = 20240601
HORIZON = {"sp6m": 126, "sp12m": 252}
FIRST_OOS_YEAR = 2012
GARCH_START = "1963-07-01"
DM_LAGS = 26
MIN_TRAIN_MONTHS = 60
MIN_OOS_MONTHS = 120
N_SHUFFLES = 1000
LABELS = {
    "none": "No risk adjustment",
    "crra": "CRRA",
    "habit": "Habit-style",
    "weight": "Probability weighting",
    "weight_sentiment": "Probability weighting with sentiment",
}


@dataclass(frozen=True)
class Built:
    dates: pd.DatetimeIndex
    panel: sdf.Panel
    shift: FloatArr
    fit_gap: FloatArr  # largest gap between the fitted mixture and the published points
    upvar: FloatArr
    params: dict[int, fhs.GjrParams]
    kernels: FloatArr  # (D, n_grid) pricing kernel on a standardised grid, for the figure


PHYSICAL_VARIANTS = {
    "fhs": "GJR-GARCH filtered historical simulation, sample mean premium (pre-specified)",
    "fhs_zero_premium": "as fhs with a zero expected excess return",
    "fhs_double_premium": "as fhs with twice the sample mean excess return",
    "fhs_gaussian": "GJR-GARCH with Gaussian shocks instead of the empirical residuals",
    "hs_5y": "unfiltered historical simulation: daily excess returns drawn from the last 5 years",
}


def build(
    data: real.RealData, market: str, n_paths: int, step: int, physical: str = "fhs"
) -> Built:
    """Physical samples, published points and covariates for every density date.

    ``physical`` picks the physical density (PHYSICAL_VARIANTS). Only ``fhs`` is
    pre-specified; the others are the post hoc robustness check of
    bpk.study.physical_robustness.
    """
    if physical not in PHYSICAL_VARIANTS:
        raise ValueError(f"unknown physical variant {physical}")
    mk = data.market
    y_all = np.log1p(mk["mkt_rf"].to_numpy(dtype=np.float64))
    days = mk.index
    log_index = np.cumsum(np.log1p((mk["mkt_rf"] + mk["rf"]).to_numpy(dtype=np.float64)))
    g0 = int(days.searchsorted(pd.Timestamp(GARCH_START)))
    rows = data.mpd[(data.mpd["market"] == market) & (data.mpd["date"] <= days[-1])].iloc[::step]
    params: dict[int, fhs.GjrParams] = {}
    xs, pts, prs, dds, sents, shifts, gaps, upv, kern = [], [], [], [], [], [], [], [], []
    grid_u = np.linspace(0.0, 1.0, 80)
    for _, row in rows.iterrows():
        t = pd.Timestamp(row["date"])
        i = int(days.searchsorted(t))  # returns before day i are known
        if t.year not in params:
            end = int(days.searchsorted(pd.Timestamp(year=t.year, month=1, day=1)))
            params[t.year] = fhs.fit_gjr(y_all[g0:end])
        p = params[t.year]
        z, h_next = fhs.filter_state(y_all[g0:i], p)
        rng = np.random.default_rng([SEED, HORIZON[market], t.toordinal()])
        drift = float(y_all[:i].mean())
        if physical == "fhs_zero_premium":
            drift = 0.0
        elif physical == "fhs_double_premium":
            drift = 2.0 * drift
        if physical == "fhs_gaussian":
            z = rng.standard_normal(len(z))
        if physical == "hs_5y":
            pool = y_all[max(i - 1260, 0) : i]
            draws = pool[rng.integers(0, len(pool), size=(HORIZON[market], n_paths))]
            x = np.sort(draws.sum(axis=0))
        else:
            x = fhs.simulate_horizon(z, h_next, p, drift, HORIZON[market], n_paths, rng)
        st = Stats(*(float(row[c]) for c in real.MPD_COLUMNS))
        mix = fit_mixture(st)
        c = mix.forward_shift()
        raw_points, probs = st.points()
        _, k = empirical_kernel(x, mix, c, n_grid=len(grid_u))
        xs.append(x)
        pts.append(raw_points - c)
        prs.append(probs)
        dds.append(float(log_index[max(i - 1260, 0) : i].max() - log_index[i - 1]))
        prev = t.to_period("M") - 1
        sents.append(float(data.sentiment.get(prev, np.nan)))
        shifts.append(c)
        gaps.append(float(np.abs(residuals(mix, st)[4:]).max()))
        upv.append(upward_variation(k))
        kern.append(k)
    return Built(
        pd.DatetimeIndex(rows["date"]),
        sdf.Panel(np.array(xs), np.array(pts), np.array(prs), np.array(dds), np.array(sents)),
        np.array(shifts),
        np.array(gaps),
        np.array(upv),
        params,
        np.array(kern),
    )


def hac_slope(y: FloatArr, x: FloatArr, lags: int) -> tuple[float, float]:
    """OLS slope of y on x (with a constant) and its Newey-West t-statistic."""
    xc = x - x.mean()
    beta = float(xc @ y) / float(xc @ xc)
    e = (y - y.mean()) - beta * xc
    g = xc * e
    n = len(y)
    v = float(g @ g) / n
    for lag in range(1, min(lags, n - 1) + 1):
        v += 2.0 * (1.0 - lag / (lags + 1.0)) * float(g[lag:] @ g[:-lag]) / n
    se = np.sqrt(v / n) / (float(xc @ xc) / n)
    return beta, float(beta / se)


def diebold_mariano(base: FloatArr, model: FloatArr) -> dict[str, float]:
    """Loss differential base minus model; positive mean and t favour the model."""
    d = base - model
    ok = np.isfinite(d)
    t = sdf.newey_west_t(d[ok], DM_LAGS)
    return {
        "mean_diff": float(d[ok].mean()),
        "t": t,
        "p_one_sided": float(1.0 - norm.cdf(t)),
        "n": int(ok.sum()),
    }


def fit_table(
    b: Built, models: tuple[str, ...], say: Any
) -> tuple[dict[str, Any], sdf.WalkForward]:
    """In-sample and out-of-sample fit of every model on one tenor."""
    years = b.dates.year.to_numpy(dtype=np.int64)
    wf = sdf.walk_forward(b.panel, years, FIRST_OOS_YEAR, models, say)
    oos_mask = years >= FIRST_OOS_YEAR
    out: dict[str, Any] = {"models": {}, "n_dates": len(b.dates), "n_oos": int(oos_mask.sum())}
    for m in models:
        usable = (
            np.isfinite(b.panel.sentiment) if m == "weight_sentiment" else np.ones(len(years), bool)
        )
        theta, _ = sdf.fit(m, b.panel.subset(usable))
        ins = sdf.losses(m, theta, b.panel.subset(usable & oos_mask))
        o = wf.loss[m][oos_mask & usable]
        out["models"][m] = {
            "n_params": sdf.N_PARAMS[m],
            "theta_full_sample": [float(v) for v in theta],
            "rmse_in_sample": float(np.sqrt(ins.mean())),
            "rmse_oos": float(np.sqrt(np.nanmean(o))),
            "n_oos": int(np.isfinite(o).sum()),
            "theta_by_year": wf.by_year[m],
        }
    return out, wf


def monthly_frame(b: Built, wf: sdf.WalkForward, data: real.RealData) -> pd.DataFrame:
    """Month-end curvature, sentiment, variance risk premium and next month's excess return."""
    gamma = wf.theta["weight"][:, 0]
    alpha = sdf.date_curvature(b.panel, gamma)
    daily = pd.DataFrame(
        {"curvature": 1.0 - alpha, "upvar": b.upvar, "sentiment": b.panel.sentiment}, index=b.dates
    )
    m = daily.groupby(pd.DatetimeIndex(daily.index).to_period("M")).last()
    mk = data.market
    per = pd.DatetimeIndex(mk.index).to_period("M")
    ret = (1.0 + mk["mkt_rf"]).groupby(per).prod() - 1.0
    logr = pd.Series(np.log1p((mk["mkt_rf"] + mk["rf"]).to_numpy()), index=mk.index)
    rv = (logr**2).rolling(21).sum().groupby(per).last()
    vix = data.vix.groupby(pd.DatetimeIndex(data.vix.index).to_period("M")).last()
    m["vix_var"] = (vix.reindex(m.index) / 100.0) ** 2 / 12.0
    m["vrp"] = m["vix_var"] - rv.reindex(m.index)
    m["next_excess"] = ret.shift(-1).reindex(m.index)
    full = per[-1] if mk.index[-1] == (mk.index[-1] + pd.offsets.BMonthEnd(0)) else per[-1] - 1
    m.loc[m.index >= full, "next_excess"] = np.nan  # the last month has no complete next month
    m.attrs["daily"] = daily.assign(alpha=alpha, gamma=gamma)
    return m


def test_h2(m: pd.DataFrame, data: real.RealData) -> dict[str, Any]:
    out: dict[str, Any] = {}
    d = m.dropna(subset=["sentiment"])
    hist = data.sentiment
    cut = np.array([float(hist[hist.index < p - 1].quantile(2.0 / 3.0)) for p in d.index])
    high = (d["sentiment"].to_numpy() > cut).astype(np.float64)
    for col in ("curvature", "upvar"):
        y = d[col].to_numpy(dtype=np.float64)
        x = d["sentiment"].to_numpy(dtype=np.float64)
        slope, t = hac_slope(y, x, 12)
        half = len(y) // 2
        dslope, dt = hac_slope(y, high, 12)
        out[col] = {
            "slope": slope,
            "t": t,
            "n_months": len(y),
            "slope_first_half": hac_slope(y[:half], x[:half], 12)[0],
            "slope_second_half": hac_slope(y[half:], x[half:], 12)[0],
            "high_minus_rest": dslope,
            "high_minus_rest_t": dt,
            "share_high": float(high.mean()),
            "mean": float(y.mean()),
            "sd": float(y.std(ddof=1)),
        }
    out["first_month"], out["last_month"] = str(d.index[0]), str(d.index[-1])
    return out


def test_h3_h4(m: pd.DataFrame, full_run: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    d = m.dropna(subset=["next_excess", "vrp", "curvature"])
    y = d["next_excess"].to_numpy(dtype=np.float64)
    vrp = d[["vrp"]].to_numpy(dtype=np.float64)
    cur = d[["curvature"]].to_numpy(dtype=np.float64)
    both = np.column_stack([vrp, cur])
    fc = {
        "mean": oos.expanding_forecasts(y, np.empty((len(y), 0)), MIN_TRAIN_MONTHS),
        "vrp": oos.expanding_forecasts(y, vrp, MIN_TRAIN_MONTHS),
        "curvature": oos.expanding_forecasts(y, cur, MIN_TRAIN_MONTHS),
        "vrp_curvature": oos.expanding_forecasts(y, both, MIN_TRAIN_MONTHS),
    }
    n_oos = int(np.isfinite(fc["mean"]).sum())
    h3: dict[str, Any] = {
        "n_months": len(y),
        "n_oos": n_oos,
        "threshold": MIN_OOS_MONTHS,
        "first_oos": str(d.index[MIN_TRAIN_MONTHS]),
        "last_oos": str(d.index[-1]),
        "run": n_oos >= MIN_OOS_MONTHS,
        "comparisons": {},
    }
    pairs = {
        "vrp_curvature vs vrp": ("vrp", "vrp_curvature"),
        "vrp_curvature vs mean": ("mean", "vrp_curvature"),
        "curvature vs mean": ("mean", "curvature"),
        "vrp vs mean": ("mean", "vrp"),
    }
    for name, (bn, mn) in pairs.items():
        r = oos.compare(y, fc[bn], fc[mn])
        h3["comparisons"][name] = {"r2": r.r2, "clark_west": r.clark_west, "p_value": r.p_value}
    key = h3["comparisons"]["vrp_curvature vs vrp"]
    h3["placebo_p"] = oos.placebo_p_value(
        y, vrp, cur[:, 0], MIN_TRAIN_MONTHS, key["r2"], N_SHUFFLES if full_run else 50, SEED
    )
    slope, t = hac_slope(y, cur[:, 0], 3)
    h3["in_sample"] = {"slope": slope, "t": t}

    ok = np.isfinite(fc["mean"])
    var = d["vix_var"].to_numpy(dtype=np.float64)[ok]
    ex = y[ok]
    strat = {k: timing.timing_returns(v[ok], var, ex)[0] for k, v in fc.items()}
    weights = timing.timing_returns(fc["vrp_curvature"][ok], var, ex)[1]
    sr_m = np.array([float(r.mean() / r.std(ddof=1)) for r in strat.values()])
    h4: dict[str, Any] = {
        "n_months": int(ok.sum()),
        "buy_hold_sharpe": timing.sharpe(ex),
        "sharpe": {k: timing.sharpe(r) for k, r in strat.items()},
        "mean_weight": float(weights.mean()),
        "turnover_per_year": float(np.abs(np.diff(weights)).mean() * 12.0),
        "trials_sr_std": float(sr_m.std(ddof=1)),
        "returns": strat["vrp_curvature"],
    }
    diff = strat["vrp_curvature"] - ex
    h4["minus_buy_hold_t"] = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    return h3, h4


def drop_logged_rows(path: Path, phase: str) -> None:
    """Remove this study's earlier rows so a rerun replaces them and the trial count is stable."""
    log = pd.read_csv(path)
    log[log["phase"] != phase].to_csv(path, index=False, lineterminator="\r\n")


def rounded(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): rounded(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v) for v in x]
    if isinstance(x, float | np.floating):
        return float(f"{float(x):.6g}")
    if isinstance(x, np.integer | np.bool_):
        return x.item()
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    full = not args.fast
    n_paths, step = (5000, 1) if full else (1500, 8)
    out_dir = OUT if full else ROOT / "results" / "real_fast"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = real.load()

    def say(msg: str) -> None:
        print(msg, flush=True)

    b6 = build(data, "sp6m", n_paths, step)
    say(f"6 month panel: {len(b6.dates)} dates {b6.dates[0].date()} to {b6.dates[-1].date()}")
    fit6, wf6 = fit_table(b6, sdf.MODELS, say)
    b12 = build(data, "sp12m", n_paths, step)
    fit12, wf12 = fit_table(b12, sdf.MODELS[:4], say)

    years = b6.dates.year.to_numpy()
    dm = {}
    for tenor, wf in (("6m", wf6), ("12m", wf12)):
        dm[tenor] = {
            f"weight vs {base}": diebold_mariano(wf.loss[base], wf.loss["weight"])
            for base in ("crra", "habit")
        }
        dm[tenor]["crra vs none"] = diebold_mariano(wf.loss["none"], wf.loss["crra"])
    dm["6m"]["weight_sentiment vs weight"] = diebold_mariano(
        wf6.loss["weight"], wf6.loss["weight_sentiment"]
    )

    # where each model misses: mean signed gap at the five published points, out of sample
    oos_mask = years >= FIRST_OOS_YEAR
    sub = b6.panel.subset(oos_mask)
    gaps = {}
    for mname in ("none", "crra", "habit", "weight"):
        g = np.zeros(5)
        for d in range(sub.x.shape[0]):
            one = sdf.Panel(
                sub.x[d : d + 1],
                sub.points[d : d + 1],
                sub.probs[d : d + 1],
                sub.drawdown[d : d + 1],
                sub.sentiment[d : d + 1],
            )
            th = wf6.theta[mname][oos_mask][d]
            g += (sdf.implied_cdf(sdf.log_weights(mname, th, one), one.counts) - one.probs)[0]
        gaps[mname] = [float(v) for v in g / sub.x.shape[0]]

    m = monthly_frame(b6, wf6, data)
    daily = m.attrs["daily"]
    h2 = test_h2(m, data)
    h3, h4 = test_h3_h4(m, full)
    strat_returns = h4.pop("returns")

    if full:
        drop_logged_rows(LOG, "real")
        for tenor, ft in (("6m", fit6), ("12m", fit12)):
            for name, r in ft["models"].items():
                log_spec(
                    LOG,
                    phase="real",
                    spec_id=f"sdf_{name}_{tenor}",
                    status="ok",
                    params={"model": name, "tenor": tenor, "first_oos_year": FIRST_OOS_YEAR},
                    metrics={"rmse_oos": r["rmse_oos"], "rmse_in_sample": r["rmse_in_sample"]},
                    note="Amendment 3, pre-specified",
                )
        for name, r in h3["comparisons"].items():
            log_spec(
                LOG,
                phase="real",
                spec_id=f"h3_{name.replace(' ', '_')}",
                status="ok",
                params={"comparison": name, "min_train_months": MIN_TRAIN_MONTHS},
                metrics=r,
                note="Amendment 3, pre-specified",
            )
        for name, s in h4["sharpe"].items():
            log_spec(
                LOG,
                phase="real",
                spec_id=f"h4_timing_{name}",
                status="ok",
                params={"forecast": name, "risk_aversion": 3.0, "max_weight": 1.5, "cost": 0.0005},
                metrics={"sharpe": s},
                note="Amendment 3, pre-specified",
            )
    n_trials = count_specs(LOG)
    dsr, hurdle = timing.deflated_sharpe(strat_returns, n_trials, h4["trials_sr_std"])
    h4.update({"dsr": dsr, "n_trials": n_trials, "hurdle_annual": hurdle * np.sqrt(12.0)})

    alpha = daily["alpha"].to_numpy()
    metrics = rounded(
        {
            "sample": {
                "first_date": str(b6.dates[0].date()),
                "last_date": str(b6.dates[-1].date()),
                "n_dates_6m": len(b6.dates),
                "n_dates_12m": len(b12.dates),
                "n_paths": n_paths,
                "first_oos_year": FIRST_OOS_YEAR,
                "dm_lags": DM_LAGS,
            },
            "reconstruction": {
                "max_gap_median": float(np.median(b6.fit_gap)),
                "max_gap_p95": float(np.percentile(b6.fit_gap, 95)),
                "forward_shift_mean": float(b6.shift.mean()),
            },
            "physical": {
                "garch_first": vars(b6.params[min(b6.params)]) | {"year": min(b6.params)},
                "garch_last": vars(b6.params[max(b6.params)]) | {"year": max(b6.params)},
            },
            "fit": {"6m": fit6, "12m": fit12},
            "dm": dm,
            "gaps_oos_6m": gaps,
            "curvature": {
                "alpha_mean": float(alpha.mean()),
                "alpha_median": float(np.median(alpha)),
                "alpha_p10": float(np.percentile(alpha, 10)),
                "alpha_p90": float(np.percentile(alpha, 90)),
                "share_below_one": float((alpha < 1.0).mean()),
                "share_at_bound": float(((alpha < 0.2001) | (alpha > 1.4999)).mean()),
                "upvar_mean": float(b6.upvar.mean()),
                "share_nonmonotone": float((b6.upvar > 0.01).mean()),
            },
            "h2": h2,
            "h3": h3,
            "h4": h4,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    series = daily.assign(shift=b6.shift, **{f"loss_{k}": v for k, v in wf6.loss.items()})
    series.round(6).to_csv(out_dir / "daily_6m.csv", index_label="date")
    m.round(6).to_csv(out_dir / "monthly.csv", index_label="month")
    if full:
        from bpk.study.figures import make_figures

        make_figures(b6, daily, gaps, out_dir)
    say(json.dumps({k: metrics[k] for k in ("dm", "curvature", "h2", "h3", "h4")}, indent=1))
    for tenor, ft in (("6m", fit6), ("12m", fit12)):
        for name, r in ft["models"].items():
            say(
                f"{tenor} {name:18s} in {r['rmse_in_sample']:.4f} oos {r['rmse_oos']:.4f} "
                f"theta {r['theta_full_sample']}"
            )


if __name__ == "__main__":
    main()
