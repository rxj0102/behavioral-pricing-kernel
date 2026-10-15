"""Noise-injection study for the RND pipeline (synthetic panel).

Noise is applied to the raw quotes BEFORE cleaning, so the cleaning rules are part of what
is tested. The generator's own baseline (spread of max(0.025, 0.4 percent of mid) times
1 to 1.3, 0.05 tick, 0.2 percent crossed quotes, 1 percent missing asks) is present at every
level, including level zero.

Noise types (levels in brackets):
- jitter: independent Gaussian noise added to bid and to ask with standard deviation
  j times the quoted half-spread, re-rounded to the tick (j in 0.5, 1, 2, 4).
- stale: each quote is replaced, with probability p, by the previous business day quote of the
  same (maturity, strike, type) in index points, so it reflects an earlier state (p in
  0.05, 0.10, 0.20).
- drop: each quote is removed with probability d (d in 0.2, 0.4, 0.6).
- combined: all three at once, levels (1, 0.05, 0.2), (2, 0.10, 0.4), (4, 0.20, 0.6).
The same dates and the same raw chains are used for every level (paired comparison).
Variance regimes are terciles of the true state variance over the sampled dates.
Usage: python -m bpk.rnd.noise_study [--config configs/default.yaml] [--n-dates 40]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bpk.config import Config, load_config
from bpk.data.clean import clean_panel
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.rnd.grade import exact_q_density, grade_estimate
from bpk.rnd.pipeline import rnd_for_date
from bpk.rnd.run import rnd_hash
from bpk.utils.seed import make_rng
from bpk.utils.stats_io import write_stats

TAG = "synthetic_heston"
_TICK = 0.05


@dataclass(frozen=True)
class NoiseSpec:
    """One noise configuration."""

    name: str
    kind: str
    level: float
    jitter: float = 0.0
    stale: float = 0.0
    drop: float = 0.0


SPECS: tuple[NoiseSpec, ...] = (
    NoiseSpec("base", "base", 0.0),
    *(NoiseSpec(f"jitter_{j}", "jitter", j, jitter=j) for j in (0.5, 1.0, 2.0, 4.0)),
    *(NoiseSpec(f"stale_{p}", "stale", p, stale=p) for p in (0.05, 0.10, 0.20)),
    *(NoiseSpec(f"drop_{d}", "drop", d, drop=d) for d in (0.2, 0.4, 0.6)),
    NoiseSpec("combined_1", "combined", 1.0, 1.0, 0.05, 0.2),
    NoiseSpec("combined_2", "combined", 2.0, 2.0, 0.10, 0.4),
    NoiseSpec("combined_3", "combined", 3.0, 4.0, 0.20, 0.6),
)


def apply_noise(
    panel: pd.DataFrame, prev: pd.DataFrame, spec: NoiseSpec, rng: np.random.Generator
) -> pd.DataFrame:
    """Return a noisy copy of the raw panel (see module docstring)."""
    df = panel.copy()
    n = len(df)
    if spec.jitter > 0:
        half = ((df["ask"] - df["bid"]).abs() / 2.0).fillna(_TICK)  # missing ask: default width
        nb, na = rng.standard_normal(n), rng.standard_normal(n)
        df["bid"] = np.maximum(np.round((df["bid"] + spec.jitter * half * nb) / _TICK) * _TICK, 0.0)
        df["ask"] = np.round((df["ask"] + spec.jitter * half * na) / _TICK) * _TICK
    if spec.stale > 0:
        hit = rng.random(n) < spec.stale
        key = ["dte", "strike", "cp"]
        old = prev[[*key, "bid", "ask"]].rename(columns={"bid": "bid_old", "ask": "ask_old"})
        merged = df[key].merge(old, on=key, how="left")
        use = hit & merged["bid_old"].notna().to_numpy()
        df.loc[use, "bid"] = merged.loc[use, "bid_old"].to_numpy()
        df.loc[use, "ask"] = merged.loc[use, "ask_old"].to_numpy()
    if spec.drop > 0:
        df = df[rng.random(n) >= spec.drop]
    return df.reset_index(drop=True)


def _raw(
    ad: SyntheticHestonAdapter, date: pd.Timestamp, spot: float, var: float, alpha: float, seed: int
) -> pd.DataFrame:
    rng = make_rng(seed, f"noise_raw_{date.date()}")
    parts = [ad._chain(rng, date, int(d), spot, var, alpha)[0] for d in ad.cfg.maturities_days]
    return pd.concat(parts, ignore_index=True)


def _work(
    args: tuple[
        pd.Timestamp,
        tuple[float, float, float],
        tuple[float, float, float],
        Config,
        tuple[NoiseSpec, ...],
    ],
) -> list[dict[str, object]]:
    date, cur, old, cfg, specs = args
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    seed = cfg.run.seed
    prev_date = date - pd.tseries.offsets.BDay(1)
    panel = _raw(ad, date, cur[0], cur[1], cur[2], seed)
    prev = _raw(ad, prev_date, old[0], old[1], old[2], seed)
    rows: list[dict[str, object]] = []
    for spec in specs:
        rng = make_rng(seed, f"noise_{spec.name}_{date.date()}")
        noisy = apply_noise(panel, prev, spec, rng)
        clean = clean_panel(noisy, cfg.cleaning).quotes
        base = {"spec": spec.name, "kind": spec.kind, "level": spec.level, "date": date}
        if clean.empty:
            rows.append(
                {
                    **base,
                    "row": "date",
                    "n_slices": 0,
                    "horizon_ok": False,
                    "calendar_points": 0,
                    "clean_rows": 0,
                }
            )
            continue
        res = rnd_for_date(clean, cfg.data.horizon_days, cfg.rnd)
        rows.append(
            {
                **base,
                "row": "date",
                "n_slices": len(res.slices),
                "horizon_ok": res.horizon is not None,
                "calendar_points": sum(res.calendar_violations.values()),
                "clean_rows": len(clean),
            }
        )
        items = [(d, e, False) for d, e in res.native.items()]
        if res.horizon is not None:
            items.append((cfg.data.horizon_days, res.horizon, True))
        for dte, e, interp in items:
            truth = exact_q_density(ad, cur[1], dte, cur[2], e.k)
            g = {str(r["region"]): float(r["l1"]) for r in grade_estimate(e, truth)}  # type: ignore[arg-type]
            s = res.slices.get(dte)
            rows.append(
                {
                    **base,
                    "row": "slice",
                    "dte": dte,
                    "interpolated": interp,
                    "l1_total": g["total"],
                    "l1_tail_left": g["tail_ext_left"],
                    "l1_tail_right": g["tail_ext_right"],
                    "rmse_vol": s.rmse_vol if s else np.nan,
                    "method": s.method if s else "interp",
                    "bfly_points": s.butterfly_violations if s else 0,
                    "n_negative": e.n_negative,
                }
            )
    return rows


def _q(x: pd.Series, p: float) -> float:
    return float(np.quantile(x.dropna(), p)) if x.notna().any() else float("nan")


def summarize(raw: pd.DataFrame, n_dates: int) -> pd.DataFrame:
    """One row per noise spec."""
    sl = raw[raw["row"] == "slice"]
    dt = raw[raw["row"] == "date"]
    out = []
    for name, grp in sl.groupby("spec", sort=False):
        nat = grp[~grp["interpolated"].astype(bool)]
        h = grp[grp["interpolated"].astype(bool)]
        d = dt[dt["spec"] == name]
        out.append(
            {
                "spec": name,
                "kind": grp["kind"].iloc[0],
                "level": grp["level"].iloc[0],
                "slices_fitted_share": len(nat) / (n_dates * 5),
                "horizon_ok_share": float(d["horizon_ok"].mean()),
                "l1_native_median": float(nat["l1_total"].median()),
                "l1_native_p90": _q(nat["l1_total"], 0.9),
                "l1_30d_median": float(h["l1_total"].median()) if len(h) else np.nan,
                "l1_30d_p90": _q(h["l1_total"], 0.9),
                "tail_l1_native_median": float(
                    (nat["l1_tail_left"] + nat["l1_tail_right"]).median()
                ),
                "svi_rmse_median": float(nat["rmse_vol"].median()),
                "svi_rmse_p90": _q(nat["rmse_vol"], 0.9),
                "convex_fallback_share": float((nat["method"] == "convex").mean()),
                "slices_with_butterfly_viol": int((nat["bfly_points"] > 0).sum()),
                "butterfly_viol_points": int(nat["bfly_points"].sum()),
                "calendar_viol_points": int(d["calendar_points"].sum()),
                "dates_with_calendar_viol": int((d["calendar_points"] > 0).sum()),
                "negative_density_points": int(grp["n_negative"].sum()),
            }
        )
    return pd.DataFrame(out)


def by_regime(raw: pd.DataFrame, var_by_date: dict[pd.Timestamp, float]) -> pd.DataFrame:
    """90th percentile L1 by noise spec and true-variance tercile."""
    v = pd.Series(var_by_date)
    edges = [float(v.quantile(1 / 3)), float(v.quantile(2 / 3))]
    reg = v.map(lambda x: "low" if x <= edges[0] else ("mid" if x <= edges[1] else "high"))
    sl = raw[raw["row"] == "slice"].copy()
    sl["regime"] = sl["date"].map(reg)
    out = []
    for (name, r), grp in sl.groupby(["spec", "regime"], sort=False):
        nat = grp[~grp["interpolated"].astype(bool)]
        h = grp[grp["interpolated"].astype(bool)]
        out.append(
            {
                "spec": name,
                "regime": r,
                "n_dates": int(grp["date"].nunique()),
                "l1_native_p90": _q(nat["l1_total"], 0.9),
                "l1_30d_p90": _q(h["l1_total"], 0.9),
                "l1_30d_median": float(h["l1_total"].median()) if len(h) else np.nan,
            }
        )
    return pd.DataFrame(out)


def run(cfg: Config, n_dates: int, out: Path, specs: tuple[NoiseSpec, ...] = SPECS) -> pd.DataFrame:
    """Run the study and write tables. Returns the summary."""
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    avail = pd.bdate_range(cfg.data.start, cfg.data.end)
    st = ad.simulate_state(avail)
    rows_of = {
        d: (float(sp), float(v), float(a))
        for d, sp, v, a in zip(st["date"], st["spot"], st["var"], st["alpha_t"], strict=True)
    }
    rng = make_rng(cfg.run.seed, "noise_dates")
    dates = sorted(avail[1:][rng.choice(len(avail) - 1, size=n_dates, replace=False)])
    jobs = [(d, rows_of[d], rows_of[d - pd.tseries.offsets.BDay(1)], cfg, specs) for d in dates]
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        raw = pd.DataFrame([r for rows in ex.map(_work, jobs) for r in rows])
    summ = summarize(raw, n_dates)
    regime = by_regime(raw, {d: rows_of[d][1] for d in dates})
    meta = {
        "source": TAG,
        "config_hash": rnd_hash(cfg),
        "n_dates": n_dates,
        "first_date": dates[0].date().isoformat(),
        "last_date": dates[-1].date().isoformat(),
        "specs": json.dumps([s.name for s in specs]),
        "note": "synthetic panel, method validation only; baseline defects present at all levels",
    }
    write_stats(summ.round(6), out / f"noise_study__{TAG}.csv", meta)
    write_stats(regime.round(6), out / f"noise_study_by_regime__{TAG}.csv", meta)
    raw_out = raw.drop(columns=["row"]).assign(date=lambda d: d["date"].dt.date.astype(str))
    write_stats(raw_out.round(8), out / f"noise_study_raw__{TAG}.csv", meta)
    return summ


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-dates", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pd.set_option("display.width", 250)
    print(run(cfg, args.n_dates, cfg.tables_dir).to_string(index=False))


if __name__ == "__main__":
    main()
