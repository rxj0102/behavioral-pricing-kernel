"""Run the RND pipeline on a sample of dates of the cleaned synthetic panel and grade it.

Usage: python -m bpk.rnd.run [--config configs/fast.yaml]
Outputs (tables_dir): rnd_fit_diagnostics, rnd_arbitrage, rnd_grading, rnd_grading_summary
(CSV with provenance header) and the 30-day densities (parquet in cache_dir).
Grading needs the generator truth, so it runs only for the synthetic source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bpk.config import Config, load_config, result_hash
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.rnd.grade import exact_q_density, grade_estimate
from bpk.rnd.pipeline import rnd_for_date
from bpk.utils.seed import make_rng
from bpk.utils.stats_io import write_stats

TAG = "synthetic_heston"


def rnd_hash(cfg: Config) -> str:
    """Hash of the data settings plus the rnd section."""
    payload = {"data": result_hash(cfg), "rnd": cfg.rnd.model_dump(mode="json")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def sample_dates(available: pd.DatetimeIndex, cfg: Config) -> list[pd.Timestamp]:
    """Seeded random sample of dates plus the nearest available date to each named date."""
    rng = make_rng(cfg.run.seed, "rnd_sample")
    n = min(cfg.rnd.n_dates, len(available))
    picks = set(available[rng.choice(len(available), size=n, replace=False)])
    for nd in cfg.rnd.named_dates:
        j = int(np.argmin(np.abs((available - pd.Timestamp(nd)).days)))
        picks.add(available[j])
    return sorted(picks)


def _work(
    args: tuple[pd.Timestamp, pd.DataFrame, float, float, Config],
) -> dict[str, list[dict[str, object]]]:
    date, quotes, var, alpha_t, cfg = args
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    res = rnd_for_date(quotes, cfg.data.horizon_days, cfg.rnd)
    fits, arb, grades = [], [], []
    base = {"date": date.date().isoformat()}
    for d, s in res.slices.items():
        e = res.native[d]
        fits.append(
            {
                **base,
                "dte": d,
                "method": s.method,
                "n_points": s.n_points,
                "rmse_vol": s.rmse_vol,
                "butterfly_violations_raw_svi": s.butterfly_violations,
                "mass_raw": e.mass_raw,
                "mean_gross": e.mean_gross,
                "n_negative": e.n_negative,
                "splice_left": e.k_left,
                "splice_right": e.k_right,
                "alpha_left": e.alpha_left,
                "alpha_right": e.alpha_right,
                "xi_left": e.xi_left,
                "xi_right": e.xi_right,
                "edge_splice_left": e.edge_splice_left,
                "edge_splice_right": e.edge_splice_right,
            }
        )
        truth = exact_q_density(ad, var, d, alpha_t, e.k)
        grades += [{**base, "dte": d, "interpolated": False, **r} for r in grade_estimate(e, truth)]
    for (d1, d2), n in res.calendar_violations.items():
        arb.append({**base, "dte_short": d1, "dte_long": d2, "calendar_violations": n})
    if res.horizon is not None:
        h = cfg.data.horizon_days
        truth = exact_q_density(ad, var, h, alpha_t, res.horizon.k)
        grades += [
            {**base, "dte": h, "interpolated": True, **r}
            for r in grade_estimate(res.horizon, truth)
        ]
        fits.append(
            {
                **base,
                "dte": h,
                "method": "interp",
                "mass_raw": res.horizon.mass_raw,
                "mean_gross": res.horizon.mean_gross,
                "n_negative": res.horizon.n_negative,
            }
        )
    dens = (
        [
            {"date": date, "k": float(k), "density": float(v)}
            for k, v in zip(res.horizon.k, res.horizon.density, strict=True)
        ]
        if res.horizon
        else []
    )
    return {"fits": fits, "arb": arb, "grades": grades, "dens": dens}


def summarize(grades: pd.DataFrame) -> pd.DataFrame:
    """Median, mean, 90th percentile and relative L1 across dates by maturity and region."""
    g = grades.assign(rel=lambda d: d["l1"] / d["true_mass"].where(d["true_mass"] > 1e-6))
    out = g.groupby(["dte", "interpolated", "region"], sort=False).agg(
        n=("l1", "size"),
        l1_median=("l1", "median"),
        l1_mean=("l1", "mean"),
        l1_p90=("l1", lambda x: float(np.quantile(x, 0.9))),
        true_mass=("true_mass", "mean"),
        rel_median=("rel", "median"),
    )
    return out.reset_index()


def run(cfg: Config) -> Path:
    """Run the pipeline and write tables. Returns the grading table path."""
    path = cfg.data.cache_dir / f"clean_quotes__{TAG}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the data build first (make data)")
    avail = pd.bdate_range(cfg.data.start, cfg.data.end)
    dates = sample_dates(avail, cfg)
    quotes = pd.read_parquet(path, filters=[("date", "in", dates)])
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    state = ad.simulate_state(avail)
    var_of = dict(zip(state["date"], state["var"].tolist(), strict=True))
    alpha_of = dict(zip(state["date"], state["alpha_t"].tolist(), strict=True))
    jobs = [
        (
            d,
            quotes[quotes["date"] == d],
            var_of[d],
            alpha_of[d],
            cfg,
        )
        for d in dates
    ]
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        outs = list(ex.map(_work, jobs))
    tables = {k: pd.DataFrame([r for o in outs for r in o[k]]) for k in ("fits", "arb", "grades")}
    dens = pd.DataFrame([r for o in outs for r in o["dens"]])
    meta = {
        "source": TAG,
        "config_hash": rnd_hash(cfg),
        "requested_start": cfg.data.start.isoformat(),
        "requested_end": cfg.data.end.isoformat(),
        "n_dates": len(dates),
        "first_date": dates[0].date().isoformat(),
        "last_date": dates[-1].date().isoformat(),
        "horizon_days": cfg.data.horizon_days,
        "note": "synthetic panel, method validation only",
    }
    out = cfg.tables_dir
    write_stats(tables["fits"], out / f"rnd_fit_diagnostics__{TAG}.csv", meta)
    write_stats(tables["arb"], out / f"rnd_arbitrage__{TAG}.csv", meta)
    gpath = out / f"rnd_grading__{TAG}.csv"
    write_stats(tables["grades"].round(8), gpath, meta)
    write_stats(summarize(tables["grades"]).round(6), out / f"rnd_grading_summary__{TAG}.csv", meta)
    cfg.data.cache_dir.mkdir(parents=True, exist_ok=True)
    dens.to_parquet(cfg.data.cache_dir / f"rnd30__{TAG}.parquet")
    return gpath


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    print(run(cfg))


if __name__ == "__main__":
    main()
