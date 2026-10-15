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

Pipeline variants compared on identical noisy quotes (so each change has its own before/after):
- legacy: linear weighted least squares SVI, no maturity monotonicity enforcement.
- robust: spread-weighted SVI, soft-L1 robust loss (scale: one bid-ask spread), no enforcement.
- robust_monotone: robust fit plus isotonic enforcement of monotone total variance in maturity.
Coverage (share of maturity slices and of 30 day horizons that could be fitted) is reported next to
every L1, failures are broken out by variance regime, and an L1 with failures counted at the worst
case value 2 is reported as well, since L1 on successful fits alone flatters the noisy settings.
Bootstrap intervals resample dates. The regime tables use the 152-date grading set by default.
Usage: python -m bpk.rnd.noise_study [--config configs/default.yaml] [--specs a,b] [--tag v2]
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
from bpk.rnd.run import rnd_hash, sample_dates
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


VARIANTS: dict[str, dict[str, object]] = {
    "legacy": {"svi_loss": "linear", "enforce_calendar": False},
    "robust": {"svi_loss": "soft_l1", "enforce_calendar": False},
    "robust_monotone": {"svi_loss": "soft_l1", "enforce_calendar": True},
}
RERUN_SPECS = (
    "base",
    "stale_0.05",
    "stale_0.1",
    "stale_0.2",
    "combined_1",
    "combined_2",
    "combined_3",
)
WORST_L1 = 2.0  # total variation maximum of an L1 distance between two densities


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
        for vname, over in VARIANTS.items():
            base = {
                "spec": spec.name,
                "kind": spec.kind,
                "level": spec.level,
                "variant": vname,
                "date": date,
            }
            if clean.empty:
                rows.append(
                    {
                        **base,
                        "row": "date",
                        "n_slices": 0,
                        "horizon_ok": False,
                        "calendar_points": 0,
                        "calendar_points_after": 0,
                        "mono_adjust_rel": 0.0,
                        "clean_rows": 0,
                    }
                )
                continue
            res = rnd_for_date(clean, cfg.data.horizon_days, cfg.rnd.model_copy(update=over))
            rows.append(
                {
                    **base,
                    "row": "date",
                    "n_slices": len(res.slices),
                    "horizon_ok": res.horizon is not None,
                    "calendar_points": sum(res.calendar_violations.values()),
                    "calendar_points_after": sum(res.calendar_violations_after.values()),
                    "mono_adjust_rel": res.mono_adjust_rel,
                    "clean_rows": len(clean),
                }
            )
            items = [(d, e, False) for d, e in res.native.items()]
            if res.horizon is not None:
                items.append((cfg.data.horizon_days, res.horizon, True))
            for dte, e, interp in items:
                truth = exact_q_density(ad, cur[1], dte, cur[2], e.k)
                g = {str(r["region"]): float(r["l1"]) for r in grade_estimate(e, truth)}  # type: ignore[arg-type]
                sl = res.slices.get(dte)
                rows.append(
                    {
                        **base,
                        "row": "slice",
                        "dte": dte,
                        "interpolated": interp,
                        "l1_total": g["total"],
                        "l1_tail_left": g["tail_ext_left"],
                        "l1_tail_right": g["tail_ext_right"],
                        "rmse_vol": sl.rmse_vol if sl else np.nan,
                        "method": sl.method if sl else "interp",
                        "bfly_points": sl.butterfly_violations if sl else 0,
                        "n_negative": e.n_negative,
                    }
                )
    return rows


def _q(x: pd.Series, p: float) -> float:
    return float(np.quantile(x.dropna(), p)) if x.notna().any() else float("nan")


def _boot_p90(
    per_date: list[np.ndarray], rng: np.random.Generator, b: int = 400
) -> tuple[float, float]:
    """95 percent bootstrap interval of the 90th percentile, resampling dates."""
    n = len(per_date)
    if n < 3:
        return float("nan"), float("nan")
    stats = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        vals = np.concatenate([per_date[i] for i in idx])
        stats.append(np.quantile(vals, 0.9) if len(vals) else np.nan)
    lo, hi = np.nanquantile(stats, [0.025, 0.975])
    return float(lo), float(hi)


def _cell(
    grp: pd.DataFrame, dts: pd.DataFrame, n_dates: int, n_mat: int, rng: np.random.Generator
) -> dict[str, object]:
    """Metrics for one (variant, spec[, regime]) cell. grp: slice rows, dts: date rows."""
    nat = grp[~grp["interpolated"].astype(bool)]
    h = grp[grp["interpolated"].astype(bool)]
    failed_h = int(n_dates - len(h))
    h_by_date = [g["l1_total"].to_numpy() for _, g in h.groupby("date")]
    h_all = np.concatenate([*h_by_date, np.full(failed_h, WORST_L1)]) if n_dates else np.array([])
    lo, hi = _boot_p90(h_by_date, rng)
    return {
        "n_dates": n_dates,
        "slice_coverage": len(nat) / (n_dates * n_mat) if n_dates else np.nan,
        "horizon_coverage": len(h) / n_dates if n_dates else np.nan,
        "failed_slices": n_dates * n_mat - len(nat),
        "failed_horizons": failed_h,
        "l1_native_median": float(nat["l1_total"].median()),
        "l1_native_p90": _q(nat["l1_total"], 0.9),
        "l1_30d_median": float(h["l1_total"].median()) if len(h) else np.nan,
        "l1_30d_p90": _q(h["l1_total"], 0.9),
        "l1_30d_p90_ci_lo": lo,
        "l1_30d_p90_ci_hi": hi,
        "l1_30d_p90_failures_at_max": float(np.quantile(h_all, 0.9)) if len(h_all) else np.nan,
        "svi_rmse_median": float(nat["rmse_vol"].median()),
        "convex_fallback_share": float((nat["method"] == "convex").mean()) if len(nat) else np.nan,
        "butterfly_viol_points": int(nat["bfly_points"].sum()),
        "calendar_points_before": int(dts["calendar_points"].sum()),
        "calendar_points_after": int(dts["calendar_points_after"].sum()),
        "mono_adjust_rel_mean": float(dts["mono_adjust_rel"].mean()),
        "negative_points_native": int(nat["n_negative"].sum()),
        "negative_points_30d": int(h["n_negative"].sum()),
    }


def summarize(raw: pd.DataFrame, n_mat: int, seed: int) -> pd.DataFrame:
    """One row per (variant, spec)."""
    sl = raw[raw["row"] == "slice"]
    dt = raw[raw["row"] == "date"]
    rng = make_rng(seed, "noise_boot")
    out = []
    for (v, name), grp in sl.groupby(["variant", "spec"], sort=False):
        d = dt[(dt["variant"] == v) & (dt["spec"] == name)]
        out.append(
            {
                "variant": v,
                "spec": name,
                "kind": grp["kind"].iloc[0],
                "level": grp["level"].iloc[0],
                **_cell(grp, d, d["date"].nunique(), n_mat, rng),
            }
        )
    return pd.DataFrame(out)


def by_regime(
    raw: pd.DataFrame, var_by_date: dict[pd.Timestamp, float], n_mat: int, seed: int
) -> pd.DataFrame:
    """Coverage, failures and 90th percentile L1 by variant, spec and true-variance tercile."""
    v = pd.Series(var_by_date)
    edges = [float(v.quantile(1 / 3)), float(v.quantile(2 / 3))]
    reg = v.map(lambda x: "low" if x <= edges[0] else ("mid" if x <= edges[1] else "high"))
    raw = raw.assign(regime=raw["date"].map(reg))
    sl = raw[raw["row"] == "slice"]
    dt = raw[raw["row"] == "date"]
    rng = make_rng(seed, "noise_boot_regime")
    out = []
    for (var_, name, r), grp in sl.groupby(["variant", "spec", "regime"], sort=False):
        d = dt[(dt["variant"] == var_) & (dt["spec"] == name) & (dt["regime"] == r)]
        out.append(
            {
                "variant": var_,
                "spec": name,
                "regime": r,
                "regime_edges": f"{edges[0]:.5f}|{edges[1]:.5f}",
                **_cell(grp, d, d["date"].nunique(), n_mat, rng),
            }
        )
    return pd.DataFrame(out)


def run(
    cfg: Config,
    out: Path,
    specs: tuple[NoiseSpec, ...] = SPECS,
    tag: str = "v2",
    n_dates: int | None = None,
) -> pd.DataFrame:
    """Run the study on the grading date set (or a smaller seeded subset) and write tables."""
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    avail = pd.bdate_range(cfg.data.start, cfg.data.end)
    st = ad.simulate_state(avail)
    rows_of = {
        d: (float(sp), float(v), float(a))
        for d, sp, v, a in zip(st["date"], st["spot"], st["var"], st["alpha_t"], strict=True)
    }
    dates = [d for d in sample_dates(avail, cfg) if d - pd.tseries.offsets.BDay(1) in rows_of]
    if n_dates is not None:
        dates = dates[:n_dates]
    jobs = [(d, rows_of[d], rows_of[d - pd.tseries.offsets.BDay(1)], cfg, specs) for d in dates]
    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as ex:
        raw = pd.DataFrame([r for rows in ex.map(_work, jobs) for r in rows])
    n_mat = len(cfg.synthetic.maturities_days)
    summ = summarize(raw, n_mat, cfg.run.seed)
    regime = by_regime(raw, {d: rows_of[d][1] for d in dates}, n_mat, cfg.run.seed)
    meta = {
        "source": TAG,
        "config_hash": rnd_hash(cfg),
        "n_dates": len(dates),
        "first_date": dates[0].date().isoformat(),
        "last_date": dates[-1].date().isoformat(),
        "specs": json.dumps([s.name for s in specs]),
        "variants": json.dumps(VARIANTS),
        "note": "synthetic panel, method validation only; baseline defects present at all levels",
    }
    sfx = f"_{tag}" if tag else ""
    write_stats(summ.round(6), out / f"noise_study{sfx}__{TAG}.csv", meta)
    write_stats(regime.round(6), out / f"noise_study{sfx}_by_regime__{TAG}.csv", meta)
    raw_out = raw.drop(columns=["row"]).assign(date=lambda d: d["date"].dt.date.astype(str))
    write_stats(raw_out.round(8), out / f"noise_study{sfx}_raw__{TAG}.csv", meta)
    return summ


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--specs", default=",".join(RERUN_SPECS), help="comma separated, or all")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--n-dates", type=int, default=None, help="use only the first n grading dates")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = [s.name for s in SPECS] if args.specs == "all" else args.specs.split(",")
    specs = tuple(s for s in SPECS if s.name in names)
    pd.set_option("display.width", 250)
    print(run(cfg, cfg.tables_dir, specs, args.tag, args.n_dates).to_string(index=False))


if __name__ == "__main__":
    main()
