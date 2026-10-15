"""Apply the pre-registered decision rule for the default SVI fit variant.

The rule is written in docs/fit_variant_decision.md and fixed before results were read.
Input: raw tables written by bpk.rnd.noise_study (one row per date, spec, variant for the
date-level rows and per slice for the density rows).
Usage: python -m bpk.rnd.decision --primary RAW.csv [--confirm RAW2.csv] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bpk.utils.results_log import log_spec
from bpk.utils.seed import make_rng
from bpk.utils.stats_io import read_stats, write_stats

SPECS = ("stale_0.05", "combined_2")
ORDER = ("legacy", "robust", "robust_monotone")  # simplest first, for exact ties
WORST_L1 = 2.0
BOOT = 2000
COVERAGE_TOL = 0.02
LEGACY = "legacy"


@dataclass(frozen=True)
class Decision:
    """Outcome of the rule on one set of dates."""

    table: pd.DataFrame
    winner: str


def _per_date_l1(raw: pd.DataFrame, spec: str, variant: str) -> tuple[np.ndarray, float, float]:
    """30 day L1 per date (worst value where missing), horizon coverage, slice coverage."""
    sub = raw[(raw["spec"] == spec) & (raw["variant"] == variant)]
    date_rows = sub[sub["n_slices"].notna()]
    dates = date_rows["date"].unique()
    h = sub[(sub["interpolated"] == True) & sub["l1_total"].notna()]  # noqa: E712
    l1 = h.groupby("date")["l1_total"].mean().reindex(dates)
    horizon_cov = float(l1.notna().mean())
    n_mat = int(sub.loc[sub["interpolated"] == False, "dte"].nunique())  # noqa: E712
    slice_cov = float(date_rows["n_slices"].sum() / (len(dates) * max(n_mat, 1)))
    return l1.fillna(WORST_L1).to_numpy(dtype=float), horizon_cov, slice_cov


def boot_mean_p90(values: np.ndarray, rng: np.random.Generator, b: int = BOOT) -> float:
    """Mean over bootstrap resamples (of dates) of the 90th percentile."""
    n = len(values)
    idx = rng.integers(0, n, size=(b, n))
    return float(np.mean(np.quantile(values[idx], 0.9, axis=1)))


def decide(raw: pd.DataFrame, seed: int) -> Decision:
    """Compute objectives, eligibility and the winner on one raw table."""
    stats: dict[str, dict[str, float]] = {}
    for v in ORDER:
        row: dict[str, float] = {}
        objs = []
        for spec in SPECS:
            vals, h, s = _per_date_l1(raw, spec, v)
            o = boot_mean_p90(vals, make_rng(seed, f"decision_boot_{spec}_{v}"))
            objs.append(o)
            row[f"boot_mean_p90_{spec}"] = o
            row[f"horizon_cov_{spec}"] = h
            row[f"slice_cov_{spec}"] = s
        row["objective"] = float(np.mean(objs))
        stats[v] = row
    cov_cols = [f"{k}_{sp}" for sp in SPECS for k in ("horizon_cov", "slice_cov")]
    eligible = {
        v: all(stats[v][c] >= stats[LEGACY][c] - COVERAGE_TOL - 1e-12 for c in cov_cols)
        for v in ORDER
    }
    cand = [v for v in ORDER if eligible[v]]
    best = min(stats[v]["objective"] for v in cand)
    winner = next(v for v in cand if abs(stats[v]["objective"] - best) <= 1e-12)
    table = pd.DataFrame([{"variant": v, **stats[v], "eligible": eligible[v]} for v in ORDER])
    return Decision(table, winner)


def final_default(primary: Decision, confirm: Decision | None) -> tuple[str, str]:
    """Apply the confirmation rule. Returns (default variant, status)."""
    if confirm is None:
        return primary.winner, "primary only (not yet confirmed)"
    if confirm.winner == primary.winner:
        return primary.winner, "confirmed"
    return LEGACY, f"not confirmed (confirmation winner: {confirm.winner}); reverted to legacy"


def fallback_diagnostic(raw: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """L1 of native slices by fit method, and of 30 day densities by whether a bracket slice
    used the convex fallback. Reproducible companion of the decision (not part of the rule)."""
    nat = raw[(raw["interpolated"] == False) & raw["l1_total"].notna()]  # noqa: E712
    out: list[dict[str, object]] = []

    def add(block: str, variant: str, group: str, vals: pd.Series) -> None:
        if len(vals):
            out.append(
                {
                    "block": block,
                    "variant": variant,
                    "group": group,
                    "n": len(vals),
                    "l1_median": float(vals.median()),
                    "l1_p90": float(np.quantile(vals, 0.9)),
                }
            )

    for v in ORDER:
        for m in ("svi", "convex"):
            add(
                "native_by_method",
                v,
                m,
                nat.loc[(nat["variant"] == v) & (nat["method"] == m), "l1_total"],
            )
    dtes = sorted(int(d) for d in nat["dte"].unique())
    below = [d for d in dtes if d <= horizon_days]
    above = [d for d in dtes if d >= horizon_days]
    if below and above:
        bracket = {below[-1], above[0]}
        conv = nat[(nat["method"] == "convex") & nat["dte"].isin(bracket)]
        keys = set(zip(conv["variant"], conv["spec"], conv["date"], strict=True))
        h = raw[(raw["interpolated"] == True) & raw["l1_total"].notna()]  # noqa: E712
        flag = pd.Series(
            [(v, s, d) in keys for v, s, d in zip(h["variant"], h["spec"], h["date"], strict=True)],
            index=h.index,
        )
        for v in ORDER:
            for b in (False, True):
                add(
                    "30d_by_bracket_convex",
                    v,
                    str(b),
                    h.loc[(h["variant"] == v) & (flag == b), "l1_total"],
                )
    return pd.DataFrame(out)


def _span(meta: dict[str, str]) -> str:
    return f"{meta.get('first_date')}..{meta.get('last_date')} n={meta.get('n_dates')}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True)
    ap.add_argument("--confirm", default=None)
    ap.add_argument("--out", default="results/tables")
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--log", default="results/results_log.csv")
    args = ap.parse_args()
    out = Path(args.out)
    praw, pmeta = read_stats(Path(args.primary))
    prim = decide(praw, args.seed)
    conf = None
    cmeta: dict[str, str] = {}
    craw: pd.DataFrame | None = None
    if args.confirm:
        craw, cmeta = read_stats(Path(args.confirm))
        conf = decide(craw, args.seed)
    default, status = final_default(prim, conf)
    meta = {
        "rule": "docs/fit_variant_decision.md",
        "primary_config_hash": pmeta.get("config_hash", ""),
        "primary_dates": _span(pmeta),
        "confirm_dates": _span(cmeta) if conf is not None else "none",
        "primary_winner": prim.winner,
        "confirm_winner": conf.winner if conf is not None else "none",
        "final_default": default,
        "status": status,
    }
    raws: dict[str, pd.DataFrame | None] = {"primary": praw, "confirm": craw}
    for name, d in (("primary", prim), ("confirm", conf)):
        if d is None:
            continue
        rawd = raws[name]
        if rawd is not None:
            write_stats(
                fallback_diagnostic(rawd).round(6),
                out / f"fallback_diagnostic_{name}__synthetic_heston.csv",
                meta,
            )
        write_stats(
            d.table.round(6), out / f"fit_variant_decision_{name}__synthetic_heston.csv", meta
        )
        for _, r in d.table.iterrows():
            log_spec(
                Path(args.log),
                phase="phase2",
                spec_id=f"fit_variant_{r['variant']}_{name}",
                status="ok" if r["variant"] == d.winner else "rejected",
                params={"set": name, "specs": list(SPECS)},
                metrics={
                    k: float(r[k]) for k in d.table.columns if k not in ("variant", "eligible")
                }
                | {"eligible": float(r["eligible"])},
                note="pre-registered rule, docs/fit_variant_decision.md",
            )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
