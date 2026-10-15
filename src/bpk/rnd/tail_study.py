"""Tail-rule specification study, logged to the results log (rule: record every spec tried).

Compares tail rules on a seeded sample of dates of the cleaned synthetic panel against the
exact Q density and appends one results-log row per rule. The sample is part of the same
synthetic panel later used for grading, so this is an in-sample comparison on synthetic data;
it says nothing about real option tails.
GEV variants were added after the first five were logged; the same seeded dates are used, so
rows are comparable.
Usage: python -m bpk.rnd.tail_study --config configs/fast.yaml --log results/results_log.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bpk.config import RndConfig, load_config
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.rnd.grade import exact_q_density, grade_estimate
from bpk.rnd.pipeline import rnd_for_date
from bpk.utils.results_log import log_spec
from bpk.utils.seed import make_rng

VARIANTS: dict[str, dict[str, object]] = {
    "tail_slope_w0.02": {"tail_rule": "slope", "slope_window": 0.02},
    "tail_slope_w0.05": {"tail_rule": "slope", "slope_window": 0.05},
    "tail_slope_w0.10": {"tail_rule": "slope", "slope_window": 0.10},
    "tail_fixed_xi0": {"tail_rule": "fixed", "xi_fixed": 0.0},
    "tail_fixed_xi-0.1": {"tail_rule": "fixed", "xi_fixed": -0.1},
    "tail_gev_xi0": {"tail_rule": "gev", "xi_fixed": 0.0},
    "tail_gev_xi-0.1": {"tail_rule": "gev", "xi_fixed": -0.1},
}
CHOSEN = "tail_fixed_xi0"  # fixed a priori (neutral exponential tail), before seeing scores


def main() -> None:
    """Run the study and append to the results log."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fast.yaml")
    ap.add_argument("--log", default="results/results_log.csv")
    ap.add_argument("--n-dates", type=int, default=8)
    ap.add_argument("--only", default="", help="comma separated variant names (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    avail = pd.bdate_range(cfg.data.start, cfg.data.end)
    state = ad.simulate_state(avail).set_index("date")
    rng = make_rng(cfg.run.seed, "tail_study")
    dates = sorted(avail[rng.choice(len(avail), size=args.n_dates, replace=False)])
    q = pd.read_parquet(
        cfg.data.cache_dir / "clean_quotes__synthetic_heston.parquet",
        filters=[("date", "in", dates)],
    )
    todo = [v for v in args.only.split(",") if v] or list(VARIANTS)
    for name in todo:
        kw = VARIANTS[name]
        rc: RndConfig = cfg.rnd.model_copy(update=kw)
        rows = []
        for d in dates:
            r = rnd_for_date(q[q["date"] == d], cfg.data.horizon_days, rc)
            ests = list(r.native.items())
            if r.horizon is not None:
                ests.append((cfg.data.horizon_days, r.horizon))
            for dte, e in ests:
                t = exact_q_density(
                    ad, float(state.loc[d, "var"]), dte, float(state.loc[d, "alpha_t"]), e.k
                )
                g = {x["region"]: float(x["l1"]) for x in grade_estimate(e, t)}  # type: ignore[arg-type]
                rows.append(g)
        df = pd.DataFrame(rows)
        log_spec(
            Path(args.log),
            phase="phase2",
            spec_id=name,
            status="ok" if name == CHOSEN else "rejected",
            params={**kw, "n_dates": len(dates), "config": args.config},
            metrics={
                "l1_total_mean": float(np.mean(df["total"])),
                "l1_tail_left_mean": float(np.mean(df["tail_ext_left"])),
                "l1_tail_right_mean": float(np.mean(df["tail_ext_right"])),
            },
            note="synthetic in-sample comparison; chosen rule fixed a priori"
            if name == CHOSEN
            else "scored worse than the chosen rule on synthetic truth (not evidence for real data)"
            if name.startswith("tail_slope")
            else "GEV tail alternative, not chosen (default stays GPD xi=0)"
            if name.startswith("tail_gev")
            else "alternative fixed shape, not chosen",
        )
        print(name, round(float(df["total"].mean()), 5))


if __name__ == "__main__":
    main()
