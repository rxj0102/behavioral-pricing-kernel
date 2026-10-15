"""Phase 2 diagnostic figures (synthetic panel). Written to paper/generated/figures.

Figure 1: implied-volatility quotes and fitted smiles for four dates.
Figure 2: estimated 30-day risk-neutral density versus the exact Q density (log scale).
Figure 3: median L1 error by maturity and region of k, from the grading table.
Dates carry real-looking labels (2008-10-10, 2020-03-16) only because the spec asks for
those periods; the state on those dates is synthetic and unrelated to the real market.
Usage: python -m bpk.viz.rnd_figures [--config configs/fast.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bpk.config import Config, load_config
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.rnd.bs import implied_total_variance
from bpk.rnd.grade import exact_q_density
from bpk.rnd.pipeline import rnd_for_date
from bpk.rnd.run import TAG, sample_dates
from bpk.utils.stats_io import read_stats

NOTE = "synthetic panel: method validation only"


def pick_dates(cfg: Config, state: pd.DataFrame, dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Named dates (if present) plus the lowest and highest variance sampled dates."""
    named = [d for d in dates if d.date() in set(cfg.rnd.named_dates)]
    var_of = dict(zip(state["date"], state["var"].tolist(), strict=True))
    by_var = sorted(dates, key=lambda d: var_of[d])
    extra = [by_var[0], by_var[-1]]
    out: list[pd.Timestamp] = []
    for d in [*named, *extra, *dates]:
        if d not in out:
            out.append(d)
    return out[:4]


def make_figures(cfg: Config, out_dir: Path) -> list[Path]:
    """Create the three figures and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ad = SyntheticHestonAdapter(cfg.synthetic, cfg.run.seed)
    avail = pd.bdate_range(cfg.data.start, cfg.data.end)
    state = ad.simulate_state(avail)
    dates = sample_dates(avail, cfg)
    chosen = pick_dates(cfg, state, dates)
    quotes = pd.read_parquet(
        cfg.data.cache_dir / f"clean_quotes__{TAG}.parquet", filters=[("date", "in", chosen)]
    )
    var_of = dict(zip(state["date"], state["var"].tolist(), strict=True))
    alpha_of = dict(zip(state["date"], state["alpha_t"].tolist(), strict=True))
    results = {
        d: rnd_for_date(quotes[quotes["date"] == d], cfg.data.horizon_days, cfg.rnd) for d in chosen
    }

    fig1, axs1 = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for ax1, ax2, d in zip(axs1.ravel(), axs2.ravel(), chosen, strict=False):
        r = results[d]
        qd = quotes[quotes["date"] == d]
        for c, (dte, s) in zip(colors, r.slices.items(), strict=False):
            g = qd[qd["dte"] == dte]
            k = np.log(g["strike"].to_numpy() / s.forward)
            call = (g["cp"] == "C").to_numpy()
            pn = g["mid"].to_numpy() * np.exp(s.rate * s.t) / s.forward
            iv = np.sqrt(implied_total_variance(pn, k, call) / s.t)
            kk = np.linspace(s.k_lo, s.k_hi, 200)
            ax1.plot(k, iv, ".", color=c, ms=3)
            ax1.plot(
                kk, np.sqrt(s.w_func(kk) / s.t), "-", color=c, lw=1, label=f"{dte}d {s.method}"
            )
        ax1.set(title=str(d.date()), xlabel="ln(K/F)", ylabel="implied vol")
        ax1.legend(fontsize=7)
        h = r.horizon
        if h is not None:
            truth = exact_q_density(ad, var_of[d], cfg.data.horizon_days, alpha_of[d], h.k)
            ax2.semilogy(h.k, np.maximum(truth, 1e-8), "k-", lw=1.5, label="exact Q")
            ax2.semilogy(h.k, np.maximum(h.density, 1e-8), "C3--", lw=1.2, label="estimated")
            ax2.axvline(h.k_left, color="grey", ls=":")
            ax2.axvline(h.k_right, color="grey", ls=":")
            ax2.set(
                title=f"{d.date()} 30d", xlabel="k = ln(S_T/F)", ylabel="density", ylim=(1e-6, 30)
            )
            ax2.legend(fontsize=7)
    fig1.suptitle(f"Smile fits ({NOTE})")
    fig2.suptitle(f"30-day RND vs exact Q, dotted: tail splice points ({NOTE})")
    for f in (fig1, fig2):
        f.tight_layout()
    p1, p2 = out_dir / "rnd_smiles.png", out_dir / "rnd_density_vs_exact.png"
    fig1.savefig(p1, dpi=110)
    fig2.savefig(p2, dpi=110)

    summ, _ = read_stats(cfg.tables_dir / f"rnd_grading_summary__{TAG}.csv")
    regs = ["k<-0.15", "-0.15..-0.05", "-0.05..0.05", "0.05..0.15", "k>0.15", "total"]
    piv = summ[summ["region"].isin(regs)].pivot_table(
        index="dte", columns="region", values="l1_median"
    )
    piv = piv[regs]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    im = ax3.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax3.set_xticks(range(len(regs)), regs, rotation=30)
    ax3.set_yticks(range(len(piv)), [f"{i}d" for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax3.text(j, i, f"{piv.iat[i, j]:.4f}", ha="center", va="center", color="w", fontsize=7)
    fig3.colorbar(im, ax=ax3, label="median L1 over dates")
    ax3.set_title(f"L1 error vs exact Q by maturity and region ({NOTE})")
    fig3.tight_layout()
    p3 = out_dir / "rnd_l1_by_region.png"
    fig3.savefig(p3, dpi=110)
    plt.close("all")
    return [p1, p2, p3]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="paper/generated/figures")
    args = ap.parse_args()
    for p in make_figures(load_config(args.config), Path(args.out)):
        print(p)


if __name__ == "__main__":
    main()
