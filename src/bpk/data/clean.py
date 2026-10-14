"""Quote cleaning with a full audit trail.

Steps (each counted in the statistics table):
 1. missing_quote: bid or ask is NaN.
 2. crossed: bid > ask.
 3. zero_bid: bid <= 0 (no resting bid, price is not informative).
 4. low_open_interest: open interest below the minimum.
 5. low_mid: mid below the minimum price.
 6. wide_spread: (ask - bid) / mid above the cap.
 7. forward inference per (date, dte) by put-call parity. For a strike with both
    quotes, C - P = exp(-rT) (F - K), so F_K = K + (C - P) exp(rT). With pairs in a
    moneyness window around the median F_K, F is the median F_K. Pairs whose F_K
    deviates from F by more than parity_tol (relative) are dropped. A maturity with
    fewer than min_pairs usable pairs is dropped.
 8. itm_duplicate: keep only out-of-the-money quotes (puts for K < F, calls for K >= F).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bpk.config import CleaningConfig
from bpk.data.schema import validate_panel

_KEYS = ["date", "dte"]


def infer_forward(
    strikes: pd.Series,
    call_mid: pd.Series,
    put_mid: pd.Series,
    disc_inv: float,
    cfg: CleaningConfig,
) -> tuple[float, pd.Series]:
    """Forward by put-call parity from aligned strike-wise call and put mids.

    Returns (forward, boolean mask over the input of pairs that agree with it).
    Forward is NaN if fewer than cfg.min_pairs pairs remain.
    """
    f_k = strikes + (call_mid - put_mid) * disc_inv
    f0 = float(np.median(f_k))
    near = (strikes / f0 - 1.0).abs() <= cfg.parity_window
    if near.sum() < cfg.min_pairs:
        return float("nan"), pd.Series(False, index=strikes.index)
    f1 = float(np.median(f_k[near]))
    ok = near & ((f_k / f1 - 1.0).abs() <= cfg.parity_tol)
    if ok.sum() < cfg.min_pairs:
        return float("nan"), pd.Series(False, index=strikes.index)
    return float(np.median(f_k[ok])), ok


@dataclass(frozen=True)
class CleanResult:
    """Cleaned quotes and the per-step audit table."""

    quotes: pd.DataFrame
    stats: pd.DataFrame


def clean_panel(panel: pd.DataFrame, cfg: CleaningConfig) -> CleanResult:
    """Apply the filters in order and return cleaned OTM quotes plus statistics."""
    validate_panel(panel)
    df = panel.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    rows: list[tuple[str, int, int]] = []

    def step(name: str, keep: pd.Series) -> None:
        nonlocal df
        before = len(df)
        df = df[keep.reindex(df.index, fill_value=False)]
        rows.append((name, before, len(df)))

    n0 = len(df)
    step("missing_quote", df["bid"].notna() & df["ask"].notna())
    step("crossed", df["bid"] <= df["ask"])
    step("zero_bid", df["bid"] > 0)
    step("low_open_interest", df["open_interest"] >= cfg.min_open_interest)
    step("low_mid", df["mid"] >= cfg.min_mid)
    step("wide_spread", (df["ask"] - df["bid"]) / df["mid"] <= cfg.max_rel_spread)

    # Forward inference by parity, per (date, dte).
    fwd_parts: list[pd.DataFrame] = []
    before = len(df)
    for (_d, dte_key), g in df.groupby(_KEYS, sort=True):
        calls = g[g["cp"] == "C"].set_index("strike")["mid"]
        puts = g[g["cp"] == "P"].set_index("strike")["mid"]
        both = calls.index.intersection(puts.index)
        if len(both) < cfg.min_pairs:
            continue
        rf = float(g["rf"].iloc[0])
        disc_inv = float(np.exp(rf * int(str(dte_key)) / 365.0))
        strikes = pd.Series(both.to_numpy(dtype=float), index=both)
        f, ok = infer_forward(strikes, calls.loc[both], puts.loc[both], disc_inv, cfg)
        if np.isnan(f):
            continue
        keep_strikes = both[ok.to_numpy()]
        sel = g[g["strike"].isin(keep_strikes)].copy()
        sel["forward"] = f
        fwd_parts.append(sel)
    df = pd.concat(fwd_parts) if fwd_parts else df.iloc[0:0].assign(forward=np.nan)
    rows.append(("parity_forward", before, len(df)))

    before = len(df)
    otm = ((df["cp"] == "P") & (df["strike"] < df["forward"])) | (
        (df["cp"] == "C") & (df["strike"] >= df["forward"])
    )
    df = df[otm]
    rows.append(("itm_duplicate", before, len(df)))

    stats = pd.DataFrame(rows, columns=["step", "rows_before", "rows_after"])
    stats["dropped"] = stats["rows_before"] - stats["rows_after"]
    stats["share_of_raw"] = stats["dropped"] / max(n0, 1)
    out = df.sort_values(["date", "dte", "strike"]).reset_index(drop=True)
    return CleanResult(quotes=out, stats=stats)
