"""Quote cleaning with a full audit trail, including a moneyness breakdown.

Steps, each counted in aggregate and by moneyness bucket (ln(K/spot)):
 1. missing_quote: bid or ask is NaN.
 2. crossed: bid > ask.
 3. zero_bid: bid <= 0 (no resting bid, price is not informative).
 4. low_open_interest: open interest below the minimum.
 5. low_mid: mid below the minimum price.
 6. wide_spread: (ask - bid) / mid above the cap.
 7. no_forward: maturities (date, dte) where near-ATM parity gives no usable forward.
 8. itm_duplicate: keep only out-of-the-money quotes (puts for K < F, calls for K >= F).

Forward inference uses only near-ATM call-put pairs. For a strike with both quotes,
C - P = exp(-rT) (F - K), so F_K = K + (C - P) exp(rT). The reference F0 is the median F_K
over the (up to) five pairs with strikes closest to spot. Pairs within parity_window of
F0 vote: F is their median, and pairs whose F_K deviates from F by more than parity_tol
are rejected from the vote (counted, not dropped as quotes). Quotes without a counterpart
are never removed for that reason: wings are kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bpk.config import CleaningConfig
from bpk.data.schema import validate_panel

_KEYS = ["date", "dte"]
BUCKET_EDGES = (-np.inf, -0.30, -0.15, -0.05, 0.05, 0.15, 0.30, np.inf)
BUCKET_LABELS = (
    "lt_-0.30",
    "-0.30_-0.15",
    "-0.15_-0.05",
    "-0.05_0.05",
    "0.05_0.15",
    "0.15_0.30",
    "gt_0.30",
)


def moneyness_bucket(strike: pd.Series, spot: pd.Series) -> pd.Series:
    """Bucket label of ln(K/spot)."""
    cut = pd.cut(
        np.log(strike / spot), bins=list(BUCKET_EDGES), labels=list(BUCKET_LABELS), right=True
    )
    return pd.Series(cut.astype(str), index=strike.index)


def infer_forward(
    strikes: pd.Series,
    call_mid: pd.Series,
    put_mid: pd.Series,
    disc_inv: float,
    cfg: CleaningConfig,
    spot: float,
) -> tuple[float, pd.Series]:
    """Forward by put-call parity from aligned strike-wise call and put mids, near ATM only.

    Returns (forward, boolean mask of pairs that voted and agree with it). Forward is NaN
    if fewer than cfg.min_pairs pairs remain.
    """
    none = pd.Series(False, index=strikes.index)
    f_k = strikes + (call_mid - put_mid) * disc_inv
    nearest = (strikes - spot).abs().nsmallest(min(5, len(strikes))).index
    f0 = float(np.median(f_k.loc[nearest]))
    near = (strikes / f0 - 1.0).abs() <= cfg.parity_window
    if near.sum() < cfg.min_pairs:
        return float("nan"), none
    f1 = float(np.median(f_k[near]))
    ok = near & ((f_k / f1 - 1.0).abs() <= cfg.parity_tol)
    if ok.sum() < cfg.min_pairs:
        return float("nan"), none
    return float(np.median(f_k[ok])), ok


@dataclass(frozen=True)
class CleanResult:
    """Cleaned quotes and the audit tables."""

    quotes: pd.DataFrame
    stats: pd.DataFrame  # one row per step
    stats_by_moneyness: pd.DataFrame  # step x bucket
    parity: pd.DataFrame  # one row per (date, dte) with a forward


def clean_panel(panel: pd.DataFrame, cfg: CleaningConfig) -> CleanResult:
    """Apply the filters in order and return cleaned OTM quotes plus statistics."""
    validate_panel(panel)
    df = panel.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["bucket"] = moneyness_bucket(df["strike"], df["spot"])
    rows: list[tuple[str, int, int]] = []
    brows: list[pd.DataFrame] = []

    def record(name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        rows.append((name, len(before), len(after)))
        b = pd.DataFrame(
            {
                "rows_before": before["bucket"].value_counts(),
                "rows_after": after["bucket"].value_counts(),
            }
        ).reindex(list(BUCKET_LABELS), fill_value=0)
        b = b.fillna(0).astype(int)
        b.insert(0, "bucket", b.index)
        b.insert(0, "step", name)
        brows.append(b.reset_index(drop=True))

    def step(name: str, keep: pd.Series) -> None:
        nonlocal df
        before = df
        df = df[keep.reindex(df.index, fill_value=False)]
        record(name, before, df)

    n0 = len(df)
    step("missing_quote", df["bid"].notna() & df["ask"].notna())
    step("crossed", df["bid"] <= df["ask"])
    step("zero_bid", df["bid"] > 0)
    step("low_open_interest", df["open_interest"] >= cfg.min_open_interest)
    step("low_mid", df["mid"] >= cfg.min_mid)
    step("wide_spread", (df["ask"] - df["bid"]) / df["mid"] <= cfg.max_rel_spread)

    # Forward inference per (date, dte) from near-ATM parity pairs.
    fwd: dict[tuple[object, object], float] = {}
    diag: list[dict[str, object]] = []
    for (d, dte_key), g in df.groupby(_KEYS, sort=True):
        calls = g[g["cp"] == "C"].set_index("strike")["mid"]
        puts = g[g["cp"] == "P"].set_index("strike")["mid"]
        both = calls.index.intersection(puts.index)
        if len(both) < cfg.min_pairs:
            continue
        rf = float(g["rf"].iloc[0])
        disc_inv = float(np.exp(rf * int(str(dte_key)) / 365.0))
        strikes = pd.Series(both.to_numpy(dtype=float), index=both)
        f, ok = infer_forward(
            strikes, calls.loc[both], puts.loc[both], disc_inv, cfg, float(g["spot"].iloc[0])
        )
        if np.isnan(f):
            continue
        fwd[(d, dte_key)] = f
        diag.append(
            {"date": d, "dte": dte_key, "pairs": len(both), "voting": int(ok.sum()), "forward": f}
        )
    before = df
    key = pd.MultiIndex.from_frame(df[_KEYS])
    df = df.assign(forward=pd.Series(list(key), index=df.index).map(fwd))
    df = df[df["forward"].notna()]
    record("no_forward", before, df)

    before = df
    otm = ((df["cp"] == "P") & (df["strike"] < df["forward"])) | (
        (df["cp"] == "C") & (df["strike"] >= df["forward"])
    )
    df = df[otm]
    record("itm_duplicate", before, df)

    stats = pd.DataFrame(rows, columns=["step", "rows_before", "rows_after"])
    stats["dropped"] = stats["rows_before"] - stats["rows_after"]
    stats["share_of_raw"] = (stats["dropped"] / max(n0, 1)).round(6)
    by = pd.concat(brows, ignore_index=True)
    by["dropped"] = by["rows_before"] - by["rows_after"]
    out = df.drop(columns=["bucket"]).sort_values(["date", "dte", "strike"]).reset_index(drop=True)
    return CleanResult(out, stats, by, pd.DataFrame(diag))
