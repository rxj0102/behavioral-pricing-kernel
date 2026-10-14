"""Synthetic Heston SPX-like option panel. For method validation only.

Nothing produced here may be reported as an empirical result. The panel is
generated daily on business days. The latent state (S_t, v_t) follows the
physical dynamics described in SyntheticConfig; quotes are Heston-Q prices
with a bid-ask spread, a 0.05 tick, and injected defects (crossed quotes,
missing asks) at known rates so the cleaning statistics can be checked.
The truth table (spot, variance) is kept apart and never given to the cleaner.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from bpk.config import SyntheticConfig
from bpk.data.base import DataAdapter, SeriesSpec
from bpk.data.heston import heston_call
from bpk.utils.seed import make_rng

_TICK = 0.05
_DT = 1.0 / 252.0


class SyntheticHestonAdapter(DataAdapter):
    """Deterministic synthetic option panel generator."""

    source_name = "synthetic_heston"

    def __init__(self, cfg: SyntheticConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed

    def is_available(self) -> bool:
        """Always available: no network or credentials needed."""
        return True

    def simulate_state(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Simulate (spot, variance) on the physical measure with full-truncation Euler.

        dlnS = (r - q + eta v - v/2) dt + sqrt(v) dW1,
        dv = (kappa theta - (kappa + lam) v) dt + sigma sqrt(v) dW2, corr(dW1, dW2) = rho.
        """
        c = self.cfg
        rng = make_rng(self.seed, "synthetic_state")
        n = len(dates)
        z1 = rng.standard_normal(n)
        z2 = c.rho * z1 + np.sqrt(1 - c.rho**2) * rng.standard_normal(n)
        spot = np.empty(n)
        var = np.empty(n)
        s, v = c.spot0, c.v0
        for i in range(n):
            spot[i], var[i] = s, v
            vp = max(v, 0.0)
            s *= np.exp(
                (c.rate - c.div_yield + c.eta * vp - 0.5 * vp) * _DT + np.sqrt(vp * _DT) * z1[i]
            )
            v = (
                v
                + (c.kappa * c.theta - (c.kappa + c.lam) * vp) * _DT
                + c.sigma * np.sqrt(vp * _DT) * z2[i]
            )
        return pd.DataFrame({"date": dates, "spot": spot, "var": np.maximum(var, 1e-6)})

    def generate(self, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (panel, truth) for business days in [start, end]."""
        c = self.cfg
        dates = pd.bdate_range(start, end)
        truth = self.simulate_state(dates)
        rng = make_rng(self.seed, "synthetic_quotes")
        frames: list[pd.DataFrame] = []
        for dt_, s, v in zip(truth["date"], truth["spot"], truth["var"], strict=True):
            for dte in c.maturities_days:
                frames.append(self._chain(rng, dt_, int(dte), float(s), float(v)))
        panel = pd.concat(frames, ignore_index=True)
        return panel, truth

    def _chain(
        self, rng: np.random.Generator, dt_: pd.Timestamp, dte: int, s: float, v: float
    ) -> pd.DataFrame:
        c = self.cfg
        t = dte / 365.0
        fwd = s * np.exp((c.rate - c.div_yield) * t)
        near = np.arange(np.round(0.90 * fwd / 5) * 5, 1.10 * fwd, 5.0)
        wing = np.arange(np.round(0.55 * fwd / 25) * 25, 1.45 * fwd, 25.0)
        strikes = np.unique(np.concatenate([near, wing[(wing < near[0]) | (wing > near[-1])]]))
        call = heston_call(fwd, strikes, t, c.rate, v, c.kappa, c.theta, c.sigma, c.rho)
        disc = np.exp(-c.rate * t)
        put = call - disc * (fwd - strikes)  # parity
        out = []
        for cp, mid in (("C", call), ("P", put)):
            mid = np.maximum(mid, 0.0)
            half = np.maximum(0.025, 0.004 * mid) * (1 + 0.3 * rng.random(len(mid)))
            bid = np.floor(np.maximum(mid - half, 0.0) / _TICK) * _TICK
            ask = np.ceil((mid + half) / _TICK) * _TICK
            m = np.log(strikes / fwd)
            oi = np.floor(np.exp(rng.normal(6.0 - 25.0 * m * m, 1.0))).astype(int)
            n = len(strikes)
            crossed = rng.random(n) < c.crossed_rate
            bid, ask = np.where(crossed, ask + _TICK, bid), np.where(crossed, bid, ask)
            ask = np.where(rng.random(n) < c.missing_rate, np.nan, ask)
            out.append(
                pd.DataFrame(
                    {
                        "date": dt_,
                        "dte": dte,
                        "strike": strikes,
                        "cp": cp,
                        "bid": bid,
                        "ask": ask,
                        "open_interest": oi,
                        "spot": s,
                        "rf": c.rate,
                    }
                )
            )
        return pd.concat(out, ignore_index=True)

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Return the option panel (spec.name must be spx_options)."""
        if spec.name != "spx_options":
            raise ValueError(f"synthetic adapter only provides spx_options, got {spec.name}")
        return self.generate(start, end)[0]
