"""Synthetic SPX-like option panel. For method validation only.

Nothing produced here may be reported as an empirical result. The panel is generated
daily on business days. The latent state (S_t, v_t, sentiment s_t) is simulated on the
physical measure; quotes carry a spread, a 0.05 tick and injected defects (crossed quotes,
missing asks) at known rates so the cleaning statistics can be checked. Truth tables are
returned separately and are never given to the cleaner.

Pricing modes (SyntheticConfig.pricing_mode):

heston_q: option prices are Heston prices under Q (semi-closed form).

behavioral_sdf: option prices are computed from the physical density times a known SDF.
 1. y = ln(S_T/S_0) - (r-q)T has Heston P-dynamics density f(y) given the current variance:
    dlnS = (r - q + eta v - v/2) dt + sqrt(v) dW1, dv = (kappa theta - (kappa + lam) v) dt + ...
 2. SDF m(y) = exp(-gamma y) w'(S(y)), S(y) = P(Y > y), Prelec w with curvature alpha_t.
 3. Location shift xi so the same SDF prices the bond and the index:
        I0 = int f m dy,  I1 = int f m e^y dy,  xi = (r-q)T - ln(I1/I0).
    The physical density of the log return is x = y + xi, the risk-neutral density is
    q(y) = f(y) m(y) / I0 on the same shifted axis. Then E_Q[S_T] = S_0 e^{(r-q)T} exactly,
    so put-call parity holds with forward F = S_0 e^{(r-q)T}.
 4. Calls: C(K) = e^{-rT} [S_0 E_Q[e^x 1{x>k}] - K Q(x>k)], k = ln(K/S_0), by cumulative Simpson
    and cubic spline on the grid. Puts by parity.
 Known simplification: the daily state path uses the plain Heston P-dynamics, while the
 pricing density is that density shifted by xi (reported in sdf_truth). The path therefore
 matches the pricing density only approximately, and xi measures the gap.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline

from bpk.config import SyntheticConfig
from bpk.data.base import DataAdapter, SeriesSpec
from bpk.data.heston import DensityInverter, heston_call
from bpk.models.weighting import rdu_kernel
from bpk.utils.seed import make_rng

FloatArr = NDArray[np.float64]
_TICK = 0.05
_DT = 1.0 / 252.0
_Y = np.linspace(-1.5, 1.0, 2001)
_DENSITY_FLOOR = 1e-10
_VAR_FLOOR = 0.008  # about 9 percent volatility; avoids a near-singular density


class SyntheticData(NamedTuple):
    """Output of the generator: panel, per-date state truth, per-chain SDF truth."""

    panel: pd.DataFrame
    truth: pd.DataFrame
    sdf_truth: pd.DataFrame


class DensityPair(NamedTuple):
    """Known densities on the return grid x for one date and maturity."""

    x: FloatArr  # log return ln(S_T/S_0)
    physical: FloatArr
    risk_neutral: FloatArr
    kernel: FloatArr  # m on the grid, normalized so that E_P[m] = exp(-rT)
    xi: float


class SyntheticHestonAdapter(DataAdapter):
    """Deterministic synthetic option panel generator."""

    source_name = "synthetic_heston"

    def __init__(self, cfg: SyntheticConfig, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self._inv = DensityInverter(_Y)

    def is_available(self) -> bool:
        """Always available: no network or credentials needed."""
        return True

    def truth_params(self) -> dict[str, object]:
        """Known generating parameters, stored for the recovery test."""
        return {
            "pricing_mode": self.cfg.pricing_mode,
            "seed": self.seed,
            "heston": self.cfg.model_dump(mode="json", exclude={"sdf"}),
            "sdf": self.cfg.sdf.model_dump(mode="json"),
        }

    def simulate_state(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Simulate (spot, variance, sentiment, alpha_t) on the physical measure.

        dlnS = (r - q + eta v - v/2) dt + sqrt(v) dW1,
        dv = (kappa theta - (kappa + lam) v) dt + sigma sqrt(v) dW2 (full truncation Euler),
        corr(dW1, dW2) = rho; s_t = phi s_{t-1} + sqrt(1 - phi^2) e_t;
        alpha_t = clip(alpha0 - delta s_t, alpha_min, 1).
        """
        c, sd = self.cfg, self.cfg.sdf
        rng = make_rng(self.seed, "synthetic_state")
        n = len(dates)
        z1 = rng.standard_normal(n)
        z2 = c.rho * z1 + np.sqrt(1 - c.rho**2) * rng.standard_normal(n)
        eps = make_rng(self.seed, "synthetic_sentiment").standard_normal(n)
        sent = np.empty(n)
        spot = np.empty(n)
        var = np.empty(n)
        s, v, x = c.spot0, c.v0, 0.0
        for i in range(n):
            spot[i], var[i], sent[i] = s, v, x
            vp = max(v, _VAR_FLOOR)
            s *= np.exp(
                (c.rate - c.div_yield + c.eta * vp - 0.5 * vp) * _DT + np.sqrt(vp * _DT) * z1[i]
            )
            v = (
                v
                + (c.kappa * c.theta - (c.kappa + c.lam) * vp) * _DT
                + c.sigma * np.sqrt(vp * _DT) * z2[i]
            )
            x = sd.sentiment_phi * x + np.sqrt(1 - sd.sentiment_phi**2) * eps[i]
        alpha_t = np.clip(sd.alpha0 - sd.delta * sent, sd.alpha_min, 1.0)
        return pd.DataFrame(
            {
                "date": dates,
                "spot": spot,
                "var": np.maximum(var, _VAR_FLOOR),
                "sentiment": sent,
                "alpha_t": alpha_t,
            }
        )

    def densities(self, v: float, dte: int, alpha_t: float) -> DensityPair:
        """Physical and risk-neutral log-return densities for the behavioral SDF mode."""
        c, sd = self.cfg, self.cfg.sdf
        t = dte / 365.0
        f = self._inv.density(t, v, c.kappa + c.lam, c.kappa * c.theta, c.sigma, c.rho, 0.5 - c.eta)
        f = np.where(f < _DENSITY_FLOOR, 0.0, f)
        f = f / np.trapezoid(f, _Y)
        m = rdu_kernel(_Y, f, sd.gamma, alpha_t, sd.beta)
        i0 = float(np.trapezoid(f * m, _Y))
        i1 = float(np.trapezoid(f * m * np.exp(_Y), _Y))
        xi = (c.rate - c.div_yield) * t - np.log(i1 / i0)
        scale = np.exp(-c.rate * t) / i0  # so that E_P[m] = exp(-rT)
        return DensityPair(_Y + xi, f, f * m / i0, m * scale, float(xi))

    def _behavioral_calls(
        self, s0: float, v: float, dte: int, alpha_t: float, strikes: FloatArr
    ) -> tuple[FloatArr, float]:
        """Call prices from the known SDF and the physical density. Returns (calls, xi)."""
        c = self.cfg
        t = dte / 365.0
        d = self.densities(v, dte, alpha_t)
        cum0 = cumulative_simpson(d.risk_neutral, x=_Y, initial=0.0)
        cum1 = cumulative_simpson(d.risk_neutral * np.exp(_Y), x=_Y, initial=0.0)
        k = np.clip(np.log(strikes / s0) - d.xi, _Y[0], _Y[-1])
        a0 = cum0[-1] - CubicSpline(_Y, cum0)(k)  # Q(x > k)
        a1 = np.exp(d.xi) * (cum1[-1] - CubicSpline(_Y, cum1)(k))  # E_Q[e^x 1{x>k}]
        disc = np.exp(-c.rate * t)
        calls = disc * (s0 * a1 - strikes * a0)
        floor = np.maximum(disc * (s0 * np.exp((c.rate - c.div_yield) * t) - strikes), 0.0)
        return np.maximum(calls, floor), d.xi

    def generate(self, start: date, end: date) -> SyntheticData:
        """Return (panel, truth, sdf_truth) for business days in [start, end]."""
        c = self.cfg
        dates = pd.bdate_range(start, end)
        truth = self.simulate_state(dates)
        rng = make_rng(self.seed, "synthetic_quotes")
        frames: list[pd.DataFrame] = []
        sdf_rows: list[dict[str, object]] = []
        for dt_, s, v, a in zip(
            truth["date"], truth["spot"], truth["var"], truth["alpha_t"], strict=True
        ):
            for dte in c.maturities_days:
                chain, xi = self._chain(rng, dt_, int(dte), float(s), float(v), float(a))
                frames.append(chain)
                sdf_rows.append({"date": dt_, "dte": int(dte), "xi": xi, "alpha_t": float(a)})
        sdf_truth = pd.DataFrame(sdf_rows)
        sdf_truth["gamma"] = c.sdf.gamma
        sdf_truth["beta"] = c.sdf.beta
        return SyntheticData(pd.concat(frames, ignore_index=True), truth, sdf_truth)

    def _chain(
        self, rng: np.random.Generator, dt_: pd.Timestamp, dte: int, s: float, v: float, a: float
    ) -> tuple[pd.DataFrame, float]:
        c = self.cfg
        t = dte / 365.0
        fwd = s * np.exp((c.rate - c.div_yield) * t)
        near = np.arange(np.round(0.90 * fwd / 5) * 5, 1.10 * fwd, 5.0)
        wing = np.arange(np.round(0.55 * fwd / 25) * 25, 1.45 * fwd, 25.0)
        strikes = np.unique(np.concatenate([near, wing[(wing < near[0]) | (wing > near[-1])]]))
        if c.pricing_mode == "behavioral_sdf":
            call, xi = self._behavioral_calls(s, v, dte, a, strikes)
        else:
            call = heston_call(fwd, strikes, t, c.rate, v, c.kappa, c.theta, c.sigma, c.rho)
            xi = float("nan")
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
        return pd.concat(out, ignore_index=True), xi

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Return the option panel (spec.name must be spx_options)."""
        if spec.name != "spx_options":
            raise ValueError(f"synthetic adapter only provides spx_options, got {spec.name}")
        return self.generate(start, end).panel
