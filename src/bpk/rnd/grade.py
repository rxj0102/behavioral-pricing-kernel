"""Grade estimated RNDs against the exact risk-neutral density of the synthetic generator.

The exact Q density of k = ln(S_T / F) comes from the generator itself: in behavioral_sdf
mode from the physical density times the known SDF, in heston_q mode from Fourier inversion
of the Q characteristic function. Both are evaluated at any maturity, so the interpolated
30-day estimate is graded against the exact 30-day density, which includes the interpolation
error. L1 error is int |f_hat - f| dk, reported in total, by region of k, and for the part of
the density produced by the generalized Pareto tails (outside the splice points).
Regions of k (equivalent to moneyness ln(K/F) of the terminal price): below -0.15, -0.15 to
-0.05, -0.05 to 0.05, 0.05 to 0.15, above 0.15.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bpk.data.heston import DensityInverter
from bpk.data.synthetic import _Y, SyntheticHestonAdapter
from bpk.rnd.density import RndEstimate

FloatArr = NDArray[np.float64]
REGION_EDGES = (-np.inf, -0.15, -0.05, 0.05, 0.15, np.inf)
REGION_LABELS = ("k<-0.15", "-0.15..-0.05", "-0.05..0.05", "0.05..0.15", "k>0.15")


def exact_q_density(
    ad: SyntheticHestonAdapter, v: float, dte: int, alpha_t: float, kgrid: FloatArr
) -> FloatArr:
    """Exact Q density of k = ln(S_T/F) on kgrid (zero outside the generator grid)."""
    c = ad.cfg
    t = dte / 365.0
    if c.pricing_mode == "behavioral_sdf":
        d = ad.densities(v, dte, alpha_t)
        x = d.x - (c.rate - c.div_yield) * t
        dens = d.risk_neutral
    else:
        inv: DensityInverter = ad._inv
        dens = inv.density(t, v, c.kappa, c.kappa * c.theta, c.sigma, c.rho, 0.5)
        x = _Y
    return np.asarray(np.interp(kgrid, x, dens, left=0.0, right=0.0), dtype=np.float64)


def _trapz_mask(y: FloatArr, x: FloatArr, mask: np.ndarray) -> float:
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(y[mask], x[mask]))


def grade_estimate(est: RndEstimate, truth: FloatArr) -> list[dict[str, object]]:
    """Rows of L1 errors: total, each region, and the left/right tail-extension parts."""
    k = est.k
    err = np.abs(est.density - truth)
    rows: list[dict[str, object]] = []

    def add(name: str, mask: np.ndarray) -> None:
        rows.append(
            {
                "region": name,
                "l1": _trapz_mask(err, k, mask),
                "true_mass": _trapz_mask(truth, k, mask),
                "est_mass": _trapz_mask(est.density, k, mask),
            }
        )

    add("total", np.ones_like(k, dtype=bool))
    for i, lab in enumerate(REGION_LABELS):
        add(lab, (k >= REGION_EDGES[i]) & (k <= REGION_EDGES[i + 1]))
    add("tail_ext_left", k <= est.k_left)
    add("tail_ext_right", k >= est.k_right)
    add("interior_between_splices", (k >= est.k_left) & (k <= est.k_right))
    return rows
