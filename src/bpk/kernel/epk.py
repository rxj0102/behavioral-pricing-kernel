"""The empirical pricing kernel on one date and a measure of how non-monotone it is."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

from bpk.kernel.mixture import Mixture

FloatArr = NDArray[np.float64]


def empirical_kernel(
    samples: FloatArr, mix: Mixture, shift: float, n_grid: int = 80
) -> tuple[FloatArr, FloatArr]:
    """Risk-neutral over physical density between the 5th and 95th physical percentile.

    samples are physical draws of x (log return in excess of the forward); the mixture is the
    risk-neutral density of the raw log return, so it is evaluated at x + shift.
    """
    lo, hi = np.percentile(samples, [5.0, 95.0])
    grid = np.linspace(lo, hi, n_grid)
    p = gaussian_kde(samples)(grid)
    q = mix.pdf(grid + shift)
    return grid, np.asarray(q / p, dtype=np.float64)


def upward_variation(kernel: FloatArr) -> float:
    """Sum of the upward moves of the log kernel. Zero for a kernel that never rises."""
    steps = np.diff(np.log(kernel))
    return float(steps[steps > 0.0].sum())
