"""Heston semi-closed-form call prices (method validation only).

Under Q with log-forward x = ln(F/K), the call price is
    C = exp(-rT) (F P1 - K P2),
    P_j = 1/2 + (1/pi) int_0^inf Re[ exp(i u x) psi_j(u) / (i u) ] du,
    psi_2(u) = psi(u),  psi_1(u) = psi(u - i),
    psi(u) = exp(A(u) + B(u) v0),
with the little trap form (Albrecher et al. 2007):
    d = sqrt((kappa - rho sigma i u)^2 + sigma^2 (i u + u^2)),
    g = (kappa - rho sigma i u - d) / (kappa - rho sigma i u + d),
    A = (kappa theta / sigma^2) [(kappa - rho sigma i u - d) T - 2 ln((1 - g e^{-dT}) / (1 - g))],
    B = ((kappa - rho sigma i u - d) / sigma^2) (1 - e^{-dT}) / (1 - g e^{-dT}).
The u-integral uses composite Gauss-Legendre panels on [0, 500].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
_PANELS = ((0.0, 5.0), (5.0, 25.0), (25.0, 100.0), (100.0, 500.0))
_NODES_PER_PANEL = 128


def _quadrature() -> tuple[FloatArr, FloatArr]:
    x, w = np.polynomial.legendre.leggauss(_NODES_PER_PANEL)
    us, ws = [], []
    for a, b in _PANELS:
        us.append(0.5 * (b - a) * x + 0.5 * (b + a))
        ws.append(0.5 * (b - a) * w)
    return np.concatenate(us), np.concatenate(ws)


_U, _W = _quadrature()


def _psi(
    u: NDArray[np.complex128],
    t: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> NDArray[np.complex128]:
    """Normalized Heston characteristic function psi(u) (forward-measure, no drift term)."""
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma**2 * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * t)
    a_term = (kappa * theta / sigma**2) * ((b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    b_term = ((b - d) / sigma**2) * (1.0 - e) / (1.0 - g * e)
    return np.asarray(np.exp(a_term + b_term * v0), dtype=np.complex128)


def heston_call(
    forward: float,
    strikes: FloatArr,
    t: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> FloatArr:
    """Undiscounted-forward Heston call prices for an array of strikes."""
    x = np.log(forward / np.asarray(strikes, dtype=float))[:, None]
    u = _U[None, :].astype(complex)
    psi2 = _psi(u, t, v0, kappa, theta, sigma, rho)
    psi1 = _psi(u - 1j, t, v0, kappa, theta, sigma, rho)
    kern = np.exp(1j * u * x) / (1j * u)
    p1 = 0.5 + (kern * psi1).real @ _W / np.pi
    p2 = 0.5 + (kern * psi2).real @ _W / np.pi
    k = np.asarray(strikes, dtype=float)
    return np.asarray(np.exp(-rate * t) * (forward * p1 - k * p2), dtype=np.float64)
