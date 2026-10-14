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
The u-integral uses composite Gauss-Legendre panels on [0, 2000].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
_PANELS = (
    (0.0, 5.0),
    (5.0, 25.0),
    (25.0, 60.0),
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
)
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
    kappa_theta: float,
    sigma: float,
    rho: float,
    c: float = 0.5,
) -> NDArray[np.complex128]:
    """Heston characteristic function psi(u) of the return without its deterministic drift.

    The log return has drift -c v dt (c = 1/2 is standard Heston under Q) and the
    variance has drift kappa_theta - kappa v. With b = kappa - rho sigma i u and
    d = sqrt(b^2 + sigma^2 (2 c i u + u^2)), the module docstring formulas hold with
    (i u + u^2) replaced by (2 c i u + u^2) and kappa theta by kappa_theta.
    """
    iu = 1j * u
    b = kappa - rho * sigma * iu
    d = np.sqrt(b * b + sigma**2 * (2.0 * c * iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * t)
    a_term = (kappa_theta / sigma**2) * ((b - d) * t - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
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
    psi2 = _psi(u, t, v0, kappa, kappa * theta, sigma, rho)
    psi1 = _psi(u - 1j, t, v0, kappa, kappa * theta, sigma, rho)
    kern = np.exp(1j * u * x) / (1j * u)
    p1 = 0.5 + (kern * psi1).real @ _W / np.pi
    p2 = 0.5 + (kern * psi2).real @ _W / np.pi
    k = np.asarray(strikes, dtype=float)
    return np.asarray(np.exp(-rate * t) * (forward * p1 - k * p2), dtype=np.float64)


class DensityInverter:
    """Fourier inversion of a centered return density on a fixed grid.

    f(y) = (1/pi) int_0^inf Re[exp(-i u y) psi(u)] du
         = (1/pi) sum_k w_k [cos(u_k y) Re psi(u_k) + sin(u_k y) Im psi(u_k)].
    The cos and sin matrices are built once per grid, so each density costs two
    matrix-vector products.
    """

    def __init__(self, y: FloatArr) -> None:
        self.y = np.asarray(y, dtype=float)
        arg = np.outer(self.y, _U)
        self._cos = np.cos(arg) * _W
        self._sin = np.sin(arg) * _W

    def density(
        self,
        t: float,
        v0: float,
        kappa: float,
        kappa_theta: float,
        sigma: float,
        rho: float,
        c: float = 0.5,
    ) -> FloatArr:
        """Density of the return y = ln(S_T/S_0) minus its deterministic drift."""
        psi = _psi(_U.astype(complex), t, v0, kappa, kappa_theta, sigma, rho, c)
        f = (self._cos @ psi.real + self._sin @ psi.imag) / np.pi
        return np.asarray(f, dtype=np.float64)
