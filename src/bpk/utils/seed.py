"""Deterministic random number generation.

Every stochastic component draws from a generator derived from the global
seed and a string key, so adding a new consumer never changes the stream of
an existing one. Derivation: SeedSequence(entropy=[seed, h(key)]) where h is
the first 8 bytes of SHA-256 of the key, read as an unsigned integer.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _key_to_int(key: str) -> int:
    """Map a string key to a stable 64-bit integer."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def make_rng(seed: int, key: str = "") -> np.random.Generator:
    """Return a reproducible generator for (seed, key)."""
    return np.random.default_rng(np.random.SeedSequence([seed, _key_to_int(key)]))
