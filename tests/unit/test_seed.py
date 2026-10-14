"""Determinism of seeded generators."""

import numpy as np

from bpk.utils.seed import make_rng


def test_same_seed_same_key_identical() -> None:
    a = make_rng(1, "x").standard_normal(100)
    b = make_rng(1, "x").standard_normal(100)
    assert np.array_equal(a, b)


def test_different_key_differs() -> None:
    a = make_rng(1, "x").standard_normal(100)
    b = make_rng(1, "y").standard_normal(100)
    assert not np.array_equal(a, b)


def test_different_seed_differs() -> None:
    assert not np.array_equal(make_rng(1, "x").random(10), make_rng(2, "x").random(10))
