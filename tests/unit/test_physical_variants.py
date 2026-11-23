"""The physical density variants used by the post hoc robustness check."""

import pytest

from bpk.study.run import PHYSICAL_VARIANTS, build


def test_pre_specified_density_is_listed_first() -> None:
    assert next(iter(PHYSICAL_VARIANTS)) == "fhs"


def test_unknown_variant_is_rejected() -> None:
    with pytest.raises(ValueError):
        build(None, "sp6m", 10, 1, physical="nope")  # type: ignore[arg-type]
