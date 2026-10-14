"""Config loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from bpk.config import Config, load_config

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["default.yaml", "fast.yaml"])
def test_shipped_configs_validate(name: str) -> None:
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.data.horizon_days == 30
    assert cfg.run.seed == 20240601


def test_fast_config_is_smaller() -> None:
    fast = load_config(ROOT / "configs" / "fast.yaml")
    full = load_config(ROOT / "configs" / "default.yaml")
    assert fast.run.fast and not full.run.fast
    assert fast.estimation.n_bootstrap < full.estimation.n_bootstrap


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"start": "2019-01-01", "end": "2019-12-31"}, "typo_key": 1})


def test_bad_window_rejected() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"data": {"start": "2020-01-01", "end": "2019-01-01"}})
