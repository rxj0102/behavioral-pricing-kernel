"""Pydantic-validated configuration loaded from YAML.

Unknown keys are rejected so that a typo in a config cannot silently fall
back to a default. The config selected by the environment variable
BPK_CONFIG (or configs/default.yaml) is the single source of run settings.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Sample window and cache locations."""

    start: date
    end: date
    cache_dir: Path = Path("results/cache")
    manifest_path: Path = Path("results/data_manifest.json")
    horizon_days: int = Field(30, gt=0, description="Calendar days of the pricing horizon")

    @model_validator(mode="after")
    def _check_window(self) -> DataConfig:
        if self.end <= self.start:
            raise ValueError("data.end must be after data.start")
        return self


class RunConfig(_Strict):
    """Run-level settings."""

    seed: int = 20240601
    results_log: Path = Path("results/results_log.csv")
    fast: bool = False


class EstimationConfig(_Strict):
    """Estimator settings (used from Phase 5)."""

    n_starts: int = Field(8, gt=0)
    n_bootstrap: int = Field(200, gt=0)


class Config(_Strict):
    """Top-level configuration."""

    run: RunConfig = RunConfig()
    data: DataConfig
    estimation: EstimationConfig = EstimationConfig()


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config. Falls back to BPK_CONFIG then the default file."""
    chosen = Path(path or os.environ.get("BPK_CONFIG", "configs/default.yaml"))
    with chosen.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
