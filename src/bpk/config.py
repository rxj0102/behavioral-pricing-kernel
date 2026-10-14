"""Pydantic-validated configuration loaded from YAML.

Unknown keys are rejected so that a typo in a config cannot silently fall
back to a default. The config selected by the environment variable
BPK_CONFIG (or configs/default.yaml) is the single source of run settings.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    """Sample window and cache locations."""

    start: date
    end: date
    source: Literal["synthetic", "wrds", "cboe"] = "synthetic"
    cache_dir: Path = Path("results/cache")
    manifest_path: Path = Path("results/data_manifest.json")
    horizon_days: int = Field(30, gt=0, description="Calendar days of the pricing horizon")

    @model_validator(mode="after")
    def _check_window(self) -> DataConfig:
        if self.end <= self.start:
            raise ValueError("data.end must be after data.start")
        return self


class SyntheticConfig(_Strict):
    """Heston option panel used ONLY for method validation, never for empirical claims.

    Option prices use risk-neutral (Q) dynamics:
        dlnS = (r - q - v/2) dt + sqrt(v) dW1,  dv = kappa (theta - v) dt + sigma sqrt(v) dW2.
    The state path follows physical (P) dynamics: equity drift adds eta * v, and the
    variance process has kappa_P = kappa + lam, theta_P = kappa theta / (kappa + lam).
    """

    kappa: float = Field(3.0, gt=0)
    theta: float = Field(0.04, gt=0)
    sigma: float = Field(0.5, gt=0)
    rho: float = Field(-0.7, gt=-1, lt=1)
    v0: float = Field(0.04, gt=0)
    spot0: float = Field(1000.0, gt=0)
    rate: float = 0.02
    div_yield: float = 0.015
    eta: float = 2.0
    lam: float = 1.5
    maturities_days: tuple[int, ...] = (21, 45, 80, 120, 180)
    crossed_rate: float = Field(0.002, ge=0, lt=1)
    missing_rate: float = Field(0.01, ge=0, lt=1)


class CleaningConfig(_Strict):
    """Quote filters. Each filter is counted in the cleaning statistics table."""

    min_open_interest: int = Field(10, ge=0)
    min_mid: float = Field(0.10, ge=0)
    max_rel_spread: float = Field(0.5, gt=0)
    parity_window: float = Field(0.05, gt=0, description="Moneyness window for forward inference")
    parity_tol: float = Field(0.01, gt=0, description="Max relative deviation of a pair forward")
    min_pairs: int = Field(3, ge=2)


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
    synthetic: SyntheticConfig = SyntheticConfig()
    cleaning: CleaningConfig = CleaningConfig()
    tables_dir: Path = Path("results/tables")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config. Falls back to BPK_CONFIG then the default file."""
    chosen = Path(path or os.environ.get("BPK_CONFIG", "configs/default.yaml"))
    with chosen.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
