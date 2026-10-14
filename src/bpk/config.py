"""Pydantic-validated configuration loaded from YAML.

Unknown keys are rejected so that a typo in a config cannot silently fall
back to a default. The config selected by the environment variable
BPK_CONFIG (or configs/default.yaml) is the single source of run settings.
"""

from __future__ import annotations

import hashlib
import json
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


class SdfTruthConfig(_Strict):
    """Known behavioral SDF used by the synthetic generator (stored for the recovery test).

    SDF on the physical log return y: m(y) proportional to exp(-gamma y) w'(S(y)), with Prelec
    weighting w(p) = exp(-beta (-ln p)^alpha_t). Sentiment s_t is a standardized AR(1) and
    alpha_t = clip(alpha0 - delta s_t, alpha_min, 1), so positive delta means high sentiment
    gives stronger probability-weighting curvature (lower alpha).
    """

    gamma: float = Field(2.0, ge=0)
    alpha0: float = Field(0.9, gt=0, le=1)
    beta: float = Field(1.0, gt=0)
    delta: float = 0.05
    alpha_min: float = Field(0.5, gt=0, le=1)
    sentiment_phi: float = Field(0.99, ge=0, lt=1)


class SyntheticConfig(_Strict):
    """Heston option panel used ONLY for method validation, never for empirical claims.

    Option prices use risk-neutral (Q) dynamics:
        dlnS = (r - q - v/2) dt + sqrt(v) dW1,  dv = kappa (theta - v) dt + sigma sqrt(v) dW2.
    In pricing_mode behavioral_sdf the option prices instead come from the physical return
    density (Heston P-dynamics) multiplied by the known SDF in sdf, see synthetic.py.
    The state path follows physical (P) dynamics: equity drift adds eta * v, and the
    variance process has kappa_P = kappa + lam, theta_P = kappa theta / (kappa + lam).
    """

    pricing_mode: Literal["heston_q", "behavioral_sdf"] = "behavioral_sdf"
    sdf: SdfTruthConfig = SdfTruthConfig()
    kappa: float = Field(3.0, gt=0)
    theta: float = Field(0.04, gt=0)
    sigma: float = Field(0.4, gt=0)
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


def result_hash(cfg: Config) -> str:
    """SHA-256 of the settings that determine results (not output paths).

    Covers run.seed, the data window, source and horizon, the synthetic and cleaning sections.
    """
    payload = {
        "seed": cfg.run.seed,
        "data": cfg.data.model_dump(mode="json", exclude={"cache_dir", "manifest_path"}),
        "synthetic": cfg.synthetic.model_dump(mode="json"),
        "cleaning": cfg.cleaning.model_dump(mode="json"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a config. Falls back to BPK_CONFIG then the default file."""
    chosen = Path(path or os.environ.get("BPK_CONFIG", "configs/default.yaml"))
    with chosen.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config.model_validate(raw)
