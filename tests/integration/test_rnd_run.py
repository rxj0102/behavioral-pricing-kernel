"""End-to-end RND run on a tiny synthetic panel, with arbitrage counts and provenance."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bpk.config import Config
from bpk.data.build import build
from bpk.rnd.run import rnd_hash, run
from bpk.utils.stats_io import read_stats

pytestmark = pytest.mark.integration


def _cfg(root: Path) -> Config:
    return Config.model_validate(
        {
            "data": {
                "start": "2019-03-04",
                "end": "2019-03-08",
                "cache_dir": str(root / "cache"),
                "manifest_path": str(root / "manifest.json"),
            },
            "rnd": {"n_dates": 2, "named_dates": []},
            "tables_dir": str(root / "tables"),
        }
    )


def test_run_produces_graded_tables_with_provenance(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    build(cfg)
    run(cfg)
    tdir = tmp_path / "tables"
    grading, meta = read_stats(tdir / "rnd_grading__synthetic_heston.csv")
    assert meta["config_hash"] == rnd_hash(cfg) and meta["n_dates"] == "2"
    assert set(grading["region"]) >= {"total", "tail_ext_left", "tail_ext_right", "k<-0.15"}
    assert set(grading["dte"]) >= {21, 30, 180}
    tot = grading[grading["region"] == "total"]
    assert (tot["l1"] < 0.1).all() and (np.abs(tot["est_mass"] - 1) < 1e-6).all()
    fits, _ = read_stats(tdir / "rnd_fit_diagnostics__synthetic_heston.csv")
    nat = fits[fits["method"] != "interp"]
    assert (nat["n_negative"] == 0).all() and (np.abs(nat["mass_raw"] - 1) < 0.01).all()
    arb, _ = read_stats(tdir / "rnd_arbitrage__synthetic_heston.csv")
    assert (arb["calendar_violations"] >= 0).all() and len(arb) > 0
    dens = pd.read_parquet(cfg.data.cache_dir / "rnd30__synthetic_heston.parquet")
    assert dens["density"].min() >= 0


def test_run_is_deterministic(tmp_path: Path) -> None:
    outs = []
    for name in ("a", "b"):
        cfg = _cfg(tmp_path / name)
        build(cfg)
        run(cfg)
        g, _ = read_stats(tmp_path / name / "tables" / "rnd_grading__synthetic_heston.csv")
        outs.append(g)
    pd.testing.assert_frame_equal(outs[0], outs[1])
