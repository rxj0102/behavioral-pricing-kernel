"""Noise injection mechanics and a tiny end-to-end run."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bpk.config import Config, SyntheticConfig
from bpk.data.synthetic import SyntheticHestonAdapter
from bpk.rnd.noise_study import SPECS, NoiseSpec, apply_noise, run
from bpk.utils.seed import make_rng
from bpk.utils.stats_io import read_stats


@pytest.fixture(scope="module")
def two_days() -> tuple[pd.DataFrame, pd.DataFrame]:
    ad = SyntheticHestonAdapter(SyntheticConfig(), 3)
    p = ad.generate(date(2019, 3, 4), date(2019, 3, 5)).panel
    d = sorted(p["date"].unique())
    return p[p["date"] == d[1]].reset_index(drop=True), p[p["date"] == d[0]].reset_index(drop=True)


def test_zero_noise_is_identity(two_days: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cur, prev = two_days
    out = apply_noise(cur, prev, NoiseSpec("z", "base", 0.0), make_rng(1, "a"))
    pd.testing.assert_frame_equal(out, cur)


def test_jitter_widens_dispersion_and_is_deterministic(
    two_days: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    cur, prev = two_days
    spec = NoiseSpec("j", "jitter", 2.0, jitter=2.0)
    a = apply_noise(cur, prev, spec, make_rng(1, "a"))
    b = apply_noise(cur, prev, spec, make_rng(1, "a"))
    pd.testing.assert_frame_equal(a, b)
    ok = cur["ask"].notna() & a["ask"].notna()
    assert np.abs(a.loc[ok, "ask"] - cur.loc[ok, "ask"]).mean() > 0.02
    assert (a["bid"] >= 0).all() and len(a) == len(cur)


def test_stale_quotes_take_previous_day_values(
    two_days: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    cur, prev = two_days
    out = apply_noise(cur, prev, NoiseSpec("s", "stale", 1.0, stale=1.0), make_rng(1, "a"))
    key = ["dte", "strike", "cp"]
    m = out.merge(prev[[*key, "bid", "ask"]], on=key, suffixes=("", "_p"))
    assert len(m) > 0.5 * len(cur)
    assert np.allclose(m["bid"], m["bid_p"]) and np.allclose(
        m["ask"].fillna(-1), m["ask_p"].fillna(-1)
    )


def test_drop_rate_is_close_to_requested(two_days: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    cur, prev = two_days
    out = apply_noise(cur, prev, NoiseSpec("d", "drop", 0.4, drop=0.4), make_rng(1, "a"))
    assert abs(1 - len(out) / len(cur) - 0.4) < 0.03


def test_spec_names_unique_and_levels_ordered() -> None:
    names = [s.name for s in SPECS]
    assert len(set(names)) == len(names)
    for kind in ("jitter", "stale", "drop", "combined"):
        lv = [s.level for s in SPECS if s.kind == kind]
        assert lv == sorted(lv)


@pytest.mark.integration
def test_tiny_study_runs_and_writes_tables(tmp_path: Path) -> None:
    cfg = Config.model_validate(
        {
            "data": {
                "start": "2019-03-01",
                "end": "2019-03-29",
                "cache_dir": str(tmp_path / "c"),
                "manifest_path": str(tmp_path / "m.json"),
            },
            "tables_dir": str(tmp_path / "t"),
        }
    )
    specs = (
        SPECS[0],
        NoiseSpec("jitter_2.0", "jitter", 2.0, jitter=2.0),
        NoiseSpec("drop_0.4", "drop", 0.4, drop=0.4),
    )
    summ = run(cfg, 2, tmp_path / "t", specs)
    assert list(summ["spec"]) == ["base", "jitter_2.0", "drop_0.4"]
    assert (summ["l1_native_median"] < 0.2).all()
    tbl, meta = read_stats(tmp_path / "t" / "noise_study_by_regime__synthetic_heston.csv")
    assert meta["n_dates"] == "2" and set(tbl["regime"]) <= {"low", "mid", "high"}
