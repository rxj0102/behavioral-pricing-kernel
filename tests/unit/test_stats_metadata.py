"""Stats files carry the config hash and date range, and they are checked."""

from pathlib import Path

import pandas as pd

from bpk.config import Config, result_hash
from bpk.data.build import build
from bpk.utils.stats_io import read_stats, write_stats


def _cfg(root: Path, seed: int = 5, end: str = "2019-01-11") -> Config:
    return Config.model_validate(
        {
            "run": {"seed": seed},
            "data": {
                "start": "2019-01-02",
                "end": end,
                "cache_dir": str(root / "cache"),
                "manifest_path": str(root / "manifest.json"),
            },
            "tables_dir": str(root / "tables"),
        }
    )


def test_stats_roundtrip_with_metadata(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    write_stats(df, tmp_path / "s.csv", {"config_hash": "abc", "requested_start": "2019-01-01"})
    out, meta = read_stats(tmp_path / "s.csv")
    pd.testing.assert_frame_equal(out, df)
    assert meta == {"config_hash": "abc", "requested_start": "2019-01-01"}


def test_every_stats_file_has_hash_and_range(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    build(cfg)
    files = sorted((tmp_path / "tables").glob("*.csv"))
    assert len(files) == 2  # aggregate and by-moneyness
    for f in files:
        _, meta = read_stats(f)
        assert meta["config_hash"] == result_hash(cfg)
        assert meta["requested_start"] == "2019-01-02" and meta["requested_end"] == "2019-01-11"
        assert "2019-01-02" <= meta["observed_start"] <= meta["observed_end"] <= "2019-01-11"
        assert int(meta["clean_rows"]) <= int(meta["raw_rows"])
        assert meta["source"] == "synthetic_heston"


def test_hash_tracks_result_relevant_settings_only(tmp_path: Path) -> None:
    base = result_hash(_cfg(tmp_path / "a"))
    assert result_hash(_cfg(tmp_path / "b")) == base  # output paths do not matter
    assert result_hash(_cfg(tmp_path / "c", seed=6)) != base
    assert result_hash(_cfg(tmp_path / "d", end="2019-01-14")) != base
