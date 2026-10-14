"""Adapter interface, parquet cache, manifest and integrity checks."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bpk.data.base import DataAdapter, DataUnavailableError, SeriesSpec
from bpk.data.cache import CachedSource, CacheIntegrityError
from bpk.data.manifest import Manifest
from bpk.data.wrds_adapter import WrdsAdapter


class FakeAdapter(DataAdapter):
    source_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        self.calls += 1
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"x": range(len(idx))}, index=idx)


SPEC = SeriesSpec("demo", "daily")
START, END = date(2020, 1, 1), date(2020, 1, 10)


def _source(tmp_path: Path, adapter: FakeAdapter) -> CachedSource:
    return CachedSource(
        adapter, tmp_path / "cache", Manifest(tmp_path / "m.json"), today=lambda: date(2026, 1, 2)
    )


def test_cache_hit_skips_fetch_and_records_manifest(tmp_path: Path) -> None:
    ad = FakeAdapter()
    src = _source(tmp_path, ad)
    first = src.get(SPEC, START, END)
    second = src.get(SPEC, START, END)
    assert ad.calls == 1
    assert first["x"].tolist() == second["x"].tolist()
    entry = next(iter(Manifest(tmp_path / "m.json").entries.values()))
    assert entry.source == "fake" and entry.rows == 10 and entry.retrieved == "2026-01-02"
    assert len(entry.sha256) == 64


def test_tampered_cache_detected(tmp_path: Path) -> None:
    src = _source(tmp_path, FakeAdapter())
    src.get(SPEC, START, END)
    parquet = next((tmp_path / "cache").glob("*.parquet"))
    pd.DataFrame({"x": [999]}).to_parquet(parquet)
    with pytest.raises(CacheIntegrityError):
        src.get(SPEC, START, END)


def test_series_name_validation() -> None:
    with pytest.raises(ValueError):
        SeriesSpec("a/b", "daily")


def test_wrds_unavailable_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    ad = WrdsAdapter()
    assert not ad.is_available()
    with pytest.raises(DataUnavailableError):
        ad.fetch(SPEC, START, END)
