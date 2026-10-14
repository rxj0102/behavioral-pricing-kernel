"""Parquet cache wrapping any DataAdapter, with manifest bookkeeping.

On a cache hit the adapter is not called and the stored file hash is
verified against the manifest, so silent corruption or edits are detected.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from bpk.data.base import DataAdapter, SeriesSpec
from bpk.data.manifest import Manifest, ManifestEntry
from bpk.utils.hashing import sha256_file


class CacheIntegrityError(RuntimeError):
    """Cached file hash differs from the manifest."""


class CachedSource:
    """Fetch through a parquet cache. The clock is injectable for tests."""

    def __init__(
        self,
        adapter: DataAdapter,
        cache_dir: Path,
        manifest: Manifest,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.adapter = adapter
        self.cache_dir = cache_dir
        self.manifest = manifest
        self._today = today

    def _path(self, spec: SeriesSpec, start: date, end: date) -> Path:
        stem = f"{self.adapter.source_name}__{spec.name}__{start:%Y%m%d}_{end:%Y%m%d}"
        return self.cache_dir / f"{stem}.parquet"

    def get(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Return the dataset, fetching and caching it on first use."""
        path = self._path(spec, start, end)
        key = path.stem
        if path.exists():
            entry = self.manifest.entries.get(key)
            if entry is None or entry.sha256 != sha256_file(path):
                raise CacheIntegrityError(f"cache file {path.name} does not match manifest")
            return pd.read_parquet(path)
        df = self.adapter.fetch(spec, start, end)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        self.manifest.record(
            ManifestEntry(
                series=key,
                source=self.adapter.source_name,
                start=start.isoformat(),
                end=end.isoformat(),
                sha256=sha256_file(path),
                rows=len(df),
                retrieved=self._today().isoformat(),
            )
        )
        return df
