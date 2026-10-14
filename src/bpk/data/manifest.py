"""Data manifest: provenance of every cached dataset.

Each entry records source, series, date range, SHA-256 of the cached parquet
file, row count and retrieval date. The manifest is JSON, sorted by key, so
diffs are readable and the file is itself deterministic given its inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    """Provenance record for one cached dataset."""

    series: str
    source: str
    start: str
    end: str
    sha256: str
    rows: int
    retrieved: str  # ISO date


class Manifest:
    """Mutable collection of entries persisted to a JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[str, ManifestEntry] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.entries = {k: ManifestEntry(**v) for k, v in raw.items()}

    def record(self, entry: ManifestEntry) -> None:
        """Insert or replace the entry for entry.series and save."""
        self.entries[entry.series] = entry
        self.save()

    def save(self) -> None:
        """Write the manifest to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: asdict(v) for k, v in sorted(self.entries.items())}
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
