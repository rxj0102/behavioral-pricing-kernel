"""Statistics tables with provenance metadata.

Every stats file starts with comment lines of the form `# key: value` (config hash,
source, requested and observed date range, row counts), followed by a normal CSV.
Reading skips the comment lines, so the file stays readable by pandas.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd


def write_stats(df: pd.DataFrame, path: Path, meta: Mapping[str, object]) -> None:
    """Write df as CSV with a metadata header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "".join(f"# {k}: {v}\n" for k, v in meta.items())
    path.write_text(head + df.to_csv(index=False), encoding="utf-8")


def read_stats(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read a stats file written by write_stats. Returns (table, metadata)."""
    meta: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("# "):
            break
        k, _, v = line[2:].partition(": ")
        meta[k] = v
    return pd.read_csv(path, comment="#"), meta
