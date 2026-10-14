"""Append-only log of every specification tried, including failures.

Each call to log_spec appends one row to a CSV. The paper reports the full
set, and the Deflated Sharpe Ratio uses the row count as the number of
trials, so nothing may be run without being logged.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

COLUMNS = ("phase", "spec_id", "status", "params_json", "metrics_json", "note")
_VALID_STATUS = {"ok", "failed", "rejected"}


def log_spec(
    path: Path,
    *,
    phase: str,
    spec_id: str,
    status: str,
    params: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
    note: str = "",
) -> None:
    """Append one specification row. Creates the file with a header if needed."""
    if status not in _VALID_STATUS:
        raise ValueError(f"status must be one of {sorted(_VALID_STATUS)}, got {status}")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        w.writerow(
            [
                phase,
                spec_id,
                status,
                json.dumps(dict(params), sort_keys=True),
                json.dumps(dict(metrics or {}), sort_keys=True),
                note,
            ]
        )


def count_specs(path: Path) -> int:
    """Number of specifications logged so far (0 if the log does not exist)."""
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        return max(sum(1 for _ in fh) - 1, 0)
