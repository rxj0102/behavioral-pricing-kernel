"""OptionMetrics via WRDS (primary source). Phase 0: availability check only.

Credentials are read from the environment variable WRDS_USERNAME (and the
standard WRDS password mechanisms: PGPASSWORD or a pgpass file). Values are
never printed, logged or stored. The query logic is added in Phase 1 once
access is confirmed.
"""

from __future__ import annotations

import importlib.util
import os
from datetime import date

import pandas as pd

from bpk.data.base import DataAdapter, DataUnavailableError, SeriesSpec


class WrdsAdapter(DataAdapter):
    """Adapter for OptionMetrics (SPX) through WRDS."""

    source_name = "wrds_optionmetrics"

    def is_available(self) -> bool:
        """True if the wrds package is installed and WRDS_USERNAME is set."""
        has_pkg = importlib.util.find_spec("wrds") is not None
        return has_pkg and bool(os.environ.get("WRDS_USERNAME"))

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Not implemented until WRDS access is confirmed (Phase 1)."""
        if not self.is_available():
            raise DataUnavailableError(
                "WRDS unavailable: install the wrds extra and set WRDS_USERNAME"
            )
        raise NotImplementedError("WRDS queries are implemented in Phase 1")
