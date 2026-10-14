"""Abstract data adapter interface.

A source (WRDS, FRED, a live chain collector, a synthetic generator) is a
DataAdapter. Downstream code depends only on this interface, so sources can
be swapped without touching the methodology modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class SeriesSpec:
    """Identifies one dataset: a logical name plus the source parameters."""

    name: str
    frequency: str  # for example daily, monthly, panel

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("series name must be non-empty and contain no slash")


class DataUnavailableError(RuntimeError):
    """Raised when a source cannot be used (missing credentials, no network)."""


class DataAdapter(ABC):
    """Interface every data source implements."""

    #: Short stable identifier recorded in the manifest.
    source_name: str

    @abstractmethod
    def is_available(self) -> bool:
        """True if the source can be queried now (credentials present, etc.).

        Must never print or return credential values.
        """

    @abstractmethod
    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Return the series for [start, end] with a sorted DatetimeIndex or a date column.

        Raises DataUnavailableError if the source is not available.
        """
