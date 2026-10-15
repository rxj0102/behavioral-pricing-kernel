"""Option panel schema shared by every source.

Columns: date (datetime64), dte (calendar days to expiry), strike, cp (C or P),
bid, ask, open_interest, spot, rf (continuously compounded risk-free rate).
"""

from __future__ import annotations

import pandas as pd

PANEL_COLUMNS = ("date", "dte", "strike", "cp", "bid", "ask", "open_interest", "spot", "rf")


def validate_panel(df: pd.DataFrame) -> None:
    """Raise ValueError if required columns are missing or cp has bad values."""
    missing = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")
    if not set(df["cp"].unique()) <= {"C", "P"}:
        raise ValueError("cp must be C or P")
