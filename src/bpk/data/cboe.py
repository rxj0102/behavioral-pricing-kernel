"""CBOE delayed SPX option chain (free, no login), snapshot only.

STATUS: the parser is tested only against a hand-built fixture that follows the
layout I expect (data.current_price, data.options[*].option, bid, ask,
open_interest). It has NOT been checked against the live endpoint because the
build sandbox could not reach it. The first live fetch must be inspected by a
person before any result uses it. The endpoint is not a documented API and its
terms of use need checking by the owner.

Option symbol layout assumed: ROOT + YYMMDD + C|P + strike*1000 as 8 digits.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd

from bpk.data.base import DataAdapter, DataUnavailableError, SeriesSpec

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
_SYMBOL = re.compile(r"^(SPXW|SPX)(\d{6})([CP])(\d{8})$")


def parse_snapshot(payload: Mapping[str, Any], asof: date, rf: float) -> pd.DataFrame:
    """Convert a CBOE chain JSON payload to the panel schema.

    Rows with an unparseable symbol are dropped (counted by the caller via length).
    """
    data = payload["data"]
    if not isinstance(data, dict):
        raise ValueError("payload missing data object")
    spot = float(data["current_price"])
    rows = []
    for opt in data["options"]:
        m = _SYMBOL.match(str(opt["option"]))
        if m is None:
            continue
        expiry = pd.to_datetime(m.group(2), format="%y%m%d").date()
        rows.append(
            {
                "date": pd.Timestamp(asof),
                "dte": (expiry - asof).days,
                "strike": int(m.group(4)) / 1000.0,
                "cp": m.group(3),
                "bid": float(opt["bid"]),
                "ask": float(opt["ask"]),
                "open_interest": int(opt["open_interest"]),
                "spot": spot,
                "rf": rf,
            }
        )
    return pd.DataFrame(rows)


class CboeSnapshotAdapter(DataAdapter):
    """Live snapshot of today's chain. Cannot return history."""

    source_name = "cboe_delayed"

    def __init__(self, rf: float) -> None:
        self.rf = rf

    def is_available(self) -> bool:
        """True if the endpoint answers a HEAD-like request within 10 seconds."""
        try:
            with urllib.request.urlopen(URL, timeout=10):
                return True
        except OSError:
            return False

    def fetch(self, spec: SeriesSpec, start: date, end: date) -> pd.DataFrame:
        """Fetch the current chain. Only valid when start == end == today."""
        today = date.today()
        if spec.name != "spx_options" or start != today or end != today:
            raise ValueError("CBOE adapter can only snapshot spx_options for today")
        try:
            with urllib.request.urlopen(URL, timeout=30) as r:
                payload = json.load(r)
        except OSError as exc:
            raise DataUnavailableError("CBOE endpoint unreachable") from exc
        return parse_snapshot(payload, today, self.rf)
