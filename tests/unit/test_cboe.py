"""CBOE parser against a HAND-BUILT fixture (live format unverified)."""

from datetime import date

from bpk.data.cboe import parse_snapshot
from bpk.data.schema import validate_panel

FIXTURE = {
    "data": {
        "current_price": 5000.0,
        "options": [
            {"option": "SPX241220C05000000", "bid": 100.0, "ask": 101.0, "open_interest": 500},
            {"option": "SPXW241115P04900000", "bid": 20.0, "ask": 21.0, "open_interest": 90},
            {"option": "BADSYMBOL", "bid": 1.0, "ask": 2.0, "open_interest": 1},
        ],
    }
}


def test_parse_snapshot_fields() -> None:
    df = parse_snapshot(FIXTURE, date(2024, 11, 1), rf=0.045)
    validate_panel(df)
    assert len(df) == 2  # unparseable symbol dropped
    row = df.iloc[0]
    assert row["strike"] == 5000.0 and row["cp"] == "C" and row["dte"] == 49
    assert df.iloc[1]["strike"] == 4900.0 and df.iloc[1]["cp"] == "P"
