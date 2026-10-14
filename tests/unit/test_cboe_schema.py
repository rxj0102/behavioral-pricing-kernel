"""Schema checker and structural fixture builder (checked on a constructed payload)."""

import copy
from datetime import date
from typing import Any

from bpk.data.cboe import parse_snapshot
from bpk.data.cboe_schema import check_snapshot_schema, describe_schema, structural_fixture
from tests.unit.test_cboe import FIXTURE


def _good() -> dict[str, Any]:
    base = copy.deepcopy(FIXTURE)
    base["data"]["options"] = [
        {
            "option": f"SPX290119C{(5000 + 5 * i) * 1000:08d}",
            "bid": 1.0,
            "ask": 2.0,
            "open_interest": 10,
            "iv": 0.2,
        }
        for i in range(200)
    ]
    return base


def test_good_payload_passes() -> None:
    assert check_snapshot_schema(_good()) == []


def test_bad_payloads_are_reported() -> None:
    assert check_snapshot_schema({}) == ["missing data object"]
    bad = _good()
    bad["data"]["options"][0].pop("bid")
    assert any("missing keys" in p for p in check_snapshot_schema(bad))
    bad2 = _good()
    for o in bad2["data"]["options"]:
        o["option"] = "NOPE"
    assert any("symbols" in p for p in check_snapshot_schema(bad2))
    bad3 = _good()
    for o in bad3["data"]["options"]:
        o["bid"], o["ask"] = 5.0, 1.0
    assert any("bid <= ask" in p for p in check_snapshot_schema(bad3))


def test_structural_fixture_has_same_layout_but_no_copied_values() -> None:
    src = _good()
    fx = structural_fixture(src, n=6)
    assert describe_schema(fx)["option_keys"] == describe_schema(src)["option_keys"]
    assert check_snapshot_schema(fx) == []
    src_symbols = {o["option"] for o in src["data"]["options"]}
    assert not src_symbols & {o["option"] for o in fx["data"]["options"]}
    assert len(parse_snapshot(fx, date(2026, 1, 2), 0.04)) == 6
