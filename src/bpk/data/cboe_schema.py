"""Schema check for CBOE delayed chain snapshots.

The check lists problems instead of raising, so a changed endpoint layout is reported in
full. Required layout: data.current_price (number); data.options (non-empty list) whose
items have option (symbol string), bid, ask (numbers), open_interest (number). At least
99 percent of symbols must match ROOT + YYMMDD + C|P + 8 digits, and at least 99 percent
of quotes must have bid <= ask.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from bpk.data.cboe import _SYMBOL

REQUIRED_OPTION_KEYS = ("option", "bid", "ask", "open_interest")
MIN_MATCH_SHARE = 0.99


def _is_num(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def check_snapshot_schema(payload: Mapping[str, Any]) -> list[str]:
    """Return a list of schema problems (empty if the snapshot matches the layout)."""
    data = payload.get("data")
    if not isinstance(data, Mapping):
        return ["missing data object"]
    problems: list[str] = []
    if not _is_num(data.get("current_price")):
        problems.append("data.current_price missing or not a number")
    opts = data.get("options")
    if not isinstance(opts, list) or not opts:
        return [*problems, "data.options missing or empty"]
    n_sym = n_cross_ok = 0
    for o in opts:
        miss = [k for k in REQUIRED_OPTION_KEYS if k not in o]
        if miss:
            problems.append(f"option item missing keys {miss}")
            break
        n_sym += bool(_SYMBOL.match(str(o["option"])))
        n_cross_ok += bool(_is_num(o["bid"]) and _is_num(o["ask"]) and o["bid"] <= o["ask"])
    if n_sym / len(opts) < MIN_MATCH_SHARE:
        problems.append(f"only {n_sym}/{len(opts)} symbols match the assumed layout")
    if n_cross_ok / len(opts) < MIN_MATCH_SHARE:
        problems.append(f"only {n_cross_ok}/{len(opts)} quotes numeric with bid <= ask")
    return problems


def describe_schema(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Structure-only description: key names and value types, no prices or symbols."""
    data = payload["data"]
    opt0 = data["options"][0]
    return {
        "top_level_keys": sorted(payload),
        "data_keys": sorted(data),
        "option_keys": {k: type(v).__name__ for k, v in sorted(opt0.items())},
        "n_options_is_positive": len(data["options"]) > 0,
    }


def structural_fixture(payload: Mapping[str, Any], n: int = 6) -> dict[str, Any]:
    """A small fixture with the real key layout and constructed values (no raw quotes).

    Symbols, prices and open interest are generated, never copied from the snapshot.
    """
    data = payload["data"]
    template = data["options"][0]
    options = []
    for i in range(n):
        item = {k: (type(v)(0) if _is_num(v) else v) for k, v in template.items()}
        cp = "C" if i % 2 == 0 else "P"
        item["option"] = f"SPX301220{cp}{(5000 + 50 * i) * 1000:08d}"
        item["bid"], item["ask"], item["open_interest"] = 10.0 + i, 11.0 + i, 100 + i
        options.append(item)
    top = {k: v for k, v in payload.items() if k != "data"}
    inner = {k: v for k, v in data.items() if k != "options"}
    inner = {k: (5000.0 if k == "current_price" else v) for k, v in inner.items()}
    return {**top, "data": {**inner, "options": options}}
