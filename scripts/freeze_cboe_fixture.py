"""Freeze the structure of one live CBOE snapshot as a parser fixture.

Run when the network allows cdn.cboe.com. It
 1. downloads the SPX delayed chain,
 2. saves the RAW snapshot to data_raw/cboe/ (git-ignored, never committed),
 3. runs the schema check and refuses to write fixtures if it fails,
 4. writes tests/fixtures/cboe_schema.json (key names and types only) and
    tests/fixtures/cboe_structural_fixture.json (real layout, constructed values),
 5. with --terms, also saves the CBOE legal pages to data_raw/cboe/ for a person to read.
The terms of use are NOT interpreted by this script. Record the human conclusion in docs/data.md.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from bpk.data.cboe import URL
from bpk.data.cboe_schema import check_snapshot_schema, describe_schema, structural_fixture

ROOT = Path(__file__).resolve().parents[1]
TERMS_URLS = ("https://www.cboe.com/about/legal/", "https://www.cboe.com/delayed_quotes/spx/")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as r:
        data: bytes = r.read()
    return data


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", action="store_true", help="also save legal pages for reading")
    args = ap.parse_args()
    raw_dir = ROOT / "data_raw" / "cboe"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = _get(URL)
    (raw_dir / f"spx_chain_{stamp}.json").write_bytes(raw)
    payload = json.loads(raw)
    problems = check_snapshot_schema(payload)
    if problems:
        raise SystemExit("schema check failed, no fixtures written: " + "; ".join(problems))
    fx = ROOT / "tests" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    (fx / "cboe_schema.json").write_text(
        json.dumps(describe_schema(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (fx / "cboe_structural_fixture.json").write_text(
        json.dumps(structural_fixture(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("fixtures written; raw snapshot kept in data_raw/cboe (git-ignored)")
    if args.terms:
        for i, u in enumerate(TERMS_URLS):
            (raw_dir / f"terms_{i}_{stamp}.html").write_bytes(_get(u))
        print("legal pages saved to data_raw/cboe for a person to read")


if __name__ == "__main__":
    main()
