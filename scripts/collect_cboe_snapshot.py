"""Entry point for the daily CBOE snapshot (see docs/collector.md).

Environment: BPK_CBOE_TERMS_ACK (see bpk.data.collector), BPK_CBOE_RAW_DIR (absolute path
outside the repository), optional BPK_CBOE_SCHEMA_REF (default tests/fixtures/cboe_schema.json).
Exit codes: 0 ok, 2 schema change, 3 not enabled, 4 fetch failure, 5 configuration, 1 other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bpk.data.collector import ENV_RAW_DIR, EXIT_CODES, CollectorError, collect, fetch_live

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run one snapshot and return the process exit code."""
    raw = os.environ.get(ENV_RAW_DIR)
    if not raw:
        print(f"ERROR: {ENV_RAW_DIR} is not set", file=sys.stderr)
        return EXIT_CODES["config"]
    ref = Path(os.environ.get("BPK_CBOE_SCHEMA_REF", ROOT / "tests/fixtures/cboe_schema.json"))
    try:
        row = collect(fetch_live, Path(raw), ref, ROOT)
    except CollectorError as exc:
        print(f"COLLECTOR FAILED [{exc.exit_kind}]: {exc}", file=sys.stderr)
        return EXIT_CODES[exc.exit_kind]
    print(f"ok {row.file} options={row.n_options} sha256={row.sha256[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
