"""Daily CBOE chain snapshot collector with safety rails.

Design rules:
- Disabled by default. It runs only if the environment variable BPK_CBOE_TERMS_ACK is
  exactly the string in ACK_VALUE, which the owner sets after reading the CBOE terms.
- Raw output goes to a private directory that must be an absolute path OUTSIDE the
  repository and outside any git work tree, created with mode 0700 (files 0600).
- A manifest (manifest.jsonl, one JSON object per snapshot) records file name, UTC time,
  size, SHA-256, option count and schema hash.
- A schema reference frozen from a human-inspected snapshot (tests/fixtures/cboe_schema.json,
  written by scripts/freeze_cboe_fixture.py) must exist. Any deviation from the checker in
  cboe_schema.py or from the frozen reference raises SchemaChangeError, writes an ALERT file
  into the raw directory, appends to failures.jsonl and makes the script exit nonzero.
  The deviating snapshot is kept under quarantine/, never silently parsed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bpk.data.cboe import URL
from bpk.data.cboe_schema import check_snapshot_schema, describe_schema

ACK_VALUE = "i-have-read-the-cboe-terms-of-use"
ENV_ACK = "BPK_CBOE_TERMS_ACK"
ENV_RAW_DIR = "BPK_CBOE_RAW_DIR"
EXIT_CODES = {"ok": 0, "other": 1, "schema": 2, "not_enabled": 3, "fetch": 4, "config": 5}


class CollectorError(RuntimeError):
    """Base class; exit_kind selects the process exit code."""

    exit_kind = "other"


class NotEnabledError(CollectorError):
    """Terms acknowledgement missing."""

    exit_kind = "not_enabled"


class ConfigError(CollectorError):
    """Bad raw directory or missing schema reference."""

    exit_kind = "config"


class SchemaChangeError(CollectorError):
    """Snapshot does not match the checker or the frozen reference."""

    exit_kind = "schema"


class FetchError(CollectorError):
    """Network or decoding failure."""

    exit_kind = "fetch"


@dataclass(frozen=True)
class ManifestRow:
    """One successful snapshot."""

    file: str
    utc: str
    bytes: int
    sha256: str
    n_options: int
    schema_hash: str


def fetch_live() -> bytes:
    """Download the live chain."""
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            data: bytes = r.read()
    except OSError as exc:
        raise FetchError(f"CBOE fetch failed: {exc}") from exc
    return data


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _in_git_tree(path: Path) -> bool:
    return any((p / ".git").exists() for p in (path, *path.parents))


def prepare_raw_dir(raw_dir: Path, repo_root: Path) -> Path:
    """Validate and create the private raw directory."""
    if not raw_dir.is_absolute():
        raise ConfigError("raw directory must be an absolute path")
    raw = raw_dir.resolve()
    if _inside(raw, repo_root.resolve()) or _in_git_tree(raw):
        raise ConfigError(f"raw directory {raw} is inside the repository or a git work tree")
    if not raw.exists():
        raw.mkdir(parents=True, mode=0o700)
    if stat.S_IMODE(raw.stat().st_mode) & 0o077:
        raise ConfigError(f"raw directory {raw} must not be accessible to group or others (0700)")
    return raw


def _schema_hash(desc: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest()


def _append(path: Path, obj: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def collect(
    fetch: Callable[[], bytes],
    raw_dir: Path,
    schema_ref: Path,
    repo_root: Path,
    env: dict[str, str] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ManifestRow:
    """Take one snapshot. Raises a CollectorError subclass on any problem."""
    environ = os.environ if env is None else env
    if environ.get(ENV_ACK) != ACK_VALUE:
        raise NotEnabledError(
            f"collector disabled: set {ENV_ACK} only after reading the CBOE terms of use"
        )
    raw = prepare_raw_dir(raw_dir, repo_root)
    if not schema_ref.exists():
        raise ConfigError(
            f"frozen schema {schema_ref} missing: run scripts/freeze_cboe_fixture.py and inspect it"
        )
    stamp = now().strftime("%Y%m%dT%H%M%SZ")
    try:
        body = fetch()
    except CollectorError as exc:
        _append(raw / "failures.jsonl", {"utc": stamp, "kind": "fetch", "detail": str(exc)})
        raise
    problems: list[str] = []
    desc: dict[str, Any] = {}
    try:
        payload = json.loads(body)
        problems = check_snapshot_schema(payload)
        if not problems:
            desc = describe_schema(payload)
            ref = json.loads(schema_ref.read_text(encoding="utf-8"))
            if desc != ref:
                problems.append(f"structure differs from frozen reference: now {desc}, was {ref}")
    except (ValueError, KeyError, TypeError) as exc:
        problems.append(f"unparseable snapshot: {exc}")
    if problems:
        q = raw / "quarantine"
        q.mkdir(exist_ok=True, mode=0o700)
        _write_private(q / f"spx_chain_{stamp}.json", body)
        _append(raw / "failures.jsonl", {"utc": stamp, "kind": "schema", "detail": problems})
        (raw / "ALERT_SCHEMA_CHANGE").write_text(
            f"{stamp}: CBOE snapshot failed the schema check. Inspect quarantine/ before "
            f"re-enabling. Problems: {problems}\n",
            encoding="utf-8",
        )
        raise SchemaChangeError("; ".join(problems))
    name = f"spx_chain_{stamp}.json"
    _write_private(raw / name, body)
    row = ManifestRow(
        name,
        stamp,
        len(body),
        hashlib.sha256(body).hexdigest(),
        len(payload["data"]["options"]),
        _schema_hash(desc),
    )
    _append(raw / "manifest.jsonl", asdict(row))
    return row
