"""Collector safety rails, tested with an injected fetch function."""

import json
import os
import stat
from pathlib import Path

import pytest

from bpk.data.cboe_schema import describe_schema
from bpk.data.collector import (
    ACK_VALUE,
    ENV_ACK,
    ConfigError,
    NotEnabledError,
    SchemaChangeError,
    collect,
)

GOOD = {
    "timestamp": "x",
    "data": {
        "current_price": 5000.0,
        "options": [
            {
                "option": f"SPX290119C{(5000 + 5 * i) * 1000:08d}",
                "bid": 1.0,
                "ask": 2.0,
                "open_interest": 10,
                "iv": 0.2,
            }
            for i in range(200)
        ],
    },
}
ENV = {ENV_ACK: ACK_VALUE}


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(describe_schema(GOOD)))
    return repo, ref, tmp_path / "private" / "raw"


def _fetch(payload: object) -> bytes:
    return json.dumps(payload).encode()


def test_disabled_without_ack(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, raw = setup
    with pytest.raises(NotEnabledError):
        collect(lambda: _fetch(GOOD), raw, ref, repo, env={})
    assert not raw.exists()


def test_raw_dir_inside_repo_refused(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, _ = setup
    with pytest.raises(ConfigError):
        collect(lambda: _fetch(GOOD), repo / "data_raw", ref, repo, env=ENV)
    with pytest.raises(ConfigError):
        collect(lambda: _fetch(GOOD), Path("relative/dir"), ref, repo, env=ENV)


def test_missing_frozen_reference_refused(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, raw = setup
    ref.unlink()
    with pytest.raises(ConfigError, match="freeze"):
        collect(lambda: _fetch(GOOD), raw, ref, repo, env=ENV)


def test_success_writes_private_files_and_manifest(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, raw = setup
    row = collect(lambda: _fetch(GOOD), raw, ref, repo, env=ENV)
    assert stat.S_IMODE(raw.stat().st_mode) == 0o700
    f = raw / row.file
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    line = json.loads((raw / "manifest.jsonl").read_text().splitlines()[0])
    assert line["sha256"] == row.sha256 and line["n_options"] == 200
    assert not (raw / "ALERT_SCHEMA_CHANGE").exists()


@pytest.mark.parametrize("mutation", ["drop_key", "rename_field", "garbage"])
def test_schema_change_fails_loudly(setup: tuple[Path, Path, Path], mutation: str) -> None:
    repo, ref, raw = setup
    bad = json.loads(json.dumps(GOOD))
    body: bytes
    if mutation == "drop_key":
        del bad["data"]["options"][0]["bid"]
        body = _fetch(bad)
    elif mutation == "rename_field":
        for o in bad["data"]["options"]:
            o["open_int"] = o.pop("open_interest")
        body = _fetch(bad)
    else:
        body = b"<html>not json</html>"
    with pytest.raises(SchemaChangeError):
        collect(lambda: body, raw, ref, repo, env=ENV)
    assert (raw / "ALERT_SCHEMA_CHANGE").exists()
    assert len(list((raw / "quarantine").iterdir())) == 1
    assert json.loads((raw / "failures.jsonl").read_text().splitlines()[0])["kind"] == "schema"
    assert not (raw / "manifest.jsonl").exists()


def test_extra_unseen_field_is_a_schema_change(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, raw = setup
    changed = json.loads(json.dumps(GOOD))
    for o in changed["data"]["options"]:
        o["new_field"] = 1
    with pytest.raises(SchemaChangeError):
        collect(lambda: _fetch(changed), raw, ref, repo, env=ENV)


def test_open_permissions_rejected(setup: tuple[Path, Path, Path]) -> None:
    repo, ref, raw = setup
    raw.mkdir(parents=True)
    os.chmod(raw, 0o755)
    with pytest.raises(ConfigError):
        collect(lambda: _fetch(GOOD), raw, ref, repo, env=ENV)
