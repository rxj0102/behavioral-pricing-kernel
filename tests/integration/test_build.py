"""End-to-end dataset build on a tiny synthetic window, with determinism check."""

from pathlib import Path

import pandas as pd
import pytest

from bpk.config import Config
from bpk.data.build import build
from bpk.data.manifest import Manifest

pytestmark = pytest.mark.integration


def _cfg(root: Path, seed: int) -> Config:
    return Config.model_validate(
        {
            "run": {"seed": seed},
            "data": {
                "start": "2019-01-02",
                "end": "2019-01-11",
                "cache_dir": str(root / "cache"),
                "manifest_path": str(root / "manifest.json"),
            },
            "tables_dir": str(root / "tables"),
        }
    )


def test_build_is_deterministic_and_manifested(tmp_path: Path) -> None:
    q1, s1 = build(_cfg(tmp_path / "a", 5))
    q2, s2 = build(_cfg(tmp_path / "b", 5))
    q3, _ = build(_cfg(tmp_path / "c", 6))
    pd.testing.assert_frame_equal(pd.read_parquet(q1), pd.read_parquet(q2))
    assert s1.read_text() == s2.read_text()
    assert not pd.read_parquet(q1).equals(pd.read_parquet(q3))
    entry = next(iter(Manifest(tmp_path / "a" / "manifest.json").entries.values()))
    assert entry.source == "synthetic_heston"
