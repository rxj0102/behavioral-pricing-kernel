"""Results log records every spec, failures included."""

from pathlib import Path

import pandas as pd
import pytest

from bpk.utils.results_log import count_specs, log_spec


def test_log_appends_and_counts(tmp_path: Path) -> None:
    p = tmp_path / "log.csv"
    assert count_specs(p) == 0
    log_spec(p, phase="p0", spec_id="a", status="ok", params={"k": 1}, metrics={"m": 2.0})
    log_spec(p, phase="p0", spec_id="b", status="failed", params={"k": 2}, note="did not converge")
    assert count_specs(p) == 2
    df = pd.read_csv(p)
    assert df["status"].tolist() == ["ok", "failed"]


def test_invalid_status_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        log_spec(tmp_path / "l.csv", phase="p", spec_id="s", status="great", params={})
