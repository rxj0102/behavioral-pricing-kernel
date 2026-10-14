"""Forward inference, cleaning audit table, defect handling."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bpk.config import CleaningConfig, SyntheticConfig
from bpk.data.clean import clean_panel, infer_forward
from bpk.data.synthetic import SyntheticHestonAdapter

CFG = CleaningConfig()


def test_parity_forward_recovers_truth_with_noise() -> None:
    rng = np.random.default_rng(0)
    f_true, r, t = 1000.0, 0.02, 0.25
    k = np.arange(900.0, 1101.0, 10.0)
    disc = np.exp(-r * t)
    call = 40.0 + 0.5 * (f_true - k) * disc + 0.5 * rng.normal(0, 0.02, len(k))
    put = call - disc * (f_true - k) + rng.normal(0, 0.02, len(k))
    f, ok = infer_forward(pd.Series(k), pd.Series(call), pd.Series(put), 1 / disc, CFG)
    assert abs(f / f_true - 1) < 2e-4
    assert 9 <= ok.sum() <= 11  # strikes within about 5 percent of the forward


def test_parity_forward_rejects_outlier_pair() -> None:
    k = np.arange(950.0, 1051.0, 10.0)
    disc = np.exp(-0.02 * 0.25)
    call = 30.0 + 0.5 * (1000.0 - k) * disc
    put = call - disc * (1000.0 - k)
    put[3] += 20.0  # a bad put quote, 2 percent of the forward
    f, ok = infer_forward(pd.Series(k), pd.Series(call), pd.Series(put), 1 / disc, CFG)
    assert abs(f / 1000.0 - 1) < 1e-6
    assert not ok.iloc[3]


def test_too_few_pairs_gives_nan() -> None:
    f, ok = infer_forward(
        pd.Series([1000.0, 1010.0]), pd.Series([30.0, 25.0]), pd.Series([28.0, 30.0]), 1.0, CFG
    )
    assert np.isnan(f) and not ok.any()


@pytest.fixture(scope="module")
def synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    ad = SyntheticHestonAdapter(SyntheticConfig(), seed=7)
    return ad.generate(date(2019, 1, 2), date(2019, 1, 11))


def test_audit_table_is_consistent(synth: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    panel, _ = synth
    res = clean_panel(panel, CFG)
    s = res.stats
    assert s["rows_before"].iloc[0] == len(panel)
    assert (s["rows_before"].iloc[1:].to_numpy() == s["rows_after"].iloc[:-1].to_numpy()).all()
    assert s["rows_after"].iloc[-1] == len(res.quotes)
    assert (s["dropped"] == s["rows_before"] - s["rows_after"]).all()


def test_cleaned_quotes_are_otm_valid_and_forward_accurate(
    synth: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    panel, truth = synth
    q = clean_panel(panel, CFG).quotes
    assert (q["bid"] > 0).all() and (q["bid"] <= q["ask"]).all()
    assert (q["open_interest"] >= CFG.min_open_interest).all()
    puts, calls = q[q["cp"] == "P"], q[q["cp"] == "C"]
    assert (puts["strike"] < puts["forward"]).all() and (calls["strike"] >= calls["forward"]).all()
    g = q.groupby(["date", "dte"]).agg(f=("forward", "first"), s=("spot", "first")).reset_index()
    f_true = g["s"] * np.exp((0.02 - 0.015) * g["dte"] / 365.0)
    assert ((g["f"] / f_true - 1).abs() < 1e-4).all()
    assert len(truth) == 8


def test_crossed_and_missing_defects_are_injected_and_removed(
    synth: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    panel, _ = synth
    assert panel["ask"].isna().any() and (panel["bid"] > panel["ask"]).any()
    q = clean_panel(panel, CFG).quotes
    assert q[["bid", "ask"]].notna().all().all()


def test_synthetic_is_deterministic() -> None:
    a = SyntheticHestonAdapter(SyntheticConfig(), seed=3).generate(
        date(2019, 1, 2), date(2019, 1, 4)
    )
    b = SyntheticHestonAdapter(SyntheticConfig(), seed=3).generate(
        date(2019, 1, 2), date(2019, 1, 4)
    )
    c = SyntheticHestonAdapter(SyntheticConfig(), seed=4).generate(
        date(2019, 1, 2), date(2019, 1, 4)
    )
    pd.testing.assert_frame_equal(a[0], b[0])
    assert not a[1]["spot"].equals(c[1]["spot"])
