# behavioral-pricing-kernel

[![CI](https://github.com/rxj0102/behavioral-pricing-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/behavioral-pricing-kernel/actions/workflows/ci.yml)

Research question: does a behavioral stochastic discount factor (probability weighting plus a sentiment shift) explain the shape of the empirical pricing kernel of the S&P 500 better than CRRA and habit-style benchmarks, and does the estimated distortion predict returns out of sample?

The hypotheses H1 to H4 were written down before any estimation and are in `docs/hypotheses.md`. Later changes are dated amendments that leave the original text intact.

## Status

This is a work in progress, built in phase gates. Only the first stages are implemented.

| Stage | Module | State |
|---|---|---|
| Data layer (CBOE snapshot collector, WRDS adapter, cleaning, manifest) | `bpk.data` | implemented |
| Risk-neutral density extraction (monotone, convex, tail rules, grading) | `bpk.rnd` | implemented, validated on synthetic data |
| Behavioral weighting functions | `bpk.models.weighting` | implemented |
| Physical density, empirical kernel, estimation, prediction, backtest | `bpk.physical`, `bpk.kernel`, `bpk.estimation`, `bpk.predict`, `bpk.backtest` | not started; `make` targets fail loudly |

No result in this repo is evidence about real markets. The tables in `results/tables` come from a synthetic Heston panel and serve as method validation only: they show what the density extraction can and cannot recover when the truth is known. H2 cannot be tested on synthetic data at all, because the synthetic sentiment series drives the distortion by construction (see Amendment 2 in the hypotheses file). H3 and H4 are planned as a power study on the synthetic panel, because the free real-data window will be too short for out-of-sample tests.

## Design choices

- Free data only. WRDS is optional and read from environment variables; nothing paid is required and no credentials are stored.
- Every specification tried is appended to `results/results_log.csv` by code, including rejected ones.
- Nothing is tuned on the out-of-sample period.
- Known limits are tracked in `docs/limitations.md`.

## Quickstart

    make setup
    make lint
    make test
    make data CONFIG=configs/fast.yaml
    make rnd CONFIG=configs/fast.yaml

## Layout

    src/bpk/      package (data, rnd, models, and stubs for later stages)
    configs/      run configurations
    docs/         hypotheses, methodology, limitations, phase plans
    results/      results log and synthetic-panel tables
    tests/        unit and integration tests

Citation metadata is in `CITATION.cff`.
