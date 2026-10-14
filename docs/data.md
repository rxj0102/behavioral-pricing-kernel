# Data

Status: Phase 0. Sources are chosen, nothing is ingested yet. This file is completed in Phase 1 with every series, its frequency, coverage and known limitations.

## Data decision

Primary: OptionMetrics through WRDS, SPX option prices and volatility surface. Requires WRDS credentials supplied by the owner as environment variable WRDS_USERNAME (and the usual WRDS password mechanism). Credentials are never hardcoded, printed or logged.

Fallback, if WRDS is not available: to be chosen by the owner at the end of Phase 0. Candidates: (a) a daily collector snapshotting the live SPX option chain going forward, which gives a short sample, stated plainly in the paper; (b) a synthetic Heston option panel, used only to validate methods and never for empirical claims.

## Planned series

| Series | Source | Frequency | Known limitation (to confirm in Phase 1) |
|---|---|---|---|
| SPX options, volatility surface | OptionMetrics via WRDS | daily | needs credentials |
| SPX total return, daily returns | to be decided | daily | |
| Realized variance | free source or WRDS | daily | measurement frequency affects level |
| VIX | CBOE | daily | |
| Risk-free rate | FRED | daily or monthly | |
| Baker-Wurgler sentiment | Jeffrey Wurgler web page | monthly | revisions, publication lag |
| AAII sentiment | AAII | weekly | survey based, licensing to check |
| Fama-French factors | Ken French data library | daily, monthly | |
| Macro controls | FRED | monthly | revisions, release lags |

## Provenance

Every cached dataset is recorded in the manifest (results/data_manifest.json) with source, series, date range, SHA-256 of the cached file, row count and retrieval date. A cached file whose hash differs from the manifest raises an error.
