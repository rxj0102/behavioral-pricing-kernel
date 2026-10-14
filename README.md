# behavioral-pricing-kernel

Research repository testing whether a behavioral stochastic discount factor (probability weighting plus a sentiment shift) explains the shape of the empirical pricing kernel of the S&P 500 better than CRRA and habit-style benchmarks, and whether the estimated distortion predicts returns out of sample.

Status: Phase 0 (scaffolding). No empirical results exist yet. Headline results will be added in Phase 9, taken only from code that ran in this repo.

Hypotheses are pre-registered in docs/hypotheses.md.

## Quickstart

    make setup
    make lint
    make test

Credentials (WRDS) are read only from environment variables, see docs/data.md.
