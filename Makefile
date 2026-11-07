# Every target uses the config selected by CONFIG (default: configs/default.yaml).
# Targets for later phases fail loudly rather than pretend to work.
PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: setup data rnd physical epk estimate predict backtest figures paper dashboard test lint all

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

data:
	$(PY) -m bpk.data.build --config $(CONFIG)
rnd:
	$(PY) -m bpk.rnd.run --config $(CONFIG)
physical:
	@echo "physical: not implemented until Phase 3" >&2; exit 1
epk:
	@echo "epk: not implemented until Phase 4" >&2; exit 1
estimate:
	@echo "estimate: not implemented until Phase 5" >&2; exit 1
predict:
	@echo "predict: not implemented until Phase 6" >&2; exit 1
backtest:
	@echo "backtest: not implemented until Phase 7" >&2; exit 1
dashboard:
	@echo "dashboard: not implemented until Phase 8" >&2; exit 1
figures:
	$(PY) -m bpk.viz.rnd_figures --config $(CONFIG)
paper:
	@echo "paper: not implemented until Phase 9" >&2; exit 1

# Runs only what exists today. Extended as phases land.
all: lint test
