# Every target uses the config selected by CONFIG (default: configs/default.yaml).
# Targets for later phases fail loudly rather than pretend to work.
PY ?= python
CONFIG ?= configs/default.yaml

.PHONY: setup data rnd study study-fast check refresh-data figures paper dashboard test lint all

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
study:  ## the real-data study of Amendment 3 (about 25 minutes), then refresh the README
	$(PY) -m bpk.study.run
	$(PY) -m bpk.study.render
study-fast:  ## a thinned run for a quick check; writes results/real_fast only
	$(PY) -m bpk.study.run --fast
check:  ## confirm the README matches results/real/metrics.json
	$(PY) -m bpk.study.render --check
refresh-data:  ## download the free series again and overwrite data/real
	$(PY) -m bpk.data.real
dashboard:
	@echo "dashboard: not implemented until Phase 8" >&2; exit 1
figures:
	$(PY) -m bpk.viz.rnd_figures --config $(CONFIG)
paper:
	@echo "paper: not implemented until Phase 9" >&2; exit 1

all: lint test check
