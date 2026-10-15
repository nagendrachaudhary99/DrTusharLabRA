# Convenience targets; every command is also documented in README.md.
PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: venv test lint typecheck check demo docker docker-demo fixtures clean

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

typecheck:
	$(BIN)/mypy

check: lint typecheck test

demo:
	$(BIN)/itrialspace-eval analyze tests/fixtures/lesions_sample.csv \
	  --patients tests/fixtures/patients_sample.csv --by diameterMm --by arm \
	  --bootstrap 2000 --seed 42 --out report.json
	$(BIN)/itrialspace-eval agreement tests/fixtures/synthetic_models.csv tests/fixtures/real_models.csv

fixtures:
	$(BIN)/python tests/fixtures/make_fixtures.py

docker:
	docker build -t itrialspace-eval .

docker-demo: docker
	docker run --rm -v "$(CURDIR)/tests/fixtures:/data:ro" itrialspace-eval analyze /data/lesions_sample.csv \
	  --patients /data/patients_sample.csv --by diameterMm --by arm --seed 42

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache report.json
