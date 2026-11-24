PYTHON ?= python

.PHONY: install install-dev lint format test eval clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,serve]"

lint:
	ruff check src tests

format:
	ruff format src tests

test:
	$(PYTHON) -m pytest -v

eval:
	embedtune-eval --out results --threads 4

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
