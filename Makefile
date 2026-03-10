PYTHON ?= python

.PHONY: help install test lint check-m0

help:
	@echo "Targets:"
	@echo "  install   Install package with dev dependencies"
	@echo "  test      Run unit tests"
	@echo "  lint      Run ruff checks"
	@echo "  check-m0  Run M0 bootstrap checks"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

check-m0:
	$(PYTHON) scripts/check_m0.py
