.PHONY: help install install-dev lint format check test test-unit test-integration test-e2e train-prediction train-synthesis clean

help:
	@echo "Available commands:"
	@echo "  make install          Install dependencies"
	@echo "  make install-dev      Install with dev dependencies"
	@echo "  make lint             Run ruff check with auto-fix"
	@echo "  make format           Format code with ruff"
	@echo "  make check            Run linting and type checking"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-e2e         Run end-to-end tests only"
	@echo "  make train-prediction Train prediction model"
	@echo "  make train-synthesis  Train synthesis model"
	@echo "  make clean            Clean cache and temporary files"

install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run ruff check --fix generative_handwriting

format:
	uv run ruff format generative_handwriting

check:
	uv run ruff check generative_handwriting
	uv run mypy generative_handwriting

test:
	uv run pytest

test-unit:
	uv run pytest -m unit_test

test-integration:
	uv run pytest -m integration_test

test-e2e:
	uv run pytest -m end_to_end_test

train-prediction:
	uv run python generative_handwriting/train_handwriting_prediction.py

train-synthesis:
	uv run python generative_handwriting/train_handwriting_synthesis.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
