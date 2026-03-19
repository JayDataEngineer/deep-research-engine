.PHONY: all format lint type test test_watch install help

# Default target executed when no arguments are given to make.
all: help

.EXPORT_ALL_VARIABLES:
UV_FROZEN = true

######################
# INSTALLATION
######################

install: ## Install dependencies
	uv sync --frozen --all-extras --group dev

install-dev: ## Install development dependencies
	uv sync --group dev

######################
# TESTING AND COVERAGE
######################

TEST ?= tests/
TEST_FILE ?= tests/

test: ## Run unit tests
	uv run pytest $(TEST)

test-file: ## Run tests in a specific file
	uv run pytest $(TEST_FILE) -v

test-watch: ## Run tests in watch mode
	uv run ptw --now . -- -vv $(TEST)

coverage: ## Run tests with coverage report
	uv run pytest --cov=src/deep_research_engine \
		--cov-config=pyproject.toml \
		--cov-report=term-missing:skip-covered \
		--cov-report=xml \
		$(TEST)

######################
# LINTING AND FORMATTING
######################

PYTHON_FILES ?= .

format: ## Run code formatters
	uv run ruff format $(PYTHON_FILES)
	uv run ruff check --fix $(PYTHON_FILES)

format-check: ## Check code formatting (no changes)
	uv run ruff format $(PYTHON_FILES) --check

lint: ## Run linters
	uv run ruff check $(PYTHON_FILES)

lint-fix: ## Run linters and fix issues
	uv run ruff check --fix $(PYTHON_FILES)

type: ## Run type checking
	uv run mypy src/deep_research_engine

type-check: type

check: lint type test ## Run all checks (lint, type, test)

######################
# CLEANUP
######################

clean: ## Clean build artifacts and cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/**/__pycache__
	rm -rf tests/**/__pycache__
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

######################
# HELP
######################

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
