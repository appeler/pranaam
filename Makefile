.PHONY: help clean install test test-cov lint format type-check docs docs-serve build upload dev-install

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf docs/_build/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

install: ## Install package
	uv pip install -e .

dev-install: ## Install package with development dependencies
	uv sync --dev
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=pranaam --cov-report=html --cov-report=term

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff format .
	ruff check --fix .

type-check: ## Run type checker
	mypy pranaam/

docs: ## Build documentation
	cd docs && make clean && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

build: ## Build package
	uv build

upload: ## Upload to PyPI
	uv publish

ci: lint type-check test ## Run CI checks
