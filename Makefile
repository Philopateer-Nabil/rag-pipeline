.PHONY: install install-dev install-ui test lint format serve ui ingest evaluate docker-up docker-down clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-ui:
	pip install -e ".[ui]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src/rag_pipeline --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

serve:
	rag serve

ui:
	rag ui

ingest:
	rag ingest data/sample/

evaluate:
	rag evaluate

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf data/faiss_index/ .pytest_cache/ .ruff_cache/ __pycache__/ dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
