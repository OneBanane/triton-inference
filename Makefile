.PHONY: install run check format lint clean

install:
	uv sync --dev
	uv run pre-commit install

run:
	uv run uvicorn triton_inference.main:app --host 127.0.0.1 --port 8000

check:
	uv run pre-commit run --all-files

format:
	uv run ruff format .
	uv run ruff check . --fix

lint:
	uv run ruff check .

clean:
	rm -rf .ruff_cache .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
