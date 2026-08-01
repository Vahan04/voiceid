.PHONY: install format lint type test check

install:
	pip install -e ".[dev]"

format:
	ruff check --fix .
	black .

lint:
	ruff check .

type:
	mypy src

test:
	pytest -q --cov=voiceid --cov-report=term-missing

check: lint type test