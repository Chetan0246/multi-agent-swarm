.PHONY: install test lint bench chart all

install:
	pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check swarm tests scripts
	mypy swarm

bench:
	python -m swarm bench --seeds 3 --out docs/assets/benchmark.json

chart:
	python scripts/plot_benchmark.py docs/assets/benchmark.json

all: lint test bench chart
