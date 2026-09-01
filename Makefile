.PHONY: install lint test simulate report evaluate render demo api clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests && ruff format --check src tests

test:
	pytest --cov=retail_vision --cov-report=term-missing

simulate:
	rva simulate --frames 600

report:
	rva report data/events.jsonl

evaluate:
	rva evaluate --frames 600

render:
	rva render --frames 600 --every 0 --out-dir data/frames

demo: simulate report evaluate

api:
	rva serve-api --db-path data/events.duckdb

clean:
	rm -rf data/events.jsonl data/frames .pytest_cache .ruff_cache
