.PHONY: setup test lint eval eval-live demo serve

setup:
	uv venv --python 3.11 .venv && . .venv/bin/activate && uv pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src

eval:
	.venv/bin/career-agent eval --fault-rate 0.25

eval-live:
	.venv/bin/career-agent eval --mode live --store delta --trace

demo:
	.venv/bin/career-agent analyze --sample helix --fault-rate 0.5

serve:
	.venv/bin/career-agent serve
