ruff := "uvx ruff@0.12.10"
pre-commit := "uvx pre-commit@4.3.0"

help:
    just -l -u

lint-dry-run:
    {{ ruff }} check
    {{ ruff }} format --check

lint:
    {{ pre-commit }} run -a

watch:
    {{ ruff }} check -w

install-pre-commit:
    {{ pre-commit }} install

test:
    uv run pytest -s
