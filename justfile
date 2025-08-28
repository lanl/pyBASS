ruff := "uvx ruff@0.12.10"
pre-commit := "uv run pre-commit"

help:
    just -l -u

# Run linter without changing code
lint-dry-run:
    {{ ruff }} check
    {{ ruff }} format --check

# Run linter
lint:
    {{ pre-commit }} run -a

# Start lint server to identify errors without change code
watch:
    {{ ruff }} check -w

# Install git hooks to lint and format prior to `git commits`
install-pre-commit:
    {{ pre-commit }} install

# Run tests in current environment
test:
    uv run pytest -s

# Run tests with latest dependencies and highest python version
test-highest:
    uv run -p 3.14 --isolated --resolution=highest pytest -s

# Run tests with oldest supported dependencies and smallest python version
test-lowest:
    uv run -p 3.9 --isolated --resolution=lowest-direct pytest -s

test-all: test-highest test test-lowest
