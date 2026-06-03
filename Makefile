.PHONY: format lint typecheck complexity dead-code deps imports security semgrep test mutation check pre-commit-install

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

complexity:
	@output="$$(uv run radon cc local_llm main.py --min B)"; \
	if [ -n "$$output" ]; then \
		printf '%s\n' "$$output"; \
		exit 1; \
	fi

dead-code:
	uv run vulture local_llm main.py tests

deps:
	uv run deptry .

imports:
	uv run lint-imports

security:
	uv run bandit -r local_llm main.py

semgrep:
	uv run semgrep --error --config semgrep-rules .

test:
	uv run pytest

mutation:
	uv run mutmut run

check: lint typecheck complexity dead-code deps imports security semgrep test

pre-commit-install:
	uv run pre-commit install
