.PHONY: format lint typecheck complexity dead-code deps imports security semgrep test mutation check

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	uv run pyright

complexity:
	@output="$$(uv run radon cc src --min B)"; \
	if [ -n "$$output" ]; then printf '%s\n' "$$output"; exit 1; fi

dead-code:
	uv run vulture

deps:
	uv run deptry .

imports:
	uv run lint-imports

security:
	uv run bandit -qr src -c pyproject.toml

semgrep:
	uv run semgrep --error --config ../../semgrep-rules .

test:
	uv run pytest

mutation:
	uv run mutmut run
	@survivors="$$(uv run mutmut results | grep -E ': (survived|no tests)' || true)"; \
	if [ -n "$$survivors" ]; then \
		printf 'Surviving mutants (add tests to kill them):\n%s\n' "$$survivors"; \
		exit 1; \
	fi; \
	echo "No surviving mutants."

check: lint typecheck complexity dead-code deps imports security semgrep test mutation
