.PHONY: format lint typecheck complexity dead-code deps imports security semgrep \
        test integration mutation gates check sync pre-commit-install

# Code-bearing workspace members (the umbrella `little-harness` ships no code).
CODE_PACKAGES := little-harness-core little-harness-llama-cpp \
                 little-harness-calculator little-harness-litellm

sync:
	uv sync --all-packages

format:
	uv run ruff format .

lint:
	uv run ruff check .

typecheck:
	@for pkg in $(CODE_PACKAGES); do \
		echo "pyright: $$pkg"; \
		uv run --directory packages/$$pkg pyright || exit 1; \
	done

complexity:
	@output="$$(uv run radon cc packages/*/little_harness packages/*/little_harness_* --min B)"; \
	if [ -n "$$output" ]; then printf '%s\n' "$$output"; exit 1; fi

dead-code:
	@for pkg in $(CODE_PACKAGES); do \
		echo "vulture: $$pkg"; \
		uv run --directory packages/$$pkg vulture || exit 1; \
	done

deps:
	@for pkg in $(CODE_PACKAGES); do \
		echo "deptry: $$pkg"; \
		uv run --directory packages/$$pkg deptry . || exit 1; \
	done

imports:
	@for pkg in $(CODE_PACKAGES); do \
		echo "import-linter: $$pkg"; \
		uv run --directory packages/$$pkg lint-imports || exit 1; \
	done

security:
	uv run bandit -qr packages -x '*/tests/*'

semgrep:
	uv run semgrep --error --config semgrep-rules packages

test:
	@for pkg in $(CODE_PACKAGES); do \
		echo "pytest: $$pkg"; \
		uv run --directory packages/$$pkg pytest || exit 1; \
	done

integration:
	uv run --directory packages/little-harness-llama-cpp pytest -m integration --no-cov

mutation:
	@for pkg in $(CODE_PACKAGES); do \
		echo "mutmut: $$pkg"; \
		( cd packages/$$pkg && uv run mutmut run ); \
		survivors="$$(cd packages/$$pkg && uv run mutmut results | grep -E ': (survived|no tests)' || true)"; \
		if [ -n "$$survivors" ]; then \
			printf 'Surviving mutants in %s (add tests to kill them):\n%s\n' "$$pkg" "$$survivors"; \
			exit 1; \
		fi; \
	done; \
	echo "No surviving mutants."

# Every gate except mutation; the fast set CI runs as its own job.
gates: lint typecheck complexity dead-code deps imports security semgrep test

check: gates mutation

pre-commit-install:
	uv run pre-commit install
