.PHONY: format lint typecheck complexity dead-code deps imports security semgrep \
        test integration mutation gates check sync pre-commit-install \
        models e2e e2e-local e2e-remote e2e-repeat \
        docs docs-serve

# Code-bearing workspace members (the umbrella `little-harness` ships no code).
CODE_PACKAGES := little-harness-core little-harness-llama-cpp \
                 little-harness-calculator little-harness-litellm \
                 little-harness-file-tools little-harness-ripgrep \
                 little-harness-ast little-harness-json-policy \
                 little-harness-logging little-harness-rich \
                 little-harness-session-jsonl

# Members with deterministic through-core integration tests (marked `integration`).
# Real-provider/model tests stay under umbrella e2e targets so CI is reproducible.
INTEGRATION_PACKAGES := little-harness-calculator little-harness-file-tools \
                        little-harness-ripgrep little-harness-ast \
                        little-harness-json-policy little-harness-logging

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
	@for pkg in $(CODE_PACKAGES); do \
		echo "bandit: $$pkg"; \
		uv run --directory packages/$$pkg bandit -qr src -c pyproject.toml || exit 1; \
	done

semgrep:
	uv run semgrep --error --config semgrep-rules packages

test:
	@for pkg in $(CODE_PACKAGES); do \
		echo "pytest: $$pkg"; \
		uv run --directory packages/$$pkg pytest || exit 1; \
	done

integration:
	@for pkg in $(INTEGRATION_PACKAGES); do \
		echo "integration: $$pkg"; \
		uv run --directory packages/$$pkg pytest -m integration --no-cov || exit 1; \
	done

# Download the GGUF models the local e2e suite drives (Q4_K_M only). No global
# install needed: uvx runs the Hugging Face CLI in a throwaway environment.
models:
	uvx --from huggingface_hub hf download LiquidAI/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-Q4_K_M.gguf --local-dir models
	uvx --from huggingface_hub hf download LiquidAI/LFM2.5-350M-GGUF LFM2.5-350M-Q4_K_M.gguf --local-dir models

# Real-provider, cross-package smoke tests in the umbrella package. Opt-in only:
# never part of `check`. Each scenario skips cleanly when its model/key is absent.
# `e2e-local` needs a GGUF (see `make models`); `e2e-remote` needs GEMINI_API_KEY,
# which the remote targets load from a gitignored root `.env` so the key never
# enters source and `make e2e` stays reproducible.
e2e:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	uv run --directory packages/little-harness pytest -m "local_model or network" --no-cov --timeout=120 -s

e2e-local:
	uv run --directory packages/little-harness pytest -m local_model --no-cov --timeout=120 -s

e2e-remote:
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	uv run --directory packages/little-harness pytest -m network --no-cov --timeout=120 -s

# Run the local e2e suite N times (default 3) to surface flakiness.
# Usage: make e2e-repeat [N=5]
e2e-repeat:
	@passed=0; n=$(or $(N),3); \
	for i in $$(seq $$n); do \
		printf '=== Run %d/%d ===\n' "$$i" "$$n"; \
		$(MAKE) --no-print-directory e2e-local && passed=$$((passed+1)) || true; \
	done; \
	printf '\nPass rate: %d/%d\n' "$$passed" "$$n"; \
	[ "$$passed" -eq "$$n" ]

DOCS_PACKAGE := little-harness-docs

docs:
	uv run --directory packages/$(DOCS_PACKAGE) sphinx-build -b html source build

docs-serve:
	uv run --directory packages/$(DOCS_PACKAGE) sphinx-autobuild -b html source build

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
