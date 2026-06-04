# Contributing

Thanks for your interest in little-harness. This is a [uv](https://docs.astral.sh/uv/)
workspace of one core distribution plus one distribution per integration.

## Setup

Requires `uv` and Python 3.12.

```
uv sync --all-packages     # install every workspace member (editable)
make pre-commit-install     # run the gates automatically on each commit
```

## Quality gates

Every package passes the full gate set on its own; the workspace `make check`
aggregates them:

```
make check   # lint, typecheck (pyright strict), complexity, dead-code, deps,
             # imports, security, semgrep, tests, mutation (zero survivors)
```

`make gates` runs everything except mutation (the fast subset CI runs as its own
job). Individual targets also run per package, e.g. `make test`, `make typecheck`,
`make mutation`. After adding or renaming a plugin, run `uv sync --all-packages`
so its entry points register for discovery.

Expectations:
- **TDD** — every new function gets a test; bug fixes get a regression test.
- **Mutation** — `make mutation` must report zero surviving mutants.
- **Strong typing** — pyright strict, no unjustified `Any`/ignores/casts.

End-to-end tests against a real local model (`make integration`) need a GGUF
model and provider keys, so they run locally and are not part of CI.

## Architecture convention

Core stays provider-agnostic and vendor-free. Every provider or tool is a
separate distribution that depends on core, implements a port, and registers an
entry point — exposed as an umbrella extra. Adding an integration is a new
package + extra, never a core edit. See the [README](README.md) ("Extending it")
and [AGENTS.md](AGENTS.md).

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`,
`fix:`, `refactor:`, `build:`, `ci:`, `docs:`, `test:`), with `!` /
`BREAKING CHANGE:` for breaking changes.
