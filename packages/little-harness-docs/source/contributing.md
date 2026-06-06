# Contributing

## Setup

```bash
uv sync --all-packages
make pre-commit-install
```

## Quality gates

```bash
make gates    # lint, typecheck, complexity, dead-code, deps, imports, security, semgrep, tests
make check    # gates + mutation (zero survivors)
make docs     # build Sphinx documentation
```

Individual targets run per package:

```bash
make test          # unit tests
make typecheck     # pyright strict per package
make lint          # ruff check
make mutation      # mutation testing (zero survivors)
make integration   # through-core integration tests
make docs          # build documentation
```

## Expectations

- **TDD** — every new function gets a test; bug fixes get a regression test.
- **Mutation** — `make mutation` must report zero surviving mutants.
- **Docstrings** — all public functions use Google-style docstrings.
- **Strong typing** — pyright strict, no unjustified `Any`/ignores/casts.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add async support for provider plugins
fix: handle empty prompt edge case in argument parser
docs: add plugin development guide
```

Breaking changes use `!` or `BREAKING CHANGE:`.

## Architecture convention

Core stays provider-agnostic and vendor-free. Every provider or tool is a
separate distribution that depends on core, implements a port, and registers an
entry point — exposed as an umbrella extra. Adding an integration is a new
package + extra, never a core edit.
