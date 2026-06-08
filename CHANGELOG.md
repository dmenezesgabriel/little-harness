# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Filesystem & shell tools** (`little-harness[file-tools]`): `read_file`,
  `write_file`, `edit_file`, and `bash` — pure-stdlib plugins. `bash` runs behind
  an injectable `DangerousCommandGuardrail` (blocks `rm -rf`, fork bombs, `dd` to
  a device, `mkfs`, `shutdown`, …) with a timeout.
- **Ripgrep search tool** (`little-harness[ripgrep]`): a pure-Python search plugin offering a ripgrep-style query interface without system binary dependencies.
- **Tree-sitter AST tools** (`little-harness[ast]`): `ast_grep` (structural
  search) and `ast_edit` (structure-aware replace of the unique `@match` node),
  built on `py-tree-sitter` (python + javascript out of the box).
- **`--tools`** flag to enable a subset of installed tools (comma-separated;
  defaults to all); unknown names fail loudly with the installed list.
- **Human-in-the-loop permissions**: a `ToolSpec.requires_approval` declaration,
  an `ApprovalHook` at the pre-tool-use seam, and a `PermissionRequester` port
  (interactive prompt with a TTY; auto-approve for `--yes`/non-interactive runs).
- Per-tool **integration tests** that drive each tool through the real agent
  core with a scripted, LLM-free provider (run via `make integration`).
- Opt-in provider/tool **extras** on the umbrella: `little-harness[llama-cpp]`,
  `[litellm]`, `[calculator]`, `[file-tools]`, `[ripgrep]`, `[ast]`, `[all]`.
- PEP 561 `py.typed` markers so installed packages ship their types.
- Root MIT `LICENSE`; `license`, `authors`, `readme`, and project URLs on every
  distribution; a README for core and each plugin.
- GitHub Actions CI (`gates` + `mutation` jobs), Dependabot config, and
  `.env.example` documenting provider keys.

### Changed
- Replaced the external `rg` binary dependency in the `ripgrep` search tool with a pure-Python grep search implementation.
- Hardened the core-vs-extension boundary by specifying precise signatures on `NullHook` and `NullObserver` instead of variadic arguments.
- Adopted the `src/` layout across all workspace packages.
- The CLI default provider now resolves to the sole installed provider via
  discovery instead of a hardcoded `llama_cpp`; the default prompt is
  provider-agnostic.

### Breaking
- Bare `little-harness` installs the core CLI only — no provider. Install a
  provider via an extra (e.g. `little-harness[llama-cpp]`). With several
  providers installed, `--provider` is required.
