# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Opt-in provider/tool **extras** on the umbrella: `little-harness[llama-cpp]`,
  `[litellm]`, `[calculator]`, `[all]`.
- PEP 561 `py.typed` markers so installed packages ship their types.
- Root MIT `LICENSE`; `license`, `authors`, `readme`, and project URLs on every
  distribution; a README for core and each plugin.
- GitHub Actions CI (`gates` + `mutation` jobs), Dependabot config, and
  `.env.example` documenting provider keys.

### Changed
- Adopted the `src/` layout across all workspace packages.
- The CLI default provider now resolves to the sole installed provider via
  discovery instead of a hardcoded `llama_cpp`; the default prompt is
  provider-agnostic.

### Breaking
- Bare `little-harness` installs the core CLI only — no provider. Install a
  provider via an extra (e.g. `little-harness[llama-cpp]`). With several
  providers installed, `--provider` is required.
