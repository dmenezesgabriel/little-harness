# Installation

## Prerequisites

- Python **3.12** (strictly `>=3.12,<3.13`)
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`

## Install from source

```bash
git clone https://github.com/dmenezesgabriel/little-harness
cd little-harness
uv sync --all-packages
```

The CLI is now available:

```bash
uv run little-harness --help
```

## Install from PyPI

```bash
uv pip install "little-harness[all]"
```

Or with only the provider/tool you need:

```bash
uv pip install "little-harness[llama-cpp,calculator]"
```

### Available extras

| Extra | Packages | Entry-point groups |
|-------|----------|--------------------|
| `llama-cpp` | `little-harness-llama-cpp` | `chat_model_providers` |
| `litellm` | `little-harness-litellm` | `chat_model_providers` |
| `calculator` | `little-harness-calculator` | `tools` |
| `file-tools` | `little-harness-file-tools` | `tools` |
| `ripgrep` | `little-harness-ripgrep` | `tools` |
| `ast` | `little-harness-ast` | `tools` |
| `json-policy` | `little-harness-json-policy` | `agent_policies` |
| `logging` | `little-harness-logging` | `observers` |
| `rich` | `little-harness-rich` | `uis` |
| `all` | All of the above | — |

The extras are a curated shorthand. All distributions are independently
installable by name:

```bash
uv pip install little-harness-core little-harness-litellm
```

## Provider keys

Some providers need environment variables:

```bash
export GEMINI_API_KEY="…"
export OPENAI_API_KEY="…"
export ANTHROPIC_API_KEY="…"
```

See `.env.example` in the repository root.
