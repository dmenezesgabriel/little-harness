# little-harness

A small, fully local LLM agent loop with a **plugin architecture**. The core is a
provider-agnostic reason-act loop (strict-JSON protocol, tool calling, streaming,
lifecycle hooks); chat-model providers and tools are separate, independently
installable distributions discovered at runtime via packaging **entry points**.

Install only what you need — the core imports no vendor SDK until you select a
provider that is installed.

## Install & run

`little-harness` is core-only; pick the provider/tool **extras** you want:

```
uv pip install "little-harness[llama-cpp,calculator]"
little-harness --model models/LFM2-8B-A1B-Q4_K_M.gguf \
  -p "What is 144 divided by 12? Then tell me if the result is even or odd."
```

Extras are `llama-cpp`, `litellm`, `calculator`, and `all`. With exactly one
provider installed, `--provider` is optional; the sole provider is the default.

Pick a different provider by its extra:

```
uv pip install "little-harness[litellm]"
export GEMINI_API_KEY=...   # litellm reads provider keys from the environment
little-harness --provider litellm --model gemini/gemini-2.5-flash -p "Hello!"
```

Composing the distributions by name (`uv pip install little-harness-core
little-harness-litellm`) is equivalent — the extras are just a curated shorthand.

### CLI flags

The core CLI is provider-agnostic. `--model` and the sampling/loop flags are
first-class; any other provider-specific setting is passed as repeatable
`-o KEY=VALUE` and validated by the selected provider plugin.

- `--provider` — the installed provider plugin to use (default `llama_cpp`).
- `-m, --model` — the model to use; provider-specific (a model name for
  `litellm` like `gemini/gemini-2.5-flash`, a GGUF path for `llama_cpp`).
  Shorthand for `-o model=…`.
- `-o, --option KEY=VALUE` — provider-specific setting, repeatable. For
  `llama_cpp`: `model_path`, `n_ctx`, `n_threads`, `n_gpu_layers`. For `litellm`:
  `api_base`, `api_key` (the key is also read from the environment, e.g.
  `GEMINI_API_KEY`/`OPENAI_API_KEY`).
- `--temperature`, `--max-tokens`, `--max-iterations` — sampling and loop bounds.
- `--stream` — stream generated tokens to stdout (the strict-JSON protocol means
  streamed text is JSON).
- `--log` — emit one structured JSON log line per agent event, each carrying a
  `run_id` correlation key and per-call `elapsed_seconds`.

Selecting a provider that is not installed fails with a clear error listing the
installed providers.

## Architecture

A uv **workspace** (monorepo) of one core distribution plus one distribution per
integration — the convention used by LangChain (`langchain-*`) and LlamaIndex
(`llama-index-*`).

```
packages/
  little-harness-core/        # import root: little_harness
  little-harness-llama-cpp/    # provider plugin (entry point: llama_cpp)
  little-harness-calculator/   # tool plugin (entry point: calculator)
  little-harness-litellm/      # provider plugin (entry point: litellm)
  little-harness/              # umbrella meta-distribution (no code; maps extras to plugins)
```

`little-harness-core` follows Clean Architecture — four concentric layers whose
dependencies point only inward, mechanically enforced by `import-linter`:

| Layer | Responsibility |
| --- | --- |
| `domain/` | Pure entities + validated value objects. Zero outward dependencies. |
| `application/` | `AgentRuntime` loop and the ports (`ChatModel`, `AgentTool`, `AgentPolicy`, `AgentObserver`, `TokenSink`, `LifecycleHook`), plus `ToolRegistry` and `HookChain`. |
| `infrastructure/` | Core defaults only: strict-JSON `policy/`, structured-logging `observability/`, `hooks/`. |
| `presentation/` | CLI: argument parsing into a validated `AppConfig` and plain-text rendering. |

`little_harness/plugin_discovery.py` is the single place that loads plugins
dynamically: `entry_point.load()` imports a provider/tool adapter (and its vendor
SDK) only when selected. Each plugin distribution depends on core, implements a
port, and registers an entry point — so core discovers plugins without importing
them. That is Dependency Inversion at the package boundary.

### Extending it — add a distribution, not a branch

**The convention:** core stays provider-agnostic and vendor-free — it names no
provider and imports no SDK. Every provider and tool is a separate distribution
that depends on core, implements a port, and registers an entry point; the
umbrella exposes each as an opt-in extra (`little-harness[<name>]`). Adding an
integration is a new package + extra, never a branch or a core edit.

- **New provider** — a `little-harness-<name>` package implementing `ChatModel`
  (`complete_streaming` + `close`, vendor SDK sealed inside it) with a
  `build(options) -> ChatModel` registered under the
  `little_harness.chat_model_providers` entry-point group. No core change;
  selectable via `--provider <name>`.
- **New tool** — a package implementing `AgentTool` with a `build() -> AgentTool`
  registered under `little_harness.tools`. Discovered into the `ToolRegistry`.
- **Tracing/metrics, live output, interception** — implement `AgentObserver`,
  `TokenSink`, or `LifecycleHook` and wire them in the core composition root.

Key patterns: Ports & Adapters, entry-point plugin discovery, Strategy
(`AgentPolicy`), Visitor (decisions, hook decisions, AST evaluation), Factory
(provider builders), Composite (`HookChain`), Observer, Null Object, and
first-class collections.

## Development

Each package passes the full gate set on its own; the workspace `make check`
aggregates them:

```
make check   # lint, typecheck (pyright strict), complexity, dead-code, deps,
             # imports, security, semgrep, tests, mutation (zero survivors)
```

Individual targets run per package (e.g. `make test`, `make typecheck`,
`make mutation`). End-to-end tests against the real local model are opt-in and
run with `make integration`; they skip automatically when the GGUF model file is
absent. After adding or renaming a plugin, run `uv sync --all-packages` so its
entry points register for discovery.
