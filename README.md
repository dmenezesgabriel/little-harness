# Local LLM

A small, fully local LLM agent loop running a GGUF model through
[`llama.cpp`](https://github.com/ggerganov/llama.cpp). The model reasons in a
strict-JSON protocol, can call tools (e.g. a safe arithmetic calculator), and
produces a final answer — all on CPU, no network.

## Run

```
uv run python main.py --threads 4 -p "What is 144 divided by 12? Then tell me if the result is even or odd."
```

The default `--model-path` is `models/LFM2-8B-A1B-Q4_K_M.gguf`. See
`uv run python main.py --help` for context size, GPU layers, temperature,
token, and iteration flags.

## Architecture

The code follows Clean Architecture: four concentric layers whose dependencies
point only inward, mechanically enforced by `import-linter` contracts in
`pyproject.toml` (`make imports`). The single composition root
(`local_llm/composition.py`) is the only place that wires concrete adapters to
the application's ports.

| Layer | Package | Responsibility |
| --- | --- | --- |
| `domain/` | entities + value objects | Pure types: `ChatMessage`, `AgentDecision` (polymorphic), `Number`, and validated value objects. Zero outward dependencies. |
| `application/` | use cases + ports | `AgentRuntime` loop, the `ChatModel` / `AgentTool` / `AgentPolicy` / `AgentObserver` / `StructuredLogger` ports, and `ToolRegistry`. |
| `infrastructure/` | adapters | `llama_cpp/` model adapter, `tools/calculator/` (AST visitor), `policy/` (strict-JSON), `observability/` (structured logging), `providers/` (model factory). |
| `presentation/` | CLI delivery | Argument parsing into a validated `AppConfig` and plain-text result rendering. |

Dependencies flow `presentation`/`infrastructure` → `application` → `domain`;
`presentation` and `infrastructure` never reference each other.

**Extending it is closed for modification:**
- a **new tool** — implement `AgentTool`, register it in `composition.py`;
- a **new LLM provider** — add a builder to `infrastructure/providers/chat_model_factory.py`;
- **tracing/metrics** — implement `AgentObserver` and pass it to `build_application`
  (the loop emits events; `NullObserver` is the default, `StructuredLoggingObserver`
  logs JSON).

Key patterns: Ports & Adapters, Strategy (`AgentPolicy`), Visitor
(`DecisionVisitor`, AST node evaluators), Factory (provider selection), Observer
(`AgentObserver`), Null Object (`NullObserver`), and first-class collections
(`ToolRegistry`, `MessageHistory`, `AgentSteps`).

## Development

All quality gates run through a single command:

```
make check   # lint, typecheck, complexity, dead-code, deps, imports, security, semgrep, test
```

Individual targets are available too (`make test`, `make typecheck`,
`make imports`, `make mutation`, …). See the `Makefile`.
