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
token, and iteration flags. Notable flags:

- `--stream` — stream generated tokens to stdout as they are produced (off by
  default; the strict-JSON protocol means streamed text is JSON).
- `--log` — emit a structured JSON log line per agent event, each carrying a
  `run_id` correlation key and per-call `elapsed_seconds` (off by default).
- `--provider` — select the chat-model provider (default `llama_cpp`).

## Architecture

The code follows Clean Architecture: four concentric layers whose dependencies
point only inward, mechanically enforced by `import-linter` contracts in
`pyproject.toml` (`make imports`). The single composition root
(`local_llm/composition.py`) is the only place that wires concrete adapters to
the application's ports.

| Layer | Package | Responsibility |
| --- | --- | --- |
| `domain/` | entities + value objects | Pure types: `ChatMessage`, `AgentDecision` (polymorphic), `Number`, and validated value objects. Zero outward dependencies. |
| `application/` | use cases + ports | `AgentRuntime` loop, the `ChatModel` (streaming + `Closeable`) / `AgentTool` / `AgentPolicy` / `AgentObserver` / `TokenSink` / `StructuredLogger` ports, and `ToolRegistry`. |
| `infrastructure/` | adapters | `llama_cpp/` model adapter, `tools/calculator/` (AST visitor), `policy/` (strict-JSON), `observability/` (structured logging), `providers/` (model factory). |
| `presentation/` | CLI delivery | Argument parsing into a validated `AppConfig` and plain-text result rendering. |

Dependencies flow `presentation`/`infrastructure` → `application` → `domain`;
`presentation` and `infrastructure` never reference each other.

**Extending it is closed for modification:**
- a **new tool** — implement `AgentTool`, register it in `composition.py`;
- a **new LLM provider** — add an `infrastructure/<provider>/` adapter implementing
  `ChatModel` (`complete_streaming` + `close`, vendor SDK kept inside that package),
  add its config fields, and register one builder in `composition.CHAT_MODEL_BUILDERS`.
  No application/domain change; selectable via `--provider`;
- **tracing/metrics** — implement `AgentObserver` and pass it to `build_application`
  (the loop emits events carrying `run_id` + `elapsed`; `NullObserver` is the default,
  `StructuredLoggingObserver` logs JSON);
- **live output** — implement `TokenSink` (`NullTokenSink` is the default,
  `StdoutTokenSink` streams to the terminal under `--stream`).

Key patterns: Ports & Adapters, Strategy (`AgentPolicy`), Visitor
(`DecisionVisitor`, AST node evaluators), Factory (provider selection), Observer
(`AgentObserver`), Null Object (`NullObserver`, `NullTokenSink`), and first-class
collections (`ToolRegistry`, `MessageHistory`, `AgentSteps`).

## Development

All quality gates run through a single command:

```
make check   # lint, typecheck, complexity, dead-code, deps, imports, security, semgrep, test
```

Individual targets are available too (`make test`, `make typecheck`,
`make imports`, `make mutation`, …). See the `Makefile`.

End-to-end tests against the real local model are opt-in (deselected from
`make check`) and run with `make integration`; they skip automatically when the
GGUF model file is absent.
