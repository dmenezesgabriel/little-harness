# CLI reference

## Synopsis

```bash
little-harness [OPTIONS]
```

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-p, --prompt` | `str` | `None` | Prompt to send to the model; omit for interactive REPL |
| `--provider` | `str` | auto-detected | Provider plugin name; defaults to the sole installed provider |
| `--policy` | `str` | auto-detected | Agent policy plugin name; defaults to the sole installed policy |
| `-m, --model` | `str` | `None` | Model name or path; shorthand for `-o model=…` |
| `-o, --option` | `KEY=VALUE` | `[]` | Provider-specific setting, repeatable |
| `--temperature` | `float` | `0.0` | Sampling temperature (0.0..2.0) |
| `--top-p` | `float` | `None` | Nucleus sampling threshold (0.0..1.0); provider default when unset |
| `--repeat-penalty` | `float` | `None` | Repetition penalty (0.0..2.0, 1.0=off); provider default when unset |
| `--max-tokens` | `int` | `512` | Maximum generated tokens per step |
| `--max-iterations` | `int` | `5` | Maximum agent loop iterations |
| `--stream` | flag | `False` | Stream tokens to stdout |
| `--log` | flag | `False` | Shorthand for `--observer logging` |
| `--observer` | `str` | `None` | Observer plugin name (e.g. `logging`) |
| `--tools` | `str` | all installed | Comma-separated tool names to enable |
| `--yes` | flag | `False` | Auto-approve every sensitive tool without prompting |

## Interactive REPL

When `-p` / `--prompt` is omitted, `little-harness` starts an interactive
read-eval-print loop:

```bash
little-harness --provider llama_cpp --model models/LFM2.5-8B-A1B-Q4_K_M.gguf
```

Type prompts directly and see responses. Slash commands are available:

| Command       | Description                              |
|---------------|------------------------------------------|
| `/exit`       | Exit the REPL                            |
| `/quit`       | Exit the REPL (alias)                    |
| `/clear`      | Clear conversation history               |
| `/help`       | Show available commands                  |
| `/history`    | Show the number of turns in this session |

## Examples

### Local GGUF model

```bash
little-harness --provider llama_cpp \
  --model models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  -o n_ctx=4096 -o n_gpu_layers=35 \
  -p "Write a haiku about Linux."
```

### Remote provider via LiteLLM

```bash
export GEMINI_API_KEY="…"

little-harness --provider litellm \
  --model gemini/gemini-2.5-flash \
  --temperature 0.7 --stream --log \
  -p "Explain the CAP theorem in one paragraph."
```

### Tool selection

```bash
little-harness --provider litellm --model gemini/gemini-2.5-flash \
  --tools calculator,ripgrep \
  -p "Search for FIXME comments and count them."
```

### Auto-approve sensitive tools

```bash
little-harness --provider litellm --model gemini/gemini-2.5-flash \
  --tools all --yes \
  -p "List the current directory contents."
```
