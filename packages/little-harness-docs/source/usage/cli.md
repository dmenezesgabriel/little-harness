# CLI reference

## Synopsis

```bash
little-harness [OPTIONS]
```

Exit codes: `0` on success, `1` on error.

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-p, --prompt` | `str` | `None` | Prompt to send to the model; omit for interactive REPL |
| `--provider` | `str` | auto-detected | Provider plugin name; defaults to the sole installed provider |
| `--policy` | `str` | auto-detected | Agent policy plugin name; defaults to the sole installed policy |
| `-m, --model` | `str` | `None` | Model name or path; shorthand for `-o model=…` |
| `-o, --option` | `KEY=VALUE` | `[]` | Provider-specific setting, repeatable |
| `--temperature` | `float` | `0.0` | Sampling temperature (0.0..2.0, 0.0=greedy) |
| `--top-p` | `float` | `None` | Nucleus sampling threshold (0.0..1.0); provider default when unset |
| `--repeat-penalty` | `float` | `None` | Repetition penalty (0.0..2.0, 1.0=off); provider default when unset |
| `--max-tokens` | `int` | `512` | Maximum generated tokens per step |
| `--max-iterations` | `int` | `5` | Maximum agent loop iterations |
| `--stream` | flag | `False` | Stream tokens to stdout |
| `--log` | flag | `False` | Shorthand for `--observer logging` |
| `--observer` | `str` | `None` | Observer plugin name (e.g. `logging`) |
| `--tools` | `str` | all installed | Comma-separated tool names to enable |
| `--ui` | `str` | `"default"` | Interactive UI plugin to use (e.g. `rich`, `default`) |
| `--yes` | flag | `False` | Auto-approve every sensitive tool without prompting |
| `-s, --session` | `str` | `None` | Resume or fork a past session by ID |

### Provider-specific options (`-o`)

Each provider accepts its own set of `-o` / `--option` keys, passed as
`KEY=VALUE` pairs. The `--model` / `-m` flag is shorthand for `-o model=…`.

#### llama_cpp

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `models/LFM2.5-8B-A1B-Q4_K_M.gguf` | Path to the GGUF model file; also set via `--model` |
| `n_ctx` | `8192` | Context window size |
| `n_threads` | `8` | Number of CPU threads |
| `n_gpu_layers` | `0` | GPU layers to offload (0 = CPU only) |
| `n_batch` | `512` | Batch size for prompt processing |
| `flash_attn` | `true` | Enable flash attention (`true`/`false`) |
| `seed` | `42` | Random seed for reproducibility |

```bash
little-harness --provider llama_cpp \
  -m models/my-model.gguf \
  -o n_ctx=4096 -o n_gpu_layers=35
```

#### litellm

| Key | Default | Description |
|-----|---------|-------------|
| `model` | _required_ | LiteLLM model string (e.g. `gemini/gemini-2.5-flash`, `gpt-4o`); also set via `--model` |
| `api_base` | `None` | Custom API base URL (for proxies or self-hosted) |
| `api_key` | `None` | API key (or use `GEMINI_API_KEY` / `OPENAI_API_KEY` env vars) |
| `num_retries` | `0` | Number of retries on transient failures |

```bash
export GEMINI_API_KEY="…"

little-harness --provider litellm \
  -m gemini/gemini-2.5-flash \
  -o api_base=https://my-proxy/v1
```

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
| `/history`    | Show conversation history                |
| `/skill`      | List loaded skills                       |
| `/skill reload` | Re-read skills from disk               |

### UI Plugins

By default, the REPL uses a standard terminal plain text interface (`--ui default`). If you install the `rich` plugin (`pip install "little-harness[rich]"`), you can run with a beautiful, enhanced Terminal User Interface (TUI) by specifying `--ui rich`:

```bash
little-harness --ui rich
```

The Rich UI features:
* **Welcome Panel**: A styled welcome message on session start.
* **Formatted Panels**: Assistant output is wrapped in clean panels.
* **Markdown Rendering**: Responses and history are rendered on the fly as formatted markdown.
* **Thinking Spinner**: Shows a visual loading status spinner while the agent or tools are executing.
* **Interactive Tool Approvals**: Seamlessly prompts the operator to approve sensitive tool calls using a styled confirmation prompt.
* **Color-coded History**: `/history` shows user messages in green, assistant in blue, system in magenta.
* **Keyboard Interrupt Handling**: Clean exit with `Ctrl+C` or `Ctrl+D`.

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

### Resume a session

```bash
little-harness --provider llama_cpp --model models/LFM2.5-8B-A1B-Q4_K_M.gguf \
  --session abc12345-...
```
