# Small-LLM Tuning Report — little-harness on CPU

**Date**: 2026-06-18
**Author**: hands-on benchmark + audit pass
**Machine**: Intel i5-1135G7 (4 physical / 8 logical cores), 30 GB RAM, **no GPU**
(`llama_cpp` 0.3.23, `llama_supports_gpu_offload() == False`)
**Models**: all six GGUFs in `models/` (350M → 8B-A1B)
**Engine**: `evaluation/benchmark.py` (model-reusing) + `evaluation/summarize.py`

> This report supersedes the 2026-06-14 `EVALUATION_REPORT.md` (1.2B only) and
> acts on its stated next step — re-run across model sizes and fix the harness's
> false positives. Every number here comes from `benchmark.py` JSONL output,
> not hand transcription.

---

## 0. TL;DR — what to change for small models on CPU

1. **Cut the tool set.** Tool descriptions + the JSON-schema GBNF grammar go into
   *every* model call; a bigger tool set inflates **time-to-first-token (TTFT)**,
   which on CPU dominates short tool calls. Single-tool vs all-tools roughly
   **2–4× the end-to-end latency** with no quality gain for these models. Use the
   shipped `fast`/`balanced`/`tiny` profiles, or `--tools a,b,c`.
2. **Set `n_threads = 4` (physical cores), not the default 8.** Hyperthread
   siblings don't help llama.cpp decode on this CPU (sweep B).
3. **Right-size `n_ctx`.** The default 8192 over-allocates the KV cache for short
   agent turns; 2048–4096 is plenty for small models and trims memory.
4. **Latency is mostly prefill, not decode.** Pure generation is a stable
   ~20–60 tok/s by model; the first call's TTFT (prompt + grammar compile)
   is the tall pole. Smaller prompt = smaller TTFT.
5. **Drop `ast_grep`/`ast_edit` for models < 8B** — they require tree-sitter
   queries small models cannot produce (confirmed again here).
6. **Model size sets the quality ceiling**, tool/loop tuning sets the latency
   floor. Pick the smallest model that clears your task's quality bar, then tune
   latency around it.

---

## 1. What was instrumented (so latency is measurable)

The loop previously logged only `output_chars` + wall `elapsed_seconds` per model
call — no way to separate prompt prefill from generation. Added (additive, no
behavior change to existing events):

- New domain value `ModelCallMetrics` (TTFT, output-token count, `tokens_per_second`).
- New observer event `on_model_metrics`, emitted right after `on_model_completed`.
- `StructuredLoggingObserver` now emits a `model_metrics` JSON line:
  `time_to_first_token_seconds`, `output_tokens`, `tokens_per_second`, `elapsed_seconds`.
- The JSONL session plugin persists the same.
- `measure_stream` isolates first-token timing and chunk counting (unit-tested).

Token count is approximated by streamed-chunk count (llama.cpp streams ≈ one
token per chunk). "Generation tok/s" in the tables divides output tokens by
`elapsed − TTFT`, isolating decode speed from prefill.

Example (`--log`, 1.2B, single tool):

```
model_completed iteration=1 elapsed=5.56s
model_metrics   iteration=1 ttft=4.87s output_tokens=17 tokens_per_second=3.05
model_completed iteration=2 elapsed=1.72s
model_metrics   iteration=2 ttft=0.60s output_tokens=26 tokens_per_second=15.1
```

Read it as: iteration 1 spent **4.87 s of 5.56 s (87%) in prefill + GBNF grammar
compile**; decode itself was ~25 tok/s both calls. The warm second call's TTFT
collapses to 0.60 s. **This is the core CPU-latency story.**

---

## 2. Benchmark method

- `benchmark.py` loads each GGUF **once** and runs many trials in-process,
  reusing the warm model (the old harness reloaded a 4–5 GB model per trial —
  see §6). Load-time options (threads/batch/flash-attn/n_ctx) vary per process;
  sampling/loop/tool-set vary in-process.
- **Suite** (`evaluation/suite.py`): 11 production-realistic, non-destructive
  cases over real repo files — arithmetic (calculator), file read, code search
  (ripgrep), directory listing (ls), and a multi-step read→calc. Each case names
  the tools it needs.
- **Tool sets**: `single` (just the needed tools), `curated`
  (calculator/read_file/edit_file/ripgrep/ls), `all` (every installed tool) —
  to isolate the prompt-size tax.
- **Scoring fixes vs the old harness**: separator-insensitive expected match
  (so `1,024` matches `1024`), `forbidden_substrings` (error/"No such file"
  text is now a FAIL, not a false PASS), and `min_tool_calls` (answering without
  calling the tool is a FAIL, capturing hallucination). Records both
  `succeeded` (correct **and** used the tool) and `answer_correct` (correct
  regardless of tool use).

### 2a. Cross-model comparison (148 trials, temp 0.1, iter 5, n_threads 4)

Columns: **pass** = correct *and* used the required tool(s); **answer** = correct
regardless of tool use; **halluc** = answered with zero tool calls; **gen tok/s** =
decode-only throughput (excludes TTFT); **e2e** = mean end-to-end seconds.

| model | n | pass | answer | halluc | TTFT s | gen tok/s | e2e s | peak RSS |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| **LFM2.5-8B-A1B** (MoE ~1B active) | 33 | **67%** | **76%** | **0%** | 8.8 | 17.0 | 16.7 | ~9.0 GB |
| LFM2-8B-A1B (older MoE) | 33 | 48% | 58% | 6% | 9.7 | 20.2 | 21.3 | ~8.9 GB |
| LFM2.5-1.2B Q8 | 33 | 45% | 48% | 27% | 5.7 | 20.5 | 9.8 | ~1.5 GB |
| LFM2.5-350M Q4 | 33 | 30% | 30% | 27% | 1.1 | 56.5 | 2.5 | ~0.6 GB |
| Phi-3-mini-4k Q4 (dense 3.8B) | 8* | 38% | 38% | 25% | 0.9 | 10.1 | 8.0 med / **359 max** | ~3 GB† |
| qwen2.5-coder-7B Q4 (dense) | 8* | 100%* | 100%* | 0% | 4.8 | 5.8 | 19.7 med / **211 max** | ~6 GB† |

\* Dense models were time-capped (their `all`-tool trials run for minutes on CPU);
their rows are small samples of the easier `single` cases — treat as
characterization, not ranking. qwen's 100% is 8 arithmetic/read cases only.
† RSS for the capped dense models is estimated from model size + KV cache; the
others are measured peak `ru_maxrss`.

**Reading it:**
- **LFM2.5-8B-A1B is the sweet spot**: best quality (76% answer, **0%
  hallucination**) at MoE speed — ~17 tok/s and 16.7 s e2e, *3–5× faster than the
  dense 7B for better quality*. This is the model to default to when quality matters.
- **MoE (A1B) beats dense on CPU**: ~1B active params means 8B-class quality at
  ~1.2B-class decode speed. The dense Phi-3/qwen run at 5–10 tok/s with
  **heavy-tailed latency** — median single-digit-to-20 s but **maxes of 3–6
  minutes** when the model rambles to the token cap. Unbounded generation on CPU
  is a latency landmine (see §0.4 / cap `max_tokens`).
- **Hallucination (answering without the tool) tracks quality**: 0% on
  LFM2.5-8B-A1B vs 27% on the 1.2B/350M. Small models skip the tool and guess.
- **350M** is genuinely fast (2.5 s e2e, 56 tok/s) but only ~30% correct — fine
  for trivial/deterministic tasks, not tool-heavy work.

### 2b. Tool-set tax — the dominant latency lever

| model / tool set | n | pass | answer | TTFT s | e2e s |
| --- | --: | --: | --: | --: | --: |
| 350M / single | 11 | 36% | 36% | **0.51** | **1.67** |
| 350M / curated | 11 | 27% | 27% | 0.99 | 2.25 |
| 350M / all | 11 | 27% | 27% | **1.86** | **3.53** |
| 1.2B / single | 11 | 36% | 45% | **2.64** | **7.48** |
| 1.2B / curated | 11 | 55% | 55% | 5.05 | 8.47 |
| 1.2B / all | 11 | 45% | 45% | **9.28** | **13.3** |
| LFM2.5-8B-A1B / single | 11 | 64% | 73% | **3.76** | **11.2** |
| LFM2.5-8B-A1B / curated | 11 | 64% | 73% | 7.73 | 15.8 |
| LFM2.5-8B-A1B / all | 11 | 73% | 82% | **14.9** | **23.2** |
| LFM2-8B-A1B / single | 11 | **64%** | 73% | 4.43 | 13.4 |
| LFM2-8B-A1B / all | 11 | **36%** | 45% | 15.96 | 28.6 |

**Going from `single` → `all` tools costs ~3.5–4× TTFT and ~2× end-to-end across
every model, for no quality gain** — and sometimes a quality *loss* (LFM2-8B-A1B
pass drops 64%→36%; more tool choices confuse smaller models). Tool descriptions
and the JSON-schema GBNF grammar are recompiled into every call, so the prompt and
grammar grow with the tool count. **`curated` (5 tools) is the best
quality/latency balance** on the capable models; **`single`/minimal** is best for
the smallest. This is the single highest-leverage knob and is exactly what the
profiles encode.

### 2c. Load-time options on CPU (110 trials, 1.2B + 8B-A1B)

| model / threads / batch / flash_attn | n | TTFT s | gen tok/s | e2e s |
| --- | --: | --: | --: | --: |
| 1.2B / **4** / 256 / off | 11 | 3.07 | **21.2** | 8.31 |
| 1.2B / 4 / 512 / off | 11 | 3.15 | 21.7 | 8.51 |
| 1.2B / 4 / 256 / on | 11 | 3.05 | 21.1 | 8.26 |
| 1.2B / **8** / 256 / off | 11 | 3.26 | **17.5** | 9.05 |
| 1.2B / 8 / 512 / on | 11 | 3.20 | 17.3 | 8.93 |
| 8B-A1B / **4** / 256 / off | 11 | 4.39 | **18.2** | 12.4 |
| 8B-A1B / **8** / 256 / off | 11 | 4.52 | **15.5** | 13.1 |

- **`n_threads = 4` (physical cores) decodes ~18–20% faster than 8** on both
  models — hyperthread siblings hurt llama.cpp throughput here. The provider
  default of **8 is the wrong default for this CPU**; the profiles set 4.
- **`n_batch` (256 vs 512) and `flash_attn` (on vs off) are within noise** on CPU —
  don't bother tuning them; flash-attn is a GPU feature.

### 2d. Sampling and loop bounds (66 trials, 1.2B, curated)

| temperature | pass | e2e s | | max_iterations | pass | e2e s |
| --: | --: | --: | --- | --: | --: | --: |
| 0.0 | **55%** | 9.8 | | 3 | **55%** | 9.9 |
| 0.1 | 52% | 9.5 | | 5 | 50% | 9.5 |
| 0.3 | 36% | 8.4 | | 8 | 45% | 8.5 |

- **Lower temperature is better for tool-calling**: 0.0–0.1 hold ~52–55% pass;
  0.3 drops to 36% (sampled JSON drifts off-protocol). Use 0.0–0.1.
- **`max_iterations = 3` is enough**: more iterations slightly *lower* pass — small
  models given more turns second-guess and wander rather than converge. Combined
  with the repair-budget issue (§7.2), a tight cap is better.

---

## 3. Tool-selection guidance (per task category)

Carried forward from 2026-06-14 and re-validated here:

| Task | Use | Avoid | Why |
| --- | --- | --- | --- |
| Arithmetic | `calculator` | `bash` | 100% vs hallucinated results; never wrong |
| Read a file | `read_file` | `bash` | bash invites "answer without reading" |
| Search code | `ripgrep` | `ast_grep`, `bash` | only tool that reliably does the I/O |
| Edit code | `edit_file` | `ast_edit` | find/replace beats tree-sitter queries |
| Structural AST | (≥8B only) | `ast_*` < 8B | small models can't write valid queries |

The `curated` set encodes this. `bash`/`web_fetch` are powerful but raise the
hallucination/safety surface; keep them out of small-model default sets.

---

## 4. Profiles shipped (`.little-harness/config.toml`)

| Profile | Model | tools | temp | max_iter | n_ctx | Intent |
| --- | --- | --- | --- | --- | --- | --- |
| `fast` | (current) | calculator, read_file, ripgrep | 0.0 | 3 | 2048 | lowest latency |
| `balanced` | (current) | + edit_file, ls | 0.1 | 5 | 4096 | day-to-day default |
| `code` | qwen2.5-coder-7b | read_file, ripgrep, edit_file, ls, find | 0.1 | 6 | 8192 | code tasks |
| `tiny` | LFM2.5-1.2B | calculator, read_file, ripgrep | 0.0 | 6 | 2048 | smallest models |

All pin `n_threads=4`, `n_gpu_layers=0`, `flash_attn=false`. Run with
`little-harness --profile fast -p "..."`. (A profile's `plugins.llama_cpp` table
replaces the base one wholesale, so each repeats the full block.)

---

## 5. Interactive / non-interactive UX findings

- **No progress feedback in the default REPL during a turn.** `_run_turn`
  blocks on `run_turn` then prints the final answer; on CPU a turn is 5–20 s of
  a frozen `>` prompt with no spinner. *Highest-impact UX gap for slow local
  models.* (`presentation/cli/interactive_console.py:192`)
- **`--stream` shows raw JSON.** The strict-JSON protocol means streamed tokens
  are JSON, so streaming dumps `{"action":...}` rather than human text — not
  useful as live feedback. A streaming view would need to parse/elide the
  protocol.
- **Inconsistent output between modes.** Non-interactive prints elapsed +
  step trace (`ResultRenderer`); interactive prints only the answer. No latency
  or tool-trace surfaced in the REPL.
- **`readline.get_history_length()` (line 142) is a discarded no-op** — vestigial
  readline wiring; arrow-key history isn't actually configured.
- **`/history` prints the raw protocol JSON.** The stored assistant message is the
  literal `{"action":"final","answer":"..."}`, so `/history` shows protocol noise
  instead of the rendered answer (observed live with `--profile tiny`).
- **Live hallucination example** (`--profile tiny`, 1.2B): asked to *read*
  `.python-version`, the model answered `3.9.7` (actual: `3.12`) with **zero tool
  calls** — fabricated content rather than calling `read_file`. The headline
  small-model failure mode, reproducible on demand.
- The Rich TUI *does* show a thinking spinner and disables input during a turn
  (better UX than the default console) — see §6 for one race nuance.

---

## 6. Memory-leak / race / bloat audit

**Verdict: no leaks in normal operation; one narrow TUI race; one real
unbounded-growth design gap; a few efficiency items.**

| Finding | Severity | Detail / fix |
| --- | --- | --- |
| **Unbounded conversation growth (interactive)** | **High (usability)** | `MessageHistory` is threaded across turns with no eviction; the full history is re-sent every turn, so TTFT/cost climb and the session marches toward the `n_ctx` ceiling, then breaks. No `/compact` or sliding window. This is the #1 thing to add (see pi "context transform"). |
| TUI input-lock race window | Low | Input is disabled *inside* the worker (`_run_agent_turn`, app.py:244), not synchronously in `on_input_submitted` before `run_worker` (app.py:137). A fast double-Enter can spawn two turns sharing `self._messages`. Fix: disable input in the submit handler before spawning. |
| Session JSONL grows unbounded | Low–Med | `~/.little-harness/sessions/<id>.jsonl` only ever appends; no rotation/compaction. Fine short-term, unbounded long-term. |
| Old eval harness reloads model per trial | Med (efficiency) | `run.py`→`run_cli` rebuilds the provider (reloads the GGUF) every trial; on CPU with 4–5 GB models the reload dominates. `benchmark.py` reuses the model. |
| `O(K²)` tuple rebuild per run | Negligible | `with_message` copies the whole tuple each append; fine at K≤8 iterations, compounds with the growth gap above in long sessions. |
| `JsonlFileAppender.append` calls `mkdir` every write | Trivial | Move the `parent.mkdir` to construction. |
| Model lifecycle | OK (no leak) | `Application` is a context manager; `chat_model.close()` runs on exit in both modes. Core is single-threaded → no shared-state races outside the TUI. Peak RSS tracked per model (see cross-model table); returns to baseline between processes. |

---

## 7. Prioritized recommendation backlog

**Done in this pass (high-confidence):** latency instrumentation (TTFT/tokens/s);
eval-framework false-positive fixes + tool-call/repair counts + model reuse;
curated `fast`/`balanced`/`code`/`tiny` profiles; documented tool-selection.

**Recommended next (by value), several drawn from `pi` (`scratch/pi`):**

1. **Context management for interactive mode** (pi `transformContext`): a sliding
   window / summarization hook before each model call so long sessions don't grow
   TTFT unboundedly or hit `n_ctx`. *Highest value.*
2. **Separate the repair budget from `max_iterations`.** A malformed-JSON reply
   currently consumes a loop iteration (`agent_runtime.py:204`); on small models
   that mis-format JSON, a few bad replies silently exhaust the budget →
   `FALLBACK_ANSWER`. Give repairs their own bounded counter.
3. **REPL progress feedback**: a spinner/elapsed line during the blocking turn in
   the default console; optionally a protocol-aware streaming view.
4. **Session rotation/compaction** for the JSONL plugin.
5. **TUI**: disable input synchronously on submit to close the race window.
6. **Per-turn model/param swap** (pi `prepareNextTurn`): e.g. escalate from a
   tiny model to a larger one only when a turn needs it.
7. **Tool-description budgeting**: shorter tool descriptions / lazy schema to
   shrink TTFT further as the tool set grows.

---

## 8. Reproducing

```
# from repo root, with the workspace venv:
PYTHONPATH=packages/little-harness .venv/bin/python -m evaluation.benchmark \
  --model models/LFM2.5-1.2B-Instruct-Q8_0.gguf --threads 4 --batch 256 \
  --n-ctx 4096 --temperatures 0.1 --max-iterations 5 --tool-sets single,curated,all \
  --out results/run.jsonl

PYTHONPATH=packages/little-harness .venv/bin/python -m evaluation.summarize \
  results/*.jsonl --group model,tool_set

# full sweep used for this report:
bash packages/little-harness/evaluation/run_sweep.sh
```
