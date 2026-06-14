# Tool Overlap Evaluation Report

**Date**: 2026-06-14
**Model**: `LFM2.5-1.2B-Instruct-Q8_0.gguf` (1.2B params, Q8_0)
**Provider**: llama.cpp (CPU, 6 threads, n_ctx=4096)
**Suite**: 25 evaluation cases × 2–3 candidate tools = 53 trials

---

## 1. Arithmetic: calculator vs bash (8 cases)

| Metric | calculator | bash |
|--------|-----------|------|
| Pass rate | **100%** (8/8) | 88% (7/8) |
| Avg time | 12.5s | **8.4s** |
| Failure pattern | — | `large_numbers`: hallucinated 6912 → 7902 |

**Conclusion**: Both tools are reliable for arithmetic. `bash` is faster because
it can answer without an explicit tool call (the model just does arithmetic in its
response). `calculator` is more reliable (100% vs 88%). **Use calculator** —
it's safer and never hallucinates a wrong result.

---

## 2. Read File: read_file vs bash (5 cases)

| Metric | read_file | bash |
|--------|-----------|------|
| Pass rate | **40%** (2/5) | **40%** (2/5) |
| Avg time | 18.0s | 8.4s |

**Failure pattern (read_file)**: The model hallucinated the filename — it called
`Read error: No such file or directory: 'README.txt'` instead of `note.txt`.
This is a model-side instruction-following bug, not a tool bug.

**Failure pattern (bash)**: The model answered without reading — it generated
plausible-sounding but fake content ("project timeline", "product names, prices").

**Conclusion**: Neither tool works well with this small model for reading.
The model frequently bypasses the tool and hallucinates. **read_file is still
preferable** — when it does call the tool, it returns real content. The failures
are model capability, not tool design.

---

## 3. Search Code: ripgrep vs ast_grep vs bash (7 cases)

| Metric | ripgrep | ast_grep | bash |
|--------|---------|----------|------|
| Pass rate | **57%** (4/7) | 33% (1/3) | 43% (3/7) |
| Avg time | 11.3s | 8.4s | **5.8s** |

**Failure pattern (ripgrep)**: Model doesn't pass correct flags (e.g. `-i` for
case-insensitive search). Searches sometimes match the evaluation YAML files
instead of the target file.

**Failure pattern (ast_grep)**: Critical — the model **cannot generate valid
tree-sitter queries**. Every failure was `Invalid query: 'function calls'` or
`'import.*'` instead of a proper `(call) @match` query. The passes were false
positives where `expected_substring` matched the error text.

**Failure pattern (bash)**: Model frequently answered without calling any tool.

**Conclusion**: **ripgrep is the best choice for search** — it's the only tool
that actually performs file I/O. `ast_grep` is unusable with this model size
(models can't produce correct queries). `bash` hallucinates answers.

---

## 4. Edit Code: edit_file vs ast_edit (5 cases)

| Metric | edit_file | ast_edit |
|--------|-----------|----------|
| Pass rate | **60%** (3/5) | **60%** (3/5) |
| Avg time | 13.5s | 13.7s |

**Failure pattern (edit_file)**: Model occasionally forgot the exact replacement
syntax. The tool worked correctly (`Replaced 1 occurrence`) but the model's
final answer didn't match the expected substring (e.g. hallucinated "TIMEOUT = 30"
instead of "TIMEOUT = 60").

**Failure pattern (ast_edit)**: Same critical query problem as `ast_grep` —
model couldn't generate correct tree-sitter queries. The passes were false
positives matching error messages.

**Conclusion**: **edit_file is more usable** — text find-and-replace is
simpler for models. `ast_edit` fundamentally requires tree-sitter query
generation that's beyond small models. Consider providing query templates
in the tool description, or only use `ast_edit` with models ≥8B.

---

## 5. Set A vs Set B: Single-tool vs All-tool Selection

| Metric | Set A (single tool) | Set B (all tools) |
|--------|---------------------|-------------------|
| Avg time | 5.5s | **20.9s** (3.8× slower) |
| Pass rate | 56% (5/9) | 56% (5/9) (partial) |

**Observations**:
- All-tool mode is 2–5× slower because the model must choose from 8 tools.
- Pass rates were identical for the tests that completed — the model chose the
  same tools it would have been given in single-tool mode.
- The all-tool tests add confidence that the model won't misuse tools, but
  are expensive for CI.

**Recommendation**: Run Set B only with stronger models (≥8B) or as an
occasional check, not in every CI run.

---

## 6. Evaluation Framework Quality

**Issues found in the evaluation itself**:

| Issue | Impact | Fix |
|-------|--------|-----|
| `expected_substring` matched error text | 5 false passes (`ast_grep`, `ast_edit`) | Add `not_in_error: true` or use structured output checks |
| `expected_substring` matched hallucinated text | 2 false passes (`read_nonexistent` for `bash`) | Check for "error" or "not found" instead of filename |
| Model answered without calling any tool | ~15% of trials are measuring hallucination, not tools | Track `tool_call_count` in `TrialResult` |
| No output-quality measurement | Can't distinguish "correct file read" from "plausible lie" | Add `reported_content_accuracy` metric |

---

## Overall Conclusions

1. **Model size is the dominant factor** — the 1.2B model is too small to
   reliably call tools that require parameter construction (filenames, flags,
   tree-sitter queries). Upgrade to ≥8B for meaningful tool evaluation.

2. **Best tool per category**:
   - Arithmetic → **calculator** (100% pass, never hallucinates)
   - File reading → **read_file** (hallucinates less than bash)
   - Code search → **ripgrep** (only tool that actually reads files)
   - Code editing → **edit_file** (simpler syntax than ast_edit)

3. **`ast_grep`/`ast_edit` need query help** — the model can't generate
   correct tree-sitter queries. Consider:
   - Adding common query templates to tool descriptions
   - Providing a query library
   - Detecting invalid queries and returning helpful suggestions

4. **Evaluation is valuable** — it caught issues the BDD tests miss:
   hallucination rates, tool-argument construction failures, and false
   positives. It should live alongside the test suite, not replace it.

5. **Next step**: Re-run with `LFM2.5-8B-A1B-Q4_K_M.gguf` to confirm which
   failures are model-size vs tool-design issues.
