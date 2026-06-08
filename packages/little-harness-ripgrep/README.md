# little-harness-ripgrep

A fast content-search tool plugin for [little-harness](https://github.com/dmenezesgabriel/little-harness),
offering a pure-Python search backend that mimics the interface of [ripgrep](https://github.com/BurntSushi/ripgrep).

| Tool | Input | Approval |
| --- | --- | --- |
| `ripgrep` | a ripgrep-style argument line (pattern, paths, flags) | no (read-only) |

The input is parsed into a structured grep query and executed entirely in standard Python. Search results are formatted to match ripgrep's default format, interpreting standard exit codes: matches (0), "no matches" (1), and errors (2).

## Install

```bash
uv pip install "little-harness[ripgrep]"
```

## Examples

```
TODO src              # find TODO under src/
-i error logs         # case-insensitive search for "error" under logs/
"def main" -t py      # a quoted pattern restricted to Python files
```
