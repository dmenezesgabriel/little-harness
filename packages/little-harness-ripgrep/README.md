# little-harness-ripgrep

A fast content-search tool plugin for [little-harness](https://github.com/dmenezesgabriel/little-harness),
backed by [ripgrep](https://github.com/BurntSushi/ripgrep).

| Tool | Input | Approval |
| --- | --- | --- |
| `ripgrep` | a ripgrep argument line (pattern, paths, flags) | no (read-only) |

The input is split with shell rules but executed **without a shell** (`rg` is
invoked directly), so there is no shell-injection surface. ripgrep's exit codes
are interpreted for the agent: matches (0) and "no matches" (1) are both
successes; a real error (2+) is a failure, as is an absent binary or a timeout.

## Requirements

The `rg` binary must be installed and on `PATH` (e.g. `apt install ripgrep`,
`brew install ripgrep`). It is a system dependency, not a Python package.

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
