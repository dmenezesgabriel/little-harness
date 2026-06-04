# little-harness-file-tools

Filesystem and shell tool plugins for [little-harness](https://github.com/dmenezesgabriel/little-harness):

| Tool | Input | Approval | Notes |
| --- | --- | --- | --- |
| `read_file` | a path string | no | Returns the UTF-8 contents of a file. |
| `write_file` | `{"path", "content"}` | yes | Writes text, creating parent directories and overwriting. |
| `edit_file` | `{"path", "old", "new"}` | yes | Replaces the **unique** occurrence of `old` with `new`. |
| `bash` | a command line | yes | Runs a shell command with a timeout, behind a guardrail. |

All four are pure-stdlib (`pathlib`, `json`, `subprocess`) — no third-party dependencies.

## Install

```bash
uv pip install "little-harness[file-tools]"
```

## Safety model

Two independent layers protect the dangerous tools:

- **Human-in-the-loop approval.** `write_file`, `edit_file`, and `bash` declare
  `requires_approval` in their `ToolSpec`. With a terminal attached, the agent
  prompts before each call; `--yes` (or a non-interactive run) auto-approves.
- **Bash guardrails.** `DangerousCommandGuardrail` blocks destructive commands
  (`rm -rf`, fork bombs, `mkfs`, `dd` to a device, `shutdown`, …) *before*
  execution — always on, even in an approved session. The denylist is injected,
  so it can be extended or replaced.

`bash` runs through the system shell on purpose, so pipes and globs work; the
`subprocess` use is confined to `SubprocessShellRunner` and documented in
`[tool.bandit]`.

## Selecting tools

Use the core `--tools` flag to enable a subset, e.g. `--tools read_file,ripgrep`.
Omitting `--tools` enables every installed tool.
