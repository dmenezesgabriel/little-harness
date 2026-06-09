# Error types

All custom exceptions live in `little_harness.domain.errors`. Every error message
includes the offending value and the expected shape.

| Error | Meaning |
|-------|---------|
| `AgentProtocolError(ValueError)` | Model output could not be parsed into a valid decision |
| `ToolRegistrationError(ValueError)` | Duplicate tool name registered in `ToolRegistry` |
| `UnknownProviderError(ValueError)` | `--provider` name not found among installed plugins |
| `UnknownToolError(ValueError)` | Tool name in `--tools` not found among installed plugins |
| `UnknownPolicyError(ValueError)` | `--policy` name not found among installed plugins |
| `UnknownObserverError(ValueError)` | `--observer` name not found among installed plugins |
| `UnknownUiError(ValueError)` | `--ui` name not found among installed plugins |
| `UnknownPermissionRequesterError(ValueError)` | Permission requester name not found |
| `UnknownSessionPluginError(ValueError)` | Session plugin name not found |

## CLI exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Any error (parsing, unknown plugin, runtime failure) |

All errors caught by the CLI `run()` wrapper are printed as `error: <message>`
to stderr. Pass `--log` for a full traceback.
