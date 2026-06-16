# Implementation Plan: little-harness features from pi

Based on analysis of [pi](https://github.com/earendil-works/pi) vs little-harness.

## Phase 1: Core (`little-harness-core`)

### 1.1 Tool Output Truncation (done)

- **Files**:
  - `domain/values/truncation.py` — `TruncationConfig`, `TruncationResult` value objects
  - `application/ports/tool_truncator.py` — `ToolTruncator` protocol
  - `infrastructure/truncation/head_truncator.py` — `HeadTruncator`
  - `infrastructure/truncation/tail_truncator.py` — `TailTruncator`
  - `application/agent_dependencies.py` — add `truncation_config`
  - `application/agent_runtime.py` — wire truncation after tool run
- **Tests**:
  - `tests/domain/test_truncation_value_objects.py`
  - `tests/infrastructure/test_truncators.py`
  - `tests/application/test_truncation_in_runtime.py`

### 1.2 Reasoning/Thinking (done)

- **New files**:
  - `domain/values/thinking.py` — `ThinkingLevel` enum, `ThinkingBudget`, `ThinkingContent`
- **Changes**:
  - `application/ports/chat_model.py` — add `thinking_level`, `thinking_budget` to `ChatCompletionRequest`, add `supports_thinking()`
  - `domain/message.py` — `MessageContent` accepts `ThinkingContent` alongside text
- **Tests**:
  - `tests/domain/test_thinking_values.py`
- **Implementation note**: TDD. Start with value objects (frozen dataclasses/enums matching existing pattern), then port changes, then message integration. No thinking UI or provider support in this phase — just the domain values and ports.

### 1.3 Skills System (done)

- **New files**:
  - `domain/skill.py` — `Skill` entity (`name`, `description`, `content`, `file_path`)
  - `domain/values/skill_values.py` — `SkillName`, `SkillDescription`
  - `application/ports/skill_loader.py` — `SkillLoader` protocol, `SkillDiagnostic`, `SkillDiagnosticCode`
  - `infrastructure/skills/file_system_skill_loader.py` — `FileSystemSkillLoader`
- **Changes**:
  - `application/agent_dependencies.py` — added `skill_loader: SkillLoader`
  - `application/agent_runtime.py` — `build_system_message()` appends skills XML block
  - `composition.py` — wired `FileSystemSkillLoader` with `config.skill_paths`
  - `presentation/cli/app_config.py` — added `skill_paths` config field
  - `presentation/cli/argument_parser.py` — passes `skill_paths` to `AppConfig`
- **Tests**:
  - `tests/domain/test_skill_values.py` (8 tests)
  - `tests/domain/test_skill_entity.py` (3 tests)
  - `tests/infrastructure/skills/test_file_system_skill_loader.py` (8 tests)
  - `tests/application/test_skills_in_system_prompt.py` (4 tests)

### 1.4 Skill Command (done)

- **New files**: None
- **Changes**:
  - `presentation/cli/repl_command.py` — `command_args` property on `ReplConsole` protocol, `list_skills()`, `reload_skills()`, `SkillCommand` class, registered in `builtin_commands()`
  - `presentation/cli/interactive_console.py` — optional `SkillLoader` param, `command_args` property, `list_skills()` and `reload_skills()` methods, smarter `_process_command()` that extracts base command name and args
  - `composition.py` — created `skill_loader` in `run_cli()`, passed to `InteractiveConsole`; threaded `skill_loader` param through `build_application()` and `build_dependencies()`
- **Tests**:
  - `tests/presentation/test_repl_command.py` — `SkillCommand` conformance, execute, help output
  - `tests/presentation/test_interactive_console.py` — `/skill` and `/skill reload` REPL tests
  - `tests/test_composition.py` / `test_composition_config.py` — updated mocks for `skill_loader` param

## Phase 2: Tool Plugins (new packages)

### 2.1 `little-harness-find` (done)

- Pure Python glob search (no external deps)
- `name: "find"`, parameters: `pattern`, `path`, `limit`
- 15 tests, 98% coverage
- `packages/little-harness-find/src/little_harness_find/`

### 2.2 `little-harness-ls` (done)

- Pure Python: `os.listdir` + `os.stat`
- `name: "ls"`, parameters: `path`, `limit`
- 17 tests, 96% coverage
- `packages/little-harness-ls/src/little_harness_ls/`

### 2.3 `little-harness-web-fetch` (done)

- Stdlib `urllib.request` for GET
- `name: "web_fetch"`, parameters: `url`, `timeout`, `format`
- 17 tests, 100% coverage
- `packages/little-harness-web-fetch/src/little_harness_web_fetch/`
- URL opener injectable for testing (`FakeUrlOpener`)

## Phase 3: Session Plugin Fix (done)

### 3.1 CLI and Config
- **`presentation/cli/argument_parser.py`** — Added `-s` / `--session` CLI arg (dest=`session_id`)
- **`presentation/cli/app_config.py`** — Added `session_id: SessionId | None = None`
- **`config_types.py`** — Added `session_id: str | None = None`
- **Tests**: `tests/presentation/test_argument_parser.py` — session arg parsing tests

### 3.2 Plugin Discovery
- **`plugin_discovery.py`** — Added `SESSION_PLUGIN_GROUP`, `discover_session_plugin()`, `default_session_plugin_name()`
- Uses `require_sole_installed` pattern matching existing observer/provider discovery

### 3.3 Composition Wiring
- **`composition.py`** — Added `_build_session_plugin()` and `_load_session_history()`
- `run_cli()`: always builds session plugin in interactive mode, uses its observer, loads saved history on resume
- One-shot mode (`--prompt`): builds plugin only if `--session` provided; uses `run_turn()` with loaded history
- Falls back to configured/null observer when no session plugin installed
- **Tests**: `tests/test_composition.py` — 7 tests for wiring, observer threading, history loading

### 3.4 Interactive Console
- **`interactive_console.py`** — Accepts `_initial_messages` to pre-populate conversation on resume

### 3.5 Session Protocol
- **`application/ports/session_plugin.py`** — Added `session_id` property and `fork()` method to protocol

### 3.6 Tree/Branching (session-jsonl)
- **`plugin.py`** — `JsonlSessionPlugin.fork()` creates child session with `parent_id`
- **`jsonl_observer.py`** — `_with_parent()` helper adds `parent_id` to every event when set
- **Tests**: 14 existing + structural conformance to updated protocol

## Phase 4: Hook System Enhancement (pending)

- More hook points (`on_tool_call`, `on_tool_result`, `on_context_build`, etc.)
- Middleware chain composition in `HookChain`

## Provider Updates (pending)

- Add thinking support to `little-harness-llama-cpp`
- Add thinking support to `little-harness-litellm`
