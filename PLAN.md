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

## Phase 2: Tool Plugins (new packages)

### 2.1 `little-harness-find` (pending)

- Pure Python glob search (no external deps)
- Optional `fd` subprocess fallback
- `name: "find"`, parameters: `pattern`, `path`, `limit`

### 2.2 `little-harness-ls` (pending)

- Pure Python: `os.listdir` + `os.stat`
- `name: "ls"`, parameters: `path`, `limit`

### 2.3 `little-harness-web-fetch` (optional)

- Stdlib `urllib.request` for GET
- Optional `web_search` tool

## Phase 3: Session Plugin Fix (pending)

- Wire `session-jsonl` into composition root
- Add tree/branching with JSONL id/parentId

## Phase 4: Hook System Enhancement (pending)

- More hook points (`on_tool_call`, `on_tool_result`, `on_context_build`, etc.)
- Middleware chain composition in `HookChain`

## Provider Updates (pending)

- Add thinking support to `little-harness-llama-cpp`
- Add thinking support to `little-harness-litellm`
