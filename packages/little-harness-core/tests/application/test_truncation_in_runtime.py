from __future__ import annotations

from little_harness.application.agent_dependencies import AgentDependencies
from little_harness.application.agent_runtime import AgentRuntime, AgentRuntimeConfig
from little_harness.application.tool_registry import ToolRegistry
from little_harness.domain.tool_result import ToolRunRequest, ToolRunResult
from little_harness.domain.tool_spec import ToolInputSchema, ToolSpec
from little_harness.domain.values.numeric_values import (
    MaxIterations,
    MaxTokens,
    Temperature,
)
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    ToolName,
    ToolOutput,
)
from little_harness.domain.values.truncation import TruncationConfig
from little_harness.infrastructure.hooks.null_hook import NullHook
from little_harness.infrastructure.truncation.head_truncator import HeadTruncator

from tests.application.fakes import (
    DecisionQueuePolicy,
    RecordingChatModel,
    RecordingObserver,
    RecordingTokenSink,
    final_decision,
    tool_decision,
)


class LargeOutputTool:
    def __init__(self, name: str = "large_tool", line_count: int = 100) -> None:
        self._name = ToolName(name)
        self.line_count = line_count

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            self._name,
            "Produces many lines.",
            ToolInputSchema("none"),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        lines = [f"line_{i}" for i in range(self.line_count)]
        return ToolRunResult(self._name, ToolOutput("\n".join(lines)), succeeded=True)


class EmptyOutputTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            ToolName("empty"),
            "Returns empty string.",
            ToolInputSchema("none"),
        )

    def run(self, request: ToolRunRequest) -> ToolRunResult:
        return ToolRunResult(ToolName("empty"), ToolOutput(""), succeeded=True)


class TestTruncationInRuntime:
    def test_truncates_tool_output_when_exceeding_line_limit(self) -> None:
        policy = DecisionQueuePolicy(
            [
                tool_decision("large_tool", "run"),
                final_decision("done"),
            ]
        )
        dependencies = AgentDependencies(
            chat_model=RecordingChatModel(["tool_output", "final"]),
            tool_registry=ToolRegistry([LargeOutputTool(line_count=100)]),
            policy=policy,
            observer=RecordingObserver(),
            token_sink=RecordingTokenSink(),
            hooks=NullHook(),
            truncator=HeadTruncator(),
            truncation_config=TruncationConfig(max_lines=3, max_bytes=51200),
        )
        config = AgentRuntimeConfig(
            max_iterations=MaxIterations(3),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
        )
        runtime = AgentRuntime(dependencies, config)
        runtime.run(Prompt("run tool"))
        assert len(policy.tool_results) == 1
        output = policy.tool_results[0].output.value
        assert "line_0" in output
        assert "line_2" in output
        assert "line_3" not in output

    def test_passes_through_small_output(self) -> None:
        policy = DecisionQueuePolicy(
            [
                tool_decision("large_tool", "run"),
                final_decision("done"),
            ]
        )
        dependencies = AgentDependencies(
            chat_model=RecordingChatModel(["tool_output", "final"]),
            tool_registry=ToolRegistry([LargeOutputTool(line_count=3)]),
            policy=policy,
            observer=RecordingObserver(),
            token_sink=RecordingTokenSink(),
            hooks=NullHook(),
            truncator=HeadTruncator(),
            truncation_config=TruncationConfig(max_lines=10, max_bytes=51200),
        )
        config = AgentRuntimeConfig(
            max_iterations=MaxIterations(3),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
        )
        runtime = AgentRuntime(dependencies, config)
        runtime.run(Prompt("run tool"))
        assert len(policy.tool_results) == 1
        output = policy.tool_results[0].output.value
        assert "line_0" in output
        assert "line_2" in output

    def test_empty_output_truncation_is_noop(self) -> None:
        policy = DecisionQueuePolicy(
            [
                tool_decision("empty", "run"),
                final_decision("done"),
            ]
        )
        dependencies = AgentDependencies(
            chat_model=RecordingChatModel(["tool_output", "final"]),
            tool_registry=ToolRegistry([EmptyOutputTool()]),
            policy=policy,
            observer=RecordingObserver(),
            token_sink=RecordingTokenSink(),
            hooks=NullHook(),
            truncator=HeadTruncator(),
            truncation_config=TruncationConfig(max_lines=1, max_bytes=51200),
        )
        config = AgentRuntimeConfig(
            max_iterations=MaxIterations(3),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
        )
        runtime = AgentRuntime(dependencies, config)
        runtime.run(Prompt("run tool"))
        assert len(policy.tool_results) == 1
        assert policy.tool_results[0].output.value == ""

    def test_succeeded_flag_preserved_after_truncation(self) -> None:
        policy = DecisionQueuePolicy(
            [
                tool_decision("large_tool", "run"),
                final_decision("done"),
            ]
        )
        dependencies = AgentDependencies(
            chat_model=RecordingChatModel(["tool_output", "final"]),
            tool_registry=ToolRegistry([LargeOutputTool(line_count=100)]),
            policy=policy,
            observer=RecordingObserver(),
            token_sink=RecordingTokenSink(),
            hooks=NullHook(),
            truncator=HeadTruncator(),
            truncation_config=TruncationConfig(max_lines=1, max_bytes=51200),
        )
        config = AgentRuntimeConfig(
            max_iterations=MaxIterations(3),
            temperature=Temperature(0.0),
            max_tokens=MaxTokens(128),
        )
        runtime = AgentRuntime(dependencies, config)
        result = runtime.run(Prompt("run tool"))
        assert result.answer == MessageContent("done")
        assert len(policy.tool_results) == 1
        assert policy.tool_results[0].succeeded is True
