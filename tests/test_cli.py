from __future__ import annotations

from pathlib import Path

import pytest

from local_llm.agent import AgentDecision, AgentResult, AgentStep
from local_llm.cli import AppConfig, format_step_action, parse_args, print_result


class TestFormatStepAction:
    def test_returns_repair_when_decision_missing(self) -> None:
        # Arrange
        step = AgentStep(1, "output", None, "observation")

        # Act / Assert
        assert format_step_action(step) == "repair"

    def test_returns_final_for_final_decision(self) -> None:
        # Arrange
        decision = AgentDecision("final", None, None, "done")
        step = AgentStep(1, "output", decision, "observation")

        # Act / Assert
        assert format_step_action(step) == "final"

    def test_returns_tool_name_for_tool_decision(self) -> None:
        # Arrange
        decision = AgentDecision("tool", "calculator", "2 + 2", None)
        step = AgentStep(1, "output", decision, "observation")

        # Act / Assert
        assert format_step_action(step) == "calculator"

    def test_falls_back_to_tool_when_name_empty(self) -> None:
        # Arrange
        decision = AgentDecision("tool", "", "2 + 2", None)
        step = AgentStep(1, "output", decision, "observation")

        # Act / Assert
        assert format_step_action(step) == "tool"


class TestPrintResult:
    def test_prints_answer_and_elapsed_without_steps(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        result = AgentResult(answer="the answer", elapsed_seconds=1.5)

        # Act
        print_result(result)

        # Assert
        output = capsys.readouterr().out
        assert "the answer" in output
        assert "Elapsed: 1.50s" in output
        assert "Agent steps:" not in output

    def test_prints_steps_when_present(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Arrange
        decision = AgentDecision("tool", "calculator", "2 + 2", None)
        step = AgentStep(1, "output", decision, "4")
        result = AgentResult(answer="done", elapsed_seconds=0.5, steps=(step,))

        # Act
        print_result(result)

        # Assert
        output = capsys.readouterr().out
        assert "Agent steps:" in output
        assert "Step 1" in output
        assert "Action: calculator" in output
        assert "Observation: 4" in output


class TestParseArgs:
    def test_uses_defaults_when_no_flags_given(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr("sys.argv", ["local-llm"])

        # Act
        config = parse_args()

        # Assert: prompt is compared separately because the default is a long string.
        expected = AppConfig(
            prompt=config.prompt,
            model_path=Path("models/LFM2-8B-A1B-Q4_K_M.gguf"),
            context_size=8192,
            thread_count=8,
            gpu_layer_count=0,
            temperature=0.0,
            max_tokens=512,
            max_iterations=5,
        )
        assert config == expected
        assert "llama.cpp" in config.prompt

    def test_reads_overrides_from_argv(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setattr(
            "sys.argv",
            [
                "local-llm",
                "--prompt",
                "What is 2 + 2?",
                "--model-path",
                "/tmp/model.gguf",
                "--ctx",
                "4096",
                "--threads",
                "4",
                "--gpu-layers",
                "20",
                "--temperature",
                "0.7",
                "--max-tokens",
                "256",
                "--max-iterations",
                "3",
            ],
        )

        # Act
        config = parse_args()

        # Assert
        expected = AppConfig(
            prompt="What is 2 + 2?",
            model_path=Path("/tmp/model.gguf"),
            context_size=4096,
            thread_count=4,
            gpu_layer_count=20,
            temperature=0.7,
            max_tokens=256,
            max_iterations=3,
        )
        assert config == expected
