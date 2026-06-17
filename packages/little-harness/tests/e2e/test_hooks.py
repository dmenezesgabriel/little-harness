"""End-to-end tests for lifecycle hooks injected via the composition root.

Each scenario injects a custom hook via ``run_cli(..., extra_hooks=[…])`` and
verifies the hook affects the agent's behaviour — that the hook point is
actually wired, that ``HookChain`` folds the decision correctly, and that the
runtime applies the decision without crashing.

Tests that ``Block`` a point before the model call never call the model, yet
the hook still had to be constructed, folded into the chain, and executed
through ``build_application`` → ``AgentRuntime.run_turn`` → the hook point.
"""

from __future__ import annotations

import pytest
from little_harness.composition import run_cli
from little_harness.domain.hook_decision import Block, HookDecision, InjectContext
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.values.numeric_values import Iteration
from little_harness.domain.values.text_values import MessageContent, Prompt, RunId
from little_harness.infrastructure.hooks.null_hook import NullHook

pytestmark = [pytest.mark.integration, pytest.mark.local_model]

CANNED_ANSWER = '{"action": "final", "answer": "hook-works"}'


class BlockFirstTurnStart(NullHook):
    """Blocks on_turn_start so the agent never calls the model."""

    def on_turn_start(
        self, run_id: RunId, iteration: Iteration, prompt: Prompt, /
    ) -> HookDecision:
        return Block(MessageContent("turn-start-blocked"))


class BlockFirstModelRequest(NullHook):
    """Blocks on_model_request so the model is never called."""

    def on_model_request(
        self, run_id: RunId, iteration: Iteration, /
    ) -> HookDecision:
        return Block(MessageContent(CANNED_ANSWER))


class BlockFirstModelResponse(NullHook):
    """Replaces the model output with a canned answer."""

    def on_model_response(
        self, run_id: RunId, iteration: Iteration, output: MessageContent, /
    ) -> HookDecision:
        return Block(MessageContent(CANNED_ANSWER))


class BlockFirstTurnEnd(NullHook):
    """Replaces model output at turn end before parsing."""

    def on_turn_end(
        self, run_id: RunId, iteration: Iteration, output: MessageContent, /
    ) -> HookDecision:
        return Block(MessageContent(CANNED_ANSWER))


class InjectInstructionsAtTurnStart(NullHook):
    """Injects instructions before each model call."""

    def on_turn_start(
        self, run_id: RunId, iteration: Iteration, prompt: Prompt, /
    ) -> HookDecision:
        return InjectContext(MessageContent("Always reply with just: injected-ok"))


class InjectHintBeforeModelCall(NullHook):
    """Injects a hint before the model API call."""

    def on_model_request(
        self, run_id: RunId, iteration: Iteration, /
    ) -> HookDecision:
        return InjectContext(MessageContent("Reply with just: hint-received"))


class BlockFirstContextBuild(NullHook):
    """Blocks on_context_build so the model is never called."""

    def on_context_build(
        self, run_id: RunId, iteration: Iteration, messages: MessageHistory, /
    ) -> HookDecision:
        return Block(MessageContent(CANNED_ANSWER))


class InjectInstructionsAtContextBuild(NullHook):
    """Injects hint at context build before the model call."""

    def on_context_build(
        self, run_id: RunId, iteration: Iteration, messages: MessageHistory, /
    ) -> HookDecision:
        return InjectContext(MessageContent("Reply with just: ctx-injected"))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


class TestTurnStartHookBlock:
    """on_turn_start Block: agent aborts before calling the model."""

    def test_block_reason_appears_in_answer(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[BlockFirstTurnStart()],
        )

        assert "turn-start-blocked" in result


class TestModelRequestHookBlock:
    """on_model_request Block: model call is skipped, canned answer returned."""

    def test_canned_answer_returned_without_model_call(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[BlockFirstModelRequest()],
        )

        assert "hook-works" in result


class TestModelResponseHookBlock:
    """on_model_response Block: model output is replaced after the API call."""

    def test_output_replaced_after_model_call(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[BlockFirstModelResponse()],
        )

        assert "hook-works" in result


class TestTurnEndHookBlock:
    """on_turn_end Block: model output replaced at end of turn."""

    def test_output_replaced_at_turn_end(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[BlockFirstTurnEnd()],
        )

        assert "hook-works" in result


class TestContextBuildHookBlock:
    """on_context_build Block: model call is skipped, canned answer returned."""

    def test_canned_answer_returned_without_model_call(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[BlockFirstContextBuild()],
        )

        assert "hook-works" in result


class TestContextBuildHookInject:
    """on_context_build InjectContext: context is added before the model call."""

    def test_injected_context_appears_in_output(
        self, local_llama_options: list[str]
    ) -> None:
        provider_options = [
            item for option in local_llama_options for item in ("-o", option)
        ]
        result = run_cli(
            [
                "--provider",
                "llama_cpp",
                "--prompt",
                "say hello",
                "--yes",
                "--max-tokens",
                "512",
                "--max-iterations",
                "3",
                *provider_options,
            ],
            extra_hooks=[InjectInstructionsAtContextBuild()],
        )

        assert "ctx-injected" in result
