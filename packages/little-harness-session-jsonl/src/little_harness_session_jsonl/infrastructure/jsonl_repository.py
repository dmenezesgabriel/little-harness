"""Missing."""

import json
from pathlib import Path

from little_harness.application.ports.agent_policy import AgentPolicy
from little_harness.application.ports.session_repository import SessionRepository
from little_harness.domain.message import ChatMessage
from little_harness.domain.message_history import MessageHistory
from little_harness.domain.tool_result import ToolRunResult
from little_harness.domain.values.role import ASSISTANT, USER
from little_harness.domain.values.text_values import (
    MessageContent,
    Prompt,
    SessionId,
    ToolName,
    ToolOutput,
)


class JsonlSessionRepository(SessionRepository):
    """Restores MessageHistory from JSONL events."""

    def __init__(self, storage_dir: Path, policy: AgentPolicy) -> None:
        """Initialize with storage directory and policy."""
        self._storage_dir = storage_dir
        self._policy = policy

    def load(self, session_id: SessionId) -> MessageHistory:
        """Load message history from the JSONL file for the given session ID."""
        file_path = self._storage_dir / f"{session_id.value}.jsonl"  # pragma: no mutate
        if not file_path.exists():
            return MessageHistory()

        messages = MessageHistory()
        current_prompt: Prompt | None = None

        with file_path.open("r", encoding="utf-8") as f:  # pragma: no mutate
            for line in f:
                if not line.strip():  # pragma: no mutate
                    continue  # pragma: no mutate

                event = json.loads(line)
                event_type = event.get("type")

                if event_type == "run_started":
                    current_prompt = Prompt(event["prompt"])
                    messages = messages.with_message(
                        ChatMessage(USER, MessageContent(current_prompt.value))
                    )
                    continue

                if event_type == "model_completed":
                    messages = messages.with_message(
                        ChatMessage(ASSISTANT, MessageContent(event["output"]))
                    )
                    continue

                if event_type == "tool_invoked" and current_prompt is not None:
                    result = ToolRunResult(
                        ToolName(event["tool_name"]),
                        ToolOutput(event["output"]),
                        succeeded=event["succeeded"],
                    )
                    msg = self._policy.build_tool_observation_message(
                        current_prompt, result
                    )
                    messages = messages.with_message(msg)
                    continue

                if event_type == "repair" and current_prompt is not None:
                    # In a real scenario, we might need the actual exception type.
                    # Here we pass a generic Exception with the error message.
                    error = Exception(event["message"])
                    msg = self._policy.build_repair_message(current_prompt, error)
                    messages = messages.with_message(msg)
                    continue  # pragma: no mutate

        return messages
