"""Maps domain chat messages to LiteLLM's OpenAI-style message dicts."""

from __future__ import annotations

from little_harness.domain.message import ChatMessage


def to_litellm_message(message: ChatMessage) -> dict[str, str]:
    """Convert a domain ChatMessage to LiteLLM's OpenAI-style message dict."""
    return {"role": message.role.name, "content": message.content.value}
