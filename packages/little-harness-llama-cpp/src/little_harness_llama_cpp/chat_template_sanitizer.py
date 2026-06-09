"""Strategy-based chat template sanitization for llama.cpp GGUF models.

Some GGUF models (e.g. LFM2.5-8B) embed Jinja2 chat templates with non-standard
``{% generation %}`` / ``{% endgeneration %}`` block tags.  llama-cpp-python's
Jinja2 parser does not register these tags, so template parsing would fail.
Each strategy here provides one way to handle such quirks, and the factory
picks the right one for the model in use.

Example:
    sanitizer = ChatTemplateSanitizerFactory.create_default()
    safe = sanitizer.sanitize("{% generation %}assistant{% endgeneration %}")

"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path


class ChatTemplateSanitizer(ABC):
    """Strategy interface: transform a Jinja2 template before model loading."""

    @abstractmethod
    def sanitize(self, template: str) -> str:
        """Return a modified template safe for llama.cpp's Jinja2 parser."""


class StripGenerationTagsSanitizer(ChatTemplateSanitizer):
    """Strip ``{% generation %}`` / ``{% endgeneration %}`` markers.

    LFM2.5 GGUF templates wrap the assistant turn inside these block tags.
    The tags are not standard Jinja2, so removing them (keeping their content)
    produces a valid template.
    """

    _GENERATION_PATTERN = re.compile(r"\{%-?\s*(?:end)?generation\s*-?%\}")

    def sanitize(self, template: str) -> str:
        """Strip generation tags from the template."""
        return self._GENERATION_PATTERN.sub("", template)


class NoOpSanitizer(ChatTemplateSanitizer):
    """Pass-through strategy — no sanitization needed."""

    def sanitize(self, template: str) -> str:
        """Return the template unchanged."""
        return template


class ChatTemplateSanitizerFactory:
    """Simple factory: selects the right ``ChatTemplateSanitizer`` strategy.

    ``create_default()`` handles every model supported today.  ``create(path)``
    is the extension point for model-specific logic: inspect model metadata
    and return a different strategy per model.
    """

    _STRIP_GENERATION = StripGenerationTagsSanitizer()
    _PASSTHROUGH = NoOpSanitizer()

    @classmethod
    def create_default(cls) -> ChatTemplateSanitizer:
        """Return the sanitizer for all currently supported GGUF models."""
        return cls._STRIP_GENERATION

    @classmethod
    def create(cls, _model_path: Path) -> ChatTemplateSanitizer:
        """Return a sanitizer appropriate for *model_path* (future extension)."""
        return cls.create_default()
