from __future__ import annotations

from pathlib import Path

from little_harness_llama_cpp.chat_template_sanitizer import (
    ChatTemplateSanitizer,
    ChatTemplateSanitizerFactory,
    NoOpSanitizer,
    StripGenerationTagsSanitizer,
)


class TestStripGenerationTagsSanitizer:
    """Strategy: removes ``{% generation %}`` / ``{% endgeneration %}`` markers."""

    def test_passes_through_a_clean_template_unchanged(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()
        template = "Hello {{ name }}"

        # Act
        result = sanitizer.sanitize(template)

        # Assert
        assert result == "Hello {{ name }}"

    def test_strips_an_opening_generation_tag(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize("{% generation %}reply")

        # Assert
        assert result == "reply"

    def test_strips_a_closing_generation_tag(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize("assistant{% endgeneration %}")

        # Assert
        assert result == "assistant"

    def test_strips_both_opening_and_closing_tags(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize("{% generation %}assistant{% endgeneration %}")

        # Assert
        assert result == "assistant"

    def test_handles_whitespace_modifiers_on_the_tag(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize("{%- generation -%}middle{%- endgeneration -%}")

        # Assert
        assert result == "middle"

    def test_strips_multiple_generation_blocks(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize(
            "{% generation %}first{% endgeneration %}"
            "{% generation %}second{% endgeneration %}"
        )

        # Assert
        assert result == "firstsecond"

    def test_preserves_surrounding_jinja2_syntax(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize(
            "{{ bos_token }}"
            "{% generation %}"
            "{{ raise_exception('stop') }}"
            "{% endgeneration %}"
        )

        # Assert
        assert result == ("{{ bos_token }}{{ raise_exception('stop') }}")

    def test_returns_an_empty_string_for_an_empty_template(self) -> None:
        # Arrange
        sanitizer = StripGenerationTagsSanitizer()

        # Act
        result = sanitizer.sanitize("")

        # Assert
        assert result == ""


class TestChatTemplateSanitizerFactory:
    """Simple factory: picks the right sanitizer strategy."""

    def test_create_default_returns_a_sanitizer(self) -> None:
        # Act
        sanitizer = ChatTemplateSanitizerFactory.create_default()

        # Assert
        assert isinstance(sanitizer, ChatTemplateSanitizer)

    def test_create_default_returns_strip_generation_tags_sanitizer(self) -> None:
        # Act
        sanitizer = ChatTemplateSanitizerFactory.create_default()

        # Assert
        assert isinstance(sanitizer, StripGenerationTagsSanitizer)

    def test_create_accepts_a_model_path(self) -> None:
        # Act
        sanitizer = ChatTemplateSanitizerFactory.create(Path("/models/m.gguf"))

        # Assert
        assert isinstance(sanitizer, ChatTemplateSanitizer)

    def test_factory_reuses_the_same_default_instance(self) -> None:
        # Arrange
        first = ChatTemplateSanitizerFactory.create_default()

        # Act
        second = ChatTemplateSanitizerFactory.create_default()

        # Assert
        assert first is second


class TestNoOpSanitizer:
    """Pass-through strategy: no transformation."""

    def test_returns_the_input_unchanged(self) -> None:
        # Arrange
        sanitizer = NoOpSanitizer()

        # Act
        result = sanitizer.sanitize("anything")

        # Assert
        assert result == "anything"

    def test_handles_empty_string(self) -> None:
        # Arrange
        sanitizer = NoOpSanitizer()

        # Act
        result = sanitizer.sanitize("")

        # Assert
        assert result == ""

    def test_handles_jinja2_syntax_unchanged(self) -> None:
        # Arrange
        sanitizer = NoOpSanitizer()

        # Act
        result = sanitizer.sanitize("{{ bos }}{% generation %}")

        # Assert
        assert result == "{{ bos }}{% generation %}"
