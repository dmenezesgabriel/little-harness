"""Builds a configured llama.cpp model from settings."""

from __future__ import annotations

from llama_cpp import Llama
from llama_cpp import llama_chat_format as _chat_fmt

from little_harness_llama_cpp.chat_template_sanitizer import (
    ChatTemplateSanitizerFactory,
)
from little_harness_llama_cpp.settings import LlamaCppModelSettings

# Some GGUF models embed non-standard Jinja2 tags (``{% generation %}`` etc.)
# that llama-cpp-python's parser rejects.  The sanitizer strategy strips them
# before the formatter sees the template.  The monkey-patch is applied once at
# import time so ``Llama()`` always gets a safe template.
_sanitizer = ChatTemplateSanitizerFactory.create_default()
_original_formatter_init = _chat_fmt.Jinja2ChatFormatter.__init__


def _formatter_init_without_generation_tags(
    self: _chat_fmt.Jinja2ChatFormatter,
    template: str,
    eos_token: str,
    bos_token: str,
    add_generation_prompt: bool = True,
    stop_token_ids: list[int] | None = None,
) -> None:
    _original_formatter_init(
        self,
        _sanitizer.sanitize(template),
        eos_token,
        bos_token,
        add_generation_prompt,
        stop_token_ids,
    )


_chat_fmt.Jinja2ChatFormatter.__init__ = _formatter_init_without_generation_tags  # type: ignore[method-assign]


def create_llama_model(settings: LlamaCppModelSettings) -> Llama:
    path = settings.model_path.value

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Expected a local GGUF model file."
        )

    return Llama(
        model_path=str(path),
        seed=settings.seed.value,
        n_ctx=settings.context_size.value,
        n_threads=settings.thread_count.value,
        # Reuse the inference threads for prefill: small local runs rarely gain
        # from a separate batch pool, and an unset value would fall back to 1.
        n_threads_batch=settings.thread_count.value,
        n_batch=settings.batch_size.value,
        n_gpu_layers=settings.gpu_layer_count.value,
        flash_attn=settings.flash_attention,
        verbose=False,
    )
