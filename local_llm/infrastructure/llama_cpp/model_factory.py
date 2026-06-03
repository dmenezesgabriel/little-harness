"""Builds a configured llama.cpp model from settings."""

from __future__ import annotations

from llama_cpp import Llama

from local_llm.infrastructure.llama_cpp.settings import LlamaCppModelSettings


def create_llama_model(settings: LlamaCppModelSettings) -> Llama:
    path = settings.model_path.value

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Expected a local GGUF model file."
        )

    return Llama(
        model_path=str(path),
        n_ctx=settings.context_size.value,
        n_threads=settings.thread_count.value,
        n_gpu_layers=settings.gpu_layer_count.value,
        verbose=False,
    )
