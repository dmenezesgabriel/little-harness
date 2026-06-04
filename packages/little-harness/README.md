# little-harness

Batteries-included install of the little-harness local LLM agent: the core CLI
(`little-harness`), the llama.cpp chat-model provider, and the calculator tool.

```
uv pip install little-harness
little-harness --provider llama_cpp -o model_path=models/model.gguf -p "2 + 2?"
```

Install only what you need instead by combining `little-harness-core` with
individual provider/tool plugins (e.g. `little-harness-litellm`). See the
repository root README for the full architecture.
