# little-harness

Umbrella for the little-harness local LLM agent. Installing it bare pulls in the
core CLI (`little-harness`) only; providers and tools are opt-in **extras**.

```
uv pip install "little-harness[llama-cpp,calculator]"
little-harness -o model_path=models/model.gguf -p "2 + 2?"
```

Available extras: `llama-cpp`, `litellm`, `calculator`, and `all`. Combining
`little-harness-core` with individually-named plugin distributions (e.g.
`little-harness-litellm`) is equivalent. See the repository root README for the
full architecture.
