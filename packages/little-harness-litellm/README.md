# little-harness-litellm

A [little-harness](https://github.com/dmenezesgabriel/little-harness) chat-model
provider plugin backed by [LiteLLM](https://github.com/BerriAI/litellm), giving
access to many hosted providers (OpenAI, Gemini, Anthropic, …) through one
adapter. It implements the core `ChatModel` port and registers a `litellm`
provider under the `little_harness.chat_model_providers` entry-point group.

```
uv pip install "little-harness[litellm]"
export GEMINI_API_KEY=...   # keys are read from the environment
little-harness --provider litellm --model gemini/gemini-2.5-flash -p "Hello!"
```

Options: `api_base`, `api_key` (the key is also read from the environment, e.g.
`GEMINI_API_KEY` / `OPENAI_API_KEY`).
