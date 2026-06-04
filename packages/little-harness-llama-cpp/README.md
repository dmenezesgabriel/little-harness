# little-harness-llama-cpp

A [little-harness](https://github.com/dmenezesgabriel/little-harness) chat-model
provider plugin that runs a local GGUF model via
[llama.cpp](https://github.com/abetlen/llama-cpp-python). It implements the core
`ChatModel` port (`complete_streaming` + `close`) and registers a `llama_cpp`
provider under the `little_harness.chat_model_providers` entry-point group; the
vendor SDK stays sealed inside this package.

```
uv pip install "little-harness[llama-cpp]"
little-harness --provider llama_cpp -o model_path=models/model.gguf -p "2 + 2?"
```

Options: `model_path`, `n_ctx`, `n_threads`, `n_gpu_layers`.
