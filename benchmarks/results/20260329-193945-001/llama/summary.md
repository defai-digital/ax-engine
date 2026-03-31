# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 708.9 | 710.3 |
| Decode | 48.1 | 47.8 |
