# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 481.7 | 480.5 |
| Decode | 42.5 | 42.8 |
