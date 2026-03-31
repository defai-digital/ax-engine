# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `64`
- Decode: `1` @ depth `64`
- Samples: `3`
- Cooldown: `10s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 499.5 | 491.5 |
| Decode | 45.3 | 45.3 |
