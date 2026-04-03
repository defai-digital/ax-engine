# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 1169.5 | 1140.6 |
| Decode | 58.4 | 57.7 |
