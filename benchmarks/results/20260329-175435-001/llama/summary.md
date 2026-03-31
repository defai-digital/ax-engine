# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 708.7 | 705.1 |
| Decode | 47.7 | 48.0 |
