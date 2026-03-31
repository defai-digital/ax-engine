# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 477.0 | 476.5 |
| Decode | 39.3 | 39.1 |
