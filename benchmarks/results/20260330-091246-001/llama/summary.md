# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 501.1 | 499.6 |
| Decode | 41.2 | 41.1 |
