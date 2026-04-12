# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 150.4 | 150.4 |
| Decode | 15.8 | 15.8 |
