# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 1176.6 | 1019.5 |
| Decode | 72.1 | 70.8 |
