# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 495.5 | 495.3 |
| Decode | 41.1 | 41.0 |
