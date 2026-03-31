# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 773.6 | 769.9 |
| Decode | 66.9 | 66.9 |
