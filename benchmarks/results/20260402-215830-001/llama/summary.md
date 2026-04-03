# llama.cpp Serial Median Benchmark

- Engine: `llama.cpp`
- Binary: `/opt/homebrew/bin/llama-bench`
- Model: `models/meta-llama-3-70b-instruct.Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 57.1 | 57.9 |
| Decode | 5.6 | 5.6 |
