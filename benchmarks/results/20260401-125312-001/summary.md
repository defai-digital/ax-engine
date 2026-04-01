# llama.cpp Benchmark (llama-only)

- Label: `qwen3-32b-llama-only`
- Model: `models/Qwen3-32B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 160.4 | 160.0 |
| Decode | 16.2 | 16.7 |

- Prefill samples: `[161.523728, 158.12034, 160.418563]`
- Decode samples: `[17.722754, 16.232759, 16.240135]`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-125312-001/llama/summary.json`
