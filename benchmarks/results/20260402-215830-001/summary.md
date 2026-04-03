# llama.cpp Benchmark (llama-only)

- Label: `llama3-70b-llama-only`
- Model: `models/meta-llama-3-70b-instruct.Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`

| Phase | Median tok/s | Mean tok/s |
|---|---:|---:|
| Prefill | 57.1 | 57.9 |
| Decode | 5.6 | 5.6 |

- Prefill samples: `[54.424431, 57.052428, 62.318627]`
- Decode samples: `[5.482902, 5.625265, 5.613683]`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-215830-001/llama/summary.json`
