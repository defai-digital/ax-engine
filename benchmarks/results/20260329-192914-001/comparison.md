# Apple-to-Apple Benchmark

- Label: `bv128-merged-cb`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 316.0 | 55.1 |
| llama.cpp | 720.9 | 48.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 43.8% |
| AX / llama decode | 114.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-192914-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-192914-001/llama/summary.json`
