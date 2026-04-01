# Apple-to-Apple Benchmark

- Label: `qwen35-27b`
- Model: `models/Qwen3.5-27B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 191.4 | 17.4 |
| llama.cpp | 209.4 | 17.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 91.4% |
| AX / llama decode | 99.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-121216-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260401-121216-001/llama/summary.json`
