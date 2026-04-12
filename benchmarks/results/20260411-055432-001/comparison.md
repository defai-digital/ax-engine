# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q5_K_M-retest`
- Model: `/Users/akiralam/code/ax-engine/models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 924.4 | 36.1 |
| llama.cpp | 1151.4 | 79.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 80.3% |
| AX / llama decode | 45.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055432-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055432-001/llama/summary.json`
