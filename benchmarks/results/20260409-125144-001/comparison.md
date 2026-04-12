# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q5_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 6.3 | 4.4 |
| llama.cpp | 1151.4 | 79.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.6% |
| AX / llama decode | 5.6% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-125144-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-125144-001/llama/summary.json`
