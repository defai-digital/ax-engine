# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q4_K_M`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 6.4 | 4.7 |
| llama.cpp | 903.4 | 87.0 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.7% |
| AX / llama decode | 5.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-123702-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-123702-001/llama/summary.json`
