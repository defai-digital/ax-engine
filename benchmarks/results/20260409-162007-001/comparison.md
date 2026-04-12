# Apple-to-Apple Benchmark

- Label: `qwen3-coder-30b-a3b-q5_k_m-ax-refresh`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 42.5 | 42.6 |
| llama.cpp | 1151.4 | 79.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 3.7% |
| AX / llama decode | 53.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162007-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162007-001/llama/summary.json`
