# Apple-to-Apple Benchmark

- Label: `qwen3-coder-30b-a3b-q4_k_m-ax-refresh`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 50.5 | 50.2 |
| llama.cpp | 903.4 | 87.0 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 5.6% |
| AX / llama decode | 57.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-161719-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-161719-001/llama/summary.json`
