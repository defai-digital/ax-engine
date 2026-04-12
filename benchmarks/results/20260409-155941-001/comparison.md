# Apple-to-Apple Benchmark

- Label: `qwen3-coder-30b-a3b-q4_k_m-rerun`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 7.6 | 5.1 |
| llama.cpp | 1201.9 | 86.8 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.6% |
| AX / llama decode | 5.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-155941-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-155941-001/llama/summary.json`
