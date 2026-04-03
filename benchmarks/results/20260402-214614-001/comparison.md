# Apple-to-Apple Benchmark

- Label: `llama3-70b`
- Model: `models/meta-llama-3-70b-instruct.Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 49.5 | 8.4 |
| llama.cpp | 53.9 | 4.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 91.9% |
| AX / llama decode | 182.2% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-214614-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-214614-001/llama/summary.json`
