# Apple-to-Apple Benchmark

- Label: `gemma-4-26B-A4B-it-Q4_K_M`
- Model: `models/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 9.4 | 12.3 |
| llama.cpp | 1176.6 | 72.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.8% |
| AX / llama decode | 17.1% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260407-203605-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260407-203605-001/llama/summary.json`
