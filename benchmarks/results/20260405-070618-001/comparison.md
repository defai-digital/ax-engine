# Apple-to-Apple Benchmark

- Label: `gemma-4-31B-it-Q4_K_M`
- Model: `models/gemma-4-31B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 149.8 | 11.0 |
| llama.cpp | 91.0 | 11.9 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 164.7% |
| AX / llama decode | 92.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-070618-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-070618-001/llama/summary.json`
