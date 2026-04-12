# Apple-to-Apple Benchmark

- Label: `qwen3-coder-30b-a3b-q8_0-ax-refresh`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 19.2 | 18.9 |
| llama.cpp | 1284.0 | 70.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 1.5% |
| AX / llama decode | 26.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162712-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-162712-001/llama/summary.json`
