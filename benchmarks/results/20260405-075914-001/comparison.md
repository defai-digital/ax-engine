# Apple-to-Apple Benchmark

- Label: `Qwen3.5-35B-A3B-Q4_K_M`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 693.4 | 45.4 |
| llama.cpp | 1122.3 | 53.2 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 61.8% |
| AX / llama decode | 85.3% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-075914-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260405-075914-001/llama/summary.json`
