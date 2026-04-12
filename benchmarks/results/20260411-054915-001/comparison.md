# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q8_0-retest`
- Model: `/Users/akiralam/code/ax-engine/models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 377.2 | 38.6 |
| llama.cpp | 1284.0 | 70.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 29.4% |
| AX / llama decode | 54.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-054915-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-054915-001/llama/summary.json`
