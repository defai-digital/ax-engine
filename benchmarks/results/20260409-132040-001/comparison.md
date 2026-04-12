# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q8_0`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 8.1 | 5.4 |
| llama.cpp | 1284.0 | 70.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.6% |
| AX / llama decode | 7.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-132040-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-132040-001/llama/summary.json`
