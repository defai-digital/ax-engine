# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q4_K_M-retest`
- Model: `/Users/akiralam/code/ax-engine/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 1104.0 | 69.8 |
| llama.cpp | 633.8 | 82.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 174.2% |
| AX / llama decode | 84.6% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055229-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055229-001/llama/summary.json`
