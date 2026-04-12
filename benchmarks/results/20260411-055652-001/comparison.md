# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q6_K-retest`
- Model: `/Users/akiralam/code/ax-engine/models/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 429.2 | 57.6 |
| llama.cpp | 1204.9 | 79.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 35.6% |
| AX / llama decode | 72.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055652-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260411-055652-001/llama/summary.json`
