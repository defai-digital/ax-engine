# Apple-to-Apple Benchmark

- Label: `Qwen3-Coder-30B-A3B-Q6_K`
- Model: `models/Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 6.7 | 4.9 |
| llama.cpp | 1204.9 | 79.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.6% |
| AX / llama decode | 6.1% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-130636-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260409-130636-001/llama/summary.json`
