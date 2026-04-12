# Apple-to-Apple Benchmark

- Label: `qwen35-35b-a3b`
- Model: `models/Qwen3.5-35B-A3B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `30s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 898.6 | 15.1 |
| llama.cpp | 1127.0 | 57.0 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 79.7% |
| AX / llama decode | 26.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-012123-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260404-012123-001/llama/summary.json`
