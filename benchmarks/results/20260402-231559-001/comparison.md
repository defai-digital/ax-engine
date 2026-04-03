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
| AX Engine | 4.5 | 4.5 |
| llama.cpp | 798.6 | 53.2 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 0.6% |
| AX / llama decode | 8.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-231559-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-231559-001/llama/summary.json`
