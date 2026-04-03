# Apple-to-Apple Benchmark

- Label: `llama31-8b`
- Model: `models/meta-llama-3.1-8b-instruct-q5_k_m.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 604.5 | 56.3 |
| llama.cpp | 705.2 | 55.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 85.7% |
| AX / llama decode | 101.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-214236-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260402-214236-001/llama/summary.json`
