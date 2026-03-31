# Apple-to-Apple Benchmark

- Label: `fa2-simd-default`
- Model: `models/gemma-3-12b-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 428.7 | 41.6 |
| llama.cpp | 501.1 | 41.2 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 85.6% |
| AX / llama decode | 101.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-091246-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-091246-001/llama/summary.json`
