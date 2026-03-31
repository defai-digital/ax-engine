# Apple-to-Apple Benchmark

- Label: `fa2-simd-default`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 695.2 | 67.5 |
| llama.cpp | 766.5 | 64.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 90.7% |
| AX / llama decode | 104.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090437-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090437-001/llama/summary.json`
