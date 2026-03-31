# Apple-to-Apple Benchmark

- Label: `fa2-simd-default`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 512.6 | 44.7 |
| llama.cpp | 644.5 | 55.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 79.5% |
| AX / llama decode | 80.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090054-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090054-001/llama/summary.json`
