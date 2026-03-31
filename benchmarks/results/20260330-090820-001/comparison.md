# Apple-to-Apple Benchmark

- Label: `fa2-simd-default`
- Model: `models/Llama-3-8B-Instruct-GGUF-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 716.2 | 60.2 |
| llama.cpp | 773.6 | 66.9 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 92.6% |
| AX / llama decode | 90.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090820-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-090820-001/llama/summary.json`
