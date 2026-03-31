# Apple-to-Apple Benchmark

- Label: `fa2-simd-default`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 358.7 | 26.6 |
| llama.cpp | 433.5 | 42.9 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 82.7% |
| AX / llama decode | 61.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-085854-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-085854-001/llama/summary.json`
