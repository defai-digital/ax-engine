# Apple-to-Apple Benchmark

- Label: `verify-gdn-simd`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 426.2 | 54.2 |
| llama.cpp | 708.9 | 48.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 60.1% |
| AX / llama decode | 112.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-000711-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-000711-001/llama/summary.json`
