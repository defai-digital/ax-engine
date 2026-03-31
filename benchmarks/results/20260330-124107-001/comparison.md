# Apple-to-Apple Benchmark

- Label: `codex-qwen35-9b-20260330`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 411.2 | 51.8 |
| llama.cpp | 665.7 | 44.8 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 61.8% |
| AX / llama decode | 115.5% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-124107-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-124107-001/llama/summary.json`
