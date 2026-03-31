# Apple-to-Apple Benchmark

- Label: `qwen35-9b`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 316.2 | 54.0 |
| llama.cpp | 708.7 | 47.7 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 44.6% |
| AX / llama decode | 113.4% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-175435-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-175435-001/llama/summary.json`
