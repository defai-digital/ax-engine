# Apple-to-Apple Benchmark

- Label: `prd-alignment`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 323.7 | 53.7 |
| llama.cpp | 708.9 | 48.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 45.7% |
| AX / llama decode | 111.6% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-200352-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-200352-001/llama/summary.json`
