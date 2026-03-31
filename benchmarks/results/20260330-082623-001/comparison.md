# Apple-to-Apple Benchmark

- Label: `smart-barrier`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 675.3 | 67.9 |
| llama.cpp | 763.6 | 64.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 88.4% |
| AX / llama decode | 105.1% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-082623-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-082623-001/llama/summary.json`
