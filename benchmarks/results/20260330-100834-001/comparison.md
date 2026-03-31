# Apple-to-Apple Benchmark

- Label: `prd-final`
- Model: `models/Qwen3-8B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 693.6 | 68.4 |
| llama.cpp | 768.2 | 64.6 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 90.3% |
| AX / llama decode | 105.8% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-100834-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260330-100834-001/llama/summary.json`
