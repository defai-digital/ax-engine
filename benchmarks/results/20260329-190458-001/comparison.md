# Apple-to-Apple Benchmark

- Label: `ws1-ws2-ws4-e2e`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 319.8 | 53.9 |
| llama.cpp | 711.6 | 47.4 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 44.9% |
| AX / llama decode | 113.7% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-190458-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-190458-001/llama/summary.json`
