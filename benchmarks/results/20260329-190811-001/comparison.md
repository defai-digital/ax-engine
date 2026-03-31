# Apple-to-Apple Benchmark

- Label: `ws1-cliff-pp64`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `64`
- Decode: `1` @ depth `64`
- Samples: `3`
- Cooldown: `10s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 188.4 | 40.8 |
| llama.cpp | 499.5 | 45.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 37.7% |
| AX / llama decode | 90.0% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-190811-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-190811-001/llama/summary.json`
