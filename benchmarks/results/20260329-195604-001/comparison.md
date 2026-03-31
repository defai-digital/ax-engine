# Apple-to-Apple Benchmark

- Label: `baseline-reuse-test`
- Model: `models/Qwen3.5-9B-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `1`
- Cooldown: `0s`
- Method: serial outer-sample medians on the same machine

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 326.0 | 54.8 |
| llama.cpp | 708.9 | 48.1 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 46.0% |
| AX / llama decode | 113.9% |

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-195604-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260329-195604-001/llama/summary.json`
