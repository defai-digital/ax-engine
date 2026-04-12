# Apple-to-Apple Benchmark

- Label: `gemma4-26b-a4b-q4km`
- Model: `models/gemma-4-26B-A4B-it-Q4_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: process-per-sample outer medians on the same machine
- KV parity: AX f16=True llama f16=True

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 1283.0 | 76.7 |
| llama.cpp | 1165.7 | 73.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 110.1% |
| AX / llama decode | 104.7% |

<details><summary>Per-sample breakdown</summary>

| # | AX prefill | AX decode | llama prefill | llama decode |
|---|---:|---:|---:|---:|
| 1 | 83.1 | 77.1 | 757.5 | 69.1 |
| 2 | 1222.2 | 74.1 | 1178.5 | 73.3 |
| 3 | 1307.6 | 76.7 | 1164.3 | 73.3 |
| 4 | 1306.8 | 76.6 | 1165.7 | 73.7 |
| 5 | 1283.0 | 76.9 | 1187.7 | 72.7 |
</details>

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-123932-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-123932-001/llama/summary.json`
