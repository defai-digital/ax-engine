# Apple-to-Apple Benchmark

- Label: `gemma4-26b-a4b-q5km`
- Model: `models/gemma-4-26B-A4B-it-Q5_K_M.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: process-per-sample outer medians on the same machine
- KV parity: AX f16=True llama f16=True

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 1148.8 | 66.3 |
| llama.cpp | 1021.4 | 60.5 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 112.5% |
| AX / llama decode | 109.6% |

<details><summary>Per-sample breakdown</summary>

| # | AX prefill | AX decode | llama prefill | llama decode |
|---|---:|---:|---:|---:|
| 1 | 74.0 | 70.1 | 890.0 | 56.3 |
| 2 | 1264.0 | 67.7 | 1023.0 | 59.9 |
| 3 | 1203.2 | 66.3 | 1021.4 | 60.5 |
| 4 | 1113.8 | 57.9 | 1026.8 | 66.2 |
| 5 | 1148.8 | 63.9 | 1020.3 | 66.2 |
</details>

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-124940-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-124940-001/llama/summary.json`
