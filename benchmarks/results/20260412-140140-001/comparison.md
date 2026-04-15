# Apple-to-Apple Benchmark

- Label: `gemma4-26b-a4b-q6k-post-runtime-prepare`
- Model: `models/gemma-4-26B-A4B-it-Q6_K.gguf`
- Prompt: `512`
- Decode: `128` @ depth `512`
- Samples: `5`
- Cooldown: `20s`
- Method: process-per-sample outer medians on the same machine
- KV parity: AX f16=True llama f16=True

| Engine | Prefill tok/s | Decode tok/s |
|---|---:|---:|
| AX Engine | 1314.6 | 70.3 |
| llama.cpp | 1177.6 | 65.3 |

| Ratio | Value |
|---|---:|
| AX / llama prefill | 111.6% |
| AX / llama decode | 107.7% |

<details><summary>Per-sample breakdown</summary>

| # | AX prefill | AX decode | llama prefill | llama decode |
|---|---:|---:|---:|---:|
| 1 | 1314.6 | 68.9 | 1030.1 | 65.9 |
| 2 | 1300.6 | 70.2 | 1198.2 | 56.7 |
| 3 | 1308.4 | 70.3 | 1148.6 | 54.3 |
| 4 | 1331.0 | 72.1 | 1177.6 | 65.4 |
| 5 | 1341.1 | 71.7 | 1184.5 | 65.3 |
</details>

- AX artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-140140-001/ax.json`
- llama.cpp artifact: `/Users/akiralam/code/ax-engine/benchmarks/results/20260412-140140-001/llama/summary.json`
