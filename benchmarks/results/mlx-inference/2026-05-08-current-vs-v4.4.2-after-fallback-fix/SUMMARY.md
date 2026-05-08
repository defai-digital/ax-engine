# Current vs v4.4.x README Artifact Benchmark

- Current output root: `benchmarks/results/mlx-inference/2026-05-08-current-vs-v4.4.2-after-fallback-fix`
- Baseline: `benchmarks/results/mlx-inference/2026-05-07-v4.4.1-readme-refresh` (README calls this v4.4.x; artifact build commits are recorded in each JSON, mostly `b63c0d9` / v4.4.1, with Gemma E4B from `603b66f`).
- Scope: 14 README models, prompt tokens 128 and 512, generation=128, temperature=0, AX direct + AX default n-gram rows, 3 measured repetitions plus 1 warmup.
- References: `mlx_lm` / `mlx_swift_lm` rows were reused from the baseline artifacts; only AX rows were refreshed.

## Summary

- Compared AX rows: 56 (28 direct, 28 n-gram).
- Decode regressions <= -10% vs README artifact baseline: 15 rows (6 direct, 9 n-gram).
- N-gram status flips: 0 rows.
- Current effective n-gram rows: 27/28.
- N-gram rows with >= +10% decode gain vs baseline: 1 rows.

## Key Takeaways

- The earlier Qwen-family catastrophic fallback regression is fixed: Qwen3.6 4/5/6/8-bit rows now keep accepted n-gram tokens instead of falling back with zero accepted tokens.
- Qwen Coder Next 4-bit prompt 128 keeps the intended tuning win: current n-gram decode remains far above the README artifact baseline.
- Direct baseline remains broadly slower on Gemma E2B/E4B and Qwen Coder, which is separate from the removed fallback heuristic.
- Several effective n-gram rows are still below the older README artifact baseline, mostly because their direct/model step time is lower in this run, not because n-gram disabled itself.

## All AX Rows

| Model | Prompt | Policy | v4.4.x decode | Current decode | Delta | Current claim | Accepted tokens | Disabled steps |
|---|---:|---|---:|---:|---:|---|---:|---:|
| `gemma-4-e2b-it-4bit` | 128 | direct | 185.6 | 173.9 | -6.3% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-4bit` | 512 | direct | 180.2 | 164.6 | -8.6% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-4bit` | 128 | ngram | 572.1 | 511.2 | -10.6% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-4bit` | 512 | ngram | 566.2 | 508.8 | -10.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 128 | direct | 170.3 | 156.9 | -7.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 512 | direct | 163.7 | 151.4 | -7.5% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 128 | ngram | 451.8 | 406.0 | -10.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 512 | ngram | 448.5 | 396.1 | -11.7% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 128 | direct | 153.6 | 143.3 | -6.8% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 512 | direct | 148.2 | 135.6 | -8.5% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 128 | ngram | 421.3 | 377.6 | -10.4% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 512 | ngram | 417.3 | 367.1 | -12.0% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-8bit` | 128 | direct | 136.8 | 126.5 | -7.6% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-8bit` | 512 | direct | 132.7 | 121.6 | -8.4% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-8bit` | 128 | ngram | 453.3 | 394.0 | -13.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-8bit` | 512 | ngram | 449.9 | 385.1 | -14.4% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e4b-it-4bit` | 128 | direct | 114.1 | 111.2 | -2.5% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e4b-it-4bit` | 512 | direct | 106.7 | 107.8 | +1.0% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e4b-it-4bit` | 128 | ngram | 318.4 | 322.5 | +1.3% | `ngram_acceleration_effective_throughput` | 324 -> 318 | 0 -> 0 |
| `gemma-4-e4b-it-4bit` | 512 | ngram | 322.1 | 312.5 | -3.0% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-26b-a4b-it-4bit` | 128 | direct | 119.3 | 111.1 | -6.8% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-26b-a4b-it-4bit` | 512 | direct | 116.5 | 107.9 | -7.4% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-26b-a4b-it-4bit` | 128 | ngram | 270.1 | 242.4 | -10.2% | `ngram_acceleration_effective_throughput` | 321 -> 318 | 0 -> 0 |
| `gemma-4-26b-a4b-it-4bit` | 512 | ngram | 220.7 | 211.7 | -4.1% | `ngram_acceleration_effective_throughput` | 288 -> 291 | 0 -> 0 |
| `gemma-4-31b-it-4bit` | 128 | direct | 27.2 | 25.0 | -8.3% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-31b-it-4bit` | 512 | direct | 26.4 | 24.2 | -8.1% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-31b-it-4bit` | 128 | ngram | 64.5 | 60.5 | -6.2% | `ngram_acceleration_effective_throughput` | 324 -> 318 | 0 -> 0 |
| `gemma-4-31b-it-4bit` | 512 | ngram | 62.8 | 59.0 | -6.0% | `ngram_acceleration_effective_throughput` | 324 -> 318 | 0 -> 0 |
| `qwen3_5-9b-mlx-4bit` | 128 | direct | 93.5 | 91.8 | -1.8% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_5-9b-mlx-4bit` | 512 | direct | 94.2 | 91.5 | -2.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_5-9b-mlx-4bit` | 128 | ngram | 193.7 | 195.4 | +0.9% | `ngram_acceleration_effective_throughput` | 276 -> 276 | 0 -> 0 |
| `qwen3_5-9b-mlx-4bit` | 512 | ngram | 91.1 | 89.4 | -1.9% | `ngram_no_draft_direct_fallback` | 0 -> 0 | 333 -> 333 |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 128 | direct | 121.1 | 110.2 | -9.0% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 512 | direct | 120.2 | 109.4 | -9.0% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 128 | ngram | 278.8 | 257.8 | -7.5% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 512 | ngram | 275.3 | 257.1 | -6.6% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-5bit` | 128 | direct | 132.7 | 120.9 | -8.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-5bit` | 512 | direct | 131.8 | 120.7 | -8.4% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-5bit` | 128 | ngram | 278.7 | 262.1 | -5.9% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-5bit` | 512 | ngram | 275.1 | 259.0 | -5.9% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-6bit` | 128 | direct | 118.9 | 107.2 | -9.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-6bit` | 512 | direct | 118.3 | 106.3 | -10.1% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-6bit` | 128 | ngram | 257.9 | 239.8 | -7.0% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-6bit` | 512 | ngram | 254.1 | 236.8 | -6.8% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-8bit` | 128 | direct | 107.3 | 96.5 | -10.1% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-8bit` | 512 | direct | 105.5 | 95.3 | -9.6% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-8bit` | 128 | ngram | 258.3 | 239.1 | -7.4% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3_6-35b-a3b-8bit` | 512 | ngram | 257.8 | 235.4 | -8.7% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `qwen3-coder-next-4bit` | 128 | direct | 103.1 | 90.0 | -12.7% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3-coder-next-4bit` | 512 | direct | 103.0 | 91.7 | -10.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3-coder-next-4bit` | 128 | ngram | 137.2 | 219.3 | +59.8% | `ngram_acceleration_effective_throughput` | 138 -> 294 | 135 -> 0 |
| `qwen3-coder-next-4bit` | 512 | ngram | 258.6 | 241.9 | -6.5% | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 128 | direct | 104.9 | 92.1 | -12.2% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 512 | direct | 103.7 | 90.9 | -12.3% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 128 | ngram | 280.3 | 258.7 | -7.7% | `ngram_acceleration_effective_throughput` | 324 -> 318 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 512 | ngram | 274.9 | 251.2 | -8.6% | `ngram_acceleration_effective_throughput` | 324 -> 318 | 0 -> 0 |

## Decode Regressions <= -10%

| Model | Prompt | Policy | v4.4.x decode | Current decode | Delta | Current claim | Accepted tokens | Disabled steps |
|---|---:|---|---:|---:|---:|---|---:|---:|
| `gemma-4-e2b-it-8bit` | 512 | ngram | 449.9 | 385.1 | -14.4% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-8bit` | 128 | ngram | 453.3 | 394.0 | -13.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `qwen3-coder-next-4bit` | 128 | direct | 103.1 | 90.0 | -12.7% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 512 | direct | 103.7 | 90.9 | -12.3% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `glm-4.7-flash-4bit` | 128 | direct | 104.9 | 92.1 | -12.2% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 512 | ngram | 417.3 | 367.1 | -12.0% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 512 | ngram | 448.5 | 396.1 | -11.7% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `qwen3-coder-next-4bit` | 512 | direct | 103.0 | 91.7 | -10.9% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `gemma-4-e2b-it-4bit` | 128 | ngram | 572.1 | 511.2 | -10.6% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-6bit` | 128 | ngram | 421.3 | 377.6 | -10.4% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-26b-a4b-it-4bit` | 128 | ngram | 270.1 | 242.4 | -10.2% | `ngram_acceleration_effective_throughput` | 321 -> 318 | 0 -> 0 |
| `gemma-4-e2b-it-5bit` | 128 | ngram | 451.8 | 406.0 | -10.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `gemma-4-e2b-it-4bit` | 512 | ngram | 566.2 | 508.8 | -10.1% | `ngram_acceleration_effective_throughput` | 324 -> 336 | 0 -> 0 |
| `qwen3_6-35b-a3b-8bit` | 128 | direct | 107.3 | 96.5 | -10.1% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |
| `qwen3_6-35b-a3b-6bit` | 512 | direct | 118.3 | 106.3 | -10.1% | `direct_same_policy_baseline` | 0 -> 0 | 0 -> 0 |

## N-gram Status Flips

None.
