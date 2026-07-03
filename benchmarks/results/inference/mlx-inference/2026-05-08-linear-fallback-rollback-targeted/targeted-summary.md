# Linear Fallback Rollback Targeted Summary

- Output root: `benchmarks/results/mlx-inference/2026-05-08-linear-fallback-rollback-targeted`
- Baseline: `v4.4.2` README artifacts for the same model/prompt/policy rows.
- Change under test: removed the 4-step no-candidate early-disable heuristic while keeping direct-pipeline pending-token reuse for genuine fallback.

| Model | Prompt | Policy | v4.4.2 decode | Current decode | Delta | v4.4.2 claim | Current claim | Accepted tokens | Disabled steps |
|---|---:|---|---:|---:|---:|---|---|---:|---:|
| `qwen3-coder-next-4bit` | 128 | direct | 103.1 | 94.0 | -8.8% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3-coder-next-4bit` | 128 | ngram | 137.2 | 222.6 | 62.2% | `ngram_acceleration_effective_throughput` | `ngram_acceleration_effective_throughput` | 138 -> 294 | 0 |
| `qwen3-coder-next-4bit` | 512 | direct | 103.0 | 93.0 | -9.7% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3-coder-next-4bit` | 512 | ngram | 258.6 | 237.8 | -8.0% | `ngram_acceleration_effective_throughput` | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 |
| `qwen3_5-9b-mlx-4bit` | 128 | direct | 93.5 | 94.5 | 1.2% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3_5-9b-mlx-4bit` | 128 | ngram | 193.7 | 198.2 | 2.3% | `ngram_acceleration_effective_throughput` | `ngram_acceleration_effective_throughput` | 276 -> 276 | 0 |
| `qwen3_5-9b-mlx-4bit` | 512 | direct | 94.2 | 94.2 | -0.0% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3_5-9b-mlx-4bit` | 512 | ngram | 91.1 | 92.7 | 1.7% | `ngram_no_draft_direct_fallback` | `ngram_no_draft_direct_fallback` | 0 -> 0 | 333 |
| `qwen3_6-35b-a3b-5bit` | 128 | direct | 132.7 | 123.2 | -7.2% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3_6-35b-a3b-5bit` | 128 | ngram | 278.7 | 263.7 | -5.4% | `ngram_acceleration_effective_throughput` | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 |
| `qwen3_6-35b-a3b-5bit` | 512 | direct | 131.8 | 122.2 | -7.3% | `direct_same_policy_baseline` | `direct_same_policy_baseline` | 0 -> 0 | 0 |
| `qwen3_6-35b-a3b-5bit` | 512 | ngram | 275.1 | 260.9 | -5.2% | `ngram_acceleration_effective_throughput` | `ngram_acceleration_effective_throughput` | 300 -> 300 | 0 |

## Result

- Qwen3.5 9B prompt 128 recovered from fallback to effective n-gram acceleration: 198.2 tok/s vs v4.4.2 193.7 tok/s.
- Qwen3.6 35B A3B 5-bit recovered from fallback to effective n-gram acceleration on both prompt shapes: 263.7/260.9 tok/s.
- Qwen Coder Next 4-bit keeps the improved 128-token n-gram result: 222.6 tok/s vs v4.4.2 137.2 tok/s.
