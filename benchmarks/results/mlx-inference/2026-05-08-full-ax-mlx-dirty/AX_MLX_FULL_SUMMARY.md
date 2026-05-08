# AX Engine MLX Full Benchmark Refresh

- Output root: `benchmarks/results/mlx-inference/2026-05-08-full-ax-mlx-dirty`
- Reference rows reused from: `benchmarks/results/mlx-inference/2026-05-08-post-af0fcb2-readme-ax`
- Scope: README model set, prompt tokens 128 and 512, generation tokens 128, 3 measured repetitions per AX policy.
- Note: artifacts record commit `af0fcb24dd52a6abc8b167049b20af411eb0eec8`, but the server binary was built from a dirty worktree containing local AX MLX changes.

## Validation

Validation passed: 14/14 artifacts, each with 4 reused reference rows and 4 refreshed AX rows.

## Current AX Results

| Model | Prompt | Direct prefill | Direct decode | Direct decode vs mlx_lm | N-gram prefill | N-gram decode | N-gram decode vs mlx_lm | N-gram vs direct decode | Claim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gemma-4-26b-a4b-it-4bit` | 128 | 1118.3 | 107.9 | -8.8% | 1229.6 | 229.7 | 94.2% | 112.9% | `ngram_acceleration_effective_throughput` |
| `gemma-4-26b-a4b-it-4bit` | 512 | 2792.7 | 103.5 | -8.5% | 2848.0 | 201.7 | 78.3% | 94.9% | `ngram_acceleration_effective_throughput` |
| `gemma-4-31b-it-4bit` | 128 | 511.0 | 24.3 | -7.5% | 626.9 | 61.7 | 135.1% | 154.3% | `ngram_acceleration_effective_throughput` |
| `gemma-4-31b-it-4bit` | 512 | 642.3 | 23.5 | -5.6% | 742.0 | 60.1 | 141.2% | 155.5% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-4bit` | 128 | 3158.9 | 172.4 | -12.7% | 3744.0 | 514.2 | 160.4% | 198.3% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-4bit` | 512 | 7477.7 | 167.3 | -12.8% | 7922.3 | 508.8 | 165.1% | 204.1% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-5bit` | 128 | 3061.5 | 157.8 | -13.7% | 3622.0 | 404.1 | 120.9% | 156.1% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-5bit` | 512 | 7062.7 | 151.3 | -15.0% | 7712.0 | 399.8 | 124.5% | 164.2% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-6bit` | 128 | 3036.4 | 142.9 | -11.4% | 3605.9 | 384.8 | 138.6% | 169.3% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-6bit` | 512 | 6907.5 | 139.1 | -9.8% | 7652.4 | 381.9 | 147.6% | 174.5% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-8bit` | 128 | 3083.8 | 126.5 | -9.3% | 3567.4 | 407.3 | 192.2% | 222.1% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e2b-it-8bit` | 512 | 6867.0 | 122.6 | -8.9% | 7680.4 | 396.2 | 194.5% | 223.3% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e4b-it-4bit` | 128 | 2422.8 | 111.8 | -7.8% | 2813.7 | 328.7 | 170.9% | 193.8% | `ngram_acceleration_effective_throughput` |
| `gemma-4-e4b-it-4bit` | 512 | 4044.6 | 108.8 | -9.3% | 4323.5 | 310.9 | 159.2% | 185.9% | `ngram_acceleration_effective_throughput` |
| `glm-4.7-flash-4bit` | 128 | 820.1 | 92.5 | -0.5% | 881.1 | 257.9 | 177.3% | 178.7% | `ngram_acceleration_effective_throughput` |
| `glm-4.7-flash-4bit` | 512 | 2233.9 | 92.4 | 2.3% | 2404.0 | 254.8 | 181.9% | 175.7% | `ngram_acceleration_effective_throughput` |
| `qwen3-coder-next-4bit` | 128 | 837.1 | 93.0 | 0.9% | 892.3 | 219.7 | 138.5% | 136.3% | `ngram_acceleration_effective_throughput` |
| `qwen3-coder-next-4bit` | 512 | 2647.0 | 91.7 | 1.5% | 2788.2 | 238.3 | 163.7% | 159.7% | `ngram_acceleration_effective_throughput` |
| `qwen3_5-9b-mlx-4bit` | 128 | 1940.7 | 92.9 | -2.4% | 2191.4 | 93.3 | -2.0% | 0.4% | `ngram_no_draft_direct_fallback` |
| `qwen3_5-9b-mlx-4bit` | 512 | 2725.5 | 92.9 | -0.5% | 2860.6 | 92.9 | -0.5% | -0.0% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-5bit` | 128 | 974.7 | 120.5 | 3.2% | 1026.8 | 121.5 | 4.0% | 0.8% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-5bit` | 512 | 2493.4 | 121.3 | 6.7% | 2596.8 | 120.4 | 5.8% | -0.8% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-6bit` | 128 | 917.0 | 109.1 | 6.1% | 991.4 | 108.3 | 5.3% | -0.7% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-6bit` | 512 | 2347.4 | 108.1 | 7.0% | 2475.0 | 107.7 | 6.6% | -0.4% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-8bit` | 128 | 919.3 | 97.6 | 4.3% | 957.0 | 96.8 | 3.4% | -0.9% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-8bit` | 512 | 2318.3 | 97.8 | 7.0% | 2455.0 | 97.1 | 6.2% | -0.8% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 128 | 1001.3 | 112.3 | 4.4% | 1115.2 | 112.4 | 4.5% | 0.1% | `ngram_no_draft_direct_fallback` |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 512 | 2564.4 | 112.4 | 8.8% | 2703.4 | 110.9 | 7.4% | -1.3% | `ngram_no_draft_direct_fallback` |

## Delta Versus Previous AX Artifact

| Model | Prompt | Policy | Prefill delta | Decode delta |
|---|---:|---|---:|---:|
| `gemma-4-26b-a4b-it-4bit` | 128 | direct | -3.3% | -3.3% |
| `gemma-4-26b-a4b-it-4bit` | 128 | ngram | -2.8% | -6.0% |
| `gemma-4-26b-a4b-it-4bit` | 512 | direct | -1.0% | -5.0% |
| `gemma-4-26b-a4b-it-4bit` | 512 | ngram | -4.5% | -5.4% |
| `gemma-4-31b-it-4bit` | 128 | direct | -1.7% | -3.1% |
| `gemma-4-31b-it-4bit` | 128 | ngram | 3.0% | 0.4% |
| `gemma-4-31b-it-4bit` | 512 | direct | -4.7% | -3.4% |
| `gemma-4-31b-it-4bit` | 512 | ngram | 0.7% | 5.4% |
| `gemma-4-e2b-it-4bit` | 128 | direct | -2.2% | -0.8% |
| `gemma-4-e2b-it-4bit` | 128 | ngram | 0.6% | 0.2% |
| `gemma-4-e2b-it-4bit` | 512 | direct | 3.4% | -0.3% |
| `gemma-4-e2b-it-4bit` | 512 | ngram | -1.7% | 0.0% |
| `gemma-4-e2b-it-5bit` | 128 | direct | -0.9% | 0.8% |
| `gemma-4-e2b-it-5bit` | 128 | ngram | 0.5% | -0.3% |
| `gemma-4-e2b-it-5bit` | 512 | direct | 0.2% | 1.2% |
| `gemma-4-e2b-it-5bit` | 512 | ngram | 1.0% | -0.3% |
| `gemma-4-e2b-it-6bit` | 128 | direct | -0.7% | 0.9% |
| `gemma-4-e2b-it-6bit` | 128 | ngram | 2.7% | 2.8% |
| `gemma-4-e2b-it-6bit` | 512 | direct | -0.7% | 1.0% |
| `gemma-4-e2b-it-6bit` | 512 | ngram | -0.0% | 2.5% |
| `gemma-4-e2b-it-8bit` | 128 | direct | 0.2% | -4.7% |
| `gemma-4-e2b-it-8bit` | 128 | ngram | 0.2% | 0.6% |
| `gemma-4-e2b-it-8bit` | 512 | direct | 0.1% | -0.0% |
| `gemma-4-e2b-it-8bit` | 512 | ngram | 1.5% | -1.7% |
| `gemma-4-e4b-it-4bit` | 128 | direct | 1.1% | 1.3% |
| `gemma-4-e4b-it-4bit` | 128 | ngram | 0.1% | 1.0% |
| `gemma-4-e4b-it-4bit` | 512 | direct | -0.5% | 1.1% |
| `gemma-4-e4b-it-4bit` | 512 | ngram | -0.9% | -1.1% |
| `glm-4.7-flash-4bit` | 128 | direct | -0.4% | -0.9% |
| `glm-4.7-flash-4bit` | 128 | ngram | 0.2% | 1.2% |
| `glm-4.7-flash-4bit` | 512 | direct | 0.4% | 1.0% |
| `glm-4.7-flash-4bit` | 512 | ngram | 1.3% | 2.3% |
| `qwen3-coder-next-4bit` | 128 | direct | 0.2% | -0.6% |
| `qwen3-coder-next-4bit` | 128 | ngram | -0.6% | -0.6% |
| `qwen3-coder-next-4bit` | 512 | direct | 1.1% | 1.4% |
| `qwen3-coder-next-4bit` | 512 | ngram | -2.0% | -2.9% |
| `qwen3_5-9b-mlx-4bit` | 128 | direct | -0.2% | 1.4% |
| `qwen3_5-9b-mlx-4bit` | 128 | ngram | 1.0% | -52.6% |
| `qwen3_5-9b-mlx-4bit` | 512 | direct | -0.0% | 1.7% |
| `qwen3_5-9b-mlx-4bit` | 512 | ngram | -4.8% | 74.8% |
| `qwen3_6-35b-a3b-5bit` | 128 | direct | -1.8% | -11.3% |
| `qwen3_6-35b-a3b-5bit` | 128 | ngram | -0.2% | -53.9% |
| `qwen3_6-35b-a3b-5bit` | 512 | direct | -4.7% | -9.8% |
| `qwen3_6-35b-a3b-5bit` | 512 | ngram | 0.6% | -54.0% |
| `qwen3_6-35b-a3b-6bit` | 128 | direct | 0.1% | -0.0% |
| `qwen3_6-35b-a3b-6bit` | 128 | ngram | 2.7% | -55.2% |
| `qwen3_6-35b-a3b-6bit` | 512 | direct | -0.1% | 0.1% |
| `qwen3_6-35b-a3b-6bit` | 512 | ngram | -0.3% | -54.8% |
| `qwen3_6-35b-a3b-8bit` | 128 | direct | 0.8% | 0.8% |
| `qwen3_6-35b-a3b-8bit` | 128 | ngram | -2.3% | -60.0% |
| `qwen3_6-35b-a3b-8bit` | 512 | direct | -0.8% | 2.0% |
| `qwen3_6-35b-a3b-8bit` | 512 | ngram | 0.3% | -59.2% |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 128 | direct | -7.4% | -6.7% |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 128 | ngram | -2.7% | -59.1% |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 512 | direct | -4.1% | -6.7% |
| `qwen3_6-35b-a3b-ud-mlx-4bit` | 512 | ngram | -4.7% | -60.3% |

## Telemetry Notes

- Effective n-gram acceleration rows report `ngram_acceleration_effective_throughput` and non-zero accepted tokens.
- No-draft rows report `ngram_no_draft_direct_fallback`; these exercised the request-level fallback path and should be interpreted as correctness/regression guardrails, not acceleration claims.
- Full per-trial telemetry remains in each model JSON artifact; this summary uses medians for the table.
