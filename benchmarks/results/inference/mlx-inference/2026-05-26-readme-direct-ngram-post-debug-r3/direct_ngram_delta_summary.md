# README Direct vs N-gram Post-debug Summary

Source artifacts: `benchmarks/results/mlx-inference/2026-05-26-readme-direct-ngram-post-debug-r3/`.
Configuration: prompt tokens 128/512/2048, generation 128, repetitions 3, cooldown 0, prefix cache disabled, `mlx_lm` rows reused only as reference metadata; deltas below compare AX n-gram vs AX direct.

| Model | Prompt | Prefill Δ | Decode Δ | TTFT Δ | Direct decode | N-gram decode | Status | Accepted/bonus |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| gemma-4-e2b-it-4bit | 128 | +11.3% | -23.7% | -10.1% | 231.6 | 176.8 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-4bit | 512 | +7.7% | -21.9% | -7.1% | 221.2 | 172.7 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e2b-it-4bit | 2048 | +4.7% | -21.1% | -4.4% | 210.6 | 166.2 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e2b-it-5bit | 128 | +14.8% | -21.2% | -12.9% | 205.6 | 161.9 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e2b-it-5bit | 512 | +9.7% | -20.5% | -8.9% | 197.6 | 157.1 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e2b-it-5bit | 2048 | +5.7% | -21.0% | -5.4% | 188.8 | 149.1 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-6bit | 128 | +12.7% | -19.7% | -11.3% | 182.3 | 146.5 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-6bit | 512 | +9.8% | -19.6% | -8.9% | 176.8 | 142.1 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-6bit | 2048 | +5.5% | -18.7% | -5.2% | 169.4 | 137.7 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e2b-it-8bit | 128 | +9.8% | -19.0% | -8.9% | 159.3 | 129.1 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-8bit | 512 | +13.1% | -17.0% | -11.6% | 155.1 | 128.7 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e2b-it-8bit | 2048 | +6.9% | -17.0% | -6.5% | 150.4 | 124.9 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e4b-it-4bit | 128 | +12.6% | -16.4% | -11.2% | 141.6 | 118.3 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-e4b-it-4bit | 512 | +7.0% | -18.1% | -6.5% | 138.3 | 113.2 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-e4b-it-4bit | 2048 | +2.5% | -16.4% | -2.4% | 135.2 | 113.0 | `ngram_no_draft_direct_fallback` | 0/0 |
| gemma-4-26b-a4b-it-4bit | 128 | +10.5% | -23.6% | -9.5% | 132.4 | 101.2 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-26b-a4b-it-4bit | 512 | +6.2% | -18.8% | -5.8% | 129.2 | 104.9 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-26b-a4b-it-4bit | 2048 | +0.4% | -21.8% | -0.4% | 124.4 | 97.3 | `ngram_no_accept_fallback` | 0/0 |
| gemma-4-31b-it-4bit | 128 | +15.8% | -12.8% | -13.6% | 27.5 | 23.9 | `ngram_acceleration_effective_throughput` | 18/18 |
| gemma-4-31b-it-4bit | 512 | +2.7% | -18.3% | -2.6% | 26.9 | 22.0 | `ngram_acceleration_effective_throughput` | 3/3 |
| gemma-4-31b-it-4bit | 2048 | +4.4% | -20.3% | -4.3% | 26.6 | 21.2 | `ngram_acceleration_effective_throughput` | 6/6 |
| qwen3_6-27b-4bit | 128 | -3.0% | -3.2% | +3.1% | 33.8 | 32.7 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-4bit | 512 | -10.2% | +0.7% | +11.4% | 33.8 | 34.0 | `ngram_acceleration_effective_throughput` | 3/3 |
| qwen3_6-27b-4bit | 2048 | -5.0% | +0.3% | +5.3% | 33.8 | 33.9 | `ngram_acceleration_effective_throughput` | 3/3 |
| qwen3_6-27b-5bit | 128 | -4.0% | -2.9% | +4.1% | 27.8 | 27.0 | `ngram_acceleration_effective_throughput` | 6/6 |
| qwen3_6-27b-5bit | 512 | -0.7% | -0.1% | +0.8% | 28.0 | 28.0 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-5bit | 2048 | -2.4% | +0.4% | +2.5% | 27.7 | 27.8 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-6bit | 128 | +2.6% | -0.9% | -2.4% | 24.0 | 23.8 | `ngram_no_accept_fallback` | 0/0 |
| qwen3_6-27b-6bit | 512 | +3.5% | +0.4% | -3.4% | 24.0 | 24.1 | `ngram_acceleration_effective_throughput` | 6/6 |
| qwen3_6-27b-6bit | 2048 | +1.2% | -0.6% | -1.2% | 23.8 | 23.6 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-8bit | 128 | -0.8% | -1.7% | +0.8% | 18.8 | 18.5 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-8bit | 512 | +1.0% | -0.3% | -1.0% | 18.7 | 18.7 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-27b-8bit | 2048 | +1.6% | -1.3% | -1.5% | 18.6 | 18.4 | `ngram_no_draft_direct_fallback` | 0/0 |
| qwen3_6-35b-a3b-4bit | 128 | +0.9% | +111.6% | -0.9% | 162.4 | 343.7 | `ngram_acceleration_effective_throughput` | 300/291 |
| qwen3_6-35b-a3b-4bit | 512 | -2.1% | +78.4% | +2.1% | 160.3 | 285.9 | `ngram_acceleration_effective_throughput` | 300/294 |
| qwen3_6-35b-a3b-4bit | 2048 | -4.7% | +86.3% | +4.9% | 152.7 | 284.4 | `ngram_acceleration_effective_throughput` | 300/288 |
