# AX MLX prefill-stage profile baseline

Per-stage prefill wall-time breakdown for the 14 README models at prompt=4096 / generation=128. Captured with `--ax-prefill-profile` and `AX_MLX_PREFILL_PROFILE=1`. Stages match the `ax_mlx_prefill_profile_*` SSE telemetry keys; sub-stages are indented under their umbrella parent. `%-of-forward` is the share of `ax_mlx_prefill_forward_wall_us`. `%-of-parent` is the share of the immediate umbrella stage.

Source: `benchmarks/results/mlx-inference/2026-05-15-prefill-profile-baseline/`.

## Cross-model hot-stage summary

| Model | Forward (µs) | FFN total % | SDPA % | pre-SDPA % | post-attn-output % | per-layer-input % |
|---|---:|---:|---:|---:|---:|---:|
| .internal/models/gemma-4-26b-a4b-it-4bit | 1,173,865 | 53.8% | 19.9% | 16.4% | 6.8% | — |
| .internal/models/gemma-4-31b-it-4bit | 7,120,717 | 59.8% | 13.0% | 17.2% | 8.2% | — |
| .internal/models/gemma-4-e2b-it-4bit | 723,271 | 57.9% | 20.4% | 8.9% | 5.6% | 0.8% |
| .internal/models/gemma-4-e2b-it-5bit | 766,240 | 59.3% | 19.2% | 8.5% | 5.7% | 0.8% |
| .internal/models/gemma-4-e2b-it-6bit | 753,456 | 58.5% | 19.4% | 8.6% | 5.9% | 0.7% |
| .internal/models/gemma-4-e2b-it-8bit | 716,547 | 58.6% | 19.8% | 8.8% | 5.5% | 0.8% |
| .internal/models/gemma-4-e4b-it-4bit | 1,088,892 | 63.7% | 15.1% | 9.7% | 5.3% | 0.7% |
| .internal/models/GLM-4.7-Flash-4bit | 1,520,751 | 43.3% | — | — | — | — |
| .internal/models/Qwen3-Coder-Next-4bit | 2,532,292 | 65.6% | 3.1% | 4.7% | 1.6% | — |
| .internal/models/Qwen3.5-9B-MLX-4bit | 1,713,868 | 54.2% | 3.1% | 6.3% | 2.2% | — |
| .internal/models/Qwen3.6-35B-A3B-5bit | 1,830,732 | 60.6% | 3.4% | 5.5% | 1.9% | — |
| .internal/models/Qwen3.6-35B-A3B-6bit | 1,875,666 | 60.3% | 3.4% | 5.5% | 1.9% | — |
| .internal/models/Qwen3.6-35B-A3B-8bit | 1,878,412 | 61.5% | 3.3% | 5.2% | 1.8% | — |
| .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit | 1,774,440 | 58.7% | 3.5% | 5.6% | 2.0% | — |

## Per-model breakdown

### .internal/models/gemma-4-26b-a4b-it-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,173,865 µs** — 2 prefill step(s), 60 layer-passes, 4096 tokens, median AX direct prefill = 3,487.5 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 192,820 | 16.4% | — |
|   QKV projection | 117,917 | 10.0% | 61.2% |
|   QK norm | 17,554 | 1.5% | 9.1% |
|   RoPE + KV append | 56,686 | 4.8% | 29.4% |
| SDPA | 233,597 | 19.9% | — |
| post-attn (umbrella) | 741,917 | 63.2% | — |
|   output projection | 80,191 | 6.8% | 10.8% |
|   residual norm | 15,524 | 1.3% | 2.1% |
|   residual gate | 14,021 | 1.2% | 1.9% |
|   FFN (umbrella) | 631,902 | 53.8% | 85.2% |
|     FFN gate+up | 65,353 | 5.6% | 10.3% |
|     FFN activation | 24,403 | 2.1% | 3.9% |
|     FFN down | 37,794 | 3.2% | 6.0% |
| LM head | 1,952 | 0.2% | — |


### .internal/models/gemma-4-31b-it-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **7,120,717 µs** — 2 prefill step(s), 120 layer-passes, 4096 tokens, median AX direct prefill = 575.2 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 1,225,338 | 17.2% | — |
|   QKV projection | 966,182 | 13.6% | 78.9% |
|   QK norm | 55,543 | 0.8% | 4.5% |
|   RoPE + KV append | 202,524 | 2.8% | 16.5% |
| SDPA | 924,020 | 13.0% | — |
| post-attn (umbrella) | 4,940,016 | 69.4% | — |
|   output projection | 583,838 | 8.2% | 11.8% |
|   residual norm | 45,997 | 0.6% | 0.9% |
|   residual gate | 41,676 | 0.6% | 0.8% |
|   FFN (umbrella) | 4,259,244 | 59.8% | 86.2% |
|     FFN gate+up | 2,343,012 | 32.9% | 55.0% |
|     FFN activation | 537,610 | 7.5% | 12.6% |
|     FFN down | 1,325,348 | 18.6% | 31.1% |
| LM head | 21,476 | 0.3% | — |


### .internal/models/gemma-4-e2b-it-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **723,271 µs** — 2 prefill step(s), 70 layer-passes, 4096 tokens, median AX direct prefill = 5,658.1 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | 5,652 | 0.8% | — |
| pre-SDPA (umbrella) | 64,421 | 8.9% | — |
|   QKV projection | 18,176 | 2.5% | 28.2% |
|   QK norm | 28,099 | 3.9% | 43.6% |
|   RoPE + KV append | 17,771 | 2.5% | 27.6% |
| SDPA | 147,291 | 20.4% | — |
| post-attn (umbrella) | 502,090 | 69.4% | — |
|   output projection | 40,511 | 5.6% | 8.1% |
|   residual norm | 14,931 | 2.1% | 3.0% |
|   residual gate | 27,762 | 3.8% | 5.5% |
|   FFN (umbrella) | 418,684 | 57.9% | 83.4% |
|     FFN gate+up | 167,508 | 23.2% | 40.0% |
|     FFN activation | 134,975 | 18.7% | 32.2% |
|     FFN down | 101,759 | 14.1% | 24.3% |
| LM head | 1,482 | 0.2% | — |


### .internal/models/gemma-4-e2b-it-5bit @ prompt=4096, generation=128

Forward wall (median across trials): **766,240 µs** — 2 prefill step(s), 70 layer-passes, 4096 tokens, median AX direct prefill = 5,341.4 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | 6,091 | 0.8% | — |
| pre-SDPA (umbrella) | 64,876 | 8.5% | — |
|   QKV projection | 18,488 | 2.4% | 28.5% |
|   QK norm | 28,618 | 3.7% | 44.1% |
|   RoPE + KV append | 17,940 | 2.3% | 27.7% |
| SDPA | 147,282 | 19.2% | — |
| post-attn (umbrella) | 544,574 | 71.1% | — |
|   output projection | 43,381 | 5.7% | 8.0% |
|   residual norm | 15,260 | 2.0% | 2.8% |
|   residual gate | 30,539 | 4.0% | 5.6% |
|   FFN (umbrella) | 454,551 | 59.3% | 83.5% |
|     FFN gate+up | 183,001 | 23.9% | 40.3% |
|     FFN activation | 141,846 | 18.5% | 31.2% |
|     FFN down | 114,427 | 14.9% | 25.2% |
| LM head | 1,595 | 0.2% | — |


### .internal/models/gemma-4-e2b-it-6bit @ prompt=4096, generation=128

Forward wall (median across trials): **753,456 µs** — 2 prefill step(s), 70 layer-passes, 4096 tokens, median AX direct prefill = 5,431.9 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | 5,561 | 0.7% | — |
| pre-SDPA (umbrella) | 64,781 | 8.6% | — |
|   QKV projection | 18,095 | 2.4% | 27.9% |
|   QK norm | 29,214 | 3.9% | 45.1% |
|   RoPE + KV append | 17,379 | 2.3% | 26.8% |
| SDPA | 146,545 | 19.4% | — |
| post-attn (umbrella) | 529,112 | 70.2% | — |
|   output projection | 44,250 | 5.9% | 8.4% |
|   residual norm | 15,710 | 2.1% | 3.0% |
|   residual gate | 28,260 | 3.8% | 5.3% |
|   FFN (umbrella) | 440,671 | 58.5% | 83.3% |
|     FFN gate+up | 177,098 | 23.5% | 40.2% |
|     FFN activation | 133,561 | 17.7% | 30.3% |
|     FFN down | 114,356 | 15.2% | 26.0% |
| LM head | 1,700 | 0.2% | — |


### .internal/models/gemma-4-e2b-it-8bit @ prompt=4096, generation=128

Forward wall (median across trials): **716,547 µs** — 2 prefill step(s), 70 layer-passes, 4096 tokens, median AX direct prefill = 5,711.5 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | 5,855 | 0.8% | — |
| pre-SDPA (umbrella) | 63,270 | 8.8% | — |
|   QKV projection | 18,150 | 2.5% | 28.7% |
|   QK norm | 27,189 | 3.8% | 43.0% |
|   RoPE + KV append | 17,608 | 2.5% | 27.8% |
| SDPA | 141,682 | 19.8% | — |
| post-attn (umbrella) | 501,954 | 70.1% | — |
|   output projection | 39,515 | 5.5% | 7.9% |
|   residual norm | 14,843 | 2.1% | 3.0% |
|   residual gate | 27,376 | 3.8% | 5.5% |
|   FFN (umbrella) | 419,837 | 58.6% | 83.6% |
|     FFN gate+up | 171,320 | 23.9% | 40.8% |
|     FFN activation | 127,083 | 17.7% | 30.3% |
|     FFN down | 106,591 | 14.9% | 25.4% |
| LM head | 1,980 | 0.3% | — |


### .internal/models/gemma-4-e4b-it-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,088,892 µs** — 2 prefill step(s), 84 layer-passes, 4096 tokens, median AX direct prefill = 3,759.5 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | 7,412 | 0.7% | — |
| pre-SDPA (umbrella) | 105,328 | 9.7% | — |
|   QKV projection | 44,423 | 4.1% | 42.2% |
|   QK norm | 36,148 | 3.3% | 34.3% |
|   RoPE + KV append | 24,958 | 2.3% | 23.7% |
| SDPA | 163,911 | 15.1% | — |
| post-attn (umbrella) | 808,840 | 74.3% | — |
|   output projection | 57,867 | 5.3% | 7.2% |
|   residual norm | 19,983 | 1.8% | 2.5% |
|   residual gate | 38,461 | 3.5% | 4.8% |
|   FFN (umbrella) | 693,125 | 63.7% | 85.7% |
|     FFN gate+up | 334,497 | 30.7% | 48.3% |
|     FFN activation | 158,281 | 14.5% | 22.8% |
|     FFN down | 183,926 | 16.9% | 26.5% |
| LM head | 1,869 | 0.2% | — |


### .internal/models/GLM-4.7-Flash-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,520,751 µs** — 2 prefill step(s), 94 layer-passes, 4096 tokens, median AX direct prefill = 2,692.1 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | — | — | — |
|   QKV projection | — | — | — |
|   QK norm | — | — | — |
|   RoPE + KV append | — | — | — |
| SDPA | — | — | — |
| post-attn (umbrella) | — | — | — |
|   output projection | — | — | — |
|   residual norm | 836,500 | 55.0% | — |
|   residual gate | 22,018 | 1.4% | — |
|   FFN (umbrella) | 658,299 | 43.3% | — |
|     FFN gate+up | 7,734 | 0.5% | 1.2% |
|     FFN activation | 2,563 | 0.2% | 0.4% |
|     FFN down | 4,548 | 0.3% | 0.7% |
| LM head | 1,153 | 0.1% | — |


### .internal/models/Qwen3-Coder-Next-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **2,532,292 µs** — 8 prefill step(s), 384 layer-passes, 4096 tokens, median AX direct prefill = 1,617.1 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 118,695 | 4.7% | — |
|   QKV projection | 69,338 | 2.7% | 58.4% |
|   QK norm | 19,440 | 0.8% | 16.4% |
|   RoPE + KV append | 29,986 | 1.2% | 25.3% |
| SDPA | 77,554 | 3.1% | — |
| post-attn (umbrella) | 469,847 | 18.6% | — |
|   output projection | 41,134 | 1.6% | 8.8% |
|   residual norm | 542,188 | 21.4% | 115.4% |
|   residual gate | 73,288 | 2.9% | 15.6% |
|   FFN (umbrella) | 1,661,759 | 65.6% | 353.7% |
|     FFN gate+up | — | — | — |
|     FFN activation | — | — | — |
|     FFN down | — | — | — |
| LM head | 4,398 | 0.2% | — |


### .internal/models/Qwen3.5-9B-MLX-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,713,868 µs** — 8 prefill step(s), 256 layer-passes, 4096 tokens, median AX direct prefill = 2,389.0 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 107,429 | 6.3% | — |
|   QKV projection | 68,336 | 4.0% | 63.6% |
|   QK norm | 13,303 | 0.8% | 12.4% |
|   RoPE + KV append | 25,404 | 1.5% | 23.6% |
| SDPA | 53,229 | 3.1% | — |
| post-attn (umbrella) | 293,982 | 17.2% | — |
|   output projection | 38,536 | 2.2% | 13.1% |
|   residual norm | 512,196 | 29.9% | 174.2% |
|   residual gate | 50,305 | 2.9% | 17.1% |
|   FFN (umbrella) | 929,375 | 54.2% | 316.1% |
|     FFN gate+up | 530,726 | 31.0% | 57.1% |
|     FFN activation | 77,213 | 4.5% | 8.3% |
|     FFN down | 320,320 | 18.7% | 34.5% |
| LM head | 9,810 | 0.6% | — |


### .internal/models/Qwen3.6-35B-A3B-5bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,830,732 µs** — 8 prefill step(s), 320 layer-passes, 4096 tokens, median AX direct prefill = 2,236.6 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 99,821 | 5.5% | — |
|   QKV projection | 58,930 | 3.2% | 59.0% |
|   QK norm | 14,887 | 0.8% | 14.9% |
|   RoPE + KV append | 25,797 | 1.4% | 25.8% |
| SDPA | 63,114 | 3.4% | — |
| post-attn (umbrella) | 333,394 | 18.2% | — |
|   output projection | 34,309 | 1.9% | 10.3% |
|   residual norm | 459,936 | 25.1% | 138.0% |
|   residual gate | 58,758 | 3.2% | 17.6% |
|   FFN (umbrella) | 1,109,772 | 60.6% | 332.9% |
|     FFN gate+up | — | — | — |
|     FFN activation | — | — | — |
|     FFN down | — | — | — |
| LM head | 6,773 | 0.4% | — |


### .internal/models/Qwen3.6-35B-A3B-6bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,875,666 µs** — 8 prefill step(s), 320 layer-passes, 4096 tokens, median AX direct prefill = 2,183.0 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 102,758 | 5.5% | — |
|   QKV projection | 62,005 | 3.3% | 60.3% |
|   QK norm | 17,168 | 0.9% | 16.7% |
|   RoPE + KV append | 23,220 | 1.2% | 22.6% |
| SDPA | 64,139 | 3.4% | — |
| post-attn (umbrella) | 344,705 | 18.4% | — |
|   output projection | 36,464 | 1.9% | 10.6% |
|   residual norm | 460,287 | 24.5% | 133.5% |
|   residual gate | 64,126 | 3.4% | 18.6% |
|   FFN (umbrella) | 1,130,938 | 60.3% | 328.1% |
|     FFN gate+up | — | — | — |
|     FFN activation | — | — | — |
|     FFN down | — | — | — |
| LM head | 7,571 | 0.4% | — |


### .internal/models/Qwen3.6-35B-A3B-8bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,878,412 µs** — 8 prefill step(s), 320 layer-passes, 4096 tokens, median AX direct prefill = 2,179.9 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 98,273 | 5.2% | — |
|   QKV projection | 58,938 | 3.1% | 60.0% |
|   QK norm | 16,470 | 0.9% | 16.8% |
|   RoPE + KV append | 22,433 | 1.2% | 22.8% |
| SDPA | 62,548 | 3.3% | — |
| post-attn (umbrella) | 348,087 | 18.5% | — |
|   output projection | 34,532 | 1.8% | 9.9% |
|   residual norm | 446,252 | 23.8% | 128.2% |
|   residual gate | 61,663 | 3.3% | 17.7% |
|   FFN (umbrella) | 1,155,767 | 61.5% | 332.0% |
|     FFN gate+up | — | — | — |
|     FFN activation | — | — | — |
|     FFN down | — | — | — |
| LM head | 9,241 | 0.5% | — |


### .internal/models/Qwen3.6-35B-A3B-UD-MLX-4bit @ prompt=4096, generation=128

Forward wall (median across trials): **1,774,440 µs** — 8 prefill step(s), 320 layer-passes, 4096 tokens, median AX direct prefill = 2,307.5 tok/s.

| Stage | µs (median) | %-of-forward | %-of-parent |
|---|---:|---:|---:|
| per-layer input | — | — | — |
| pre-SDPA (umbrella) | 99,868 | 5.6% | — |
|   QKV projection | 59,526 | 3.4% | 59.6% |
|   QK norm | 16,867 | 1.0% | 16.9% |
|   RoPE + KV append | 24,516 | 1.4% | 24.5% |
| SDPA | 62,551 | 3.5% | — |
| post-attn (umbrella) | 321,611 | 18.1% | — |
|   output projection | 34,820 | 2.0% | 10.8% |
|   residual norm | 454,964 | 25.6% | 141.5% |
|   residual gate | 62,596 | 3.5% | 19.5% |
|   FFN (umbrella) | 1,041,681 | 58.7% | 323.9% |
|     FFN gate+up | — | — | — |
|     FFN activation | — | — | — |
|     FFN down | — | — | — |
| LM head | 9,213 | 0.5% | — |

