# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | Lightning v0.6.10 | Lightning+ng v0.6.10 | AX Engine v5.1.0 | AX+ngram v5.1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 693.2 | 473.0 * | 466.2 * | 685.5 | 685.1 |
| Qwen3.6 27B 4-bit | long_code | 795.2 | 658.9 * | 656.8 * | 790.3 | 789.8 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.2 | 482.7 * | 477.2 * | 691.5 | 690.7 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1529.7 | 886.2 * | 876.6 * | 1808.6 | 1817.9 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2447.7 | 1374.5 * | 1502.1 * | 2699.3 | 2696.6 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1678.9 | 964.1 * | 970.4 * | 1968.6 | 1978.0 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | Lightning v0.6.10 | Lightning+ng v0.6.10 | AX Engine v5.1.0 | AX+ngram v5.1.0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 470.6 | 680.3 * | 690.1 * | 469.0 | 469.3 |
| Qwen3.6 27B 4-bit | long_code | 902.3 | 1088.9 * | 1092.4 * | 907.9 | 908.4 |
| Qwen3.6 27B 4-bit | python_modules_long | 506.3 | 718.1 * | 727.1 * | 505.9 | 506.1 |
| Qwen3.6 35B-A3B 4-bit | flappy | 210.3 | 366.8 * | 367.8 * | 177.8 | 177.0 |
| Qwen3.6 35B-A3B 4-bit | long_code | 293.1 | 479.1 * | 479.2 * | 265.8 | 266.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 198.7 | 360.2 * | 356.4 * | 174.2 | 171.7 |

\* Lightning TTFT includes local HTTP socket overhead

