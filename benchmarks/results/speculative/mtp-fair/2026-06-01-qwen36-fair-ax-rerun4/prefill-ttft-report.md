# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | Light. MTP | Light. ngram+MTP | AX MTP | AX MTP+n-gram |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 693.2 | 473.0 * | 466.2 * | 668.4 | 667.3 |
| Qwen3.6 27B 4-bit | long_code | 795.2 | 658.9 * | 656.8 * | 775.5 | 780.3 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.2 | 482.7 * | 477.2 * | 677.6 | 677.8 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1529.7 | 886.2 * | 876.6 * | 1728.4 | 1726.1 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2447.7 | 1374.5 * | 1502.1 * | 2619.9 | 2619.1 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1678.9 | 964.1 * | 970.4 * | 1897.3 | 1890.5 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | Light. MTP | Light. ngram+MTP | AX MTP | AX MTP+n-gram |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 470.6 | 680.3 * | 690.1 * | 481.0 | 481.8 |
| Qwen3.6 27B 4-bit | long_code | 902.3 | 1088.9 * | 1092.4 * | 925.2 | 919.5 |
| Qwen3.6 27B 4-bit | python_modules_long | 506.3 | 718.1 * | 727.1 * | 516.5 | 516.2 |
| Qwen3.6 35B-A3B 4-bit | flappy | 210.3 | 366.8 * | 367.8 * | 186.1 | 186.3 |
| Qwen3.6 35B-A3B 4-bit | long_code | 293.1 | 479.1 * | 479.2 * | 273.9 | 273.9 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 198.7 | 360.2 * | 356.4 * | 182.4 | 182.3 |

\* Lightning TTFT includes local HTTP socket overhead

