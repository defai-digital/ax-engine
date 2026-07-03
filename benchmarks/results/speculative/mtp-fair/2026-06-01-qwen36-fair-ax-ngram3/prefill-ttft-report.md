# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | Light. MTP | Light. ngram+MTP | AX MTP | AX MTP+n-gram |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 693.2 | 439.6 * | 423.6 * | 640.6 | 648.3 |
| Qwen3.6 27B 4-bit | long_code | 795.2 | 632.5 * | 611.5 * | 692.2 | 694.3 |
| Qwen3.6 27B 4-bit | python_modules_long | 690.2 | 451.9 * | 442.9 * | 651.0 | 648.0 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1529.7 | 918.0 * | 875.7 * | 1664.3 | 1655.1 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2447.7 | 1664.5 * | 1606.9 * | 2306.6 | 2282.4 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1704.7 | 995.2 * | 962.1 * | 1816.1 | 1805.6 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | Light. MTP | Light. ngram+MTP | AX MTP | AX MTP+n-gram |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 470.6 | 731.5 * | 759.1 * | 502.1 | 496.1 |
| Qwen3.6 27B 4-bit | long_code | 902.3 | 1134.5 * | 1173.5 * | 1036.6 | 1033.5 |
| Qwen3.6 27B 4-bit | python_modules_long | 506.3 | 728.7 * | 789.5 * | 537.4 | 538.9 |
| Qwen3.6 35B-A3B 4-bit | flappy | 210.3 | 351.7 * | 371.1 * | 193.2 | 194.3 |
| Qwen3.6 35B-A3B 4-bit | long_code | 293.1 | 431.1 * | 446.5 * | 311.1 | 314.4 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 202.4 | 349.3 * | 360.4 * | 190.7 | 190.4 |

\* Lightning TTFT includes local HTTP socket overhead

