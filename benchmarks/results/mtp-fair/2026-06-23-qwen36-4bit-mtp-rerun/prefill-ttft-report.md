# MTP Prefill Rate and TTFT Report

## Notes

- **MTPLX**: prefill measured directly via `prompt_eval_time_s` (offline, pure GPU compute).
- **AX Engine**: prefill and TTFT measured at runner level (`ttft_source: ax_engine_runner_prefill_time`).
- **Lightning-MLX**: TTFT measured client-side via `ttft_s` (includes local HTTP socket overhead).
  Prefill rate is approximate (`prompt_tokens / ttft_s`); overstates prefill latency slightly.

## Prefill Rate (tok/s, higher is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 |
| --- | --- | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 718.3 | 672.9 |
| Qwen3.6 27B 4-bit | long_code | 818.9 | 780.9 |
| Qwen3.6 27B 4-bit | python_modules_long | 706.4 | 681.2 |
| Qwen3.6 35B-A3B 4-bit | flappy | 1631.3 | 1766.7 |
| Qwen3.6 35B-A3B 4-bit | long_code | 2483.3 | 2679.3 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 1627.0 | 1968.1 |

\* approx: Lightning prefill = prompt\_tokens / ttft\_s (includes HTTP overhead)

## TTFT (ms, lower is better)

| Model | Suite | MTPLX 0.3.7 | AX Engine v5.2.2 |
| --- | --- | ---: | ---: |
| Qwen3.6 27B 4-bit | flappy | 447.6 | 477.8 |
| Qwen3.6 27B 4-bit | long_code | 876.2 | 918.8 |
| Qwen3.6 27B 4-bit | python_modules_long | 494.8 | 513.8 |
| Qwen3.6 35B-A3B 4-bit | flappy | 198.1 | 182.6 |
| Qwen3.6 35B-A3B 4-bit | long_code | 289.0 | 267.8 |
| Qwen3.6 35B-A3B 4-bit | python_modules_long | 213.3 | 177.6 |

\* Lightning TTFT includes local HTTP socket overhead
