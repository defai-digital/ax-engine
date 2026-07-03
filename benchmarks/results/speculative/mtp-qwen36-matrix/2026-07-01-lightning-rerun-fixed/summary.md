# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 59.1 tok/s | 828.8 tok/s | 410 ms | 94.7% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | 125.3 tok/s | 1,516.8 tok/s | 216 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | 103.1 tok/s | 1,132.5 tok/s | 286 ms | 100.0% | ok |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
