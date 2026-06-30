# Qwen3.6 MTP Benchmark Matrix Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 62.9 tok/s | 682.5 tok/s | 477 ms | 99.5% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 63.2 tok/s | 694.6 tok/s | 490 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 59.4 tok/s | 861.2 tok/s | 400 ms | 94.5% | ok |

Notes:

- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
- MTPLX prefill and TTFT are derived from `prompt_eval_time_s` in the MTPLX runner.
- Lightning prefill is approximate (`prompt_tokens / client TTFT`) and includes local HTTP overhead.
- Unsupported peer lanes are listed in `plan.md` with the exact support reason.
