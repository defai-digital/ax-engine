# Qwen3.6 MTP Peer Comparison Refresh Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 59.7 tok/s | 657.4 tok/s | 489 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 63.2 tok/s | 694.6 tok/s | 490 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 59.4 tok/s | 861.2 tok/s | 400 ms | 94.5% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `ax_engine` | 41.9 tok/s | 644.7 tok/s | 500 ms | 99.4% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `ax_engine` | 172.5 tok/s | 1,819.8 tok/s | 177 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | 137.2 tok/s | 1,678.7 tok/s | 197 ms | 95.7% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `ax_engine` | 139.4 tok/s | 1,530.2 tok/s | 210 ms | 99.8% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | 115.6 tok/s | 1,349.5 tok/s | 246 ms | 96.7% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |

Notes:

- 27B 4-bit flappy AX Engine row comes from the 2026-06-30 current-code AX-only rerun.
- 27B 4-bit flappy MTPLX and lightning-mlx rows come from the 2026-06-30 same-session peer rerun.
- Other rows come from the 2026-06-29 apples-to-apples peer comparison.
- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
