# Qwen3.6 MTP Peer Comparison Refresh Summary

| Target | Suite | Engine | Decode | Prefill | TTFT | Accept | Status |
|---|---|---|---:|---:|---:|---:|---|
| Qwen3.6 27B 4-bit | `flappy` | `ax_engine` | 60.5 tok/s | 668.2 tok/s | 483 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `mtplx` | 63.2 tok/s | 694.6 tok/s | 490 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `lightning_mlx` | 59.4 tok/s | 861.2 tok/s | 400 ms | 94.5% | ok |
| Qwen3.6 27B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `ax_engine` | 41.4 tok/s | 637.1 tok/s | 507 ms | 100.0% | ok |
| Qwen3.6 27B 6-bit | `flappy` | `mtplx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `lightning_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 27B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `ax_engine` | 166.3 tok/s | 1,755.3 tok/s | 184 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `mtplx` | 137.2 tok/s | 1,678.7 tok/s | 197 ms | 95.7% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `lightning_mlx` | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 4-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `ax_engine` | 141.8 tok/s | 1,536.0 tok/s | 209 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `mtplx` | 115.6 tok/s | 1,349.5 tok/s | 246 ms | 96.7% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `lightning_mlx` | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `rapid_mlx` | - tok/s | - tok/s | - ms | - | unsupported |
| Qwen3.6 35B-A3B 6-bit | `flappy` | `omlx` | - tok/s | - tok/s | - ms | - | unsupported |

Notes:

- 27B 4-bit AX Engine row is refreshed from the 2026-06-30 targeted AX-only flappy rerun.
- Other AX Engine rows remain from the 2026-06-30 full AX-only flappy rerun.
- 27B 4-bit MTPLX and lightning-mlx rows are from the 2026-06-30 same-session peer rerun.
- Other peer rows remain from the 2026-06-29 apples-to-apples run.
- AX rows are pure MTP and fail summary generation if n-gram telemetry is non-zero.
