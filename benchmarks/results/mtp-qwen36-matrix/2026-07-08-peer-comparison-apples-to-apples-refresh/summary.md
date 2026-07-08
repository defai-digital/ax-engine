# Qwen3.6 MTP peer comparison stitched refresh

AX Engine rows are refreshed from current code on 2026-07-08. MTPLX and lightning-mlx rows are retained peer records from the previous README/docs evidence and were not rerun in this refresh.

| Target | Engine | Decode | Prefill | TTFT | Accept | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | `ax_engine` | 63.0 tok/s | 812.2 tok/s | 396 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 27B 4-bit | `mtplx` | 58.5 tok/s | 653.4 tok/s | 485 ms | 98.4% | ok; 2026-07-01 same AX sidecar |
| Qwen3.6 27B 4-bit | `lightning_mlx` | 55.7 tok/s | 414.9 tok/s | 801 ms | 96.6% | ok; 2026-07-01 same AX sidecar |
| Qwen3.6 27B 6-bit | `ax_engine` | 41.8 tok/s | 757.3 tok/s | 426 ms | 99.9% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 27B 6-bit | `mtplx` | - | - | - | - | no comparable peer artifact |
| Qwen3.6 27B 6-bit | `lightning_mlx` | - | - | - | - | no comparable peer artifact |
| Qwen3.6 35B-A3B 4-bit | `ax_engine` | 172.4 tok/s | 2,096.7 tok/s | 153 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 35B-A3B 4-bit | `mtplx` | 138.1 tok/s | 1,637.0 tok/s | 193 ms | 95.7% | ok; 2026-07-01 production config |
| Qwen3.6 35B-A3B 4-bit | `lightning_mlx` | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok; 2026-07-01 production config |
| Qwen3.6 35B-A3B 6-bit | `ax_engine` | 141.2 tok/s | 1,828.8 tok/s | 177 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 35B-A3B 6-bit | `mtplx` | 117.6 tok/s | 1,383.9 tok/s | 235 ms | 96.7% | ok; 2026-07-01 production config |
| Qwen3.6 35B-A3B 6-bit | `lightning_mlx` | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok; 2026-07-01 production config |
