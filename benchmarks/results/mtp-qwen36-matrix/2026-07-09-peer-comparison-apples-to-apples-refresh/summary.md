# Qwen3.6 MTP peer comparison stitched refresh

AX Engine rows are refreshed from current local runs on 2026-07-08. MTPLX rows are refreshed from MTPLX 2.0.1 local runs on 2026-07-09. lightning-mlx rows are retained peer records from the previous README/docs evidence and were not rerun in this refresh.

| Target | Engine | Decode | Prefill | TTFT | Accept | Source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | `ax_engine` | 63.0 tok/s | 812.2 tok/s | 396 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 27B 6-bit | `ax_engine` | 41.8 tok/s | 757.3 tok/s | 426 ms | 99.9% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 27B 6-bit | `mtplx` | - | - | - | - | no comparable peer artifact |
| Qwen3.6 35B-A3B 4-bit | `ax_engine` | 172.4 tok/s | 2,096.7 tok/s | 153 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 35B-A3B 6-bit | `ax_engine` | 141.2 tok/s | 1,828.8 tok/s | 177 ms | 100.0% | AX Engine current-code refresh, 2026-07-08 |
| Qwen3.6 27B 4-bit | `mtplx` | 58.5 tok/s | 676.1 tok/s | 485 ms | 98.8% | MTPLX 2.0.1 local rerun, 2026-07-09 |
| Qwen3.6 27B 4-bit | `lightning_mlx` | 55.7 tok/s | 414.9 tok/s | 801 ms | 96.6% | ok; retained 2026-07-01 peer artifact |
| Qwen3.6 35B-A3B 4-bit | `mtplx` | 137.9 tok/s | 1,639.7 tok/s | 191 ms | 94.7% | MTPLX 2.0.1 local rerun, 2026-07-09 |
| Qwen3.6 35B-A3B 4-bit | `lightning_mlx` | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok; retained 2026-07-01 peer artifact |
| Qwen3.6 35B-A3B 6-bit | `mtplx` | 119.0 tok/s | 1,480.6 tok/s | 221 ms | 95.6% | MTPLX 2.0.1 local rerun, 2026-07-09 |
| Qwen3.6 35B-A3B 6-bit | `lightning_mlx` | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok; retained 2026-07-01 peer artifact |
