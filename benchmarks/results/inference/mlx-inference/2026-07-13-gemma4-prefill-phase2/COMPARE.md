# Gemma4 prefill Phase 1+2 vs prior direct rebench

**New:** `2026-07-13-gemma4-prefill-phase2` (main `c0722290` + prefill compile / dual-path fuse)
**Base:** `2026-07-12-adr038-0c95182f-gemma4-direct` (pre Phase 1+2 prefill work)

| model | pt | prefill old | prefill new | Δ | decode old | decode new | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12b | 128 | 403.9 | 402.7 | -0.3% | 69.6 | 69.6 | -0.0% |
| 12b | 512 | 528.9 | 529.7 | +0.2% | 67.8 | 67.7 | -0.1% |
| 12b | 2048 | 558.6 | 558.9 | +0.1% | 65.9 | 65.6 | -0.5% |
| 26b | 128 | 593.6 | 598.7 | +0.8% | 135.1 | 134.6 | -0.4% |
| 26b | 512 | 1188.1 | 1191.8 | +0.3% | 131.8 | 131.1 | -0.6% |
| 26b | 2048 | 1531.0 | 1533.9 | +0.2% | 127.0 | 125.7 | -1.0% |

**Reading:** Phase 1+2 did **not** move cold prefill materially (±noise). Decode flat. Next lever is deeper MoE/expert Metal work + stage profiling, not more micro-fuses.
