# Measured results

M5 Max 128 GiB, matched native API/callback boundaries. Medians of five measured
requests after two warmups and three-second idle. See README for scope and controls.

## Paired AX improvement

| Pack | Workload | Base TTFT ms | Candidate TTFT ms | TTFT change | Base prefill tok/s | Candidate prefill tok/s | Decode change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiel | random128 | 235.51 | 141.38 | -39.97% | 551.95 | 945.01 | -0.65% |
| tiel | python-lru | 251.30 | 144.02 | -42.69% | 787.75 | 1397.46 | -0.47% |
| tiel | rust-jsonl | 407.68 | 281.16 | -31.03% | 2579.94 | 3785.04 | -1.07% |
| cyber | random128 | 232.69 | 139.27 | -40.15% | 559.03 | 947.50 | -1.04% |
| cyber | python-lru | 256.08 | 142.67 | -44.29% | 868.47 | 1587.57 | -0.43% |
| cyber | rust-jsonl | 413.54 | 280.36 | -32.20% | 2610.50 | 3885.68 | -1.30% |

Prefill uses AX's existing forward-wall counter, including GPU submission/evaluation
waiting. This is a latency improvement, not evidence of faster GPU arithmetic.

## Completion throughput (tok/s)

| Pack | Workload | AX candidate | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: |
| tiel | random128 | 117.83 | 109.84 | 112.59 |
| tiel | python-lru | 194.77 | 176.99 | 178.55 |
| tiel | rust-jsonl | 172.31 | 145.64 | 139.45 |
| cyber | random128 | 170.88 | 174.38 | 114.09 |
| cyber | python-lru | 219.19 | 194.36 | 201.19 |
| cyber | rust-jsonl | 168.93 | 164.18 | 158.93 |

## TTFT (ms)

| Pack | Workload | AX candidate | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: |
| tiel | random128 | 141.38 | 161.85 | 163.24 |
| tiel | python-lru | 144.02 | 157.28 | 157.73 |
| tiel | rust-jsonl | 281.16 | 292.18 | 294.82 |
| cyber | random128 | 139.27 | 161.01 | 163.47 |
| cyber | python-lru | 142.67 | 155.97 | 158.83 |
| cyber | rust-jsonl | 280.36 | 291.07 | 295.05 |

## Post-first-batch decode (tok/s)

| Pack | Workload | AX candidate | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: |
| tiel | random128 | 134.48 | 126.75 | 130.43 |
| tiel | python-lru | 217.85 | 198.39 | 199.83 |
| tiel | rust-jsonl | 211.67 | 174.17 | 165.61 |
| cyber | random128 | 208.19 | 221.00 | 132.69 |
| cyber | python-lru | 248.72 | 220.15 | 228.98 |
| cyber | rust-jsonl | 206.40 | 201.46 | 194.17 |

## Trial spread

Minimum/maximum TTFT across the five short samples; sample size does not license
population-wide significance claims. Exact per-trial records remain in trials.json.

| Pack | Workload | Base TTFT min/max ms | Candidate TTFT min/max ms |
| --- | --- | ---: | ---: |
| tiel | random128 | 233.40/241.75 | 138.51/141.95 |
| tiel | python-lru | 246.75/254.00 | 139.14/144.46 |
| tiel | rust-jsonl | 407.54/412.47 | 280.66/281.45 |
| cyber | random128 | 230.25/235.01 | 137.71/140.52 |
| cyber | python-lru | 252.54/259.24 | 141.32/143.64 |
| cyber | rust-jsonl | 410.85/417.67 | 279.65/281.24 |

## Regression controls

| Pack | Control | Workload | Base TTFT ms | Candidate TTFT ms | Decode change |
| --- | --- | --- | ---: | ---: | ---: |
| tiel | long | synthetic-long8192 | 1910.88 | 1786.45 | -0.80% |
| tiel | zeroidle | random128 | 74.57 | 73.84 | -0.37% |
| tiel | zeroidle | python-lru | 89.16 | 88.49 | +0.03% |
| tiel | zeroidle | rust-jsonl | 249.56 | 245.28 | -0.08% |
| tiel | override | random128 | 232.42 | 231.95 | +0.50% |
| tiel | override | python-lru | 249.97 | 250.75 | +0.27% |
| tiel | override | rust-jsonl | 412.57 | 412.19 | +0.15% |
| tiel | direct | random128 | 245.11 | 144.51 | -0.14% |
| tiel | direct | python-lru | 256.25 | 148.79 | -0.12% |
| tiel | direct | rust-jsonl | 407.35 | 277.23 | +0.12% |
| cyber | long | synthetic-long8192 | 1911.69 | 1798.99 | -0.68% |
| cyber | zeroidle | random128 | 72.84 | 73.06 | -0.91% |
| cyber | zeroidle | python-lru | 94.08 | 94.21 | -0.58% |
| cyber | zeroidle | rust-jsonl | 248.06 | 248.88 | -0.72% |
| cyber | override | random128 | 234.84 | 233.40 | +0.43% |
| cyber | override | python-lru | 256.27 | 254.45 | +0.00% |
| cyber | override | rust-jsonl | 411.82 | 415.34 | +0.10% |
| cyber | direct | random128 | 223.75 | 144.74 | +0.36% |
| cyber | direct | python-lru | 256.02 | 147.06 | +0.15% |
| cyber | direct | rust-jsonl | 413.43 | 276.86 | -0.19% |

Direct rows have only one measured sample per case and are correctness controls.
Zero-idle and override rows have three; the synthetic long holdout has three.

## Bounded repeat memory

| Pack | Active MiB min/max | Cache MiB min/max | Peak MiB max |
| --- | ---: | ---: | ---: |
| tiel | 21759.507/21759.507 | 77.170/77.170 | 22071.5 |
| cyber | 21760.915/21760.915 | 77.593/77.593 | 22071.4 |

Twenty measured requests in one process. These allocator bounds do not establish
endurance, mixed-model or memory-contention qualification.

## Trace attribution

Separate 194-input/32-output Metal traces; three measured requests. These rows
are diagnostic and excluded from acceptance throughput. Unwired here means
the early existing environment override; compiled post-load policy results are above.

| Runtime | First GPU after API entry, median ms | GPU active interval union, median ms |
| --- | ---: | ---: |
| ax | 146.10 | 93.50 |
| mtplx | 23.34 | 102.88 |
| ax-unwired | 25.89 | 93.33 |

The controlled residency change removes most excess waiting before first GPU
execution. GPU-active duration is nearly unchanged. This does not prove a
specific undocumented driver eviction algorithm or accelerate GPU arithmetic.

## Validation

Rust 1.97.1: 3746 tests passed. Python: 209 passed, 26 skipped and 142 subtests
passed. fmt, focused MLX library Clippy, maturin develop, script checks, both
qualification dry runs and canonical-claim checks passed. Full-workspace
Clippy still fails with the identical pre-existing error multiset reproduced
on the clean baseline; affected core source is unchanged. This is not an
all-green workspace or release qualification.

The verifier recomputes callback metrics, checks native MTP/direct routes,
exact AX output IDs, token budgets, cache controls, pack immutability, short
latency/prefill gains and regression bounds. See trials.json for raw records.
