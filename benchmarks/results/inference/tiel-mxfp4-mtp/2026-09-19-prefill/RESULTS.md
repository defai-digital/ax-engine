# Measured results

The final change is diagnostic instrumentation. No runtime optimization met the
predeclared >=10% TTFT target; rejected experiments are not enabled or shipped.

## Matched peer matrix

Medians of five measured requests after two warmups; three-second idle;
native API to last committed callback. Conservative-depth control and timing
diagnostics are off. These are matched workloads, not identical cross-engine
decode trajectories. See README for the full contract.

| Pack | Workload | AX baseline completion tok/s | AX final completion tok/s | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: | ---: |
| tiel | random128 | 109.32 | 109.41 | 109.84 | 112.40 |
| tiel | python-lru | 181.22 | 181.27 | 177.23 | 178.49 |
| tiel | rust-jsonl | 160.26 | 159.67 | 145.36 | 139.41 |
| cyber | random128 | 153.41 | 153.60 | 174.01 | 114.27 |
| cyber | python-lru | 201.21 | 201.01 | 194.55 | 201.63 |
| cyber | rust-jsonl | 156.65 | 156.45 | 163.46 | 159.17 |

| Pack | Workload | AX final TTFT ms | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: |
| tiel | random128 | 232.30 | 161.87 | 163.76 |
| tiel | python-lru | 248.25 | 157.54 | 155.65 |
| tiel | rust-jsonl | 410.49 | 293.79 | 294.89 |
| cyber | random128 | 229.26 | 162.02 | 163.21 |
| cyber | python-lru | 252.25 | 157.37 | 158.01 |
| cyber | rust-jsonl | 414.45 | 295.66 | 297.34 |

| Pack | Workload | AX final decode tok/s | MTPLX sustained | MTPLX turbo |
| --- | --- | ---: | ---: | ---: |
| tiel | random128 | 135.28 | 126.56 | 130.26 |
| tiel | python-lru | 218.96 | 198.24 | 199.47 |
| tiel | rust-jsonl | 213.64 | 173.84 | 165.44 |
| cyber | random128 | 210.31 | 221.32 | 132.75 |
| cyber | python-lru | 249.76 | 220.14 | 229.29 |
| cyber | rust-jsonl | 208.75 | 201.10 | 194.50 |

## AX regression controls

Exact committed token arrays are checked for baseline versus final, including
warmups, three short workloads, the synthetic 8192-token holdout, and direct
smokes with AX_NO_SPEC=1. Direct smokes have one warmup and one measured request
per case and are correctness controls, not a direct-path performance claim.

| Pack | Workload | TTFT change | Decode change | Baseline peak MiB | Final peak MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| tiel | random128 | +0.55% | +0.08% | 22071.5 | 22071.5 |
| tiel | python-lru | -0.27% | +0.01% | 22172.3 | 22172.3 |
| tiel | rust-jsonl | +1.05% | -0.15% | 22923.0 | 22923.0 |
| tiel | synthetic-long8192 | +0.41% | -0.68% | 24712.8 | 24712.8 |
| cyber | random128 | -0.99% | -0.17% | 22071.4 | 22071.4 |
| cyber | python-lru | +0.08% | -0.08% | 22221.0 | 22221.0 |
| cyber | rust-jsonl | +0.23% | -0.02% | 22949.1 | 22949.1 |
| cyber | synthetic-long8192 | -0.46% | +0.16% | 24712.8 | 24712.8 |

The long holdout uses one warmup/three measured requests and repeated random
tokens. It is a multi-chunk regression check, not a coding or long-context
quality evaluation. Positive TTFT change means slower; positive decode
change means faster. Small differences are not speedup claims.

## Bounded repeat memory

Twenty measured random128 requests after two warmups, same process, cold KV.

| Pack | Active MiB min/max | Cache MiB min/max | Peak MiB max |
| --- | ---: | ---: | ---: |
| tiel | 21759.507/21759.507 | 77.139/77.170 | 22071.5 |
| cyber | 21759.434/21760.915 | 77.562/79.073 | 22071.4 |

These allocator observations do not qualify endurance or measure whole-system memory.

## Phase diagnostics

Final-build short-phase runs use two warmups and three measured requests per case.
This flag also enables existing verbose diagnostics: these timings are attribution
evidence only, excluded from throughput acceptance.

| Pack | Prompt tokens | Graph build ms | Existing eval/sample ms | Retained materialization ms | Clear cache ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiel | 128 | 3.755 | 225.604 | 0.002 | 1.138 |
| tiel | 194 | 3.765 | 242.864 | 0.002 | 1.305 |
| tiel | 1039 | 4.218 | 400.117 | 0.079 | 1.378 |
| cyber | 128 | 3.763 | 228.509 | 0.002 | 1.081 |
| cyber | 219 | 3.741 | 248.347 | 0.002 | 1.242 |
| cyber | 1064 | 4.324 | 402.550 | 0.079 | 1.209 |

## Official raw reference

The required primary reference uses mlx_lm.benchmark's internal decode timer,
with p128/g128, two effective warmups, five samples and three-second idle.
It is not the native-call completion metric above; no peer ratio is computed.
Its TTFT field is derived from reported prefill throughput, not callback timing.

| Pack | Internal decode tok/s | Prefill tok/s |
| --- | ---: | ---: |
| tiel | 130.225 | 613.049 |
| cyber | 130.222 | 613.879 |

## Validation

Rust 1.97.1: 3733 Rust tests passed. Python: 209 passed, 26 skipped and 142
subtests passed. fmt, focused MLX library Clippy, maturin develop, script checks,
both qualification dry runs, and canonical-claim checks passed. Full-workspace
Clippy remains failing: an isolated clean checkout of the base commit reproduced
the identical diagnostic-message multiset. Those existing core errors are not
changed here. This is not an all-green workspace or release qualification.

The artifact verifier recomputes callback metrics, checks exact AX output identity,
native MTP/direct counters, no-prefix-cache controls, phase geometry, and unchanged
pack manifests/norms. The optimization objective remains open.
