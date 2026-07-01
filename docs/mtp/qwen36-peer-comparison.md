# Qwen3.6 MTP Peer Benchmark

This page holds the full Qwen3.6 MTP peer benchmark results for AX Engine,
MTPLX, and lightning-mlx. The README keeps only the decode-throughput view
because decode is the closest comparable metric across the three engines. The
full result set belongs here because prefill, TTFT, accept rate, model artifact
identity, seed policy, and output-quality gates all need more context than the
README should carry.

This is a **production-configuration comparison**, not an identical-weight
apples-to-apples benchmark. Each engine runs the package and settings used by
its ecosystem. Treat the rows as audit evidence and trend guidance, not as a
single definitive peer-engine ranking.

## Limitations

- **Different MTP artifacts.** AX Engine uses
  `ax-local/Qwen3.6-27B-MTP`; MTPLX and lightning-mlx use
  `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`. The tensors derive from
  Qwen's official weights, but quantization and draft-head precision differ.
- **AX optimistic verify is not a promoted peer default.** The earlier AX 27B
  4-bit optimistic row entered a periodic whitespace token cycle and inflated
  accept/decode. The current AX 27B 4-bit row reruns the same benchmark with
  strict MTP verification (`AX_MLX_MTP_OPTIMISTIC=0`) and passes the
  output-degeneracy gate.
- **Prefill and TTFT scopes differ.** AX reports runner-internal timing, MTPLX
  derives from server-side `prompt_eval_time_s`, and lightning-mlx reports
  client-observed HTTP stream TTFT. These columns are shown for provenance but
  should not be read as a clean cross-engine prefill/TTFT leaderboard.
- **Seeds differ.** The refreshed AX 27B 4-bit row uses seed 44; older stitched
  rows keep their source-run seed policy.
- **Composite artifact.** Rows are stitched from multiple runs from
  2026-06-29 through 2026-07-01, not one physical same-session measurement.
- **Dirty builds.** Some stitched raw artifacts were produced from local dirty
  checkouts. Reproducible publication should rerun every engine from clean
  tagged commits.

## Benchmark Contract

| Field | Value |
| --- | --- |
| Prompt suite | `flappy`, first 4 cases |
| Generated tokens | 1000 |
| Warmups / measured reps | 2 warmups, 5 measured |
| Cooldown | 15 s between repetitions, 10 s between prompt cases |
| Sampling | `temperature=0.6`, `top_p=0.95`, `top_k=20` |
| Mode | Pure MTP |
| Prefix cache | Cross-request prefix cache disabled for cold-prefill parity |
| AX optimistic verify | Disabled for the refreshed 27B 4-bit peer row (`AX_MLX_MTP_OPTIMISTIC=0`) |

## Decode Summary

Decode tok/s is the closest comparable metric in this peer set. The refreshed
AX 27B 4-bit row is strict and passes the output-degeneracy gate.

| Target | AX Engine | MTPLX | lightning-mlx | Readout |
| --- | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | 61.0 tok/s | 64.3 tok/s | 59.4 tok/s | AX strict row is clean; MTPLX leads this 27B 4-bit peer row |
| Qwen3.6 27B 6-bit | 41.4 tok/s | - | - | No official comparable peer 27B 6-bit MTP artifact |
| Qwen3.6 35B-A3B 4-bit | 166.3 tok/s | 138.1 tok/s | 116.2 tok/s | AX leads this production-config row |
| Qwen3.6 35B-A3B 6-bit | 141.8 tok/s | 117.6 tok/s | 96.3 tok/s | AX leads this production-config row |

![Qwen3.6 MTP peer decode comparison](../assets/perf-mtp-peer-comparison-apples-to-apples.svg)

## Full Result Table

| Target | Engine | Decode | Prefill | TTFT | Accept | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3.6 27B 4-bit | AX Engine | 61.0 tok/s | 812.3 tok/s | 396 ms | 100.0% | ok; strict verify |
| Qwen3.6 27B 4-bit | MTPLX | 64.3 tok/s | 681.4 tok/s | 470 ms | 100.0% | ok |
| Qwen3.6 27B 4-bit | lightning-mlx | 59.4 tok/s | 426.0 tok/s | 784 ms | 95.9% | ok |
| Qwen3.6 27B 6-bit | AX Engine | 41.4 tok/s | 637.1 tok/s | 507 ms | 100.0% | ok |
| Qwen3.6 27B 6-bit | MTPLX | - | - | - | - | No official 27B 6-bit MTP artifact |
| Qwen3.6 27B 6-bit | lightning-mlx | - | - | - | - | No official 27B 6-bit MTP artifact |
| Qwen3.6 35B-A3B 4-bit | AX Engine | 166.3 tok/s | 1,755.3 tok/s | 184 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 4-bit | MTPLX | 138.1 tok/s | 1,637.0 tok/s | 193 ms | 95.7% | ok |
| Qwen3.6 35B-A3B 4-bit | lightning-mlx | 116.2 tok/s | 1,466.5 tok/s | 215 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | AX Engine | 141.8 tok/s | 1,536.0 tok/s | 209 ms | 100.0% | ok |
| Qwen3.6 35B-A3B 6-bit | MTPLX | 117.6 tok/s | 1,383.9 tok/s | 235 ms | 96.7% | ok |
| Qwen3.6 35B-A3B 6-bit | lightning-mlx | 96.3 tok/s | 1,215.8 tok/s | 272 ms | 100.0% | ok |

## Full Charts

These charts are intentionally kept off the README because prefill, TTFT, and
accept rate need the limitations above to be interpreted correctly.

![Qwen3.6 MTP peer prefill comparison](../assets/perf-mtp-peer-comparison-prefill-apples-to-apples.svg)

![Qwen3.6 MTP peer TTFT comparison](../assets/perf-mtp-peer-comparison-ttft-apples-to-apples.svg)

![Qwen3.6 MTP peer accept-rate comparison](../assets/perf-mtp-peer-comparison-accept-rate-apples-to-apples.svg)

## Artifacts

- Combined summary:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-07-01-peer-comparison-apples-to-apples/summary.md),
  [`summary.json`](../../benchmarks/results/mtp-qwen36-matrix/2026-07-01-peer-comparison-apples-to-apples/summary.json)
- AX 27B 4-bit rerun:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-07-01-27b4-ax-strict-full-rerun-seed44-r1/summary.md)
- MTPLX 1.0.4 rerun:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-07-01-mtplx-v104-rerun/summary.md)
- lightning-mlx prefix-cache-disabled rerun:
  [`summary.md`](../../benchmarks/results/mtp-qwen36-matrix/2026-07-01-lightning-prefix-disabled-r1/summary.md)

## What Would Make This Fully Fair

To promote this as a strict peer-engine benchmark, rerun all engines with:

- the same target weights and the same draft-head weights;
- the same per-repetition seed sequence;
- output-degeneracy gate passing on every promoted row;
- one clean tagged build per engine;
- one physical session with interleaved runs;
- either a common client-observed TTFT/prefill contract or separate internal
  and client-observed columns.
