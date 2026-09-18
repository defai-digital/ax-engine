# Decode bandwidth utilization: Mac mini M4 Pro vs MacBook Pro M5 Max

Why the same pack decodes 2.5x faster on the M5 Max campaign laptop than on
the Mac mini M4 Pro qualification SKU, and why that gap is the hardware, not
the runtime.

Pack: `AutomatosX/AX-Qwen3.8-27B-MLX-AXQ-6bit-MTP` @
`3e290738e96972307c6aeb9934ab170ca0eae1c1`. Dense weights streamed per
decode token: **20.84 GB** (`bandwidth_accounting.bytes_used_for_estimate`
in both `ax_engine.json` artifacts). Same `flappy` contract on both hosts;
20-run medians.

Data: [`benchmarks/results/mtp-axq-peer/decode-bandwidth-utilization-2026-09-17.json`](../../benchmarks/results/mtp-axq-peer/decode-bandwidth-utilization-2026-09-17.json).
Chart: rendered by `scripts/render_readme_performance_charts.py` and checked
in CI (`--check`).

<img src="../assets/perf-decode-bandwidth-utilization.svg" alt="Decode throughput expressed as weight-stream bandwidth against Apple's published memory bandwidth for Mac mini M4 Pro and MacBook Pro M5 Max">

## Hosts

| Host | Chip | GPU | Published memory bandwidth | Source |
| --- | --- | --- | ---: | --- |
| Mac mini M4 Pro 64 GB (Mac16,11), qualification SKU | Apple M4 Pro | 20-core | 273 GB/s | [Apple support 121555](https://support.apple.com/en-us/121555) |
| MacBook Pro M5 Max 128 GB (Mac17,6), campaign host | Apple M5 Max | 40-core | 614 GB/s | [Apple MacBook Pro specs](https://www.apple.com/macbook-pro/specs/) |

Bandwidth ratio 614 / 273 = **2.25x**. GPU core ratio 40 / 20 = 2x, and the
M5 GPU adds per-core neural accelerators for matrix work.

## Decode: both hosts sit at the memory-bandwidth ceiling

Direct autoregressive decode reads every dense weight byte once per token,
so `decode tok/s x weight bytes` is the bandwidth the runtime actually
streams. mlx-lm 0.31.3 (same pack, greedy, no speculation) gives the cleanest
reading:

| Host | mlx-lm direct AR | Implied weight stream | Published bandwidth | Utilization |
| --- | ---: | ---: | ---: | ---: |
| Mac mini M4 Pro | 12.78 tok/s | 266 GB/s | 273 GB/s | **97.5%** |
| MacBook Pro M5 Max | 27.90 tok/s | 581 GB/s | 614 GB/s | **94.7%** |

There is no headroom left on either machine for single-token decode. The
2.18x mlx-lm gap is the 2.25x bandwidth gap.

MTP runtimes read the weights once per verify step and emit several tokens,
so their equivalent stream exceeds the published bandwidth. That excess is
the speculation gain, not a measurement error:

| Host | Runtime | Decode | Equivalent stream | % of published bandwidth | vs direct AR |
| --- | --- | ---: | ---: | ---: | ---: |
| Mac mini M4 Pro | AX Engine 7.4.0 MTP depth 3 | 31.05 tok/s | 647 GB/s | 237% | 2.43x |
| Mac mini M4 Pro | MTPLX 2.11.3 depth 3 | 28.16 tok/s | 587 GB/s | 215% | 2.20x |
| Mac mini M4 Pro | OMLX 0.6.4 Lightning depth 1 | 15.02 tok/s | 313 GB/s | 115% | 1.18x |
| MacBook Pro M5 Max | AX Engine 7.4.0 MTP depth 3 | 76.90 tok/s | 1602 GB/s | 261% | 2.76x |
| MacBook Pro M5 Max | MTPLX 2.11.2 depth 3 | 70.62 tok/s | 1472 GB/s | 240% | 2.53x |
| MacBook Pro M5 Max | OMLX 0.6.4 Lightning depth 1 | 38.47 tok/s | 802 GB/s | 131% | 1.38x |

## Cross-runtime ratios are the same on both hosts

If the AX decode or prefill path were broken on the mini, only the AX ratio
would move. Every runtime moves together:

| Runtime | Decode, M5 Max / M4 Pro | Prefill, M5 Max / M4 Pro |
| --- | ---: | ---: |
| AX Engine | 2.48x | 6.61x |
| MTPLX | 2.51x | 6.02x |
| OMLX | 2.56x | - |
| mlx-lm | 2.18x | - |

## Prefill is compute-bound, so the M5 gap is larger

Prefill on 264-432 token prompts is matrix-multiply bound. Two GPU
generations apart, the M5 Max has twice the cores and per-core neural
accelerators, which is why AX prefill goes from 120.3 to 795.3 tok/s (6.6x)
and MTPLX from 114.0 to 686.6 tok/s (6.0x). AX prefill on the mini is not a
fallback path: the harness rows record the same `mtp_head_only_verify_loop`
route and `mtp_head_only_effective` claim status on both hosts.

## MTP status on both hosts

The MTP head-only route (recurrent depth 3, n-gram stacking off, exact Qwen
linear-attention verifier) produces the same greedy tokens on both machines
and therefore the same acceptance telemetry per case:

| Case | Depth-0 / 1 / 2 accept rate | Accepted draft tokens per 5 reps | Verify steps per 5 reps |
| --- | --- | ---: | ---: |
| flappy_pipes | 100% / 100% / 100% | 960 | 325 |
| flappy_score_gates | 100% / 98.5% / 93.8% | 950 | 330 |
| flappy_collision_checks | 100% / 98.5% / 95.4% | 955 | 330 |
| flappy_sound_channels | 100% / 100% / 100% | 960 | 325 |

Each verify step yields one target token plus about 2.9 accepted drafts, so
256 tokens take about 65 weight passes instead of 256. The realized gain is
lower than 3.9x because a depth-3 verify step is more expensive than a
single-token step (four positions through attention and the MTP head), and
that overhead is a larger share on the M4 Pro: 2.43x on the mini versus
2.76x on the M5 Max. Both runs pass the harness stability and
MTP-correctness publication gates. Certification status is unchanged:
checkpoint Tier 1, **MTP Tier 2 pending**, AX record Candidate.

## What would move the mini

Because single-token decode already streams 97.5% of the published
bandwidth, further mini gains have to come from fewer bytes per weight pass
(lower-bit experts or KV) or more tokens per pass (higher MTP acceptance or
depth), not from kernel tuning of the existing path.
