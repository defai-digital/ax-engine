# Gemma 4 12B AX-only direct and optimistic MTP

**Date:** 2026-07-10
**Status:** Diagnostic speed-ceiling result; optimistic sampled MTP is not an exact publication candidate.

## Contract

- Model: `ax-local/gemma-4-12b-it-assistant-mtp` (6-bit affine, group size 64)
- Host: Apple M5 Max, 128 GB unified memory, macOS 26.5.2
- Runtime: AX Engine release build at `08c867e6f5b32acc8f717ba532288f18338d2b00`; MLX 0.32.0
- Prompt suite: chat-templated `flappy`, four cases (309, 382, 521, and 316 input tokens)
- Decode: 1000 generated tokens; `temperature=0.6`, `top_p=0.95`, `top_k=20`; seed 0
- Measurement: two warmups, five measured repetitions, 15 s cooldown, 10 s inter-case cooldown
- Common settings: AX Engine only, cold prefill (prefix cache disabled), assistant depth 2, n-gram stacking disabled
- Optimistic setting: `AX_MLX_MTP_OPTIMISTIC=1`

## Results

Case-level values are medians of the five measured repetitions. The aggregate is
the median of the four case medians.

| Prompt case | Direct decode | Optimistic MTP decode | Speedup |
| --- | ---: | ---: | ---: |
| `flappy_pipes` | 37.94 tok/s | 95.76 tok/s | 2.52x |
| `flappy_score_gates` | 37.95 tok/s | 83.65 tok/s | 2.20x |
| `flappy_collision_checks` | 37.87 tok/s | 94.44 tok/s | 2.49x |
| `flappy_sound_channels` | 37.98 tok/s | 95.28 tok/s | 2.51x |
| **Aggregate** | **37.95 tok/s** | **94.86 tok/s** | **2.50x** |

| Metric | Direct | Optimistic MTP |
| --- | ---: | ---: |
| Prefill | 466.1 tok/s | 442.6 tok/s |
| Runner TTFT | 763 ms | 787 ms |
| Client-wall TTFT | 765 ms | 789 ms |
| Peak RSS | 11.58 GiB | 11.58 GiB |

## Correctness status

The optimistic route used assistant MTP: it recorded zero direct-fallback steps,
13,045 drafted tokens, 13,040 accepted tokens, and five rejected cycles across
the four prompt cases. It is nevertheless an approximate sampled route. All five
`flappy_score_gates` optimistic trials emitted token IDs that differed from the
same-seed direct trials; the other three cases matched. Therefore this result is
an approximate MTP speed ceiling only and is not a publication-quality sampled
MTP claim.

For the current correctness-preserving sampled policy, a requested Gemma MTP
route falls back to direct decoding: 38.02 tok/s aggregate decode, effectively
the 37.95 tok/s direct baseline.

## Raw-result provenance

The run completed with schema `ax.mlx_inference_stack.v2`. The local raw result
files used to derive this committed summary had SHA-256 values:

- Direct: `3c3d576d4114b36a49afdff55d58ae6ee925cfa57e6292f2c6f03d6a369b7480`
- Optimistic MTP: `b0164df317875312575b7f55513540c7c82bade9b5d6f4b7d847b81ef60b051a`
