# Qwen 3.8 27B: greedy prompts without the block-boundary split

**Date:** 2026-11-25 · **Hosts:** MacBook Pro M5 Max 128 GB and Mac mini
M4 Pro 64 GB, macOS 27.2, pinned MLX 0.32.4 · **Pack:** Qwen 3.8 27B AXQ
MXFP4 · **Peers:** oMLX 0.8.2, Splash 1.7.0 · **Raw paired runs:**
[`benchmarks/results/qwen38-m5-modes-20261125/`](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#27b-rows-after-the-greedy-prefill-schedule-change)

This changes when the runner runs a prefill forward on the Qwen hybrids
(48 gated-delta layers and 16 full-attention layers on the 27B), not what
a forward computes.

## Summary

Medians of three rested rounds on the M5 Max, default settings, against
the previous build's rerun in the same harness:

| 27B on M5 Max | Before | Now | Peer |
| --- | ---: | ---: | ---: |
| Direct greedy decode (tok/s) | 35.31 | 35.39 | 34.56 (oMLX) |
| Direct first text, 127–128 input tokens (s) | 0.23 | 0.18 | 0.22 (oMLX) |
| Direct first text, 1,492 input tokens (s) | 1.63 | 1.57 | 1.52 (oMLX) |
| DFlash2 first text, 127–128 input tokens (s) | 0.23 | 0.18 | 0.17 (Splash) |
| DFlash2 first text, 1,492 input tokens (s) | 1.64 | 1.58 | 1.19 (Splash) |

Short prompts are now ahead of oMLX. At 1,492 tokens the direct path is
still 3% behind oMLX by default and level with it under
`--mlx-numerics fast` (1.51 s). Splash's prefill lead on DFlash2 stands.

## What the split was

The gated-delta layers carry recurrent state that cannot be trimmed, so a
prefix snapshot is valid only where a forward stopped. To have a snapshot
at the last 16-token block boundary, a cold prompt of N tokens ran as a
head up to that boundary, a single-row step, and a tail forward of 1 to 15
rows.

Reading the restore code showed what that snapshot serves: only the replay
of a prompt whose tail alone changed. Follow-up turns and every other warm
restore floor onto the 1,024-token grid rungs, which the cold run captures
at chunk boundaries it crosses anyway. An exact repeat of a greedy prompt
restores the end-of-prompt state and runs no forward.

The tail forward is expensive for its size. MLX's quantized matmul costs
about the same for 10 rows as for 32 on the M5 Max, so the split added 55
to 106 ms to every cold unaligned prompt.

## What changed

A text-only greedy request (temperature 0, no logits processors) on a
linear-attention model with live snapshots no longer splits. Its cold
prefill runs the schedule of an uncached prefill, captures the grid rungs
it crosses, and keeps the end of the prompt as the end-of-prompt state
alone. Its replay restores that state or a grid rung, never a block
boundary, including one a sampled request stored.

Which schedule a request takes is a function of the request, the model and
the host, so a warm run repeats its cold run's forwards. Sampled requests
keep the split and the boundary replay; they need the final-row logits,
which the end-of-prompt state does not hold yet.

`AX_MLX_PREFIX_END_STATE_ONLY=0` restores the split for every request.
Disk snapshots of these models written before this change miss.

### Hosts without the Neural Accelerator

On the M4 Pro the quantized matmul works in 32-row tiles and a partly
filled tile costs a whole one, about 240 ms across the 27B. A prompt that
ends 1 to 7 tokens past a multiple of 32 is cheaper with the split, whose
head ends on the tile. Those prompts keep the split on M3, M4 and M2; all
others run unsplit.

## Cold first-token time by prompt length

One direct greedy request per length, one output token.

MacBook Pro M5 Max 128 GB:

| Prompt tokens | Split | Now | Prefix cache off |
| ---: | ---: | ---: | ---: |
| 17 | 146 ms | 88 ms | 88 ms |
| 31 | 200 ms | 93 ms | 94 ms |
| 127 | 272 ms | 167 ms | 167 ms |
| 131 | 229 ms | 216 ms | 217 ms |
| 257 | 338 ms | 329 ms | 339 ms |
| 1,498 | 1,609 ms | 1,516 ms | 1,502 ms |

The 1,498-token row was measured with MLX's gated-delta recurrence on (see
below). With the default sequential recurrence the first cold request at
that length reads 1,547 ms, and 1,492 ms under `--mlx-numerics fast`. The
rows up to 127 tokens are below that recurrence's 128-row minimum.

Mac mini M4 Pro 64 GB, which carried other work during the measurement:

| Prompt tokens | Split | Now | Schedule taken |
| ---: | ---: | ---: | --- |
| 17 | 379 ms | 264 ms | unsplit |
| 31 | 573 ms | 271 ms | unsplit |
| 33 | 388 ms | 388 ms | split |
| 63 | 824 ms | 513 ms | unsplit |
| 97 | 932 ms | 887 ms | split |
| 127 | 1,352 ms | 1,049 ms | unsplit |
| 143 | 1,374 ms | 1,266 ms | unsplit |
| 257 | 2,166 ms | 2,108 ms | split |

Decode on the M4 Pro is unchanged (16.09 against 16.07 tok/s, two rounds
each). The DFlash2 arm of that host's harness did not activate DFlash2 in
either configuration, so it gives no DFlash2 number there.

## What it costs

- A greedy prompt that changes only its last 1 to 15 tokens, or whose
  end-of-prompt state was evicted, recomputes from the rung below instead
  of from the block boundary. On the M5 Max a 1,498-token prompt with its
  last three tokens changed replays in 518 to 557 ms from the 1,024-token
  rung, where the boundary replay took 93 to 97 ms. Under 1,024 tokens it
  recomputes the whole prompt.
- The exact repeat still runs no forward (1 ms).
- Sampled requests gain nothing yet.

## Replies

- On the real pack, over 48 forced tokens: exact repeats, tail edits,
  one-token tails, follow-up turns, a greedy request after a sampled one
  stored a boundary snapshot, and a sampled request after a greedy one
  each equal their cold reply. Logits were not compared.
- Against the previous build, one of the three 64-token benchmark replies
  changed. The new reply is the one the previous release gives with the
  prefix cache off, and the one oMLX gives. The split schedule did not
  reproduce it.

## MLX's gated-delta recurrence stays an opt-in

MLX's chunked `gated_delta_update` takes a 1,496-row prefill from 1,534 to
1,473 ms on the M5 Max. It was briefly on by default on M5 and later
because the benchmark replies matched the sequential recurrence under the
split schedule. Under the unsplit schedule one of the three replies
differs from the sequential recurrence's, and no quality evaluation
separates the two, so the default stays sequential:

- `--mlx-numerics fast` selects the MLX recurrence on M5 and later, and
  measured 35.34 tok/s and 1.51 s first text at 1,492 tokens against
  oMLX's 34.55 and 1.52 (three rested rounds).
- `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=1` selects it on any host, and `=0`
  keeps it off under `fast`.

## What is left

- First text at 1,492 tokens, default settings: 1.57 against oMLX's
  1.52 s. The remaining per-layer costs measured in the gated-delta layers
  are a copy that un-interleaves the packed projection and FP32 casts
  around the gate kernel, about 1 ms per layer together.
- DFlash2 first text at 1,492 tokens: 1.58 against Splash's 1.19 s.
- Sampled requests: keeping the final-row logits in the end-of-prompt
  state would let them skip the split too.
- Long generations and logits-level comparisons were not run.
