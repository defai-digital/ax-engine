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

| 27B on M5 Max | Before | Schedule change | With the MLX recurrence (now) | Peer |
| --- | ---: | ---: | ---: | ---: |
| Direct greedy decode (tok/s) | 35.31 | 35.39 | 35.46 | 34.67 (oMLX) |
| Direct first text, 127–128 input tokens (s) | 0.23 | 0.18 | 0.17 | 0.22 (oMLX) |
| Direct first text, 1,492 input tokens (s) | 1.63 | 1.57 | 1.50 | 1.51 (oMLX) |
| DFlash2 first text, 127–128 input tokens (s) | 0.23 | 0.18 | 0.18 | 0.17 (Splash) |
| DFlash2 first text, 1,492 input tokens (s) | 1.64 | 1.58 | 1.50 | 1.19 (Splash) |

Direct greedy is now ahead of oMLX on all three measures. The lead at
1,492 tokens is 9 ms: each of the six AX Engine probes (1.497 to 1.501 s)
is below each of oMLX's (1.507 to 1.513 s). Splash's prefill lead on
DFlash2 stands.

A rerun on 2026-11-26, after the server stopped copying the tokenizer for
every request ([note](server-tokenizer-shared-2026-11-26.md)), reads 0.16 s
at 127 to 128 tokens and 1.487 to 1.491 s at 1,492 against oMLX's 1.510 s:
a lead of 20 ms. Decode is unchanged (35.46 against 34.68 tok/s).

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

With `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0` the first cold request at
1,498 tokens reads 1,547 ms. The rows up to 127 tokens are below the MLX
recurrence's 128-row minimum.

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
- The schedule change alone changes one of the three 64-token benchmark
  replies (the second) against the previous build. The new reply is the
  one the previous release gives with the prefix cache off, and the one
  oMLX gives. The split schedule did not reproduce it.

## MLX's gated-delta recurrence on M5 and later

MLX's chunked `gated_delta_update` takes a 1,496-row prefill from 1,534 to
1,473 ms on the M5 Max. It rounds differently from AX's sequential
recurrence, and under the unsplit schedule one of the three 64-token
benchmark replies differs between the two. A changed reply says nothing
about how far the model moved, so the next-token logits were compared
directly: 46 natural-text prompts (raw prefixes of ten documentation and
source files, 173 to 2,903 tokens, 57,124 tokens in all), the full
248,320-entry vocabulary, each configuration in its own process. Two runs of
one configuration are bit-identical.

| Change | Top token kept | Largest logit difference, median / worst | KL, median / worst |
| --- | ---: | ---: | ---: |
| Schedule only (sequential recurrence, split against unsplit) | 46 of 46 | 0.16 / 0.38 | 8.9e-5 / 1.9e-3 |
| Recurrence only (unsplit schedule) | 46 of 46 | 0.24 / 1.06 | 1.6e-4 / 2.5e-3 |
| Schedule only (MLX recurrence) | 46 of 46 | 0.16 / 1.09 | 5.9e-5 / 2.1e-3 |

- The bf16 model already moves by 0.16 to 0.38 logits when only the
  chunking of the prefill changes. The recurrence moves it by about 1.5
  times that at the median.
- The recurrence's largest differences are on unlikely tokens: the
  most-moved token is ranked 18th at best (median rank 2,487). Among tokens
  above 0.1% probability its worst difference is 0.38 against the
  schedule's 0.19, and the largest change of any token's probability is
  0.030 for both.
- A prompt that repeats one sentence hundreds of times is unstable under
  either change. Cut in the middle of the repetition (27 cuts of the
  benchmark's prefill prompts), the top token flips in 8 cases for the
  schedule change and in 13 for the recurrence. The complete prompts, which
  end in a question, agree in every configuration.

A short scored set was run as well: 80 chat prompts, each a natural-text
context of 300 to 1,800 tokens followed by a question with one answer (30
two-digit sums, differences and products, 30 six-digit codes planted in
the context, 20 capital cities), 24 output tokens. Both recurrences answer
all 80 correctly and their 80 replies are token-identical. The set is at
its ceiling, so it shows no difference on short answers and cannot detect a
small one.

On this evidence the recurrence is on by default where the Neural
Accelerator is present. `AX_MLX_QWEN_GATED_DELTA_PREFILL_MLX=0` restores the
sequential recurrence and `=1` selects the MLX one on other hosts; either
deviation carries a numerics-tier token. Limits: no graded benchmark with
hard items was run, generations longer than 64 tokens were not compared,
and only the M5 Max was measured. The logit comparison is reproducible with
the ignored test `qwen_linear_prefill_logits_dump`.

## What is left

- DFlash2 first text at 1,492 tokens: 1.50 against Splash's 1.19 s. Splash
  runs part of its prefill on the Neural Engine; no change to the GPU path
  measured so far reaches that.
- The direct lead at 1,492 tokens is under 1%. The remaining per-layer
  costs measured in the gated-delta layers are a copy that un-interleaves
  the packed projection and FP32 casts around the gate kernel, about 1 ms
  per layer together. The copy alone, timed on 2026-11-26 in front of the
  convolution it feeds, adds 0.18 ms per layer at 1,492 rows and 0.30 ms at
  2,048: about 9 ms over the 48 gated-delta layers, 0.6% of first text.
  Removing it needs a convolution that reads the packed layout, because the
  default prefill path runs MLX's `conv1d` on a contiguous input.
- DFlash2 tree rows: 16 rows decode 1 to 6% faster than the default 8 at
  64 to 256 output tokens on seven prompts, but change the greedy text
  ([note](qwen38-27b-dflash2-tree-rows-2026-11-26.md)).
- Sampled requests: keeping the final-row logits in the end-of-prompt
  state would let them skip the split too.
- A graded evaluation of the MLX recurrence with hard items, generations
  longer than 64 tokens, and M5 hosts other than the M5 Max.
