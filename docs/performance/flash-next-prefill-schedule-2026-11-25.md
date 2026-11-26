# Flash Next prefill schedule on M5 Max: dead prefix captures and long-prompt chunks

**Date:** 2026-11-25 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 · **Peer:**
oMLX 0.8.2 · **Raw paired runs:**
[`benchmarks/results/qwen38-m5-modes-20261125/`](../../benchmarks/results/qwen38-m5-modes-20261125/README.md)

This follows
[the kernel work of the same day](qwen38-m5-tree-verify-and-flash-next-2026-11-25.md).
It changes when the runner runs a prefill forward, not what a forward
computes.

## Summary

Cold first text of a direct greedy request (`--mlx-mtp-policy disabled`,
one output token, a fresh marker at the start of every prompt), median of
three prompts per length, each server measured on its own in one session:

| Input tokens | Previous commit (s) | Now (s) | oMLX 0.8.2 (s) |
| ---: | ---: | ---: | ---: |
| 36 | 0.33 | 0.18 | 0.17 |
| 127 | 0.30 | 0.22 | 0.35 |
| 231 | 0.38 | 0.25 | 0.23 |
| 413 | 0.41 | 0.33 | 0.30 |
| 803 | 0.63 | 0.51 | 0.44 |
| 1,492 | 0.95 | 0.82 | 0.71 |
| 2,233 | 1.70 | 1.35 | 1.34 |
| 2,961 | 2.43 | 2.01 | 1.45 |
| 4,443 | 4.09 | 3.56 | 2.22 |

AX Engine is 13 to 46% faster than its previous commit at every length. It
is ahead of oMLX at 127 tokens, within 10% of it at 36, 231 and 2,233, and
behind elsewhere: by 0.11 s at 1,492 tokens and by 1.3 s at 4,443.

In the paired serving runs (three per arm, interleaved), decode is 59.64
against 59.35 tok/s, a tie, and first text is 0.23 against 0.36 s at 127 to
128 input tokens and 0.83 against 0.71 s at 1,492.

## What changed

### No prefix capture that no request can restore

Flash Next without MTP reuses a stored prefix only as a whole prompt with
its greedy first token, so that a warm reply is the cold reply. The runner
nevertheless treated it like the other linear-attention models: it ended a
forward at the last 16-token block boundary of every unaligned prompt (and
at the prefill-chunk grid), serialized the state there (about 250 MB
whatever the prompt length), and ran the remaining rows as a second
forward. No request could read those snapshots.

Such a runner no longer stops a prefill to capture. A cold prompt runs the
schedule of an uncached prefill, and its reply is the one a server without
a prefix cache gives. A warm replay still restores the end-of-prompt state.
With MTP requested, where prefixes inside a prompt are restored, nothing
changes.

This is the whole difference up to 1,492 tokens in the table above. There
is no switch: the previous schedule produced state nothing consumed. A
reply can differ from the previous release on a low-margin token, because
one forward and two forwards round differently.

### 2,048-token chunks on long prompts

A prompt with 2,048 or more tokens left was prefilled in 512-token chunks,
a clamp measured on another model family in July. Every Flash Next forward
streams each routed expert stack once, about 0.2 s whatever the row count,
so four 512-row forwards pay that four times. Flash Next now keeps its
2,048-token chunk while the prompt fits the sparse-attention index-score
workspace at that chunk (32,768 tokens of context), on Neural Accelerator
hosts, without expert streaming.

Same binary, `AX_MLX_FLASH_NEXT_LONG_PROMPT_WIDE_CHUNK=0` against the
default, cold first text (s); the rows from 9,123 tokens up ran with
`--total-blocks 4096`:

| Input tokens | 512-token chunks | 2,048-token chunks | Change |
| ---: | ---: | ---: | ---: |
| 2,233 | 1.60 | 1.35 | −16% |
| 2,961 | 2.27 | 2.01 | −11% |
| 4,443 | 3.93 | 3.56 | −9% |
| 9,123 | 8.98 | 8.35 | −7% |
| 14,973 | 16.42 | 15.13 | −8% |
| 23,423 | 26.38 | 24.01 | −9% |
| 31,223 | 36.89 | 33.34 | −10% |

The chunk changes how rows are batched: of twelve long prompts, four gave a
different 64-token greedy continuation than with 512-token chunks. The kill
is therefore a numerics-tier token (`flash_next_wide_chunk=0`). Other model
families, and Flash Next on hosts without a Neural Accelerator, keep the
clamp.

## Measured and not shipped: capturing the boundary inside the forward

With MTP requested the boundary snapshot is used, so the split there costs
a second forward for a reason. Flash Next state is a set of immutable
arrays, which makes it possible to take the snapshot inside the single
forward: appended attention rows and index keys as a prefix of the final
state, convolution windows as slices, and the gated-delta recurrence as two
kernel calls split at the boundary. That was built and measured. It saved
50 to 130 ms of cold first text on sampled requests, and it was removed
before merge for two reasons a real-pack test produced:

- **It breaks warm equals cold.** The cold request computes the rows after
  the boundary together with the rows before it. A request that restores
  the snapshot computes them as a forward of their own. On a low-margin
  prompt the two pick different first tokens (3 of 16 synthetic cases).
- **The snapshot is not always the split state.** It equals the state of a
  prefill that stops at the boundary bit for bit only while both forwards
  use the same matmul tiling. MLX 0.32.4 changes the block tiling of a
  dense matmul when rows times output width reaches 2^20. For this model
  that is at 103, 205, 1,639 and 2,048 rows (the 10,240-, 5,120-, 640- and
  512-wide projections), and a prompt and its boundary on opposite sides of
  one of those differ by rounding in every later layer.

The second point also bounds what any row-count-dependent prefill schedule
can promise: two schedules of the same prompt agree bit for bit only when
none of their forwards crosses one of those row counts differently.

## What is left

A later change the same day closed the 1,492-token gap described below by
running the compiled elementwise stretches in prefill forwards:
[measurements](flash-next-prefill-fused-stretches-2026-11-25.md).

- **Beyond the dense attention budget.** Up to 2,051 tokens of context the
  sparse-attention layers attend densely. Past that, every row selects 512
  key blocks and attends over gathered keys, which costs about 0.45 to
  0.6 ms per row here; that is the gap from 2,961 tokens up, where first
  text grows by about 1.05 ms per input token against 0.5 ms for oMLX. The
  opt-in indexed kernel (`AX_MLX_FLASH_NEXT_QSA_INDEXED_PREFILL=1`, which
  `--mlx-numerics fast` turns on) brings 4,443 tokens to 2.97 s.
- **At 1,492 tokens** one forward takes about 0.76 s in process against
  about 0.65 s for the peer. Stage skips put the routed experts near
  350 ms, the gated-delta layers near 275 ms (the recurrence itself 88 ms)
  and sparse attention near 70 ms. A microbenchmark of one gather over a
  concatenated gate and up stack against two gathers measured 4.06 against
  4.22 ms per layer at 1,496 rows, so fusing them is worth about 8 ms per
  forward, not the tens of milliseconds hoped for.
- **Forwards wider than 2,048 rows** fall off MLX's recurrence kernel and
  are not used.

## Reproduce

The paired serving runs use the
[campaign runner](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#replay).
The scan by length sends completions with `max_tokens` 1 and `temperature`
0 to a server started with `--mlx-mtp-policy disabled
--disable-ngram-acceleration --max-concurrent-requests 1`, three warmups
first, then three prompts per length that differ in their first line. The
AX Engine columns above are from runs of each binary on its own: started
right after the peer had served the same pack, the same binary measured 5
to 9% slower from 2,961 tokens up (2.11 and 3.88 s) than it did alone.
