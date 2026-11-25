# Qwen 3.8 on M5 Max: draft-tree verify, MX-format NAX rows and Flash Next prefill

**Date:** 2026-11-25 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Packs:** Qwen 3.8 27B AXQ MXFP4 and Qwen 3.8 Flash
Next AXQ MXFP4 · **Raw results:**
[`benchmarks/results/qwen38-m5-modes-20261125/`](../../benchmarks/results/qwen38-m5-modes-20261125/README.md)

## Summary

Four changes, measured against the leading peer of each model in runs
interleaved in one session (three runs per arm, medians below):

| Model and mode | Metric | AX Engine before | AX Engine now | Peer |
| --- | --- | ---: | ---: | ---: |
| 27B, DFlash2 | Decode, 64-token outputs (tok/s) | 82.83 | 111.02 | 108.88 (Splash 1.7.0) |
| 27B, DFlash2 | Effective prefill (tok/s) | 895.73 | 951.60 | 1,321.38 |
| 27B, direct | Decode (tok/s) | 35.35 | 35.35 | 34.55 (oMLX 0.8.2) |
| Flash Next, direct | Decode (tok/s) | 50.94 | 59.58 | 59.31 (oMLX 0.8.2) |
| Flash Next, direct | First text, 127 and 128 input tokens (s) | 0.69 and 0.63 | 0.34 and 0.24 | 0.38 and 0.35 |
| Flash Next, direct | First text, 1,492 input tokens (s) | 1.11 | 0.91 | 0.71 |
| Flash Next, direct | Effective prefill (tok/s) | 3,024 | 2,217 | 3,984 |

"Before" is the published 8.1.0 row for decode and 27B prefill, and the
same-session run of the build before the prefill work for the Flash Next
latencies.

What the numbers support, and what they do not:

- **27B DFlash2 decode is ahead of Splash on this harness** (64-token
  outputs, six pooled streams): 111.66, 110.97 and 111.02 against 108.78,
  108.92 and 108.88 tok/s, 2.0% at the median. On the 256-token tree-verify
  harness AX Engine is about 9% behind (100.8 against 111.3 tok/s), and its
  effective prefill is about 28% behind: Splash moves part of prefill to
  another accelerator. First text at 1,492 input tokens is 1.64 s against
  1.20 s.
- **Flash Next decode is level with oMLX.** 59.68, 59.41 and 59.58 against
  59.31, 59.35 and 59.21 tok/s: ahead in all three pairs, by 0.5% at the
  median. That is inside what separate sessions move, so it is a tie.
- **Flash Next first text is ahead at 127 to 128 input tokens and behind at
  1,492.** Both lengths got faster. The effective-prefill estimate is the
  slope between those two probes, so it fell when the short probe improved
  more than the long one: it does not isolate prefill throughput.

## 27B: one-forward draft tree and MX-format NAX rows

DFlash2 verified one chain per round, so a wrong draft at depth `d` wasted
the rows after it. The round now verifies a small tree in one target
forward: rows are picked best first from the drafter's per-position
candidates under an ancestor mask, and the accepted path is committed with
the same rows a chain would have produced. Details, the row budget and the
simulator are in [DFlash2 tree verify](../DFLASH2-TREE-VERIFY.md).

Two pieces make an eight-row forward affordable on M5:

- MLX's Neural Accelerator `qmv` handles affine packs only at 0.32.4. The
  MX-format variant of that kernel and its dispatch are ported from MLX
  914732419 under the MIT carve-out, so six to eight verify rows of an MXFP4
  pack run through it.
- The drafter projects its draft positions through the first 148,288 and
  the last 320 rows of the LM head (about 60% of its bytes) while the last
  64 tokens of the request all lie in those rows. The target verify still
  reads the whole head.

`AX_MLX_DFLASH2_TREE_ROWS` and `AX_MLX_DFLASH2_HEAD_SUBSET` are in the
[switch table](../MLX-BACKEND.md). On a Mac mini M4 Pro 64 GB, which has no
Neural Accelerator route and keeps the chain, the branch measures 31.78,
31.54 and 30.70 tok/s against 29.82, 31.10 and 31.26 for `main`, with
identical text.

## Flash Next decode

Three changes took one-row decode from 50.9 to 59.6 tok/s: submitting the
layers built so far every few layers so the GPU overlaps graph
construction (`AX_FLASH_NEXT_EAGER_DISPATCH`), concatenating projections
that read the same input (`AX_FLASH_NEXT_FUSE_PROJECTIONS`), and a 256 MB
per-buffer byte cap. The sweep is in
[gather_qmm async serialization](gather-qmm-async-serialization.md).

A decode step reads roughly 4 GB of weights (an estimate from the tensor
sizes: routed experts, GDN and attention projections, and the dense bf16
hyper-connection projections of 96 residuals). At 59.5 tok/s that is about
240 GB/s, so the step is close to memory-bound, which is consistent with
two independent engines landing within 0.5% of each other.

## Flash Next prefill

### What was slow

A scan of one warm prefill forward by row count, with stage skips, found
two cliffs and a tail of memory passes:

| Rows | Before (ms) | Now (ms) |
| ---: | ---: | ---: |
| 32 | 175 | 111 |
| 64 | 298 | 161 |
| 127 | 562 | 173 |
| 128 | 290 | 174 |
| 209 | 221 | 202 |
| 417 | 335 | 290 |
| 806 | 552 | 459 |
| 1,497 | 963 | 766 |

1. **Gated-delta recurrence below 128 rows.** The sequential recurrence
   cost about 2.1 ms per row over the 36 GDN layers (276 ms at 127 rows);
   MLX's `gated_delta_update` kernel does 128 rows in 22 ms. Its floor was
   128, from an M2 Ultra probe.
2. **Routed experts below 205 rows.** MLX runs its batched expert kernel
   only when a sorted gather holds at least four pairs per expert. With 512
   experts and ten per token that is 205 rows; below it every pair is one
   matvec (228 ms at 127 rows, 130 ms at 209).
3. **Passes over the packed residual.** Each of the 96 gated residuals
   widened the `[rows, 10240]` residual to f32, normalized it, scaled it,
   narrowed it, mixed four streams and wrote it back through separate MLX
   ops, about a dozen passes over tens of megabytes. The routed-expert tail
   did the same over `[rows * 10, 2560]`: unsort, weight, sum.

### What changed

| Change | Switch | Numerics |
| --- | --- | --- |
| MLX recurrence from 32 rows on M5 and later | `AX_MLX_FLASH_NEXT_GDN_PREFILL_MLX_MIN_SEQ` | Rounds differently from the sequential recurrence; a non-default value is a numerics-tier token |
| Pad a short sorted gather with zero rows on the last expert so it reaches the batched kernel (from 77 rows) | `AX_MLX_FLASH_NEXT_MOE_PREFILL_PAD` | Batched kernel accumulates in another order; the kill is a numerics-tier token |
| Grouped norm, stream mix and residual write as one Metal kernel each (from 64 rows) | `AX_MLX_FLASH_NEXT_HC_PREFILL_KERNELS` | Bit for bit the MLX op chain |
| Unsort, routing weights and expert sum as one Metal kernel (from 64 rows) | `AX_MLX_FLASH_NEXT_MOE_PREFILL_SUM` | Bit for bit the MLX op chain |

The exact kernels round through bf16 wherever the MLX chain stores bf16 and
follow MLX's reduction orders: `rms_norm` sums four squares per thread, then
per simdgroup, then over the simdgroup partials; the small strided sum keeps
`min(8, k)` bf16 accumulators and folds them in order. Each kernel is
compared with its chain once per process at the caller's geometry before
first use, and a kernel that fails to build, launch or match leaves the MLX
ops in place. All four changes apply on Neural Accelerator hosts only.
One-row decode and verifier windows (a handful of rows) are below every
threshold.

Contribution at 1,497 rows, same binary: the hyper-connection kernels
968 → 802 ms, the routed-sum kernel 800 → 768 ms. At 127 rows: the padding
558 → 450 ms, the recurrence floor 450 → 184 ms.

### Server first text

Same session, four probes per length, pinned prompts with a fresh marker
per request:

| Input tokens | Before (s) | Now (s) |
| ---: | ---: | ---: |
| 39 | 0.31 | 0.31 |
| 130 | 0.67 | 0.27 |
| 143 | 0.75 | 0.35 |
| 235 | 0.39 | 0.37 |
| 417 | 0.42 | 0.38 |
| 806 | 0.70 | 0.61 |
| 1,497 | 1.13 | 0.93 |

The runner prefills up to the last 16-token boundary, snapshots there for
prefix reuse, and then runs the remaining `n mod 16` rows as a second
forward. That tail costs up to 100 ms (130 against 143 tokens above), and
a 31-row head stays on the sequential recurrence (39 tokens).

### What is left at 1,497 rows

From stage skips on the scan (ms, before the kernels): routed experts 383,
GDN 275 (recurrence 88), sparse attention 71, everything else 275. The
kernels above took about 195 ms out of "everything else" and the expert
tail. What remains is mostly at a floor this engine shares with its peer:

- Reading every expert stack once per gather costs about 1.45 ms per
  419 MB projection whatever the row count, about 210 ms per forward.
- The GDN and expert projections run at the rate MLX's quantized matmul
  reaches at 0.32.4; a source build of MLX 914732419 measured no prefill
  gain (1.108 against 1.100 s at 1,492 tokens).

Not done: capturing the block-boundary state inside one forward so the
tail needs no second forward; deferring snapshot serialization past the
first token; a pipelined recurrence.

## Reproduce

The scan and the stage skips are test-only:

```bash
AX_FLASH_NEXT_PROFILE_PACK=/path/to/flash-next-pack \
AX_FLASH_NEXT_PROMPT_IDS_FILE=/path/to/ids.json \
AX_FLASH_NEXT_SCAN_ROWS=127,209,1497 AX_FLASH_NEXT_SCAN_REPEATS=3 \
AX_FLASH_NEXT_SCAN_SKIP=moe \
cargo test -p ax-engine-mlx --release --lib -- --ignored --nocapture \
  flash_next_prefill_latency_scan
```

`AX_FLASH_NEXT_SCAN_SKIP` takes `moe`, `gdn`, `gdn_recurrence` and `qsa`.
Each switch in the table above takes `0` to restore the previous route in
the same binary. The paired serving runs use the
[campaign runner](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#replay).
