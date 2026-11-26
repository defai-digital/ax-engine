# Flash Next long prompts on M5: sparse attention on the GPU tensor units

**Date:** 2026-11-26 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 · **Peer:**
oMLX 0.8.2

This follows
[the masked dense call of the day before](flash-next-qsa-masked-dense-2026-11-25.md),
whose cost per 2,048-token block kept growing with the prompt. It changes
how the sparse-attention layers attend once the history reaches 4,096
tokens. Results differ from the previous path by rounding
([numerics](#numerics)).

## Summary

Cold first text of a direct greedy request (`--mlx-mtp-policy disabled`,
one output token), M5 Max, rested host, the previous default and this one
in the order before, after, before:

| Input tokens | Masked dense (s) | Tensor-unit kernel (s) | Masked dense again (s) | Change |
| ---: | ---: | ---: | ---: | ---: |
| 4,443 | 2.14 to 2.19 | 2.13 to 2.18 | 2.14 to 2.19 | 0% |
| 6,523 | 3.16 to 3.21 | 3.10 to 3.15 | 3.16 to 3.22 | −2% |
| 9,123 | 4.47 to 4.52 | 4.30 to 4.34 | 4.50 to 4.52 | −4% |
| 13,023 | 6.73 to 6.76 | 6.27 to 6.31 | 6.75 to 6.81 | −7% |

Two requests per length (four at 4,443), 60 seconds of rest before each
length.

Against oMLX, the scan by length of the
[campaign](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#flash-next-first-text-by-input-length)
(three requests per length, twelve lengths back to back, servers in the
order AX Engine, oMLX, AX Engine with 150 seconds of rest before each):

| Input tokens | AX Engine before (s) | AX Engine now (s) | oMLX (s) | Now against oMLX |
| ---: | ---: | ---: | ---: | ---: |
| 2,961 | 1.458 | 1.422 | 1.421 | +0% |
| 4,443 | 2.269 | 2.282 | 2.176 | +5% |
| 6,523 | 3.385 | 3.346 | 3.208 | +4% |
| 9,123 | 4.934 | 4.675 | 4.497 | +4% |
| 13,023 | 7.210 | 6.745 | 6.443 | +5% |

The gap at 13,023 tokens goes from 12% to 5%. oMLX is still ahead from
4,443 tokens up. The shorter lengths do not reach the kernel and are
unchanged (36 to 2,233 tokens: +1%, −42%, +1%, +2%, +3%, −1%, −16%).

The scan reaches its long prompts after a minute of continuous load, and
that costs AX Engine more than it costs oMLX. The same lengths with 120
seconds of rest before each, several requests in a row, GPU power read
from the SoC energy counters:

| Input tokens | Engine | First text, requests in order (s) | GPU energy per request (J) | GPU power (W) |
| ---: | --- | --- | ---: | ---: |
| 2,961 | AX Engine | 1.51, 1.39, 1.39, 1.39, 1.39, 1.39 | 116 to 124 | 82 to 88 |
| | oMLX | 1.63, 1.40, 1.40, 1.40, 1.40, 1.40 | 99 to 115 | 71 to 76 |
| 4,443 | AX Engine | 2.19, 2.13, 2.13, 2.13, 2.18, 2.21 | 168 to 187 | 76 to 88 |
| | oMLX | 2.18, 2.11, 2.11, 2.12, 2.11, 2.13 | 154 to 158 | 72 to 74 |
| 6,523 | AX Engine | 3.15, 3.10, 3.09, 3.20 | 249 to 270 | 78 to 87 |
| | oMLX | 3.12, 3.06, 3.06, 3.07 | 219 to 231 | 70 to 75 |
| 9,123 | AX Engine | 4.34, 4.27, 4.45, 4.51 | 335 to 365 | 75 to 86 |
| | oMLX | 4.29, 4.22, 4.24, 4.29 | 311 to 316 | 72 to 75 |
| 13,023 | AX Engine | 6.20, 6.32, 6.50, 6.59, 6.64 | 436 to 521 | 66 to 84 |
| | oMLX | 6.09, 6.06, 6.14, 6.24, 6.28 | 402 to 450 | 66 to 74 |

- Rested, AX Engine is ahead at 2,961 tokens (1.39 against 1.40 s) and
  1 to 2% behind from 4,443 tokens up.
- AX Engine draws 84 to 88 W on the GPU where oMLX draws 72 to 76 W, for
  about the same wall time: 15 to 20% more energy per request.
- After 8 to 10 seconds at that power the host lowers AX Engine's GPU
  power (to 66 to 80 W) and its requests slow by 2 to 7%. oMLX, drawing
  less, slows later and less. An idle gap of 8 seconds restores the first
  figures; a gap in which another process keeps the GPU busy does not.

So the remaining gap in the back-to-back scan is mostly energy: the two
engines take nearly the same time per request while the host is cool, and
AX Engine spends more GPU energy doing it.

## What changed

Past 2,051 tokens every query of a sparse-attention layer attends to its
own selection: up to 512 four-token blocks and a tail, 2,051 key and value
rows out of the whole history. The masked dense call reads the whole
history for every query and masks the rest away, so one layer of a
2,048-query chunk costs it 5.6 ms at 4,096 tokens of history, 11.0 ms at
8,192 and 16.7 ms at 12,288.

The new kernel (`metal/kernels/flash_next_qsa_indexed_nax.metal`) reads
each query's selected rows in place and runs the two products on the GPU
tensor units. One layer of the same chunk costs it 5.7 to 5.9 ms at any
history.

- One threadgroup of two simdgroups handles one query and one key head.
  The 12 query heads that share the key head are the rows of a 16-row
  tile, and each simdgroup owns half of the 256-wide head dimension.
- The threadgroup walks the query's selection 32 keys at a time: partial
  scores from both halves meet in threadgroup memory, both run the same
  float32 online softmax, and each accumulates its half of the output.
- Nothing is gathered or padded. Keys and values are read from the sealed
  prefix and the tail of the K/V history where they are stored.
- Scores, softmax and accumulation are float32. A probability enters the
  second product as two float16 pieces whose sum is the float32 value;
  one float16 or bfloat16 piece would round it to 11 or 8 bits. The
  output is rounded once to the activation type.

It takes chunks of 64 or more queries over 4,096 or more tokens of
history, at any length: beyond 32,768 tokens, where the masked call stops
and padded sub-batches cost 61 to 71 ms per layer, the kernel still costs
what it costs at 4,096. Shorter histories, verifier windows and decode
keep their paths. Below 4,096 tokens the masked call is as fast or faster.

### Where it runs

The kernel's fragment layout is that of the tensor units of M5 and later.
A GPU without them compiles the same source and returns garbage, so two
things admit it:

- the host has the Neural Accelerator, and
- a self-test passed in this process: at model load the kernel attends a
  small fixed selection, and its output must match attention computed on
  the CPU. Any failure keeps the previous paths.

M4 and earlier are unchanged.

## Numerics

The kernel attends the same rows with the same scale as the paths it
replaces. Only the arithmetic differs, and it is closer to exact
attention than either of them.

One layer against float64 attention on the same bfloat16 inputs (48
queries, 8,192 tokens of history, 2,051 selected rows each, production
geometry):

| Path | RMS error | Output elements equal to the correctly rounded value |
| --- | ---: | ---: |
| Correct rounding to bfloat16 | 1.09e-4 | 100% |
| Tensor-unit kernel | 1.09e-4 | 99.9% |
| Padded sub-batches | 1.54e-4 | 58.3% |
| Masked dense call | 4.95e-4 | 23.3% |

With float32 activations the kernel is not exact: the tensor units round
float32 operands (RMS error 1.3e-4 against 1.6e-8 for the padded path),
so it is admitted for 16-bit activations only.

A row's output does not depend on the other rows of its chunk or on the
length of the history, bit for bit.

A slot past the end of a query's selection reads the first stored row with
zero weight. A finite row cannot change a bit of the output that way; a
non-finite value in that one row would reach it, as a non-finite value in
any row reaches every query through the masked dense call. Every way of
pointing such a slot at another row made the kernel 10 to 45% slower.

Next-token logits over the whole 248,320-entry vocabulary, 36 natural-text
prompts of 4,609 to 12,907 tokens (documentation and source files, 304,948
tokens in all), each configuration in its own process:

| Pair | Top token kept | Largest logit difference, median / worst | Same, among tokens above 0.1% | KL, median / worst |
| --- | ---: | ---: | ---: | ---: |
| Masked dense, two runs | bit-identical | | | |
| Tensor unit, two runs | bit-identical | | | |
| Masked dense against padded | 36 of 36 | 1.18 / 2.57 | 0.50 / 0.94 | 7.2e-4 / 3.4e-2 |
| Masked dense, 2,048- against 1,024-row chunks | 36 of 36 | 1.39 / 2.63 | 0.47 / 1.88 | 7.4e-4 / 3.1e-2 |
| Tensor unit against masked dense | 36 of 36 | 1.40 / 6.41 | 0.44 / 2.13 | 1.3e-3 / 3.4e-2 |
| Tensor unit against padded | 36 of 36 | 1.14 / 5.75 | 0.50 / 2.25 | 1.1e-3 / 4.3e-2 |

- At the median the kernel moves the logits as far as the two existing
  paths differ from each other, or as far as halving the chunk does.
- Its worst cases are larger. The 6.41 is on an unlikely token of a prompt
  whose top token leads by 10 logits (KL 2.9e-5); the largest change of any
  token's probability is 0.088 against masked dense and 0.118 against
  padded, where the two existing paths differ by 0.067.
- No top token changes in any pair.

### Warm and cold replies

A direct Flash Next request (no MTP, the default) reuses a stored prefix
only as a whole prompt with its greedy token, and under the default server
flags it stores none: all 50 requests of the control below ran a full
prefill on a server with the prefix cache and on one without. So no reply
of this mode continues a state another path computed.

The control: prompts of 4,130, 4,161, 4,500, 6,300 and 8,250 tokens, each
sent twice, then with its last three tokens changed, extended by 5, 20,
63, 64, 70 and 300 tokens, and forked 8 tokens before its end; 48 forced
output tokens and their log-probabilities. The 50 replies of the two
servers are identical.

With `--mlx-mtp-policy required` the runner does keep snapshots inside a
prompt, and there the same control does not hold, with or without the
kernel:

| MTP required, 50 requests | Replies that differ |
| --- | ---: |
| Default server, run twice | 0 |
| Prefix cache on against off, masked dense | 29 |
| Prefix cache on against off, tensor-unit kernel | 33 |

Both servers report the same prefill schedule for 48 of the 50 requests,
so in that mode something other than the reported schedule depends on the
prefix cache. That was so before this change and is not explained here.
Only two of the 50 requests restored a snapshot, so the control says
little about a restore followed by a short tail, the case in which the
rows of a warm run take another attention path than the rows of the cold
run took.

## Switch

`AX_MLX_FLASH_NEXT_QSA_INDEXED_NAX_PREFILL` is on by default on Neural
Accelerator hosts. `0` restores the masked dense call and the padded
sub-batches and is recorded as the numerics-tier token
`qsa_indexed_nax=0`, so prefix snapshots written with and without it do
not mix.

Two things for whoever merges this:

- The default numerics of Flash Next prompts past 4,096 tokens change on
  M5 and later. Two reviewers were given the measurements above and both
  chose this default; one asked for a graded multi-token benchmark before
  calling the change harmless for whole replies, which has not been run.
- A disk prefix snapshot written by an earlier build under the default
  settings carries no numerics token, so this build restores it. A reply
  served from such a snapshot continues the earlier build's state.

## What is left

- **The routed experts.** One 2,047-row forward that continues 2,048
  tokens, each engine in its own process, with one stage replaced by a
  stub at a time. The stage's cost is the difference to the whole forward:

  | Stage | AX Engine | oMLX |
  | --- | ---: | ---: |
  | Whole forward | 1,056 ms, 81.9 J | 1,028 ms, 70.4 J |
  | Routed and shared experts | 427 ms, 39.3 J | 335 ms, 28.9 J |
  | Gated-delta layers | 360 ms, 26.0 J | 167 ms, 25.4 J |
  | Sparse attention | 187 ms, 14.8 J | 194 ms, 14.0 J |

  Energy adds up across the stages (80.1 of 81.9 J and 68.3 of 70.4 J);
  wall time does not for oMLX, which overlaps its gated-delta work with
  other stages. The whole energy difference is in the experts.

  Both engines run the expert projections on the tensor units. AX Engine
  calls MLX's sorted gather; oMLX has its own kernels. One layer on the
  real weights, 20,470 (row, expert) pairs, both timed in oMLX's process:

  | Projection | MLX 0.32.4 | oMLX's kernel |
  | --- | ---: | ---: |
  | Gate and up, one packed stack | 5.2 ms, 0.52 J | 4.2 ms, 0.40 J |
  | Down | 3.0 ms, 0.29 J | 2.4 ms, 0.20 J |

  That is 1.6 ms and 0.20 J per layer, or 77 ms and 10 J per forward:
  the gap. Reading the token rows through a map and applying the
  activation inside the first kernel, which oMLX also does, changes its
  4.2 ms by less than 0.1 ms. The gain is in how the kernel tiles the rows
  of one expert and streams its weights, so the next step is a sorted
  gather kernel of AX Engine's own:
  [done the same day](flash-next-moe-gather-tiles-2026-11-26.md), which
  puts AX Engine ahead of oMLX from 127 to 13,023 tokens.
- **Block selection** still grows with the history: 15 to 36 ms for the
  indexer and 8 to 17 ms for the block scores per 2,048-token block at
  13,023 tokens (12 layers), because the partial sort that picks the 512
  blocks sorts all of them.
- **Short prompts.** 413 and 803 tokens are 1 to 3% behind oMLX, as
  before.
- **MTP required.** Replies there depend on whether the prefix cache is on
  ([above](#warm-and-cold-replies)), with or without this kernel.
- **A graded benchmark** of whole replies with the kernel on and off, and
  M5 hosts other than the M5 Max.

## Reproduce

- Kernel correctness: `cargo test -p ax-engine-mlx --lib tensor_unit` on a
  Neural Accelerator host. On other hosts the numeric tests skip.
- Logits: the ignored test `qwen_linear_prefill_logits_dump` with
  `AX_QWEN_LOGITS_LAST_ROW_STEP=1 AX_QWEN_PROFILE_CHUNK=2048`, with and
  without `AX_QWEN_LOGITS_TENSOR_UNIT=1`. Unit tests keep the kernel off
  unless they opt in, so that they run the same forwards on every host.
- Stage profile: the ignored test `flash_next_prefill_stage_profile` with
  `AX_FLASH_NEXT_PROFILE_TENSOR_UNIT=1`.
- The per-layer timings and the float64 comparison come from Python
  scripts that launch the kernel file through `mx.fast.metal_kernel` on
  random tensors of the production shape.
- GPU energy is the `GPU Energy` channel of the SoC's `Energy Model`
  IOReport group, read without elevated rights before and after a request
  or a forward. The stage stubs are `AX_FLASH_NEXT_SCAN_SKIP` of the
  ignored test `flash_next_prefill_latency_scan` on the AX Engine side.
