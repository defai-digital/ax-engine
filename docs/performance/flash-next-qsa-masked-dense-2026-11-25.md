# Flash Next long prompts on M5 Max: sparse attention as one masked dense call

**Date:** 2026-11-25 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 · **Peer:**
oMLX 0.8.2

This follows
[the compiled prefill stretches of the same day](flash-next-prefill-fused-stretches-2026-11-25.md),
which left AX Engine 26 to 46% behind oMLX from about 3,000 input tokens
up. It changes how the sparse-attention layers attend once a prompt is
longer than 2,051 tokens. Unlike that change it is **not bit-identical**:
results differ from the previous path by rounding
([numerics](#numerics)).

## Summary

Cold first text of a direct greedy request (`--mlx-mtp-policy disabled`,
one output token, a fresh marker at the start of every prompt), median of
three prompts per length. The servers ran one after another in the order
AX Engine, oMLX, AX Engine, with 150 seconds of rest before each:

| Input tokens | AX Engine, first run (s) | oMLX 0.8.2 (s) | AX Engine, second run (s) | AX Engine against oMLX |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.173 | 0.164 | 0.175 | +5 to +7% |
| 127 | 0.215 | 0.349 | 0.215 | −38% |
| 231 | 0.236 | 0.225 | 0.236 | +5% |
| 413 | 0.307 | 0.291 | 0.306 | +5% |
| 803 | 0.450 | 0.441 | 0.450 | +2% |
| 1,492 | 0.712 | 0.698 | 0.700 | 0 to +2% |
| 2,233 | 1.167 | 1.335 | 1.181 | −12 to −13% |
| 2,961 | 1.458 | 1.437 | 1.465 | +1 to +2% |
| 4,443 | 2.281 | 2.195 | 2.288 | +4% |
| 6,523 | 3.414 | 3.218 | 3.402 | +6% |
| 9,123 | 4.935 | 4.574 | 5.006 | +8 to +9% |
| 13,023 | 7.255 | 6.573 | 7.307 | +10 to +11% |

AX Engine was built from commit `c688907a6` with default settings. The
per-prompt records are
[first-text-by-length.jsonl](../../benchmarks/results/qwen38-m5-modes-20261125/first-text-by-length.jsonl),
and the README chart is drawn from them. An earlier scan of the same
kind, on a build that differed only outside the default path, read within
0.07 s of these at every length.

Up to 2,233 tokens the change does nothing, and those rows are the
previous note's result measured again. From 2,961 tokens AX Engine was 26
to 46% behind and is now about 2 to 11% behind, more with length.

The same binary with the new path switched off and on, each server on its
own (the request adds a 25-token wrapper to the lengths above):

| Input tokens | Padded sub-batches (s) | Masked dense (s) | Change |
| ---: | ---: | ---: | ---: |
| 1,517 | 0.713 | 0.712 | 0% |
| 2,258 | 1.264 | 1.188 | −6% |
| 2,986 | 1.870 | 1.490 | −20% |
| 4,468 | 3.407 | 2.308 | −32% |
| 6,548 | 5.515 | 3.470 | −37% |
| 9,148 | 8.132 | 4.988 | −39% |

## What changed

Past 2,051 tokens of context every query row of a sparse-attention layer
selects 512 blocks of four tokens plus the newest tokens, about 2,051
tokens, and softmax runs over those.

The previous path copied each query's selected key and value rows into
padded sub-batches of about 255 queries and attended one query per batch
entry. For a 2,048-row chunk that is about 8.6 GB of copies per layer, and
MLX runs a batch of single queries on its single-query kernel.

The new path leaves the keys and values where they are. It builds a
`[queries, tokens]` boolean mask that marks each query's selected tokens
and calls MLX's attention once over the whole history. That is the
multi-query kernel the model already uses for contexts up to 2,051 tokens.
Each query attends to exactly the tokens it selected; masked tokens
contribute exact zeros.

One sparse-attention layer, 2,048 queries, by history length:

| History tokens | Padded sub-batches (ms) | Masked dense (ms) |
| ---: | ---: | ---: |
| 4,096 | 61.2 | 6.0 |
| 8,192 | 61.6 | 11.8 |
| 32,768 | 66.8 | 46.7 |
| 65,536 | 71.0 | 97.2 |

The masked call reads every key, so its cost grows with the history; the
copy's does not. The path is therefore used for chunks of 64 or more
queries up to 8,192 tokens of history and for chunks of 512 or more up to
32,768. Other chunks, verifier windows and decode steps keep their paths.

One 2,048-row prefill forward that starts at 2,048 tokens of context went
from 1,798 ms to 918 ms. The same forward with the sparse-attention layers
skipped takes 737 ms, so those twelve layers went from about 1,060 ms to
about 180 ms.

## Numerics

The masked call sums a query's keys in cache order and in different
groups than the padded call, which summed them in selection order. Both
accumulate in float32 and store bfloat16.

On the real pack, a 4,443-token prompt in 2,048-row chunks, against the
padded sub-batches:

- The chunk below the 2,051-token budget is bit-identical (the path is
  not used).
- In the first sparse-attention layer of the next chunk 13% of the output
  elements differ, by at most one bfloat16 step (0.0156 at magnitude 2).
- Later layers amplify that: the last row's logits differ by up to 1.7 at
  the end of a chunk, with the same top token and a margin of 5.3 against
  5.4. The existing indexed kernel (`--mlx-numerics fast`) moves the same
  logits by up to 1.5 on this prompt.
- 256 greedy tokens after that prompt are identical.

Eight other prompts of 3,249 to 14,732 tokens (source files with a
question), 160 greedy tokens each, path off against on: two replies are
identical and six diverge somewhere, at the first near-tie. The two
diverging pairs that were read say the same thing in different words. The
off arm run twice gave eight identical replies.

Unit tests hold the path to the per-query attention within 2e-5 in
float32, and bit for bit to itself across a chunk split, a segmented
history, a second batch row, and the device and host forms of the same
selection. Tokens outside a query's selection cannot change a bit of its
output.

A row's result does not depend on the other rows of its chunk. On hosts
without a Neural Accelerator it can depend on the length of the history
the call reads, as the dense path below 2,052 tokens already does.

## Switch and hosts

`AX_MLX_FLASH_NEXT_QSA_MASKED_DENSE_PREFILL` is on by default on Neural
Accelerator hosts (M5 and later). `0` restores the padded sub-batches and
is recorded as the numerics-tier token `qsa_masked_dense=0`, so prefix
snapshots written with and without it do not mix.

Other hosts keep the padded sub-batches. On a Mac mini M4 Pro 64 GB the
same kernel is about six times slower: one layer of 2,048 queries takes
35 ms masked against 177 ms padded at 4,096 tokens and 70 against 176 at
8,192, but 298 against 196 at 32,768. The path would help there up to
about 16,000 tokens and needs its own bound and an end-to-end measurement
first.

Where the masked call covers a chunk it also takes precedence over the
indexed kernel of `--mlx-numerics fast`, which is slower.

Two things for whoever merges this:

- The default numerics of long Flash Next prompts change on M5 and later.
  Two reviewers were asked: one chose this M5-only default, the other a
  default for every host.
- A disk prefix snapshot written by an earlier build under the default
  settings carries no numerics token, so this build restores it. A reply
  served from such a snapshot is the earlier build's reply.

## What is left

- **Long prompts.** From 2,961 tokens each further token costs about 0.54
  to 0.60 ms against 0.49 to 0.52 ms for oMLX. The masked call's share
  grows with the history (6 ms per layer at 4,096 tokens, 12 at 8,192).
  The peer attends the selected keys in place with a kernel on the tensor
  units, whose cost does not depend on the history.
- **Up to 803 tokens** AX Engine is 0.01 to 0.02 s behind, as before,
  except at 127 tokens, where it is ahead.
- **Hosts without a Neural Accelerator**, as above.

## Reproduce

The scan by length sends completions with `max_tokens` 1 and `temperature`
0 to a server started with `--mlx-mtp-policy disabled
--disable-ngram-acceleration --max-concurrent-requests 1`, three warmups
first, then three prompts per length that differ in their first line.
Rest the host between servers: a scan started right after another one
read 0.83 s at 1,492 tokens where the rested one reads 0.70 s.

The drift numbers come from the ignored test
`flash_next_indexed_prefill_attention_drift_on_real_pack` of
`ax-engine-mlx` with `AX_FLASH_NEXT_DRIFT_CANDIDATE=masked_dense` and
`AX_FLASH_NEXT_INDEXED_CHUNK_SIZE=2048`. The per-layer timings come from a
Python script that issues the same MLX calls on random tensors of the
production shape (24 query heads, 2 key heads, head dimension 256, 2,051
selected tokens).
