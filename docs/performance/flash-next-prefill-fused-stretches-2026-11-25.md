# Flash Next prefill on M5 Max: the compiled elementwise stretches at prefill width

**Date:** 2026-11-25 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 · **Peer:**
oMLX 0.8.2 · **Raw paired runs:**
[`benchmarks/results/qwen38-m5-modes-20261125/`](../../benchmarks/results/qwen38-m5-modes-20261125/README.md)

This follows
[the prefill-schedule change of the same day](flash-next-prefill-schedule-2026-11-25.md),
which left AX Engine 0.11 s behind oMLX at 1,492 input tokens. It changes
how a prefill forward runs its elementwise work, not what the forward
computes: logits and state are bit-identical.

## Summary

Cold first text of a direct greedy request (`--mlx-mtp-policy disabled`,
one output token, a fresh marker at the start of every prompt), median of
three prompts per length, each server measured on its own:

| Input tokens | Previous commit (s) | Now (s) | oMLX 0.8.2 (s) |
| ---: | ---: | ---: | ---: |
| 36 | 0.18 | 0.17 | 0.17 |
| 127 | 0.22 | 0.22 | 0.35 |
| 231 | 0.25 | 0.24 | 0.23 |
| 413 | 0.33 | 0.31 | 0.30 |
| 803 | 0.51 | 0.45 | 0.44 |
| 1,492 | 0.82 | 0.70 | 0.71 |
| 2,233 | 1.35 | 1.23 | 1.34 |
| 2,961 | 2.01 | 1.82 | 1.45 |
| 4,443 | 3.56 | 3.24 | 2.22 |

AX Engine is 4 to 14% faster than its previous commit from 231 tokens up.
Against oMLX it is ahead at 127 and 2,233 tokens, level at 1,492, within
0.01 to 0.02 s behind from 36 to 803, and still well behind from 2,961
tokens up, where sparse attention selects key blocks
([what is left](#what-is-left)).

In the paired serving runs (three per arm, interleaved), decode is 59.36
against 59.39 tok/s, a tie as before. First text is 0.22 against 0.36 s at
127 to 128 input tokens and 0.705 against 0.706 s at 1,492, where it was
0.83 against 0.71.

## What changed

Flash Next runs the elementwise work between its projections (norms,
gates, activations, scales, casts) as compiled closures, one fused kernel
per chain instead of one GPU pass per operation. There are 19 such
compiled stretches, each proven bit-identical to its chain of MLX
operations. They were built for the decode step and were only allowed
there: a stretch ran for one row (eight at most), and a prefill forward
ran every chain operation by operation.

A profile of one prefill forward against the peer's, stage by stage,
showed the difference was not in a projection or in the recurrence kernel
but spread over those chains. At 1,497 rows every operation of a chain
reads and writes a tensor of up to 15 million elements.

A prefill forward of 64 rows or more now runs the same compiled stretches.
One forward in process, the previous path against the compiled one:

| Rows | MLX operations (ms) | Compiled stretches (ms) | Change |
| ---: | ---: | ---: | ---: |
| 64 | 160.5 | 156.6 | −2% |
| 103 | 166.6 | 162.3 | −3% |
| 206 | 199.8 | 189.6 | −5% |
| 777 | 449.3 | 396.0 | −12% |
| 1,411 | 730.0 | 622.0 | −15% |
| 1,497 | 769.7 | 653.7 | −15% |

The same forward in the peer's runtime, with the kernels its server
installs, takes about 657 ms at 1,497 rows.

### Same numbers

The final logits and the complete model state (every layer's convolution
window, recurrent state, attention keys and index keys) were hashed after
each forward, with the stretches off and on:

- six lengths from 64 to 1,497 rows, each from a fresh process: identical;
- 41 forwards in one process over 36 distinct lengths, with lengths
  repeated at the end: identical. That run takes every closure past its
  trace limit and through a rebuild.

Unit tests compare the gated-delta layer, the routed experts, the
hyper-connection read and write, the rotary embedding and the attention
gate against the MLX operations bit for bit at 64 and 97 rows, on closures
that serve one row count, another, and the first again.

There is no numerics token: the default `exact` preset keeps its
replies.

### Closures and prompt lengths

A compiled closure is traced once per input shape, and MLX searches a
closure's traces one by one on every call. A server sees a new prompt
length on most requests, so the registry bounds that:

- Prefill has its own closure per stretch, apart from the decode one. A
  decode closure keeps the single trace its hot path looks up, and a
  failure at prefill width leaves decode compiled.
- A prefill closure is rebuilt after it has traced 32 input shapes.
- Every new shape is evaluated once before its output is used, where a
  fallback to the MLX operations is still possible. A stretch that fails
  stays on the MLX operations for the rest of the process.

A sequence of 40 prompt lengths from 111 to 2,141 tokens, three cold
requests per length, same binary with the switch off and on: the first
request of every length was faster with the stretches on (the smallest
difference was 0.6 ms, at 111 tokens), and the 40 first requests took
25.4 s against 28.6 s (−11%). In process, the first forward at a new
length is 2 to 3 ms slower than the MLX operations at 64 and 103 rows,
which the next forward of that length wins back, and faster from 206 rows.

### What stays decode-only

Three of the 22 stretches are hand-written Metal kernels rather than
compiled closures. At prefill width they gained about 7 ms per forward and
cost 85 to 141 ms the first time each row count was seen. They keep their
decode-only gate. Rows between 9 and 63 also stay on the MLX operations.

### Switch and hosts

`AX_FLASH_NEXT_FUSE_PREFILL` is on by default on Neural Accelerator hosts
(M5 and later), the only ones that can hold this pack resident and so the
only ones it was measured on. `0` turns it off; `1` turns it on for other
hosts. `AX_FLASH_NEXT_FUSE=0` still turns every stretch off, decode
included.

## What is left

A later change the same day attends those long prompts with one masked
dense call per layer and brought the gap from 2,961 tokens up to about 2 to 11%:
[measurements](flash-next-qsa-masked-dense-2026-11-25.md).

- **From 2,961 tokens up** the gap is unchanged in kind: past 2,051 tokens
  of context the sparse-attention layers select 512 key blocks per row and
  attend over gathered keys. First text grows by about 0.95 ms per input
  token there, against 0.5 ms for oMLX.
- **Up to 803 tokens** AX Engine is 0.01 to 0.02 s behind.
- **The routed experts** are the largest remaining stage of a forward. The
  peer runs the gate and up projections of an expert and the SwiGLU
  between them as one kernel; here they are two gathers and a compiled
  stretch.
- **Hosts without a Neural Accelerator** run the previous path until
  someone measures the switch there.

## Reproduce

The paired serving runs use the
[campaign runner](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#replay).
The scan by length sends completions with `max_tokens` 1 and `temperature`
0 to a server started with `--mlx-mtp-policy disabled
--disable-ngram-acceleration --max-concurrent-requests 1`, three warmups
first, then three prompts per length that differ in their first line.

The in-process numbers come from the ignored test
`flash_next_prefill_latency_scan` of `ax-engine-mlx`, which runs one
prefill forward per row count and reports the fastest of its warm repeats.
A unit-test build starts with every stretch off, so the compiled arm sets
`AX_FLASH_NEXT_FUSE=1 AX_FLASH_NEXT_FUSE_PREFILL=1`. Keep the list of row
counts short: on this laptop the same 1,497-row forward went from 655 to
742 ms over 30 back-to-back repeats, with the stretches on or off alike.
