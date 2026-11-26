# Flash Next on M5: the routed expert projections in 64-row tiles

**Date:** 2026-11-26 · **Host:** MacBook Pro M5 Max 128 GB, macOS 27.2,
pinned MLX 0.32.4 · **Pack:** Qwen 3.8 Flash Next AXQ MXFP4 · **Peer:**
oMLX 0.8.2

[The note on the tensor-unit sparse attention](flash-next-qsa-tensor-unit-2026-11-26.md#what-is-left)
ended with the routed experts: they held the whole GPU energy difference to
oMLX. This note changes the kernel that computes their projections during
prefill. Results are bit-identical to the previous path
([numerics](#numerics)).

## Summary

Cold first text of a direct greedy request (`--mlx-mtp-policy disabled`,
one output token), M5 Max, one build with the kernel on, off and on again.
Every arm is a fresh server that gets the same prompts, three per length,
with 150 seconds of rest before it. Medians:

| Input tokens | 64-row tiles (s) | MLX's gather (s) | 64-row tiles again (s) | Change |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.162 | 0.167 | 0.166 | −2% |
| 127 | 0.186 | 0.206 | 0.189 | −9% |
| 231 | 0.220 | 0.226 | 0.219 | −3% |
| 413 | 0.282 | 0.296 | 0.285 | −4% |
| 803 | 0.414 | 0.440 | 0.417 | −6% |
| 1,492 | 0.641 | 0.686 | 0.640 | −7% |
| 2,233 | 1.031 | 1.129 | 1.048 | −8% |
| 2,961 | 1.338 | 1.458 | 1.346 | −8% |
| 4,443 | 2.091 | 2.266 | 2.091 | −8% |
| 6,523 | 3.068 | 3.325 | 3.084 | −7% |
| 9,123 | 4.443 | 4.784 | 4.411 | −7% |
| 13,023 | 6.387 | 6.800 | 6.335 | −6% |

Against oMLX, the scan by length of the
[campaign](../../benchmarks/results/qwen38-m5-modes-20261125/README.md#flash-next-first-text-by-input-length),
servers in the order AX Engine, oMLX, AX Engine with 150 seconds of rest
before each:

| Input tokens | AX Engine before (s) | Before against oMLX | AX Engine now (s) | oMLX (s) | Now against oMLX |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 36 | 0.167 | +1% | 0.168 | 0.165 | +2% |
| 127 | 0.205 | −42% | 0.189 | 0.350 | −46% |
| 231 | 0.226 | +1% | 0.219 | 0.232 | −6% |
| 413 | 0.296 | +2% | 0.285 | 0.291 | −2% |
| 803 | 0.440 | +3% | 0.416 | 0.432 | −4% |
| 1,492 | 0.686 | −1% | 0.642 | 0.692 | −7% |
| 2,233 | 1.108 | −16% | 1.046 | 1.327 | −21% |
| 2,961 | 1.422 | 0% | 1.350 | 1.421 | −5% |
| 4,443 | 2.282 | +5% | 2.102 | 2.170 | −3% |
| 6,523 | 3.346 | +4% | 3.087 | 3.191 | −3% |
| 9,123 | 4.675 | +4% | 4.405 | 4.522 | −3% |
| 13,023 | 6.745 | +5% | 6.352 | 6.441 | −1% |

AX Engine is ahead at eleven of the twelve lengths, including the range
from 4,443 tokens up where it was 4 to 5% behind, and 3 ms behind at 36
tokens. The "before" columns are the scan of the previous build
(commit `7a8767085`) from the note linked above. The 36-token prompt is
below the sorted gather and does not reach the kernel; its rows show the
spread between runs.

The same lengths with 120 seconds of rest before each and several requests
in a row, GPU power read from the SoC energy counters:

| Input tokens | Engine | First text, requests in order (s) | GPU energy per request (J) | GPU power (W) |
| ---: | --- | --- | ---: | ---: |
| 2,961 | AX Engine | 1.42, 1.29, 1.29, 1.29, 1.29, 1.29 | 106 to 114 | 80 to 84 |
| | oMLX | 1.48, 1.40, 1.40, 1.40, 1.40, 1.40 | 101 to 105 * | 70 to 75 * |
| 4,443 | AX Engine | 2.06, 2.00, 2.00, 1.99, 2.00, 2.04 | 161 to 211 | 79 to 106 |
| | oMLX | 2.19, 2.11, 2.12, 2.11, 2.12, 2.13 | 151 to 160 | 72 to 75 |
| 6,523 | AX Engine | 2.95, 2.89, 2.90, 2.98 | 224 to 278 | 75 to 96 |
| | oMLX | 3.13, 3.06, 3.06, 3.07 | 224 to 227 | 72 to 74 |
| 9,123 | AX Engine | 4.06, 3.99, 4.10, 4.18 | 299 to 328 | 71 to 82 |
| | oMLX | 4.29, 4.22, 4.26, 4.28 | 276 to 348 | 64 to 82 |
| 13,023 | AX Engine | 5.80, 5.74, 5.85, 6.21, 6.21 | 381 to 479 | 61 to 83 |
| | oMLX | 6.09, 6.06, 6.16, 6.26, 6.31 | 420 to 453 | 66 to 75 |

\* Two oMLX samples at 2,961 tokens read 0 and 40 W, a gap in the power
log; they are left out of the ranges.

With rest, AX Engine is ahead at every length here, by 3 to 8%; the
previous build was 0 to 2% behind from 4,443 tokens up. Its GPU energy per
request is about 5% above oMLX's, where the previous build drew 10 to 15%
more (the 106 W and 96 W readings are single samples). At 13,023 tokens the host lowers AX
Engine's GPU power to 61 W from the fourth request on, and its first text
rises to 6.21 s, still below oMLX's 6.26 and 6.31 s at the same positions.

## What changed

A Flash Next layer routes every token to 10 of 512 experts. Prefill sorts
the (token, expert) pairs by expert and computes the gate, up and down
projections of all pairs with one gather each, so the weights of an expert
are read once per run of rows.

MLX's kernel for that gather works on the tensor units in row tiles of one
expert: a threadgroup dequantizes a 64 by 64 piece of the expert's weights
and multiplies up to one tile of rows with it. It takes 32-row tiles when
the stack holds fewer than 64 pairs per expert, and 64-row tiles otherwise.
The rule counts all 512 experts. Real routing does not use them evenly.
One 2,047-row chunk of the benchmark prompt, per layer:

| | Fewest | Most |
| --- | ---: | ---: |
| Experts with any row, of 512 | 84 | 327 |
| Rows of the median expert in use | 11 | 158 |
| Rows of the busiest expert | 759 | 2,047 |

A busy expert fills many tiles, and every tile dequantizes the same
weights again. A 64-row tile halves that for the busy experts. A 16-row
fragment past the last row of a short run is skipped, so a short run costs
what it costs in a 32-row tile.

AX Engine now runs these three projections through its own port of MLX's
kernel with 64-row tiles (the kernel source and license notice are in
[`THIRD_PARTY_NOTICES`](../../THIRD_PARTY_NOTICES)). It accumulates in the
order MLX's kernel does.

The projections alone, on the real weights and the real routing of twelve
layers, gate, up and down together, each kernel launched 20 times:

| Rows in the chunk | MLX 0.32.4 | 64-row tiles | 96-row tiles | 128-row tiles |
| ---: | ---: | ---: | ---: | ---: |
| 126 | 55.5 ms, 1.41 J | 48.1 ms, 1.09 J | 48.6 ms, 1.17 J | 51.5 ms, 1.17 J |
| 412 | 61.1 ms, 2.34 J | 56.2 ms, 1.95 J | 57.9 ms, 2.01 J | 60.0 ms, 2.02 J |
| 802 | 75.6 ms, 3.79 J | 65.3 ms, 3.03 J | 68.0 ms, 3.00 J | 69.0 ms, 2.99 J |
| 1,491 | 95.7 ms, 6.28 J | 81.2 ms, 4.83 J | 84.9 ms, 4.69 J | 87.2 ms, 4.50 J |
| 2,047 | 109.5 ms, 7.89 J | 91.3 ms, 5.90 J | 93.4 ms, 5.66 J | 94.6 ms, 5.48 J |

The 126-row chunk is one MLX computes with zero rows appended
(`AX_MLX_FLASH_NEXT_MOE_PREFILL_PAD`), which the port does not need. 64
rows are the fastest at every length. 128 rows draw the least energy on
long chunks and are 4% slower.

Measured and not adopted, all on the 2,047-row chunk:

- Two weight tiles, so that a step dequantizes the next piece while it
  multiplies with the current one and needs one barrier instead of two: 6%
  slower, 1% less energy.
- One or four simdgroups along the rows of a tile instead of two: 9 to 18%
  slower.
- 128 output columns or 128 K values per step: no gain.

### Where it runs

- Prefill forwards whose gather is sorted: chunks of at least 64 rows.
- Only where MLX would run its batched expert kernel: at least four pairs
  per expert of the stack (205 rows for Flash Next), or the range the
  padding switch covers (from 77 rows), which the port serves without the
  padding. Below that MLX computes one pair at a time with other rounding,
  and those gathers stay on MLX.
- MXFP4 stacks with groups of 32, at least eight experts, input and output
  widths that are multiples of 64, and 16-bit activations.
- Hosts with the Neural Accelerator (M5 and later). The kernel is written
  for the tensor units' fragment layout. On other hosts nothing changes.
- One-row decode and MTP verifier windows do not sort and are unchanged.

## Numerics

The port is bit-identical to MLX's batched gather: both accumulate
ascending 16-value steps of K into float32 and round once at the store,
and a row's result does not depend on its tile.

- Unit tests compare the port with MLX's gather bit for bit on runs that
  fill a tile, end inside each fragment row, skip experts, leave one row
  and cross the scheduler's 32-expert scan (100 experts), at both Flash
  Next projection shapes and for both 16-bit activation types. A module
  test compares the whole routed block, sort and unsort included, with the
  port on and off, once where MLX needs the padding and once where it does
  not.
- On the real pack, the final logits and the whole state of a prefill
  forward (recurrent state and K/V of all 48 layers) hash identically with
  the port on and off at 231, 803 and 2,047 rows from an empty history and
  at 2,047 rows after 2,048 tokens.
- Forced 48-token greedy replies to 9 prompts of 100, 127, 231, 413, 803, 1,492, 2,233, 4,443, 9,123 tokens from a server with the
  port and from one without it: 9 of 9 are token-identical.
- At model load the port runs each projection shape against MLX's gather
  on a fixed 42-expert stack. A kernel that fails to build, launch or match
  leaves the routed experts on MLX's gather for the life of the process and
  logs a warning.

## Switch

`AX_MLX_FLASH_NEXT_MOE_GATHER_NAX_PREFILL=0` turns the port off. It is on
by default on M5 and later and cannot be turned on elsewhere. The switch
changes no rounding, so it is not a numerics-tier token.

## Notes for the merger

- The kernel is a port under ADR-053, which names `fp_quantized_nax.h`.
  Two points go beyond the earlier port and need the maintainer's explicit
  decision before merge: this is a gather used in prefill, where the
  earlier port was a matvec used in the verifier, and the tile scheduling
  helper comes from MLX's `utils.h`.
- Mac mini M4 Pro 64 GB: the library tests pass (2,355) and the path is
  off, so first text and decode there are unchanged by construction. No
  timing was taken on that host.

## What is left

- **The 36-token row** does not reach the sorted gather (it starts at 64
  rows) and stays within run-to-run spread of oMLX.
- **Block selection** of the sparse-attention layers still grows with the
  history (23 to 53 ms per 2,048-token block at 13,023 tokens). The score
  projection, the flag scan that each row's first thread runs over all
  scores, and the partial sort should be timed apart before any of them
  changes; the partial sort already sorts only the 512 selected blocks.
- **Gate and up in one launch** would read the activations once instead of
  twice. It needs a packed copy of both stacks, which a Flash Next pack does
  not ship. Measured with 64-row tiles on five layers it takes 15 to 20%
  less time than the two launches at about the same energy.
- **Hosts other than the M5 Max.** Only this host was timed. The port is
  off before M5.
- **Sampled requests** and **MTP required** were not measured with the
  port; it changes no rounding, so their replies do not change.

## Reproduce

```sh
# Kernel against MLX, bit for bit (needs an M5 or later for the comparison):
cargo test -p ax-engine-mlx --release --lib -- moe_gather ported_prefill_gather

# One arm of the scan: a fresh server, three prompts per length.
ax-engine-server --mlx --mlx-model-artifacts-dir <pack> --mlx-mtp-policy disabled \
  --disable-ngram-acceleration --stream-experts off --max-concurrent-requests 1
# Kernel off for the same server:
AX_MLX_FLASH_NEXT_MOE_GATHER_NAX_PREFILL=0 ax-engine-server ...
```

The scan sends `/v1/completions` requests with `max_tokens` 1 and
`temperature` 0 and reads the time to the first streamed text. Its prompts
repeat one sentence 1 to 1,000 times after a first line that differs per
request.
