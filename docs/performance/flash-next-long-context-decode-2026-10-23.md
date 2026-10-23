# Flash Next long-context decode — 2026-10-23

Qwen 3.8 Flash Next (`qwen4_exp`, MXFP4 MTP pack `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`
at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`) decoded at about 42 tok/s with 2k tokens of context and
32.7 tok/s with 39k on an Apple M5 Max 128 GB. This note records where the difference came from, what
changed, and what was measured. It is a single-host engineering record, **not a speedup claim and not
a certification run**: no comparison runtime is involved and the host is a campaign host. It builds on
[the n-gram read and fused-stretch note](flash-next-ngram-reads-and-fused-decode-2026-10-22.md).

## Finding

QSA selects a bounded set of rows (about 2,300 tokens) for every query, so attention itself costs the
same at any context length. A synchronized stage profile of a singleton decode step (new test-only
`flash_next_decode_context_stage_profile`, with marks inside the QSA layer) shows where the step grows
with context:

| QSA stage, ms per step (synchronized) | 2.6k tokens | 39k tokens |
| --- | ---: | ---: |
| K/V append (`qsa_c_append`) | 3.07 | 16.99 |
| indexer | 3.34 | 5.89 |
| projections, rotary prep, gather and attention, output | 9.15 | 10.59 |
| QSA total | 15.56 | 33.48 |

Every other stage (gated delta network, MoE, hyper-connection read and write, lm head) is flat within
noise. The K/V append is a full-history copy: `append_cache` ran
`contiguous(concatenate([past, new]))` for K and for V in each of the 12 QSA layers on every decode
step, 40 MB per tensor per layer at 39k tokens. A standalone MLX microbenchmark of that concatenation
(K, V and the indexer keys, one layer) grows from 0.23 ms at 2k tokens to 0.56 ms at 39k.

The usual fix, a preallocated cache updated in place (as `mlx-lm`'s `KVCache` does), relies on MLX
buffer donation, which needs a uniquely referenced array. The engine's state contract forbids that:
`forward` never modifies the cache it is given and returns a staged next state, and rejected
speculative windows, forks and prefix-cache snapshots keep older states alive, so an in-place update
would silently degrade to a copy of the whole buffer.

## What changed

| Change | Where | Kill switch |
| --- | --- | --- |
| The K/V history of each QSA layer is an immutable **sealed prefix plus a tail** of at most 256 rows. An append copies only the tail and folds it into the prefix once per 256 rows; below 4,096 tokens the history stays a single array and append is the old concatenation | `model/shared/qwen4_exp_kv_history.rs`, `qwen4_exp_attention.rs`, `qwen4_exp.rs` | `AX_FLASH_NEXT_KV_SEGMENTS=0` |
| Selected attention gathers its rows from the two segments directly: the index list is clamped per segment, each segment is gathered, and a select on `idx < sealed_rows` picks the owner. A gather is a pure copy, so the result is the gather from the concatenation, and a single-segment history emits the same ops as before. The dense kernels, batched prefill windows and the snapshot writer still flatten | same | same |
| The QSA RoPE cos/sin tables depend only on the first position, step, count, rotary width, base, dtype, rank and sequence axis, not on the layer, yet each layer rebuilt them up to four times per step (about 22 of its 86 MLX ops on the dense path). A small per-thread cache now hands the finished tables to the later layers; MRoPE (image positions) bypasses it | `qwen4_exp_qsa.rs` | `AX_FLASH_NEXT_ROPE_CACHE=0` |

## Numerics

Both changes are exact by construction (copies and reuse of the outputs of the same ops), and checked
bit for bit rather than to a tolerance:

- unit tests compare raw bits for the rotary cache (3 dtypes, 4 offsets, 2 layouts, hit and miss, key
  separation, eviction, MRoPE bypass), the segment gather against `take` on the concatenation (bf16
  and f32; ascending, shuffled, duplicated, boundary, all-sealed, all-tail and 2,100-row index lists),
  selected attention on segmented and flat histories (singleton, multi-row device window, host
  window), the sealed/tail invariants of `append` (seal exactly at the bound, multi-row appends that
  cross it, fork isolation, switching the kill switch mid-run), and that the snapshot wire bytes are
  identical whatever the segmentation;
- on the real pack, logits and serialized state are identical to the flat history for a chunked
  prefill, verifier windows of 1, 2, 3, 4 and 8 rows under both `Shared` and `RowExact`, and a 70-step
  chain, with test limits of 64/32 and 300/16 so the chain crosses several seals; counters prove every
  decode step used the segmented gather and none rebuilt the flat history;
- the rotary cache is checked the same way on a 2,600-token prompt (selected attention active).

## Results (Apple M5 Max 128 GB)

One request at a time, MTP off, page cache purged before each arm, first-seen prompts, greedy 128
tokens, same build with the switch on (default) and off, two rounds each. Generated text SHA-256 is
identical in every arm and round.

| Prompt tokens | K/V history, decode tok/s off -> on | TTFT off / on |
| ---: | --- | --- |
| 2,207 | 42.3 -> 42.3 (-0.1%, segmentation not active below 4,096) | 2.38 s / 2.38 s |
| 8,976 | 40.0 -> 40.6 (+1.4%) | 13.7 s / 13.6 s |
| 39,225 | 32.7 -> 37.5 (+14.5%) | 75.5 s / 74.9 s |

Under canonical MTP (`--mlx-mtp-policy required`) the 39k decode goes 29.4 -> 34.3 tok/s (+17%) and
under the opt-in batched verifier 32.3 -> 36.9 (+14%); 9k and 2k change by at most 1.3%, and text is
identical to the unsegmented run in every case. A request cancelled mid-prefill leaves the server
producing the same text for the next request (same SHA with and without segments), and 16 back-to-back
9k requests give one text, steady decode (40.3-40.8 tok/s) and flat resident memory (72.4 -> 72.1 GB).

After the change the synchronized profile shows the K/V append at 1.79 ms for 39k tokens (it was 16.99)
and the QSA total at 14.9 / 15.3 / 17.6 ms for 2.6k / 9k / 39k tokens (it was 15.6 and 33.5 at 2.6k and
39k). The rotary table cache removes 22 of the 86 MLX ops per QSA layer on the dense path and gives
+0.5% to +1.5% decode (42.0 -> 42.2 tok/s at 2k, 39.3 -> 39.9 at 9k), text identical.

## Limits

- One host per number; no Ultra-class run. The 39k row is a single prompt shape.
- Segmentation starts at 4,096 tokens; below that the code path is the old one. The seal copy
  (one full-history concatenation per 256 rows and layer) is a latency spike of a few milliseconds on
  that step; prefill chunks seal at once and cost what they cost before.
- What is left of the growth at 39k is the indexer: the pooled block keys of every complete block are
  recomputed each step (about 1.1 ms of the synchronized profile), plus the block score matmul and the
  gather itself. Caching the pooled keys per sealed segment is possible but worth about 3% at 39k.
- The dense kernels (context up to the QSA budget), the batched prefill windows and the snapshot
  writer still build a flat history, which is the cost they had before.
