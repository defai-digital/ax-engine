# Flash Next long-context decode and decode glue — 2026-10-23

Qwen 3.8 Flash Next (`qwen4_exp`, MXFP4 MTP pack `AutomatosX/AX-Qwen3.8-Flash-Next-MLX-AXQ-MXFP4-MTP`
at `ff2a28485eb89bb60e8fe35dd6c65c51e63ee7b3`) decoded at about 42 tok/s with 2k tokens of context and
32.7 tok/s with 39k on an Apple M5 Max 128 GB at the start of this work. This note records where the
difference and the remaining host overhead came from, what changed, and what was measured. It is a
single-host engineering record, **not a speedup claim and not a certification run**: no comparison runtime is
involved and the host is a campaign host. It builds on
[the n-gram read and fused-stretch note](flash-next-ngram-reads-and-fused-decode-2026-10-22.md).

## Finding 1: the K/V history was copied on every step

QSA selects a bounded set of rows (about 2,300 tokens) for every query, so attention itself costs the same at
any context length. A synchronized stage profile of a singleton decode step (new test-only
`flash_next_decode_context_stage_profile`, with marks inside the QSA layer) shows where the step grows with
context:

| QSA stage, ms per step (synchronized) | 2.6k tokens | 39k tokens |
| --- | ---: | ---: |
| K/V append (`qsa_c_append`) | 3.07 | 16.99 |
| indexer | 3.34 | 5.89 |
| projections, rotary prep, gather and attention, output | 9.15 | 10.59 |
| QSA total | 15.56 | 33.48 |

Every other stage is flat within noise. The K/V append is a full-history copy: `append_cache` ran
`contiguous(concatenate([past, new]))` for K and V in each of the 12 QSA layers on every decode step, 40 MB per
tensor per layer at 39k tokens. The usual fix, a preallocated cache updated in place (as `mlx-lm`'s `KVCache`
does), relies on MLX buffer donation, which needs a uniquely referenced array. The engine's state contract
forbids that: `forward` never modifies the cache it is given and returns a staged next state, and rejected
speculative windows, forks and prefix-cache snapshots keep older states alive, so an in-place update would
silently degrade to a copy of the whole buffer.

## Finding 2: an op census of the decode step

`mlx_sys::op_trace_begin/op_trace_end` (new, opt-in) records every MLX op wrapper by name and calling Rust
function, with source lines in a dev build. The first census of one decode step showed 9,851 ops with no compiled
stretches and 5,331 with the 15 stretches of the earlier note, and two things nobody had written down: the
production decode ran the 26-op portable GDN recurrence (the native kernel validated in September was still
default-off), and 290 `transpose` ops per step rebuilt the same weight views. Per change below, after all of them
the step is 4,036 ops, of which about 1,080 are compiled-stretch dispatches and 900 are GEMMs.

## What changed

All changes are bit-identical to the previous behaviour by construction or by gate, and each has a kill switch.

| Change | Where | Kill switch |
| --- | --- | --- |
| The K/V history of each QSA layer is an immutable **sealed prefix plus a tail** of at most 256 rows; an append copies only the tail and folds it into the prefix once per 256 rows. Selected attention gathers its rows from the two segments directly (clamp per segment, take twice, select on `idx < sealed_rows`: pure copies). Below 4,096 tokens the history stays one array. Dense kernels, batched prefill windows and the snapshot writer still flatten | `model/shared/qwen4_exp_kv_history.rs`, `qwen4_exp_attention.rs`, `qwen4_exp.rs` | `AX_FLASH_NEXT_KV_SEGMENTS=0` |
| The QSA **pooled block keys** (mean of each block, RMS norm, rotation) were recomputed for every complete block each step. They are now derived data in the index key cache and extended only by newly completed blocks (three steps in four pool nothing); never serialized, media steps bypass | `qwen4_exp_qsa.rs` | `AX_FLASH_NEXT_POOLED_CACHE=0` |
| The QSA **RoPE cos/sin tables** were rebuilt up to four times per layer per step; a per-thread cache hands the finished tables to later layers. The rotation itself is two compiled stretches (`qsa_rope_lo`, `qsa_rope_hi`) over the half tables, without the rotate-half concatenation | `qwen4_exp_qsa.rs` | `AX_FLASH_NEXT_ROPE_CACHE=0`, `AX_FLASH_NEXT_FUSE` |
| Two more compiled stretches, `qsa_gate` (attention times the f32 sigmoid of the gate) and `moe_route_tail` (top-k renormalisation and cast): 19 stretches in all | `qwen4_exp_attention.rs`, `qwen4_exp_moe.rs` | `AX_FLASH_NEXT_FUSE` |
| The **native singleton GDN kernel** (two pointwise Metal dispatches that keep both MLX reductions, validated bit-exact against the portable recurrence in September) is default-on, and reads the key heads unexpanded so the two `expand_heads` gathers per layer disappear | `qwen4_exp_gdn_metal.rs`, `qwen4_exp_gdn.rs` | `AX_MLX_FLASH_NEXT_GDN_METAL=0` |
| `QuantizedWeight` caches the transposed **view** of an unquantized weight (metadata only, so the same strided operand reaches the same GEMM); a clone starts empty and replacing the weight invalidates the view | `weights.rs`, `model/shared/utils.rs` | none (view only) |
| RMS norm gains of the QSA and indexer layers are cast to f32 once; full-range slices are skipped when batch or sequence is one | `qwen4_exp_qsa.rs`, `qwen4_exp_attention.rs` | none |

## Numerics

Every change is checked bit for bit rather than to a tolerance:

- unit tests compare raw bits for the K/V segments and the segment gather against `take` on the concatenation
  (bf16 and f32; ascending, shuffled, duplicated, boundary, all-sealed, all-tail and 2,100-row index lists),
  selected attention on segmented and flat histories (singleton, multi-row device window, host window), the
  `append` invariants (seal exactly at the bound, multi-row appends that cross it, fork isolation, switching the
  kill switch mid-run), snapshot wire bytes, the pooled range against the full pooling in three dtypes and the
  selection step by step with and without the cache (device and host selection, single-query and window, restore,
  fork, media bypass), the rotary stretches and the gate and route-tail stretches on adversarial inputs (NaN
  payloads, +-inf, +-0, subnormals, overflow) against both forced-eager and the inline original ops, dtype reuse
  without a registry reset, failure injection per stretch, the row cap, the grouped GDN kernel against the
  expanded kernel and the portable recurrence (groups 1-3, batch 1-2, subnormal inputs, eight state-feedback
  steps), and a stale-weight-view regression;
- on the real pack, a 2,600-token prompt (selected attention active) is run chunked, then as verifier windows of
  1, 2, 3, 4 and 8 rows under both `Shared` and `RowExact`, then as a 16- or 70-step chain, under: K/V segments
  with test limits 64/32 and 300/16; the pooled cache on and off; the rotary cache on and off; and the
  arms {compiled stretches, native GDN kernel, both} against an all-off baseline. Logits and the serialized final
  state are identical in every case, counters prove the segmented gather, the stretches and the kernel ran, and
  the 96-id fused-versus-eager gate checks each of the 19 stretches over the chain;
- the same gates and the 1,858-test library suite pass on an Apple M2 Ultra 192 GB.

## Results (Apple M5 Max 128 GB)

One request at a time, MTP off, page cache purged before each arm, first-seen prompts, greedy 128 tokens. Generated
text SHA-256 is identical in every arm and round of every comparison below.

| Comparison (decode tok/s, after the first token) | 2,207 tokens | 8,976 tokens | 39,225 tokens |
| --- | ---: | ---: | ---: |
| main at the start of this campaign | 31.2 | 30.4 | 26.9 |
| after the n-gram reads and 7 stretches (earlier note) | 40.1 | 37.9 | 31.6 |
| final build, all of the above | 44.8 | 43.5 | 42.3 |
| final build with today's kill switches set (flat K/V, portable GDN, no pooled or rotary cache, 15 stretches) | 41.9 | 39.6 | 32.4 |

Individual steps, each against the same build with only that switch changed (two rounds): K/V segments 42.3 -> 42.3
(-0.1%, inactive below 4,096 tokens), 40.0 -> 40.6 (+1.4%), 32.7 -> 37.5 (+14.5%); native GDN kernel 42.1 -> 43.6
(+3.5%), 44.3 -> 45.9 (+3.7%, 1,905 tokens), 40.4 -> 41.7 (+3.4%); the glue commit of this note's second half
(grouped heads, gains, gate, route tail, slices) +1.2% / +0.9% / +1.2%. TTFT is unchanged by all of them.

Under canonical MTP (`--mlx-mtp-policy required`) the same switches give 40.7 -> 43.0, 36.1 -> 39.7 and 28.4 -> 38.7
tok/s (+5.5%, +9.9%, +36%) at 1.9k, 9k and 39k tokens, and under the opt-in batched verifier 35.9 -> 36.3, 31.7 ->
32.8 and 32.1 -> 38.8, text identical to the unchanged build in every case. A request cancelled mid-prefill leaves
the server producing the same text for the next request, 16 back-to-back 9k requests give one text with steady
decode (43.6-43.9 tok/s) and flat resident memory (72.44 -> 72.09 GB), 30 short requests stay at 48.4-48.8 tok/s with
flat memory, and the streamed-experts and image-prefill smoke requests produce the same text as the previous builds.

After the changes the synchronized step (sum of all stages, which is larger than the real step because every stage
is evaluated on its own) grows by 5.2 ms from 2.6k to 39k tokens of context, where it grew by 20.9 ms before: the
K/V append is 1.88 ms at 39k (it was 16.99 ms), the pooled block keys 0.65 ms, and the rest of the QSA layer is
flat. The raw profiles are in the benchmark record.

## Evaluated and not built

- **HC read as fewer dispatches.** Per hyper-connection read the sequence is norm stretch, GEMM, low stretch, GEMM,
  mix stretch, GEMM, gate stretch: four fused elementwise kernels around three GEMMs, already one dispatch each.
  Fewer dispatches would need GEMM prologue/epilogue fusion, that is custom GEMMs, and another MLX-based engine
  measured hand-written GEMVs losing in all three attempts; MLX does not fuse unrelated elementwise graphs, so
  merging the independent gate and low stretches gains nothing.
- **Custom kernels for the GDN decay and L2 chains.** The eager remainder after the stretches is `power` (72 ops
  per step) and `log1p` (36), under 1% of the step, and a custom kernel would need explicit `precise::` math and a
  load-time bitwise self-check to match them.
- The router's `softmax`, `argpartition` and `take_along_axis` are primitives, not fusable glue.

## Limits

- One host per number; no Ultra-class run. The 39k row is a single prompt shape.
- Segmentation starts at 4,096 tokens; below that the code path is the old one. The seal copy (one full-history
  concatenation per 256 rows and layer) is a latency spike of several milliseconds on that step; prefill chunks
  seal at once and cost what they cost before.
- The dense kernels (context up to the QSA budget), the batched prefill windows and the snapshot writer still
  build a flat history, which is the cost they had before. The opt-in batched MTP verifier is unchanged at short
  context.
- The native GDN kernel accepts f32 state with key widths 32, 64, 128 and 256; anything else takes the portable
  recurrence (exactly as before).
