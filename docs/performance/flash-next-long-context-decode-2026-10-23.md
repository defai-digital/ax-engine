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
default-off), and 290 `transpose` ops per step rebuilt the same weight views. After all the changes below the step is
3,040 ops (3,712 before the native hyper-connection read), of which about 800 are fused or native elementwise dispatches and about 900 are GEMMs.

## What changed

All changes are bit-identical to the previous behaviour by construction or by gate, and each has a kill switch. The default build runs 22 compiled or native stretches.

| Change | Where | Kill switch |
| --- | --- | --- |
| The K/V history of each QSA layer is an immutable **sealed prefix plus a tail** of at most 256 rows; an append copies only the tail and folds it into the prefix once per 256 rows. Selected attention gathers its rows from the two segments directly (clamp per segment, take twice, select on `idx < sealed_rows`: pure copies). Below 4,096 tokens the history stays one array. Dense kernels, batched prefill windows and the snapshot writer still flatten | `model/shared/qwen4_exp_kv_history.rs`, `qwen4_exp_attention.rs`, `qwen4_exp.rs` | `AX_FLASH_NEXT_KV_SEGMENTS=0` |
| The QSA **pooled block keys** (mean of each block, RMS norm, rotation) were recomputed for every complete block each step. They are now derived data in the index key cache and extended only by newly completed blocks (three steps in four pool nothing); never serialized, media steps bypass | `qwen4_exp_qsa.rs` | `AX_FLASH_NEXT_POOLED_CACHE=0` |
| The QSA **RoPE cos/sin tables** were rebuilt up to four times per layer per step; a per-thread cache hands the finished tables to later layers. The rotation itself is two compiled stretches (`qsa_rope_lo`, `qsa_rope_hi`) over the half tables, without the rotate-half concatenation | `qwen4_exp_qsa.rs` | `AX_FLASH_NEXT_ROPE_CACHE=0`, `AX_FLASH_NEXT_FUSE` |
| Two more compiled stretches, `qsa_gate` (attention times the f32 sigmoid of the gate) and `moe_route_tail` (top-k renormalisation and cast) | `qwen4_exp_attention.rs`, `qwen4_exp_moe.rs` | `AX_FLASH_NEXT_FUSE` |
| The **native singleton GDN kernel** (two pointwise Metal dispatches that keep both MLX reductions, validated bit-exact against the portable recurrence in September) is default-on, and reads the key heads unexpanded so the two `expand_heads` gathers per layer disappear | `qwen4_exp_gdn_metal.rs`, `qwen4_exp_gdn.rs` | `AX_MLX_FLASH_NEXT_GDN_METAL=0` |
| `QuantizedWeight` caches the transposed **view** of an unquantized weight (metadata only, so the same strided operand reaches the same GEMM); a clone starts empty and replacing the weight invalidates the view | `weights.rs`, `model/shared/utils.rs` | none (view only) |
| RMS norm gains of the QSA and indexer layers are cast to f32 once; full-range slices are skipped when batch or sequence is one | `qwen4_exp_qsa.rs`, `qwen4_exp_attention.rs` | none |
| Two **native pointwise Metal kernels** for the GDN gates: `ax_gdn_decay_beta_v1` (beta, decay pre/post and the eager `log1p`: four dispatches become one) and `ax_gdn_qkv_scale_v1` (inverse-norm scaling of q and k through `precise::pow` and the f32 cast of v: two eager `power` ops, two scale stretches and a cast become one). The sums of squares stay MLX reductions. Each kernel copies MLX's own Sigmoid, Minimum/Maximum, log1p and Power definitions, spells the math functions `precise::`, disables contraction, and runs a raw-bit self-check per dtype on first use that disables it for the process on any mismatch or rejected launch | `qwen4_exp_native_gdn.rs`, `qwen4_exp_gdn.rs` | `AX_FLASH_NEXT_FUSE` (`gdn_native_decay`, `gdn_native_qk`) |
| A **native hyper-connection read**: two Metal kernels that reproduce MLX's own `gemv` arithmetic and carry the elementwise work in the projection epilogue. `ax_hc_down_gate_v1` computes the `read_down` and `write_inject` rows straight from the two stored weights (no merged copy) and ends with `silu(down / C)` or `sigmoid(logits / C) * scale`; `ax_hc_up_mix_v1` computes `read_up` with one simdgroup per hidden column (its four accumulators are the column's four streams), ends with `sigmoid(up) * normalized` and the stream mean in MLX's reduction order. Six dispatches (two projections, the low and gate stretches, the up projection, the mix) become two. The accumulation order (per-lane sequential f32 over the K stripes, `simd_shuffle_down` 16..1, sequential threadgroup sum, leftover zero padding) is MLX's, each kernel is only used for the shapes where MLX is known to pick that `gemv` variant, and each layer proves itself on its real weights at first use | `qwen4_exp_native_hc_gemv.rs`, `qwen4_exp_residual.rs` | `AX_FLASH_NEXT_FUSE` (`hc_native_low_gate`) |

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
- the native kernels are proven rather than sampled: on both an Apple M2 Ultra and an M5 Max, every positive finite f32 bit pattern goes
  through `pow(s, -0.5)` (kernel B against MLX's `power`), every f32 bit pattern through the whole decay chain (three
  `exp_a_log` values) and the f32 sigmoid, and every bf16 and f16 pattern through the beta path; the hyper-connection
  kernel takes all 65,536 bf16 and f16 patterns in both the low-rank and the gate role. All compare raw bits, NaN
  payloads included;
- on the real pack, a 2,600-token prompt (selected attention active) is run chunked, then as verifier windows of
  1, 2, 3, 4 and 8 rows under both `Shared` and `RowExact`, then as a 16- or 70-step chain, under: K/V segments
  with test limits 64/32 and 300/16; the pooled cache on and off; the rotary cache on and off; and the
  arms {compiled stretches only, all stretches including the native kernels, MLX's native GDN recurrence kernel and
  their combinations} against an all-off baseline. Logits and the serialized final
  state are identical in every case, counters prove the segmented gather, the stretches and the kernel ran, and
  the 96-id fused-versus-eager gate checks each of the 22 stretches over the chain;
- the same gates, the exhaustive sweeps and the 1,876-test library suite pass on an Apple M2 Ultra 192 GB.

## Results (Apple M5 Max 128 GB)

One request at a time, MTP off, page cache purged before each arm, first-seen prompts, greedy 128 tokens. Generated
text SHA-256 is identical in every arm and round of every comparison below.

| Comparison (decode tok/s, after the first token) | 2,207 tokens | 8,976 tokens | 39,225 tokens |
| --- | ---: | ---: | ---: |
| main at the start of this campaign | 31.2 | 30.4 | 26.9 |
| after the n-gram reads and 7 stretches (earlier note) | 40.1 | 37.9 | 31.6 |
| the build before the native hyper-connection read | 45.4 | 44.0 | 42.6 |
| final build, with the native hyper-connection read | 47.3 | 45.7 | 44.5 |
| final build with today's kill switches set (flat K/V, portable GDN, no pooled or rotary cache, 15 stretches) | 41.7 | 39.5 | 32.4 |

Individual steps, each against the same build with only that switch changed (two rounds): K/V segments 42.3 -> 42.3
(-0.1%, inactive below 4,096 tokens), 40.0 -> 40.6 (+1.4%), 32.7 -> 37.5 (+14.5%); native GDN kernel 42.1 -> 43.6
(+3.5%), 44.3 -> 45.9 (+3.7%, 1,905 tokens), 40.4 -> 41.7 (+3.4%); the glue commit of this note's second half
(grouped heads, gains, gate, route tail, slices) +1.2% / +0.9% / +1.2%; the native GDN gate kernels +0.8% / +0.9% / +0.8% at 2.2k / 1.9k / 9k tokens (three rounds, same binary, mask on versus off). TTFT is unchanged by all of them.

Under canonical MTP (`--mlx-mtp-policy required`) the same switches give 40.8 -> 43.3, 36.1 -> 40.1 and 28.6 -> 39.0
tok/s (+6.2%, +10.9%, +36%) at 1.9k, 9k and 39k tokens, and under the opt-in batched verifier 35.7 -> 36.4, 31.4 ->
32.9 and 31.9 -> 39.0, text identical to the unchanged build in every case. With the native hyper-connection read the same
switches give 40.6 -> 45.5, 36.1 -> 41.7 and 28.4 -> 40.7 tok/s under canonical MTP and 35.7 -> 36.2, 31.5 -> 32.7 and
32.3 -> 39.0 under the batched verifier, text identical to the build with the switches set and to the build before
the read in every case. A request cancelled mid-prefill leaves
the server producing the same text for the next request, 16 back-to-back 9k requests give one text with steady
decode (43.8-44.2 tok/s) and flat resident memory (72.33 -> 72.08 GB), 30 short requests stay at 48.9-49.2 tok/s with
flat memory, and the streamed-experts and image-prefill smoke requests produce the same text as the previous builds.
The same lanes on the final build: one text for all 16 back-to-back 9k requests (45.7-46.2 tok/s, resident memory
72.43 -> 72.10 GB) and for all 30 short requests (51.2-51.7 tok/s, 71.42 -> 71.40 GB), the same text after a cancelled
prefill, and the same streamed-experts and image-prefill hashes.

After the changes the synchronized step (sum of all stages, which is larger than the real step because every stage
is evaluated on its own) grows by 5.2 ms from 2.6k to 39k tokens of context, where it grew by 20.9 ms before: the
K/V append is 1.88 ms at 39k (it was 16.99 ms), the pooled block keys 0.65 ms, and the rest of the QSA layer is
flat. The raw profiles are in the benchmark record.

## The native kernels: what the measurements said

- **GDN gates (kernels A and B): shipped default-on.** Server A/B, same binary with the two stretches in the mask or
  not, three rounds, text identical: +0.79% (2,207 tokens), +0.87% (1,905), +0.83% (8,976). The 16-step real-pack
  chain is 351.6 -> 342.2 ms on the M5 and 503 -> 497 ms on the M2.
- **Hyper-connection read: first version slower, second version shipped default-on.** The first version fused only
  the elementwise tail (`ax_hc_low_gate_v1` behind one GEMV over the concatenated `read_down` and `write_inject`). It
  was exact and cheaper in isolation (24.6 -> 23.0 us per read when 384 reads are submitted back to back, M5), but
  end to end it was slower: -1.0% and -0.8% (same binary, mask on versus off, three rounds, p1 and p3). An
  allocation-only control (the merged weights built and kept, never used) was -0.4% and -0.2%, so about a third of
  the loss was the 637 MB of duplicated weights and the rest the native path itself; the serialised merged GEMV also
  removed the overlap the write-gate GEMV had with the low-rank chain. The conclusion was that a fusion which keeps
  MLX's GEMVs and only shortens the tail cannot win, so the second version moves the elementwise work **into** the
  GEMV epilogues and reads the stored weights directly (no merged copy, no extra dispatch between the projection
  and its tail). It is exact against MLX's own `gemv`: 64 random seeds at production geometry with overflowing,
  cancelling, NaN, infinity, signed-zero and subnormal inputs; every one of the 65,536 bf16 patterns through each
  epilogue in each role, one stream at a time so a NaN in one stream cannot hide an error in another; a
  tie-engineered stream-mean case in which the four candidate associations round differently; and mutation checks
  (reversed shuffle or threadgroup order, a different mean association, a dropped `+ 0.0f`, an unrounded sigmoid
  each make a test fail). The real-pack gates (prefill, verifier windows of 1/2/3/4/8 rows under Shared and
  RowExact, a 16-step chain, final state) are raw-bit identical to the eager baseline on the M2 Ultra and the M5 Max,
  with the stretch's counter proving it ran on every one of the 1,536 reads of the chain. Server A/B, same binary,
  the stretch in the mask or not, three rounds, text identical: 45.05 -> 47.01 tok/s (+4.3%, 2,207 tokens), 47.40 ->
  49.76 (+5.0%, 1,905), 43.84 -> 45.77 (+4.4%, 8,976); TTFT unchanged. On the M2 Ultra the same A/B is +0.5%, +0.9%
  and +0.3% (two rounds, text identical); the gain there is small and the cause was not investigated.
- The router's `softmax`, `argpartition` and `take_along_axis` are primitives, not fusable glue.

## Limits

- One host per number; no Ultra-class run. The 39k row is a single prompt shape.
- Segmentation starts at 4,096 tokens; below that the code path is the old one. The seal copy (one full-history
  concatenation per 256 rows and layer) is a latency spike of several milliseconds on that step; prefill chunks
  seal at once and cost what they cost before.
- The dense kernels (context up to the QSA budget), the batched prefill windows and the snapshot writer still
  build a flat history, which is the cost they had before. The opt-in batched MTP verifier is unchanged at short
  context.
- MLX's native GDN recurrence kernel accepts f32 state with key widths 32, 64, 128 and 256; anything else takes the
  portable recurrence (exactly as before).
- The new GDN gate kernels need `dk == dv` for the q/k/v kernel, rows 1-8, and a dtype mix of bf16/f16/f32; a
  non-contiguous input costs one extra copy dispatch (`ensure_row_contiguous`). Any self-check mismatch or rejected
  launch disables a kernel for the process and the compiled stretches run instead.
