# Batched decode ceiling — the lever is real but the forward captures ~none of it

**Date:** 2026-07-17 · **MLX 0.32.0 (minos 26.2)** · **M3 Max** · Phase 3.0/3.1
audit for the decode-dispatch-efficiency plan.

## Why batched decode

Decode is bandwidth- and host-graph-encoding-bound
([`ax-decode-overlap-residual.md`](ax-decode-overlap-residual.md)): batch=1
reads all active weights and encodes the whole graph per token. A batched
forward reads the weights and encodes the graph once for B tokens, so aggregate
throughput should grow toward B×. This is the highest-ceiling remaining lever
(vLLM/llama.cpp show near-linear aggregate on bandwidth-bound decode).

## What the current code supports

`decode_batched_forward` / `BatchedDecodeSession` (default OFF,
`AX_MLX_BATCHED_DECODE`) is **dense full-attention only**. Structural
rejections (`architecture.rs::dense_batched_decode_structural_rejections`):
`moe`, `linear_attention`, `mla`, `sliding_window`, `layer_gating`, `diffusion`,
plus `mtp`. **Qwen3-Coder-Next (qwen3_next) is MoE + linear-attention → rejected
on both counts.** Extending to it means batched MoE and batched linear-attention
forwards — a large effort, which is why the ceiling had to be validated first.

## Ceiling measurement (dense, the supported path)

`batched-decode-ceiling-probe` (new microbench): prefill one 32-token prompt,
seed N clones into a `BatchedDecodeSession`, step the cohort, report aggregate
tok/s. Llama-3.1-8B-4bit, 0.32.0:

| batch | agg tok/s | per-req tok/s | step µs | scaling |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 53.3 | 53.3 | 18,766 | 1.00× |
| 2 | 58.7 | 29.4 | 34,065 | 1.10× |
| 4 | 64.0 | 16.0 | 62,529 | 1.20× |
| 8 | 66.3 | 8.3 | 120,698 | **1.24×** |

**Only 1.24× at batch=8** — step time grows ~6.5× for 8× batch, so the forward
barely amortizes. Far from the ×2–4 premise.

## Isolation — it is not what you'd guess

- **Not the compiled-path custom-kernel fallback.** The run spams
  `CustomKernel cannot infer output shapes` (the compiled dense-FFN path fails
  to shapeless-compile the packed-SwiGLU Metal kernel every layer and falls
  back — the same wall as MoE per-layer compile). But disabling it
  (`AX_MLX_DENSE_FFN_COMPILE=0 AX_MLX_DENSE_SWIGLU_PACKED_METAL=0`) gives
  **identical 1.24×** — the fallback is clean, not the cap.
- **The matmul ceiling is ~2.7×, not ~5× (earlier number was an artifact).**
  A per-call `mx.eval` around each matmul gives 5×, but that is amortizing the
  fixed submit overhead, not the weight read. Chaining 100 matmuls into one
  graph and evaluating once (the real forward's structure — one eval per step)
  gives the true weight-read amortization: **2.69× at batch=8** (1.62× @2,
  2.40× @4, 3.59× @16). It is not a shape issue — decode-shaped `[B,1,H]`
  matches 2D. So Apple Silicon 4-bit quantized matmul amortizes ~2.7× at
  batch=8 (compute-limited past that), and the forward nets 1.24× — still
  ~2× below the real matmul ceiling.
- **Not the attention/KV path (leading suspect REFUTED).** Sweeping the prefill
  length changes the per-request KV size 32× but barely moves scaling: seq=8 →
  1.24×, seq=256 → 1.20× at batch=8. If attention/KV were the non-amortizing
  cost it would degrade sharply with seq; it does not. The non-amortizing
  component is **seq-independent**.
- **What it is: a large seq-independent structural cost in the batched forward.**
  Solving the two-regime model (matmuls amortize 2.69×, the rest scales ~8×)
  against the measured b1/b8 step times puts ~69% of the batch=1 step in a
  non-matmul, seq-independent, batch-linear cost — far too large to be
  elementwise norms/RoPE. Pinpointing it requires stage-instrumenting the
  batched forward (`decode_batched_forward` / `layer_forward_batched`), the
  Phase 3.3 first move.

## Conclusion: viable, but the first work is fixing DENSE amortization

The lever is real — the weight-bound matmuls amortize ~5× at batch=8 in
isolation — but the AX batched forward captures almost none of it (1.24×). The
gap is **implementation**, not a hardware/MLX limit, and it is on the **dense**
path, before any MoE/linear-attention extension. The leading suspect for the
non-amortizing component is the per-request attention/KV path (each row has its
own KV cache; batched SDPA reads B caches and runs B attentions with no
cross-batch weight to amortize), plus possible structural overhead in
`layer_forward_batched`; confirming it needs stage-timing the batched step.

## Phase 3.3 update (2026-07-17): ceiling is ~3.3×, and AX's 1.24× is a fixable AX bug

A pure-MLX replica of the FULL batched decode step — per-layer QKV/O/FFN
quantized matmuls + SDPA (with a ragged validity mask) + lm_head + argmax,
×32 layers, single eval — amortizes **3.3× at batch=8** (1.92× @2, 3.0× @4).
So the realistic ceiling is ~3.3× (higher than the 2.69× square-matmul
estimate; the big FFN/lm_head matmuls are more bandwidth-bound), and AX's 1.24×
is **~2.7× below a ceiling that pure MLX reaches with the same structure** —
i.e. a recoverable AX-side implementation issue, not a hardware/MLX limit.

Ruled out as the culprit (each tested):
- **RoPE per-row loop** — real inefficiency (O(batch) slice+rope+concat, now
  fixed via `rope_dynamic` with a `[batch]` offset array, parity-clean, 25
  batched tests pass), but removing it did **not** move the ceiling → the step
  is not dispatch-bound.
- **SDPA validity mask** — replica with vs without: 3.31× vs 3.30×.
- **lm_head + argmax** — replica with vs without: 3.28× vs 3.30×.
- **Attention key length** — seq 8 vs 256: 1.24× vs 1.20× (seq-independent).

## Phase 3.4 update (2026-07-17): ROOT-CAUSED — the FFN's per-row `RowExact` policy

Stage-instrumented the real `layer_forward_batched` (env-gated
`AX_MLX_BATCHED_PROFILE`, barriers per stage). Per-stage amortization
batch 1→8 (Llama-8B):

| stage | b1 µs | b8 µs | amortization |
| --- | ---: | ---: | ---: |
| pre_attn (norm+qkv+qk_norm+rope) | 11,330 | 21,392 | 4.2× |
| attention (append+SDPA) | 8,428 | 11,855 | 5.7× |
| o_proj (+post_norm) | 9,530 | 16,196 | 4.7× |
| **ffn** | **24,004** | **116,464** | **1.65×** |

The FFN is the culprit — it grows 4.85× for 8× batch (49%→73% of the step)
while every other stage amortizes 4–5.7×. Root cause: **`ffn_swiglu_batched`
passes `ProjectionBatchPolicy::RowExact`**, and `qw_with_policy` with `RowExact`
(`utils.rs:54`) **loops over batch rows and does a separate `quantized_matmul`
per row, then concatenates** — re-reading the gate_up and down weights B times
instead of once. Zero weight-read amortization, by construction.

Why RowExact exists: batched `quantized_matmul` is **not** bit-identical to
per-row (measured max_abs_diff **2.3e-2** on `[8,1,4096]×[4096,14336]` bf16 —
the batched kernel accumulates in a different order), and batched-decode
certification requires per-row parity with single-request decode. RowExact
buys that exactness at the cost of all amortization — the same bf16
reduction-order-drift tension as the fused-downproj report.

## Phase 3.5 result (2026-07-17): +46% aggregate, and greedy tokens DON'T diverge

Shipped `AX_MLX_BATCHED_SHARED_PROJ` (default OFF): routes the batched QKV,
attention-output, and FFN projections through the `Shared` policy (one batched
`quantized_matmul`) instead of `RowExact`. A/B on Llama-8B (0.32.0):

| batch | RowExact agg tok/s | Shared agg tok/s | Shared scaling | slot-0 token stream |
| ---: | ---: | ---: | ---: | --- |
| 1 | 53.0 | 52.9 | 1.00× | identical |
| 2 | 57.3 | 81.0 | 1.53× | identical |
| 4 | 62.6 | 93.0 | 1.76× | identical |
| 8 | 65.2 | **95.0** | **1.80×** | identical (`bd951636…`) |

**+46% aggregate throughput at batch=8** (65.2 → 95.0 tok/s), and — despite the
2.3e-2 raw-matmul drift — the **greedy token stream is byte-identical** to
RowExact, and the **25 batched correctness tests pass with the flag on**. So on
these workloads the bf16 accumulation drift does not flip argmax; Shared is
greedy-equivalent in practice, not merely close.

## Phase 3.6 result (2026-07-17): lm_head fix (+56% total), and DEFAULT-FLIPPED to ON

Profiling the Shared residual found the last big per-row site: the **lm_head**
(`[B, hidden] × [hidden, 128256]`, ~262 MB at 4-bit) was hardcoded `RowExact`
in `decode_batched_forward` — outside the per-layer profiler, re-read B times
per step. Routing it through the same flag: batch=8 on Llama-8B **95.0 → 97.2
tok/s (1.80× → 1.92×)**. Combined with 3.5, Shared is now **+56% over RowExact**
(65 → 97 tok/s), token stream still identical.

**Default-flip certification (met):** the batched-vs-per-row bf16 drift does not
flip greedy argmax — the decoded token stream is **byte-identical** to RowExact
on **three dense checkpoints** (Llama-3.1-8B, Qwen3-4B, Ministral-8B), the 25
batched correctness tests pass with the policy on, and the full workspace suite
is green with it as the default. `AX_MLX_BATCHED_SHARED_PROJ` is now **default
ON** with a kill-switch that restores the bit-exact per-row path.

Remaining gap to the ~3.3× replica ceiling is the compute-bound FFN matmuls
(2.8× true amortization for that size on Apple Silicon 4-bit) plus fixed
per-step overhead — diminishing returns on dense.

## Phase 3.7 handoff (2026-07-17): batched MoE + linear-attention — scoped and de-risked

The original target (Qwen3-Coder-Next) is MoE + linear-attention, both rejected
by `dense_batched_decode_structural_rejections`. Scoping the extension found it
is an **integration** effort, not a kernel rewrite — the compute kernels are
already batch-capable:

- **MoE experts** — `moe_experts_forward` is `leading_elements`-general (it
  already serves multi-token prefill `[1, seq, H]`), so a batched decode
  `[B, 1, H]` input flows through the same `gather_qmm` with `[B, top_k]`
  indices. The MoE half is a clean wiring change.
- **Linear-attention** — `gated_delta_decode_kernel` already takes a `batch: i32`
  parameter and per-row SSM state, so it decodes B rows in one dispatch.

**The real work (the integration gaps):**

1. **Batched recurrent-state store.** `BatchedKvCache` holds only full-attention
   `[B, kv_heads, cap, head_dim]` K/V — it has **no** linear-attention SSM state.
   Need a sibling batched store for the per-row gated-delta recurrent state
   (`[B, ...state_shape]`), with the same oracle contract BatchedKvCache already
   meets: **row r byte-identical to a single-sequence run**. This is where
   silent numerical bugs live; it must be oracle-tested first, in isolation.
2. **Route by layer kind in `layer_forward_batched`.** Replace the
   `router_proj.is_none()` / `linear_attn.is_none()` asserts with: MoE layers →
   the leading-general MoE block; linear-attention layers → the batch-param
   gated-delta with the batched state store. Lift the corresponding entries in
   `dense_batched_decode_structural_rejections`.
3. **Certify per-model on greedy-token divergence** (the 3.6 pattern), not
   bit-identity — batched `gather_qmm`/gated-delta will bf16-drift vs per-row.

**Why it wasn't completed here:** every local MoE checkpoint is hybrid (MoE +
linear-attention / sliding / MLA — verified: Qwen3.6-35B-A3B is also MoE +
linear-attn), so there is **no MoE-only or linear-only test target** to land an
end-to-end increment against — both halves must ship together for any real
model. Each half CAN be unit-tested in isolation first (batched MoE block vs
per-row; batched recurrent-state store vs single-sequence oracle), which is the
right first increment. The batched recurrent-state integration is a focused
multi-day effort best done fresh, not rushed at a marathon's tail — the
fused-downproj report's lesson (rushing weight/reduction code is how silent
numerical bugs enter a forward pass) applies directly.

**Revised Phase 3 plan (updated after 3.2/3.3):**
- **Realistic ceiling: ~2.7× aggregate at batch=8** (the true matmul
  amortization on this hardware), i.e. ~53→~143 agg tok/s on Llama-8B, not the
  naive ×8. Meaningful for concurrent serving; zero for single-request.
- **3.2 DONE:** attention/KV ruled out (seq-independent); the gap is a large
  seq-independent structural cost in the batched forward, ~2× recoverable
  (1.24×→2.69×).
- **3.3 — stage-instrument `decode_batched_forward` / `layer_forward_batched`**
  (matmuls vs attention vs elementwise vs KV/mask/lm_head, batch 1 vs 8) to
  name the seq-independent non-amortizing stage, then fix it. Parity-gated.
- **3.4+ — extend to MoE / linear-attention** only after dense delivers toward
  the 2.7× ceiling.

This inverts the naive plan (extend to Coder-Next first): the measurement shows
the supported dense path itself only returns 1.24×, so that is where Phase 3
starts. The +28% auto-buffer-caps win remains the shipped single-request result;
batched decode is a real but currently-unrealized aggregate lever.

## Reproduction

```
cargo run -p ax-engine-microbench --release --bin batched-decode-ceiling-probe -- <dense_model_dir>
# vs the compile fallback:
AX_MLX_DENSE_FFN_COMPILE=0 AX_MLX_DENSE_SWIGLU_PACKED_METAL=0 cargo run ... batched-decode-ceiling-probe -- <dir>
```
