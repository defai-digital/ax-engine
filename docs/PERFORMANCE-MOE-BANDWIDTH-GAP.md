# MoE decode bandwidth-utilization gap (Qwen3-Coder-Next)

## Summary

On Qwen3-Coder-Next (MoE, 10-of-512 experts, batch=1 decode), AX Engine
sustains **39% of the M5 Max GPU peak memory bandwidth** (227 GB/s of
577 GB/s) at 115.4 tok/s, reading 1.96 GB/token. On **dense** models on
the same hardware AX reaches **78–86% of peak** — so there is a ~40-point
bandwidth-utilization gap that is specific to batch=1 MoE decode.

This is **not** a gap to llama.cpp (llama.cpp is at 42% but reads 1.44×
the bytes, so it decodes slowest). It is a gap to AX's own dense-model
ceiling, and it represents real headroom: if AX could sustain dense-level
bandwidth utilization on this MoE footprint, decode would roughly double.

This document records *why* the bus is idle, *what* the leverage points
are, and *what* must be measured first. It is a companion to
[`PERFORMANCE-DECODE-GAP.md`](PERFORMANCE-DECODE-GAP.md), which covers the
separate (and largely closed) 1–6% dense-decode gap to `mlx_lm`.

## The measured state

| Engine / quantization | Weights/token | Decode tok/s | Effective BW | % of 577 GB/s peak |
|---|---:|---:|---:|---:|
| AX — MLX 4-bit | 1.96 GB | 115.4 | 227 GB/s | 39% |
| mlx-lm — MLX 4-bit | 1.96 GB | 99.2 | 195 GB/s | 34% |
| llama.cpp — Q4_K_M | 2.83 GB | 86.2 | 244 GB/s | 42% |

Source: `benchmarks/results/mlx-inference/2026-06-13-qwen3-coder-next-prefill-probe/`
and `benchmarks/results/llama-cpp-metal/2026-06-13-qwen3-coder-next-9620-fa/`.
The 577 GB/s peak is an MLX reduction probe over a 6 GB array (same probe
used for the Gemma 4 dense-model bandwidth table).

The bandwidth numbers are **derived** (bytes/token × tok/s ÷ peak), not
directly measured with performance counters. The bench JSON shows
`peak_bandwidth_gb_s: null` for this run.

## Why the bus is idle (the core diagnosis)

Decode tok/s = effective_bandwidth ÷ bytes_per_token. AX already reads the
fewest bytes (1.96 GB), so the only way to go faster is to raise the
effective bandwidth — i.e. keep the memory bus busier during each token's
decode.

In **dense** decode, each layer's FFN is 3 large quantized matmuls (gate,
up, down) that each read a substantial slice of the model weights and run
long enough to saturate the bus for hundreds of microseconds. The bus duty
cycle is high.

In **batch=1 MoE** decode, the picture is different:

1. **Each gather_qmm is small.** Per layer, the expert gate_up gather reads
   10 experts × expert_size(512) × hidden(2048) × 0.5 bytes (4-bit) ≈ 5.2 MB,
   and the down gather reads a similar amount. These kernels finish in tens
   of microseconds — too short to fully amortize the fixed per-dispatch
   overhead (kernel launch, command-buffer submission, graph evaluation).

2. **There are many non-bandwidth dispatches between the gather_qmms.**
   Per layer, the MoE block executes roughly:
   - Router: 1× quantized_matmul (8-bit) + softmax_precise + argpartition +
     take_along_axis + sum + divide ≈ **6–7 dispatches** (none read expert
     weights; the bus is idle).
   - Expert gate_up: 1× gather_qmm (4-bit) — **bus-active**.
   - Activation: slice + slice + silu_mul/swiglu ≈ **3 dispatches** (bus idle).
   - Expert down: 1× gather_qmm (4-bit) — **bus-active**.
   - Weighted sum: 1× custom kernel `ax_qwen3_moe_weighted_sum_v1`
     (bus idle — tiny tensor).
   - Shared expert: 2–3× quantized_matmul (4-bit) + silu_mul + 1×
     quantized_matmul (8-bit gate) + sigmoid + multiply ≈ **6–7 dispatches**
     (partly bus-active, partly idle).
   - Add(moe_out, shared_out): 1 dispatch (bus idle).
   - Post FFN norm: 1× rms_norm (bus idle).

   That is **~20 dispatches per layer × 48 layers ≈ ~960 dispatches per
   token**, of which only ~4–6 per layer actually read expert weights. The
   bus duty cycle is low because most dispatches do tiny elementwise /
   reduction work that never touches the weight tensors.

3. **The bus duty cycle, not any single kernel's efficiency, is the gap.**
   No single kernel is "slow" — the gather_qmm dispatches into MLX's
   optimized C++ runtime. The problem is the *gaps* between them: launch
   overhead, host-side graph construction, and tiny inter-kernel dispatches
   that occupy the pipeline without moving bytes.

This is consistent with the README framing: "decode here is **dispatch-bound,
not bandwidth-bound**."

## What is already optimized

The codebase already has substantial MoE-specific work:

- **Packed `gate_up_exps_packed`** (`mlp.rs:1507-1518`) — collapses the
  expert gate + up projections into one gather_qmm per layer (saves 1
  dispatch + 1 weight read).
- **Custom weighted-sum kernel** `ax_qwen3_moe_weighted_sum_v1`
  (`mlp.rs:403-501`) — fuses multiply + reduce + cast into one dispatch.
- **Packed SwiGLU kernel** `ax_qwen_packed_swiglu_v1` (`mlp.rs:342-356,
  528-535`) — fuses the gate/up split + silu + multiply. **However, this is
  only wired to the DENSE FFN path** (`ffn_swiglu` → `packed_ffn_activation`
  at `mlp.rs:768,305-315`); the MoE expert path
  (`moe_experts_forward_impl` at `mlp.rs:1541`) uses the unfused
  `dense_ffn_activation` → `silu_mul`/`swiglu` instead.
- **Packed QKVZ/BA linear-attention projections** — load-time packing for
  the 36 linear-attention layers (gated by `AX_MLX_PACK_LINEAR_ATTENTION_PROJECTIONS`,
  default ON, `fastpath.rs:477-491`).
- **Fused gated-delta decode kernel** `qwen35_gated_delta_decode_v1`
  (`linear_attention_ops.rs:446-533`) — one dispatch for the entire SSM
  recurrent update (eliminates ~8 dispatches per linear-attention layer).

So AX already beats `mlx_lm` by +13.8–16.3% on this model (the custom
weighted-sum kernel and the gather-qmm routing are the runtime wins). The
question is how to close the remaining gap to AX's own dense ceiling.

## What must be measured first (Tier 0 — prerequisite)

**There is currently no instrumentation that can attribute MoE decode time
to sub-stages on Qwen3-Next.** The existing profilers are:

- `AX_MLX_GEMMA4_MOE_PROFILE=1` — has exactly the right stages (Attention,
  Dense, Router, Expert, Post) but is gated by `cfg.gemma4_moe_router`
  (`standard.rs:162`), which is false for Qwen3-Next. Does not fire.
- `AX_MLX_DECODE_PROFILE=1` — fires for all families (`seq == 1` gate), but
  the `PostAttnFfnGateUp` / `PostAttnFfnActivation` / `PostAttnFfnDown`
  sub-stages are only instrumented inside `ffn_swiglu` (the **dense** FFN
  path, `mlp.rs:768+`). The MoE path (`moe_experts_forward_impl`) records
  nothing at sub-stage granularity — the entire MoE block collapses into
  the coarse `PostAttnFfn` number.

**Action 0A — Add MoE sub-stage profiling.** Either un-gate the existing
`Gemma4MoeProfileSnapshot` (rename to a family-neutral MoE profile) or add
explicit `record_*` calls inside `moe_experts_forward_impl` and
`shared_expert_forward`. Without this, every subsequent optimization is
blind. The harness already supports `--ax-gemma4-moe-profile`; a parallel
`--ax-moe-profile` (or family-neutral rename) is a low-risk, diagnostic-only
change.

## Leverage points, ranked

### Tier 1 — Fuse within the MoE block (low–medium effort, incremental)

Each of these eliminates real dispatches. Per `PERFORMANCE-DECODE-GAP.md`,
single-op fusion on the *dense* path showed ~0.2% gains because dense ops
are already bandwidth-bound. **The MoE path is dispatch-bound, not
bandwidth-bound, so that prior ceiling does not apply here** — eliminating
a dispatch on the MoE path directly raises bus duty cycle.

**1A — Fuse shared-expert add into the weighted-sum kernel.**
The `ax_qwen3_moe_weighted_sum_v1` kernel already loops over top-k and
accumulates `down_out[k] * weight[k]`. Extend it to take a third input
(shared_expert_out) and add it in-kernel:
`out[idx] = sum_k(down_out[k] * weight[k]) + shared_out[idx]`. Eliminates
1 `add` dispatch per layer × 48 = 48 dispatches/token.
Effort: low (modify existing kernel + caller). Risk: low.

**1B — Route MoE expert SwiGLU through a packed Metal kernel.**
The gather_qmm on `gate_up_exps_packed` already produces a packed
`[1, seq, top_k, 2*expert_size]` output. Currently this is sliced into
gate/up (`mlx_slice_last_dim` × 2) then passed to `dense_ffn_activation`
→ `silu_mul`/`swiglu` (3 dispatches). The dense path's
`ax_qwen_packed_swiglu_v1` does the split + silu + mul in one kernel.
Adapt that kernel for the `[., ., top_k, 2*expert_size]` shape (or write a
thin variant) and call it directly on the packed gather_qmm output.
Eliminates 2 slices + 1 activation = ~3 dispatches × 48 = ~144/token.
Effort: medium. Risk: low–medium (new Metal kernel, needs correctness tests
against the current `silu_mul` reference).

**1C — Fuse router softmax + top-k + renorm.**
Currently 6–7 dispatches (quantized_matmul + softmax_precise + argpartition
+ take_along_axis + sum + divide). A single Metal kernel could do the full
router in one dispatch. Eliminates ~5 dispatches × 48 = ~240/token.
Effort: medium–high (GPU argpartition / top-k selection is non-trivial).
Risk: medium (ranking correctness is output-corrupting if wrong). This is
the highest-risk Tier 1 item; defer until 1A/1B are validated.

**Tier 1 aggregate estimate:** eliminating ~7 dispatches/layer (1A+1B)
raises bus duty cycle modestly. A rough ceiling: if dispatch overhead is
~1–3µs each and there are ~960 dispatches/token, removing ~336 (7×48)
saves ~0.3–1.0ms/token out of an 8.66ms step (115.4 tok/s) — roughly +4–12%.
This is consistent with the README's "+8% if AX matched llama.cpp's 42%."
The Tier 1 ceiling is probably in the +5–10% range, not the +100% needed
to reach dense-level utilization.

### Tier 2 — Deep expert-block fusion (high effort, high impact)

**2A — Fuse gate_up gather_qmm + SwiGLU + down gather_qmm + weighted-sum
into one "MoE expert block" Metal kernel.** Instead of materializing the
intermediate gate_up tensor, the hidden tensor, and the down_out tensor,
stream expert weights through registers / SIMD-shared memory and emit only
the final per-token output. Collapses ~5 dispatches → 1 and eliminates 3
intermediate tensor writes/reads.

This is the "deeper gather+GEMV+weighted-sum fusion" the README names as the
lever. The challenge: MLX's `gather_qmm` is upstream C++ and already
well-tuned for its general case. A specialized kernel (fixed top-k=10,
expert_size=512, batch=1) must beat MLX's general kernel *and* eliminate
the intermediate traffic to win. This is a multi-week Metal engineering
effort with real correctness risk (4-bit affine dequant, scatter-gather
indexing, activation fusion).

Effort: very high. Risk: high. Impact: theoretical max — this is the only
single change that could plausibly move AX from 39% toward dense-level
utilization, because it attacks the dispatch density directly.

### Tier 3 — Graph-level compilation (highest leverage, high effort)

**3A — `mx.compile`-analog for the per-layer MoE forward.** Wrap the entire
MoE block (router + experts + shared + weighted-sum + add + norm) in a
compiled MLX graph so per-op dispatch is amortized into a single compiled
dispatch at first call. This is item 1 of "what might still work" in
`PERFORMANCE-DECODE-GAP.md`. MLX's compiler can fuse elementwise ops and
optimize the dispatch schedule without hand-written kernels.

Blockers: requires `mlx-sys` exposure of `mlx_compile` (the FFI does not
currently wrap it), careful handling of the KV-cache / recurrent SSM state
as non-traced inputs/outputs, and debugging graph-compilation failures on
dynamic shapes (top-k indices).

Effort: high. Risk: medium–high (compile can fail on dynamic shapes).
Impact: potentially closes most of the dispatch-overhead gap across *all*
layers, not just the MoE block. This is the most leveraged single investment.

**3B — Whole-layer Metal kernel.** Item 2 of
`PERFORMANCE-DECODE-GAP.md`. One kernel for the entire post-attention FFN.
Effort: extreme. Risk: very high (maintainability, correctness drift).
Not recommended unless 3A proves insufficient.

## What prior art rules out (and what does not apply here)

`PERFORMANCE-DECODE-GAP.md` closed the dense-decode gap investigation with
these negative results, which are **binding for the dense path** but **may
not apply to MoE**:

- **Fused `add + rms_norm` regressed on dense** because dense `rms_norm` is
  already bandwidth-bound. On MoE, the post-FFN `rms_norm` follows a block
  that is *dispatch-bound*, so fusion here is about dispatch elimination,
  not bandwidth — the prior regression does not automatically transfer.
- **Direct C++ Gemma4 post-attn FFN route regressed.** That was a whole-FFN
  C++ shim on a dense path. It does not speak to MoE gather_qmm fusion.
- **Single-op fusion has a low ceiling (~0.2%) on dense.** True for dense,
  where each op is bandwidth-bound. On MoE, each eliminated dispatch raises
  bus duty cycle — the ceiling is higher (Tier 1 estimate: +5–10%).

The binding conclusion from prior art that **does** transfer: **single-op
fusion alone will not reach dense-level utilization.** Reaching 78–86% on
MoE requires either Tier 2 (deep kernel fusion) or Tier 3 (graph compile).

## Recommended path

1. **0A (measure).** Add MoE sub-stage profiling. Run
   `AX_MLX_DECODE_PROFILE=1` + the new MoE sub-stages on Qwen3-Coder-Next
   to confirm the dispatch-count hypothesis and get per-stage wall time.
   ~1 day. Zero risk.
2. **1A + 1B (incremental fusion).** Fuse shared-expert add into
   weighted-sum; route MoE SwiGLU through the packed kernel. Each is
   independently shippable behind an env flag with A/B evidence. ~1 week.
   Expected: +2–5%.
3. **3A (graph compile).** Expose `mlx_compile` in `mlx-sys` and wrap the
   per-layer MoE forward. This is the highest-leverage single investment
   and the only Tier 3 option that does not require hand-written Metal.
   ~2–4 weeks. Expected: potentially closes most of the dispatch-overhead
   gap.
4. **2A (deep expert-block kernel) — only if 3A is insufficient.** This is
   the Metal engineering path to dense-level utilization. ~4–8 weeks.

The single most important caveat: **the 39% number is derived, not measured
with performance counters, and `peak_bandwidth_gb_s` was null in the bench
run.** Before investing in any of the above, a direct Metal performance-counter
measurement of bus utilization during MoE decode would confirm (or revise)
the ~40-point gap estimate. The derived number is internally consistent
with the dense-model comparison, but a counter-based reading is the
ground truth.

## Related artifacts

- README bandwidth table and headroom note: `README.md:684-698`
- MoE expert forward: `crates/ax-engine-mlx/src/model/shared/mlp.rs:1493-1591`
- Router: `crates/ax-engine-mlx/src/model/shared/mlp.rs:1127-1155`
- Shared expert: `crates/ax-engine-mlx/src/model/shared/mlp.rs:1048-1095`
- Weighted-sum kernel: `crates/ax-engine-mlx/src/model/shared/mlp.rs:403-501`
- Packed SwiGLU kernel (dense only): `crates/ax-engine-mlx/src/model/shared/mlp.rs:342-356,528-535`
- Profiler (Gemma4-gated): `crates/ax-engine-mlx/src/model/profile.rs:209-278`
- Prior dense-gap investigation: `docs/PERFORMANCE-DECODE-GAP.md`
- Roadmap MoE track: `docs/ROADMAP.md:24`
