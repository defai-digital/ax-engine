# Batched hybrid decode: MoE + linear-attention (Qwen3-Next) — Phase 3.7

**Status:** capability complete and numerically correct; **stays opt-in / not
certified for default-on** because batched decode of an MoE model cannot be
bit-exact with per-row decode (the amortization mechanism *is* the drift source).
Host: Apple M3 Max, MLX 0.32.0.

## What was built

Phase 3.1–3.6 batched decode covered **dense full-attention** models. Qwen3-Next
(Qwen3-Coder-Next, Qwen3.6-35B-A3B) is a **hybrid**: gated-delta linear-attention
layers interleaved with periodic full-attention layers, all with **MoE** FFN.
Phase 3.7 extends the batched path to that shape:

- **`BatchedLinearState`** (`batched_linear_state.rs`) — per-row gated-delta
  recurrent + conv1d state store, the sibling of `BatchedKvCache` for linear
  layers. Oracle-tested: row `r` byte-identical to the single-row state.
- **`linear_attention_forward_batched`** (`model/shared/linear_attention.rs`) —
  the `[B,1,H]` batched linear-attention sublayer. Uses the portable projection +
  conv1d + qk-norm composition (batch-general; the batch=1-shaped Metal/direct-C++
  fast paths are bypassed) and the already-batch-native `gated_delta` kernel
  (dispatches over `batch * num_value_heads`, state `[B, Hv, Dv, Dk]`).
- **`ffn_batched`** (`model/families/standard.rs`) — dense SwiGLU **or** qwen3-MoE
  FFN for `[B,1,H]`. MoE is batch-general (`moe_router_qwen3` falls back off its
  batch=1 fused kernel; `gather_qmm` broadcasts the batch dim), after guarding one
  batch=1-shaped activation fast path (`moe_fused_activation_unsort_metal`, which
  hardcodes `[1,1,top_k,hidden]` and would drop lanes 1..B-1).
- **`layer_forward_batched_linear`** + per-layer routing in
  `decode_batched_forward` (full-attention → KV path, linear → recurrent path),
  and **hybrid seeding** in `BatchedDecodeSession` (lazily provisions
  `BatchedLinearState`, seeds KV for full-attention layers and recurrent state for
  linear layers, auto-detected from the prefill cache; swap-removes both stores in
  lockstep on leave).
- **Structural eligibility** (`ax-engine-core/architecture.rs`) — linear attention
  is no longer a rejection. MoE is allowed only when the explicit
  `batched_qwen3_moe_router` capability is set (from model family at
  `ArchitectureSpec` construction, and fail-closed from weight markers on the
  runner path). Linear attention no longer proxies router eligibility.
  Gemma-4 / GPT-OSS / GLM / DeepSeek MoE stay rejected. Runner projection
  completeness is **per-layer kind** (linear layers do not need split Q/K/V/O).

## Correctness: the forward is right

- **Linear unit oracle** (`batched_linear_attention_row_is_byte_identical_to_single_row`):
  row `r` of a batch-`B` linear-attention step is **byte-identical** to a
  standalone batch-1 step — output *and* post-step conv/recurrent state.
- **End-to-end B=1** (`batched_decode_2a_probe`, Qwen3-Coder-Next-4bit, portable
  linear + RowExact projections): **token-exact for 32 steps** vs single-sequence
  decode. At B=1 every batched kernel (`gather_qmm[1,…]`, router, lm_head) is
  shape-identical to per-row, so this isolates the forward's correctness from
  batched-kernel drift.

## Amortization: the win

`batched-decode-ceiling-probe <coder-next> 32`, aggregate tok/s (M3 Max):

| batch | agg tok/s | scaling |
| ----- | --------- | ------- |
| 1     | 62.7      | 1.00×   |
| 2     | 108.8     | 1.73×   |
| 4     | 165.8     | 2.64×   |
| 8     | 240.8     | 3.84×   |

Batched hybrid decode amortizes the expert + projection weight reads across the
cohort exactly as the dense path does.

## Why it is not certified for default-on

`batched_decode_2a_probe` with distinct prompts diverges from single-sequence
greedy at B>1 (first flip around position 8–13, then chaotic). Isolation shows
**this is batched-kernel drift, not a batching bug**:

1. **B=1 is token-exact** (above) — the per-row forward is correct.
2. Forcing *both* arms onto the portable linear path moves the first divergence
   from position 3 to 13 — i.e. most of the raw failure is the batched path being
   portable while production single-row uses fused Metal/shim linear kernels
   (a reduction-order difference, not batching).
3. The residual is the **batched quantized matmuls**: at B>1 MLX selects
   different `gather_qmm` / `quantized_matmul` reductions than per-row (the same
   bf16 drift Phase 3.5 documented for the Shared projection policy). For MoE this
   is sharper than for dense: the router's discrete top-k `argpartition` can flip
   a selection under a 1-bit logit perturbation, which then cascades.

This is **fundamental, not fixable without giving up the win**: batched MoE
amortizes the expert read *through* batched `gather_qmm`, which is precisely the
kernel whose reduction differs from per-row. Bit-exact greedy parity with per-row
decode (ADR-003 D5) and batched-MoE amortization are mutually exclusive. Same
class as the fused-router non-promotion.

**Outcome:** the path is gated by the existing `AX_MLX_BATCHED_DECODE` opt-in and
stays **uncertified**, so it is OFF by default (rejected as `certification_missing`)
and only routes under `AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED=1`. Certification
would require either RowExact MoE experts (which forfeits the amortization) or an
ADR change to a corpus-level greedy-token-agreement bar instead of bit-exactness.

## Reproduce

```bash
export MLX_LIB_DIR=/opt/homebrew/Cellar/mlx/0.32.0/lib
export MLX_INCLUDE_DIR=/opt/homebrew/Cellar/mlx/0.32.0/include
export DYLD_LIBRARY_PATH="$MLX_LIB_DIR:$DYLD_LIBRARY_PATH"; unset MLX_USE_CPU
CN=~/models/models--mlx-community--Qwen3-Coder-Next-4bit/snapshots/*/

# B=1 token-exact (forward correctness):
AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_INPUTS=0 \
AX_MLX_QWEN_LINEAR_ATTENTION_DECODE_POST_INPUT_METAL=0 \
AX_MLX_QWEN_DIRECT_CPP_LINEAR_ATTENTION_POST_INPUT=0 \
AX_MLX_LINEAR_ATTENTION_RMS_NORM_GATE_METAL=0 AX_MLX_BATCHED_SHARED_PROJ=0 \
AX_BATCH=1 AX_PROMPT_LEN=16 AX_GEN=32 \
  ./target/debug/batched_decode_2a_probe "$CN"

# amortization:
./target/debug/batched-decode-ceiling-probe "$CN" 32
```
