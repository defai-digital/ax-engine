# AX-side decode overlap residual — the next local decode lever

**Date:** 2026-07-17 · **MLX 0.32.0 (minos 26.2, shipped build)** · **M3 Max** ·
**Model:** Qwen3-Coder-Next-4bit · Companion to
[`gather-qmm-async-serialization.md`](gather-qmm-async-serialization.md).

## The finding

After `AX_MLX_AUTO_BUFFER_CAPS` (which fixed the MLX-side gather_qmm command-
buffer splitting), the AX direct-decode pipeline **still overlaps host graph
build with GPU execution only ~11%**, while a structurally-identical pure-MLX
loop overlaps 94% on the same MLX 0.32.0. This is an **ax-side** gap, not
upstream, and it is the last local decode lever.

| measurement (0.32.0, caps on) | value |
| --- | ---: |
| decode | 97.8 tok/s |
| total wall | 10,223 µs/tok |
| forward (host graph build) | 2,387 µs/tok (23% of step) |
| host-sleep injection absorbed (of 4 ms) | 447 µs (**11% overlap**) |
| pure-MLX equivalent overlap | **94%** |

Barrier diagnostic (`AX_MLX_DIRECT_PIPELINE_BARRIER=1`, forces N+1 GPU
completion mid-loop):

| bucket | value |
| --- | ---: |
| forward (host build) | 2,134 µs |
| `async_eval` submit call | **6,365 µs** |
| next-complete (N+1 GPU work) | 5,459 µs |
| pending eval (await N) | 0.3 µs |

Two facts from the barrier split:

1. **There is real GPU work to hide under** — next-complete is 5.5 ms, well over
   the 2.4 ms host build. An ideal double buffer would hide the host build
   entirely.
2. **`async_eval` is effectively blocking** — the submit call itself measures
   ~6.4 ms. A true non-blocking submit is microseconds. So the "double buffer"
   is not double-buffering: `async_eval(next_token_arr)` is doing the GPU wait
   inline, and the subsequent `eval(pending)` is then a ~0µs no-op.

## Prime suspect

`generate.rs::advance_direct_pipeline_with_timings` submits only the token:

```rust
// KV cache is in next_token_arr's computation graph (via SDPA), so no extra
// refs needed — they would only add one GPU command buffer per layer (≈85µs each).
async_eval(&[&next_token_arr]);
```

Hypothesis: `next_token_arr = argmax(logits)` reduces the whole
`[1, vocab]` logits to a scalar, so its graph, once submitted, may force MLX to
schedule the reduction in a way that blocks — or the KV-writes (deliberately not
in the async refs) are what actually keeps the GPU busy across the next build,
and by leaving them out the submit returns only after the token is done rather
than launching-and-returning. Either way the ordering that overlaps in pure-MLX
does not here.

## Why this is the right next step (and its risk)

- **Local and in-scope.** No upstream dependency (pure-MLX overlaps fine on the
  same 0.32.0), no product sign-off (single-request decode), no new kernels.
- **Sized.** ~2.4 ms/step of a 10.2 ms step is serial host build; closing it
  toward the pure-MLX 94% is worth ~15–20% decode on MoE-class models.
- **Risk: it is the hottest path.** Changing the async/eval ordering or the
  async-eval ref set is throughput-sensitive and must be parity-gated
  (greedy checksum unchanged) and A/B'd. This is the I7 "zero-bubble" item from
  the skeleton spec, now shown to be **ax-side-addressable**, not upstream-gated
  as previously (wrongly) concluded.

## Suggested first experiment (parity-gated, low commitment)

Add the KV-cache arrays to the `async_eval` refs (undo the `≈85µs/layer`
optimization) and re-measure overlap + decode. If overlap jumps toward the
pure-MLX 94%, the KV-not-in-refs decision was starving the GPU and the ~85µs/
layer cost is dwarfed by the overlap win. If it does not move, the blocker is
the argmax-reduction submit and the next experiment is submitting the pre-argmax
logits (or the KV refs) instead. Each is a one-line change, parity-checked
against CN + Llama, A/B'd on 0.32.0.
