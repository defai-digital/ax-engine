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

## Experiment 1 — KV-in-async-refs: REFUTED (2026-07-17)

Added the KV-cache arrays to the `async_eval` submit (env-gated,
`AX_MLX_DIRECT_ASYNC_KV_REFS`), rebuilt against 0.32.0, A/B'd on
Qwen3-Coder-Next-4bit:

| | async_eval submit | decode (sleep 0) | decode (sleep 4 ms) | parity |
| --- | ---: | ---: | ---: | --- |
| off | 7,725 µs | 99.0 tok/s | 74.4 | `296c5760a58d6aa2` |
| on (KV refs) | 7,863 µs | 97.6 tok/s | 74.0 | identical |

**No effect.** The submit still blocks at ~7.8 ms, overlap is unchanged (a 4 ms
injection still degrades both to ~74 tok/s), decode is flat-to-slightly-worse.
So the KV-not-in-refs decision is **not** what starves the pipeline — the
`≈85µs/layer` comment's premise is fine, and the blocker is elsewhere. The
env flag was reverted (do-nothing branch in the hottest path); the negative
result stands on its own.

## Remaining hypotheses (deeper, each a rebuild)

1. **Argmax-scalar submit.** `async_eval(&[&next_token_arr])` submits a scalar
   (`argmax` of `[1, vocab]`). Submitting a scalar may force MLX to schedule the
   whole reduction chain synchronously, where the pure-MLX repro (which submits
   a `[1, H]` tensor) does not. Test: submit the pre-argmax logits (or the last
   hidden) and read the token from the materialized logits, comparing overlap.
2. **MLX async_eval scheduler semantics for single-outstanding graphs.** The
   pure-MLX repro keeps exactly one async in flight and consumes it immediately;
   AX does too, but the real forward's graph is far larger. Whether MLX's
   `async_eval` returns before launch depends on internal scheduler state that
   may differ for a 48-layer quantized graph vs the 24-layer repro. Test:
   instrument `mlx_async_eval` return timing vs graph size directly.

## Experiment 2 — argmax-scalar submit: REFUTED (clean-room, 2026-07-17)

Before touching the ax hot path, tested hypothesis 1 in the minimal MLX repro
(`mlx_scalar_submit_test.py`): same 24-layer gather_qmm chain, but async_eval
handed an `argmax(h @ vocab)` **scalar** (the AX pattern) vs the `[1, H]`
**tensor**, both at raised caps on 0.32.0:

| submit | overlap (4 ms injection absorbed) |
| --- | ---: |
| tensor (control) | 102% |
| **argmax scalar (AX pattern)** | **99%** |

**Refuted.** Submitting a scalar overlaps just as well as a tensor in pure MLX,
so the scalar submit is not what blocks the AX pipeline. The clean-room test
cost a 5-minute Python edit and saved an ax rebuild+A/B cycle chasing a wrong
hypothesis.

## RESOLVED (2026-07-17): it is not a sync bug — decode is host-graph-encoding-bound

Continuing the clean-room discipline refuted every structural hypothesis and
then found the real mechanism:

- **mlx-sys shim async_eval is fine** — the `async-eval-overlap-probe` (mlx-sys,
  not Python) overlaps 94–100% on 0.32.0 at raised caps, async-call <600µs.
- **Not gather_qmm, KV-cache slice_update+SDPA, argmax-scalar submit, or custom
  Metal kernel dispatch** — a probe reproducing each overlaps fine.
- **The block is AT async_eval, not mid-forward** — decode-trace's forward-build
  bucket is only 2,387µs (lazy graph construction); the cost is the 7,819µs
  `async_eval` call itself.
- **`async_eval` host cost scales ~linearly with graph size** — quadrupling the
  probe's layer count (24→96) grew async-call ~400µs→~1,400µs. So the 7.8ms on
  the real 48-layer linear-attn+MoE forward is **host-side graph encoding**, not
  a GPU wait.

**The reframe.** Per step on Qwen3-Coder-Next-4bit (0.32.0):

| | µs/step |
| --- | ---: |
| host: forward build | 2,387 |
| host: async_eval graph encoding | 7,819 |
| **host total** | **~10,200** |
| GPU compute (barrier next-complete) | 5,459 |

Host (10.2 ms) **exceeds** GPU (5.5 ms), so decode is **host/graph-encoding-bound,
not GPU-bound** — there is nothing to overlap because the host critical path is
already longer than the GPU work. The pure-MLX repro overlaps 99% only because
there GPU (5.6 ms) > host (0.4 ms); the real forward flips host-bound because its
graph is larger (host↑) and MoE reads few bytes (GPU↓).

**The lever, and why it is gated.** The only way to cut the 7.8ms/step encoding
is to **encode the graph once and reuse it** — graph compilation (`mx.compile` /
the per-layer compiled-closure path). That is the SAME lever as the MoE
per-layer-compile track, which is **blocked on MLX's inability to
shapeless-compile GatherQMM** (`docs/performance/moe-bandwidth-gap.md`,
`fastpath.rs` AX_MLX_MOE_LAYER_COMPILE history). So the decode-overlap residual
and the per-layer-compile track are ONE lever: reduce per-step host
graph-encoding cost via compilation, gated on MLX compile support for gather
ops. Async reordering cannot help — there is no GPU idle to fill.

Ceiling: closing the host-encoding gap toward the 5.5ms GPU floor is up to ~1.85×
decode (10.2→5.5ms) on this model, but it is gated upstream (MLX GatherQMM
compile), which is why it is not a local quick win.

## (superseded) earlier framing: BOTH cheap hypotheses dead

Both concrete, cheap hypotheses are now refuted with evidence:
KV-in-async-refs (experiment 1) and argmax-scalar-submit (experiment 2). The AX
async_eval blocking (~7.8 ms submit, ~11% overlap) **does not reproduce in the
minimal MLX repro**, which overlaps 99–102% with the same scalar-submit + raised
caps. So the blocker is AX-forward-specific — the leading remaining suspect is
the KV-cache mutable-state dependency chain (`slice_update` into persistent
cache arrays every step, which the stateless repro has no analogue for), not
anything in the submit shape.

Finding it requires **instrumenting the real AX forward** (bisect which stage —
attention/SDPA, router argpartition, KV writes — makes async_eval block), an
open-ended investigation on the hottest path, not a quick experiment. Given the
+28% already banked (auto-buffer-caps) and that the two cheap levers are
exhausted, this track is **parked at a natural boundary**: the remaining ~15–20%
is real but now behind open-ended ax-forward work. Recommendation is to weigh
that against Phase 3 batched decode (×2–4 ceiling, needs concurrency-priority
sign-off) rather than continue speculative hot-path spelunking here.
