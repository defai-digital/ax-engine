# Upstream MLX issue (DRAFT for review) — gather_qmm buffer accounting defeats async_eval overlap on batch=1 MoE

> **Status:** draft, awaiting owner review before filing at
> [ml-explore/mlx](https://github.com/ml-explore/mlx/issues). Everything below
> the line is the proposed issue body. The repro (`mlx_overlap_repro.py`) is
> pure MLX — no downstream code — and is included inline.

---

## Title

`gather_qmm` counts the full expert tensor against `MLX_MAX_MB_PER_BUFFER`, forcing command-buffer splits that defeat `async_eval` overlap on batch=1 MoE decode

## Summary

On batch=1 Mixture-of-Experts decode, each layer's `gather_qmm` reads only
`top_k / num_experts` of the expert weight tensor, but the Metal command
encoder accounts the **full** tensor `data_size()` toward the per-command-buffer
byte budget. A single expert stack (hundreds of MB) exceeds the default
`MLX_MAX_MB_PER_BUFFER` (40–50 MB by architecture) on its own, so every MoE
layer forces a command-buffer commit. With ~24–48 commits per decode step, the
`eval_impl` scheduler backpressure (`n_active_tasks() > MAX_ACTIVE_TASKS`) turns
`async_eval` into a de-facto barrier: host graph build for step N+1 no longer
overlaps GPU execution of step N.

The effect is large and fully reproducible in pure MLX (repro below, M3 Max,
MLX 0.31.2):

| caps | host/GPU overlap (4 ms injection absorbed) |
| --- | ---: |
| default | **42%** |
| `MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000` | **99%** |

Raising the cap so the expert tensor fits restores overlap to 99% and, in a
real Qwen3-Next-class runtime, is worth +14–25% decode throughput
(token-stream-identical). Dense and linear-attention models are unaffected
(their per-layer inputs stay near the default cap).

## Why this is a defect, not just a tuning knob

`gather_qmm` with runtime `rhs_indices` physically reads only the routed
experts (`GatherQMM::eval_gpu` keeps indices on-GPU, no over-read — that part is
correct). But the command-buffer size accounting does not know that: it adds
the whole `[num_experts, ...]` tensor's `data_size()` when the buffer is first
seen (`backend/metal/device.cpp`, roughly
`if (all_inputs_.insert(a.buffer().ptr()).second) buffer_sizes_ += a.data_size();`
then commit when `(buffer_sizes_ >> 20) > max_mb`). So the accounting charges
the full tensor while the kernel reads a small slice of it, and a workload that
is *not* actually large-per-buffer is treated as if it were — splitting on every
MoE layer for no bandwidth reason.

The user-visible consequence is that batch=1 MoE decode silently loses
host/GPU pipelining unless the user knows to raise an undocumented-for-this-
purpose env var to a value derived from their model's expert-tensor size.

## Repro (pure MLX, no downstream code)

```python
# mlx_overlap_repro.py
import os, time
import mlx.core as mx

LAYERS, H, E, TOPK, GROUP, BITS = 24, 3072, 64, 8, 64, 4
ITERS, WARMUP, SPIN_US = 30, 5, 4000

def spin(us):
    if us <= 0: return
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e6 < us: pass

w = mx.random.normal([E, H, H]).astype(mx.bfloat16) * 1e-3
wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
dense = (mx.random.normal([H, H]) * 5e-3).astype(mx.bfloat16)
idx = mx.arange(TOPK, dtype=mx.uint32)               # constant routed experts
mx.eval(wq, scales, biases, dense, idx)

def step(x):
    for _ in range(LAYERS):
        g = mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx, transpose=True,
                          group_size=GROUP, bits=BITS)   # [TOPK, 1, H]
        x = mx.tanh(mx.sum(g, axis=0) + (x @ dense))      # [1, H]
    return x

def run(spin_us):
    x = (mx.ones([1, H]) * 0.01).astype(mx.bfloat16); mx.eval(x)
    pending = step(x); mx.async_eval(pending)
    for _ in range(WARMUP):
        nxt = step(pending); mx.async_eval(nxt); mx.eval(pending); pending = nxt
    t0 = time.perf_counter()
    for _ in range(ITERS):
        spin(spin_us)          # emulate host build time for step N+1
        nxt = step(pending)    # build N+1
        mx.async_eval(nxt)     # submit N+1
        mx.eval(pending)       # await N
        pending = nxt
    mx.eval(pending)
    return (time.perf_counter() - t0) / ITERS * 1e6

caps = (os.environ.get("MLX_MAX_MB_PER_BUFFER", "default"),
        os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"))
print(f"MLX {mx.__version__}  caps(MB,ops)={caps}")
base, inj = run(0), run(SPIN_US)
absorbed = SPIN_US - (inj - base)
print(f"  spin=0     : {base:8.1f} us/iter")
print(f"  spin={SPIN_US}us : {inj:8.1f} us/iter (delta {inj-base:+.1f})")
print(f"  absorbed   : {absorbed:8.1f} us of {SPIN_US} ({absorbed/SPIN_US*100:.0f}% overlap)")
```

Observed on Apple M3 Max 128 GB, reproduced on **both** the latest release
(0.32.0) and 0.31.2 — behavior is essentially identical, so it is not a
regression and is not fixed on latest:

```
MLX 0.32.0  caps(MB,ops)=('default', 'default')
  spin=0     :   5784.3 us/iter
  spin=4000us :   8013.6 us/iter (delta +2229.3)
  absorbed   :   1770.7 us of 4000 (44% overlap)

MLX 0.32.0  caps(MB,ops)=('1024', '1000')
  spin=0     :   5636.5 us/iter
  spin=4000us :   5864.5 us/iter (delta +228.0)
  absorbed   :   3772.0 us of 4000 (94% overlap)

MLX 0.31.2  default caps → 42% overlap ; raised caps → 99% overlap
```

The `spin` emulates host graph-build time for the next step; if the pipeline
overlaps, it is absorbed under the in-flight GPU step. Default caps absorb 42%;
caps large enough to hold the expert tensor absorb 99%.

## Suggested fixes (maintainers' call)

1. **Gather-aware accounting (preferred):** for `gather_qmm` / `gather_mm`,
   charge the index-selected footprint (`~ top_k/num_experts` of the tensor, or
   the per-dispatch resident slice) toward `buffer_sizes_` rather than the full
   `data_size()`. This matches what the kernel actually reads.
2. **Exempt already-resident read-only weight buffers** from re-counting across
   consecutive dispatches in the same command-buffer sequence, so a repeatedly-
   referenced expert stack does not force a split every layer.
3. **At minimum, document** that batch=1 MoE decode benefits from raising
   `MLX_MAX_MB_PER_BUFFER` to cover the largest expert tensor, since the current
   default silently disables pipelining for this common case.

## Environment

- MLX **0.32.0** (pip, latest at time of filing) and MLX 0.31.2 — reproduced
  identically on both. Apple M3 Max, 128 GB, macOS.

## Scope note

This issue is strictly about buffer-size accounting for gather ops. `async_eval`
itself is not implicated: the repro shows it overlaps to 99% once the accounting
no longer forces per-layer splits.
