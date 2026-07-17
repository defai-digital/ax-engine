#!/usr/bin/env python3
"""Minimal pure-MLX repro: batch=1 gather_qmm chain gets no host/GPU overlap
under async_eval, and MLX_MAX_MB_PER_BUFFER changes it.

No ax-engine. Just MLX. Measures whether injected host time between building
step N+1 and awaiting step N is absorbed (overlap) or added 1:1 (serialized).

Run:
  python3 mlx_overlap_repro.py
  MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000 python3 mlx_overlap_repro.py
"""

import os
import time

import mlx.core as mx

LAYERS = 24
H = 3072
E = 64  # experts
TOPK = 8
GROUP = 64
BITS = 4
ITERS = 30
WARMUP = 5
SPIN_US = 4000


def spin(us):
    if us <= 0:
        return
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e6 < us:
        pass


# Shared 4-bit expert stack [E, H, H], quantized once.
w = mx.random.normal([E, H, H]).astype(mx.bfloat16) * 1e-3
wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
dense = (mx.random.normal([H, H]) * 5e-3).astype(mx.bfloat16)
idx = mx.arange(TOPK, dtype=mx.uint32)  # constant routed experts [TOPK]
mx.eval(wq, scales, biases, dense, idx)


def step(x):
    # 24-layer batch=1 MoE-shaped chain: gather_qmm over routed experts + a
    # dense ballast, tanh to keep magnitude bounded.
    for _ in range(LAYERS):
        g = mx.gather_qmm(
            x, wq, scales, biases, rhs_indices=idx, transpose=True,
            group_size=GROUP, bits=BITS,
        )  # [TOPK, 1, H]
        x = mx.tanh(mx.sum(g, axis=0) + (x @ dense))  # [1, H]
    return x


def run(spin_us):
    x0 = (mx.ones([1, H]) * 0.01).astype(mx.bfloat16)
    mx.eval(x0)
    pending = step(x0)
    mx.async_eval(pending)
    for _ in range(WARMUP):
        nxt = step(pending)
        mx.async_eval(nxt)
        mx.eval(pending)
        pending = nxt
    t0 = time.perf_counter()
    for _ in range(ITERS):
        spin(spin_us)               # emulate host graph-build time for N+1
        nxt = step(pending)         # build N+1
        mx.async_eval(nxt)          # submit N+1
        mx.eval(pending)            # await N
        pending = nxt
    mx.eval(pending)
    return (time.perf_counter() - t0) / ITERS * 1e6  # us/iter


caps = (
    os.environ.get("MLX_MAX_MB_PER_BUFFER", "default"),
    os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"),
)
print(f"MLX {mx.__version__}  caps(MB,ops)={caps}")
base = run(0)
inj = run(SPIN_US)
absorbed = SPIN_US - (inj - base)
print(f"  spin=0     : {base:8.1f} us/iter")
print(f"  spin={SPIN_US}us : {inj:8.1f} us/iter  (delta {inj - base:+.1f})")
print(
    f"  absorbed   : {absorbed:8.1f} us of {SPIN_US} injected "
    f"({absorbed / SPIN_US * 100:.0f}% overlap)"
)
