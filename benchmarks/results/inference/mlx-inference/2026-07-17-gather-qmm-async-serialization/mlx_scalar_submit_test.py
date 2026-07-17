#!/usr/bin/env python3
"""Hypothesis test: does submitting an argmax SCALAR to async_eval collapse
overlap vs submitting the [1,H] tensor? Mirrors AX's direct pipeline, which
async_eval's argmax(logits) (a scalar) while chaining the hidden state.

Both modes chain the hidden [1,H]; they differ only in WHAT is handed to
async_eval — the tensor (h) or the scalar token (argmax(h @ vocab)).

Run under raised caps (where the tensor version overlapped ~94-99%):
  MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000 \
    ./mlx032venv/bin/python mlx_scalar_submit_test.py tensor
  MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000 \
    ./mlx032venv/bin/python mlx_scalar_submit_test.py scalar
"""

import sys
import time

import mlx.core as mx

LAYERS, H, E, TOPK, GROUP, BITS, VOCAB = 24, 3072, 64, 8, 64, 4, 32000
ITERS, WARMUP, SPIN_US = 30, 5, 4000
MODE = sys.argv[1] if len(sys.argv) > 1 else "tensor"


def spin(us):
    if us <= 0:
        return
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e6 < us:
        pass


w = mx.random.normal([E, H, H]).astype(mx.bfloat16) * 1e-3
wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
dense = (mx.random.normal([H, H]) * 5e-3).astype(mx.bfloat16)
vocab = (mx.random.normal([H, VOCAB]) * 5e-3).astype(mx.bfloat16)
idx = mx.arange(TOPK, dtype=mx.uint32)
mx.eval(wq, scales, biases, dense, vocab, idx)


def step(x):
    for _ in range(LAYERS):
        g = mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx, transpose=True,
                          group_size=GROUP, bits=BITS)
        x = mx.tanh(mx.sum(g, axis=0) + (x @ dense))
    return x  # [1, H]


def run(spin_us):
    h = (mx.ones([1, H]) * 0.01).astype(mx.bfloat16)
    mx.eval(h)

    def build(prev_h):
        nh = step(prev_h)                       # [1, H], full chain
        tok = mx.argmax(nh @ vocab, axis=-1)    # scalar token, depends on nh
        return nh, tok

    ph, pt = build(h)
    submit = pt if MODE == "scalar" else ph
    mx.async_eval(submit)
    for _ in range(WARMUP):
        nh, nt = build(ph)
        mx.async_eval(nt if MODE == "scalar" else nh)
        mx.eval(pt if MODE == "scalar" else ph)
        ph, pt = nh, nt
    t0 = time.perf_counter()
    for _ in range(ITERS):
        spin(spin_us)
        nh, nt = build(ph)
        mx.async_eval(nt if MODE == "scalar" else nh)
        mx.eval(pt if MODE == "scalar" else ph)
        ph, pt = nh, nt
    mx.eval(ph, pt)
    return (time.perf_counter() - t0) / ITERS * 1e6


base, inj = run(0), run(SPIN_US)
absorbed = SPIN_US - (inj - base)
print(f"MLX {mx.__version__}  submit={MODE}")
print(f"  spin=0     : {base:8.1f} us/iter")
print(f"  spin={SPIN_US}us : {inj:8.1f} us/iter (delta {inj-base:+.1f})")
print(f"  absorbed   : {absorbed:8.1f} us of {SPIN_US} ({absorbed/SPIN_US*100:.0f}% overlap)")
