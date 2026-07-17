#!/usr/bin/env python3
"""Leading-suspect test: does a persistent, per-step slice_update'd KV cache
(the cross-step mutable-state dependency AX has and the stateless repro lacks)
reproduce the async_eval blocking / overlap loss?

Adds real KV-cache attention to the gather_qmm chain: persistent K/V buffers
slice_update'd at the current position every step, attention over [:t+1].
Compares overlap WITHOUT vs WITH the KV cache, both at raised caps on 0.32.0.

  MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000 python mlx_kvcache_dep_test.py nokv
  MLX_MAX_MB_PER_BUFFER=1024 MLX_MAX_OPS_PER_BUFFER=1000 python mlx_kvcache_dep_test.py kv
"""

import sys
import time

import mlx.core as mx

LAYERS, H, E, TOPK, GROUP, BITS = 24, 3072, 64, 8, 64, 4
HEADS, HD, MAXT = 24, 128, 512  # HEADS*HD = 3072 = H
ITERS, WARMUP, SPIN_US = 30, 5, 4000
MODE = sys.argv[1] if len(sys.argv) > 1 else "kv"


def spin(us):
    if us <= 0:
        return
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e6 < us:
        pass


w = mx.random.normal([E, H, H]).astype(mx.bfloat16) * 1e-3
wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
dense = (mx.random.normal([H, H]) * 5e-3).astype(mx.bfloat16)
wqkv = (mx.random.normal([H, 3 * H]) * 5e-3).astype(mx.bfloat16)
mx.eval(wq, scales, biases, dense, wqkv)


def ffn(x):
    for _ in range(LAYERS):
        g = mx.gather_qmm(x, wq, scales, biases, rhs_indices=mx.arange(TOPK, dtype=mx.uint32),
                          transpose=True, group_size=GROUP, bits=BITS)
        x = mx.tanh(mx.sum(g, axis=0) + (x @ dense))
    return x


def run(spin_us):
    x = (mx.ones([1, H]) * 0.01).astype(mx.bfloat16)
    # Persistent KV buffers [1, HEADS, MAXT, HD], slice_update'd each step.
    K = mx.zeros([1, HEADS, MAXT, HD], dtype=mx.bfloat16)
    V = mx.zeros([1, HEADS, MAXT, HD], dtype=mx.bfloat16)
    mx.eval(x, K, V)
    state = {"K": K, "V": V, "t": 0}

    def build(h):
        if MODE == "kv":
            qkv = (h @ wqkv).reshape(1, 3, HEADS, HD)
            q = qkv[:, 0].reshape(1, HEADS, 1, HD)
            k = qkv[:, 1].reshape(1, HEADS, 1, HD)
            v = qkv[:, 2].reshape(1, HEADS, 1, HD)
            t = state["t"]
            # slice_update K/V at position t — the persistent mutable dependency.
            state["K"] = mx.slice_update(state["K"], k, mx.array([0, 0, t, 0]), axes=(0, 1, 2, 3))
            state["V"] = mx.slice_update(state["V"], v, mx.array([0, 0, t, 0]), axes=(0, 1, 2, 3))
            state["t"] = t + 1
            attn = mx.fast.scaled_dot_product_attention(
                q, state["K"][:, :, : t + 1], state["V"][:, :, : t + 1], scale=1.0 / (HD ** 0.5))
            h = h + attn.reshape(1, H)
        nh = ffn(h)
        return nh

    ph = build(x)
    mx.async_eval(ph)
    for _ in range(WARMUP):
        nh = build(ph)
        mx.async_eval(nh)
        mx.eval(ph)
        ph = nh
    t0 = time.perf_counter()
    for _ in range(ITERS):
        spin(spin_us)
        nh = build(ph)
        mx.async_eval(nh)
        mx.eval(ph)
        ph = nh
    mx.eval(ph)
    return (time.perf_counter() - t0) / ITERS * 1e6


base, inj = run(0), run(SPIN_US)
absorbed = SPIN_US - (inj - base)
print(f"MLX {mx.__version__}  mode={MODE}")
print(f"  spin=0     : {base:8.1f} us/iter")
print(f"  spin={SPIN_US}us : {inj:8.1f} us/iter (delta {inj-base:+.1f})")
print(f"  absorbed   : {absorbed:8.1f} us of {SPIN_US} ({absorbed/SPIN_US*100:.0f}% overlap)")
