import time, mlx.core as mx

def bench(fn, reps=30, warm=5):
    for _ in range(warm):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

for n in (2048, 9000, 39000):
    k = mx.random.normal((1, n, 2, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, n, 2, 256)).astype(mx.bfloat16)
    ik = mx.random.normal((1, n, 128)).astype(mx.bfloat16)
    nk = mx.random.normal((1, 1, 2, 256)).astype(mx.bfloat16)
    nv = mx.random.normal((1, 1, 2, 256)).astype(mx.bfloat16)
    nik = mx.random.normal((1, 1, 128)).astype(mx.bfloat16)
    mx.eval(k, v, ik, nk, nv, nik)

    def concat_all():
        outs = [mx.contiguous(mx.concatenate([k, nk], axis=1)),
                mx.contiguous(mx.concatenate([v, nv], axis=1)),
                mx.contiguous(mx.concatenate([ik, nik], axis=1))]
        mx.eval(*outs)
    per_layer = bench(concat_all)

    # pooled block keys recompute (ratio 4), rms + rope-free proxy
    w = mx.ones((128,), dtype=mx.bfloat16)
    def pooled():
        blocks = n // 4
        g = ik[:, : blocks * 4, :].reshape(1, blocks, 4, 128).astype(mx.float32)
        m = (g.sum(axis=2) / 4.0).astype(mx.bfloat16)
        out = mx.fast.rms_norm(m.astype(mx.float32), w.astype(mx.float32), 1e-6).astype(mx.bfloat16)
        mx.eval(out)
    pooled_ms = bench(pooled)

    # alternative: preallocated buffer with in-place slice update (donation)
    cap = n + 256
    buf = mx.zeros((1, cap, 2, 256), dtype=mx.bfloat16)
    mx.eval(buf)
    state = {"buf": buf, "pos": n}
    def inplace():
        b = state["buf"]
        b[:, state["pos"] : state["pos"] + 1, :, :] = nk
        mx.eval(b)
    inplace_ms = bench(inplace)

    print(f"n={n:6d}  concat K+V+idx per layer {per_layer:7.3f} ms  x12 = {per_layer*12:7.2f} ms | pooled-keys proxy {pooled_ms:6.3f} ms x12 = {pooled_ms*12:6.2f} ms | in-place append (one tensor) {inplace_ms:6.3f} ms")
