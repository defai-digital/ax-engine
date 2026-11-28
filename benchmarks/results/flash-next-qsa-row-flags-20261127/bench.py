import sys, time, mlx.core as mx
S = sys.argv[1]
body = open(S + "/old_block_key_body.metal").read(); hdr = open(S + "/block_key_header.metal").read()
scan = body[body.index("        int finite = 0;"):body.index("flags[row] = (bad != 0 || finite < keep_q) ? 1 : 0;") + len("flags[row] = (bad != 0 || finite < keep_q) ? 1 : 0;")]
keys_only = body.replace("if (block == 0) {\n    const size_t row = i / blocks;\n    if (decode_keys != 0) {\n        flags[row] = 0;\n    } else {\n" + scan + "\n    }\n}", "")
assert keys_only != body and "flags" not in keys_only, keys_only[-400:]
FLAG = r"""
const int rows_n = limits[0];
const int blocks_n = limits[1];
const int seq_n = limits[2];
const int keep_q = limits[3];
const uint lane = thread_index_in_simdgroup;
const size_t row = thread_position_in_grid.x / 32;
if (blocks_n <= 0 || seq_n <= 0 || row >= (size_t)rows_n) return;
const int limit = completes[row % (size_t)seq_n];
const device float* line = scores + row * (size_t)blocks_n;
uint bad = 0u;
for (int b = (int)lane; b < limit; b += 32) {
    bad |= (((as_type<uint>(line[b]) >> 23) & 0xffu) == 0xffu) ? 1u : 0u;
}
bad = simd_max(bad);
if (lane == 0u) {
    flags[row] = (bad != 0u || limit < keep_q) ? 1 : 0;
}
"""
k_cur = mx.fast.metal_kernel(name="cur", input_names=["scores","completes","limits"], output_names=["keys","flags"], source=body, header=hdr)
k_keys = mx.fast.metal_kernel(name="keys", input_names=["scores","completes","limits"], output_names=["keys"], source=keys_only, header=hdr)
k_flag = mx.fast.metal_kernel(name="flag", input_names=["scores","completes","limits"], output_names=["flags"], source=FLAG, header=hdr)
def cur(scores, completes, seq, blocks, keep):
    count = seq * blocks; lim = mx.array([count, blocks, seq, keep, 0, 0, 0, 0], dtype=mx.int32)
    return k_cur(inputs=[scores, completes, lim], grid=((count+63)//64*64,1,1), threadgroup=(64,1,1),
                 output_shapes=[(1, seq, blocks), (1, seq)], output_dtypes=[mx.int64, mx.int32])
def new(scores, completes, seq, blocks, keep):
    count = seq * blocks; lim = mx.array([count, blocks, seq, keep, 0, 0, 0, 0], dtype=mx.int32)
    keys = k_keys(inputs=[scores, completes, lim], grid=((count+63)//64*64,1,1), threadgroup=(64,1,1),
                  output_shapes=[(1, seq, blocks)], output_dtypes=[mx.int64])[0]
    flim = mx.array([seq, blocks, seq, keep, 0, 0, 0, 0], dtype=mx.int32)
    flags = k_flag(inputs=[scores, completes, flim], grid=(32 * seq, 1, 1), threadgroup=(32, 1, 1),
                   output_shapes=[(1, seq)], output_dtypes=[mx.int32])[0]
    return keys, flags
def t(fn, n=30):
    mx.eval(fn())
    t0 = time.perf_counter()
    for _ in range(n): mx.eval(fn())
    return (time.perf_counter() - t0) / n * 1e3
seq = 2048
ok = True
for offset_blocks in [0, 100, 512, 1600, 2744]:
    blocks = offset_blocks + 512
    scores = mx.random.normal((1, seq, blocks)).astype(mx.float32)
    completes = mx.array([min(blocks, offset_blocks + (q + 1) // 4) for q in range(seq)], dtype=mx.int32)
    for poison in [None, "nan", "inf", "future", "sub"]:
        s = scores
        if poison:
            s = mx.array(s)  # copy
            import numpy as np
            a = np.array(s); r = 1500; c = {"nan": 3, "inf": 7, "future": blocks - 1, "sub": 5}[poison]
            a[0, r, c] = {"nan": np.nan, "inf": -np.inf, "future": np.nan, "sub": np.float32(1e-40)}[poison]
            s = mx.array(a)
        a = cur(s, completes, seq, blocks, 512); b = new(s, completes, seq, blocks, 512); mx.eval(a, b)
        same = bool(mx.array_equal(a[0], b[0]).item()) and bool(mx.array_equal(a[1], b[1]).item())
        ok &= same
        if not same: print("MISMATCH", blocks, poison, mx.sum(a[1]).item(), mx.sum(b[1]).item())
    print(f"blocks={blocks}: current {t(lambda: cur(scores, completes, seq, blocks, 512)):.3f} ms, "
          f"keys+flag kernels {t(lambda: new(scores, completes, seq, blocks, 512)):.3f} ms, flagged rows {mx.sum(a[1]).item()}")
print("all identical:", ok)
