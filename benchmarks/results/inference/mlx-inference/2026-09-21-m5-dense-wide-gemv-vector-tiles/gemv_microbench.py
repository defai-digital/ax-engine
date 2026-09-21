"""Micro-benchmark: dense multi-row GEMV over a [in, out] bf16 weight (lm_head shape)."""
import sys, time
import mlx.core as mx

IN, OUT = 5120, 248320
SCALAR = r"""
    uint n = thread_position_in_grid.x;
    if (n >= (uint)OutDim) { return; }
    float acc[Leading];
    for (uint s = 0; s < (uint)Leading; ++s) { acc[s] = 0.0f; }
    for (uint k = 0; k < (uint)InputDim; ++k) {
        float w = static_cast<float>(weight_t[k * (uint)OutDim + n]);
        for (uint s = 0; s < (uint)Leading; ++s) {
            acc[s] = fma(w, static_cast<float>(x[s * (uint)InputDim + k]), acc[s]);
        }
    }
    for (uint s = 0; s < (uint)Leading; ++s) { out[s * (uint)OutDim + n] = static_cast<OutT>(acc[s]); }
"""
# Vectorized: each thread owns Cols adjacent columns, loads Cols bf16 (2*Cols bytes) per k as raw bits.
def vec_source(cols, unroll):
    load = {4: "uint2", 8: "uint4", 2: "uint"}[cols]
    body = f"""
    uint n0 = thread_position_in_grid.x * (uint){cols};
    if (n0 >= (uint)OutDim) {{ return; }}
    float acc[Leading][{cols}];
    for (uint s = 0; s < (uint)Leading; ++s) for (uint c = 0; c < {cols}; ++c) acc[s][c] = 0.0f;
    device const {load}* wv = (device const {load}*)(weight_t);
    threadgroup float xs[Leading * {unroll}];
    uint od = (uint)OutDim / {cols};
    for (uint k = 0; k < (uint)InputDim; k += {unroll}) {{
        #pragma unroll
        for (uint u = 0; u < {unroll}; ++u) {{
            {load} raw = wv[(k + u) * od + (n0 / {cols})];
            float w[{cols}];
"""
    # unpack bf16 bits -> f32
    comps = ["x","y","z","w"][: max(1, cols//2)]
    lines = []
    for i in range(cols):
        comp = comps[i//2] if cols > 2 else ""
        src = f"raw.{comp}" if cols > 2 else "raw"
        shift = "" if i % 2 == 0 else " >> 16"
        lines.append(f"            w[{i}] = as_type<float>(((uint){src}{shift}) << 16);")
    body += "\n".join(lines) + f"""
            for (uint s = 0; s < (uint)Leading; ++s) {{
                float xv = static_cast<float>(x[s * (uint)InputDim + k + u]);
                for (uint c = 0; c < {cols}; ++c) acc[s][c] = fma(w[c], xv, acc[s][c]);
            }}
        }}
    }}
    for (uint s = 0; s < (uint)Leading; ++s) for (uint c = 0; c < {cols}; ++c)
        out[s * (uint)OutDim + n0 + c] = static_cast<OutT>(acc[s][c]);
"""
    return body

def make(name, src):
    return mx.fast.metal_kernel(name=name, input_names=["x","weight_t"], output_names=["out"], source=src)

def run(kernel, x, w, leading, threads_per_col, tg=256):
    ncols_threads = (OUT + threads_per_col - 1) // threads_per_col
    grid_x = ((ncols_threads + tg - 1) // tg) * tg
    return kernel(inputs=[x, w], template=[("OutT", mx.bfloat16), ("OutDim", OUT), ("InputDim", IN), ("Leading", leading)],
                  grid=(grid_x,1,1), threadgroup=(tg,1,1), output_shapes=[(leading, OUT)], output_dtypes=[mx.bfloat16])[0]

def bench(fn, iters=8):
    for _ in range(2): mx.eval(fn())
    mx.synchronize()
    t=time.perf_counter()
    for _ in range(iters): mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter()-t)/iters*1000

mx.random.seed(0)
w = (mx.random.normal((IN, OUT)) * 0.02).astype(mx.bfloat16)
mx.eval(w)
gb = IN*OUT*2/1e9
scalar = make("ax_dense_wide_gemv_wt_v1", SCALAR)
variants = {"scalar": (scalar, 1)}
for cols in (4, 8):
    for unroll in (1, 4):
        variants[f"vec{cols}_u{unroll}"] = (make(f"ax_dense_wide_gemv_wt_c{cols}_u{unroll}", vec_source(cols, unroll)), cols)
for leading in (1, 4, 8):
    x = (mx.random.normal((leading, IN))).astype(mx.bfloat16); mx.eval(x)
    ref = run(scalar, x, w, leading, 1)
    mm = bench(lambda: mx.matmul(x, w))
    print(f"S={leading} matmul(x, W_t)      {mm:7.2f} ms  {gb/mm*1000:6.0f} GB/s")
    for name,(k,cols) in variants.items():
        try:
            ms = bench(lambda: run(k, x, w, leading, cols))
            outv = run(k, x, w, leading, cols); mx.eval(outv)
            exact = bool(mx.array_equal(outv.view(mx.uint16), ref.view(mx.uint16)).item())
            print(f"S={leading} {name:14s}   {ms:7.2f} ms  {gb/ms*1000:6.0f} GB/s  bit_exact_vs_scalar={exact}")
        except Exception as e:
            print(f"S={leading} {name} FAILED: {str(e)[:200]}")
