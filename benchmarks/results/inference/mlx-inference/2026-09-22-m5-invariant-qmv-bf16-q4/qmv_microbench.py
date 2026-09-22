"""Micro-benchmark: invariant affine qmv_fast microbatch kernel, generic vs bf16/4-bit specialisation vs MLX.

Extracts both kernel sources from crates/ax-engine-mlx/src/model/shared/utils.rs
so the benchmark always measures the tree it runs in. 16 launches per eval
over 8 distinct weight copies: single launches of a 20-50 MB projection are
dominated by dispatch and synchronisation overhead and report meaningless
bandwidth.
"""
import pathlib, re, statistics, sys, time
import mlx.core as mx

ROOT = pathlib.Path(__file__).resolve().parents[5] if len(sys.argv) < 2 else pathlib.Path(sys.argv[1])
utils = (ROOT / "crates/ax-engine-mlx/src/model/shared/utils.rs").read_text()
def kernel_source(name):
    return re.search(rf'const {name}: &str = r#"(.*?)"#;', utils, re.S).group(1)
GENERIC = kernel_source("INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE")
BF16_Q4 = kernel_source("INVARIANT_AFFINE_QMV_FAST_BF16_Q4_KERNEL_SOURCE")

def make(name, src):
    return mx.fast.metal_kernel(name=name, input_names=["x", "weight", "scales", "biases"], output_names=["out"], source=src)

def run(kernel, x, wq, sc, bi, leading, K, N, gs, bits):
    return kernel(inputs=[x, wq, sc, bi],
                  template=[("InputT", mx.bfloat16), ("OutT", mx.bfloat16), ("Leading", leading), ("OutDim", N), ("InputDim", K),
                            ("PackedCols", wq.shape[1]), ("GroupSize", gs), ("GroupCount", K // gs), ("Bits", bits),
                            ("PackFactor", 32 // bits), ("QuantMask", (1 << bits) - 1)],
                  grid=(32, (N // 8) * 2, 1), threadgroup=(32, 2, 1), output_shapes=[(leading, N)], output_dtypes=[mx.bfloat16])[0]

COPIES, LAUNCHES = 8, 16
def bench(fn_for_copy, iters=5):
    for _ in range(2): mx.eval(*[fn_for_copy(i % COPIES) for i in range(LAUNCHES)])
    mx.synchronize()
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(*[fn_for_copy(i % COPIES) for i in range(LAUNCHES)]); mx.synchronize()
        ts.append((time.perf_counter() - t) * 1000 / LAUNCHES)
    return statistics.median(ts)

# Qwen3.8-27B AXQ 6bit-MTP projections that take the invariant kernel under the exact profile.
shapes = [("in_proj_qkv", 5120, 10240, 32, 4), ("down_proj", 17408, 5120, 64, 4), ("out_proj", 6144, 5120, 32, 4),
          ("in_proj_z", 5120, 6144, 32, 4), ("attn_o", 6144, 5120, 32, 4)]
k_generic = make("ax_inv_qmv_generic", GENERIC)
k_bf16q4 = make("ax_inv_qmv_bf16_q4", BF16_Q4)
mx.random.seed(0)
for name, K, N, gs, bits in shapes:
    ws = []
    for _ in range(COPIES):
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits)
        mx.eval(wq, sc, bi)
        ws.append((wq, sc, bi))
    wq, sc, bi = ws[0]
    gb = (wq.nbytes + sc.nbytes + bi.nbytes) / 1e9
    for leading in (1, 2, 3, 4):
        x = (mx.random.normal((leading, K)) * 0.5).astype(mx.bfloat16)
        mx.eval(x)
        o1 = run(k_generic, x, wq, sc, bi, leading, K, N, gs, bits)
        o2 = run(k_bf16q4, x, wq, sc, bi, leading, K, N, gs, bits)
        ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=gs, bits=bits)
        mx.eval(o1, o2, ref)
        exact = bool(mx.array_equal(o1.view(mx.uint16), o2.view(mx.uint16)))
        maxdiff = float(mx.max(mx.abs(o1.astype(mx.float32) - ref.astype(mx.float32))))
        tm = bench(lambda i: mx.quantized_matmul(x, ws[i][0], ws[i][1], ws[i][2], transpose=True, group_size=gs, bits=bits))
        t1 = bench(lambda i: run(k_generic, x, *ws[i], leading, K, N, gs, bits))
        t2 = bench(lambda i: run(k_bf16q4, x, *ws[i], leading, K, N, gs, bits))
        print(f"{name:12s} K={K:5d} N={N:5d} gs={gs} S={leading}  generic {t1:6.3f} ms ({gb/t1*1000:4.0f} GB/s)  bf16_q4 {t2:6.3f} ms ({gb/t2*1000:4.0f} GB/s)  ratio {t1/t2:.3f}  mlx {tm:6.3f} ms ({gb/tm*1000:4.0f} GB/s)  bf16_q4==generic bits:{exact} |generic-mlx|max={maxdiff:.4f}", flush=True)
