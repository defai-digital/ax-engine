"""M5 microbench: MLX quantized_matmul at 6-bit (gs 64) for the Qwen3.8-27B 5120-wide verify shapes, M=1..6,
plus the repo's generic invariant qmv_fast kernel at 6-bit for Leading<=4. 16 launches per eval over 8 weight copies."""
import pathlib, re, statistics, sys, time
import mlx.core as mx
utils = pathlib.Path(sys.argv[1]).read_text()
def kernel_source(name):
    return re.search(rf'const {name}: &str = r#"(.*?)"#;', utils, re.S).group(1)
GENERIC = kernel_source("INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE")
k_generic = mx.fast.metal_kernel(name="ax_inv_qmv_generic", input_names=["x", "weight", "scales", "biases"], output_names=["out"], source=GENERIC)
def run(kernel, x, wq, sc, bi, leading, K, N, gs, bits):
    return kernel(inputs=[x, wq, sc, bi],
                  template=[("InputT", mx.bfloat16), ("OutT", mx.bfloat16), ("Leading", leading), ("OutDim", N), ("InputDim", K),
                            ("PackedCols", wq.shape[1]), ("GroupSize", gs), ("GroupCount", K // gs), ("Bits", bits),
                            ("PackFactor", 4 if bits == 6 else 32 // bits), ("QuantMask", (1 << bits) - 1)],
                  grid=(32, (N // 8) * 2, 1), threadgroup=(32, 2, 1), output_shapes=[(leading, N)], output_dtypes=[mx.bfloat16])[0]
COPIES, LAUNCHES = 8, 16
def bench(fn, iters=5):
    for _ in range(2): mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)])
    mx.synchronize(); ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)]); mx.synchronize()
        ts.append((time.perf_counter() - t) * 1000 / LAUNCHES)
    return statistics.median(ts)
bits, gs = 6, 64
shapes = [("in_proj_qkv", 5120, 10240), ("in_proj_z", 5120, 6144), ("out_proj", 6144, 5120), ("attn_o", 6144, 5120),
          ("down_proj", 17408, 5120), ("gate_proj", 5120, 17408)]
mx.random.seed(0)
for name, K, N in shapes:
    ws = []
    for _ in range(COPIES):
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits); mx.eval(wq, sc, bi); ws.append((wq, sc, bi))
    gb = sum(a.nbytes for a in ws[0]) / 1e9
    for leading in (1, 2, 3, 4, 5, 6):
        x = (mx.random.normal((leading, K)) * 0.5).astype(mx.bfloat16); mx.eval(x)
        tm = bench(lambda i: mx.quantized_matmul(x, ws[i][0], ws[i][1], ws[i][2], transpose=True, group_size=gs, bits=bits))
        line = f"{name:12s} K={K:5d} N={N:5d} M={leading}  mlx {tm:6.3f} ms ({gb/tm*1000:4.0f} GB/s)"
        if leading <= 4:
            ref = mx.quantized_matmul(x, *ws[0], transpose=True, group_size=gs, bits=bits)
            o = run(k_generic, x, *ws[0], leading, K, N, gs, bits); mx.eval(o, ref)
            md = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))))
            tg = bench(lambda i: run(k_generic, x, *ws[i], leading, K, N, gs, bits))
            line += f"  generic_q6 {tg:6.3f} ms ({gb/tg*1000:4.0f} GB/s) |diff|max={md:.4f}"
        print(line, flush=True)
