"""Micro-benchmark: relaxed-verifier split-K QMM (gate/up at S=2..4) vs MLX qmm."""
import pathlib, re, statistics, sys, time
import mlx.core as mx
ROOT = pathlib.Path(sys.argv[1])
src_rs = (ROOT / "crates/ax-engine-mlx/src/model/shared/verify_qmm.rs").read_text()
SPLITK = re.search(r'const VERIFY_QMM_SPLIT_K_SOURCE: &str = r#"(.*?)"#;', src_rs, re.S).group(1)
k_split = mx.fast.metal_kernel(name="ax_splitk", input_names=["x", "weight", "scales", "biases"], output_names=["out"], source=SPLITK)
def run_split(x, wq, sc, bi, m, K, N, gs, bits, kparts):
    return k_split(inputs=[x, wq, sc, bi], template=[("InputT", mx.bfloat16), ("MRows", m), ("Bits", bits), ("GroupSize", gs), ("KDim", K), ("NDim", N), ("KParts", kparts)],
                   grid=(32 * kparts, N // 4, 1), threadgroup=(32 * kparts, 1, 1), output_shapes=[(m, N)], output_dtypes=[mx.bfloat16])[0]
COPIES, LAUNCHES = 8, 16
def bench(fn, iters=5):
    for _ in range(2): mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)])
    mx.synchronize(); ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)]); mx.synchronize(); ts.append((time.perf_counter() - t) * 1000 / LAUNCHES)
    return statistics.median(ts)
mx.random.seed(0)
for name, K, N, gs, bits in [("down_proj", 17408, 5120, 64, 4), ("out_proj", 6144, 5120, 32, 4), ("in_proj_z", 5120, 6144, 32, 4), ("in_proj_qkv", 5120, 10240, 32, 4), ("attn_o", 6144, 5120, 32, 4)]:
    ws = []
    for _ in range(COPIES):
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits); sc = sc.astype(mx.bfloat16); bi = bi.astype(mx.bfloat16)
        mx.eval(wq, sc, bi); ws.append((wq, sc, bi))
    gb = sum(a.nbytes for a in ws[0]) / 1e9
    for m in (2, 3, 4):
        x = (mx.random.normal((m, K)) * 0.5).astype(mx.bfloat16); mx.eval(x)
        for kparts in (2, 4):
            o = run_split(x, *ws[0], m, K, N, gs, bits, kparts); ref = mx.quantized_matmul(x, *ws[0], transpose=True, group_size=gs, bits=bits); mx.eval(o, ref)
            maxdiff = float(mx.max(mx.abs(o.astype(mx.float32) - ref.astype(mx.float32))))
            ts = bench(lambda i: run_split(x, *ws[i], m, K, N, gs, bits, kparts))
            tm = bench(lambda i: mx.quantized_matmul(x, *ws[i], transpose=True, group_size=gs, bits=bits))
            print(f"{name:14s} K={K} N={N:5d} M={m} KParts={kparts}  split_k {ts:6.3f} ms ({gb/ts*1000:4.0f} GB/s)  mlx {tm:6.3f} ms ({gb/tm*1000:4.0f} GB/s)  |diff|max={maxdiff:.4f}", flush=True)
