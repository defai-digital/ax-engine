import time, mlx.core as mx
H = 4096
CHAIN = 100  # many matmuls in ONE graph, eval once — no per-call submit overhead
w = mx.random.normal([H, H]).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b)
print(f"MLX {mx.__version__}  CHAINED {CHAIN} matmuls [B,{H}]x[{H},{H}] 4-bit, single eval")
print("batch  wall_us/eval  us_per_row  scaling(true amortization)")
base=None
for B in [1,2,4,8,16]:
    x0 = mx.random.normal([B,H]).astype(mx.bfloat16); mx.eval(x0)
    def run():
        x = x0
        for _ in range(CHAIN):
            x = mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True, group_size=64, bits=4)
            x = x * mx.array(0.9, dtype=mx.bfloat16)  # keep magnitude bounded
        return x
    for _ in range(3): mx.eval(run())
    t0=time.perf_counter(); N=20
    for _ in range(N): mx.eval(run())
    us=(time.perf_counter()-t0)/N*1e6
    per_row = us/B
    agg = B/(us*1e-6)
    if base is None: base=agg
    print(f"{B:>5}  {us:>11.1f}  {per_row:>10.1f}  {agg/base:>5.2f}x")
