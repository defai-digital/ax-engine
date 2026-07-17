import time, mlx.core as mx
H, OUT = 4096, 14336  # Llama-8B FFN gate/up-ish
w = mx.random.normal([OUT, H]).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4)
mx.eval(wq, s, b)
print(f"MLX {mx.__version__}  quantized_matmul [B,{H}] x [{H},{OUT}] (4-bit)")
print("batch  wall_us/step  agg_rows/s  scaling")
base = None
for B in [1, 2, 4, 8, 16]:
    x = mx.random.normal([B, H]).astype(mx.bfloat16); mx.eval(x)
    for _ in range(5):
        mx.eval(mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True, group_size=64, bits=4))
    t0 = time.perf_counter()
    N = 100
    for _ in range(N):
        mx.eval(mx.quantized_matmul(x, wq, scales=s, biases=b, transpose=True, group_size=64, bits=4))
    us = (time.perf_counter()-t0)/N*1e6
    agg = B/(us*1e-6)
    if base is None: base = agg
    print(f"{B:>5}  {us:>11.1f}  {agg:>10.0f}  {agg/base:>5.2f}x")
