import time, mlx.core as mx
H, OUT = 4096, 14336
w = mx.random.normal([OUT, H]).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq, s, b)
def bench(shape_fn, label):
    print(f"\n{label}")
    print("batch  wall_us  scaling")
    base=None
    for B in [1,4,8]:
        x = shape_fn(B); mx.eval(x)
        for _ in range(5): mx.eval(mx.quantized_matmul(x,wq,scales=s,biases=b,transpose=True,group_size=64,bits=4))
        t0=time.perf_counter(); N=100
        for _ in range(N): mx.eval(mx.quantized_matmul(x,wq,scales=s,biases=b,transpose=True,group_size=64,bits=4))
        us=(time.perf_counter()-t0)/N*1e6; agg=B/(us*1e-6)
        if base is None: base=agg
        print(f"{B:>5}  {us:>7.1f}  {agg/base:>5.2f}x")
bench(lambda B: mx.random.normal([B,H]).astype(mx.bfloat16), "2D [B, H] (raw)")
bench(lambda B: mx.random.normal([B,1,H]).astype(mx.bfloat16), "3D [B, 1, H] (forward decode shape)")
