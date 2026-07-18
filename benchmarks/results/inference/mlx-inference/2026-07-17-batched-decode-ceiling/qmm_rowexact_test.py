import mlx.core as mx
H, OUT, B = 4096, 14336, 8
w = mx.random.normal([OUT, H]).astype(mx.bfloat16) * 1e-2
wq, s, b = mx.quantize(w, group_size=64, bits=4); mx.eval(wq,s,b)
x = mx.random.normal([B, 1, H]).astype(mx.bfloat16); mx.eval(x)
def qmm(xx): return mx.quantized_matmul(xx, wq, scales=s, biases=b, transpose=True, group_size=64, bits=4)
# batched: one call over all B rows
batched = qmm(x); mx.eval(batched)
# per-row (RowExact): slice each row, matmul, concat
rows = [qmm(x[r:r+1]) for r in range(B)]
perrow = mx.concatenate(rows, axis=0); mx.eval(perrow)
maxdiff = float(mx.max(mx.abs(batched - perrow)).item())
print(f"MLX {mx.__version__}  batched vs per-row quantized_matmul [B={B},1,{H}]x[{H},{OUT}]")
print(f"  max_abs_diff = {maxdiff:.3e}")
print("BIT-IDENTICAL" if maxdiff == 0.0 else ("CLOSE" if maxdiff < 1e-3 else "DIFFERENT"))
