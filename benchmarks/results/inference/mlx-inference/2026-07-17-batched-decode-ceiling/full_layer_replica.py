import time, mlx.core as mx
# Llama-8B-ish dims
H, KVH, HD, FFN, LAYERS, KVLEN, VOCAB = 4096, 1024, 128, 14336, 32, 40, 128256
NHEADS = H // HD
def q4(shape):
    w = mx.random.normal(shape).astype(mx.bfloat16) * 1e-2
    return mx.quantize(w, group_size=64, bits=4)
Wq = q4([H, H]); Wk = q4([KVH, H]); Wv = q4([KVH, H]); Wo = q4([H, H])
Wg = q4([FFN, H]); Wu = q4([FFN, H]); Wd = q4([H, FFN])
Wlm = q4([VOCAB, H])
def qmm(x, W): return mx.quantized_matmul(x, W[0], scales=W[1], biases=W[2], transpose=True, group_size=64, bits=4)
import os
print(f"MLX {mx.__version__}  full replica x{LAYERS}  mask={os.environ.get(chr(85)+chr(83)+chr(69)+chr(95)+chr(77)+chr(65)+chr(83)+chr(75),"0")}")
print("batch  step_us  scaling")
base=None
for B in [1,2,4,8]:
    x0 = mx.random.normal([B,1,H]).astype(mx.bfloat16)
    K0 = mx.random.normal([B,KVH//HD,KVLEN,HD]).astype(mx.bfloat16)
    import os
    USE_MASK = os.environ.get('USE_MASK','0')=='1'
    mask = mx.zeros([B,1,1,KVLEN+1], dtype=mx.bfloat16) if USE_MASK else None
    V0 = mx.random.normal([B,KVH//HD,KVLEN,HD]).astype(mx.bfloat16)
    mx.eval(x0,K0,V0)
    def step():
        x = x0
        for _ in range(LAYERS):
            q = qmm(x, Wq).reshape(B,1,NHEADS,HD).transpose(0,2,1,3)
            k = qmm(x, Wk).reshape(B,1,KVH//HD,HD).transpose(0,2,1,3)
            v = qmm(x, Wv).reshape(B,1,KVH//HD,HD).transpose(0,2,1,3)
            kk = mx.concatenate([K0, k], axis=2); vv = mx.concatenate([V0, v], axis=2)
            a = mx.fast.scaled_dot_product_attention(q, kk, vv, scale=1.0/(HD**0.5), mask=mask)
            a = a.transpose(0,2,1,3).reshape(B,1,H)
            x = x + qmm(a, Wo)
            g = qmm(x, Wg); u = qmm(x, Wu)
            x = x + qmm((g*mx.sigmoid(g))*u, Wd)
        logits = qmm(x, Wlm)
        tok = mx.argmax(logits, axis=-1)
        return tok
    for _ in range(3): mx.eval(step())
    t0=time.perf_counter(); N=15
    for _ in range(N): mx.eval(step())
    us=(time.perf_counter()-t0)/N*1e6
    agg = B/(us*1e-6)
    if base is None: base=agg
    print(f"{B:>5}  {us:>7.0f}  {agg/base:>5.2f}x")
