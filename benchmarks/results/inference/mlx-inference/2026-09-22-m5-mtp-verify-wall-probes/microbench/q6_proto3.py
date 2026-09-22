"""Prototype: bf16-input / 6-bit affine microbatch qmv kernel (Leading 1..8) vs MLX quantized_matmul.
16 values per thread (12 bytes = 3 uints of contiguous little-endian 6-bit fields), 512-wide K blocks,
x kept as raw bf16 bit pairs, weights unpacked to unscaled 6-bit integers, f32 accumulate, simd_sum per row."""
import statistics, sys, time
import mlx.core as mx

Q6_E = r"""
    constexpr uint ValuesPerThread = 8;
    constexpr uint BytesPerThread = 6;
    constexpr uint BlockSize = ValuesPerThread * 32;
    constexpr uint RowsPerSimd = 4;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * RowsPerSimd;

    float result[Leading][RowsPerSimd];
    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        uint x_bits[Leading][4];
        float x_sums[Leading];
        for (uint token = 0; token < (uint)Leading; ++token) {
            const device uint4* x_row = reinterpret_cast<const device uint4*>(
                x + token * (uint)InputDim + k + lane * ValuesPerThread);
            uint4 xl = x_row[0];
            x_bits[token][0] = xl.x; x_bits[token][1] = xl.y;
            x_bits[token][2] = xl.z; x_bits[token][3] = xl.w;
            float s = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                uint p = x_bits[token][i];
                s += as_type<float>(p << 16) + as_type<float>(p & 0xffff0000u);
            }
            x_sums[token] = s;
        }

        for (uint row = 0; row < RowsPerSimd; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize + lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device ushort* packed_src = reinterpret_cast<const device ushort*>(
                weight_bytes + current_row * (uint)PackedCols * 4 + (k * 6) / 8 + lane * BytesPerThread);
            uint h0 = packed_src[0];
            uint h1 = packed_src[1];
            uint h2 = packed_src[2];
            uint w0 = h0 | (h1 << 16);   // bits 0..31
            uint w1 = h2;                // bits 32..47
            // 8 contiguous little-endian 6-bit fields across 48 bits.
            for (uint token = 0; token < (uint)Leading; ++token) {
                uint p0 = x_bits[token][0], p1 = x_bits[token][1], p2 = x_bits[token][2], p3 = x_bits[token][3];
                float accum = 0.0f;
                accum = fma(as_type<float>(p0 << 16),          (float)(w0 & 0x3fu), accum);
                accum = fma(as_type<float>(p0 & 0xffff0000u), (float)((w0 >> 6) & 0x3fu), accum);
                accum = fma(as_type<float>(p1 << 16),          (float)((w0 >> 12) & 0x3fu), accum);
                accum = fma(as_type<float>(p1 & 0xffff0000u), (float)((w0 >> 18) & 0x3fu), accum);
                accum = fma(as_type<float>(p2 << 16),          (float)((w0 >> 24) & 0x3fu), accum);
                accum = fma(as_type<float>(p2 & 0xffff0000u), (float)((w0 >> 30) | ((w1 & 0x0fu) << 2)), accum);
                accum = fma(as_type<float>(p3 << 16),          (float)((w1 >> 4) & 0x3fu), accum);
                accum = fma(as_type<float>(p3 & 0xffff0000u), (float)((w1 >> 10) & 0x3fu), accum);
                result[token][row] = fma(scale, accum, fma(x_sums[token], bias, result[token][row]));
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] = static_cast<OutT>(total);
            }
        }
    }
"""
Q6_A = r"""
    constexpr uint ValuesPerThread = 16;
    constexpr uint BytesPerThread = 12;
    constexpr uint BlockSize = ValuesPerThread * 32;
    constexpr uint RowsPerSimd = 4;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * RowsPerSimd;

    float result[Leading][RowsPerSimd];
    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        uint x_bits[Leading][8];
        float x_sums[Leading];
        for (uint token = 0; token < (uint)Leading; ++token) {
            const device uint4* x_row = reinterpret_cast<const device uint4*>(
                x + token * (uint)InputDim + k + lane * ValuesPerThread);
            uint4 xl = x_row[0];
            uint4 xh = x_row[1];
            x_bits[token][0] = xl.x; x_bits[token][1] = xl.y;
            x_bits[token][2] = xl.z; x_bits[token][3] = xl.w;
            x_bits[token][4] = xh.x; x_bits[token][5] = xh.y;
            x_bits[token][6] = xh.z; x_bits[token][7] = xh.w;
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                uint p = x_bits[token][i];
                s += as_type<float>(p << 16) + as_type<float>(p & 0xffff0000u);
            }
            x_sums[token] = s;
        }

        for (uint row = 0; row < RowsPerSimd; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize + lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device uint* packed_src = reinterpret_cast<const device uint*>(
                weight_bytes + current_row * (uint)PackedCols * 4 + (k * 6) / 8 + lane * BytesPerThread);
            uint w0 = packed_src[0];
            uint w1 = packed_src[1];
            uint w2 = packed_src[2];
            // 16 contiguous little-endian 6-bit fields across w0..w2.
            float accum[Leading];
            for (uint token = 0; token < (uint)Leading; ++token) { accum[token] = 0.0f; }
            #define AXQ6(i, qv) { float qf = (float)(qv); for (uint token = 0; token < (uint)Leading; ++token) { \
                uint p = x_bits[token][(i) / 2]; float xv = ((i) & 1u) ? as_type<float>(p & 0xffff0000u) : as_type<float>(p << 16); \
                accum[token] = fma(xv, qf, accum[token]); } }
            AXQ6(0, w0 & 0x3fu); AXQ6(1, (w0 >> 6) & 0x3fu); AXQ6(2, (w0 >> 12) & 0x3fu); AXQ6(3, (w0 >> 18) & 0x3fu);
            AXQ6(4, (w0 >> 24) & 0x3fu); AXQ6(5, (w0 >> 30) | ((w1 & 0x0fu) << 2));
            AXQ6(6, (w1 >> 4) & 0x3fu); AXQ6(7, (w1 >> 10) & 0x3fu); AXQ6(8, (w1 >> 16) & 0x3fu); AXQ6(9, (w1 >> 22) & 0x3fu);
            AXQ6(10, (w1 >> 28) | ((w2 & 0x03u) << 4));
            AXQ6(11, (w2 >> 2) & 0x3fu); AXQ6(12, (w2 >> 8) & 0x3fu); AXQ6(13, (w2 >> 14) & 0x3fu); AXQ6(14, (w2 >> 20) & 0x3fu);
            AXQ6(15, w2 >> 26);
            #undef AXQ6
            for (uint token = 0; token < (uint)Leading; ++token) {
                result[token][row] = fma(scale, accum[token], fma(x_sums[token], bias, result[token][row]));
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] = static_cast<OutT>(total);
            }
        }
    }
"""
Q6 = r"""
    constexpr uint ValuesPerThread = 16;
    constexpr uint BytesPerThread = 12;
    constexpr uint BlockSize = ValuesPerThread * 32;
    constexpr uint RowsPerSimd = 4;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * RowsPerSimd;

    float result[Leading][RowsPerSimd];
    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        uint x_bits[Leading][8];
        float x_sums[Leading];
        for (uint token = 0; token < (uint)Leading; ++token) {
            const device uint4* x_row = reinterpret_cast<const device uint4*>(
                x + token * (uint)InputDim + k + lane * ValuesPerThread);
            uint4 xl = x_row[0];
            uint4 xh = x_row[1];
            x_bits[token][0] = xl.x; x_bits[token][1] = xl.y;
            x_bits[token][2] = xl.z; x_bits[token][3] = xl.w;
            x_bits[token][4] = xh.x; x_bits[token][5] = xh.y;
            x_bits[token][6] = xh.z; x_bits[token][7] = xh.w;
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                uint p = x_bits[token][i];
                s += as_type<float>(p << 16) + as_type<float>(p & 0xffff0000u);
            }
            x_sums[token] = s;
        }

        for (uint row = 0; row < RowsPerSimd; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize + lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device uint* packed_src = reinterpret_cast<const device uint*>(
                weight_bytes + current_row * (uint)PackedCols * 4 + (k * 6) / 8 + lane * BytesPerThread);
            uint w0 = packed_src[0];
            uint w1 = packed_src[1];
            uint w2 = packed_src[2];
            // 16 contiguous little-endian 6-bit fields across w0..w2.
            uint q[16];
            q[0] = w0 & 0x3fu;          q[1] = (w0 >> 6) & 0x3fu;   q[2] = (w0 >> 12) & 0x3fu;  q[3] = (w0 >> 18) & 0x3fu;
            q[4] = (w0 >> 24) & 0x3fu;  q[5] = (w0 >> 30) | ((w1 & 0x0fu) << 2);
            q[6] = (w1 >> 4) & 0x3fu;   q[7] = (w1 >> 10) & 0x3fu;  q[8] = (w1 >> 16) & 0x3fu;  q[9] = (w1 >> 22) & 0x3fu;
            q[10] = (w1 >> 28) | ((w2 & 0x03u) << 4);
            q[11] = (w2 >> 2) & 0x3fu;  q[12] = (w2 >> 8) & 0x3fu;  q[13] = (w2 >> 14) & 0x3fu; q[14] = (w2 >> 20) & 0x3fu;
            q[15] = w2 >> 26;
            for (uint token = 0; token < (uint)Leading; ++token) {
                float accum = 0.0f;
                for (uint i = 0; i < 8; ++i) {
                    uint p = x_bits[token][i];
                    accum = fma(as_type<float>(p << 16), (float)q[2 * i], accum);
                    accum = fma(as_type<float>(p & 0xffff0000u), (float)q[2 * i + 1], accum);
                }
                result[token][row] = fma(scale, accum, fma(x_sums[token], bias, result[token][row]));
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] = static_cast<OutT>(total);
            }
        }
    }
"""
Q6_ER = r"""
    constexpr uint ValuesPerThread = 8;
    constexpr uint BytesPerThread = 6;
    constexpr uint BlockSize = ValuesPerThread * 32;
    constexpr uint RowsPerSimd = Rows;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * (2 * RowsPerSimd) + simd_group * RowsPerSimd;

    float result[Leading][RowsPerSimd];
    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        uint x_bits[Leading][4];
        float x_sums[Leading];
        for (uint token = 0; token < (uint)Leading; ++token) {
            const device uint4* x_row = reinterpret_cast<const device uint4*>(
                x + token * (uint)InputDim + k + lane * ValuesPerThread);
            uint4 xl = x_row[0];
            x_bits[token][0] = xl.x; x_bits[token][1] = xl.y;
            x_bits[token][2] = xl.z; x_bits[token][3] = xl.w;
            float s = 0.0f;
            for (uint i = 0; i < 4; ++i) {
                uint p = x_bits[token][i];
                s += as_type<float>(p << 16) + as_type<float>(p & 0xffff0000u);
            }
            x_sums[token] = s;
        }

        for (uint row = 0; row < RowsPerSimd; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize + lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device ushort* packed_src = reinterpret_cast<const device ushort*>(
                weight_bytes + current_row * (uint)PackedCols * 4 + (k * 6) / 8 + lane * BytesPerThread);
            uint h0 = packed_src[0];
            uint h1 = packed_src[1];
            uint h2 = packed_src[2];
            uint w0 = h0 | (h1 << 16);   // bits 0..31
            uint w1 = h2;                // bits 32..47
            // 8 contiguous little-endian 6-bit fields across 48 bits.
            for (uint token = 0; token < (uint)Leading; ++token) {
                uint p0 = x_bits[token][0], p1 = x_bits[token][1], p2 = x_bits[token][2], p3 = x_bits[token][3];
                float accum = 0.0f;
                accum = fma(as_type<float>(p0 << 16),          (float)(w0 & 0x3fu), accum);
                accum = fma(as_type<float>(p0 & 0xffff0000u), (float)((w0 >> 6) & 0x3fu), accum);
                accum = fma(as_type<float>(p1 << 16),          (float)((w0 >> 12) & 0x3fu), accum);
                accum = fma(as_type<float>(p1 & 0xffff0000u), (float)((w0 >> 18) & 0x3fu), accum);
                accum = fma(as_type<float>(p2 << 16),          (float)((w0 >> 24) & 0x3fu), accum);
                accum = fma(as_type<float>(p2 & 0xffff0000u), (float)((w0 >> 30) | ((w1 & 0x0fu) << 2)), accum);
                accum = fma(as_type<float>(p3 << 16),          (float)((w1 >> 4) & 0x3fu), accum);
                accum = fma(as_type<float>(p3 & 0xffff0000u), (float)((w1 >> 10) & 0x3fu), accum);
                result[token][row] = fma(scale, accum, fma(x_sums[token], bias, result[token][row]));
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < RowsPerSimd; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] = static_cast<OutT>(total);
            }
        }
    }
"""
VARIANTS = {"E8R4": Q6_ER, "E8R8": Q6_ER}
ROWS = {"E8R4": 4, "E8R8": 8}
KS = {n: mx.fast.metal_kernel(name="ax_q6_" + n, input_names=["x", "weight", "scales", "biases"], output_names=["out"], source=s) for n, s in VARIANTS.items()}
def run(k_q6, x, wq, sc, bi, leading, K, N, gs, rows=4):
    return k_q6(inputs=[x, wq, sc, bi],
                template=[("OutT", mx.bfloat16), ("Leading", leading), ("OutDim", N), ("InputDim", K),
                          ("PackedCols", wq.shape[1]), ("GroupSize", gs), ("GroupCount", K // gs), ("Rows", rows)],
                grid=(32, (N // (2 * rows)) * 2, 1), threadgroup=(32, 2, 1), output_shapes=[(leading, N)], output_dtypes=[mx.bfloat16])[0]
COPIES, LAUNCHES = 8, 16
def bench(fn, iters=5):
    for _ in range(2): mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)])
    mx.synchronize(); ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(*[fn(i % COPIES) for i in range(LAUNCHES)]); mx.synchronize()
        ts.append((time.perf_counter() - t) * 1000 / LAUNCHES)
    return statistics.median(ts)
bits = 6
shapes = [("out_proj", 6144, 5120, 32), ("gate_proj", 5120, 17408, 64)] if len(sys.argv) > 2 and sys.argv[2] == "quick" else [("in_proj_qkv", 5120, 10240, 32), ("in_proj_z", 5120, 6144, 32), ("out_proj", 6144, 5120, 32), ("attn_o", 6144, 5120, 32), ("down_proj", 17408, 5120, 64), ("gate_proj", 5120, 17408, 64)]
leadings = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3, 4, 5, 6]
mx.random.seed(0)
for name, K, N, gs in shapes:
    ws = []
    for _ in range(COPIES):
        w = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(w, group_size=gs, bits=bits); mx.eval(wq, sc, bi); ws.append((wq, sc, bi))
    gb = sum(a.nbytes for a in ws[0]) / 1e9
    for leading in leadings:
        x = (mx.random.normal((leading, K)) * 0.5).astype(mx.bfloat16); mx.eval(x)
        ref = mx.quantized_matmul(x, *ws[0], transpose=True, group_size=gs, bits=bits)
        wd = mx.dequantize(ws[0][0], ws[0][1], ws[0][2], group_size=gs, bits=bits).astype(mx.float32)
        exact = x.astype(mx.float32) @ wd.T
        tm = bench(lambda i: mx.quantized_matmul(x, ws[i][0], ws[i][1], ws[i][2], transpose=True, group_size=gs, bits=bits))
        line = f"{name:12s} K={K:5d} N={N:5d} gs={gs} M={leading}  mlx {tm:6.3f} ms ({gb/tm*1000:4.0f} GB/s)"
        for vn, kk in KS.items():
            o = run(kk, x, *ws[0], leading, K, N, gs, ROWS[vn]); mx.eval(o)
            md = float(mx.max(mx.abs(o.astype(mx.float32) - exact)))
            tq = bench(lambda i: run(kk, x, *ws[i], leading, K, N, gs, ROWS[vn]))
            line += f"  {vn} {tq:6.3f} ms ({gb/tq*1000:4.0f} GB/s) x{tm/tq:.3f} err={md:.4f}"
        print(line, flush=True)
