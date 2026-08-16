use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxDtype, MlxMetalKernel, add, astype,
    async_eval, concatenate, contiguous, dequantize_with_mode, eval, expand_dims_axes, gather_mm,
    matmul, multiply, reshape, slice, slice_last_dim, take, tanh, transpose,
};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::super::config::ModelConfig;
use crate::fastpath;
use crate::weights::{DECODE_LM_HEAD_QUANT_BITS, DECODE_LM_HEAD_QUANT_GROUP_SIZE, QuantizedWeight};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct QkvSlices {
    pub q: (i32, i32),
    pub gate: Option<(i32, i32)>,
    pub k: (i32, i32),
    pub v: (i32, i32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProjectionBatchPolicy {
    Shared,
    /// Preserve the single-row reduction graph for every decode row.
    RowExact,
}

thread_local! {
    static QWEN_PREFILL_DEQUANT_DENSE_FAMILY: Cell<bool> = const { Cell::new(false) };
    static PREFILL_DEQUANT_DENSE_T: RefCell<HashMap<usize, MlxArray>> =
        RefCell::new(HashMap::new());
    static QWEN_PREFILL_SKIP_EMBED_CLIP: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_SKIP_F32_SDPA: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_BF16_EMBED_DEQUANT: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_NATIVE_OFFSET_CAUSAL: Cell<bool> = const { Cell::new(false) };
    static QWEN_PREFILL_SKIP_SWIGLU_COMPILE: Cell<bool> = const { Cell::new(false) };
    static GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED: Cell<bool> = const { Cell::new(false) };
}

/// Mark the current Qwen3.5 / Qwen3-Next layer so `qw` may dequant+dense.
pub(crate) fn set_qwen_prefill_dequant_dense_family(active: bool) {
    QWEN_PREFILL_DEQUANT_DENSE_FAMILY.set(active);
}

/// Enable skipping the unused embed-id `clip` for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_embed_clip(active: bool) {
    QWEN_PREFILL_SKIP_EMBED_CLIP.set(active);
}

/// Whether [`set_qwen_prefill_skip_embed_clip`] is set for this forward.
pub(crate) fn qwen_prefill_skip_embed_clip_active() -> bool {
    QWEN_PREFILL_SKIP_EMBED_CLIP.get()
}

/// Enable skipping the unused f32 SDPA upcast for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_f32_sdpa(active: bool) {
    QWEN_PREFILL_SKIP_F32_SDPA.set(active);
}

/// Whether [`set_qwen_prefill_skip_f32_sdpa`] is set for this forward.
pub(crate) fn qwen_prefill_skip_f32_sdpa_active() -> bool {
    QWEN_PREFILL_SKIP_F32_SDPA.get()
}

/// Arms [`set_qwen_prefill_skip_f32_sdpa`] for the rest of this forward.
pub(crate) struct QwenPrefillSkipF32SdpaGuard;

impl QwenPrefillSkipF32SdpaGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_skip_f32_sdpa(active);
        Self
    }
}

impl Drop for QwenPrefillSkipF32SdpaGuard {
    fn drop(&mut self) {
        set_qwen_prefill_skip_f32_sdpa(false);
    }
}

/// Enable BF16 embedding dequant for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_bf16_embed_dequant(active: bool) {
    QWEN_PREFILL_BF16_EMBED_DEQUANT.set(active);
}

/// Whether [`set_qwen_prefill_bf16_embed_dequant`] is set for this forward.
pub(crate) fn qwen_prefill_bf16_embed_dequant_active() -> bool {
    QWEN_PREFILL_BF16_EMBED_DEQUANT.get()
}

/// Enable native offset-causal SDPA for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_native_offset_causal(active: bool) {
    QWEN_PREFILL_NATIVE_OFFSET_CAUSAL.set(active);
}

/// Whether [`set_qwen_prefill_native_offset_causal`] is set for this forward.
pub(crate) fn qwen_prefill_native_offset_causal_active() -> bool {
    QWEN_PREFILL_NATIVE_OFFSET_CAUSAL.get()
}

/// Arms [`set_qwen_prefill_native_offset_causal`] for the rest of this forward.
pub(crate) struct QwenPrefillNativeOffsetCausalGuard;

impl QwenPrefillNativeOffsetCausalGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_native_offset_causal(active);
        Self
    }
}

impl Drop for QwenPrefillNativeOffsetCausalGuard {
    fn drop(&mut self) {
        set_qwen_prefill_native_offset_causal(false);
    }
}

/// Enable skipping unused SwiGLU compile for this Qwen prefill forward.
pub(crate) fn set_qwen_prefill_skip_swiglu_compile(active: bool) {
    QWEN_PREFILL_SKIP_SWIGLU_COMPILE.set(active);
}

/// Whether [`set_qwen_prefill_skip_swiglu_compile`] is set for this forward.
pub(crate) fn qwen_prefill_skip_swiglu_compile_active() -> bool {
    QWEN_PREFILL_SKIP_SWIGLU_COMPILE.get()
}

/// Arms [`set_qwen_prefill_skip_swiglu_compile`] for the rest of this forward.
pub(crate) struct QwenPrefillSkipSwigluCompileGuard;

impl QwenPrefillSkipSwigluCompileGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_qwen_prefill_skip_swiglu_compile(active);
        Self
    }
}

impl Drop for QwenPrefillSkipSwigluCompileGuard {
    fn drop(&mut self) {
        set_qwen_prefill_skip_swiglu_compile(false);
    }
}

pub(crate) fn set_gemma4_prefill_skip_last_ffn_packed(active: bool) {
    GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED.set(active);
}

pub(crate) fn gemma4_prefill_skip_last_ffn_packed_active() -> bool {
    GEMMA4_PREFILL_SKIP_LAST_FFN_PACKED.get()
}

pub(crate) struct Gemma4PrefillSkipLastFfnPackedGuard;

impl Gemma4PrefillSkipLastFfnPackedGuard {
    pub(crate) fn arm(active: bool) -> Self {
        set_gemma4_prefill_skip_last_ffn_packed(active);
        Self
    }
}

impl Drop for Gemma4PrefillSkipLastFfnPackedGuard {
    fn drop(&mut self) {
        set_gemma4_prefill_skip_last_ffn_packed(false);
    }
}

/// Skip a no-op BF16 astype when the gather is already BF16.
pub(crate) fn qwen_prefill_maybe_skip_bf16_astype(
    x: &MlxArray,
    model_family: &str,
    seq: i32,
) -> MlxArray {
    if (fastpath::should_qwen_prefill_skip_bf16_astype(model_family, seq)
        || fastpath::should_gemma4_prefill_bf16_embed(model_family, seq))
        && x.dtype() == MlxDtype::Bfloat16
    {
        x.clone()
    } else {
        astype(x, MlxDtype::Bfloat16, None)
    }
}

/// Submit the embedding gather so GPU starts while the first layer is built.
pub(crate) fn qwen_prefill_maybe_async_embed(hidden: &MlxArray, model_family: &str, seq: i32) {
    qwen_prefill_maybe_async_embed_for(
        hidden,
        fastpath::qwen_prefill_async_embed_enabled(),
        model_family,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_embed`].
pub(crate) fn qwen_prefill_maybe_async_embed_for(
    hidden: &MlxArray,
    enabled: bool,
    model_family: &str,
    seq: i32,
) {
    if fastpath::should_qwen_prefill_async_embed_for(enabled, model_family, seq) {
        async_eval(&[hidden]);
    }
}

fn activation_seq_len(x: &MlxArray) -> i32 {
    let shape = x.shape();
    match shape.len() {
        0 => 0,
        1 => shape[0],
        _ => shape[shape.len() - 2],
    }
}

fn qwen_prefill_dequant_dense_applies(x: &MlxArray) -> bool {
    QWEN_PREFILL_DEQUANT_DENSE_FAMILY.get()
        && fastpath::should_qwen_prefill_dequant_dense_for(
            fastpath::qwen_prefill_dequant_dense_enabled(),
            "qwen3_5",
            activation_seq_len(x),
        )
}

fn cached_prefill_dequant_weight_t(qw: &QuantizedWeight) -> Option<MlxArray> {
    let scales = qw.scales.as_ref()?;
    let key = qw as *const QuantizedWeight as usize;
    PREFILL_DEQUANT_DENSE_T.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        let dense = dequantize_with_mode(
            &qw.weight,
            scales,
            quant_biases,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            None,
            Some(MlxDtype::Bfloat16),
            None,
        );
        let weight_t = transpose(&dense, &[1, 0], None);
        eval(&[&weight_t]);
        cache.borrow_mut().insert(key, weight_t.clone());
        Some(weight_t)
    })
}

static INVARIANT_AFFINE_QMV_FAST_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static INVARIANT_DENSE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Microbatch form of MLX 0.32's affine `qmv_fast` reduction.
///
/// Each simdgroup still computes four output rows with the exact singleton
/// lane assignment, per-block accumulation, and `simd_sum` reduction. The only
/// extension is a fixed array of up to four independent input rows. Packed
/// weight bytes, scale, and bias are loaded once, then reused for every row in
/// the microbatch. Thus `Leading=2..4` amortizes weight traffic without changing
/// the arithmetic graph observed by any individual row.
const INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE: &str = r#"
    constexpr uint PacksPerThread = 2;
    constexpr uint QmvPackFactor = Bits == 6 ? 4 : 32 / Bits;
    constexpr uint BytesPerPack = Bits == 6 ? 3 : 4;
    constexpr uint ValuesPerThread = QmvPackFactor * PacksPerThread;
    constexpr uint BytesPerThread = BytesPerPack * PacksPerThread;
    constexpr uint BlockSize = ValuesPerThread * 32;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint out_row = threadgroup_position_in_grid.y * 8 + simd_group * 4;

    float result[4][4];
    for (uint token = 0; token < 4; ++token) {
        for (uint row = 0; row < 4; ++row) {
            result[token][row] = 0.0f;
        }
    }

    const device uchar* weight_bytes = reinterpret_cast<const device uchar*>(weight);
    for (uint k = 0; k < (uint)InputDim; k += BlockSize) {
        float x_values[4][16];
        float x_sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (uint token = 0; token < (uint)Leading; ++token) {
            const device InputT* x_row =
                x + token * (uint)InputDim + k + lane * ValuesPerThread;
            if (Bits == 4) {
                for (uint i = 0; i < ValuesPerThread; i += 4) {
                    OutT group_sum =
                        static_cast<OutT>(x_row[i]) +
                        static_cast<OutT>(x_row[i + 1]) +
                        static_cast<OutT>(x_row[i + 2]) +
                        static_cast<OutT>(x_row[i + 3]);
                    x_sums[token] += static_cast<float>(group_sum);
                    float x0 = static_cast<float>(x_row[i]);
                    float x1 = static_cast<float>(x_row[i + 1]);
                    float x2 = static_cast<float>(x_row[i + 2]);
                    float x3 = static_cast<float>(x_row[i + 3]);
                    x_values[token][i] = x0;
                    x_values[token][i + 1] = x1 / 16.0f;
                    x_values[token][i + 2] = x2 / 256.0f;
                    x_values[token][i + 3] = x3 / 4096.0f;
                }
            } else if (Bits == 6) {
                for (uint i = 0; i < ValuesPerThread; i += 4) {
                    OutT group_sum =
                        static_cast<OutT>(x_row[i]) +
                        static_cast<OutT>(x_row[i + 1]) +
                        static_cast<OutT>(x_row[i + 2]) +
                        static_cast<OutT>(x_row[i + 3]);
                    x_sums[token] += static_cast<float>(group_sum);
                    float x0 = static_cast<float>(x_row[i]);
                    float x1 = static_cast<float>(x_row[i + 1]);
                    float x2 = static_cast<float>(x_row[i + 2]);
                    float x3 = static_cast<float>(x_row[i + 3]);
                    x_values[token][i] = x0;
                    x_values[token][i + 1] = x1 / 64.0f;
                    x_values[token][i + 2] = x2 / 16.0f;
                    x_values[token][i + 3] = x3 / 4.0f;
                }
            } else {
                for (uint i = 0; i < ValuesPerThread; ++i) {
                    float value = static_cast<float>(x_row[i]);
                    x_sums[token] += value;
                    x_values[token][i] = value;
                }
            }
        }

        for (uint row = 0; row < 4; ++row) {
            uint current_row = out_row + row;
            uint group = k / (uint)GroupSize +
                lane / ((uint)GroupSize / ValuesPerThread);
            uint sidecar_index = current_row * (uint)GroupCount + group;
            float scale = static_cast<float>(scales[sidecar_index]);
            float bias = static_cast<float>(biases[sidecar_index]);
            const device uchar* packed_src =
                weight_bytes + current_row * (uint)PackedCols * 4 +
                k * (uint)Bits / 8 + lane * BytesPerThread;
            uchar packed[8];
            for (uint byte = 0; byte < BytesPerThread; ++byte) {
                packed[byte] = packed_src[byte];
            }

            for (uint token = 0; token < (uint)Leading; ++token) {
                float accum = 0.0f;
                if (Bits == 4) {
                    for (uint pack = 0; pack < PacksPerThread; ++pack) {
                        uint byte = pack * 4;
                        ushort packed16_0 =
                            static_cast<ushort>(packed[byte]) |
                            (static_cast<ushort>(packed[byte + 1]) << 8);
                        ushort packed16_1 =
                            static_cast<ushort>(packed[byte + 2]) |
                            (static_cast<ushort>(packed[byte + 3]) << 8);
                        uint value = pack * 8;
                        accum +=
                            (x_values[token][value] * (packed16_0 & 0x000f) +
                             x_values[token][value + 1] * (packed16_0 & 0x00f0) +
                             x_values[token][value + 2] * (packed16_0 & 0x0f00) +
                             x_values[token][value + 3] * (packed16_0 & 0xf000));
                        accum +=
                            (x_values[token][value + 4] * (packed16_1 & 0x000f) +
                             x_values[token][value + 5] * (packed16_1 & 0x00f0) +
                             x_values[token][value + 6] * (packed16_1 & 0x0f00) +
                             x_values[token][value + 7] * (packed16_1 & 0xf000));
                    }
                } else if (Bits == 6) {
                    for (uint pack = 0; pack < PacksPerThread; ++pack) {
                        uint byte = pack * 3;
                        uint value = pack * 4;
                        accum += (packed[byte] & 0x3f) * x_values[token][value];
                        accum += (packed[byte] & 0xc0) * x_values[token][value + 1];
                        accum +=
                            (packed[byte + 1] & 0x0f) *
                            (x_values[token][value + 1] * 256.0f);
                        accum += (packed[byte + 1] & 0xf0) *
                            x_values[token][value + 2];
                        accum +=
                            (packed[byte + 2] & 0x03) *
                            (x_values[token][value + 2] * 256.0f);
                        accum += (packed[byte + 2] & 0xfc) *
                            x_values[token][value + 3];
                    }
                } else {
                    for (uint value = 0; value < ValuesPerThread; ++value) {
                        accum += x_values[token][value] * packed[value];
                    }
                }
                result[token][row] +=
                    scale * accum + x_sums[token] * bias;
            }
        }
    }

    for (uint token = 0; token < (uint)Leading; ++token) {
        for (uint row = 0; row < 4; ++row) {
            float total = simd_sum(result[token][row]);
            if (lane == 0) {
                out[token * (uint)OutDim + out_row + row] =
                    static_cast<OutT>(total);
            }
        }
    }
"#;

/// Dense (non-quantized) microbatch projection with invariant per-row reduction.
const INVARIANT_DENSE_PROJECTION_KERNEL_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256;
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (row >= OutDim) {
        return;
    }

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint row_base = row * InputDim;
    for (uint input_col = tid; input_col < InputDim; input_col += 256) {
        float w = static_cast<float>(weight[row_base + input_col]);
        for (uint token = 0; token < (uint)Leading; ++token) {
            float x_v = static_cast<float>(x[token * InputDim + input_col]);
            acc[token] = fma(x_v, w, acc[token]);
        }
    }

    threadgroup float partials[32]; // four rows × eight simdgroups
    for (uint token = 0; token < (uint)Leading; ++token) {
        float sum = simd_sum(acc[token]);
        if (lane == 0) {
            partials[token * 8 + sg] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint token = 0; token < (uint)Leading; ++token) {
            float total = 0.0f;
            for (uint group = 0; group < 8; ++group) {
                total += partials[token * 8 + group];
            }
            out[token * OutDim + row] = static_cast<OutT>(total);
        }
    }
"#;

pub(crate) fn qkv_slices(cfg: &ModelConfig, head_dim: usize, kv_head_count: usize) -> QkvSlices {
    let q_size = (cfg.n_heads * head_dim) as i32;
    let kv_size = (kv_head_count * head_dim) as i32;
    let gate = cfg.attn_output_gate.then_some((q_size, q_size * 2));
    let kv_start = if cfg.attn_output_gate {
        q_size * 2
    } else {
        q_size
    };
    QkvSlices {
        q: (0, q_size),
        gate,
        k: (kv_start, kv_start + kv_size),
        v: (kv_start + kv_size, kv_start + kv_size * 2),
    }
}

/// Infer the KV head count encoded in one packed QKV projection.
///
/// This must use the projection's actual row count rather than the model's
/// base KV geometry: Gemma 4 global-attention layers can use both a wider head
/// dimension and a different KV head count than their sliding layers.
pub(crate) fn packed_qkv_kv_head_count(
    cfg: &ModelConfig,
    head_dim: usize,
    packed_rows: usize,
) -> Option<usize> {
    let q_rows = cfg.n_heads.checked_mul(head_dim)?;
    let packed_q_rows = if cfg.attn_output_gate {
        q_rows.checked_mul(2)?
    } else {
        q_rows
    };
    let remaining = packed_rows.checked_sub(packed_q_rows)?;
    if !remaining.is_multiple_of(2) {
        return None;
    }
    let kv_rows = remaining / 2;
    if head_dim == 0 || !kv_rows.is_multiple_of(head_dim) {
        return None;
    }
    let kv_head_count = kv_rows / head_dim;
    (kv_head_count > 0).then_some(kv_head_count)
}

/// Flatten `[B,S,H]` to `[B*S,H]` for steel qmm when the flat-qmm flag is on.
fn qwen_prefill_maybe_flat_qmm(x: &MlxArray, qmm: impl FnOnce(&MlxArray) -> MlxArray) -> MlxArray {
    qwen_prefill_maybe_flat_qmm_for(x, fastpath::qwen_prefill_flat_qmm_enabled(), qmm)
}

/// Pure helper for [`qwen_prefill_maybe_flat_qmm`].
pub(crate) fn qwen_prefill_maybe_flat_qmm_for(
    x: &MlxArray,
    enabled: bool,
    qmm: impl FnOnce(&MlxArray) -> MlxArray,
) -> MlxArray {
    let shape = x.shape();
    let seq = match shape.len() {
        3 => shape[1],
        _ => 0,
    };
    if !fastpath::should_qwen_prefill_flat_qmm_for(enabled, seq, shape.len()) {
        return qmm(x);
    }
    let batch = shape[0];
    let hidden = shape[2];
    let flat = reshape(x, &[batch * seq, hidden], None);
    let out = qmm(&flat);
    let out_last = *out.shape().last().unwrap_or(&hidden);
    reshape(&out, &[batch, seq, out_last], None)
}

/// Tile `[B,S,H]` into 512-token slices for steel qmm, then concatenate.
fn qwen_prefill_maybe_tile_qmm(x: &MlxArray, qmm: impl Fn(&MlxArray) -> MlxArray) -> MlxArray {
    qwen_prefill_maybe_tile_qmm_for(x, fastpath::qwen_prefill_tile_qmm_enabled(), "qwen3_5", qmm)
}

/// Pure helper for [`qwen_prefill_maybe_tile_qmm`].
pub(crate) fn qwen_prefill_maybe_tile_qmm_for(
    x: &MlxArray,
    enabled: bool,
    model_family: &str,
    qmm: impl Fn(&MlxArray) -> MlxArray,
) -> MlxArray {
    let shape = x.shape();
    if shape.len() != 3 {
        return qmm(x);
    }
    let seq = shape[1];
    if !fastpath::should_qwen_prefill_tile_qmm_for(enabled, model_family, seq) {
        return qmm(x);
    }
    let tile = fastpath::QWEN_PREFILL_QMM_TILE;
    if seq <= tile {
        return qmm(x);
    }
    let batch = shape[0];
    let hidden = shape[2];
    let mut parts: Vec<MlxArray> = Vec::new();
    let mut start = 0i32;
    while start < seq {
        let end = (start + tile).min(seq);
        let chunk = contiguous(
            &slice(x, &[0, start, 0], &[batch, end, hidden], &[1, 1, 1], None),
            None,
        );
        parts.push(qmm(&chunk));
        start = end;
    }
    let refs: Vec<&MlxArray> = parts.iter().collect();
    concatenate(&refs, 1, None)
}

pub(crate) fn qw(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    qw_with_policy(x, qw, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qw_with_policy(
    x: &MlxArray,
    qw: &QuantizedWeight,
    policy: ProjectionBatchPolicy,
) -> MlxArray {
    // Under an invariant projection scope, the S=1 baseline uses this same
    // kernel through Shared and S>1 RowExact must use it too. The scope owner
    // is responsible for applying the arithmetic contract symmetrically.
    if policy == ProjectionBatchPolicy::RowExact
        && fastpath::qwen_linear_mtp_exact_enabled()
        && let Some(invariant) = invariant_projection_metal_impl(x, qw)
    {
        return invariant;
    }
    // Outside an invariant scope, RowExact stays on per-row MLX so it matches
    // an ordinary pure-direct singleton.
    if policy == ProjectionBatchPolicy::RowExact
        && let Some(row_exact) = qw_row_exact_mlx(x, qw)
    {
        return row_exact;
    }
    // Shared (default): invariant when exact profile scopes it. MXFP4
    // quantized_matmul is already singleton-exact at S=2, so a batched
    // qmm is safe here (and much faster than a per-row loop).
    qw_direct(x, qw)
}

/// Per-row MLX projection so S>1 / B>1 matches the corresponding singleton.
fn qw_row_exact_mlx(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    let shape = x.shape();
    if shape.len() != 3 {
        return None;
    }
    // Batch-decode: B>1, S=1 — one projection per batch row.
    if shape[0] > 1 && shape[1] == 1 {
        let rows: Vec<MlxArray> = (0..shape[0])
            .map(|row| {
                let row = slice(x, &[row, 0, 0], &[row + 1, 1, shape[2]], &[1, 1, 1], None);
                qw_direct_mlx(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = rows.iter().collect();
        return Some(concatenate(&refs, 0, None));
    }
    // Multi-token teacher-forced / MTP verify: B=1, S>1.
    if shape[0] == 1 && shape[1] > 1 {
        let cols: Vec<MlxArray> = (0..shape[1])
            .map(|t| {
                let row = slice(x, &[0, t, 0], &[1, t + 1, shape[2]], &[1, 1, 1], None);
                qw_direct_mlx(&contiguous(&row, None), qw)
            })
            .collect();
        let refs: Vec<&MlxArray> = cols.iter().collect();
        return Some(concatenate(&refs, 1, None));
    }
    None
}

/// Runtime 2-bit `lm_head` is a decode GEMV. Prefill (S>1) stays on the
/// BF16 `W_t` GEMM — q4 qmm was a wash at p2048 and 2-bit is worse there.
fn decode_lm_head_quant_cache_eligible(x: &MlxArray) -> bool {
    let shape = x.shape();
    if shape.len() < 2 {
        return shape.first().copied() == Some(1);
    }
    shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))
        == Some(1)
}

fn qw_direct_mlx(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Always MLX quantized_matmul / dense matmul (no invariant). Used by
    // RowExact so multi-token rows match pure-direct MLX singletons.
    let y = if qwen_prefill_dequant_dense_applies(x)
        && let Some(weight_t) = cached_prefill_dequant_weight_t(qw)
    {
        matmul(x, &weight_t, None)
    } else if let Some(scales) = &qw.scales {
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        mlx_sys::quantized_matmul_with_mode(
            x,
            &qw.weight,
            scales,
            quant_biases,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            None,
        )
    } else if decode_lm_head_quant_cache_eligible(x)
        && let (Some(q_w), Some(q_s), Some(q_b)) = (
            qw.decode_q4_weight.as_ref(),
            qw.decode_q4_scales.as_ref(),
            qw.decode_q4_biases.as_ref(),
        )
    {
        mlx_sys::quantized_matmul_with_mode(
            x,
            q_w,
            q_s,
            Some(q_b),
            true,
            Some(DECODE_LM_HEAD_QUANT_GROUP_SIZE),
            Some(DECODE_LM_HEAD_QUANT_BITS),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
        )
    } else if let Some(weight_t) = &qw.decode_weight_t {
        matmul(x, weight_t, None)
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    };
    if let Some(bias) = &qw.linear_bias {
        add(&y, bias, None)
    } else {
        y
    }
}

fn qw_direct(x: &MlxArray, qw: &QuantizedWeight) -> MlxArray {
    // Dense Linear bias (`QuantizedWeight.linear_bias`) is separate from affine
    // group-quant biases (`qw.biases`). Matches mlx-lm `nn.Linear` /
    // `QuantizedLinear`: y = x @ W + b. Required for GPT-OSS Q/K/V/O + router.
    let y = if fastpath::qwen_linear_mtp_exact_enabled()
        && let Some(invariant) = invariant_projection_metal_impl(x, qw)
    {
        invariant
    } else if qwen_prefill_dequant_dense_applies(x)
        && let Some(weight_t) = cached_prefill_dequant_weight_t(qw)
    {
        matmul(x, &weight_t, None)
    } else if let Some(scales) = &qw.scales {
        // MXFP8/MXFP4 have no affine group-bias channel; pass None for those modes.
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        qwen_prefill_maybe_tile_qmm(x, |tiled| {
            qwen_prefill_maybe_flat_qmm(tiled, |flat| {
                mlx_sys::quantized_matmul_with_mode(
                    flat,
                    &qw.weight,
                    scales,
                    quant_biases,
                    true,
                    Some(qw.group_size),
                    Some(qw.bits),
                    mode,
                    None,
                )
            })
        })
    } else if decode_lm_head_quant_cache_eligible(x)
        && let (Some(q_w), Some(q_s), Some(q_b)) = (
            qw.decode_q4_weight.as_ref(),
            qw.decode_q4_scales.as_ref(),
            qw.decode_q4_biases.as_ref(),
        )
    {
        mlx_sys::quantized_matmul_with_mode(
            x,
            q_w,
            q_s,
            Some(q_b),
            true,
            Some(DECODE_LM_HEAD_QUANT_GROUP_SIZE),
            Some(DECODE_LM_HEAD_QUANT_BITS),
            mlx_sys::MlxQuantizationMode::Affine,
            None,
        )
    } else if let Some(weight_t) = &qw.decode_weight_t {
        matmul(x, weight_t, None)
    } else {
        let wt = transpose(&qw.weight, &[1, 0], None);
        matmul(x, &wt, None)
    };
    if let Some(bias) = &qw.linear_bias {
        add(&y, bias, None)
    } else {
        y
    }
}

static UNQUANTIZED_DECODE_PROJECTION_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static UNQUANTIZED_DECODE_PROJECTION_HITS: AtomicU64 = AtomicU64::new(0);

/// Metal GEMV for a single-token dense projection. Reads `weight` as
/// `[out, in]` in place — no `[out, in] → [in, out]` transpose buffer.
/// Eight output rows per threadgroup via simdgroup_matrix (phase1 sg_bf16).
const UNQUANTIZED_DECODE_PROJECTION_SOURCE: &str = r#"
    constexpr uint Tile = 8;
    uint tg = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    uint row_base = tg * Tile;
    if (row_base >= (uint)OutDim) {
        return;
    }
    const uint N = (uint)InputDim;
    if (row_base + Tile > (uint)OutDim) {
        if (lane < Tile) {
            uint row = row_base + lane;
            if (row < (uint)OutDim) {
                float val = 0.0f;
                uint row_kbase = row * N;
                for (uint c = 0; c < N; ++c) {
                    val = fma(static_cast<float>(weight[row_kbase + c]),
                              static_cast<float>(x[c]), val);
                }
                out[row] = static_cast<OutT>(val);
            }
        }
        return;
    }
    simdgroup_matrix<float, 8, 8> acc;
    acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint k = 0;
    for (; k + Tile <= N; k += Tile) {
        simdgroup_matrix<OutT, 8, 8> w_tile;
        simdgroup_matrix<float, 8, 8> h_tile;
        simdgroup_load(w_tile, weight + row_base * N + k, (ulong)N, ulong2(0, 0), false);
        simdgroup_load(h_tile, x + k, 0, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, w_tile, h_tile, acc);
    }
    threadgroup float out_buf[Tile * Tile];
    simdgroup_store(acc, out_buf, Tile, ulong2(0, 0));
    if (lane < Tile) {
        uint row = row_base + lane;
        float val = out_buf[lane * Tile];
        uint row_kbase = row * N;
        for (uint c = k; c < N; ++c) {
            val = fma(static_cast<float>(weight[row_kbase + c]),
                      static_cast<float>(x[c]), val);
        }
        out[row] = static_cast<OutT>(val);
    }
"#;

/// Decode-only unquantized projection: `logits = x @ weight.T` without
/// materializing `weight.T`. `weight` is `[out, in]`, `x` is rank ≥ 1
/// with last dim `in` and all other dims product 1.
///
/// Wired decode remasure on AXQ 27B (df-macbookpro-m5): 29.47 vs 28.78
/// (1.024×), slower than `qw` 30.20. Production stays on `qw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn project_unquantized_decode(x: &MlxArray, weight: &MlxArray) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || x.dtype() != weight.dtype()
    {
        return None;
    }
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_dim = weight_shape[0];
    let input_dim = weight_shape[1];
    if out_dim <= 0 || input_dim <= 0 {
        return None;
    }
    let x_shape = x.shape();
    if x_shape.last().copied() != Some(input_dim) {
        return None;
    }
    let leading = x_shape[..x_shape.len() - 1]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))?;
    if leading != 1 {
        return None;
    }
    let tiles = out_dim.saturating_add(7) / 8;
    let grid_x = tiles.checked_mul(32)?;
    let x_flat = reshape(x, &[input_dim], None);
    // simdgroup_load into a float tile requires a float source (phase1
    // decode_logits_projection_sg_*). Hidden is 5k elements — not a
    // 2.54 GB weight transpose.
    let x_f32 = if x.dtype() == MlxDtype::Float32 {
        x_flat
    } else {
        astype(&x_flat, MlxDtype::Float32, None)
    };
    let kernel = UNQUANTIZED_DECODE_PROJECTION_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_unquantized_decode_projection_sg_v2",
            &["x", "weight"],
            &["out"],
            UNQUANTIZED_DECODE_PROJECTION_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[&x_f32, weight],
            &[KernelOutputSpec {
                shape: vec![out_dim],
                dtype: x.dtype(),
            }],
            &[
                KernelTemplateArg::Dtype {
                    name: "OutT",
                    dtype: x.dtype(),
                },
                KernelTemplateArg::Int {
                    name: "OutDim",
                    value: out_dim,
                },
                KernelTemplateArg::Int {
                    name: "InputDim",
                    value: input_dim,
                },
            ],
            (grid_x, 1, 1),
            (32, 1, 1),
            None,
        )
        .ok()?;
    let flat = outputs.pop()?;
    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    Some(reshape(&flat, &out_shape, None))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn unquantized_decode_projection_hits() -> u64 {
    UNQUANTIZED_DECODE_PROJECTION_HITS.load(Ordering::Relaxed)
}

/// Unquantized decode `lm_head` entry: no-copy GEMV, else `qw`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn project_lm_head(x: &MlxArray, lm_head: &QuantizedWeight) -> MlxArray {
    if lm_head.scales.is_none()
        && let Some(y) = project_unquantized_decode(x, &lm_head.weight)
    {
        UNQUANTIZED_DECODE_PROJECTION_HITS.fetch_add(1, Ordering::Relaxed);
        return if let Some(bias) = &lm_head.linear_bias {
            add(&y, bias, None)
        } else {
            y
        };
    }
    qw(x, lm_head)
}

/// Slice the last axis of `x` to `[start, end)`.
fn slice_trailing_cols(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    let shape = x.shape();
    let ndim = shape.len();
    let mut starts = vec![0_i32; ndim];
    let mut ends: Vec<i32> = shape.to_vec();
    starts[ndim - 1] = start;
    ends[ndim - 1] = end;
    let strides = vec![1_i32; ndim];
    contiguous(&slice(x, &starts, &ends, &strides, None), None)
}

fn invariant_projection_metal_impl(x: &MlxArray, qw: &QuantizedWeight) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let x_shape = x.shape();
    let input_dim = *x_shape.last()?;
    if input_dim <= 0 || x_shape.len() < 2 {
        return None;
    }
    let leading = x_shape[..x_shape.len() - 1]
        .iter()
        .try_fold(1_i32, |product, dimension| product.checked_mul(*dimension))?;
    if !(1..=4).contains(&leading) {
        return None;
    }

    let weight_shape = qw.weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_dim = weight_shape[0];
    if out_dim <= 0 {
        return None;
    }

    // Split dimensions that are not aligned to one complete qmv_fast lane
    // block: 512 values for 4-bit, 256 for 6/8-bit. Both pure-direct and
    // multi-token use this split so A/B identity holds.
    let qmv_block_size = match qw.bits {
        4 => 512,
        6 | 8 => 256,
        _ => 512,
    };
    if matches!(
        qw.mlx_quantization_mode(),
        mlx_sys::MlxQuantizationMode::Affine
    ) && qw.bits > 0
        && qw.bits <= 8
        && qw.group_size > 0
        && input_dim % qmv_block_size != 0
        && input_dim > qmv_block_size
        && input_dim % qw.group_size == 0
        && (input_dim * qw.bits) % 32 == 0
    {
        let aligned = (input_dim / qmv_block_size) * qmv_block_size;
        let rem = input_dim - aligned;
        if aligned > 0 && rem > 0 && rem % qw.group_size == 0 && (rem * qw.bits) % 32 == 0 {
            let packed_al = aligned * qw.bits / 32;
            let packed_rem = rem * qw.bits / 32;
            let groups_al = aligned / qw.group_size;
            let groups_rem = rem / qw.group_size;
            if weight_shape[1] == packed_al + packed_rem
                && let (Some(scales), Some(biases)) = (qw.scales.as_ref(), qw.biases.as_ref())
                && scales.shape() == [out_dim, input_dim / qw.group_size]
                && biases.shape() == scales.shape()
            {
                let x_al = slice_trailing_cols(x, 0, aligned);
                let x_rem = slice_trailing_cols(x, aligned, input_dim);
                let w_al = contiguous(
                    &slice(&qw.weight, &[0, 0], &[out_dim, packed_al], &[1, 1], None),
                    None,
                );
                let w_rem = contiguous(
                    &slice(
                        &qw.weight,
                        &[0, packed_al],
                        &[out_dim, packed_al + packed_rem],
                        &[1, 1],
                        None,
                    ),
                    None,
                );
                let s_al = contiguous(
                    &slice(scales, &[0, 0], &[out_dim, groups_al], &[1, 1], None),
                    None,
                );
                let s_rem = contiguous(
                    &slice(
                        scales,
                        &[0, groups_al],
                        &[out_dim, groups_al + groups_rem],
                        &[1, 1],
                        None,
                    ),
                    None,
                );
                let b_al = contiguous(
                    &slice(biases, &[0, 0], &[out_dim, groups_al], &[1, 1], None),
                    None,
                );
                let b_rem = contiguous(
                    &slice(
                        biases,
                        &[0, groups_al],
                        &[out_dim, groups_al + groups_rem],
                        &[1, 1],
                        None,
                    ),
                    None,
                );
                let qw_al = QuantizedWeight {
                    weight: w_al,
                    scales: Some(s_al),
                    biases: Some(b_al),
                    group_size: qw.group_size,
                    bits: qw.bits,
                    mode: qw.mode.clone(),
                    linear_bias: None,
                    decode_weight_t: None,
                    decode_q4_weight: None,
                    decode_q4_scales: None,
                    decode_q4_biases: None,
                };
                let qw_rem = QuantizedWeight {
                    weight: w_rem,
                    scales: Some(s_rem),
                    biases: Some(b_rem),
                    group_size: qw.group_size,
                    bits: qw.bits,
                    mode: qw.mode.clone(),
                    linear_bias: None,
                    decode_weight_t: None,
                    decode_q4_weight: None,
                    decode_q4_scales: None,
                    decode_q4_biases: None,
                };
                // The aligned prefix hits qmv_fast.
                let y_al = invariant_projection_metal_impl(&x_al, &qw_al)?;
                // Remainder: MLX singleton (Leading=1) or RowExact MLX
                // (Leading>1) so multi-token matches pure-direct.
                let y_rem = if leading > 1 {
                    let x_rem_shape = x_rem.shape();
                    // x_rem is [1, S, rem] or similar with leading product S.
                    let seq = x_rem_shape[x_rem_shape.len() - 2];
                    let cols: Vec<MlxArray> = (0..seq)
                        .map(|t| {
                            let ndim = x_rem_shape.len();
                            let mut starts = vec![0_i32; ndim];
                            let mut ends: Vec<i32> = x_rem_shape.to_vec();
                            starts[ndim - 2] = t;
                            ends[ndim - 2] = t + 1;
                            let strides = vec![1_i32; ndim];
                            let row =
                                contiguous(&slice(&x_rem, &starts, &ends, &strides, None), None);
                            mlx_sys::quantized_matmul_with_mode(
                                &row,
                                &qw_rem.weight,
                                qw_rem.scales.as_ref().unwrap(),
                                qw_rem.biases.as_ref(),
                                true,
                                Some(qw_rem.group_size),
                                Some(qw_rem.bits),
                                qw_rem.mlx_quantization_mode(),
                                None,
                            )
                        })
                        .collect();
                    let refs: Vec<&MlxArray> = cols.iter().collect();
                    // Concat on the sequence axis (second-to-last for 3D).
                    concatenate(&refs, (x_rem_shape.len() - 2) as i32, None)
                } else {
                    mlx_sys::quantized_matmul_with_mode(
                        &x_rem,
                        &qw_rem.weight,
                        qw_rem.scales.as_ref().unwrap(),
                        qw_rem.biases.as_ref(),
                        true,
                        Some(qw_rem.group_size),
                        Some(qw_rem.bits),
                        qw_rem.mlx_quantization_mode(),
                        None,
                    )
                };
                return Some(add(&y_al, &y_rem, None));
            }
        }
    }

    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;

    let mut common_template_args = vec![
        KernelTemplateArg::Dtype {
            name: "InputT",
            dtype: x.dtype(),
        },
        KernelTemplateArg::Dtype {
            name: "OutT",
            dtype: x.dtype(),
        },
        KernelTemplateArg::Int {
            name: "Leading",
            value: leading,
        },
        KernelTemplateArg::Int {
            name: "OutDim",
            value: out_dim,
        },
        KernelTemplateArg::Int {
            name: "InputDim",
            value: input_dim,
        },
    ];

    let mut outputs = if let Some(scales) = qw.scales.as_ref() {
        if !matches!(
            qw.mlx_quantization_mode(),
            mlx_sys::MlxQuantizationMode::Affine
        ) {
            return None;
        }
        let biases = qw.biases.as_ref()?;
        let output_dtype = promote_projection_dtype(x.dtype(), scales.dtype(), biases.dtype())?;
        if let Some(KernelTemplateArg::Dtype { dtype, .. }) = common_template_args.get_mut(1) {
            *dtype = output_dtype;
        }
        if qw.bits <= 0 || qw.bits > 8 || qw.group_size <= 0 {
            return None;
        }
        let pack_factor = 32 / qw.bits;
        let packed_shape_matches = if qw.bits == 6 {
            input_dim % 16 == 0
                && input_dim
                    .checked_mul(qw.bits)?
                    .checked_div(32)
                    .is_some_and(|packed_cols| packed_cols == weight_shape[1])
        } else {
            pack_factor > 0 && pack_factor.checked_mul(weight_shape[1])? == input_dim
        };
        if !packed_shape_matches || input_dim % qw.group_size != 0 {
            return None;
        }
        let group_count = input_dim / qw.group_size;
        let expected_sidecar = vec![out_dim, group_count];
        if scales.shape() != expected_sidecar || biases.shape() != expected_sidecar {
            return None;
        }
        common_template_args.extend([
            KernelTemplateArg::Int {
                name: "PackedCols",
                value: weight_shape[1],
            },
            KernelTemplateArg::Int {
                name: "GroupSize",
                value: qw.group_size,
            },
            KernelTemplateArg::Int {
                name: "GroupCount",
                value: group_count,
            },
            KernelTemplateArg::Int {
                name: "Bits",
                value: qw.bits,
            },
            KernelTemplateArg::Int {
                name: "PackFactor",
                value: pack_factor,
            },
            KernelTemplateArg::Int {
                name: "QuantMask",
                value: (1_i32 << qw.bits) - 1,
            },
        ]);
        let values_per_thread = match qw.bits {
            4 => 16,
            6 | 8 => 8,
            _ => 0,
        };
        let block_size = values_per_thread * 32;
        let qmv_fast_eligible = values_per_thread > 0
            && out_dim % 8 == 0
            && input_dim % block_size == 0
            && qw.group_size >= values_per_thread
            && qw.group_size % values_per_thread == 0;
        if qmv_fast_eligible {
            let kernel = INVARIANT_AFFINE_QMV_FAST_KERNEL.get_or_init(|| {
                MlxMetalKernel::new(
                    "ax_invariant_affine_qmv_fast_v1",
                    &["x", "weight", "scales", "biases"],
                    &["out"],
                    INVARIANT_AFFINE_QMV_FAST_KERNEL_SOURCE,
                    "",
                    true,
                )
            });
            kernel
                .try_apply_with_template(
                    &[x, &qw.weight, scales, biases],
                    &[KernelOutputSpec {
                        shape: out_shape,
                        dtype: output_dtype,
                    }],
                    &common_template_args,
                    (32, (out_dim / 8).saturating_mul(2), 1),
                    (32, 2, 1),
                    None,
                )
                .ok()?
        } else {
            // Non-fast custom kernel does not match MLX for Gemma-like shapes.
            // Use MLX quantized_matmul (bitexact for Leading=1; RowExact for S>1).
            let shape = x.shape();
            let out = if shape.len() == 3 && shape[0] == 1 && shape[1] > 1 {
                let cols: Vec<MlxArray> = (0..shape[1])
                    .map(|t| {
                        let row = contiguous(
                            &slice(x, &[0, t, 0], &[1, t + 1, shape[2]], &[1, 1, 1], None),
                            None,
                        );
                        mlx_sys::quantized_matmul_with_mode(
                            &row,
                            &qw.weight,
                            scales,
                            Some(biases),
                            true,
                            Some(qw.group_size),
                            Some(qw.bits),
                            qw.mlx_quantization_mode(),
                            None,
                        )
                    })
                    .collect();
                let refs: Vec<&MlxArray> = cols.iter().collect();
                concatenate(&refs, 1, None)
            } else {
                mlx_sys::quantized_matmul_with_mode(
                    x,
                    &qw.weight,
                    scales,
                    Some(biases),
                    true,
                    Some(qw.group_size),
                    Some(qw.bits),
                    qw.mlx_quantization_mode(),
                    None,
                )
            };
            vec![out]
        }
    } else {
        if weight_shape[1] != input_dim
            || !matches!(
                qw.weight.dtype(),
                MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
            )
        {
            return None;
        }
        let kernel = INVARIANT_DENSE_PROJECTION_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_invariant_dense_projection_v1",
                &["x", "weight"],
                &["out"],
                INVARIANT_DENSE_PROJECTION_KERNEL_SOURCE,
                "",
                true,
            )
        });
        kernel
            .try_apply_with_template(
                &[x, &qw.weight],
                &[KernelOutputSpec {
                    shape: out_shape,
                    dtype: x.dtype(),
                }],
                &common_template_args,
                (out_dim.saturating_mul(256), 1, 1),
                (256, 1, 1),
                None,
            )
            .ok()?
    };
    outputs.pop()
}

fn promote_projection_dtype(
    input: MlxDtype,
    scales: MlxDtype,
    biases: MlxDtype,
) -> Option<MlxDtype> {
    if scales != biases {
        return None;
    }
    if input == scales {
        return Some(input);
    }
    if input == MlxDtype::Float32 || scales == MlxDtype::Float32 {
        return Some(MlxDtype::Float32);
    }
    if matches!(input, MlxDtype::Bfloat16 | MlxDtype::Float16)
        && matches!(scales, MlxDtype::Bfloat16 | MlxDtype::Float16)
    {
        return Some(MlxDtype::Float32);
    }
    None
}

pub(crate) fn mlx_slice_last_dim(x: &MlxArray, start: i32, end: i32) -> MlxArray {
    slice_last_dim(x, start, end, None)
}

pub(crate) fn scale_hidden_pub(hidden: &MlxArray, scale: f32) -> MlxArray {
    scale_hidden(hidden, scale)
}

pub(crate) fn scale_hidden(hidden: &MlxArray, scale: f32) -> MlxArray {
    // `cached_scalar` deduplicates the (value, dtype) pair across the process,
    // so steady-state decode pays one `multiply` op per call instead of
    // (astype + multiply). Saves ~4 ops/step on Gemma 4 E2B (one per scale
    // site: hidden_states_scale + 3 inside compute_per_layer_inputs_arr).
    let s_arr = mlx_sys::ops::cached_scalar(scale, hidden.dtype());
    multiply(hidden, &s_arr, None)
}

static ADD_MUL_SCALAR_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const ADD_MUL_SCALAR_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    float av = static_cast<float>(a[idx]);
    float bv = static_cast<float>(b[idx]);
    float scale_v = static_cast<float>(scale[0]);
    T rounded_sum = static_cast<T>(av + bv);
    out[idx] = static_cast<T>(static_cast<float>(rounded_sum) * scale_v);
"#;

pub(crate) fn add_then_multiply_scalar(a: &MlxArray, b: &MlxArray, scalar: &MlxArray) -> MlxArray {
    add_then_multiply_scalar_metal(a, b, scalar)
        .unwrap_or_else(|| multiply(&add(a, b, None), scalar, None))
}

fn add_then_multiply_scalar_metal(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if !fastpath::layer_scalar_fused_add_enabled()
        || !layer_scalar_fused_add_shape_supported(&a.shape())
    {
        return None;
    }
    add_then_multiply_scalar_metal_impl(a, b, scalar)
}

fn layer_scalar_fused_add_shape_supported(shape: &[i32]) -> bool {
    shape.get(1).copied().unwrap_or(1) == 1
}

fn add_then_multiply_scalar_metal_impl(
    a: &MlxArray,
    b: &MlxArray,
    scalar: &MlxArray,
) -> Option<MlxArray> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() || scalar.dtype() != a.dtype() {
        return None;
    }
    if !matches!(
        a.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let scalar_elements = scalar
        .shape()
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if scalar_elements != 1 {
        return None;
    }
    let shape = a.shape();
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = ADD_MUL_SCALAR_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_add_mul_scalar_v1",
            &["a", "b", "scale"],
            &["out"],
            ADD_MUL_SCALAR_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[a, b, scalar],
        &[KernelOutputSpec {
            shape,
            dtype: a.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: a.dtype(),
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
        ],
        (element_count, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

pub(crate) fn scalar_like(value: f32, dtype: MlxDtype) -> MlxArray {
    // Retained for callers outside the steady-state decode hot path
    // (e.g. MoE router masking, test fixtures) where the per-call astype is
    // not the bottleneck and value uniqueness is not guaranteed.
    let scalar = MlxArray::from_raw_data(
        &value as *const f32 as *const u8,
        std::mem::size_of::<f32>(),
        &[1_i32],
        MlxDtype::Float32,
    );
    astype(&scalar, dtype, None)
}

pub(crate) fn apply_final_logit_softcap(cfg: &ModelConfig, logits: &MlxArray) -> MlxArray {
    let Some(cap) = cfg.final_logit_softcapping.filter(|cap| *cap > 0.0) else {
        return logits.clone();
    };
    let inv_cap = 1.0_f32 / cap;
    let inv_cap_arr = mlx_sys::ops::cached_scalar(inv_cap, logits.dtype());
    let cap_arr = mlx_sys::ops::cached_scalar(cap, logits.dtype());
    let scaled = multiply(logits, &inv_cap_arr, None);
    multiply(&tanh(&scaled, None), &cap_arr, None)
}

pub(crate) fn shape_element_count(shape: &[i32]) -> usize {
    shape
        .iter()
        .map(|dim| usize::try_from(*dim).expect("MLX shape dims must be non-negative"))
        .product()
}

pub(crate) fn squeeze_switch_singleton(x: &MlxArray) -> MlxArray {
    let mut shape = x.shape();
    let ndim = shape.len();
    if ndim >= 2 && shape[ndim - 2] == 1 {
        shape.remove(ndim - 2);
        reshape(x, &shape, None)
    } else {
        x.clone()
    }
}

/// Gather-matmul for expert weights (quantized or dense).
///
/// `x`: [..., hidden], `qw.weight`: [num_experts, expert_size, hidden] (or packed).
/// `indices`: [..., top_k].  Returns [..., top_k, out_size].
pub(crate) fn qw_gather(
    x: &MlxArray,
    qw: &QuantizedWeight,
    indices: &MlxArray,
    sorted_indices: bool,
) -> MlxArray {
    let y = if let Some(scales) = &qw.scales {
        // MXFP4 has no affine group-bias channel; pass None for non-affine modes.
        let mode = qw.mlx_quantization_mode();
        let quant_biases = match mode {
            mlx_sys::MlxQuantizationMode::Affine => qw.biases.as_ref(),
            _ => None,
        };
        mlx_sys::gather_qmm_with_mode(
            x,
            &qw.weight,
            scales,
            quant_biases,
            indices,
            true,
            Some(qw.group_size),
            Some(qw.bits),
            mode,
            sorted_indices,
            None,
        )
    } else {
        // Dense experts: weight shape [N, out, in] → need [N, in, out] for gather_mm.
        let ndim = qw.weight.ndim();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        let last = axes.len() - 1;
        axes.swap(last - 1, last);
        let wt = transpose(&qw.weight, &axes, None);
        gather_mm(x, &wt, indices, sorted_indices, None)
    };

    // Dense SwitchLinear bias: y += bias[indices]  (mlx-lm switch_layers.py).
    // bias shape [num_experts, out]; indices select experts → [..., top_k, out]
    // after expand for broadcast against gather output.
    if let Some(linear_bias) = &qw.linear_bias {
        apply_expert_linear_bias(&y, linear_bias, indices)
    } else {
        y
    }
}

/// `y + expand_dims(linear_bias[indices], -2)` matching mlx-lm QuantizedSwitchLinear.
fn apply_expert_linear_bias(y: &MlxArray, linear_bias: &MlxArray, indices: &MlxArray) -> MlxArray {
    // take along expert axis 0: bias[indices] with indices of any rank.
    // Use take for flat indices then reshape to indices.shape + [out].
    let out_dim = *linear_bias
        .shape()
        .last()
        .expect("expert linear bias must be [E, out]");
    let flat_idx = reshape(indices, &[-1], None);
    let gathered = take(linear_bias, &flat_idx, 0, None); // [N, out]
    let mut bias_shape = indices.shape();
    bias_shape.push(out_dim);
    let gathered = reshape(&gathered, &bias_shape, None);
    // gather_qmm output often has a singleton dim before the last (SwitchGLU
    // expand_dims); expand bias so it broadcasts: insert dim at -2 when needed.
    let y_shape = y.shape();
    let bias = if y_shape.len() == gathered.ndim() + 1
        && y_shape.get(y_shape.len().saturating_sub(2)) == Some(&1)
    {
        expand_dims_axes(&gathered, &[-2], None)
    } else {
        gathered
    };
    add(y, &bias, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{
        MlxQuantizationMode, clear_cache, contiguous, eval, get_peak_memory, matmul, quantize,
        quantized_matmul, reset_peak_memory, transpose,
    };

    #[test]
    fn qw_applies_dense_linear_bias() {
        // Dense Linear: y = x @ W^T + b  (mlx-lm nn.Linear with bias=True).
        let w_data = [1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity
        let weight = array_f32(&w_data, &[2, 2]);
        let bias = array_f32(&[0.5, -0.25], &[2]);
        let qw = QuantizedWeight {
            weight,
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: Some(bias),
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let x = array_f32(&[1.0, 2.0], &[1, 1, 2]);
        let out = super::qw(&x, &qw);
        eval(&[&out]);
        let got = out.data_f32();
        // identity + bias → [1.5, 1.75]
        assert!((got[0] - 1.5).abs() < 1e-5, "got {}", got[0]);
        assert!((got[1] - 1.75).abs() < 1e-5, "got {}", got[1]);
    }

    #[test]
    fn project_unquantized_decode_matches_x_at_weight_t() {
        let hidden = 8;
        let vocab = 16;
        let mut w_data = vec![0.0f32; (vocab * hidden) as usize];
        for row in 0..vocab {
            for col in 0..hidden {
                w_data[(row * hidden + col) as usize] = (row + 1) as f32 * 0.1 + col as f32 * 0.01;
            }
        }
        let weight = array_f32(&w_data, &[vocab, hidden]);
        let x_data: Vec<f32> = (0..hidden).map(|i| (i + 1) as f32 * 0.25).collect();
        let x = array_f32(&x_data, &[1, 1, hidden]);
        let hits_before = super::unquantized_decode_projection_hits();
        let metal = super::project_unquantized_decode(&x, &weight).expect("decode GEMV eligible");
        let lm = QuantizedWeight {
            weight: weight.clone(),
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let shipped = super::project_lm_head(&x, &lm);
        let reference = matmul(&x, &transpose(&weight, &[1, 0], None), None);
        eval(&[&metal, &shipped, &reference]);
        let got = metal.data_f32();
        let shipped_got = shipped.data_f32();
        let want = reference.data_f32();
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "idx {i}: metal {} vs ref {}",
                got[i],
                want[i]
            );
            assert!(
                (shipped_got[i] - want[i]).abs() < 1e-4,
                "idx {i}: shipped {} vs ref {}",
                shipped_got[i],
                want[i]
            );
        }
        assert!(
            super::unquantized_decode_projection_hits() > hits_before,
            "shipped lm_head must take the no-copy GEMV"
        );
    }

    #[test]
    fn project_unquantized_decode_skips_full_weight_transpose_buffer() {
        // Large enough that a materialized W.T is visible in peak memory.
        let hidden = 128i32;
        let vocab = 4096i32;
        let w_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i % 17) as f32 * 0.01)
            .collect();
        let x_data: Vec<f32> = (0..hidden).map(|i| (i % 5) as f32 * 0.1).collect();
        let weight = array_f32(&w_data, &[vocab, hidden]);
        let x = array_f32(&x_data, &[1, hidden]);
        eval(&[&x, &weight]);

        clear_cache();
        reset_peak_memory();
        let metal = super::project_unquantized_decode(&x, &weight).expect("decode GEMV eligible");
        eval(&[&metal]);
        let peak_metal = get_peak_memory();

        clear_cache();
        reset_peak_memory();
        let transposed = contiguous(&transpose(&weight, &[1, 0], None), None);
        let reference = matmul(&x, &transposed, None);
        eval(&[&reference]);
        let peak_transpose = get_peak_memory();

        let weight_bytes = (vocab as usize) * (hidden as usize) * 4;
        assert!(
            peak_metal + weight_bytes / 2 < peak_transpose,
            "no-copy GEMV peak {peak_metal} should beat materialized transpose peak {peak_transpose} by ~half of {weight_bytes} weight bytes"
        );
    }

    #[test]
    fn qw_pretransposed_lm_head_matches_lazy_transpose() {
        let hidden = 8i32;
        let vocab = 16i32;
        let w_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect();
        let x_data: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.25).collect();
        let weight = array_f32(&w_data, &[vocab, hidden]);
        let x = array_f32(&x_data, &[1, 1, hidden]);
        let mut prepared = QuantizedWeight {
            weight: weight.clone(),
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        prepared.prepare_contiguous_decode_weight_t();
        let weight_t = prepared
            .decode_weight_t
            .as_ref()
            .expect("unquantized rank-2 lm_head must materialize W_t once");
        assert_eq!(weight_t.shape(), vec![hidden, vocab]);
        assert_eq!(
            prepared.weight.shape(),
            vec![vocab, hidden],
            "original [out, in] layout must remain a lazy view of W_t"
        );

        let lazy = QuantizedWeight {
            weight,
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let got = super::qw(&x, &prepared);
        let want = super::qw(&x, &lazy);
        eval(&[&got, &want]);
        let got = got.data_f32();
        let want = want.data_f32();
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert!(
                (got[i] - want[i]).abs() < 1e-4,
                "idx {i}: prepared {} vs lazy {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn prepare_contiguous_decode_weight_t_skips_quantized() {
        let weight = array_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let scales = array_f32(&[1.0, 1.0], &[2, 1]);
        let biases = array_f32(&[0.0, 0.0], &[2, 1]);
        let mut quantized = QuantizedWeight {
            weight,
            scales: Some(scales),
            biases: Some(biases),
            group_size: 2,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        quantized.prepare_contiguous_decode_weight_t();
        quantized.prepare_decode_q4_lm_head();
        assert!(
            quantized.decode_weight_t.is_none(),
            "quantized lm_head must not grow a dense W_t copy"
        );
        assert!(
            quantized.decode_q4_weight.is_none(),
            "already-quantized tensors must not grow a decode q4 cache"
        );
    }

    #[test]
    fn prepare_decode_q4_lm_head_is_decode_only() {
        let hidden = 64i32;
        let vocab = 32i32;
        let seq = 4i32;
        let w_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i % 13) as f32) * 0.05 - 0.3)
            .collect();
        let x_decode_data: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32) * 0.1 - 0.3).collect();
        let x_prefill_data: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let weight = array_f32(&w_data, &[vocab, hidden]);
        let x_decode = array_f32(&x_decode_data, &[1, 1, hidden]);
        let x_prefill = array_f32(&x_prefill_data, &[1, seq, hidden]);
        let mut prepared = QuantizedWeight {
            weight: weight.clone(),
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        prepared.prepare_decode_q4_lm_head();
        prepared.prepare_contiguous_decode_weight_t();
        assert!(
            prepared.decode_q4_weight.is_some(),
            "unquantized rank-2 hidden%64==0 must build a decode quant cache"
        );
        assert!(
            prepared.decode_weight_t.is_some(),
            "prefill keeps a contiguous BF16 W_t"
        );
        let got_decode = super::qw(&x_decode, &prepared);
        let got_prefill = super::qw(&x_prefill, &prepared);
        let want_prefill = matmul(&x_prefill, prepared.decode_weight_t.as_ref().unwrap(), None);
        eval(&[&got_decode, &got_prefill, &want_prefill]);
        let got_decode = got_decode.data_f32();
        assert!(
            got_decode.iter().all(|v| v.is_finite()),
            "2-bit decode lm_head must produce finite logits"
        );
        let got_prefill = got_prefill.data_f32();
        let want_prefill = want_prefill.data_f32();
        assert_eq!(got_prefill.len(), want_prefill.len());
        let mut max_abs = 0.0f32;
        for i in 0..got_prefill.len() {
            max_abs = max_abs.max((got_prefill[i] - want_prefill[i]).abs());
        }
        assert!(
            max_abs < 1e-4,
            "prefill must use BF16 W_t, not the 2-bit decode cache, max_abs={max_abs}"
        );
    }

    #[test]
    fn qwen_prefill_maybe_skip_bf16_astype_skips_when_already_bf16() {
        let data = [1.0f32, -0.5, 0.25, 2.0];
        let x = array_f32(&data, &[1, 2, 2]);
        let bf = astype(&x, MlxDtype::Bfloat16, None);
        eval(&[&bf]);
        let skipped = super::qwen_prefill_maybe_skip_bf16_astype(&bf, "qwen3_5", 1024);
        let forced = astype(&bf, MlxDtype::Bfloat16, None);
        eval(&[&skipped, &forced]);
        assert_eq!(skipped.dtype(), MlxDtype::Bfloat16);
        assert_eq!(skipped.shape(), forced.shape());
        let cast = super::qwen_prefill_maybe_skip_bf16_astype(&x, "qwen3_5", 1024);
        eval(&[&cast]);
        assert_eq!(cast.dtype(), MlxDtype::Bfloat16);
        assert!(
            fastpath::should_qwen_prefill_skip_bf16_astype_for(true, "qwen3_5", 1024),
            "shipped skip-astype gate must accept prefill seq"
        );
        let gemma_skipped = super::qwen_prefill_maybe_skip_bf16_astype(&bf, "gemma4", 128);
        eval(&[&gemma_skipped]);
        assert_eq!(gemma_skipped.dtype(), MlxDtype::Bfloat16);
        assert_eq!(gemma_skipped.shape(), forced.shape());
        assert!(
            fastpath::should_gemma4_prefill_bf16_embed_for(true, "gemma4", 128),
            "shipped Gemma 4 bf16 embed must skip unused astype at p128"
        );
    }

    #[test]
    fn qwen_prefill_maybe_async_embed_submits_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let hidden = array_f32(&data, &[1, 4, 8]);
        super::qwen_prefill_maybe_async_embed_for(&hidden, true, "qwen3_5", 1024);
        eval(&[&hidden]);
        assert_eq!(hidden.shape(), vec![1, 4, 8]);
        assert!(
            hidden.data_f32().iter().all(|v| v.is_finite()),
            "async embed must leave a finite materialized tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_async_embed_for(true, "qwen3_5", 1024),
            "shipped async-embed gate must accept the p2048 chunk length"
        );
        super::qwen_prefill_maybe_async_embed_for(&hidden, false, "qwen3_5", 1024);
        super::qwen_prefill_maybe_async_embed_for(&hidden, true, "qwen3_5", 512);
        super::qwen_prefill_maybe_async_embed_for(&hidden, true, "gemma4", 1024);
    }

    #[test]
    fn qwen_prefill_dequant_dense_matches_quantized_matmul() {
        let input_dim = 64i32;
        let output_dim = 128i32;
        let seq = 1024i32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * 0.015625)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let qw = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..seq * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let x = array_f32(&input_data, &[1, seq, input_dim]);
        super::set_qwen_prefill_dequant_dense_family(false);
        let via_qmm = super::qw(&x, &qw);
        let weight_t = super::cached_prefill_dequant_weight_t(&qw)
            .expect("affine 4-bit weight must dequantize");
        let via_dense = matmul(&x, &weight_t, None);
        eval(&[&via_qmm, &via_dense]);
        let qmm = via_qmm.data_f32();
        let dense = via_dense.data_f32();
        assert_eq!(qmm.len(), dense.len());
        let mut max_abs = 0.0f32;
        for i in 0..qmm.len() {
            max_abs = max_abs.max((qmm[i] - dense[i]).abs());
        }
        assert!(
            max_abs < 2e-2,
            "dequant+dense must match steel qmm, max_abs={max_abs}"
        );
        let x_decode = array_f32(&input_data[..input_dim as usize], &[1, 1, input_dim]);
        super::set_qwen_prefill_dequant_dense_family(true);
        let decode_out = super::qw(&x_decode, &qw);
        super::set_qwen_prefill_dequant_dense_family(false);
        let decode_qmm = super::qw(&x_decode, &qw);
        eval(&[&decode_out, &decode_qmm]);
        assert_eq!(
            decode_out.data_f32(),
            decode_qmm.data_f32(),
            "seq=1 must stay on steel qmm"
        );
    }

    #[test]
    fn qwen_prefill_maybe_flat_qmm_matches_3d_quantized_matmul() {
        let input_dim = 32i32;
        let output_dim = 64i32;
        let seq = 1024i32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 63) as f32 - 31.0) * 0.015625)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let input_data: Vec<f32> = (0..seq * input_dim)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let x = array_f32(&input_data, &[1, seq, input_dim]);
        let qmm_3d = mlx_sys::quantized_matmul_with_mode(
            &x,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
        );
        let qmm_flat = super::qwen_prefill_maybe_flat_qmm_for(&x, true, |flat| {
            mlx_sys::quantized_matmul_with_mode(
                flat,
                &quantized[0],
                &quantized[1],
                Some(&quantized[2]),
                true,
                Some(32),
                Some(4),
                MlxQuantizationMode::Affine,
                None,
            )
        });
        eval(&[&qmm_3d, &qmm_flat]);
        assert_eq!(qmm_flat.shape(), vec![1, seq, output_dim]);
        let a = qmm_3d.data_f32();
        let b = qmm_flat.data_f32();
        assert_eq!(a.len(), b.len());
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1e-4,
            "flat 2-D qmm must match 3-D steel qmm, max_abs={max_abs}"
        );
        assert!(
            fastpath::should_qwen_prefill_flat_qmm_for(true, 1024, 3),
            "shipped flat-qmm gate must accept the p2048 chunk length"
        );
        let skipped = super::qwen_prefill_maybe_flat_qmm_for(&x, false, |inner| inner.clone());
        eval(&[&skipped]);
        assert_eq!(skipped.shape(), x.shape());
    }

    #[test]
    fn qwen_prefill_tile_qmm_matches_oneshot() {
        let input_dim = 32i32;
        let output_dim = 64i32;
        let seq = 1024i32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 63) as f32 - 31.0) * 0.015625)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let input_data: Vec<f32> = (0..seq * input_dim)
            .map(|index| ((index % 17) as f32 - 8.0) * 0.03125)
            .collect();
        let x = array_f32(&input_data, &[1, seq, input_dim]);
        let oneshot = mlx_sys::quantized_matmul_with_mode(
            &x,
            &quantized[0],
            &quantized[1],
            Some(&quantized[2]),
            true,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
        );
        let tiled = super::qwen_prefill_maybe_tile_qmm_for(&x, true, "qwen3_5", |chunk| {
            mlx_sys::quantized_matmul_with_mode(
                chunk,
                &quantized[0],
                &quantized[1],
                Some(&quantized[2]),
                true,
                Some(32),
                Some(4),
                MlxQuantizationMode::Affine,
                None,
            )
        });
        eval(&[&oneshot, &tiled]);
        assert_eq!(tiled.shape(), vec![1, seq, output_dim]);
        let a = oneshot.data_f32();
        let b = tiled.data_f32();
        assert_eq!(a.len(), b.len());
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1e-4,
            "tiled qmm must match oneshot steel qmm, max_abs={max_abs}"
        );
        assert!(
            fastpath::should_qwen_prefill_tile_qmm_for(true, "qwen3_5", 1024),
            "shipped tile-qmm gate must accept the p2048 chunk length"
        );
        let skipped =
            super::qwen_prefill_maybe_tile_qmm_for(&x, false, "qwen3_5", |inner| inner.clone());
        eval(&[&skipped]);
        assert_eq!(skipped.shape(), x.shape());
    }

    #[test]
    fn prepare_contiguous_decode_weight_t_keeps_one_physical_buffer() {
        // Large enough that a second full copy would show in peak memory.
        let hidden = 128i32;
        let vocab = 4096i32;
        let w_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i % 17) as f32 * 0.01)
            .collect();
        let weight = array_f32(&w_data, &[vocab, hidden]);
        eval(&[&weight]);
        let mut prepared = QuantizedWeight {
            weight,
            scales: None,
            biases: None,
            group_size: 1,
            bits: 32,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        prepared.prepare_contiguous_decode_weight_t();
        let weight_t = prepared.decode_weight_t.as_ref().expect("prepared W_t");
        clear_cache();
        reset_peak_memory();
        eval(&[&prepared.weight, weight_t]);
        let peak = get_peak_memory();
        let one_copy = (vocab as usize) * (hidden as usize) * 4;
        assert!(
            peak < one_copy.saturating_mul(2),
            "eval of W view + W_t must not materialize two full copies (peak {peak}, one copy {one_copy})"
        );
        assert_eq!(prepared.weight.shape(), vec![vocab, hidden]);
        assert_eq!(weight_t.shape(), vec![hidden, vocab]);
    }

    #[test]
    fn project_unquantized_decode_matches_bf16_x_at_weight_t() {
        let hidden = 8;
        let vocab = 16;
        let w_data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i as f32) * 0.01 - 0.5)
            .collect();
        let x_data: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.25).collect();
        let weight = astype(
            &array_f32(&w_data, &[vocab, hidden]),
            MlxDtype::Bfloat16,
            None,
        );
        let x = astype(
            &array_f32(&x_data, &[1, 1, hidden]),
            MlxDtype::Bfloat16,
            None,
        );
        let metal = super::project_unquantized_decode(&x, &weight).expect("bf16 GEMV eligible");
        let reference = matmul(&x, &transpose(&weight, &[1, 0], None), None);
        eval(&[&metal, &reference]);
        let got = astype(&metal, MlxDtype::Float32, None);
        let want = astype(&reference, MlxDtype::Float32, None);
        eval(&[&got, &want]);
        for (g, w) in got.data_f32().iter().zip(want.data_f32().iter()) {
            assert!((g - w).abs() < 2.0e-2, "bf16 gemv {g} vs ref {w}");
        }
    }

    #[test]
    fn project_unquantized_decode_rejects_multi_token() {
        let weight = array_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let x = array_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        assert!(super::project_unquantized_decode(&x, &weight).is_none());
    }

    #[test]
    fn expert_linear_bias_matches_mlx_lm_index_add() {
        // bias: [E=3, out=2], indices: [1, 2] → bias[[1,2]] = [[2,3],[4,5]]
        let bias_data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let bias = MlxArray::from_raw_data(
            bias_data.as_ptr() as *const u8,
            bias_data.len() * 4,
            &[3, 2],
            MlxDtype::Float32,
        );
        let y_data = [10.0f32, 20.0, 30.0, 40.0];
        let y = MlxArray::from_raw_data(
            y_data.as_ptr() as *const u8,
            y_data.len() * 4,
            &[1, 2, 2],
            MlxDtype::Float32,
        );
        let idx_data = [1u32, 2];
        let indices = MlxArray::from_raw_data(
            idx_data.as_ptr() as *const u8,
            idx_data.len() * 4,
            &[1, 2],
            MlxDtype::Uint32,
        );
        let out = apply_expert_linear_bias(&y, &bias, &indices);
        eval(&[&out]);
        let got = out.data_f32();
        // y[0] + bias[1] = [10,20]+[2,3] = [12,23]
        // y[1] + bias[2] = [30,40]+[4,5] = [34,45]
        assert!((got[0] - 12.0).abs() < 1e-5);
        assert!((got[1] - 23.0).abs() < 1e-5);
        assert!((got[2] - 34.0).abs() < 1e-5);
        assert!((got[3] - 45.0).abs() < 1e-5);
    }

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn layer_scalar_fused_add_is_decode_only() {
        assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 4]));
        assert!(layer_scalar_fused_add_shape_supported(&[1, 1, 35, 4]));
        assert!(!layer_scalar_fused_add_shape_supported(&[1, 2, 4]));
        assert!(!layer_scalar_fused_add_shape_supported(&[1, 2048, 4]));
    }

    #[test]
    fn add_then_multiply_scalar_metal_matches_unfused_float32() {
        let a = array_f32(&[0.5, -1.0, 2.0, 3.5, -4.0, 8.0], &[2, 3]);
        let b = array_f32(&[1.0, 4.0, -2.0, 0.5, 3.0, -8.0], &[2, 3]);
        let scalar = array_f32(&[0.25], &[1]);

        let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
            .expect("scalar fused add should support float32 inputs");
        let reference = multiply(&add(&a, &b, None), &scalar, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![2, 3]);
        assert_eq!(direct.data_f32(), reference.data_f32());
    }

    #[test]
    fn row_exact_projection_matches_independent_quantized_rows() {
        let input_dim = 64;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 127) as f32 - 63.0) * 0.015625)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(64),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 64,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 31) as f32 - 15.0) * 0.03125)
            .collect();
        let input = array_f32(&input_data, &[2, 1, input_dim]);
        let batched = qw_with_policy(&input, &weight, ProjectionBatchPolicy::RowExact);

        for row in 0..2i32 {
            let row_start = (row as usize) * (input_dim as usize);
            let single_input = array_f32(
                &input_data[row_start..row_start + input_dim as usize],
                &[1, 1, input_dim],
            );
            let expected = qw(&single_input, &weight);
            let actual = contiguous(
                &slice(
                    &batched,
                    &[row, 0, 0],
                    &[row + 1, 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn batched_mxfp4_multi_token_without_exact_vs_singleton() {
        let input_dim = 64i32;
        let output_dim = 32i32;
        let seq = 2i32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let input = array_f32(&input_data, &[1, seq, input_dim]);
        let _exact_off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        let batched = qw(&input, &weight);
        let mut max_abs = 0.0f32;
        for t in 0..seq {
            let start = (t * input_dim) as usize;
            let single = array_f32(
                &input_data[start..start + input_dim as usize],
                &[1, 1, input_dim],
            );
            let expected = qw(&single, &weight);
            let actual = contiguous(
                &slice(
                    &batched,
                    &[0, t, 0],
                    &[1, t + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            eval(&[&actual, &expected]);
            let got = actual.data_f32();
            let exp = expected.data_f32();
            for i in 0..got.len() {
                max_abs = max_abs.max((got[i] - exp[i]).abs());
            }
        }
        eprintln!("batched MXFP4 S=2 vs singleton max_abs={max_abs}");
        assert!(
            max_abs < 1.0e-5,
            "if this fails, batched MXFP4 qmm is not singleton-exact (max_abs={max_abs})"
        );
    }

    #[test]
    fn exact_shared_mxfp4_multi_token_matches_singleton_rows() {
        let input_dim = 64i32;
        let output_dim = 32i32;
        let seq = 2i32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.01171875)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Mxfp4,
            None,
            None,
        );
        assert_eq!(quantized.len(), 2, "mxfp4 quant returns [packed, scales]");
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: None,
            group_size: 32,
            bits: 4,
            mode: "mxfp4".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..(seq * input_dim) as usize)
            .map(|index| ((index % 29) as f32 - 14.0) * 0.03125)
            .collect();
        let input = array_f32(&input_data, &[1, seq, input_dim]);
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let batched = qw(&input, &weight);
        for t in 0..seq {
            let start = (t * input_dim) as usize;
            let single = array_f32(
                &input_data[start..start + input_dim as usize],
                &[1, 1, input_dim],
            );
            let expected = qw(&single, &weight);
            let actual = contiguous(
                &slice(
                    &batched,
                    &[0, t, 0],
                    &[1, t + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            eval(&[&actual, &expected]);
            let got = actual.data_f32();
            let exp = expected.data_f32();
            let mut max_abs = 0.0f32;
            for i in 0..got.len() {
                max_abs = max_abs.max((got[i] - exp[i]).abs());
            }
            assert!(
                max_abs < 1.0e-5,
                "MXFP4 exact S>1 Shared qw must match singleton row {t}, max_abs={max_abs}"
            );
        }
    }

    #[test]
    fn invariant_affine_projection_is_bit_exact_across_microbatch_shapes() {
        let input_dim = 64;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 113) as f32 - 56.0) * 0.01171875)
            .collect();
        let weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let quantized = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: quantized[0].clone(),
            scales: Some(quantized[1].clone()),
            biases: Some(quantized[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 43) as f32 - 21.0) * 0.02734375)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let microbatch = invariant_projection_metal_impl(&input, &weight)
            .expect("affine invariant projection should support two rows");

        for row in 0..2 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let expected = invariant_projection_metal_impl(&single, &weight)
                .expect("affine invariant projection should support one row");
            let actual = contiguous(
                &slice(
                    &microbatch,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            let expected = astype(&expected, MlxDtype::Float32, None);
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn invariant_split_2816_matches_mlx_and_microbatch() {
        // Split path: 2560 qmv_fast + 256 MLX rem. Must match full MLX and microbatch.
        let input_dim = 2816;
        let output_dim = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..3 * input_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 3, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let mb = invariant_projection_metal_impl(&input, &weight).expect("split 2816");
        for row in 0..3 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let inv_s = invariant_projection_metal_impl(&single, &weight).expect("single");
            let mlx_s = quantized_matmul(
                &single,
                &q[0],
                &q[1],
                Some(&q[2]),
                true,
                Some(group_size),
                Some(bits),
                None,
            );
            let actual = contiguous(
                &slice(
                    &mb,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let a = astype(&actual, MlxDtype::Float32, None);
            let b = astype(&inv_s, MlxDtype::Float32, None);
            let c = astype(&mlx_s, MlxDtype::Float32, None);
            eval(&[&a, &b, &c]);
            assert_eq!(a.data_f32(), b.data_f32(), "microbatch vs single row {row}");
            // Split sum may have tiny float assoc vs full MLX — allow ulp-level.
            let da = a.data_f32();
            let dc = c.data_f32();
            let mut maxd = 0.0f32;
            for i in 0..da.len() {
                maxd = maxd.max((da[i] - dc[i]).abs());
            }
            eprintln!("row {row}: max|split-mlx|={maxd} bitexact={}", da == dc);
            assert!(maxd < 1e-2, "split vs full MLX too far: {maxd}");
        }
    }

    #[test]
    fn invariant_qmv_2816_matches_mlx_singleton() {
        // Gemma hidden=2816 is eleven complete 6-bit qmv lane blocks.
        let input_dim = 2816;
        let output_dim = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..input_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let inv = invariant_projection_metal_impl(&input, &weight).expect("qmv 2816");
        let mlx = quantized_matmul(
            &input,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            None,
        );
        let a = astype(&inv, MlxDtype::Float32, None);
        let b = astype(&mlx, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let da = a.data_f32();
        let db = b.data_f32();
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        assert!(maxd < 1e-3, "qmv 2816 vs MLX maxΔ={maxd}");
    }

    #[test]
    fn invariant_gemma26_shapes_self_consistent_and_mlx() {
        // Realistic Gemma 26B-A4B projection shapes under Shared microbatch S=3.
        let cases: &[(i32, i32, i32, i32)] = &[
            // (input_dim, output_dim, bits, group_size)
            (2816, 4096, 6, 32), // q_proj
            (2816, 2048, 6, 32), // k/v
            (4096, 2816, 6, 32), // o_proj (512-aligned in)
            (2816, 2112, 6, 64), // dense gate/up
            (2112, 2816, 6, 64), // dense down
            (2816, 2112, 6, 32),
            (2112, 2816, 6, 32),
            (2816, 128, 8, 64), // router-ish
            (2816, 128, 8, 32),
            (704, 2816, 4, 32), // expert down-like (if used as linear)
            (2816, 704, 4, 32),
        ];
        for &(input_dim, output_dim, bits, group_size) in cases {
            let n = (input_dim * output_dim) as usize;
            let weight_data: Vec<f32> = (0..n)
                .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
                .collect();
            let source = array_f32(&weight_data, &[output_dim, input_dim]);
            let q = quantize(
                &source,
                Some(group_size),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let weight = QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: Some(q[2].clone()),
                group_size,
                bits,
                mode: "affine".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            };
            let s = 3_i32;
            let input_data: Vec<f32> = (0..(s * input_dim) as usize)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
                .collect();
            let input = astype(
                &array_f32(&input_data, &[1, s, input_dim]),
                MlxDtype::Bfloat16,
                None,
            );
            let Some(mb) = invariant_projection_metal_impl(&input, &weight) else {
                eprintln!(
                    "SKIP no invariant in={input_dim} out={output_dim} b={bits} gs={group_size}"
                );
                continue;
            };
            let mut max_self = 0.0f32;
            let mut max_mlx = 0.0f32;
            let mut bitexact_self = true;
            let mut bitexact_mlx = true;
            for row in 0..s {
                let single = contiguous(
                    &slice(
                        &input,
                        &[0, row, 0],
                        &[1, row + 1, input_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let inv_s =
                    invariant_projection_metal_impl(&single, &weight).expect("singleton invariant");
                let mlx_s = quantized_matmul(
                    &single,
                    &q[0],
                    &q[1],
                    Some(&q[2]),
                    true,
                    Some(group_size),
                    Some(bits),
                    None,
                );
                let actual = contiguous(
                    &slice(
                        &mb,
                        &[0, row, 0],
                        &[1, row + 1, output_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let a = astype(&actual, MlxDtype::Float32, None);
                let b = astype(&inv_s, MlxDtype::Float32, None);
                let c = astype(&mlx_s, MlxDtype::Float32, None);
                eval(&[&a, &b, &c]);
                let da = a.data_f32();
                let db = b.data_f32();
                let dc = c.data_f32();
                if da != db {
                    bitexact_self = false;
                }
                if da != dc {
                    bitexact_mlx = false;
                }
                for i in 0..da.len() {
                    max_self = max_self.max((da[i] - db[i]).abs());
                    max_mlx = max_mlx.max((da[i] - dc[i]).abs());
                }
            }
            eprintln!(
                "in={input_dim} out={output_dim} b={bits} gs={group_size}: self_exact={bitexact_self} max_self={max_self:.6e} mlx_exact={bitexact_mlx} max_mlx={max_mlx:.6e}"
            );
            assert!(
                max_self == 0.0,
                "SELF FAIL in={input_dim} out={output_dim}: max_self={max_self}"
            );
        }
    }

    #[test]
    fn invariant_2d_vs_3d_singleton_shape() {
        let input_dim = 2816i32;
        let output_dim = 4096i32;
        let bits = 6i32;
        let group_size = 32i32;
        let n = (input_dim * output_dim) as usize;
        let weight_data: Vec<f32> = (0..n)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..input_dim as usize)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let x3 = astype(
            &array_f32(&input_data, &[1, 1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let x2 = astype(
            &array_f32(&input_data, &[1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let y3 = invariant_projection_metal_impl(&x3, &weight).expect("3d");
        let y2 = invariant_projection_metal_impl(&x2, &weight).expect("2d");
        let a = astype(&y3, MlxDtype::Float32, None);
        let b = astype(&y2, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let da = a.data_f32();
        let db = b.data_f32();
        assert_eq!(da.len(), db.len());
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        eprintln!("2d vs 3d maxd={maxd} bitexact={}", da == db);
        assert_eq!(da, db, "2d vs 3d shape mismatch maxd={maxd}");
    }

    #[test]
    fn invariant_nonfast_2112_matches_mlx_singleton() {
        // intermediate_size-like: non-fast path must match MLX bitexact.
        let input_dim: i32 = 2112;
        let output_dim: i32 = 64;
        let bits = 6;
        let group_size = 32;
        let weight_data: Vec<f32> = (0..(input_dim * output_dim) as usize)
            .map(|i| ((i % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source = array_f32(&weight_data, &[output_dim, input_dim]);
        let q = quantize(
            &source,
            Some(group_size),
            Some(bits),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let weight = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size,
            bits,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..input_dim as usize)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 1, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let inv = invariant_projection_metal_impl(&input, &weight).expect("non-fast MLX");
        let mlx = quantized_matmul(
            &input,
            &q[0],
            &q[1],
            Some(&q[2]),
            true,
            Some(group_size),
            Some(bits),
            None,
        );
        let a = astype(&inv, MlxDtype::Float32, None);
        let b = astype(&mlx, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        // 2112 uses split (2048 qmv_fast + 64 MLX); absolute MLX match is ulp-level.
        let da = a.data_f32();
        let db = b.data_f32();
        let mut maxd = 0.0f32;
        for i in 0..da.len() {
            maxd = maxd.max((da[i] - db[i]).abs());
        }
        assert!(maxd < 1e-3, "2112 split vs MLX maxΔ={maxd}");
    }

    #[test]
    fn invariant_affine_qmv_fast_matches_mlx_singleton_and_microbatch() {
        let input_dim = 512;
        let output_dim = 64;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 251) as f32 - 125.0) * 0.00390625)
            .collect();
        let source_weight = array_f32(&weight_data, &[output_dim, input_dim]);
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 89) as f32 - 44.0) * 0.015625)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );

        for bits in [4, 6, 8] {
            let quantized = quantize(
                &source_weight,
                Some(64),
                Some(bits),
                MlxQuantizationMode::Affine,
                None,
                None,
            );
            let weight = QuantizedWeight {
                weight: quantized[0].clone(),
                scales: Some(quantized[1].clone()),
                biases: Some(quantized[2].clone()),
                group_size: 64,
                bits,
                mode: "affine".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            };
            let microbatch = invariant_projection_metal_impl(&input, &weight)
                .expect("fast invariant affine projection should support two rows");

            for row in 0..2 {
                let single = contiguous(
                    &slice(
                        &input,
                        &[0, row, 0],
                        &[1, row + 1, input_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let invariant_single = invariant_projection_metal_impl(&single, &weight)
                    .expect("fast invariant affine projection should support one row");
                let mlx_single = quantized_matmul(
                    &single,
                    &quantized[0],
                    &quantized[1],
                    Some(&quantized[2]),
                    true,
                    Some(64),
                    Some(bits),
                    None,
                );
                let actual = contiguous(
                    &slice(
                        &microbatch,
                        &[0, row, 0],
                        &[1, row + 1, output_dim],
                        &[1, 1, 1],
                        None,
                    ),
                    None,
                );
                let actual = astype(&actual, MlxDtype::Float32, None);
                let invariant_single = astype(&invariant_single, MlxDtype::Float32, None);
                let mlx_single = astype(&mlx_single, MlxDtype::Float32, None);
                eval(&[&actual, &invariant_single, &mlx_single]);
                assert_eq!(
                    actual.data_f32(),
                    invariant_single.data_f32(),
                    "microbatch row {row}, bits={bits}"
                );
                assert_eq!(
                    invariant_single.data_f32(),
                    mlx_single.data_f32(),
                    "MLX singleton parity, bits={bits}"
                );
            }
        }
    }

    #[test]
    fn invariant_dense_projection_is_bit_exact_across_microbatch_shapes() {
        let input_dim = 64;
        let output_dim = 48;
        let weight_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|index| ((index % 97) as f32 - 48.0) * 0.009765625)
            .collect();
        let weight = QuantizedWeight {
            weight: astype(
                &array_f32(&weight_data, &[output_dim, input_dim]),
                MlxDtype::Bfloat16,
                None,
            ),
            scales: None,
            biases: None,
            group_size: 0,
            bits: 16,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let input_data: Vec<f32> = (0..2 * input_dim)
            .map(|index| ((index % 37) as f32 - 18.0) * 0.0234375)
            .collect();
        let input = astype(
            &array_f32(&input_data, &[1, 2, input_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let microbatch = invariant_projection_metal_impl(&input, &weight)
            .expect("dense invariant projection should support two rows");

        for row in 0..2 {
            let single = contiguous(
                &slice(
                    &input,
                    &[0, row, 0],
                    &[1, row + 1, input_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let expected = invariant_projection_metal_impl(&single, &weight)
                .expect("dense invariant projection should support one row");
            let actual = contiguous(
                &slice(
                    &microbatch,
                    &[0, row, 0],
                    &[1, row + 1, output_dim],
                    &[1, 1, 1],
                    None,
                ),
                None,
            );
            let actual = astype(&actual, MlxDtype::Float32, None);
            let expected = astype(&expected, MlxDtype::Float32, None);
            eval(&[&actual, &expected]);
            assert_eq!(actual.data_f32(), expected.data_f32(), "row {row}");
        }
    }

    #[test]
    fn add_then_multiply_scalar_metal_matches_unfused_bf16_rounding() {
        let a = astype(
            &array_f32(&[0.333, -1.125, 2.75, 3.125], &[1, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let b = astype(
            &array_f32(&[1.875, 4.25, -2.375, 0.625], &[1, 4]),
            MlxDtype::Bfloat16,
            None,
        );
        let scalar = astype(&array_f32(&[0.3125], &[1]), MlxDtype::Bfloat16, None);

        let direct = add_then_multiply_scalar_metal_impl(&a, &b, &scalar)
            .expect("scalar fused add should support bf16 inputs");
        let reference = multiply(&add(&a, &b, None), &scalar, None);
        let direct = astype(&direct, MlxDtype::Float32, None);
        let reference = astype(&reference, MlxDtype::Float32, None);
        eval(&[&direct, &reference]);

        assert_eq!(direct.shape(), vec![1, 4]);
        assert_eq!(direct.data_f32(), reference.data_f32());
    }

    #[test]
    fn add_then_multiply_scalar_metal_rejects_broadcast_vector_scale() {
        let a = array_f32(&[1.0, 2.0], &[1, 2]);
        let b = array_f32(&[3.0, 4.0], &[1, 2]);
        let vector_scale = array_f32(&[0.5, 0.25], &[2]);

        assert!(
            add_then_multiply_scalar_metal_impl(&a, &b, &vector_scale).is_none(),
            "only exact scalar layer-scale tensors are fused"
        );
    }
}
