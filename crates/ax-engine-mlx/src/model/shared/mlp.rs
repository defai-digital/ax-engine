use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, add, argpartition_axis, argsort_axis, astype, divide, expand_dims,
    expand_dims_axes, gelu_approx_mul, gelu_approx_mul_quantized_matmul, multiply,
    quantized_matmul_rms_norm, reshape, rms_norm, silu_mul, slice_last_dim, softmax,
    softmax_precise, sum_axis, take, take_along_axis, topk_axis,
};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::fastpath;
use crate::per_layer_compile::{apply_layer_dense_ffn_decode, apply_layer_dense_ffn_prefill};
use crate::weights::{LayerWeights, QuantizedWeight};

use super::super::config::{GlmRouterConfig, ModelConfig};
use super::super::profile::{
    DecodeProfileStage, MoeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    moe_profile_enabled, prefill_profile_enabled, record_moe_profile_layer,
    record_moe_profile_stage, record_moe_profile_total,
    record_qwen_dense_ffn_gate_up_matvec_metal_attempt,
    record_qwen_dense_ffn_gate_up_matvec_metal_fallback,
    record_qwen_dense_ffn_gate_up_matvec_metal_hit, saturating_profile_us,
};
use super::utils::{
    ProjectionBatchPolicy, mlx_slice_last_dim, qkv_slices, qw, qw_gather, qw_with_policy,
    scalar_like, scale_hidden, shape_element_count, squeeze_switch_singleton,
};

static GELU_MUL_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

/// Maximum sequence length for which the fused MoE shared-expert weighted-sum
/// Metal kernel is attempted. Beyond this threshold, the weighted-sum is
/// bandwidth-bound on a large tensor, where the fused kernel's extra input read
/// costs more than the dispatch it saves.
const MOE_SHARED_FUSION_SEQ_THRESHOLD: usize = 64;

const GELU_MUL_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    T gate_v = gate[idx];
    T x_v = x[idx];
    float gate_f = static_cast<float>(gate_v);
    // gelu_approx(gate) saturates to identity (gate > 10) or zero (gate < -10)
    // out here; skip tanh in that range because the cubic inner term overflows
    // half/bfloat16 intermediates and fast-math tanh(inf) returns NaN.
    if (gate_f > 10.0f) {
        out[idx] = static_cast<T>(gate_f * static_cast<float>(x_v));
        return;
    }
    if (gate_f < -10.0f) {
        out[idx] = static_cast<T>(0.0f);
        return;
    }
    // In-range math rounds through T after every step to stay bit-identical
    // with mlx-lm's imperative op-by-op gelu_approx(gate) * x chain.
    T half_v = static_cast<T>(0.5f);
    T one_v = static_cast<T>(1.0f);
    T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
    T coeff_v = static_cast<T>(0.044715f);

    T gate2 = static_cast<T>(static_cast<float>(gate_v) * static_cast<float>(gate_v));
    T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_v));
    T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
    T inner = static_cast<T>(static_cast<float>(gate_v) + static_cast<float>(cubic));
    T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
    T t = static_cast<T>(tanh(static_cast<float>(scaled)));
    T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
    T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_v));
    T activated = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
    out[idx] = static_cast<T>(static_cast<float>(activated) * static_cast<float>(x_v));
"#;

pub(crate) fn qkv_project(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(cfg, w, x, head_dim, false, ProjectionBatchPolicy::Shared)
}

pub(crate) fn qkv_project_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(cfg, w, x, head_dim, false, ProjectionBatchPolicy::RowExact)
}

/// Embedding variant of `qkv_project`: prefers a packed-QKV single-matmul when
/// the layer has materialised `qkv_packed` weights. For split Q/K/V Qwen
/// embedding weights (the common case), still prefers split projections —
/// packing would require a runtime concat of three quantized matrices.
pub(crate) fn qkv_project_embed(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    // Prefer packed when present for any batch/seq; previously only short
    // seq<=16 batches forced packed, leaving long ingest shapes on 3-way split.
    let force_packed = w.qkv_packed.is_some();
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        force_packed,
        ProjectionBatchPolicy::Shared,
    )
}

fn qkv_project_inner(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
    force_packed: bool,
    projection_policy: ProjectionBatchPolicy,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let slices = qkv_slices(cfg, head_dim);
    let batch = x.shape().first().copied().unwrap_or(1);
    let prefer_split = !force_packed
        && projection_policy == ProjectionBatchPolicy::Shared
        && batch > 1
        && w.q_proj.is_some()
        && w.k_proj.is_some();
    if !prefer_split && let Some(packed) = &w.qkv_packed {
        let out = qw_with_policy(x, packed, projection_policy);
        let (q, gate) = if let Some((gate_start, gate_end)) = slices.gate {
            // attn_output_gate=true: the q section of the packed output preserves
            // q_proj's per-head interleaved layout `[h0_q, h0_gate, h1_q, h1_gate, ...]`,
            // so a flat slice `[0, q_size)` would mix one head's q with its gate
            // instead of yielding all heads' q. Reshape per-head and slice the
            // last dim, matching the split path below and mlx-lm's
            // `mx.split(q.reshape(B, L, n_heads, -1), 2, axis=-1)`.
            debug_assert_eq!(slices.q.0, 0);
            debug_assert_eq!(slices.q.1, gate_start);
            let seq = out.shape()[1];
            let qg = mlx_slice_last_dim(&out, 0, gate_end);
            let qg_heads = reshape(
                &qg,
                &[batch, seq, cfg.n_heads as i32, 2 * head_dim as i32],
                None,
            );
            let q = reshape(
                &slice_last_dim(&qg_heads, 0, head_dim as i32, None),
                &[batch, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            let gate = reshape(
                &slice_last_dim(&qg_heads, head_dim as i32, 2 * head_dim as i32, None),
                &[batch, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            (q, Some(gate))
        } else {
            (mlx_slice_last_dim(&out, slices.q.0, slices.q.1), None)
        };
        let k = mlx_slice_last_dim(&out, slices.k.0, slices.k.1);
        let v = mlx_slice_last_dim(&out, slices.v.0, slices.v.1);
        (q, k, v, gate)
    } else {
        let q_full = qw_with_policy(x, w.q_proj.as_ref().unwrap(), projection_policy);
        let (q, gate) = if slices.gate.is_some() {
            // attn_output_gate=true: q_proj output is [B, L, n_heads, 2*head_dim] interleaved.
            // Split by reshaping to [B, L, n_heads, 2*head_dim] and slicing last dim,
            // matching mlx-lm's `mx.split(q_proj_out.reshape(B, L, n_heads, -1), 2, axis=-1)`.
            let seq = q_full.shape()[1];
            let q_heads = reshape(
                &q_full,
                &[batch, seq, cfg.n_heads as i32, 2 * head_dim as i32],
                None,
            );
            let q = reshape(
                &slice_last_dim(&q_heads, 0, head_dim as i32, None),
                &[batch, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            let gate = reshape(
                &slice_last_dim(&q_heads, head_dim as i32, 2 * head_dim as i32, None),
                &[batch, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            (q, Some(gate))
        } else {
            // q_proj output is exactly [B, L, n_heads * head_dim] — no slice needed.
            (q_full, None)
        };
        let k = qw_with_policy(x, w.k_proj.as_ref().unwrap(), projection_policy);
        let v = w
            .v_proj
            .as_ref()
            .map(|v_proj| qw_with_policy(x, v_proj, projection_policy))
            .unwrap_or_else(|| k.clone());
        (q, k, v, gate)
    }
}

pub(crate) fn attention_output_projection(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
) -> MlxArray {
    attention_output_projection_with_policy(
        attn_flat,
        attn_gate,
        o_proj,
        ProjectionBatchPolicy::Shared,
    )
}

pub(crate) fn attention_output_projection_batched(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
) -> MlxArray {
    attention_output_projection_with_policy(
        attn_flat,
        attn_gate,
        o_proj,
        ProjectionBatchPolicy::RowExact,
    )
}

fn attention_output_projection_with_policy(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
    projection_policy: ProjectionBatchPolicy,
) -> MlxArray {
    let gated = if let Some(gate) = attn_gate {
        multiply(attn_flat, &mlx_sys::ops::sigmoid(gate, None), None)
    } else {
        attn_flat.clone()
    };
    qw_with_policy(&gated, o_proj, projection_policy)
}

/// Gemma-family GeGLU activation.
///
/// This preserves mlx-lm's `nn.gelu_approx(gate) * x` math while using AX's
/// direct MLX shim to collapse the scalar-heavy activation chain behind one
/// stable FFI call. The older compiled-closure experiment was removed from the
/// production surface because MLX compiled closures are thread/stream-registry
/// sensitive in the Rust server worker model.
pub(crate) fn geglu(gate: &MlxArray, x: &MlxArray) -> MlxArray {
    if let Some(out) = gelu_approx_mul_metal(gate, x, fastpath::geglu_mul_metal_enabled()) {
        return out;
    }
    gelu_approx_mul(gate, x, None)
}

pub(crate) fn per_layer_input_gate(gate: &MlxArray, per_layer_input: &MlxArray) -> MlxArray {
    // Keep Gemma4 per-layer input gating on the exact MLX op chain. The
    // Metal GELU approximation is close in isolation, but its small bf16
    // activation error is applied at every layer and can flip first-token
    // argmax on E4B pattern prompts.
    gelu_approx_mul(gate, per_layer_input, None)
}

fn gelu_approx_mul_metal(gate: &MlxArray, x: &MlxArray, enabled: bool) -> Option<MlxArray> {
    if !enabled {
        return None;
    }
    if gate.shape() != x.shape() || gate.dtype() != x.dtype() {
        return None;
    }
    if !matches!(
        gate.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    let shape = gate.shape();
    let element_count = shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;
    let kernel = GELU_MUL_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_gemma_gelu_mul_v4",
            &["gate", "x"],
            &["out"],
            GELU_MUL_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[gate, x],
        &[KernelOutputSpec {
            shape,
            dtype: gate.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: gate.dtype(),
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

pub(crate) fn per_layer_input_gate_project(
    gate: &MlxArray,
    per_layer_input: &MlxArray,
    proj_w: &QuantizedWeight,
) -> MlxArray {
    if let Some(scales) = proj_w.scales.as_ref() {
        return gelu_approx_mul_quantized_matmul(
            gate,
            per_layer_input,
            &proj_w.weight,
            scales,
            proj_w.biases.as_ref(),
            proj_w.group_size,
            proj_w.bits,
            None,
        );
    }
    qw(&per_layer_input_gate(gate, per_layer_input), proj_w)
}

/// SwiGLU compiled helper — mirrors `geglu()` but with SiLU activation.
/// Wraps `silu(gate) * up` in a per-thread `Mutex<HashMap<ThreadId,
/// MlxClosure>>` compile cache. Same thread-locality + fail-closed
/// contract: `try_apply` falls back to the imperative path on
/// cross-thread / stream-contract mismatch. Process-static (NOT
/// `thread_local!`) for the same SIGSEGV-at-drop reason documented on
/// `geglu()`.
pub(crate) fn swiglu(gate: &MlxArray, up: &MlxArray) -> MlxArray {
    use std::collections::HashMap;
    use std::collections::hash_map::Entry;
    use std::thread::ThreadId;

    // SwiGLU's `silu(gate) * up` op tree empirically tolerates 3D and 4D
    // inputs under the same compiled closure (verified on Qwen 3.6 35B-A3B,
    // Coder Next, GLM 4.7 Flash — all rank-mixed dense+MoE and stable),
    // unlike the `gelu_approx + multiply` tree that ABORTS — see the
    // companion comment on `geglu()`. A single per-thread closure is
    // sufficient here.
    static SWIGLU_COMPILE_CACHE: OnceLock<Mutex<HashMap<ThreadId, MlxClosure>>> = OnceLock::new();

    let cache = SWIGLU_COMPILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();
    // Graceful degradation on mutex poison: skip the compiled path and fall
    // through to the uncached `silu_mul` below. Under `panic = "abort"` a
    // poisoned mutex would otherwise crash the process.
    let outputs = if let Ok(mut guard) = cache.lock() {
        if let Entry::Vacant(slot) = guard.entry(tid)
            && let Ok(compiled) = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
                let gate = inputs.get(0);
                let up = inputs.get(1);
                vec![silu_mul(&gate, &up, None)]
            })
            .compile(true)
        {
            slot.insert(compiled);
        }
        guard
            .get(&tid)
            .and_then(|cls| cls.try_apply(&[gate, up]).ok())
    } else {
        None
    };

    if let Some(mut outputs) = outputs
        && let Some(out) = outputs.pop()
    {
        return out;
    }
    silu_mul(gate, up, None)
}

pub(crate) fn dense_ffn_activation(cfg: &ModelConfig, gate: &MlxArray, up: &MlxArray) -> MlxArray {
    if cfg.uses_geglu {
        geglu(gate, up)
    } else if fastpath::prefill_ffn_compile_swiglu_enabled() {
        swiglu(gate, up)
    } else {
        silu_mul(gate, up, None)
    }
}

fn packed_ffn_activation(
    cfg: &ModelConfig,
    gate_up: &MlxArray,
    hidden_dim: i32,
) -> Option<MlxArray> {
    if cfg.uses_geglu {
        packed_geglu_metal(gate_up, hidden_dim)
    } else {
        packed_swiglu_metal(gate_up, hidden_dim)
    }
}

static PACKED_GEGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static PACKED_SWIGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GEMMA4_MOE_WEIGHTED_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GEMMA4_MOE_WEIGHTED_SCALED_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN3_MOE_WEIGHTED_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN3_MOE_WEIGHTED_SUM_WITH_SHARED_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const PACKED_GEGLU_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint col = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint gate_idx = row * (HiddenDim * 2) + col;
    uint up_idx = gate_idx + HiddenDim;

    T gate_v = gate_up[gate_idx];
    T up_v = gate_up[up_idx];
    float gate_f = static_cast<float>(gate_v);
    // See GELU_MUL_KERNEL_SOURCE above: saturate to identity/zero outside
    // [-10, 10] to avoid fast-math tanh(inf) = NaN, and round through T at
    // every step in range to stay bit-identical with mlx-lm's imperative
    // gelu_approx(gate) * up chain.
    if (gate_f > 10.0f) {
        out[idx] = static_cast<T>(gate_f * static_cast<float>(up_v));
        return;
    }
    if (gate_f < -10.0f) {
        out[idx] = static_cast<T>(0.0f);
        return;
    }
    T half_v = static_cast<T>(0.5f);
    T one_v = static_cast<T>(1.0f);
    T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
    T coeff_v = static_cast<T>(0.044715f);

    T gate2 = static_cast<T>(static_cast<float>(gate_v) * static_cast<float>(gate_v));
    T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_v));
    T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
    T inner = static_cast<T>(static_cast<float>(gate_v) + static_cast<float>(cubic));
    T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
    T t = static_cast<T>(tanh(static_cast<float>(scaled)));
    T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
    T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_v));
    T activated = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
    out[idx] = static_cast<T>(static_cast<float>(activated) * static_cast<float>(up_v));
"#;

const PACKED_SWIGLU_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint col = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint gate_idx = row * (HiddenDim * 2) + col;
    uint up_idx = gate_idx + HiddenDim;

    float gate_v = static_cast<float>(gate_up[gate_idx]);
    float up_v = static_cast<float>(gate_up[up_idx]);
    float activated = gate_v / (1.0f + exp(-gate_v));
    out[idx] = static_cast<T>(activated * up_v);
"#;

const QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL_SOURCE: &str = r#"
    uint row = thread_position_in_grid.x / 32;
    uint lane = thread_index_in_simdgroup;
    if (row >= OutDim) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (uint packed_col = lane; packed_col < PackedCols; packed_col += 32) {
        uint gate_packed = gate_weight[row * PackedCols + packed_col];
        uint up_packed = up_weight[row * PackedCols + packed_col];
        for (uint packed_lane = 0; packed_lane < PackFactor; ++packed_lane) {
            uint input_col = packed_col * PackFactor + packed_lane;
            uint gate_q = (gate_packed >> (packed_lane * Bits)) & QuantMask;
            uint up_q = (up_packed >> (packed_lane * Bits)) & QuantMask;
            uint group = input_col / GroupSize;
            uint scale_idx = row * GroupCount + group;
            float x_v = static_cast<float>(x[input_col]);
            float gate_scale = static_cast<float>(gate_scales[scale_idx]);
            float gate_bias = static_cast<float>(gate_biases[scale_idx]);
            float up_scale = static_cast<float>(up_scales[scale_idx]);
            float up_bias = static_cast<float>(up_biases[scale_idx]);
            gate_acc = fma(x_v, static_cast<float>(gate_q) * gate_scale + gate_bias, gate_acc);
            up_acc = fma(x_v, static_cast<float>(up_q) * up_scale + up_bias, up_acc);
        }
    }

    float gate_sum = simd_sum(gate_acc);
    float up_sum = simd_sum(up_acc);
    if (lane == 0) {
        float activated = gate_sum / (1.0f + exp(-gate_sum));
        out[row] = static_cast<OutT>(activated * up_sum);
    }
"#;

const GEMMA4_MOE_WEIGHTED_SUM_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        acc += y * w;
    }

    out[idx] = static_cast<OutT>(acc);
"#;

const GEMMA4_MOE_WEIGHTED_SCALED_SUM_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        uint expert_idx = top_k_indices[weight_base + k];
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        float scale = static_cast<float>(expert_scale[expert_idx]);
        acc += y * w * scale;
    }

    out[idx] = static_cast<OutT>(acc);
"#;

const QWEN3_MOE_WEIGHTED_SUM_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        acc += y * w;
    }

    out[idx] = static_cast<OutT>(acc);
"#;

const QWEN3_MOE_WEIGHTED_SUM_WITH_SHARED_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    uint hidden_idx = idx % HiddenDim;
    uint row = idx / HiddenDim;
    uint down_base = row * TopK * HiddenDim + hidden_idx;
    uint weight_base = row * TopK;
    float acc = 0.0f;

    for (uint k = 0; k < TopK; ++k) {
        float y = static_cast<float>(down_out[down_base + k * HiddenDim]);
        float w = static_cast<float>(top_k_weights[weight_base + k]);
        acc += y * w;
    }
    acc += static_cast<float>(shared_out[idx]);

    out[idx] = static_cast<OutT>(acc);
"#;

fn qwen3_moe_weighted_sum_metal(
    down_out: &MlxArray,
    top_k_weights: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !matches!(
        down_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        top_k_weights.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        output_dtype,
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }

    let down_shape = down_out.shape();
    let weights_shape = top_k_weights.shape();
    if down_shape.len() != weights_shape.len() + 1 || weights_shape.is_empty() {
        return None;
    }
    let hidden_dim = *down_shape.last()?;
    let top_k = *weights_shape.last()?;
    if top_k <= 0 || hidden_dim <= 0 {
        return None;
    }
    if down_shape[..down_shape.len() - 1] != weights_shape[..] {
        return None;
    }

    let mut out_shape = weights_shape[..weights_shape.len() - 1].to_vec();
    out_shape.push(hidden_dim);
    let element_count = out_shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = QWEN3_MOE_WEIGHTED_SUM_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen3_moe_weighted_sum_v1",
            &["down_out", "top_k_weights"],
            &["out"],
            QWEN3_MOE_WEIGHTED_SUM_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[down_out, top_k_weights],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: top_k,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
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

/// Weighted-sum kernel variant that fuses the shared-expert add.
/// Equivalent to `qwen3_moe_weighted_sum_metal(down_out, top_k_weights, dtype)`
/// followed by `add(out, shared_out)`, but in a single Metal dispatch.
fn qwen3_moe_weighted_sum_with_shared_metal(
    down_out: &MlxArray,
    top_k_weights: &MlxArray,
    shared_out: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !matches!(
        down_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        top_k_weights.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        shared_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        output_dtype,
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }

    let down_shape = down_out.shape();
    let weights_shape = top_k_weights.shape();
    let shared_shape = shared_out.shape();
    if down_shape.len() != weights_shape.len() + 1 || weights_shape.is_empty() {
        return None;
    }
    let hidden_dim = *down_shape.last()?;
    let top_k = *weights_shape.last()?;
    if top_k <= 0 || hidden_dim <= 0 {
        return None;
    }
    if down_shape[..down_shape.len() - 1] != weights_shape[..] {
        return None;
    }
    // shared_out must match the output shape [.., hidden_dim] (weights minus top_k dim).
    let expected_shared_shape = &weights_shape[..weights_shape.len() - 1];
    let mut expected_shared_with_hidden = expected_shared_shape.to_vec();
    expected_shared_with_hidden.push(hidden_dim);
    if shared_shape != expected_shared_with_hidden {
        return None;
    }

    let mut out_shape = weights_shape[..weights_shape.len() - 1].to_vec();
    out_shape.push(hidden_dim);
    let element_count = out_shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = QWEN3_MOE_WEIGHTED_SUM_WITH_SHARED_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen3_moe_weighted_sum_with_shared_v1",
            &["down_out", "top_k_weights", "shared_out"],
            &["out"],
            QWEN3_MOE_WEIGHTED_SUM_WITH_SHARED_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[down_out, top_k_weights, shared_out],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: top_k,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
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

fn packed_geglu_metal(gate_up: &MlxArray, hidden_dim: i32) -> Option<MlxArray> {
    if !fastpath::dense_geglu_packed_metal_enabled() {
        return None;
    }
    packed_geglu_metal_impl(gate_up, hidden_dim)
}

fn packed_swiglu_metal(gate_up: &MlxArray, hidden_dim: i32) -> Option<MlxArray> {
    if !fastpath::dense_swiglu_packed_metal_enabled() {
        return None;
    }
    packed_swiglu_metal_impl(gate_up, hidden_dim)
}

fn packed_geglu_metal_impl(gate_up: &MlxArray, hidden_dim: i32) -> Option<MlxArray> {
    packed_glu_metal_impl(
        gate_up,
        hidden_dim,
        &PACKED_GEGLU_KERNEL,
        "ax_gemma_packed_geglu_v4",
        PACKED_GEGLU_KERNEL_SOURCE,
    )
}

fn packed_swiglu_metal_impl(gate_up: &MlxArray, hidden_dim: i32) -> Option<MlxArray> {
    packed_glu_metal_impl(
        gate_up,
        hidden_dim,
        &PACKED_SWIGLU_KERNEL,
        "ax_qwen_packed_swiglu_v1",
        PACKED_SWIGLU_KERNEL_SOURCE,
    )
}

fn packed_glu_metal_impl(
    gate_up: &MlxArray,
    hidden_dim: i32,
    kernel_cell: &'static OnceLock<MlxMetalKernel>,
    kernel_name: &'static str,
    kernel_source: &'static str,
) -> Option<MlxArray> {
    if !matches!(
        gate_up.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    if hidden_dim <= 0 {
        return None;
    }
    let shape = gate_up.shape();
    let last_dim = *shape.last()?;
    if last_dim != hidden_dim.saturating_mul(2) {
        return None;
    }
    let mut out_shape = shape;
    *out_shape.last_mut()? = hidden_dim;
    let element_count = out_shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = kernel_cell.get_or_init(|| {
        MlxMetalKernel::new(kernel_name, &["gate_up"], &["out"], kernel_source, "", true)
    });
    let mut outputs = kernel.apply_with_template(
        &[gate_up],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: gate_up.dtype(),
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: gate_up.dtype(),
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
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

fn qwen_dense_ffn_gate_up_swiglu_metal(
    cfg: &ModelConfig,
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<MlxArray> {
    if !fastpath::qwen_dense_ffn_gate_up_matvec_metal_enabled()
        || cfg.uses_geglu
        || !cfg.model_family.starts_with("qwen")
    {
        return None;
    }
    let x_shape = x.shape();
    let leading_elements = x_shape[..x_shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if leading_elements != 1 {
        return None;
    }
    record_qwen_dense_ffn_gate_up_matvec_metal_attempt();
    let out = qwen_dense_ffn_gate_up_swiglu_metal_impl(x, gate, up);
    if out.is_some() {
        record_qwen_dense_ffn_gate_up_matvec_metal_hit();
    } else {
        record_qwen_dense_ffn_gate_up_matvec_metal_fallback();
    }
    out
}

fn qwen_dense_ffn_gate_up_swiglu_metal_impl(
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<MlxArray> {
    if !matches!(
        x.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }

    let x_shape = x.shape();
    let input_dim = *x_shape.last()?;
    if input_dim <= 0 {
        return None;
    }
    let leading_elements = x_shape[..x_shape.len().saturating_sub(1)]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if leading_elements != 1 {
        return None;
    }

    let (Some(gate_scales), Some(gate_biases), Some(up_scales), Some(up_biases)) = (
        gate.scales.as_ref(),
        gate.biases.as_ref(),
        up.scales.as_ref(),
        up.biases.as_ref(),
    ) else {
        return None;
    };
    if gate.bits != up.bits || gate.group_size != up.group_size {
        return None;
    }
    if gate.bits != 4 || gate.group_size <= 0 {
        return None;
    }

    let gate_weight_shape = gate.weight.shape();
    let up_weight_shape = up.weight.shape();
    if gate_weight_shape.len() != 2 || gate_weight_shape != up_weight_shape {
        return None;
    }
    let out_dim = gate_weight_shape[0];
    let packed_cols = gate_weight_shape[1];
    if out_dim <= 0 || packed_cols <= 0 {
        return None;
    }

    let pack_factor = 32 / gate.bits;
    if packed_cols.checked_mul(pack_factor)? != input_dim {
        return None;
    }
    if input_dim % gate.group_size != 0 {
        return None;
    }
    let group_count = input_dim / gate.group_size;
    let expected_sidecar_shape = vec![out_dim, group_count];
    if gate_scales.shape() != expected_sidecar_shape
        || gate_biases.shape() != expected_sidecar_shape
        || up_scales.shape() != expected_sidecar_shape
        || up_biases.shape() != expected_sidecar_shape
    {
        return None;
    }

    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    let element_count = out_dim;
    let quant_mask = (1_i32 << gate.bits) - 1;
    let kernel = QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_dense_ffn_gate_up_swiglu_simd_v1",
            &[
                "x",
                "gate_weight",
                "gate_scales",
                "gate_biases",
                "up_weight",
                "up_scales",
                "up_biases",
            ],
            &["out"],
            QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                x,
                &gate.weight,
                gate_scales,
                gate_biases,
                &up.weight,
                up_scales,
                up_biases,
            ],
            &[KernelOutputSpec {
                shape: out_shape,
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
                    name: "PackedCols",
                    value: packed_cols,
                },
                KernelTemplateArg::Int {
                    name: "GroupSize",
                    value: gate.group_size,
                },
                KernelTemplateArg::Int {
                    name: "GroupCount",
                    value: group_count,
                },
                KernelTemplateArg::Int {
                    name: "Bits",
                    value: gate.bits,
                },
                KernelTemplateArg::Int {
                    name: "PackFactor",
                    value: pack_factor,
                },
                KernelTemplateArg::Int {
                    name: "QuantMask",
                    value: quant_mask,
                },
            ],
            (element_count.saturating_mul(32), 1, 1),
            (32, 1, 1),
            None,
        )
        .ok()?;
    outputs.pop()
}

// ---------------------------------------------------------------------------
// D2: Fused MoE expert block kernel — decode-only.
//
// Fuses activation (SwiGLU/GeGLU) + squeeze + unsort into a single Metal
// dispatch.  Replaces the chain: packed_swiglu_metal_impl →
// squeeze_switch_singleton → gather_inputs.unsort() with one kernel call.
// The output is the hidden tensor in original (unsorted) expert order,
// ready for the down-projection gather_qmm or the weighted-sum kernel.
// ---------------------------------------------------------------------------

static MOE_FUSED_ACTIVATION_UNSORT_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const MOE_FUSED_ACTIVATION_UNSORT_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    // Layout: out[original_k][d] where idx = original_k * HiddenDim + d.
    uint hidden_idx = idx % HiddenDim;
    uint orig_k = idx / HiddenDim;

    // Map original expert position → sorted position via inv_order.
    uint sorted_k = inv_order[orig_k];

    // Read gate and up from the packed gate_up output at sorted position.
    // `T` is the input dtype template (bf16/f16/f32), matching whichever
    // dtype the gate_up array was passed in as.
    uint gate_up_base = sorted_k * TwoExpertSize;
    T gate_v = gate_up[gate_up_base + hidden_idx];
    T up_v = gate_up[gate_up_base + HiddenDim + hidden_idx];

    float activated;
    // `USE_GEGLU` is a compile-time bool *template parameter* (bound via
    // KernelTemplateArg::Bool), not a preprocessor macro — a preprocessor
    // `#if USE_GEGLU` cannot see it (the preprocessor runs before template
    // substitution, so an unset macro name is always 0), which silently
    // always took the SwiGLU branch regardless of the configured activation.
    // `if constexpr` on the template parameter is the correct, MLX-idiomatic
    // way to specialize on it (see fp_quantized_nax.h in the vendored MLX
    // kernels for the same pattern).
    if constexpr (USE_GEGLU) {
        // GeGLU: gelu_approx(gate) * up. Saturate outside [-10, 10] to avoid
        // fast-math tanh(inf) = NaN, and round through T at every step in
        // range to stay bit-identical with mlx-lm's imperative
        // gelu_approx(gate) * up chain (see GELU_MUL_KERNEL_SOURCE above).
        float gate_f = static_cast<float>(gate_v);
        if (gate_f > 10.0f) {
            activated = gate_f * static_cast<float>(up_v);
        } else if (gate_f < -10.0f) {
            activated = 0.0f;
        } else {
            T half_v = static_cast<T>(0.5f);
            T one_v = static_cast<T>(1.0f);
            T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
            T coeff_v = static_cast<T>(0.044715f);

            T gate2 = static_cast<T>(static_cast<float>(gate_v) * static_cast<float>(gate_v));
            T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_v));
            T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
            T inner = static_cast<T>(static_cast<float>(gate_v) + static_cast<float>(cubic));
            T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
            T t = static_cast<T>(tanh(static_cast<float>(scaled)));
            T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
            T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_v));
            T activated_t = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
            activated = static_cast<float>(static_cast<T>(static_cast<float>(activated_t) * static_cast<float>(up_v)));
        }
    } else {
        // SwiGLU: silu(gate) * up.
        float gate_v_f = static_cast<float>(gate_v);
        float up_v_f = static_cast<float>(up_v);
        float sigmoid = 1.0f / (1.0f + exp(-gate_v_f));
        activated = (gate_v_f * sigmoid) * up_v_f;
    }

    out[idx] = static_cast<OutT>(activated);
"#;

/// Fused activation + squeeze + unsort for MoE decode (seq==1).
///
/// Takes the packed gate_up output `[1, 1, TopK_sorted, 2*ExpertSize]` and
/// produces the hidden state `[1, 1, TopK_original, ExpertSize]` with the
/// activation (SwiGLU or GeGLU) applied and the expert positions unsorted
/// back to their original order. Eliminates 3 separate dispatches.
fn moe_fused_activation_unsort_metal(
    gate_up_out: &MlxArray,
    inv_order: &MlxArray,
    hidden_dim: i32,
    top_k: i32,
    output_dtype: MlxDtype,
    uses_geglu: bool,
) -> Option<MlxArray> {
    if !matches!(
        gate_up_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        output_dtype,
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }
    if hidden_dim <= 0 || top_k <= 0 {
        return None;
    }
    let element_count = top_k.checked_mul(hidden_dim)?;

    let two_expert_size = hidden_dim.checked_mul(2)?;

    let kernel = MOE_FUSED_ACTIVATION_UNSORT_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_moe_fused_activation_unsort_v4",
            &["gate_up", "inv_order"],
            &["out"],
            MOE_FUSED_ACTIVATION_UNSORT_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[gate_up_out, inv_order],
        &[KernelOutputSpec {
            shape: vec![1, 1, top_k, hidden_dim],
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "T",
                dtype: gate_up_out.dtype(),
            },
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
            },
            KernelTemplateArg::Int {
                name: "TwoExpertSize",
                value: two_expert_size,
            },
            KernelTemplateArg::Int {
                name: "ElementCount",
                value: element_count,
            },
            KernelTemplateArg::Bool {
                name: "USE_GEGLU",
                value: uses_geglu,
            },
        ],
        (element_count, 1, 1),
        (256, 1, 1),
        None,
    );
    outputs.pop()
}

fn gemma4_moe_weighted_sum_metal(
    down_out: &MlxArray,
    top_k_weights: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !matches!(
        down_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        top_k_weights.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        output_dtype,
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) {
        return None;
    }

    let down_shape = down_out.shape();
    let weights_shape = top_k_weights.shape();
    if down_shape.len() != weights_shape.len() + 1 || weights_shape.is_empty() {
        return None;
    }
    let hidden_dim = *down_shape.last()?;
    let top_k = *weights_shape.last()?;
    if top_k <= 0 || hidden_dim <= 0 {
        return None;
    }
    if down_shape[..down_shape.len() - 1] != weights_shape[..] {
        return None;
    }

    let mut out_shape = weights_shape[..weights_shape.len() - 1].to_vec();
    out_shape.push(hidden_dim);
    let element_count = out_shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = GEMMA4_MOE_WEIGHTED_SUM_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_gemma4_moe_weighted_sum_v1",
            &["down_out", "top_k_weights"],
            &["out"],
            GEMMA4_MOE_WEIGHTED_SUM_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[down_out, top_k_weights],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: top_k,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
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

fn gemma4_moe_weighted_scaled_sum_metal(
    down_out: &MlxArray,
    top_k_weights: &MlxArray,
    top_k_indices: &MlxArray,
    expert_scale: &MlxArray,
    output_dtype: MlxDtype,
) -> Option<MlxArray> {
    if !matches!(
        down_out.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || !matches!(
        top_k_weights.dtype(),
        MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
    ) || top_k_indices.dtype() != MlxDtype::Uint32
        || !matches!(
            expert_scale.dtype(),
            MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
        )
        || !matches!(
            output_dtype,
            MlxDtype::Bfloat16 | MlxDtype::Float16 | MlxDtype::Float32
        )
    {
        return None;
    }

    let down_shape = down_out.shape();
    let weights_shape = top_k_weights.shape();
    if weights_shape != top_k_indices.shape()
        || down_shape.len() != weights_shape.len() + 1
        || weights_shape.is_empty()
        || expert_scale.shape().len() != 1
    {
        return None;
    }
    let hidden_dim = *down_shape.last()?;
    let top_k = *weights_shape.last()?;
    if top_k <= 0 || hidden_dim <= 0 {
        return None;
    }
    if down_shape[..down_shape.len() - 1] != weights_shape[..] {
        return None;
    }

    let mut out_shape = weights_shape[..weights_shape.len() - 1].to_vec();
    out_shape.push(hidden_dim);
    let element_count = out_shape
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    let element_count = i32::try_from(element_count).ok()?;

    let kernel = GEMMA4_MOE_WEIGHTED_SCALED_SUM_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_gemma4_moe_weighted_scaled_sum_v1",
            &["down_out", "top_k_weights", "top_k_indices", "expert_scale"],
            &["out"],
            GEMMA4_MOE_WEIGHTED_SCALED_SUM_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let mut outputs = kernel.apply_with_template(
        &[down_out, top_k_weights, top_k_indices, expert_scale],
        &[KernelOutputSpec {
            shape: out_shape,
            dtype: output_dtype,
        }],
        &[
            KernelTemplateArg::Dtype {
                name: "OutT",
                dtype: output_dtype,
            },
            KernelTemplateArg::Int {
                name: "TopK",
                value: top_k,
            },
            KernelTemplateArg::Int {
                name: "HiddenDim",
                value: hidden_dim,
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

pub(crate) fn ffn_swiglu(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
) -> MlxArray {
    ffn_swiglu_with_policy(
        cfg,
        w,
        x,
        post_norm,
        layer_idx,
        ProjectionBatchPolicy::Shared,
    )
}

pub(crate) fn ffn_swiglu_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
) -> MlxArray {
    ffn_swiglu_with_policy(
        cfg,
        w,
        x,
        post_norm,
        layer_idx,
        ProjectionBatchPolicy::RowExact,
    )
}

fn ffn_swiglu_with_policy(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
    projection_policy: ProjectionBatchPolicy,
) -> MlxArray {
    let seq = x.shape().get(1).copied().unwrap_or(1);
    let leading_elements = x.shape()[..x.shape().len().saturating_sub(1)]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))
        .unwrap_or(0);
    let profile_decode = seq == 1 && leading_elements == 1 && decode_profile_enabled();
    let profile_prefill = seq > 1 && prefill_profile_enabled();
    // Insert the rotation per `AX_MLX_EXPERIMENTAL_WEIGHT_ROTATION` mode:
    //   Enable mode (P1):  R(R(x)) ≈ x (identity sandwich)
    //   Apply  mode (P2a): R(x), expects offline-rotated weights to cancel
    // When Apply mode is paired with the AWQ-lite smoothing vector from
    // `--smoothing weight_mag` (P2b §3a), the per-input-channel multiplication
    // by `1/s` runs AFTER the rotation. The offline tool baked `* s` into
    // both gate_proj and up_proj rotated weights, so `R(x) * (1/s)` against
    // `(W @ R) * s` matmuls cancels back to W @ x.
    let rotated = crate::weight_rotation::maybe_apply_rotation_identity(x);
    let smoothed = if let Some(smoothing_inv) = w.rotation_smoothing_inverse.as_ref() {
        mlx_sys::ops::multiply(&rotated, smoothing_inv, None)
    } else {
        rotated
    };
    let x = &smoothed;

    // Compiled dense FFN (SwiGLU + packed gate_up only). GEGLU's gelu_approx
    // tree aborts under MLX compilation. Decode uses shapeless=true (seq=1);
    // prefill uses a fixed-shape per-geometry cache (see
    // `apply_layer_dense_ffn_prefill`).
    if !cfg.uses_geglu
        && let Some(packed) = &w.gate_up_packed
    {
        let packed_dim = packed
            .weight
            .shape()
            .first()
            .copied()
            .expect("packed FFN weight must have an output dimension");
        let half_dim = packed_dim / 2;
        let down_qw = w.down_proj.as_ref();
        let (inputs, schema) = flatten_dense_ffn_inputs(x, Some(packed), down_qw, post_norm);
        let input_refs: Vec<&MlxArray> = inputs.iter().collect();
        let eps = cfg.rms_norm_eps;
        let body = move |inputs: &MlxVectorArray| {
            let x = inputs.get(0);
            let (gate_up_qw, down_qw, post_norm) = schema.rebuild(inputs);
            let gate_up = gate_up_qw
                .as_ref()
                .expect("dense FFN compile: gate_up weight required");
            let gate_up_out = qw_with_policy(&x, gate_up, projection_policy);
            let gate = slice_last_dim(&gate_up_out, 0, half_dim, None);
            let up = slice_last_dim(&gate_up_out, half_dim, half_dim * 2, None);
            let ffn_hidden = silu_mul(&gate, &up, None);
            let down = down_qw
                .as_ref()
                .expect("dense FFN compile: down weight required");
            let out = qw_with_policy(&ffn_hidden, down, projection_policy);
            if let Some(norm_w) = post_norm {
                vec![rms_norm(&out, Some(&norm_w), eps, None)]
            } else {
                vec![out]
            }
        };
        let compiled_result = if seq == 1
            && leading_elements == 1
            && fastpath::dense_ffn_compile_enabled()
        {
            apply_layer_dense_ffn_decode(cfg.compile_cache_identity, layer_idx, &input_refs, body)
        } else if seq > 1 && leading_elements > 1 && fastpath::dense_ffn_compile_prefill_enabled() {
            apply_layer_dense_ffn_prefill(
                cfg.compile_cache_identity,
                layer_idx,
                leading_elements,
                &input_refs,
                body,
            )
        } else {
            None
        };
        if let Some(result) = compiled_result.and_then(|r| r.into_iter().next()) {
            return result;
        }
    }

    let gate_up_started = Instant::now();
    let packed_gate_up: Option<MlxArray>;
    let mut gate_up_profile_recorded = false;
    let qwen_dense_ffn = !cfg.uses_geglu && cfg.model_family.starts_with("qwen");
    // Decode keeps the split route so the opt-in Qwen dense FFN matvec Metal
    // kernel and any split-only decode compile path can engage. Prefill (and
    // multi-token verify) prefer packed gate/up when the loader materialised
    // it: one quantized matmul + packed SwiGLU Metal is the better prefill
    // geometry and is a major direct-mode TTFT lever on Qwen 3.6 27B.
    let prefer_split_gate_up = qwen_dense_ffn
        && seq == 1
        && leading_elements == 1
        && w.gate_proj.is_some()
        && w.up_proj.is_some();
    let (gate_out, up_out) = if !prefer_split_gate_up && let Some(packed) = &w.gate_up_packed {
        let out = qw_with_policy(x, packed, projection_policy);
        let packed_dim = out
            .shape()
            .last()
            .copied()
            .expect("packed FFN output must have a last dimension");
        assert!(
            packed_dim > 0 && packed_dim % 2 == 0,
            "packed FFN output last dimension must be positive and even, got {packed_dim}"
        );
        let half = packed_dim / 2;
        if cfg.uses_geglu {
            // Profiling should add barriers around the production graph, not
            // silently fall back to the split GEGLU route. Otherwise the
            // decode-profile Candidate Gate ranks a path that production does
            // not use when packed GeGLU Metal is enabled.
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnGateUp,
                gate_up_started,
                &[&out],
            );
            gate_up_profile_recorded = profile_decode || profile_prefill;
            let activation_started = Instant::now();
            if let Some(ffn_hidden) = packed_geglu_metal(&out, half) {
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnActivation,
                    activation_started,
                    &[&ffn_hidden],
                );
                let down_started = Instant::now();
                let down = w
                    .down_proj
                    .as_ref()
                    .expect("dense FFN layer must have down_proj");
                if let Some(norm_w) = post_norm {
                    if !profile_decode
                        && !profile_prefill
                        && projection_policy == ProjectionBatchPolicy::Shared
                        && fastpath::dense_qmatmul_rms_norm_enabled()
                        && let Some(scales) = down.scales.as_ref()
                    {
                        let out = quantized_matmul_rms_norm(
                            &ffn_hidden,
                            &down.weight,
                            scales,
                            down.biases.as_ref(),
                            down.group_size,
                            down.bits,
                            norm_w,
                            cfg.rms_norm_eps,
                            None,
                        );
                        forward_profile_eval_elapsed(
                            profile_decode,
                            profile_prefill,
                            DecodeProfileStage::PostAttnFfnDown,
                            down_started,
                            &[&out],
                        );
                        return out;
                    }
                    let out = qw_with_policy(&ffn_hidden, down, projection_policy);
                    forward_profile_eval_elapsed(
                        profile_decode,
                        profile_prefill,
                        DecodeProfileStage::PostAttnFfnDown,
                        down_started,
                        &[&out],
                    );
                    return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
                }
                let out = qw_with_policy(&ffn_hidden, down, projection_policy);
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnDown,
                    down_started,
                    &[&out],
                );
                return out;
            }
        } else {
            // Same packed-projection fast path as GEGLU, but with Qwen-family
            // SwiGLU math. If the Metal kernel rejects the shape/dtype, fall
            // through to the existing split + compiled/fallback SwiGLU path.
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnGateUp,
                gate_up_started,
                &[&out],
            );
            gate_up_profile_recorded = profile_decode || profile_prefill;
            let activation_started = Instant::now();
            if let Some(ffn_hidden) = packed_swiglu_metal(&out, half) {
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnActivation,
                    activation_started,
                    &[&ffn_hidden],
                );
                let down_started = Instant::now();
                let down = w
                    .down_proj
                    .as_ref()
                    .expect("dense FFN layer must have down_proj");
                if let Some(norm_w) = post_norm {
                    if !profile_decode
                        && !profile_prefill
                        && projection_policy == ProjectionBatchPolicy::Shared
                        && fastpath::dense_qmatmul_rms_norm_enabled()
                        && let Some(scales) = down.scales.as_ref()
                    {
                        let out = quantized_matmul_rms_norm(
                            &ffn_hidden,
                            &down.weight,
                            scales,
                            down.biases.as_ref(),
                            down.group_size,
                            down.bits,
                            norm_w,
                            cfg.rms_norm_eps,
                            None,
                        );
                        forward_profile_eval_elapsed(
                            profile_decode,
                            profile_prefill,
                            DecodeProfileStage::PostAttnFfnDown,
                            down_started,
                            &[&out],
                        );
                        return out;
                    }
                    let out = qw_with_policy(&ffn_hidden, down, projection_policy);
                    forward_profile_eval_elapsed(
                        profile_decode,
                        profile_prefill,
                        DecodeProfileStage::PostAttnFfnDown,
                        down_started,
                        &[&out],
                    );
                    return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
                }
                let out = qw_with_policy(&ffn_hidden, down, projection_policy);
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnDown,
                    down_started,
                    &[&out],
                );
                return out;
            }
        }
        packed_gate_up = Some(out.clone());
        let gate = mlx_slice_last_dim(&out, 0, half);
        let up = mlx_slice_last_dim(&out, half, half * 2);
        (gate, up)
    } else {
        let gate_w = w.gate_proj.as_ref().unwrap();
        let up_w = w.up_proj.as_ref().unwrap();
        if let Some(ffn_hidden) = qwen_dense_ffn_gate_up_swiglu_metal(cfg, x, gate_w, up_w) {
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnGateUp,
                gate_up_started,
                &[&ffn_hidden],
            );
            let activation_started = Instant::now();
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnActivation,
                activation_started,
                &[&ffn_hidden],
            );
            let down_started = Instant::now();
            let down = w
                .down_proj
                .as_ref()
                .expect("dense FFN layer must have down_proj");
            if let Some(norm_w) = post_norm {
                if !profile_decode
                    && !profile_prefill
                    && projection_policy == ProjectionBatchPolicy::Shared
                    && fastpath::dense_qmatmul_rms_norm_enabled()
                    && let Some(scales) = down.scales.as_ref()
                {
                    let out = quantized_matmul_rms_norm(
                        &ffn_hidden,
                        &down.weight,
                        scales,
                        down.biases.as_ref(),
                        down.group_size,
                        down.bits,
                        norm_w,
                        cfg.rms_norm_eps,
                        None,
                    );
                    forward_profile_eval_elapsed(
                        profile_decode,
                        profile_prefill,
                        DecodeProfileStage::PostAttnFfnDown,
                        down_started,
                        &[&out],
                    );
                    return out;
                }
                let out = qw_with_policy(&ffn_hidden, down, projection_policy);
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnDown,
                    down_started,
                    &[&out],
                );
                return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
            }
            let out = qw_with_policy(&ffn_hidden, down, projection_policy);
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnDown,
                down_started,
                &[&out],
            );
            return out;
        }
        packed_gate_up = None;
        let gate = qw_with_policy(x, gate_w, projection_policy);
        let up = qw_with_policy(x, up_w, projection_policy);
        (gate, up)
    };
    if (profile_decode || profile_prefill) && !gate_up_profile_recorded {
        let gate_up_profile_storage;
        let gate_up_profile_refs = if let Some(packed) = packed_gate_up.as_ref() {
            gate_up_profile_storage = vec![packed];
            gate_up_profile_storage.as_slice()
        } else {
            gate_up_profile_storage = vec![&gate_out, &up_out];
            gate_up_profile_storage.as_slice()
        };
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnGateUp,
            gate_up_started,
            gate_up_profile_refs,
        );
    }

    // Gemma4 uses GEGLU with fast-approx GELU gate (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate).
    //
    // Gemma4 uses the direct MLX GeGLU shim. It preserves mlx-lm's activation
    // math without the server-thread stream hazards of the removed compiled
    // closure experiment.
    let activation_started = Instant::now();
    let ffn_hidden = dense_ffn_activation(cfg, &gate_out, &up_out);
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::PostAttnFfnActivation,
        activation_started,
        &[&ffn_hidden],
    );
    let down_started = Instant::now();
    let down = w
        .down_proj
        .as_ref()
        .expect("dense FFN layer must have down_proj");
    if let Some(norm_w) = post_norm {
        if !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::dense_qmatmul_rms_norm_enabled()
            && let Some(scales) = down.scales.as_ref()
        {
            let out = quantized_matmul_rms_norm(
                &ffn_hidden,
                &down.weight,
                scales,
                down.biases.as_ref(),
                down.group_size,
                down.bits,
                norm_w,
                cfg.rms_norm_eps,
                None,
            );
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::PostAttnFfnDown,
                down_started,
                &[&out],
            );
            return out;
        }
        let out = qw_with_policy(&ffn_hidden, down, projection_policy);
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnDown,
            down_started,
            &[&out],
        );
        return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
    }
    let out = qw_with_policy(&ffn_hidden, down, projection_policy);
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::PostAttnFfnDown,
        down_started,
        &[&out],
    );
    out
}

pub(crate) fn shared_expert_forward(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let hidden = if let Some(packed) = w.shared_gate_up_proj.as_ref() {
        let gate_up = qw(x, packed);
        let packed_dim = gate_up
            .shape()
            .last()
            .copied()
            .expect("packed shared expert output must have a last dimension");
        assert!(
            packed_dim > 0 && packed_dim % 2 == 0,
            "packed shared expert output last dimension must be positive and even, got {packed_dim}"
        );
        let half = packed_dim / 2;
        if let Some(hidden) = packed_ffn_activation(cfg, &gate_up, half) {
            hidden
        } else {
            let gate = mlx_slice_last_dim(&gate_up, 0, half);
            let up = mlx_slice_last_dim(&gate_up, half, half * 2);
            dense_ffn_activation(cfg, &gate, &up)
        }
    } else {
        let gate = qw(
            x,
            w.shared_gate_proj
                .as_ref()
                .expect("shared expert must have gate projection"),
        );
        let up = qw(
            x,
            w.shared_up_proj
                .as_ref()
                .expect("shared expert must have up projection"),
        );
        dense_ffn_activation(cfg, &gate, &up)
    };
    let shared = qw(
        &hidden,
        w.shared_down_proj
            .as_ref()
            .expect("shared expert must have down projection"),
    );
    if let Some(shared_expert_gate) = &w.shared_expert_gate {
        let shared_gate = qw(x, shared_expert_gate);
        multiply(&mlx_sys::ops::sigmoid(&shared_gate, None), &shared, None)
    } else {
        shared
    }
}

/// Gemma4 MoE router: rms_norm(scale * hidden) → proj → argpartition → softmax.
pub(crate) fn moe_router_gemma4(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w
        .router_proj
        .as_ref()
        .expect("Gemma4 MoE layer must have router_proj");
    let combined_scale = w
        .router_combined_scale
        .as_ref()
        .expect("Gemma4 MoE layer must have precomputed router_combined_scale");
    let normed = rms_norm(hidden, Some(combined_scale), cfg.rms_norm_eps, None);

    let expert_scores = qw(&normed, router_proj);
    let (top_k_indices, top_k_weights) = top_k_by_argpartition(
        &expert_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        true,
    );
    // Per-expert output scale is applied by the Gemma4 expert tail. Deferring
    // it lets the direct Metal weighted-sum path avoid a separate gather and
    // multiply node per decode layer.
    (top_k_indices, top_k_weights)
}

// ---------------------------------------------------------------------------
// Tier 1C: Fused MoE router kernel — decode-only.
//
// Fuses argpartition + take_along_axis + softmax + renormalize into a single
// Metal dispatch, eliminating 4-5 MLX ops per MoE layer in the narrow-softmax
// router path. Takes f32 router logits, outputs (top_k_indices, top_k_weights).
// ---------------------------------------------------------------------------

static MOE_ROUTER_FUSED_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const MOE_ROUTER_FUSED_KERNEL_SOURCE: &str = r#"
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float logits_shared[ThreadgroupSize];
    threadgroup float float_reduce[ThreadgroupSize];
    threadgroup uint idx_reduce[ThreadgroupSize];

    // Load logits into threadgroup memory.
    if (tid < NumExperts) {
        logits_shared[tid] = logits_in[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Iterative top-k selection: find the maximum among unselected experts
    // for TopK rounds. Selected experts are masked to -1e38.
    uint selected_idx[TopK];
    for (uint k = 0; k < TopK; k++) {
        float local_max = -1e38f;
        uint local_max_idx = 0;
        if (tid < NumExperts) {
            local_max = logits_shared[tid];
            local_max_idx = tid;
        }

        // Threadgroup-wide max reduction via shared memory.
        float_reduce[tid] = local_max;
        idx_reduce[tid] = local_max_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ThreadgroupSize / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float other = float_reduce[tid + stride];
                if (other > float_reduce[tid] ||
                    (other == float_reduce[tid] && idx_reduce[tid + stride] < idx_reduce[tid])) {
                    float_reduce[tid] = other;
                    idx_reduce[tid] = idx_reduce[tid + stride];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        selected_idx[k] = idx_reduce[0];

        // Mask the selected expert so it is not picked again.
        if (tid == idx_reduce[0]) {
            logits_shared[tid] = -1e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute softmax over only the top-k selected logits.
    float sel_logit = -1e38f;
    if (tid < TopK) {
        sel_logit = logits_in[selected_idx[tid]];
    }

    float max_val = sel_logit;
    float_reduce[tid] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ThreadgroupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float_reduce[tid] = max(float_reduce[tid], float_reduce[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    max_val = float_reduce[0];

    float exp_val = (tid < TopK) ? exp(sel_logit - max_val) : 0.0f;
    float_reduce[tid] = exp_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_exp = 0.0f;
    for (uint stride = ThreadgroupSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float_reduce[tid] += float_reduce[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum_exp = float_reduce[0];

    // Write outputs. The weights are always the softmax over the selected
    // top-k logits: the MLX fallback (`top_k_by_argpartition` with
    // `resoftmax=true`) normalizes over the subset regardless of
    // `moe_norm_topk_prob`, so emitting raw exponentials here would scale
    // every expert output by sum_exp.
    if (tid < TopK) {
        indices_out[tid] = selected_idx[tid];
        weights_out[tid] = exp_val / sum_exp;
    }
"#;

/// One-time kernel-source validation result. MLX evaluation is lazy, so a
/// compile error in the kernel source would otherwise surface only when the
/// decode graph is evaluated — mid-step, as a process-level MLX error rather
/// than a graceful fallback. The first dispatch is `try_eval`ed eagerly; on
/// failure the fused path is disabled for the process lifetime.
static MOE_ROUTER_FUSED_KERNEL_VALIDATED: OnceLock<bool> = OnceLock::new();

/// Fused MoE router post-matmul kernel: argpartition + softmax + renormalize
/// in one Metal dispatch.
///
/// Takes f32 router logits `[1, 1, num_experts]` and returns:
/// - `top_k_indices`: `[1, 1, top_k]` (uint32)
/// - `top_k_weights`: `[1, 1, top_k]` (f32, softmax over the selected top-k)
///
/// Decode-only (seq==1). Returns `None` if the kernel is ineligible.
fn moe_router_fused_metal(
    logits_f32: &MlxArray,
    num_experts: usize,
    top_k: usize,
) -> Option<(MlxArray, MlxArray)> {
    if !fastpath::moe_router_fused_metal_enabled() {
        return None;
    }
    if logits_f32.dtype() != MlxDtype::Float32 {
        return None;
    }
    let shape = logits_f32.shape();
    if shape.len() < 2 || shape[0] != 1 || shape[1] != 1 {
        return None;
    }

    let (indices, weights) = moe_router_fused_metal_apply(logits_f32, num_experts, top_k)?;

    let validated = *MOE_ROUTER_FUSED_KERNEL_VALIDATED.get_or_init(|| {
        match mlx_sys::transforms::try_eval(&[&indices, &weights]) {
            Ok(()) => true,
            Err(message) => {
                tracing::warn!(
                    %message,
                    "fused MoE router kernel failed validation; using MLX op fallback"
                );
                false
            }
        }
    });
    if !validated {
        return None;
    }
    Some((indices, weights))
}

/// Dispatch the fused router kernel without the fastpath gate or the one-time
/// validation. Split out so tests can exercise the kernel directly.
fn moe_router_fused_metal_apply(
    logits_f32: &MlxArray,
    num_experts: usize,
    top_k: usize,
) -> Option<(MlxArray, MlxArray)> {
    if num_experts == 0 || top_k == 0 || top_k > num_experts {
        return None;
    }
    if num_experts > 1024 {
        return None;
    }

    let tg_size: i32 = if num_experts <= 32 {
        32
    } else if num_experts <= 64 {
        64
    } else if num_experts <= 128 {
        128
    } else if num_experts <= 256 {
        256
    } else if num_experts <= 512 {
        512
    } else {
        1024
    };

    let kernel = MOE_ROUTER_FUSED_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen3_moe_router_fused_v2",
            &["logits_in"],
            &["indices_out", "weights_out"],
            MOE_ROUTER_FUSED_KERNEL_SOURCE,
            "#include <metal_stdlib>\nusing namespace metal;",
            false,
        )
    });

    let out_shape = vec![1, 1, top_k as i32];
    let mut outputs = kernel
        .try_apply_with_template(
            &[logits_f32],
            &[
                KernelOutputSpec {
                    shape: out_shape.clone(),
                    dtype: MlxDtype::Uint32,
                },
                KernelOutputSpec {
                    shape: out_shape,
                    dtype: MlxDtype::Float32,
                },
            ],
            &[
                KernelTemplateArg::Int {
                    name: "NumExperts",
                    value: num_experts as i32,
                },
                KernelTemplateArg::Int {
                    name: "TopK",
                    value: top_k as i32,
                },
                KernelTemplateArg::Int {
                    name: "ThreadgroupSize",
                    value: tg_size,
                },
            ],
            (tg_size, 1, 1),
            (tg_size, 1, 1),
            None,
        )
        .ok()?;

    if outputs.len() != 2 {
        return None;
    }
    let weights = outputs.pop()?;
    let indices = outputs.pop()?;
    Some((indices, weights))
}

/// Env-gated MoE routing trace capture (`AX_MLX_MOE_ROUTER_TRACE=<path>`):
/// appends one line per router call — `<seq>;<i0>,<i1>,...` over all tokens'
/// top-k expert indices in call order — for the ADR-037 P2 expert-overlap
/// amortization probe (`moe_gather_amortization_probe`). Capture-only
/// diagnostic: it forces an eval of the indices on every router call, so it
/// must never be enabled in serving. Unset (the default) costs one cached
/// `OnceLock` read per call.
fn maybe_trace_moe_router(indices: &MlxArray, seq: usize) {
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};
    static SINK: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let Some(sink) = SINK.get_or_init(|| {
        let path = std::env::var("AX_MLX_MOE_ROUTER_TRACE").ok()?;
        let path = path.trim();
        if path.is_empty() {
            return None;
        }
        match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(file) => Some(Mutex::new(file)),
            Err(error) => {
                eprintln!("AX_MLX_MOE_ROUTER_TRACE: cannot open {path}: {error}");
                None
            }
        }
    }) else {
        return;
    };
    let indices = if indices.dtype() == MlxDtype::Uint32 {
        indices.clone()
    } else {
        astype(indices, MlxDtype::Uint32, None)
    };
    mlx_sys::eval(&[&indices]);
    let values = indices.data_u32();
    let mut line = String::with_capacity(values.len() * 4 + 8);
    line.push_str(&seq.to_string());
    line.push(';');
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            line.push(',');
        }
        line.push_str(&value.to_string());
    }
    line.push('\n');
    if let Ok(mut file) = sink.lock() {
        let _ = file.write_all(line.as_bytes());
    }
}

/// Qwen3 MoE router: proj → softmax → pick top-k by weight value (no rms_norm).
///
/// By default (kill-switch via `AX_MLX_QWEN3_MOE_NARROW_SOFTMAX=0`), uses the
/// Gemma4-style argpartition-first pattern: argpartition on raw logits
/// (monotonic with softmax → same top-k for well-separated experts), then
/// softmax only on the selected top-k subset. This eliminates the full-width
/// `softmax_precise` over all 128–256 experts.
pub(crate) fn moe_router_qwen3(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed: &MlxArray,
) -> (MlxArray, MlxArray) {
    let (indices, weights) = moe_router_qwen3_impl(cfg, w, normed);
    maybe_trace_moe_router(
        &indices,
        normed.shape().get(1).copied().unwrap_or(1) as usize,
    );
    (indices, weights)
}

fn moe_router_qwen3_impl(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w
        .router_proj
        .as_ref()
        .expect("Qwen3 MoE layer must have router_proj");
    let logits = qw(normed, router_proj);
    let last_axis = logits.ndim() as i32 - 1;

    // Narrow softmax: argpartition on raw logits, then softmax only on the
    // top-k subset. Matches the Gemma4 router pattern. Default ON after
    // validation confirmed token-for-token equivalence with mlx-lm's
    // precise=True reference. Subset softmax equals the reference's full
    // softmax + top-k renormalize ONLY under norm_topk_prob; a config without
    // it needs the un-renormalized full-width probabilities (sum < 1), so it
    // must take the reference path below.
    if cfg.moe_norm_topk_prob && fastpath::qwen3_moe_narrow_softmax_enabled() {
        // Try fused Metal router (Tier 1C): collapses argpartition +
        // take_along_axis + softmax + renormalize into one dispatch.
        // Decode-only (seq==1); falls back to the MLX op path below. The f32
        // cast feeds only the fused kernel, so build it only when that route
        // is enabled — otherwise it is a dead graph node per router call.
        if fastpath::moe_router_fused_metal_enabled() {
            let logits_f32 = if logits.dtype() == MlxDtype::Float32 {
                logits.clone()
            } else {
                astype(&logits, MlxDtype::Float32, None)
            };
            if let Some((indices, weights)) =
                moe_router_fused_metal(&logits_f32, cfg.moe_expert_count, cfg.moe_experts_per_token)
            {
                return (indices, weights);
            }
        }

        let (top_k_indices, top_k_weights) = top_k_by_argpartition(
            &logits,
            cfg.moe_expert_count,
            cfg.moe_experts_per_token,
            true, // resoftmax only the top-k subset
        );
        let top_k_weights = if cfg.moe_norm_topk_prob {
            let sum = sum_axis(&top_k_weights, last_axis, true, None);
            mlx_sys::ops::divide(&top_k_weights, &sum, None)
        } else {
            top_k_weights
        };
        return (top_k_indices, top_k_weights);
    }

    // Default: full-width softmax_precise over all experts, then argpartition.
    // mlx-lm uses mx.softmax(..., precise=True) for all MoE routers; with bf16
    // logits and many experts (e.g. 256) the tiny round-off can flip top-k
    // rankings and corrupt output.
    let weights_all = softmax_precise(&logits, last_axis, None);
    let (top_k_indices, top_k_weights) = top_k_by_argpartition(
        &weights_all,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );
    // norm_topk_prob: renormalise top-k weights to sum to 1.
    let top_k_weights = if cfg.moe_norm_topk_prob {
        let sum = sum_axis(&top_k_weights, last_axis, true, None);
        mlx_sys::ops::divide(&top_k_weights, &sum, None)
    } else {
        top_k_weights
    };
    (top_k_indices, top_k_weights)
}

pub(crate) fn moe_router_glm(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed: &MlxArray,
) -> (MlxArray, MlxArray) {
    let logits = qw(
        normed,
        w.router_proj
            .as_ref()
            .expect("GLM MoE layer must have router projection"),
    );
    let correction_bias = w
        .router_correction_bias
        .as_ref()
        .expect("GLM MoE layer must have router correction bias");
    moe_router_glm_from_logits(cfg, &logits, correction_bias)
}

/// GLM4MoELite router: sigmoid logits + correction bias selects top-k;
/// gathered weights come from the original sigmoid scores.
pub(crate) fn moe_router_glm_from_logits(
    cfg: &ModelConfig,
    logits: &MlxArray,
    correction_bias: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router = cfg.glm_router.as_ref().expect("GLM router config");
    let last_axis = logits.ndim() as i32 - 1;
    let scores = mlx_sys::ops::sigmoid(&astype(logits, MlxDtype::Float32, None), None);
    let selection_scores = add(&scores, correction_bias, None);
    let selection_scores = glm_router_apply_group_selection(cfg, router, &selection_scores);
    let (top_k_indices, _) = top_k_by_argpartition(
        &selection_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );
    let top_k_weights = take_along_axis(&scores, &top_k_indices, last_axis, None);
    let top_k_weights = if cfg.moe_experts_per_token > 1 && cfg.moe_norm_topk_prob {
        let denominator = sum_axis(&top_k_weights, last_axis, true, None);
        let epsilon = 1e-20_f32;
        let epsilon = MlxArray::from_raw_data(
            &epsilon as *const f32 as *const u8,
            std::mem::size_of::<f32>(),
            &[1_i32],
            MlxDtype::Float32,
        );
        divide(&top_k_weights, &add(&denominator, &epsilon, None), None)
    } else {
        top_k_weights
    };
    (
        top_k_indices,
        scale_hidden(&top_k_weights, router.routed_scaling_factor),
    )
}

pub(crate) fn glm_router_apply_group_selection(
    cfg: &ModelConfig,
    router: &GlmRouterConfig,
    selection_scores: &MlxArray,
) -> MlxArray {
    if router.n_group <= 1 {
        return selection_scores.clone();
    }

    assert!(
        cfg.moe_expert_count.is_multiple_of(router.n_group),
        "GLM expert count must divide evenly across router groups"
    );
    assert!(
        router.topk_group <= router.n_group,
        "GLM topk_group must be <= n_group"
    );
    let zero_group_count = router.n_group - router.topk_group;
    if zero_group_count == 0 {
        return selection_scores.clone();
    }

    let shape = selection_scores.shape();
    assert_eq!(
        shape.len(),
        3,
        "GLM router scores must be [batch, seq, experts]"
    );
    let batch = shape[0];
    let seq = shape[1];
    let experts_per_group = cfg.moe_expert_count / router.n_group;
    assert!(
        experts_per_group >= 2,
        "GLM grouped router requires at least two experts per group"
    );

    let grouped = reshape(
        selection_scores,
        &[batch, seq, router.n_group as i32, experts_per_group as i32],
        None,
    );
    // mlx-lm uses `mx.topk(..., 2, axis=-1).sum(...)` here because only the
    // top-2 values are needed for group scoring; indices are selected later.
    let group_top2 = topk_axis(&grouped, 2, -1, None);
    let group_scores = sum_axis(&group_top2, -1, true, None);
    let group_axis = group_scores.ndim() as i32 - 2;
    let group_idx = argpartition_axis(
        &group_scores,
        (zero_group_count as i32) - 1,
        group_axis,
        None,
    );
    use mlx_sys::slice;
    let group_idx = slice(
        &group_idx,
        &[0, 0, 0, 0],
        &[batch, seq, zero_group_count as i32, 1],
        &[1, 1, 1, 1],
        None,
    );
    use mlx_sys::broadcast_to;
    let group_idx = broadcast_to(
        &group_idx,
        &[
            batch,
            seq,
            zero_group_count as i32,
            experts_per_group as i32,
        ],
        None,
    );
    use mlx_sys::put_along_axis;
    let zero = scalar_like(0.0, grouped.dtype());
    let masked = put_along_axis(&grouped, &group_idx, &zero, group_axis, None);
    reshape(&masked, &[batch, seq, cfg.moe_expert_count as i32], None)
}

/// Pick top-k elements via argpartition and optionally re-apply softmax.
pub(crate) fn top_k_by_argpartition(
    scores: &MlxArray,
    num_experts: usize,
    top_k: usize,
    resoftmax: bool,
) -> (MlxArray, MlxArray) {
    let last_axis = scores.ndim() as i32 - 1;
    let part_indices = argpartition_axis(scores, -(top_k as i32), last_axis, None);
    let top_k_indices = slice_last_dim(
        &part_indices,
        (num_experts - top_k) as i32,
        num_experts as i32,
        None,
    );
    let top_k_raw = take_along_axis(scores, &top_k_indices, last_axis, None);
    let top_k_weights = if resoftmax {
        softmax(&top_k_raw, last_axis, None)
    } else {
        top_k_raw
    };
    (top_k_indices, top_k_weights)
}

/// DeepSeek V3 router: sigmoid routing with group-based expert pre-selection.
///
/// Algorithm (matches `group_expert_select` in mlx-lm deepseek_v3.py):
///   sigmoid(logits) + correction_bias
///   → group masking (n_group > 1: zero out experts in worst n_group-topk_group groups)
///   → argpartition top-k
///   → gather original sigmoid scores (pre-bias)
///   → optionally normalise
///   → scale by routed_scaling_factor
///
/// All router arithmetic stays in f32; dtype cast happens after the weighted sum
/// in `moe_experts_forward`.
pub(crate) fn moe_router_deepseek_v3(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w
        .router_proj
        .as_ref()
        .expect("DeepSeek V3 MoE layer must have router_proj");
    let logits = qw(x, router_proj);
    let last_axis = logits.ndim() as i32 - 1;

    // sigmoid scores kept in f32 throughout all router arithmetic.
    let orig_scores = mlx_sys::ops::sigmoid(&astype(&logits, MlxDtype::Float32, None), None);

    // Selection scores: add correction bias if present.
    let selection_scores = if let Some(bias) = w.router_correction_bias.as_ref() {
        add(&orig_scores, &astype(bias, MlxDtype::Float32, None), None)
    } else {
        orig_scores.clone()
    };

    // Group-based pre-selection: zero experts in the worst (n_group - topk_group) groups.
    // For n_group=1 this is a no-op (all experts visible to top-k selection).
    let selection_scores =
        deepseek_group_expert_mask(cfg, &selection_scores, cfg.moe_n_group, cfg.moe_topk_group);

    // Top-k by argpartition on the (possibly group-masked) selection scores.
    let (top_k_indices, _) = top_k_by_argpartition(
        &selection_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );

    // Gather original (pre-bias) scores for the selected experts — still f32.
    let top_k_weights = take_along_axis(&orig_scores, &top_k_indices, last_axis, None);

    // Optionally normalise top-k weights to sum to 1 (done in f32 for precision).
    let top_k_weights = if cfg.moe_experts_per_token > 1 && cfg.moe_norm_topk_prob {
        let denominator = sum_axis(&top_k_weights, last_axis, true, None);
        divide(&top_k_weights, &denominator, None)
    } else {
        top_k_weights
    };

    // Scale by routed_scaling_factor (DeepSeek V3: 2.5, others: 1.0) — still f32.
    let scaling = cfg.moe_routed_scaling_factor;
    let top_k_weights = if (scaling - 1.0).abs() > 1e-6 {
        scale_hidden(&top_k_weights, scaling)
    } else {
        top_k_weights
    };

    // dtype cast deferred to here — after all f32 arithmetic — matching the GLM router pattern.
    let top_k_weights = astype(&top_k_weights, x.dtype(), None);

    (top_k_indices, top_k_weights)
}

/// GPT-OSS MoE router: proj → softmax over ALL experts → top-k → renormalize.
///
/// This differs from Qwen3 (argpartition-first, then softmax on top-k subset)
/// and Gemma4 (rms_norm → proj → argpartition → softmax). GPT-OSS always
/// computes full softmax first, then selects the top-k by weight magnitude,
/// then renormalizes the selected weights to sum to 1.
pub(crate) fn moe_router_gpt_oss(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed: &MlxArray,
) -> (MlxArray, MlxArray) {
    let router_proj = w
        .router_proj
        .as_ref()
        .expect("GPT-OSS MoE layer must have router_proj");
    let logits = qw(normed, router_proj);
    let last_axis = logits.ndim() as i32 - 1;

    // Full softmax over all 128 experts.
    let weights_all = softmax_precise(&logits, last_axis, None);

    // Select top-k by weight magnitude.
    let (top_k_indices, top_k_raw) = top_k_by_argpartition(
        &weights_all,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );

    // Renormalize top-k weights to sum to 1.
    let top_k_weights = if cfg.moe_experts_per_token > 1 {
        let sum = sum_axis(&top_k_raw, last_axis, true, None);
        divide(&top_k_raw, &sum, None)
    } else {
        top_k_raw
    };

    (top_k_indices, top_k_weights)
}
///
/// Matches `group_expert_select` in mlx-lm deepseek_v3.py lines 206–216:
///   scores reshaped → top-2 per group → sum → argpartition worst groups → zero them.
fn deepseek_group_expert_mask(
    cfg: &ModelConfig,
    scores: &MlxArray,
    n_group: usize,
    topk_group: usize,
) -> MlxArray {
    if n_group <= 1 {
        return scores.clone();
    }
    let zero_group_count = n_group.saturating_sub(topk_group);
    if zero_group_count == 0 {
        return scores.clone();
    }

    let shape = scores.shape();
    assert_eq!(
        shape.len(),
        3,
        "DeepSeek router scores must be [batch, seq, experts]"
    );
    let batch = shape[0];
    let seq = shape[1];
    let experts_per_group = cfg.moe_expert_count / n_group;

    // Reshape to [batch, seq, n_group, experts_per_group].
    let grouped = reshape(
        scores,
        &[batch, seq, n_group as i32, experts_per_group as i32],
        None,
    );

    // Top-2 score sum per group → [batch, seq, n_group, 1].
    // mlx-lm uses `mx.topk(..., 2, axis=-1).sum(...)` here because only the
    // top-2 values are needed for group scoring; indices are selected later.
    let group_top2 = topk_axis(&grouped, 2, -1, None);
    let group_scores = sum_axis(&group_top2, -1, true, None);

    // argpartition to find the zero_group_count worst group indices.
    let group_axis = group_scores.ndim() as i32 - 2;
    let group_idx = argpartition_axis(
        &group_scores,
        (zero_group_count as i32) - 1,
        group_axis,
        None,
    );
    use mlx_sys::slice;
    let group_idx = slice(
        &group_idx,
        &[0, 0, 0, 0],
        &[batch, seq, zero_group_count as i32, 1],
        &[1, 1, 1, 1],
        None,
    );
    use mlx_sys::broadcast_to;
    let group_idx = broadcast_to(
        &group_idx,
        &[
            batch,
            seq,
            zero_group_count as i32,
            experts_per_group as i32,
        ],
        None,
    );

    use mlx_sys::put_along_axis;
    let zero = scalar_like(0.0, grouped.dtype());
    let masked = put_along_axis(&grouped, &group_idx, &zero, group_axis, None);
    reshape(&masked, &[batch, seq, cfg.moe_expert_count as i32], None)
}

/// Expert forward: applies selected experts to `x` and returns the weighted sum.
///
/// x: [1, seq, hidden] (already pre-normed via pre_feedforward_layernorm_2)
/// top_k_indices: [1, seq, top_k]   expert assignments (uint32)
/// top_k_weights: [1, seq, top_k]   softmax-normalised weights (bf16)
pub(crate) fn moe_experts_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    moe_experts_forward_impl(cfg, w, x, top_k_indices, top_k_weights, None, None)
}

/// Expert forward with shared-expert output for fused weighted-sum (Phase 1A).
/// When `shared_expert_out` is provided and the fused Metal kernel is eligible,
/// the shared-expert add is fused into the weighted-sum kernel, eliminating one
/// `add` dispatch per layer.
pub(crate) fn moe_experts_forward_with_shared(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
    shared_expert_out: &MlxArray,
) -> MlxArray {
    moe_experts_forward_impl(
        cfg,
        w,
        x,
        top_k_indices,
        top_k_weights,
        None,
        Some(shared_expert_out),
    )
}

pub(crate) fn moe_experts_forward_gemma4(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> MlxArray {
    moe_experts_forward_impl(
        cfg,
        w,
        x,
        top_k_indices,
        top_k_weights,
        w.router_expert_scale.as_ref(),
        None,
    )
}

/// Combine Gemma4 dual-path MoE sub-blocks (dense `h1` + expert `h2`).
///
/// When the expert post-norm is absent and a final FFN post-norm is present,
/// fuses `add(h1, h2)` + RMSNorm into one `add_rms_norm_pair` (saves a
/// dispatch on the decode dual-path post stage).
pub(crate) fn combine_gemma4_dual_path_outputs(
    h1: &MlxArray,
    h2: &MlxArray,
    expert_post_norm2: Option<&MlxArray>,
    ffn_post_norm: Option<&MlxArray>,
    eps: f32,
) -> MlxArray {
    let h2 = super::rms_norm_opt(h2, expert_post_norm2, eps);
    match (expert_post_norm2.is_none(), ffn_post_norm) {
        (true, Some(post)) => {
            // Fuse residual add of dense+expert with the shared post-norm.
            let (_residual, normed) = mlx_sys::add_rms_norm_pair(h1, &h2, post, eps, None);
            normed
        }
        _ => {
            let combined = add(h1, &h2, None);
            super::rms_norm_opt(&combined, ffn_post_norm, eps)
        }
    }
}

/// Standalone MoE expert forward using individually captured weight tensors.
///
/// Used by the per-layer MoE compiled closure (`apply_layer_moe_decode`) where
/// full `LayerWeights` cannot be captured (it is not `Clone`). The caller
/// clones only the expert weight tensors into this function.
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_experts_forward_with_cloned_weights(
    cfg: &ModelConfig,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
    gate_up_exps_packed: Option<QuantizedWeight>,
    gate_exps: Option<QuantizedWeight>,
    up_exps: Option<QuantizedWeight>,
    down_exps: Option<QuantizedWeight>,
    shared_expert_out: Option<MlxArray>,
    router_expert_scale: Option<MlxArray>,
) -> MlxArray {
    let w = LayerWeights {
        attn_norm: x.clone(),
        attn_post_norm: None,
        q_norm: None,
        k_norm: None,
        q_proj: None,
        k_proj: None,
        v_proj: None,
        qkv_packed: None,
        o_proj: None,
        linear_attn: None,
        glm_mla_attn: None,
        ffn_norm: x.clone(),
        ffn_post_norm: None,
        gate_proj: None,
        up_proj: None,
        gate_up_packed: None,
        down_proj: None,
        ffn_norm2: None,
        ffn_post_norm1: None,
        ffn_post_norm2: None,
        router_proj: None,
        router_correction_bias: None,
        router_scale: None,
        router_combined_scale: None,
        router_expert_scale,
        layer_scalar: None,
        per_layer_gate: None,
        per_layer_proj_w: None,
        per_layer_post_norm: None,
        shared_expert_gate: None,
        shared_gate_up_proj: None,
        shared_gate_proj: None,
        shared_up_proj: None,
        shared_down_proj: None,
        gate_up_exps_packed,
        gate_exps,
        up_exps,
        down_exps,
        mxfp4_gate_up_exps: None,
        mxfp4_down_exps: None,
        attn_sink: None,
        rotation_smoothing_inverse: None,
    };
    moe_experts_forward_impl(
        cfg,
        &w,
        x,
        top_k_indices,
        top_k_weights,
        None,
        shared_expert_out.as_ref(),
    )
}

/// Index layout for one quantized expert weight threaded through the compiled
/// MoE decode closure as explicit inputs.
///
/// MLX-C 0.6.0 rejects compiling a function with *uncaptured inputs* — every
/// MLX array the traced graph depends on must be an explicit function input,
/// not a value captured from the enclosing Rust closure. The expert weights
/// (and the optional shared-expert output) are therefore passed positionally
/// in the input vector. `group_size`/`bits` are plain scalars and stay
/// captured in the schema.
#[derive(Clone, Copy)]
pub(crate) struct QuantInputSlot {
    weight: usize,
    scales: Option<usize>,
    biases: Option<usize>,
    group_size: i32,
    bits: i32,
}

impl QuantInputSlot {
    fn rebuild(&self, inputs: &MlxVectorArray) -> QuantizedWeight {
        QuantizedWeight {
            weight: inputs.get(self.weight),
            scales: self.scales.map(|i| inputs.get(i)),
            biases: self.biases.map(|i| inputs.get(i)),
            group_size: self.group_size,
            bits: self.bits,
        }
    }
}

/// Positional layout of the compiled MoE decode closure's input vector.
///
/// Inputs `0..=2` are always `(hidden, top_k_indices, top_k_weights)`. The
/// expert weights and the optional shared-expert output follow, in the order
/// they were pushed by [`flatten_compiled_moe_inputs`]. The schema holds only
/// `Copy` index/scalar metadata, so it is cheap to capture by the closure.
#[derive(Clone, Copy, Default)]
pub(crate) struct CompiledMoeSchema {
    gate_up: Option<QuantInputSlot>,
    gate: Option<QuantInputSlot>,
    up: Option<QuantInputSlot>,
    down: Option<QuantInputSlot>,
    shared: Option<usize>,
}

fn push_optional_input(inputs: &mut Vec<MlxArray>, arr: Option<&MlxArray>) -> Option<usize> {
    arr.map(|a| {
        let i = inputs.len();
        inputs.push(a.clone());
        i
    })
}

fn push_quant_inputs(
    inputs: &mut Vec<MlxArray>,
    q: Option<&QuantizedWeight>,
) -> Option<QuantInputSlot> {
    let q = q?;
    let weight = inputs.len();
    inputs.push(q.weight.clone());
    let scales = push_optional_input(inputs, q.scales.as_ref());
    let biases = push_optional_input(inputs, q.biases.as_ref());
    Some(QuantInputSlot {
        weight,
        scales,
        biases,
        group_size: q.group_size,
        bits: q.bits,
    })
}

/// Flatten every MLX array the MoE expert forward depends on into an explicit
/// input vector, returning the vector plus a [`CompiledMoeSchema`] that records
/// where each tensor landed. The compiled closure rebuilds its weights from the
/// schema via [`CompiledMoeSchema::rebuild`], guaranteeing the traced graph has
/// no uncaptured inputs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flatten_compiled_moe_inputs(
    hidden: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
    gate_up_exps_packed: Option<&QuantizedWeight>,
    gate_exps: Option<&QuantizedWeight>,
    up_exps: Option<&QuantizedWeight>,
    down_exps: Option<&QuantizedWeight>,
    shared_expert_out: Option<&MlxArray>,
) -> (Vec<MlxArray>, CompiledMoeSchema) {
    let mut inputs: Vec<MlxArray> =
        vec![hidden.clone(), top_k_indices.clone(), top_k_weights.clone()];
    let gate_up = push_quant_inputs(&mut inputs, gate_up_exps_packed);
    let gate = push_quant_inputs(&mut inputs, gate_exps);
    let up = push_quant_inputs(&mut inputs, up_exps);
    let down = push_quant_inputs(&mut inputs, down_exps);
    let shared = push_optional_input(&mut inputs, shared_expert_out);
    (
        inputs,
        CompiledMoeSchema {
            gate_up,
            gate,
            up,
            down,
            shared,
        },
    )
}

impl CompiledMoeSchema {
    /// Rebuild the expert weights and shared-expert output from the closure's
    /// input vector, in the same layout produced by
    /// [`flatten_compiled_moe_inputs`].
    #[allow(clippy::type_complexity)]
    pub(crate) fn rebuild(
        &self,
        inputs: &MlxVectorArray,
    ) -> (
        MlxArray,
        MlxArray,
        MlxArray,
        Option<QuantizedWeight>,
        Option<QuantizedWeight>,
        Option<QuantizedWeight>,
        Option<QuantizedWeight>,
        Option<MlxArray>,
    ) {
        (
            inputs.get(0),
            inputs.get(1),
            inputs.get(2),
            self.gate_up.map(|s| s.rebuild(inputs)),
            self.gate.map(|s| s.rebuild(inputs)),
            self.up.map(|s| s.rebuild(inputs)),
            self.down.map(|s| s.rebuild(inputs)),
            self.shared.map(|i| inputs.get(i)),
        )
    }
}

// ---------------------------------------------------------------------------
// Dense FFN compile schema (mirrors CompiledMoeSchema for MoE layers).
//
// All MLX arrays the compiled dense FFN closure depends on are threaded
// through as explicit inputs. `group_size` / `bits` / `half_dim` are
// plain scalars and stay captured in the closure.
// ---------------------------------------------------------------------------

/// Index layout for one dense FFN's weight tensors threaded through the
/// compiled closure as explicit inputs.
///
/// Mirrors [`CompiledMoeSchema`]: the gate_up and down quantized weights
/// plus the optional post-norm are passed positionally in the input vector.
pub(crate) struct CompiledDenseFfnSchema {
    gate_up: Option<QuantInputSlot>,
    down: Option<QuantInputSlot>,
    post_norm: Option<usize>,
}

impl CompiledDenseFfnSchema {
    /// Rebuild the dense FFN weights from the closure's input vector.
    pub(crate) fn rebuild(
        &self,
        inputs: &MlxVectorArray,
    ) -> (
        Option<QuantizedWeight>,
        Option<QuantizedWeight>,
        Option<MlxArray>,
    ) {
        (
            self.gate_up.map(|s| s.rebuild(inputs)),
            self.down.map(|s| s.rebuild(inputs)),
            self.post_norm.map(|i| inputs.get(i)),
        )
    }
}

/// Flatten every MLX array the dense FFN forward depends on into an explicit
/// input vector, returning the vector plus a [`CompiledDenseFfnSchema`] that
/// records where each tensor landed.
pub(crate) fn flatten_dense_ffn_inputs(
    x: &MlxArray,
    gate_up: Option<&QuantizedWeight>,
    down: Option<&QuantizedWeight>,
    post_norm: Option<&MlxArray>,
) -> (Vec<MlxArray>, CompiledDenseFfnSchema) {
    let mut inputs: Vec<MlxArray> = vec![x.clone()];
    let gate_up_slot = push_quant_inputs(&mut inputs, gate_up);
    let down_slot = push_quant_inputs(&mut inputs, down);
    let post_norm_idx = push_optional_input(&mut inputs, post_norm);
    (
        inputs,
        CompiledDenseFfnSchema {
            gate_up: gate_up_slot,
            down: down_slot,
            post_norm: post_norm_idx,
        },
    )
}

// ---------------------------------------------------------------------------
// Gemma4 dual-path (dense + expert) compile schema.
//
// Wraps the entire dual-path MoE block (dense sub-block + expert sub-block
// + combine) into a single compiled closure.  All weight tensors are explicit
// inputs; only `cfg`, `eps`, `packed_dim`, and index metadata are captured.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct CompiledGemma4DualPathSchema {
    // Dense sub-block
    dense_gate_up: QuantInputSlot,
    dense_down: QuantInputSlot,
    dense_post_norm1: Option<usize>,
    // Expert sub-block
    h2_norm: usize,
    router_proj: QuantInputSlot,
    router_combined_scale: usize,
    router_expert_scale: Option<usize>,
    expert_gate_up: Option<QuantInputSlot>,
    expert_gate: Option<QuantInputSlot>,
    expert_up: Option<QuantInputSlot>,
    expert_down: QuantInputSlot,
    expert_post_norm2: Option<usize>,
    // Combine
    ffn_post_norm: Option<usize>,
    // Scalars
    packed_dim: i32,
    pub(crate) moe_expert_count: usize,
    pub(crate) moe_experts_per_token: usize,
}

/// Flatten every MLX array the Gemma4 dual-path forward depends on into an
/// explicit input vector, returning the vector plus a
/// [`CompiledGemma4DualPathSchema`] that records where each tensor landed.
///
/// Returns `None` when a weight the compiled forward needs is absent (e.g.
/// checkpoints whose dense gate/up tensors are not row-packed, or expert
/// tensors shipped neither packed nor split) — the caller falls back to the
/// imperative dual-path. A panic here would fire inside the compile trace and
/// abort the process under `panic = "abort"`, so absence must be a graceful
/// decline, not an `expect`.
pub(crate) fn flatten_gemma4_dual_path_inputs(
    normed2: &MlxArray,
    hidden: &MlxArray,
    w: &LayerWeights,
) -> Option<(Vec<MlxArray>, CompiledGemma4DualPathSchema)> {
    // Quantized packed dense weight is `[2 * intermediate, packed_in]`: the
    // GeGLU split in `forward` slices the matmul *output*, so measure the
    // output axis (axis 0), not the packed input axis.
    let packed_dim = w.gate_up_packed.as_ref()?.weight.shape().first().copied()?;
    let mut inputs: Vec<MlxArray> = vec![normed2.clone(), hidden.clone()];
    let dense_gate_up = push_quant_inputs(&mut inputs, w.gate_up_packed.as_ref())?;
    let dense_down = push_quant_inputs(&mut inputs, w.down_proj.as_ref())?;
    let dense_post_norm1 = push_optional_input(&mut inputs, w.ffn_post_norm1.as_ref());
    let h2_norm = push_optional_input(&mut inputs, w.ffn_norm2.as_ref())?;
    let router_proj = push_quant_inputs(&mut inputs, w.router_proj.as_ref())?;
    let router_combined_scale = push_optional_input(&mut inputs, w.router_combined_scale.as_ref())?;
    let router_expert_scale = push_optional_input(&mut inputs, w.router_expert_scale.as_ref());
    let expert_gate_up = push_quant_inputs(&mut inputs, w.gate_up_exps_packed.as_ref());
    let expert_gate = push_quant_inputs(&mut inputs, w.gate_exps.as_ref());
    let expert_up = push_quant_inputs(&mut inputs, w.up_exps.as_ref());
    // The expert forward needs packed gate-up or the split pair.
    if expert_gate_up.is_none() && (expert_gate.is_none() || expert_up.is_none()) {
        return None;
    }
    let expert_down = push_quant_inputs(&mut inputs, w.down_exps.as_ref())?;
    let expert_post_norm2 = push_optional_input(&mut inputs, w.ffn_post_norm2.as_ref());
    let ffn_post_norm = push_optional_input(&mut inputs, w.ffn_post_norm.as_ref());
    let schema = CompiledGemma4DualPathSchema {
        dense_gate_up,
        dense_down,
        dense_post_norm1,
        h2_norm,
        router_proj,
        router_combined_scale,
        router_expert_scale,
        expert_gate_up,
        expert_gate,
        expert_up,
        expert_down,
        expert_post_norm2,
        ffn_post_norm,
        packed_dim,
        moe_expert_count: 0,
        moe_experts_per_token: 0,
    };
    Some((inputs, schema))
}

impl CompiledGemma4DualPathSchema {
    /// Execute the dual-path forward pass from the compiled closure's input
    /// vector, rebuilding all weight tensors from the schema indices.
    pub(crate) fn forward(&self, inputs: &MlxVectorArray, cfg: &ModelConfig) -> MlxArray {
        let normed2 = inputs.get(0);
        let hidden = inputs.get(1);
        let eps = cfg.rms_norm_eps;
        // Dense sub-block
        let dense_gate_up = self.dense_gate_up.rebuild(inputs);
        let gate_up_out = qw(&normed2, &dense_gate_up);
        // Fall back to slice + geglu when the packed Metal kernel is
        // ineligible (unsupported dtype or packed width): a panic here fires
        // inside the compile trace and aborts the process (panic=abort).
        let hidden_dim = self.packed_dim / 2;
        let h1_hidden = packed_geglu_metal_impl(&gate_up_out, hidden_dim).unwrap_or_else(|| {
            let gate = slice_last_dim(&gate_up_out, 0, hidden_dim, None);
            let up = slice_last_dim(&gate_up_out, hidden_dim, self.packed_dim, None);
            geglu(&gate, &up)
        });
        let dense_down = self.dense_down.rebuild(inputs);
        let h1 = qw(&h1_hidden, &dense_down);
        let h1 = crate::model::shared::rms_norm_opt(
            &h1,
            self.dense_post_norm1.map(|i| inputs.get(i)).as_ref(),
            eps,
        );
        // Expert sub-block
        let h2_norm_w = inputs.get(self.h2_norm);
        let h2_normed = rms_norm(&hidden, Some(&h2_norm_w), eps, None);
        let router_proj = self.router_proj.rebuild(inputs);
        let combined_scale = inputs.get(self.router_combined_scale);
        let normed_router = rms_norm(&hidden, Some(&combined_scale), eps, None);
        let expert_scores = qw(&normed_router, &router_proj);
        let (top_k_indices, top_k_weights) = top_k_by_argpartition(
            &expert_scores,
            self.moe_expert_count,
            self.moe_experts_per_token,
            true,
        );
        let h2 = moe_experts_forward_with_cloned_weights(
            cfg,
            &h2_normed,
            &top_k_indices,
            &top_k_weights,
            self.expert_gate_up.map(|s| s.rebuild(inputs)),
            self.expert_gate.map(|s| s.rebuild(inputs)),
            self.expert_up.map(|s| s.rebuild(inputs)),
            Some(self.expert_down.rebuild(inputs)),
            None,
            self.router_expert_scale.map(|i| inputs.get(i)),
        );
        // Combine dense + expert (fused post-norm when possible).
        let expert_post = self.expert_post_norm2.map(|i| inputs.get(i));
        let ffn_post = self.ffn_post_norm.map(|i| inputs.get(i));
        combine_gemma4_dual_path_outputs(&h1, &h2, expert_post.as_ref(), ffn_post.as_ref(), eps)
    }
}

// ---------------------------------------------------------------------------
// D3: Expert-Parallel dispatch infrastructure for MoE prefill.
//
// Pre-computes a per-expert token assignment (bin plan) so the expert FFN
// can be dispatched in parallel across GPU threadgroups instead of
// sequentially through gather_qmm.
// ---------------------------------------------------------------------------

/// Per-expert token assignment for parallel MoE dispatch.
#[allow(dead_code)]
struct ExpertBinPlan {
    /// Number of tokens assigned to each expert.
    bin_sizes: Vec<usize>,
    /// Maximum tokens assigned to any single expert.
    max_bin_size: usize,
    /// Mean tokens per active expert (total_assignments / active_experts).
    mean_bin_size: f64,
    /// Number of experts that received at least one token.
    active_experts: usize,
}

/// Build a per-expert token bin plan from the MoE routing output.
///
/// For each token, `top_k_indices` specifies which experts are selected.
/// This function counts how many tokens are assigned to each expert and
/// computes load-balance statistics.
fn build_expert_bins(top_k_indices: &MlxArray, n_experts: usize) -> Option<ExpertBinPlan> {
    let shape = top_k_indices.shape();
    let total_tokens = shape
        .iter()
        .take(shape.len().saturating_sub(1))
        .product::<i32>() as usize;
    let top_k = *shape.last()? as usize;
    if total_tokens == 0 || top_k == 0 || n_experts == 0 {
        return None;
    }
    let flat_size = total_tokens * top_k;
    // Ensure the array is Uint32 before calling data_u32(); convert if needed.
    let u32_indices: MlxArray;
    let indices_ref: &MlxArray = if top_k_indices.dtype() == MlxDtype::Uint32 {
        top_k_indices
    } else {
        u32_indices = astype(top_k_indices, MlxDtype::Uint32, None);
        &u32_indices
    };
    let indices = indices_ref.data_u32();
    if indices.len() < flat_size {
        return None;
    }
    let mut bin_sizes = vec![0_usize; n_experts];
    for &expert_id in &indices[..flat_size] {
        let eid = expert_id as usize;
        if eid < n_experts {
            bin_sizes[eid] += 1;
        }
    }
    let active_experts = bin_sizes.iter().filter(|&&s| s > 0).count();
    let max_bin_size = *bin_sizes.iter().max().unwrap_or(&0);
    let total_assignments = total_tokens * top_k;
    let mean_bin_size = if active_experts > 0 {
        total_assignments as f64 / active_experts as f64
    } else {
        0.0
    };
    Some(ExpertBinPlan {
        bin_sizes,
        max_bin_size,
        mean_bin_size,
        active_experts,
    })
}

/// Check whether the expert-parallel dispatch should be used for this prefill.
///
/// Returns true when the flag is on, seq > 1, and the token distribution
/// is balanced enough for parallel dispatch (max_bin <= 2x mean_bin).
fn expert_parallel_eligible(
    seq: usize,
    top_k_indices: &MlxArray,
    n_experts: usize,
) -> Option<ExpertBinPlan> {
    if seq <= 1 || !fastpath::moe_expert_parallel_enabled() {
        return None;
    }
    let plan = build_expert_bins(top_k_indices, n_experts)?;
    // Load-balance check: fall back to sequential gather_qmm when skewed.
    if plan.max_bin_size as f64 > 2.0 * plan.mean_bin_size {
        return None;
    }
    Some(plan)
}

// ---------------------------------------------------------------------------
// Tier 2A: Deep expert-block fusion — decode-only.
//
// Fuses gather_qmm(gate_up) + SwiGLU + gather_qmm(down) + weighted-sum into
// a single Metal kernel dispatch, achieving dense-class bandwidth utilization
// for MoE layers. Each expert’s weights are streamed through registers,
// with activation applied inline, emitting only the final weighted-sum output.
//
// Status: scaffold — the kernel body requires multi-week Metal engineering.
// The dispatch function and fastpath flag
// (`AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL`) are in place; the kernel source
// needs to be authored.
// ---------------------------------------------------------------------------

/// Attempt deep expert-block fusion for MoE decode.
///
/// Returns `Some(output)` if the fused kernel succeeds, `None` to fall back
/// to the standard multi-dispatch path.
///
/// Gated by `AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL` (default OFF).
fn try_moe_deep_expert_block_metal(
    _cfg: &ModelConfig,
    _w: &LayerWeights,
    _x: &MlxArray,
    _top_k_indices: &MlxArray,
    _top_k_weights: &MlxArray,
) -> Option<MlxArray> {
    if !fastpath::moe_deep_expert_block_metal_enabled() {
        return None;
    }
    // TODO(kernel): implement deep expert-block fusion Metal kernel.
    // The kernel must fuse:
    //   1. gather_qmm for packed gate_up (4-bit dequant + scatter-gather)
    //   2. Inline SwiGLU activation (split + silu + multiply)
    //   3. gather_qmm for down projection
    //   4. Inline weighted-sum with top-k weights
    //
    // Key parameters (Qwen3-Coder-Next):
    //   - hidden_dim: 2048
    //   - expert_intermediate_size: 512
    //   - num_experts: 256
    //   - top_k: 10
    //   - quantization: 4-bit affine, group_size=32
    //
    // Key challenges:
    //   - Must beat MLX's general gather_qmm for fixed top_k=10
    //   - 4-bit affine dequant + scatter-gather + activation in one kernel
    //   - Down projection gather_qmm requires separate weight layout
    //   - Weighted-sum across top_k experts with float32 accumulation
    None
}

#[allow(clippy::too_many_arguments)]
fn moe_experts_forward_impl(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
    top_k_expert_scale: Option<&MlxArray>,
    shared_expert_out: Option<&MlxArray>,
) -> MlxArray {
    let seq = x.shape().get(1).copied().unwrap_or(1) as usize;
    let profile_decode = seq == 1 && decode_profile_enabled();
    let profile_prefill = seq > 1 && prefill_profile_enabled();
    let profile_moe = moe_profile_enabled();
    let moe_total_started = profile_moe.then(Instant::now);
    if profile_moe {
        record_moe_profile_layer();
    }

    // D3: check expert-parallel eligibility for prefill. When the plan is
    // available and the parallel Metal kernel is implemented, this will
    // dispatch experts in parallel across GPU threadgroups. Currently falls
    // through to the sequential gather_qmm path.
    let _ep_plan = expert_parallel_eligible(seq, top_k_indices, cfg.moe_expert_count);

    // Tier 2A: try deep expert-block fusion (decode-only). Fuses gather_qmm
    // gate_up + SwiGLU + gather_qmm down + weighted-sum into one dispatch.
    // Falls back to the standard multi-dispatch path when ineligible.
    if seq == 1
        && let Some(out) = try_moe_deep_expert_block_metal(cfg, w, x, top_k_indices, top_k_weights)
    {
        return out;
    }

    // Match MLX SwitchGLU: [batch, seq, hidden] → [batch, seq, 1, 1, hidden].
    // The extra singleton before top_k is required by gather_mm/gather_qmm broadcasting.
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let gather_inputs = switch_gather_inputs(&x_exp, top_k_indices);
    let down_exps = w.down_exps.as_ref().expect("MoE layer must have down_exps");

    // Phase 1B: when the expert gate_up is packed and the flag is on, try the
    // packed SwiGLU Metal kernel directly on the gather_qmm output, fusing the
    // last-dim split + SiLU + multiply into one dispatch. Decode-only (seq==1):
    // at prefill the tensor is large and bandwidth-bound, where the separate
    // slice+silu_mul ops are faster than the single packed dispatch. Falls back
    // to the split-activation path when the kernel is ineligible or at prefill.
    let hidden = if let Some(packed) = &w.gate_up_exps_packed {
        let gate_up_started = Instant::now();
        let out = qw_gather(
            &gather_inputs.x,
            packed,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertGateUp,
            gate_up_started,
            &[&out],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::ExpertGateUp,
                saturating_profile_us(gate_up_started),
            );
        }
        let half = cfg.moe_expert_intermediate_size as i32;
        // Try fused packed activation Metal kernel (decode-only, seq==1).
        // GeGLU path: Gemma4 MoE experts — fuses split+gelu_approx+mul.
        // SwiGLU path: Qwen3 MoE experts — fuses split+silu+mul.
        // D2 fused-expert-block path: when the flag is on and the gather is
        // unsorted, fuses activation + squeeze + unsort in a single dispatch.
        // Falls back to split slice + dense_ffn_activation otherwise.
        let fused = if cfg.uses_geglu && seq == 1 && fastpath::moe_geglu_packed_metal_enabled() {
            packed_geglu_metal_impl(&out, half)
        } else if !cfg.uses_geglu && seq == 1 && fastpath::moe_swiglu_packed_metal_enabled() {
            packed_swiglu_metal_impl(&out, half)
        } else if seq == 1
            && !gather_inputs.sorted_indices
            && fastpath::moe_fused_expert_block_enabled()
        {
            let top_k = top_k_indices.shape().last().copied().unwrap_or(0);
            moe_fused_activation_unsort_metal(
                &out,
                gather_inputs.inv_order.as_ref().unwrap_or(top_k_indices),
                half,
                top_k,
                out.dtype(),
                cfg.uses_geglu,
            )
        } else {
            None
        };
        if let Some(fused) = fused {
            let activation_started = Instant::now();
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::MoeExpertActivation,
                activation_started,
                &[&fused],
            );
            if profile_moe {
                record_moe_profile_stage(
                    MoeProfileStage::ExpertActivation,
                    saturating_profile_us(activation_started),
                );
            }
            fused
        } else {
            let gate = mlx_slice_last_dim(&out, 0, half);
            let up = mlx_slice_last_dim(&out, half, half * 2);
            let activation_started = Instant::now();
            let h = dense_ffn_activation(cfg, &gate, &up);
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::MoeExpertActivation,
                activation_started,
                &[&h],
            );
            if profile_moe {
                record_moe_profile_stage(
                    MoeProfileStage::ExpertActivation,
                    saturating_profile_us(activation_started),
                );
            }
            h
        }
    } else {
        let gate_up_started = Instant::now();
        let gate_exps = w.gate_exps.as_ref().expect("MoE layer must have gate_exps");
        let gate_out = qw_gather(
            &gather_inputs.x,
            gate_exps,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        let up_exps = w.up_exps.as_ref().expect("MoE layer must have up_exps");
        let up_out = qw_gather(
            &gather_inputs.x,
            up_exps,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertGateUp,
            gate_up_started,
            &[&gate_out, &up_out],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::ExpertGateUp,
                saturating_profile_us(gate_up_started),
            );
        }
        let activation_started = Instant::now();
        let h = dense_ffn_activation(cfg, &gate_out, &up_out);
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertActivation,
            activation_started,
            &[&h],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::ExpertActivation,
                saturating_profile_us(activation_started),
            );
        }
        h
    };

    // Down projection: [1, seq, top_k, hidden]
    let down_started = Instant::now();
    let down_out = squeeze_switch_singleton(&qw_gather(
        &hidden,
        down_exps,
        &gather_inputs.indices,
        gather_inputs.sorted_indices,
    ));
    let down_out = gather_inputs.unsort(down_out);
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::MoeExpertDown,
        down_started,
        &[&down_out],
    );
    if profile_moe {
        record_moe_profile_stage(
            MoeProfileStage::ExpertDown,
            saturating_profile_us(down_started),
        );
    }

    // Fresh timer for the weighted-sum stage so it does not include the down
    // projection time (which is already recorded under MoeExpertDown).
    let weighted_sum_started = Instant::now();

    // Phase 1A: when shared_expert_out is provided, try the fused weighted-sum
    // kernel that adds the shared expert inside the same dispatch. Decode-only
    // and short prefill tail chunks (seq <= threshold): at long prefill the
    // weighted-sum is bandwidth-bound on a large tensor, where the fused
    // kernel's extra input read costs more than the dispatch it saves. Falls
    // back to the separate `add` in the branches below at long prefill.
    if seq <= MOE_SHARED_FUSION_SEQ_THRESHOLD
        && let Some(shared) = shared_expert_out
        && fastpath::moe_fuse_shared_expert_add_enabled()
        && let Some(out) =
            qwen3_moe_weighted_sum_with_shared_metal(&down_out, top_k_weights, shared, x.dtype())
    {
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertWeightedSum,
            weighted_sum_started,
            &[&out],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::WeightedSum,
                saturating_profile_us(weighted_sum_started),
            );
            if let Some(started) = moe_total_started {
                record_moe_profile_total(saturating_profile_us(started));
            }
        }
        return out;
    }

    // Weighted sum over top_k dimension → [1, seq, hidden]. Gemma4 decode hits
    // this in every layer; fuse multiply + reduction + cast to keep the direct
    // pipeline graph smaller. Other MoE families keep the generic MLX path.
    if cfg.gemma4_moe_router {
        if let Some(expert_scale) = top_k_expert_scale {
            if let Some(expert_sum) = gemma4_moe_weighted_scaled_sum_metal(
                &down_out,
                top_k_weights,
                top_k_indices,
                expert_scale,
                x.dtype(),
            ) {
                let out = if let Some(shared) = shared_expert_out {
                    add(&expert_sum, shared, None)
                } else {
                    expert_sum
                };
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::MoeExpertWeightedSum,
                    weighted_sum_started,
                    &[&out],
                );
                if profile_moe {
                    record_moe_profile_stage(
                        MoeProfileStage::WeightedSum,
                        saturating_profile_us(weighted_sum_started),
                    );
                    if let Some(started) = moe_total_started {
                        record_moe_profile_total(saturating_profile_us(started));
                    }
                }
                return out;
            }
        } else if let Some(expert_sum) =
            gemma4_moe_weighted_sum_metal(&down_out, top_k_weights, x.dtype())
        {
            let out = if let Some(shared) = shared_expert_out {
                add(&expert_sum, shared, None)
            } else {
                expert_sum
            };
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::MoeExpertWeightedSum,
                weighted_sum_started,
                &[&out],
            );
            if profile_moe {
                record_moe_profile_stage(
                    MoeProfileStage::WeightedSum,
                    saturating_profile_us(weighted_sum_started),
                );
                if let Some(started) = moe_total_started {
                    record_moe_profile_total(saturating_profile_us(started));
                }
            }
            return out;
        }
    } else if let Some(expert_sum) =
        qwen3_moe_weighted_sum_metal(&down_out, top_k_weights, x.dtype())
    {
        // Qwen3 MoE: use Metal kernel for weighted sum (fuses multiply + reduce + cast).
        // If shared_expert_out is present but the fused kernel was ineligible, add
        // the shared expert here as a separate dispatch.
        let out = if let Some(shared) = shared_expert_out {
            add(&expert_sum, shared, None)
        } else {
            expert_sum
        };
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertWeightedSum,
            weighted_sum_started,
            &[&out],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::WeightedSum,
                saturating_profile_us(weighted_sum_started),
            );
            if let Some(started) = moe_total_started {
                record_moe_profile_total(saturating_profile_us(started));
            }
        }
        return out;
    }
    let scaled_weights;
    let top_k_weights = if let Some(expert_scale) = top_k_expert_scale {
        let gathered = take(expert_scale, top_k_indices, 0, None);
        scaled_weights = multiply(top_k_weights, &gathered, None);
        &scaled_weights
    } else {
        top_k_weights
    };
    let seq_dim = down_out.ndim() as i32;
    let top_k_axis = seq_dim - 2; // second-to-last dim
    let scores_exp = expand_dims(top_k_weights, top_k_weights.ndim() as i32, None);
    let weighted = multiply(&down_out, &scores_exp, None);
    let out = sum_axis(&weighted, top_k_axis, false, None);
    // Cast back to the input dtype. GLM scores are f32 (sigmoid over astype→f32),
    // so without this the weighted sum is f32 and contaminates all downstream
    // residuals and projections. Python's MoE does `.astype(y.dtype)` here.
    let out = astype(&out, x.dtype(), None);
    // If shared_expert_out is present and the fused kernel was ineligible, add
    // the shared expert as a separate dispatch.
    let out = if let Some(shared) = shared_expert_out {
        add(&out, shared, None)
    } else {
        out
    };
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::MoeExpertWeightedSum,
        weighted_sum_started,
        &[&out],
    );
    if profile_moe {
        record_moe_profile_stage(
            MoeProfileStage::WeightedSum,
            saturating_profile_us(weighted_sum_started),
        );
        if let Some(started) = moe_total_started {
            record_moe_profile_total(saturating_profile_us(started));
        }
    }
    out
}

pub(crate) struct SwitchGatherInputs {
    pub(crate) x: MlxArray,
    pub(crate) indices: MlxArray,
    pub(crate) sorted_indices: bool,
    pub(crate) inv_order: Option<MlxArray>,
    pub(crate) original_indices_shape: Vec<i32>,
}

impl SwitchGatherInputs {
    fn unsort(&self, x: MlxArray) -> MlxArray {
        let Some(inv_order) = &self.inv_order else {
            return x;
        };
        let unsorted = take(&x, inv_order, 0, None);
        let mut shape = self.original_indices_shape.clone();
        let hidden = *x
            .shape()
            .last()
            .expect("expert output must have hidden dim");
        shape.push(hidden);
        reshape(&unsorted, &shape, None)
    }
}

const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

pub(crate) fn switch_gather_inputs(
    x_expanded: &MlxArray,
    indices: &MlxArray,
) -> SwitchGatherInputs {
    let indices_shape = indices.shape();
    let selection_count = shape_element_count(&indices_shape);
    let top_k = indices_shape.last().copied().unwrap_or(1).max(1) as usize;
    if selection_count < SWITCH_GLU_SORT_THRESHOLD {
        return SwitchGatherInputs {
            x: x_expanded.clone(),
            indices: indices.clone(),
            sorted_indices: false,
            inv_order: None,
            original_indices_shape: indices_shape,
        };
    }

    let flat_indices = reshape(indices, &[selection_count as i32], None);
    let order = argsort_axis(&flat_indices, -1, None);
    let inv_order = argsort_axis(&order, -1, None);
    let sorted_indices = take(&flat_indices, &order, 0, None);

    let x_shape = x_expanded.shape();
    let hidden = *x_shape
        .last()
        .expect("SwitchGLU input must include hidden dim");
    let rows = selection_count / top_k;
    let x_flat = reshape(x_expanded, &[rows as i32, 1, hidden], None);
    let top_k_scalar = MlxArray::from_raw_data(
        &(top_k as u32) as *const u32 as *const u8,
        std::mem::size_of::<u32>(),
        &[1],
        MlxDtype::Uint32,
    );
    let row_indices = astype(&divide(&order, &top_k_scalar, None), MlxDtype::Uint32, None);
    let x_sorted = take(&x_flat, &row_indices, 0, None);

    SwitchGatherInputs {
        x: x_sorted,
        indices: sorted_indices,
        sorted_indices: true,
        inv_order: Some(inv_order),
        original_indices_shape: indices_shape,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_sys::{
        MlxQuantizationMode, concatenate, eval, quantize, quantized_matmul, slice_last_dim,
    };

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        let max_abs_diff = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_abs_diff <= tolerance,
            "max_abs_diff {max_abs_diff} exceeds tolerance {tolerance}"
        );
    }

    #[test]
    fn moe_router_fused_metal_kernel_compiles_and_matches_fallback() {
        let logits_data: Vec<f32> = vec![0.1, 2.0, -1.0, 0.5, 3.0, 0.0, -2.0, 1.5];
        let num_experts = logits_data.len();
        let top_k = 3usize;
        let logits = array_f32(&logits_data, &[1, 1, num_experts as i32]);

        let (indices, weights) = moe_router_fused_metal_apply(&logits, num_experts, top_k)
            .expect("fused router kernel dispatch should be eligible");
        // try_eval (not eval) so a kernel-source compile error fails the test
        // with the Metal diagnostic instead of aborting the process.
        mlx_sys::transforms::try_eval(&[&indices, &weights])
            .expect("fused router kernel must compile and evaluate");

        let (ref_indices, ref_weights) = top_k_by_argpartition(&logits, num_experts, top_k, true);
        eval(&[&ref_indices, &ref_weights]);

        let mut fused: Vec<(u32, f32)> = indices
            .data_u32()
            .iter()
            .copied()
            .zip(weights.data_f32().iter().copied())
            .collect();
        let mut reference: Vec<(u32, f32)> = ref_indices
            .data_u32()
            .iter()
            .copied()
            .zip(ref_weights.data_f32().iter().copied())
            .collect();
        // argpartition returns the top-k unordered; the kernel returns them
        // max-first. Compare as (index, weight) pairs sorted by expert index.
        fused.sort_by_key(|(index, _)| *index);
        reference.sort_by_key(|(index, _)| *index);

        assert_eq!(
            fused.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
            reference
                .iter()
                .map(|(index, _)| *index)
                .collect::<Vec<_>>()
        );
        let fused_weights: Vec<f32> = fused.iter().map(|(_, weight)| *weight).collect();
        let reference_weights: Vec<f32> = reference.iter().map(|(_, weight)| *weight).collect();
        assert_close(&fused_weights, &reference_weights, 1.0e-5);
        let weight_sum: f32 = fused_weights.iter().sum();
        assert!(
            (weight_sum - 1.0).abs() < 1.0e-5,
            "weights must be a softmax"
        );
    }

    #[test]
    fn packed_geglu_metal_matches_direct_geglu_for_bf16_packed_gate_up() {
        let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.083).collect();
        let up_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.037).collect();
        let gate = astype(&array_f32(&gate_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
        let up = astype(&array_f32(&up_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
        let packed = concatenate(&[&gate, &up], -1, None);

        let direct = astype(&geglu(&gate, &up), MlxDtype::Float32, None);
        let metal = packed_geglu_metal_impl(&packed, 8)
            .expect("packed GEGLU Metal kernel should support bf16 packed gate/up");
        let metal = astype(&metal, MlxDtype::Float32, None);
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 3, 8]);
        assert_eq!(
            metal.data_f32(),
            direct.data_f32(),
            "packed GEGLU shim must produce bit-identical output to the imperative reference"
        );
    }

    #[test]
    fn packed_swiglu_metal_matches_direct_swiglu_for_bf16_packed_gate_up() {
        let gate_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 12.0) * 0.071).collect();
        let up_data: Vec<f32> = (0..24).map(|i| ((i as f32) + 1.0) * 0.041).collect();
        let gate = astype(&array_f32(&gate_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
        let up = astype(&array_f32(&up_data, &[1, 3, 8]), MlxDtype::Bfloat16, None);
        let packed = concatenate(&[&gate, &up], -1, None);

        let direct = astype(
            &multiply(&mlx_sys::ops::silu(&gate, None), &up, None),
            MlxDtype::Float32,
            None,
        );
        let metal = packed_swiglu_metal_impl(&packed, 8)
            .expect("packed SwiGLU Metal kernel should support bf16 packed gate/up");
        let metal = astype(&metal, MlxDtype::Float32, None);
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 3, 8]);
        assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
    }

    #[test]
    fn qwen_dense_ffn_gate_up_swiglu_metal_matches_split_quantized_matmuls() {
        let x_data: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
        let gate_weight_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 180.0) * 0.0025).collect();
        let up_weight_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 96.0) * -0.001875).collect();
        let x = array_f32(&x_data, &[1, 1, 32]);
        let gate_weight = array_f32(&gate_weight_data, &[16, 32]);
        let up_weight = array_f32(&up_weight_data, &[16, 32]);
        let gate_q = quantize(
            &gate_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let up_q = quantize(
            &up_weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gate_q.len(), 3);
        assert_eq!(up_q.len(), 3);
        let gate = QuantizedWeight {
            weight: gate_q[0].clone(),
            scales: Some(gate_q[1].clone()),
            biases: Some(gate_q[2].clone()),
            group_size: 32,
            bits: 4,
        };
        let up = QuantizedWeight {
            weight: up_q[0].clone(),
            scales: Some(up_q[1].clone()),
            biases: Some(up_q[2].clone()),
            group_size: 32,
            bits: 4,
        };

        let metal = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up)
            .expect("4-bit affine gate/up SwiGLU matvec should be eligible");
        let gate_ref = quantized_matmul(
            &x,
            &gate_q[0],
            &gate_q[1],
            Some(&gate_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let up_ref = quantized_matmul(
            &x,
            &up_q[0],
            &up_q[1],
            Some(&up_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let reference = silu_mul(&gate_ref, &up_ref, None);
        mlx_sys::transforms::try_eval(&[&metal, &reference])
            .expect("Qwen dense FFN SwiGLU matvec Metal kernel must compile and evaluate");

        assert_eq!(metal.shape(), vec![1, 1, 16]);
        assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
    }

    #[test]
    fn qwen_dense_ffn_gate_up_swiglu_metal_rejects_non_decode_shapes() {
        let weight = QuantizedWeight {
            weight: mlx_sys::zeros(&[16, 4], MlxDtype::Uint32, None),
            scales: Some(mlx_sys::zeros(&[16, 1], MlxDtype::Bfloat16, None)),
            biases: Some(mlx_sys::zeros(&[16, 1], MlxDtype::Bfloat16, None)),
            group_size: 32,
            bits: 4,
        };
        let batched = mlx_sys::zeros(&[2, 1, 32], MlxDtype::Float32, None);
        let prefill = mlx_sys::zeros(&[1, 2, 32], MlxDtype::Float32, None);

        assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&batched, &weight, &weight).is_none());
        assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&prefill, &weight, &weight).is_none());
    }

    #[test]
    fn split_geglu_metal_matches_direct_geglu_for_bf16_gate_up() {
        let gate_data: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.059).collect();
        let up_data: Vec<f32> = (0..32).map(|i| ((i as f32) + 3.0) * 0.031).collect();
        let gate = astype(
            &array_f32(&gate_data, &[1, 1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );
        let up = astype(
            &array_f32(&up_data, &[1, 1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );

        let direct = astype(&gelu_approx_mul(&gate, &up, None), MlxDtype::Float32, None);
        let metal = gelu_approx_mul_metal(&gate, &up, true)
            .expect("split GEGLU Metal kernel should support bf16 gate/up");
        let metal = astype(&metal, MlxDtype::Float32, None);
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 1, 4, 8]);
        assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
    }

    #[test]
    fn gemma4_moe_weighted_sum_metal_matches_mlx_ops() {
        let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
        let down = array_f32(&down_data, &[1, 2, 3, 4]);
        let weights = array_f32(&weight_data, &[1, 2, 3]);

        let scores_exp = expand_dims(&weights, weights.ndim() as i32, None);
        let weighted = multiply(&down, &scores_exp, None);
        let direct = sum_axis(&weighted, 2, false, None);
        let metal = gemma4_moe_weighted_sum_metal(&down, &weights, MlxDtype::Float32)
            .expect("weighted-sum Metal kernel should support f32 inputs");
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 2, 4]);
        assert_close(metal.data_f32(), direct.data_f32(), 1.0e-5);
    }

    #[test]
    fn combine_gemma4_dual_path_fused_post_norm_matches_unfused() {
        // h1 + h2 then post-RMSNorm must match add_rms_norm_pair path.
        let h1 = array_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let h2 = array_f32(&[0.5, -0.5, 1.5, -1.5], &[1, 1, 4]);
        let post = array_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let eps = 1.0e-6_f32;

        let unfused = {
            let combined = add(&h1, &h2, None);
            rms_norm(&combined, Some(&post), eps, None)
        };
        let fused = combine_gemma4_dual_path_outputs(&h1, &h2, None, Some(&post), eps);
        eval(&[&unfused, &fused]);
        assert_eq!(fused.shape(), unfused.shape());
        assert_close(fused.data_f32(), unfused.data_f32(), 1.0e-5);
    }

    #[test]
    fn combine_gemma4_dual_path_with_expert_post_norm_stays_unfused_order() {
        let h1 = array_f32(&[1.0, 0.0, -1.0, 2.0], &[1, 1, 4]);
        let h2 = array_f32(&[0.25, 0.25, 0.25, 0.25], &[1, 1, 4]);
        let post2 = array_f32(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let post = array_f32(&[0.5, 0.5, 0.5, 0.5], &[4]);
        let eps = 1.0e-6_f32;

        let expected = {
            let h2n = rms_norm(&h2, Some(&post2), eps, None);
            let combined = add(&h1, &h2n, None);
            rms_norm(&combined, Some(&post), eps, None)
        };
        let got = combine_gemma4_dual_path_outputs(&h1, &h2, Some(&post2), Some(&post), eps);
        eval(&[&expected, &got]);
        assert_close(got.data_f32(), expected.data_f32(), 1.0e-5);
    }

    #[test]
    fn gemma4_moe_weighted_scaled_sum_metal_matches_mlx_ops() {
        let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
        let indices_data: Vec<u32> = vec![2, 0, 3, 1, 3, 0];
        let scale_data: Vec<f32> = vec![0.75, 1.25, 0.5, 1.5];
        let down = array_f32(&down_data, &[1, 2, 3, 4]);
        let weights = array_f32(&weight_data, &[1, 2, 3]);
        let indices = MlxArray::from_raw_data(
            indices_data.as_ptr() as *const u8,
            std::mem::size_of_val(indices_data.as_slice()),
            &[1, 2, 3],
            MlxDtype::Uint32,
        );
        let scale = array_f32(&scale_data, &[4]);

        let gathered = take(&scale, &indices, 0, None);
        let scaled_weights = multiply(&weights, &gathered, None);
        let scores_exp = expand_dims(&scaled_weights, scaled_weights.ndim() as i32, None);
        let weighted = multiply(&down, &scores_exp, None);
        let direct = sum_axis(&weighted, 2, false, None);
        let metal = gemma4_moe_weighted_scaled_sum_metal(
            &down,
            &weights,
            &indices,
            &scale,
            MlxDtype::Float32,
        )
        .expect("weighted scaled-sum Metal kernel should support f32 inputs");
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 2, 4]);
        assert_close(metal.data_f32(), direct.data_f32(), 1.0e-5);
    }

    #[test]
    fn moe_fused_activation_unsort_metal_matches_direct_geglu_for_bf16() {
        let hidden_dim = 8;
        let top_k = 3;
        let gate_data: Vec<f32> = (0..hidden_dim * top_k)
            .map(|i| ((i as f32) - 12.0) * 0.083)
            .collect();
        let up_data: Vec<f32> = (0..hidden_dim * top_k)
            .map(|i| ((i as f32) + 1.0) * 0.037)
            .collect();
        // Sorted-order gate/up, shape [top_k, hidden_dim].
        let gate = astype(
            &array_f32(&gate_data, &[top_k, hidden_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let up = astype(
            &array_f32(&up_data, &[top_k, hidden_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let packed_sorted = concatenate(&[&gate, &up], -1, None);
        let packed = reshape(&packed_sorted, &[1, 1, top_k, hidden_dim * 2], None);

        // original_k -> sorted_k: original position 0 reads sorted row 2, etc.
        let inv_order_data: Vec<u32> = vec![2, 0, 1];
        let inv_order = MlxArray::from_raw_data(
            inv_order_data.as_ptr() as *const u8,
            std::mem::size_of_val(inv_order_data.as_slice()),
            &[top_k],
            MlxDtype::Uint32,
        );

        // Reference: apply geglu in sorted order, then unsort via `take`
        // (mirrors SwitchGatherInputs::unsort's flatten + take pattern).
        let direct_sorted = geglu(&gate, &up);
        let direct = astype(
            &take(&direct_sorted, &inv_order, 0, None),
            MlxDtype::Float32,
            None,
        );

        let metal = moe_fused_activation_unsort_metal(
            &packed,
            &inv_order,
            hidden_dim,
            top_k,
            MlxDtype::Bfloat16,
            true,
        )
        .expect("MoE fused activation+unsort Metal kernel should support bf16 GEGLU inputs");
        let metal = astype(
            &reshape(&metal, &[top_k, hidden_dim], None),
            MlxDtype::Float32,
            None,
        );
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![top_k, hidden_dim]);
        assert_eq!(
            metal.data_f32(),
            direct.data_f32(),
            "MoE fused activation+unsort GEGLU branch must be bit-identical to the imperative reference"
        );
    }

    #[test]
    fn moe_fused_activation_unsort_metal_matches_direct_swiglu_for_bf16() {
        // Regression guard for the sibling `uses_geglu=false` branch: proves
        // the `if constexpr (USE_GEGLU)` specialization still selects SwiGLU
        // (not just that GEGLU no longer silently falls through to it).
        let hidden_dim = 8;
        let top_k = 3;
        let gate_data: Vec<f32> = (0..hidden_dim * top_k)
            .map(|i| ((i as f32) - 12.0) * 0.071)
            .collect();
        let up_data: Vec<f32> = (0..hidden_dim * top_k)
            .map(|i| ((i as f32) + 1.0) * 0.041)
            .collect();
        let gate = astype(
            &array_f32(&gate_data, &[top_k, hidden_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let up = astype(
            &array_f32(&up_data, &[top_k, hidden_dim]),
            MlxDtype::Bfloat16,
            None,
        );
        let packed_sorted = concatenate(&[&gate, &up], -1, None);
        let packed = reshape(&packed_sorted, &[1, 1, top_k, hidden_dim * 2], None);

        let inv_order_data: Vec<u32> = vec![2, 0, 1];
        let inv_order = MlxArray::from_raw_data(
            inv_order_data.as_ptr() as *const u8,
            std::mem::size_of_val(inv_order_data.as_slice()),
            &[top_k],
            MlxDtype::Uint32,
        );

        let direct_sorted = silu_mul(&gate, &up, None);
        let direct = astype(
            &take(&direct_sorted, &inv_order, 0, None),
            MlxDtype::Float32,
            None,
        );

        let metal = moe_fused_activation_unsort_metal(
            &packed,
            &inv_order,
            hidden_dim,
            top_k,
            MlxDtype::Bfloat16,
            false,
        )
        .expect("MoE fused activation+unsort Metal kernel should support bf16 SwiGLU inputs");
        let metal = astype(
            &reshape(&metal, &[top_k, hidden_dim], None),
            MlxDtype::Float32,
            None,
        );
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![top_k, hidden_dim]);
        assert_close(metal.data_f32(), direct.data_f32(), 1.0e-2);
    }

    #[test]
    fn packed_geglu_metal_rejects_unexpected_packed_width() {
        let data = vec![0.0_f32; 12];
        let packed = array_f32(&data, &[1, 1, 12]);
        assert!(
            packed_geglu_metal_impl(&packed, 5).is_none(),
            "packed width must be exactly 2 * hidden_dim"
        );

        let gate = slice_last_dim(&packed, 0, 6, None);
        assert!(
            packed_geglu_metal_impl(&gate, 6).is_none(),
            "already-split gate tensors must stay on the normal GEGLU path"
        );
    }

    #[test]
    fn packed_swiglu_metal_rejects_unexpected_packed_width() {
        let data = vec![0.0_f32; 12];
        let packed = array_f32(&data, &[1, 1, 12]);
        assert!(
            packed_swiglu_metal_impl(&packed, 5).is_none(),
            "packed width must be exactly 2 * hidden_dim"
        );

        let gate = slice_last_dim(&packed, 0, 6, None);
        assert!(
            packed_swiglu_metal_impl(&gate, 6).is_none(),
            "already-split gate tensors must stay on the normal SwiGLU path"
        );
    }

    #[test]
    fn qwen3_moe_weighted_sum_with_shared_metal_matches_unfused() {
        // down_out: [batch=1, seq=2, top_k=3, hidden=4]
        let down_data: Vec<f32> = (0..24).map(|i| ((i as f32) - 8.0) * 0.037).collect();
        let weight_data: Vec<f32> = vec![0.1, 0.25, 0.65, 0.5, 0.125, 0.375];
        let shared_data: Vec<f32> = (0..8).map(|i| ((i as f32) + 1.0) * 0.053).collect();

        let down = array_f32(&down_data, &[1, 2, 3, 4]);
        let weights = array_f32(&weight_data, &[1, 2, 3]);
        let shared = array_f32(&shared_data, &[1, 2, 4]);

        // Unfused reference: weighted_sum(down, weights) + shared
        let scores_exp = expand_dims(&weights, weights.ndim() as i32, None);
        let weighted = multiply(&down, &scores_exp, None);
        let expert_sum = sum_axis(&weighted, 2, false, None);
        let unfused = add(&expert_sum, &shared, None);

        // Fused kernel
        let fused =
            qwen3_moe_weighted_sum_with_shared_metal(&down, &weights, &shared, MlxDtype::Float32)
                .expect("fused weighted-sum-with-shared kernel should support f32 inputs");
        eval(&[&unfused, &fused]);

        assert_eq!(fused.shape(), vec![1, 2, 4]);
        assert_close(fused.data_f32(), unfused.data_f32(), 1.0e-5);
    }

    #[test]
    fn qwen3_moe_weighted_sum_with_shared_metal_rejects_shape_mismatch() {
        let down = array_f32(&[0.0; 24], &[1, 2, 3, 4]);
        let weights = array_f32(&[0.0; 6], &[1, 2, 3]);
        // Wrong shared shape: [1, 3, 4] instead of [1, 2, 4]
        let shared = array_f32(&[0.0; 12], &[1, 3, 4]);
        assert!(
            qwen3_moe_weighted_sum_with_shared_metal(&down, &weights, &shared, MlxDtype::Float32)
                .is_none(),
            "kernel must reject mismatched shared_expert shape"
        );
    }

    #[test]
    fn packed_swiglu_metal_matches_slice_and_silu_mul_on_moe_shaped_input() {
        // Simulates the MoE expert gate_up gather_qmm output: [batch=1, seq=1, top_k=4, 2*expert_size=16]
        let gate_data: Vec<f32> = (0..32).map(|i| ((i as f32) - 16.0) * 0.053).collect();
        let up_data: Vec<f32> = (0..32).map(|i| ((i as f32) + 2.0) * 0.031).collect();
        let gate = astype(
            &array_f32(&gate_data, &[1, 1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );
        let up = astype(
            &array_f32(&up_data, &[1, 1, 4, 8]),
            MlxDtype::Bfloat16,
            None,
        );
        let packed = concatenate(&[&gate, &up], -1, None);

        // Unfused reference: slice + silu_mul (matches the MoE fallback path)
        let half = 8_i32;
        let gate_slice = mlx_slice_last_dim(&packed, 0, half);
        let up_slice = mlx_slice_last_dim(&packed, half, half * 2);
        let direct = astype(
            &silu_mul(&gate_slice, &up_slice, None),
            MlxDtype::Float32,
            None,
        );

        // Fused packed SwiGLU kernel (same kernel as dense path, applied to MoE shape)
        let metal = packed_swiglu_metal_impl(&packed, 8)
            .expect("packed SwiGLU Metal kernel should support MoE-shaped gate_up");
        let metal = astype(&metal, MlxDtype::Float32, None);
        eval(&[&direct, &metal]);

        assert_eq!(metal.shape(), vec![1, 1, 4, 8]);
        assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
    }

    /// Guardrail probe for the Tier 3A compiled shared-expert closure.
    ///
    /// The core risk is that `shapeless=true` compilation with
    /// `quantized_matmul` is untested in this codebase (the existing compile
    /// caches use either elementwise ops with shapeless, or quantized_matmul
    /// with per-shape compilation). This probe builds a small quantized weight,
    /// compiles a shapeless closure doing `quantized_matmul -> sigmoid ->
    /// multiply` (the shared-expert gate path), and records the current
    /// fail-closed finding: the compiled output is correct for the traced
    /// shape, but not shape-polymorphic across a different sequence length.
    #[test]
    fn shapeless_compiled_linear_closure_is_not_shape_polymorphic() {
        use mlx_sys::{MlxClosure, MlxVectorArray, quantized_matmul, sigmoid};

        // Build a small non-quantized weight mimicking a shared-expert
        // projection: shape [hidden=8, out=16]. (The probe's goal is to verify
        // the shapeless compilation contract for a graph with a linear op +
        // elementwise ops across two input shapes. Quantized_matmul's packed
        // uint32 format is well-exercised by the production weight loader and
        // existing tests; the real unknown here is whether shapeless=true
        // preserves correctness for a linear graph, so a plain matmul suffices.)
        let weight_data: Vec<f32> = (0..128).map(|i| ((i as f32) - 64.0) * 0.01).collect();
        let weight = array_f32(&weight_data, &[8, 16]);
        let qw_captured = QuantizedWeight {
            weight: weight.clone(),
            scales: None,
            biases: None,
            group_size: 64,
            bits: 32,
        };

        // Capture a *clone* of the weight into the closure body. Per
        // closure.rs:191, captured MlxArrays become constants in the compiled
        // graph — this is the same mechanism the embedding closures use.
        let body_factory = || {
            let qw = qw_captured.clone();
            MlxClosure::new_dyn(move |inputs: &MlxVectorArray| {
                let x = inputs.get(0);
                let h = qw_inner(&qw, &x);
                let gate = inputs.get(1);
                let sig = sigmoid(&gate, None);
                vec![multiply(&h, &sig, None)]
            })
        };

        // Helper that mirrors qw() but takes QuantizedWeight by ref (avoids
        // lifetime issues with the closure capturing qw by value).
        fn qw_inner(qw: &QuantizedWeight, x: &MlxArray) -> MlxArray {
            if let Some(scales) = &qw.scales {
                quantized_matmul(
                    x,
                    &qw.weight,
                    scales,
                    qw.biases.as_ref(),
                    true,
                    Some(qw.group_size),
                    Some(qw.bits),
                    None,
                )
            } else {
                mlx_sys::matmul(x, &qw.weight, None)
            }
        }

        let compiled = body_factory()
            .compile(true)
            .expect("shapeless compile of quantized_matmul closure must succeed");

        // Shape 1: [1, 1, 8] (decode shape).
        let x1 = array_f32(
            &(0..8).map(|i| (i as f32) * 0.1).collect::<Vec<_>>(),
            &[1, 1, 8],
        );
        let gate1 = array_f32(&[0.3; 16], &[1, 1, 16]);

        let imperative_out_1 = {
            let h = qw_inner(&qw_captured, &x1);
            let sig = sigmoid(&gate1, None);
            multiply(&h, &sig, None)
        };
        let compiled_out_1 = compiled.apply(&[&x1, &gate1]);
        eval(&[&imperative_out_1, &compiled_out_1[0]]);
        assert_eq!(compiled_out_1[0].shape(), vec![1, 1, 16]);
        // Bit-identical: compiled graph must produce exactly the same result.
        let imp = imperative_out_1.data_f32().to_vec();
        let comp = compiled_out_1[0].data_f32().to_vec();
        let max_diff = imp
            .iter()
            .zip(&comp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1.0e-6,
            "shapeless compiled quantized_matmul closure must match imperative (shape 1): max_diff={max_diff}"
        );

        // Shape 2: [1, 4, 8] (prefill shape). Current MLX compile behavior
        // does not preserve correctness across this shape change, so Tier 3A
        // shared-expert compilation must stay out of production until this is
        // reworked with a per-shape cache or another fail-closed strategy.
        let x2 = array_f32(
            &(0..32).map(|i| (i as f32) * 0.05).collect::<Vec<_>>(),
            &[1, 4, 8],
        );
        let gate2 = array_f32(&[0.7; 64], &[1, 4, 16]);

        let imperative_out_2 = {
            let h = qw_inner(&qw_captured, &x2);
            let sig = sigmoid(&gate2, None);
            multiply(&h, &sig, None)
        };
        let compiled_out_2 = compiled.apply(&[&x2, &gate2]);
        eval(&[&imperative_out_2, &compiled_out_2[0]]);
        assert_eq!(compiled_out_2[0].shape(), vec![1, 4, 16]);
        let imp = imperative_out_2.data_f32().to_vec();
        let comp = compiled_out_2[0].data_f32().to_vec();
        let max_diff = imp
            .iter()
            .zip(&comp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1.0e-3,
            "shapeless compiled linear closure unexpectedly became shape-polymorphic; re-evaluate the Tier 3A guardrail before enabling it"
        );
    }
}
