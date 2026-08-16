use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, add, add_rms_norm_pair, argpartition_axis, argsort_axis, astype, async_eval,
    compiled_dual_gate_up_qmm, compiled_dual_gate_up_qmm_forced, compiled_gelu_approx_split_mlp,
    concatenate, contiguous, divide, dual_affine_qmm, dual_affine_qmm_forced, dual_qmm_geglu,
    dual_qmm_swiglu, dual_stream_affine_qmm, exp, expand_dims, expand_dims_axes, gelu_approx_mul,
    gelu_approx_mul_quantized_matmul, log1p, maximum, minimum, multiply, negative, power,
    quantized_matmul_rms_norm, quantized_matmul_with_mode, reshape, rms_norm,
    rms_norm_quantized_matmul, silu_mul, silu_mul_quantized_matmul, slice, slice_last_dim, softmax,
    softmax_precise, sum_axis, take, take_along_axis, topk_axis, transpose, zeros,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::fastpath;
use crate::per_layer_compile::{apply_layer_dense_ffn_decode, apply_layer_dense_ffn_prefill_min};
use crate::weights::{
    LayerWeights, QuantizedWeight, SHARED_VERIFY_COMPILE_LAYER, compile_quant_contract_salt,
};

use super::super::config::{GlmRouterConfig, ModelConfig};
use super::super::profile::{
    DecodeProfileStage, MoeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    moe_profile_enabled, prefill_profile_enabled, record_moe_profile_layer,
    record_moe_profile_stage, record_moe_profile_total, record_moe_router_fused_attempt,
    record_moe_router_fused_fallback, record_moe_router_fused_hit,
    record_qwen_dense_ffn_gate_up_matvec_metal_attempt,
    record_qwen_dense_ffn_gate_up_matvec_metal_fallback,
    record_qwen_dense_ffn_gate_up_matvec_metal_hit, saturating_profile_us,
};
use super::utils::{
    ProjectionBatchPolicy, mlx_slice_last_dim, packed_qkv_kv_head_count, qkv_slices, qw, qw_gather,
    qw_with_policy, scalar_like, scale_hidden, shape_element_count, squeeze_switch_singleton,
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
    float x_f = static_cast<float>(x_v);
    // gelu_approx(gate) saturates to identity (gate > 10) or zero (gate < -10)
    // out there; skip tanh in that range because the cubic inner term overflows
    // half/bfloat16 intermediates and fast-math tanh(inf) returns NaN.
    //
    // The saturation must stay branchless: the v3/v4 early-return guards made
    // control flow divergent, which serializes the vectorized bf16 loads and
    // stores and cost ~40% of Gemma prefill throughput at the model level.
    // Compute the bit-exact in-range chain on a clamped gate (identical to
    // the unclamped value for every in-range input) and pick the saturation
    // endpoints with ternaries that compile to uniform `select`.
    float gate_cf = clamp(gate_f, -10.0f, 10.0f);
    T gate_c = static_cast<T>(gate_cf);
    // In-range math rounds through T after every step to stay bit-identical
    // with mlx-lm's imperative op-by-op gelu_approx(gate) * x chain.
    T half_v = static_cast<T>(0.5f);
    T one_v = static_cast<T>(1.0f);
    T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
    T coeff_v = static_cast<T>(0.044715f);

    T gate2 = static_cast<T>(static_cast<float>(gate_c) * static_cast<float>(gate_c));
    T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_c));
    T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
    T inner = static_cast<T>(static_cast<float>(gate_c) + static_cast<float>(cubic));
    T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
    T t = static_cast<T>(tanh(static_cast<float>(scaled)));
    T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
    T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_c));
    T activated = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
    float prod = static_cast<float>(activated) * x_f;
    prod = (gate_f > 10.0f) ? gate_f * x_f : prod;
    prod = (gate_f < -10.0f) ? 0.0f : prod;
    out[idx] = static_cast<T>(prod);
"#;

pub(crate) fn qkv_project(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        false,
        ProjectionBatchPolicy::Shared,
        None,
        None,
    )
}

/// Like [`qkv_project`], but project Q from `last_x` (last token) while K/V
/// stay on the full sequence. Split-path only: packed QKV cannot drop the Q
/// slice independently and ignores `last_x`.
pub(crate) fn qkv_project_last_query(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    last_x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        false,
        ProjectionBatchPolicy::Shared,
        None,
        Some(last_x),
    )
}

/// Like [`qkv_project`] but per-sequence-position (and per-batch-row) exact —
/// used for Gemma MoE multi-token teacher-forced verify so each position matches
/// singleton pure-direct QKV (Shared batched matmul drifts near-ties).
pub(crate) fn qkv_project_row_exact(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        false,
        ProjectionBatchPolicy::RowExact,
        None,
        None,
    )
}

/// Multi-token hybrid: position 0 uses RowExact (matches pure-direct MLX S=1);
/// remaining positions use Shared (amortized). Reduces formal first_diff while
/// keeping Shared speed on draft positions (smokef99).
pub(crate) fn qkv_project_pos0_exact_rest_shared(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let shape = x.shape();
    // Expect [1, S, H] multi-token.
    if shape.len() != 3 || shape[0] != 1 || shape[1] <= 1 {
        return qkv_project_row_exact(cfg, w, x, head_dim);
    }
    let s = shape[1] as usize;
    let h = shape[2];
    // pos0 singleton
    let x0 = contiguous(&slice(x, &[0, 0, 0], &[1, 1, h], &[1, 1, 1], None), None);
    let (q0, k0, v0, g0) = qkv_project(cfg, w, &x0, head_dim);
    // rest Shared
    let x_rest = contiguous(
        &slice(x, &[0, 1, 0], &[1, s as i32, h], &[1, 1, 1], None),
        None,
    );
    let (qr, kr, vr, gr) = qkv_project(cfg, w, &x_rest, head_dim);
    let q = concatenate(&[&q0, &qr], 1, None);
    let k = concatenate(&[&k0, &kr], 1, None);
    let v = concatenate(&[&v0, &vr], 1, None);
    let g = match (g0, gr) {
        (Some(a), Some(b)) => Some(concatenate(&[&a, &b], 1, None)),
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
    };
    (q, k, v, g)
}

/// Like [`qkv_project`], but when `input_norm` is provided and
/// `AX_MLX_ATTN_NORM_QKV_FUSE=1`, fuses `rms_norm(x, input_norm)` into the
/// packed-QKV quantized matmul (one C++ call). Falls back to portable
/// rms_norm then project when the fuse path is unavailable.
pub(crate) fn qkv_project_with_input_norm(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
    input_norm: Option<&MlxArray>,
    eps: f32,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        false,
        ProjectionBatchPolicy::Shared,
        input_norm.map(|n| (n, eps)),
        None,
    )
}

/// Projection policy for the batched-decode path: `RowExact` (per-row,
/// bit-identical to single decode, no weight-read amortization) unless
/// `AX_MLX_BATCHED_SHARED_PROJ` opts into `Shared` (one batched matmul,
/// amortizes toward the ~3.3× ceiling, bf16-drifts vs per-row). Phase 3.5.
fn batched_projection_policy() -> ProjectionBatchPolicy {
    if fastpath::batched_shared_projections_enabled() {
        ProjectionBatchPolicy::Shared
    } else {
        ProjectionBatchPolicy::RowExact
    }
}

pub(crate) fn qkv_project_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    qkv_project_inner(
        cfg,
        w,
        x,
        head_dim,
        false,
        batched_projection_policy(),
        None,
        None,
    )
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
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn qkv_project_inner(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
    force_packed: bool,
    projection_policy: ProjectionBatchPolicy,
    input_norm: Option<(&MlxArray, f32)>,
    last_q_x: Option<&MlxArray>,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let batch = x.shape().first().copied().unwrap_or(1);
    let seq = x.shape().get(1).copied().unwrap_or(1);
    let prefer_split = prefer_split_qkv_projection(
        &cfg.model_family,
        force_packed,
        projection_policy,
        batch,
        seq,
        w.q_proj.is_some() && w.k_proj.is_some(),
    );
    let packed_kv_head_count = w.qkv_packed.as_ref().and_then(|packed| {
        let packed_rows = packed.weight.shape().first().copied().unwrap_or(0) as usize;
        packed_qkv_kv_head_count(cfg, head_dim, packed_rows)
    });
    let use_contig = fastpath::should_qwen_prefill_contiguous_attn_weights(&cfg.model_family, seq);
    let contig_packed;
    let contig_q;
    let contig_k;
    let contig_v;
    if !prefer_split
        && let (Some(packed), Some(kv_head_count)) = (&w.qkv_packed, packed_kv_head_count)
    {
        let packed = if use_contig {
            contig_packed = cached_prefill_attn_contiguous_weight(packed);
            &contig_packed
        } else {
            packed
        };
        let slices = qkv_slices(cfg, head_dim, kv_head_count);
        let out = if let Some((norm_w, eps)) = input_norm
            && projection_policy == ProjectionBatchPolicy::Shared
            && !fastpath::qwen_linear_mtp_exact_enabled()
            && !fastpath::moe_mt_bf16_identity_enabled()
            && fastpath::should_attn_norm_qkv_fuse(&cfg.model_family, seq)
            && packed.is_affine_quantized()
            && let Some(scales) = packed.scales.as_ref()
        {
            rms_norm_quantized_matmul(
                x,
                norm_w,
                eps,
                &packed.weight,
                scales,
                packed.biases.as_ref(),
                packed.group_size,
                packed.bits,
                None,
            )
        } else if let Some((norm_w, eps)) = input_norm {
            let normed = rms_norm(x, Some(norm_w), eps, None);
            qw_with_policy(&normed, packed, projection_policy)
        } else {
            qw_with_policy(x, packed, projection_policy)
        };
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
        let normed;
        let x_in = if let Some((norm_w, eps)) = input_norm {
            normed = rms_norm(x, Some(norm_w), eps, None);
            &normed
        } else {
            x
        };
        let q_w = if use_contig {
            contig_q = cached_prefill_attn_contiguous_weight(w.q_proj.as_ref().unwrap());
            &contig_q
        } else {
            w.q_proj.as_ref().unwrap()
        };
        // Last-only generate: K/V stay on the full sequence; Q is the last
        // token. Packed path above cannot drop Q independently.
        let q_in = last_q_x.unwrap_or(x_in);
        let q_full = qw_with_policy(q_in, q_w, projection_policy);
        let (q, gate) = if cfg.attn_output_gate {
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
        let k_w = if use_contig {
            contig_k = cached_prefill_attn_contiguous_weight(w.k_proj.as_ref().unwrap());
            &contig_k
        } else {
            w.k_proj.as_ref().unwrap()
        };
        let k = qw_with_policy(x_in, k_w, projection_policy);
        let v = if let Some(v_proj) = w.v_proj.as_ref() {
            let v_w = if use_contig {
                contig_v = cached_prefill_attn_contiguous_weight(v_proj);
                &contig_v
            } else {
                v_proj
            };
            qw_with_policy(x_in, v_w, projection_policy)
        } else {
            k.clone()
        };
        (q, k, v, gate)
    }
}

// Greedy cold prefill keeps the final prompt token for the first-token graph,
// so README prompt buckets 128 and 512 enter the backbone as 127 and 511.
// Long pure uses `--prefill-chunk 512` (seq=512 → packed). mbp-m5 pure A/B
// extending the cap so chunk-512 took split was ~3% slower than packed
// (2026-07-25 pure-split-qkv-chunk512-ab); keep max=511.
const GEMMA4_SPLIT_PREFILL_MIN_SEQ: i32 = 127;
const GEMMA4_SPLIT_QKV_PREFILL_MAX_SEQ: i32 = 511;

fn prefer_split_qkv_projection(
    model_family: &str,
    force_packed: bool,
    projection_policy: ProjectionBatchPolicy,
    batch: i32,
    seq: i32,
    has_split_qk: bool,
) -> bool {
    !force_packed
        && projection_policy == ProjectionBatchPolicy::Shared
        && has_split_qk
        && (batch > 1
            || ((model_family == "gemma4" || model_family == "gemma4_unified")
                && batch == 1
                && (GEMMA4_SPLIT_PREFILL_MIN_SEQ..=GEMMA4_SPLIT_QKV_PREFILL_MAX_SEQ)
                    .contains(&seq)))
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

/// Attention o_proj followed by optional post-attention RMSNorm.
///
/// When `AX_MLX_O_PROJ_QMATMUL_RMS_NORM=1`, `post_norm` is present, there is no
/// attention gate, and scales are available, fuses into one
/// `quantized_matmul_rms_norm` C++ call (pure prefill residual on Gemma
/// sandwich `post_attention_layernorm`).
pub(crate) fn attention_output_projection_with_post_norm(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
    post_norm: Option<&MlxArray>,
    eps: f32,
) -> MlxArray {
    attention_output_projection_with_post_norm_policy(
        attn_flat,
        attn_gate,
        o_proj,
        post_norm,
        eps,
        ProjectionBatchPolicy::Shared,
    )
}

pub(crate) fn attention_output_projection_with_post_norm_policy(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
    post_norm: Option<&MlxArray>,
    eps: f32,
    projection_policy: ProjectionBatchPolicy,
) -> MlxArray {
    let seq = attn_flat.shape().get(1).copied().unwrap_or(1);
    let contig_o;
    let o_proj = if fastpath::should_qwen_prefill_contiguous_attn_weights_for(
        fastpath::qwen_prefill_contiguous_attn_weights_enabled(),
        "qwen3_5",
        seq,
    ) {
        contig_o = cached_prefill_attn_contiguous_weight(o_proj);
        &contig_o
    } else {
        o_proj
    };
    // Fused o_proj+rmsnorm is Shared-only; fall back when RowExact is required.
    // Skip when invariant projections are active (MoE exact): fused kernel
    // bypasses invariant and pure-direct would then diverge from multi-token.
    if projection_policy == ProjectionBatchPolicy::Shared
        && !fastpath::qwen_linear_mtp_exact_enabled()
        && !fastpath::moe_mt_bf16_identity_enabled()
        && attn_gate.is_none()
        && let Some(norm_w) = post_norm
        && fastpath::o_proj_qmatmul_rms_norm_enabled()
        && o_proj.is_affine_quantized()
        && let Some(scales) = o_proj.scales.as_ref()
    {
        return quantized_matmul_rms_norm(
            attn_flat,
            &o_proj.weight,
            scales,
            o_proj.biases.as_ref(),
            o_proj.group_size,
            o_proj.bits,
            norm_w,
            eps,
            None,
        );
    }
    let projected =
        attention_output_projection_with_policy(attn_flat, attn_gate, o_proj, projection_policy);
    if let Some(norm_w) = post_norm {
        rms_norm(&projected, Some(norm_w), eps, None)
    } else {
        projected
    }
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
        batched_projection_policy(),
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
/// stable FFI call.
///
/// Order (mlxcel residual `compiled_geglu_approx_activation`, gemma4.rs
/// multi-token bits=8 FFN):
/// 1. Opt-in process-static `mx::compile` via `AX_MLX_COMPILED_GEGLU_ACTIVATION=1`
/// 2. Default-ON Metal GEGLU (`AX_MLX_GEGLU_MUL_METAL`)
/// 3. Imperative `gelu_approx_mul` C++ shim
pub(crate) fn geglu(gate: &MlxArray, x: &MlxArray) -> MlxArray {
    if let Some(out) = mlx_sys::compiled_geglu_approx_activation(gate, x, None) {
        return out;
    }
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
            "ax_gemma_gelu_mul_v5",
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
    model_identity: u64,
    gate: &MlxArray,
    per_layer_input: &MlxArray,
    proj_w: &QuantizedWeight,
) -> MlxArray {
    if fastpath::gemma4_per_layer_input_gate_compile_enabled()
        && gate.shape().get(1).copied() == Some(1)
        && let Some(hidden) = crate::per_layer_compile::apply_per_layer_input_gate_decode(
            model_identity,
            gate,
            per_layer_input,
        )
    {
        return qw(&hidden, proj_w);
    }
    if proj_w.is_affine_quantized()
        && let Some(scales) = proj_w.scales.as_ref()
    {
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
    if let Some(limit) = deepseek_v4_swiglu_limit(cfg) {
        return deepseek_v4_clamped_swiglu(gate, up, limit);
    }
    if cfg.uses_geglu {
        geglu(gate, up)
    } else if fastpath::prefill_ffn_compile_swiglu_enabled()
        && !super::utils::qwen_prefill_skip_swiglu_compile_active()
    {
        swiglu(gate, up)
    } else {
        silu_mul(gate, up, None)
    }
}

/// DeepSeek V4 SwiGLU clamp limit (`deepseek_v4.swiglu_limit`), when enabled.
///
/// Gates every fused SwiGLU fast path (packed/fused Metal kernels and the
/// compiled prefill closure) off for V4: they would bypass the clamp.
pub(super) fn deepseek_v4_swiglu_limit(cfg: &ModelConfig) -> Option<f32> {
    cfg.deepseek_v4
        .as_ref()
        .map(|v4| v4.swiglu_limit)
        .filter(|limit| *limit > 0.0)
}

/// DeepSeek V4 clamped SwiGLU: `silu(min(gate, limit)) * clamp(up, ±limit)`.
///
/// llama.cpp `ffn_gate_clamped` / `ffn_up_clamped` + `ggml_swiglu_split` for
/// `LLM_ARCH_DEEPSEEK4`; vLLM `SiluAndMulWithClamp(swiglu_limit)` with the
/// default `alpha=1, beta=0`.
fn deepseek_v4_clamped_swiglu(gate: &MlxArray, up: &MlxArray, limit: f32) -> MlxArray {
    let pos = mlx_sys::ops::cached_scalar(limit, gate.dtype());
    let neg = mlx_sys::ops::cached_scalar(-limit, gate.dtype());
    let gate_c = minimum(gate, &pos, None);
    let up_c = mlx_sys::clip(up, &neg, &pos, None);
    silu_mul(&gate_c, &up_c, None)
}

fn packed_ffn_activation(
    cfg: &ModelConfig,
    gate_up: &MlxArray,
    hidden_dim: i32,
) -> Option<MlxArray> {
    if deepseek_v4_swiglu_limit(cfg).is_some() {
        // The fused packed kernels apply no clamp; fall back to the split
        // path so `dense_ffn_activation` applies the V4 clamped SwiGLU.
        return None;
    }
    if cfg.uses_geglu {
        packed_geglu_metal(gate_up, hidden_dim)
    } else {
        packed_swiglu_metal(gate_up, hidden_dim)
    }
}

static PACKED_GEGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static PACKED_SWIGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN_DENSE_FFN_DOWN_MATVEC_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN_DENSE_FFN_DOWN_RESIDUAL_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GEMMA_DUAL_GATE_UP_GEGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static QWEN_PREFILL_DUAL_QMM_SWIGLU_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
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
    float up_f = static_cast<float>(up_v);
    // See GELU_MUL_KERNEL_SOURCE above: saturate to identity/zero outside
    // [-10, 10] to avoid fast-math tanh(inf) = NaN, round through T at
    // every step in range to stay bit-identical with mlx-lm's imperative
    // gelu_approx(gate) * up chain, and keep the saturation branchless
    // (divergent early returns serialize vectorized bf16 memory traffic).
    float gate_cf = clamp(gate_f, -10.0f, 10.0f);
    T gate_c = static_cast<T>(gate_cf);
    T half_v = static_cast<T>(0.5f);
    T one_v = static_cast<T>(1.0f);
    T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
    T coeff_v = static_cast<T>(0.044715f);

    T gate2 = static_cast<T>(static_cast<float>(gate_c) * static_cast<float>(gate_c));
    T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_c));
    T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
    T inner = static_cast<T>(static_cast<float>(gate_c) + static_cast<float>(cubic));
    T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
    T t = static_cast<T>(tanh(static_cast<float>(scaled)));
    T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
    T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_c));
    T activated = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
    float prod = static_cast<float>(activated) * up_f;
    prod = (gate_f > 10.0f) ? gate_f * up_f : prod;
    prod = (gate_f < -10.0f) ? 0.0f : prod;
    out[idx] = static_cast<T>(prod);
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

/// Decode matvec: affine-4bit gate/up + SwiGLU for one token.
///
/// v1d: 256 threads per output row (8 simdgroups). Cross-simdgroup reduction
/// uses only 8 floats of TG memory (not a full x cache — full x TG caching
/// regressed thr to ~41 tok/s). More K-parallelism for PackedCols=512/1536.
const QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL_SOURCE: &str = r#"
    // grid.x = OutDim * 256, threadgroup = 256
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256; // 0..255 within the row's threadgroup
    uint lane = tid % 32;
    uint sg = tid / 32; // 0..7
    if (row >= OutDim) {
        return;
    }

    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    const uint row_base = row * PackedCols;
    const uint scale_row = row * GroupCount;

    // 256-wide stride over packed columns.
    for (uint packed_col = tid; packed_col < PackedCols; packed_col += 256) {
        uint gate_packed = gate_weight[row_base + packed_col];
        uint up_packed = up_weight[row_base + packed_col];
        for (uint packed_lane = 0; packed_lane < PackFactor; ++packed_lane) {
            uint input_col = packed_col * PackFactor + packed_lane;
            uint gate_q = (gate_packed >> (packed_lane * Bits)) & QuantMask;
            uint up_q = (up_packed >> (packed_lane * Bits)) & QuantMask;
            uint group = input_col / GroupSize;
            uint scale_idx = scale_row + group;
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

    // Cross-simdgroup reduce via tiny TG buffers (8 floats each).
    threadgroup float gate_partials[8];
    threadgroup float up_partials[8];
    if (lane == 0) {
        gate_partials[sg] = gate_sum;
        up_partials[sg] = up_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float g = 0.0f;
        float u = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            g += gate_partials[i];
            u += up_partials[i];
        }
        float activated = g / (1.0f + exp(-g));
        out[row] = static_cast<OutT>(activated * u);
    }
"#;

/// Multi-token dual gate/up affine GEMM + GEGLU for Gemma pure prefill (v3).
///
/// v1: one OutDim row per TG, re-read X from global → ~8.5× pure regression.
/// v2: BM=4 / TOKEN_TILE=8 / K=64 / TG=256 but K-loop strided by TG left most
///     threads idle → ~25× regression.
/// v3: classical tiled GEMM over (BM×BN) output tiles with full-TG cooperative
///     loads of X and dequantized W. BM*BN == TG so each thread owns one
///     output element of the tile (no cross-TG reduction). Streams gate+up
///     weights once per tile K-step and fuses gelu_approx * up at writeback.
///
/// Template ints: Leading, OutDim, PackedCols, InputDim, GroupSize, GroupCount,
/// Bits, PackFactor, QuantMask. OutT is the activation dtype.
/// Dispatch: grid.x = num_row_blocks * num_token_blocks * TG, threadgroup = TG.
const GEMMA_DUAL_GATE_UP_GEGLU_KERNEL_SOURCE: &str = r#"
    // Tiled dual-qmm GEMM: each TG owns BM output rows × BN tokens.
    // BM*BN must equal TG so one thread maps to one (row, token) of the tile.
    const uint BM = 8u;
    const uint BN = 16u;
    const uint BK = 128u; // multiple of GroupSize(64) and PackFactor(4)
    const uint TG = 128u; // BM * BN

    uint flat = thread_position_in_grid.x;
    uint block = flat / TG;
    uint tid = flat % TG;

    uint num_token_blocks = ((uint)Leading + BN - 1u) / BN;
    uint row_block = block / max(num_token_blocks, 1u);
    uint token_block = block % max(num_token_blocks, 1u);
    uint row0 = row_block * BM;
    uint t0 = token_block * BN;
    if (row0 >= (uint)OutDim || t0 >= (uint)Leading) {
        return;
    }
    uint nrows = min(BM, (uint)OutDim - row0);
    uint ntok = min(BN, (uint)Leading - t0);

    // Thread owns one (r_local, t_local) of the BM×BN tile.
    uint r_local = tid / BN;
    uint t_local = tid % BN;

    // X tile [BN, BK], W tiles [BM, BK] — all float dequantized.
    threadgroup float x_tile[BN * BK];
    threadgroup float gate_w_tile[BM * BK];
    threadgroup float up_w_tile[BM * BK];

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint k0 = 0u; k0 < (uint)InputDim; k0 += BK) {
        uint nk = min(BK, (uint)InputDim - k0);

        // Cooperative load of X[ntok, nk] → x_tile (row-major BN*BK).
        for (uint i = tid; i < BN * BK; i += TG) {
            uint ti = i / BK;
            uint kk = i % BK;
            float v = 0.0f;
            if (ti < ntok && kk < nk) {
                v = static_cast<float>(x[(t0 + ti) * (uint)InputDim + (k0 + kk)]);
            }
            x_tile[i] = v;
        }

        // Cooperative dequant of gate/up W[nrows, nk] → tiles (row-major BM*BK).
        for (uint i = tid; i < BM * BK; i += TG) {
            uint r = i / BK;
            uint kk = i % BK;
            float gw = 0.0f;
            float uw = 0.0f;
            if (r < nrows && kk < nk) {
                uint row = row0 + r;
                uint input_col = k0 + kk;
                uint packed_col = input_col / (uint)PackFactor;
                uint packed_lane = input_col % (uint)PackFactor;
                uint group = input_col / (uint)GroupSize;
                uint shift = packed_lane * (uint)Bits;
                uint row_base = row * (uint)PackedCols;
                uint scale_idx = row * (uint)GroupCount + group;
                uint gate_packed = gate_weight[row_base + packed_col];
                uint up_packed = up_weight[row_base + packed_col];
                uint gate_q = (gate_packed >> shift) & (uint)QuantMask;
                uint up_q = (up_packed >> shift) & (uint)QuantMask;
                float gate_scale = static_cast<float>(gate_scales[scale_idx]);
                float gate_bias = static_cast<float>(gate_biases[scale_idx]);
                float up_scale = static_cast<float>(up_scales[scale_idx]);
                float up_bias = static_cast<float>(up_biases[scale_idx]);
                gw = static_cast<float>(gate_q) * gate_scale + gate_bias;
                uw = static_cast<float>(up_q) * up_scale + up_bias;
            }
            gate_w_tile[i] = gw;
            up_w_tile[i] = uw;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Full-TG matmul for this K tile (one FMA stream per output element).
        if (r_local < nrows && t_local < ntok) {
            for (uint kk = 0u; kk < nk; ++kk) {
                float xv = x_tile[t_local * BK + kk];
                gate_acc = fma(xv, gate_w_tile[r_local * BK + kk], gate_acc);
                up_acc = fma(xv, up_w_tile[r_local * BK + kk], up_acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Fused gelu_approx(gate) * up writeback (gelu_pytorch_tanh).
    if (r_local < nrows && t_local < ntok) {
        float cubic = gate_acc * gate_acc * gate_acc;
        float inner = tanh(0.7978846f * (gate_acc + 0.044715f * cubic));
        float activated = 0.5f * gate_acc * (1.0f + inner);
        out[(t0 + t_local) * (uint)OutDim + (row0 + r_local)] =
            static_cast<OutT>(activated * up_acc);
    }
"#;

/// Qwen prefill 4-bit dual gate/up + SwiGLU using simdgroup_matrix 8×8 MMA.
/// Distinct from the scalar Gemma dual GEMM (8.5× reject) and from host-FFI
/// `dual_qmm_swiglu` (875 vs 891). Each TG is one simdgroup owning an
/// 8-token × 8-output tile.
const QWEN_PREFILL_DUAL_QMM_SWIGLU_KERNEL_SOURCE: &str = r#"
    constexpr uint Tile = 8;
    uint token0 = threadgroup_position_in_grid.x * Tile;
    uint row0 = threadgroup_position_in_grid.y * Tile;
    uint lane = thread_index_in_simdgroup;
    if (token0 >= (uint)Leading || row0 >= (uint)OutDim) {
        return;
    }
    const uint input_dim = (uint)PackedCols * (uint)PackFactor;
    const bool partial =
        token0 + Tile > (uint)Leading || row0 + Tile > (uint)OutDim;

    if (partial) {
        uint ntok = min(Tile, (uint)Leading - token0);
        uint nrows = min(Tile, (uint)OutDim - row0);
        for (uint t = lane; t < ntok; t += 32) {
            for (uint r = 0; r < nrows; ++r) {
                float gate_acc = 0.0f;
                float up_acc = 0.0f;
                uint tok = token0 + t;
                uint row = row0 + r;
                for (uint col = 0; col < input_dim; ++col) {
                    uint packed_col = col / (uint)PackFactor;
                    uint packed_lane = col % (uint)PackFactor;
                    uint shift = packed_lane * (uint)Bits;
                    uint group = col / (uint)GroupSize;
                    uint scale_idx = row * (uint)GroupCount + group;
                    uint gq = (gate_weight[row * (uint)PackedCols + packed_col] >> shift)
                        & (uint)QuantMask;
                    uint uq = (up_weight[row * (uint)PackedCols + packed_col] >> shift)
                        & (uint)QuantMask;
                    float xv = static_cast<float>(x[tok * input_dim + col]);
                    gate_acc = fma(
                        xv,
                        static_cast<float>(gq) * static_cast<float>(gate_scales[scale_idx])
                            + static_cast<float>(gate_biases[scale_idx]),
                        gate_acc);
                    up_acc = fma(
                        xv,
                        static_cast<float>(uq) * static_cast<float>(up_scales[scale_idx])
                            + static_cast<float>(up_biases[scale_idx]),
                        up_acc);
                }
                float sigmoid = 1.0f / (1.0f + exp(-gate_acc));
                out[tok * (uint)OutDim + row] =
                    static_cast<OutT>((gate_acc * sigmoid) * up_acc);
            }
        }
        return;
    }

    threadgroup float x_tg[Tile * Tile];
    threadgroup float gw_tg[Tile * Tile];
    threadgroup float uw_tg[Tile * Tile];
    simdgroup_matrix<float, 8, 8> gate_acc;
    simdgroup_matrix<float, 8, 8> up_acc;
    gate_acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    up_acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < input_dim; k0 += Tile) {
        for (uint i = lane; i < Tile * Tile; i += 32) {
            uint t = i / Tile;
            uint kk = i % Tile;
            uint row = row0 + t;
            uint col = k0 + kk;
            uint packed_col = col / (uint)PackFactor;
            uint packed_lane = col % (uint)PackFactor;
            uint shift = packed_lane * (uint)Bits;
            uint group = col / (uint)GroupSize;
            uint scale_idx = row * (uint)GroupCount + group;
            uint gq = (gate_weight[row * (uint)PackedCols + packed_col] >> shift)
                & (uint)QuantMask;
            uint uq = (up_weight[row * (uint)PackedCols + packed_col] >> shift)
                & (uint)QuantMask;
            x_tg[i] = static_cast<float>(x[(token0 + t) * input_dim + col]);
            gw_tg[i] = static_cast<float>(gq) * static_cast<float>(gate_scales[scale_idx])
                + static_cast<float>(gate_biases[scale_idx]);
            uw_tg[i] = static_cast<float>(uq) * static_cast<float>(up_scales[scale_idx])
                + static_cast<float>(up_biases[scale_idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<float, 8, 8> x_tile;
        simdgroup_matrix<float, 8, 8> gw_tile;
        simdgroup_matrix<float, 8, 8> uw_tile;
        simdgroup_load(x_tile, x_tg, Tile, ulong2(0, 0), false);
        simdgroup_load(gw_tile, gw_tg, Tile, ulong2(0, 0), true);
        simdgroup_load(uw_tile, uw_tg, Tile, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(gate_acc, x_tile, gw_tile, gate_acc);
        simdgroup_multiply_accumulate(up_acc, x_tile, uw_tile, up_acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float gate_out_tg[Tile * Tile];
    threadgroup float up_out_tg[Tile * Tile];
    simdgroup_store(gate_acc, gate_out_tg, Tile, ulong2(0, 0));
    simdgroup_store(up_acc, up_out_tg, Tile, ulong2(0, 0));
    for (uint i = lane; i < Tile * Tile; i += 32) {
        uint t = i / Tile;
        uint r = i % Tile;
        float g = gate_out_tg[i];
        float u = up_out_tg[i];
        float sigmoid = 1.0f / (1.0f + exp(-g));
        out[(token0 + t) * (uint)OutDim + (row0 + r)] =
            static_cast<OutT>((g * sigmoid) * u);
    }
"#;

/// Single-matrix affine-4bit decode matvec (FFN down_proj).
/// v1d: 256 threads per output row (same layout as gate/up).
const QWEN_DENSE_FFN_DOWN_MATVEC_KERNEL_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256;
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (row >= OutDim) {
        return;
    }

    float acc = 0.0f;
    const uint row_base = row * PackedCols;
    const uint scale_row = row * GroupCount;

    for (uint packed_col = tid; packed_col < PackedCols; packed_col += 256) {
        uint packed = weight[row_base + packed_col];
        for (uint packed_lane = 0; packed_lane < PackFactor; ++packed_lane) {
            uint input_col = packed_col * PackFactor + packed_lane;
            uint q = (packed >> (packed_lane * Bits)) & QuantMask;
            uint group = input_col / GroupSize;
            uint scale_idx = scale_row + group;
            float x_v = static_cast<float>(x[input_col]);
            float scale = static_cast<float>(scales[scale_idx]);
            float bias = static_cast<float>(biases[scale_idx]);
            acc = fma(x_v, static_cast<float>(q) * scale + bias, acc);
        }
    }

    float sum = simd_sum(acc);
    threadgroup float partials[8];
    if (lane == 0) {
        partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partials[i];
        }
        out[row] = static_cast<OutT>(total);
    }
"#;

const QWEN_DENSE_FFN_DOWN_RESIDUAL_KERNEL_SOURCE: &str = r#"
    uint flat = thread_position_in_grid.x;
    uint row = flat / 256;
    uint tid = flat % 256;
    uint lane = tid % 32;
    uint sg = tid / 32;
    if (row >= OutDim) {
        return;
    }

    float acc = 0.0f;
    const uint row_base = row * PackedCols;
    const uint scale_row = row * GroupCount;

    for (uint packed_col = tid; packed_col < PackedCols; packed_col += 256) {
        uint packed = weight[row_base + packed_col];
        for (uint packed_lane = 0; packed_lane < PackFactor; ++packed_lane) {
            uint input_col = packed_col * PackFactor + packed_lane;
            uint q = (packed >> (packed_lane * Bits)) & QuantMask;
            uint group = input_col / GroupSize;
            uint scale_idx = scale_row + group;
            float x_v = static_cast<float>(x[input_col]);
            float scale = static_cast<float>(scales[scale_idx]);
            float bias = static_cast<float>(biases[scale_idx]);
            acc = fma(x_v, static_cast<float>(q) * scale + bias, acc);
        }
    }

    float sum = simd_sum(acc);
    threadgroup float partials[8];
    if (lane == 0) {
        partials[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 8; ++i) {
            total += partials[i];
        }
        out[row] = static_cast<OutT>(total + static_cast<float>(residual[row]));
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
        "ax_gemma_packed_geglu_v5",
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
        || fastpath::qwen_linear_mtp_exact_enabled()
        || cfg.uses_geglu
        || !cfg.model_family.starts_with("qwen")
        || (gate.bits == up.bits
            && gate.group_size == up.group_size
            && qwen_dense_ffn_gate_up_matvec_metal_regresses(
                &cfg.model_family,
                cfg.layer_count,
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.moe_expert_count,
                gate.bits,
                gate.group_size,
            ))
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

fn qwen_dense_ffn_gate_up_matvec_metal_regresses(
    model_family: &str,
    layer_count: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_expert_count: usize,
    bits: i32,
    group_size: i32,
) -> bool {
    // Qwen3.6-27B publishes the Qwen3.5 model family in its config. On the
    // 64-layer dense 5120 -> 17408 FFN at 4-bit/group-64, the custom gate/up and
    // down matvec kernels measured 19.33 tok/s median with a 16.4% spread.
    // Letting MLX use its split quantized-matmul FFN measured 22.74 tok/s with
    // a 2.9% spread under the same fixed-token decode protocol. Keep the custom
    // path for the smaller Qwen3.5-9B shape where it remains a measured win.
    model_family == "qwen3_5"
        && layer_count == 64
        && hidden_size == 5120
        && intermediate_size == 17_408
        && moe_expert_count == 0
        && bits == 4
        && group_size == 64
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
    let quant_mask = (1_i32 << gate.bits) - 1;
    let kernel = QWEN_DENSE_FFN_GATE_UP_MATVEC_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_dense_ffn_gate_up_swiglu_simd_v1d",
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
            // 256 threads (8 simdgroups) per output row.
            (out_dim.saturating_mul(256), 1, 1),
            (256, 1, 1),
            None,
        )
        .ok()?;
    outputs.pop()
}

/// Multi-token Qwen 4-bit dual gate/up + SwiGLU via simdgroup_matrix MMA.
/// Flag is call-site only so tests can drive the body after a wash flip.
fn qwen_prefill_dual_qmm_swiglu_metal(
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
    if leading_elements <= 1 || leading_elements > 2048 {
        return None;
    }
    let leading = i32::try_from(leading_elements).ok()?;
    let (Some(gate_scales), Some(gate_biases), Some(up_scales), Some(up_biases)) = (
        gate.scales.as_ref(),
        gate.biases.as_ref(),
        up.scales.as_ref(),
        up.biases.as_ref(),
    ) else {
        return None;
    };
    if gate.bits != 4
        || up.bits != 4
        || gate.group_size != up.group_size
        || (gate.group_size != 32 && gate.group_size != 64)
        || gate.group_size <= 0
    {
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
    let expected = vec![out_dim, group_count];
    if gate_scales.shape() != expected
        || gate_biases.shape() != expected
        || up_scales.shape() != expected
        || up_biases.shape() != expected
    {
        return None;
    }
    let x_flat = if x_shape.len() == 2 && x_shape[0] == leading {
        x.clone()
    } else {
        reshape(x, &[leading, input_dim], None)
    };
    let quant_mask = (1_i32 << gate.bits) - 1;
    let kernel = QWEN_PREFILL_DUAL_QMM_SWIGLU_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_qwen_prefill_dual_qmm_swiglu_sg_v1",
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
            QWEN_PREFILL_DUAL_QMM_SWIGLU_KERNEL_SOURCE,
            "",
            true,
        )
    });
    let token_tiles = (leading + 7) / 8;
    let row_tiles = (out_dim + 7) / 8;
    let grid_x = token_tiles.checked_mul(32)?;
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                &x_flat,
                &gate.weight,
                gate_scales,
                gate_biases,
                &up.weight,
                up_scales,
                up_biases,
            ],
            &[KernelOutputSpec {
                shape: vec![leading, out_dim],
                dtype: x.dtype(),
            }],
            &[
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
            (grid_x, row_tiles, 1),
            (32, 1, 1),
            None,
        )
        .ok()?;
    let flat_out = outputs.pop()?;
    if x_shape.len() == 2 && x_shape[0] == leading {
        Some(flat_out)
    } else {
        let mut restored = x_shape;
        *restored.last_mut()? = out_dim;
        Some(reshape(&flat_out, &restored, None))
    }
}

/// Multi-token Gemma dual gate/up affine Metal + fused GEGLU product.
///
/// Profile residual: pure Gemma prefill is dominated by two multi-token qmm
/// (gate_proj + up_proj) on bits=8. This kernel streams X once, dequants both
/// weights per tile, reuses dequant across a token tile, and writes
/// `gelu_approx(gate)*up` without materialising either intermediate.
fn gemma_dense_ffn_dual_gate_up_geglu_metal(
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<MlxArray> {
    if !fastpath::gemma_dual_gate_up_metal_enabled() {
        return None;
    }
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
    // Multi-token only; decode keeps MLX / existing paths.
    if leading_elements <= 1 {
        return None;
    }
    // Cap Leading for dispatch cost / register tile loops (chunk is 512).
    if leading_elements > 2048 {
        return None;
    }
    let leading = i32::try_from(leading_elements).ok()?;

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
    // Flip package: bits=8 gs64; also allow bits=4 for ffn4 packages.
    if !(gate.bits == 4 || gate.bits == 8) || gate.group_size != 64 {
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
    let expected_sidecar = vec![out_dim, group_count];
    if gate_scales.shape() != expected_sidecar
        || gate_biases.shape() != expected_sidecar
        || up_scales.shape() != expected_sidecar
        || up_biases.shape() != expected_sidecar
    {
        return None;
    }

    // Flatten leading dims so the kernel sees [Leading, InputDim].
    let x_flat = if x_shape.len() == 2 && x_shape[0] == leading {
        x.clone()
    } else {
        reshape(x, &[leading, input_dim], None)
    };
    let out_flat_shape = vec![leading, out_dim];
    let quant_mask = (1_i32 << gate.bits) - 1;
    let kernel = GEMMA_DUAL_GATE_UP_GEGLU_KERNEL.get_or_init(|| {
        MlxMetalKernel::new(
            "ax_gemma_dense_ffn_dual_gate_up_geglu_v3",
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
            GEMMA_DUAL_GATE_UP_GEGLU_KERNEL_SOURCE,
            "",
            true,
        )
    });
    // v3 tiled GEMM: BM=8 rows × BN=16 tokens per TG of 128 threads.
    const BM: i32 = 8;
    const BN: i32 = 16;
    const TG: i32 = 128; // BM * BN
    let num_row_blocks = (out_dim + BM - 1) / BM;
    let num_token_blocks = (leading + BN - 1) / BN;
    let num_blocks = num_row_blocks.saturating_mul(num_token_blocks.max(1));
    let mut outputs = kernel
        .try_apply_with_template(
            &[
                &x_flat,
                &gate.weight,
                gate_scales,
                gate_biases,
                &up.weight,
                up_scales,
                up_biases,
            ],
            &[KernelOutputSpec {
                shape: out_flat_shape,
                dtype: x.dtype(),
            }],
            &[
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
                    name: "PackedCols",
                    value: packed_cols,
                },
                KernelTemplateArg::Int {
                    name: "InputDim",
                    value: input_dim,
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
            (num_blocks.saturating_mul(TG), 1, 1),
            (TG, 1, 1),
            None,
        )
        .ok()?;
    let flat_out = outputs.pop()?;
    // Restore original leading shape if needed: [...leading_dims, out_dim].
    if x_shape.len() == 2 && x_shape[0] == leading {
        Some(flat_out)
    } else {
        let mut restored = x_shape;
        *restored.last_mut()? = out_dim;
        Some(reshape(&flat_out, &restored, None))
    }
}

/// Decode-only affine-4bit matvec for FFN down_proj (intermediate → hidden).
/// When `residual` is `Some`, the kernel writes `residual + down(x)` so the
/// caller can skip a separate add on the Qwen decode residual.
fn qwen_dense_ffn_down_matvec_metal_impl(
    x: &MlxArray,
    down: &QuantizedWeight,
    residual: Option<&MlxArray>,
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
    let (Some(scales), Some(biases)) = (down.scales.as_ref(), down.biases.as_ref()) else {
        return None;
    };
    if down.bits != 4 || down.group_size <= 0 {
        return None;
    }
    let weight_shape = down.weight.shape();
    if weight_shape.len() != 2 {
        return None;
    }
    let out_dim = weight_shape[0];
    let packed_cols = weight_shape[1];
    if out_dim <= 0 || packed_cols <= 0 {
        return None;
    }
    let pack_factor = 32 / down.bits;
    if packed_cols.checked_mul(pack_factor)? != input_dim {
        return None;
    }
    if input_dim % down.group_size != 0 {
        return None;
    }
    let group_count = input_dim / down.group_size;
    let expected_sidecar = vec![out_dim, group_count];
    if scales.shape() != expected_sidecar || biases.shape() != expected_sidecar {
        return None;
    }

    if let Some(residual) = residual {
        if residual.dtype() != x.dtype() || residual.shape().last().copied() != Some(out_dim) {
            return None;
        }
        let residual_leading = residual.shape()[..residual.shape().len().saturating_sub(1)]
            .iter()
            .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
        if residual_leading != 1 {
            return None;
        }
    }
    let mut out_shape = x_shape;
    *out_shape.last_mut()? = out_dim;
    let quant_mask = (1_i32 << down.bits) - 1;
    let (kernel, inputs): (&MlxMetalKernel, Vec<&MlxArray>) = if let Some(residual) = residual {
        let kernel = QWEN_DENSE_FFN_DOWN_RESIDUAL_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_dense_ffn_down_residual_v1",
                &["x", "weight", "scales", "biases", "residual"],
                &["out"],
                QWEN_DENSE_FFN_DOWN_RESIDUAL_KERNEL_SOURCE,
                "",
                true,
            )
        });
        (kernel, vec![x, &down.weight, scales, biases, residual])
    } else {
        let kernel = QWEN_DENSE_FFN_DOWN_MATVEC_KERNEL.get_or_init(|| {
            MlxMetalKernel::new(
                "ax_qwen_dense_ffn_down_matvec_simd_v1d",
                &["x", "weight", "scales", "biases"],
                &["out"],
                QWEN_DENSE_FFN_DOWN_MATVEC_KERNEL_SOURCE,
                "",
                true,
            )
        });
        (kernel, vec![x, &down.weight, scales, biases])
    };
    let mut outputs = kernel
        .try_apply_with_template(
            &inputs,
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
                    value: down.group_size,
                },
                KernelTemplateArg::Int {
                    name: "GroupCount",
                    value: group_count,
                },
                KernelTemplateArg::Int {
                    name: "Bits",
                    value: down.bits,
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
            (out_dim.saturating_mul(256), 1, 1),
            (256, 1, 1),
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
        // Branchless saturation, mirroring GELU_MUL_KERNEL_SOURCE: the
        // divergent early-return form serializes the vectorized loads
        // (measured ~40% of Gemma prefill throughput there). Compute the
        // bit-exact in-range chain on a clamped gate and select the
        // saturation endpoints with uniform ternaries.
        float gate_f = static_cast<float>(gate_v);
        float gate_cf = clamp(gate_f, -10.0f, 10.0f);
        T gate_c = static_cast<T>(gate_cf);
        T half_v = static_cast<T>(0.5f);
        T one_v = static_cast<T>(1.0f);
        T sqrt_2_over_pi_v = static_cast<T>(0.7978846f);
        T coeff_v = static_cast<T>(0.044715f);

        T gate2 = static_cast<T>(static_cast<float>(gate_c) * static_cast<float>(gate_c));
        T gate3 = static_cast<T>(static_cast<float>(gate2) * static_cast<float>(gate_c));
        T cubic = static_cast<T>(static_cast<float>(coeff_v) * static_cast<float>(gate3));
        T inner = static_cast<T>(static_cast<float>(gate_c) + static_cast<float>(cubic));
        T scaled = static_cast<T>(static_cast<float>(sqrt_2_over_pi_v) * static_cast<float>(inner));
        T t = static_cast<T>(tanh(static_cast<float>(scaled)));
        T one_plus_t = static_cast<T>(static_cast<float>(one_v) + static_cast<float>(t));
        T half_gate = static_cast<T>(static_cast<float>(half_v) * static_cast<float>(gate_c));
        T activated_t = static_cast<T>(static_cast<float>(half_gate) * static_cast<float>(one_plus_t));
        float in_range =
            static_cast<float>(static_cast<T>(static_cast<float>(activated_t) * static_cast<float>(up_v)));
        activated = gate_f > 10.0f
            ? gate_f * static_cast<float>(up_v)
            : (gate_f < -10.0f ? 0.0f : in_range);
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
            "ax_moe_fused_activation_unsort_v5",
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

pub(crate) fn ffn_swiglu_row_exact(
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

/// Decode FFN plus residual: `residual + swiglu_ffn(x)`.
///
/// On the Qwen metal down path this is one kernel write instead of
/// `down` then `add`. Other routes fall back to `add(residual, ffn_swiglu(...))`.
pub(crate) fn ffn_swiglu_plus_residual(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
    residual: &MlxArray,
) -> MlxArray {
    let seq = x.shape().get(1).copied().unwrap_or(1);
    if seq == 1
        && let (Some(gate), Some(up), Some(down)) = (
            w.gate_proj.as_ref(),
            w.up_proj.as_ref(),
            w.down_proj.as_ref(),
        )
        && let Some(ffn_hidden) = qwen_dense_ffn_gate_up_swiglu_metal(cfg, x, gate, up)
        && let Some(fused) =
            qwen_dense_ffn_down_matvec_metal_impl(&ffn_hidden, down, Some(residual))
    {
        return match post_norm {
            Some(norm_w) => rms_norm(&fused, Some(norm_w), cfg.rms_norm_eps, None),
            None => fused,
        };
    }
    add(residual, &ffn_swiglu(cfg, w, x, post_norm, layer_idx), None)
}

pub(crate) fn ffn_swiglu_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
) -> MlxArray {
    ffn_swiglu_with_policy(cfg, w, x, post_norm, layer_idx, batched_projection_policy())
}

fn ffn_swiglu_with_policy(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
    projection_policy: ProjectionBatchPolicy,
) -> MlxArray {
    let shape = x.shape();
    let seq = shape.get(1).copied().unwrap_or(1);
    let qwen_dense_ffn = !cfg.uses_geglu && cfg.model_family.starts_with("qwen");
    let (x_f32, restore_dtype) = qwen_prefill_ffn_f32_input(x, qwen_dense_ffn, seq);
    let x = &x_f32;
    let out = if fastpath::should_qwen_prefill_flat_ffn(&cfg.model_family, seq, shape.len()) {
        let (flat, orig) = flatten_qwen_prefill_ffn_activation(x);
        let inner =
            ffn_swiglu_with_policy_inner(cfg, w, &flat, post_norm, layer_idx, projection_policy);
        restore_qwen_prefill_ffn_activation(&inner, orig)
    } else if fastpath::should_qwen_prefill_contiguous_ffn(&cfg.model_family, seq, shape.len()) {
        let x = contiguous(x, None);
        ffn_swiglu_with_policy_inner(cfg, w, &x, post_norm, layer_idx, projection_policy)
    } else {
        ffn_swiglu_with_policy_inner(cfg, w, x, post_norm, layer_idx, projection_policy)
    };
    qwen_prefill_ffn_restore_dtype(&out, restore_dtype)
}

/// Cast Qwen prefill FFN activations to Float32 for the steel qmm.
pub(crate) fn qwen_prefill_ffn_f32_input(
    x: &MlxArray,
    qwen_dense_ffn: bool,
    seq: i32,
) -> (MlxArray, Option<MlxDtype>) {
    qwen_prefill_ffn_f32_input_for(
        x,
        fastpath::qwen_prefill_ffn_f32_input_enabled(),
        qwen_dense_ffn,
        seq,
    )
}

/// Pure helper for [`qwen_prefill_ffn_f32_input`].
pub(crate) fn qwen_prefill_ffn_f32_input_for(
    x: &MlxArray,
    enabled: bool,
    qwen_dense_ffn: bool,
    seq: i32,
) -> (MlxArray, Option<MlxDtype>) {
    if qwen_dense_ffn
        && fastpath::should_qwen_prefill_ffn_f32_input_for(enabled, seq)
        && x.dtype() != MlxDtype::Float32
    {
        (astype(x, MlxDtype::Float32, None), Some(x.dtype()))
    } else {
        (x.clone(), None)
    }
}

/// Restore the pre-FFN activation dtype after a Float32 qmm pass.
pub(crate) fn qwen_prefill_ffn_restore_dtype(y: &MlxArray, orig: Option<MlxDtype>) -> MlxArray {
    match orig {
        Some(dtype) if y.dtype() != dtype => astype(y, dtype, None),
        _ => y.clone(),
    }
}

fn ffn_swiglu_with_policy_inner(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
    layer_idx: usize,
    projection_policy: ProjectionBatchPolicy,
) -> MlxArray {
    // 3-D `[B,S,H]` uses S. Flattened prefill `[B*S,H]` uses the leading
    // token count so Metal/compile gates do not see hidden_size as seq.
    let seq = if x.shape().len() == 2 {
        x.shape()[0]
    } else {
        x.shape().get(1).copied().unwrap_or(1)
    };
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

    let has_split_gate_up = w.gate_proj.is_some() && w.up_proj.is_some();
    let qwen_dense_ffn = !cfg.uses_geglu && cfg.model_family.starts_with("qwen");
    qwen_prefill_maybe_eval_ffn_input(x, qwen_dense_ffn, seq);
    let use_prefill_ffn_gs64 = qwen_dense_ffn && fastpath::should_qwen_prefill_ffn_gs64(seq);
    let packed_gs64 = if use_prefill_ffn_gs64 {
        w.gate_up_packed.as_ref().and_then(|src| {
            cached_prefill_ffn_gs64(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_PACKED,
                src,
            )
        })
    } else {
        None
    };
    let gate_gs64 = if use_prefill_ffn_gs64 {
        w.gate_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_gs64(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_GATE,
                src,
            )
        })
    } else {
        None
    };
    let up_gs64 = if use_prefill_ffn_gs64 {
        w.up_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_gs64(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_UP,
                src,
            )
        })
    } else {
        None
    };
    let down_gs64 = if use_prefill_ffn_gs64 {
        w.down_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_gs64(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_DOWN,
                src,
            )
        })
    } else {
        None
    };
    let use_prefill_ffn_q3 = qwen_dense_ffn && fastpath::should_qwen_prefill_q3_ffn(seq);
    let packed_q3 = if use_prefill_ffn_q3 {
        w.gate_up_packed.as_ref().and_then(|src| {
            cached_prefill_ffn_q3(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_PACKED,
                src,
            )
        })
    } else {
        None
    };
    let gate_q3 = if use_prefill_ffn_q3 {
        w.gate_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_q3(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_GATE,
                src,
            )
        })
    } else {
        None
    };
    let up_q3 = if use_prefill_ffn_q3 {
        w.up_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_q3(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_UP,
                src,
            )
        })
    } else {
        None
    };
    let down_q3 = if use_prefill_ffn_q3 {
        w.down_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_q3(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_DOWN,
                src,
            )
        })
    } else {
        None
    };
    let use_prefill_ffn_contig_w =
        qwen_dense_ffn && fastpath::should_qwen_prefill_contiguous_ffn_weights(seq);
    let packed_cw = if use_prefill_ffn_contig_w {
        w.gate_up_packed.as_ref().and_then(|src| {
            cached_prefill_ffn_contiguous_weight(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_PACKED,
                src,
            )
        })
    } else {
        None
    };
    let gate_cw = if use_prefill_ffn_contig_w {
        w.gate_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_contiguous_weight(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_GATE,
                src,
            )
        })
    } else {
        None
    };
    let up_cw = if use_prefill_ffn_contig_w {
        w.up_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_contiguous_weight(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_UP,
                src,
            )
        })
    } else {
        None
    };
    let down_cw = if use_prefill_ffn_contig_w {
        w.down_proj.as_ref().and_then(|src| {
            cached_prefill_ffn_contiguous_weight(
                cfg.compile_cache_identity,
                layer_idx,
                PREFILL_FFN_GS64_DOWN,
                src,
            )
        })
    } else {
        None
    };
    let prefer_split_gate_up = prefer_split_dense_ffn_gate_up(
        &cfg.model_family,
        qwen_dense_ffn,
        seq,
        leading_elements,
        has_split_gate_up,
    ) || (qwen_dense_ffn
        && fastpath::should_qwen_prefill_split_packed(&cfg.model_family, seq)
        && (has_split_gate_up || w.gate_up_packed.is_some()));
    let use_packed = use_packed_dense_ffn_prefill(
        prefer_split_gate_up,
        w.gate_up_packed.is_some(),
        super::utils::gemma4_prefill_skip_last_ffn_packed_active(),
    );

    // Compiled dense FFN with packed gate_up.
    // - SwiGLU (Qwen): decode shapeless + prefill fixed-shape.
    // - GEGLU (Gemma): decode still uncompiled (`gelu_approx` aborts under
    //   MLX shapeless compile). Prefill uses fixed-shape compile with the
    //   Metal-backed `geglu()` helper inside the body; short prompts skip
    //   via `DENSE_FFN_PREFILL_COMPILE_MIN_LEADING`.
    if use_packed && let Some(packed) = &w.gate_up_packed {
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
        let uses_geglu = cfg.uses_geglu;
        let body = move |inputs: &MlxVectorArray| {
            let x = inputs.get(0);
            let (gate_up_qw, down_qw, post_norm) = schema.rebuild(inputs);
            let gate_up = gate_up_qw
                .as_ref()
                .expect("dense FFN compile: gate_up weight required");
            let gate_up_out = qw_with_policy(&x, gate_up, projection_policy);
            let ffn_hidden = if uses_geglu {
                packed_geglu_metal(&gate_up_out, half_dim).unwrap_or_else(|| {
                    let gate = slice_last_dim(&gate_up_out, 0, half_dim, None);
                    let up = slice_last_dim(&gate_up_out, half_dim, half_dim * 2, None);
                    geglu(&gate, &up)
                })
            } else {
                packed_swiglu_metal(&gate_up_out, half_dim).unwrap_or_else(|| {
                    let gate = slice_last_dim(&gate_up_out, 0, half_dim, None);
                    let up = slice_last_dim(&gate_up_out, half_dim, half_dim * 2, None);
                    silu_mul(&gate, &up, None)
                })
            };
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
        // Shapeless / fixed-shape compile cannot host some MXFP8 Metal kernels
        // (`CustomKernel cannot infer output shapes`). Skip compile when the
        // packed gate/up path is non-affine so we do not pay a failed compile.
        let pack_mode = packed.mlx_quantization_mode();
        let compile_ok_for_quant = matches!(
            pack_mode,
            mlx_sys::MlxQuantizationMode::Affine | mlx_sys::MlxQuantizationMode::Mxfp4
        );
        let compiled_result = if !uses_geglu
            && seq == 1
            && leading_elements == 1
            && compile_ok_for_quant
            && fastpath::dense_ffn_compile_enabled()
        {
            apply_layer_dense_ffn_decode(cfg.compile_cache_identity, layer_idx, &input_refs, body)
        } else if seq > 1
            && leading_elements
                >= if fastpath::should_gemma4_packed_ffn_compile_p128(&cfg.model_family, seq) {
                    fastpath::GEMMA4_PACKED_FFN_COMPILE_P128_LEADING
                } else {
                    fastpath::DENSE_FFN_PREFILL_COMPILE_MIN_LEADING
                }
            && dense_ffn_prefill_compile_supported(&cfg.model_family, leading_elements)
            && compile_ok_for_quant
            && fastpath::dense_ffn_compile_prefill_enabled()
        {
            apply_layer_dense_ffn_prefill_min(
                cfg.compile_cache_identity,
                layer_idx,
                leading_elements,
                if fastpath::should_gemma4_packed_ffn_compile_p128(&cfg.model_family, seq) {
                    fastpath::GEMMA4_PACKED_FFN_COMPILE_P128_LEADING
                } else {
                    fastpath::DENSE_FFN_PREFILL_COMPILE_MIN_LEADING
                },
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
    // Qwen decode keeps the split route so its opt-in dense FFN matvec Metal
    // kernel can engage. Gemma4 publication-shape prefill also keeps split gate/up: paired
    // 128/512/2048 checks found its two MLX qmatmuls faster than the packed
    // fixed-shape graph, while decode retains the packed route.
    let (gate_out, up_out) = if !prefer_split_gate_up && let Some(packed_src) = &w.gate_up_packed {
        let packed = packed_q3
            .as_ref()
            .or(packed_gs64.as_ref())
            .or(packed_cw.as_ref())
            .unwrap_or(packed_src);
        let out = qw_with_policy(x, packed, projection_policy);
        qwen_prefill_maybe_async_packed_gate_up(&out, qwen_dense_ffn, seq);
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
                        && down.is_affine_quantized()
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
                qwen_prefill_maybe_eval_ffn_hidden(&ffn_hidden, qwen_dense_ffn, seq);
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnActivation,
                    activation_started,
                    &[&ffn_hidden],
                );
                let down_started = Instant::now();
                let down_src = w
                    .down_proj
                    .as_ref()
                    .expect("dense FFN layer must have down_proj");
                let down = down_q3
                    .as_ref()
                    .or(down_gs64.as_ref())
                    .or(down_cw.as_ref())
                    .unwrap_or(down_src);
                if let Some(norm_w) = post_norm {
                    if !profile_decode
                        && !profile_prefill
                        && projection_policy == ProjectionBatchPolicy::Shared
                        && fastpath::dense_qmatmul_rms_norm_enabled()
                        && down.is_affine_quantized()
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
                    qwen_prefill_maybe_async_down(&out, qwen_dense_ffn, seq);
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
                qwen_prefill_maybe_async_down(&out, qwen_dense_ffn, seq);
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
        let split_from_packed = if w.gate_proj.is_none() {
            w.gate_up_packed
                .as_ref()
                .and_then(cached_prefill_split_packed_ffn)
        } else {
            None
        };
        let gate_src = w
            .gate_proj
            .as_ref()
            .or(split_from_packed.as_ref().map(|(g, _)| g))
            .expect("dense FFN split path requires gate_proj or packed");
        let up_src = w
            .up_proj
            .as_ref()
            .or(split_from_packed.as_ref().map(|(_, u)| u))
            .expect("dense FFN split path requires up_proj or packed");
        let gate_w = gate_q3
            .as_ref()
            .or(gate_gs64.as_ref())
            .or(gate_cw.as_ref())
            .unwrap_or(gate_src);
        let up_w = up_q3
            .as_ref()
            .or(up_gs64.as_ref())
            .or(up_cw.as_ref())
            .unwrap_or(up_src);
        // Multi-token dual qmm + GEGLU in one C++ call (opt-in; mlxcel sequence
        // without mx::compile). Pure A/B residual: gate_up ~3.3s.
        if cfg.uses_geglu
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::dual_qmm_geglu_enabled()
            && let (Some(g_s), Some(u_s)) = (gate_w.scales.as_ref(), up_w.scales.as_ref())
            && let (Some(g_b), Some(u_b)) = (gate_w.biases.as_ref(), up_w.biases.as_ref())
            && gate_w.group_size > 0
            && gate_w.bits > 0
            && up_w.group_size == gate_w.group_size
            && up_w.bits == gate_w.bits
            && let Some(ffn_hidden) = dual_qmm_geglu(
                x,
                &gate_w.weight,
                g_s,
                g_b,
                &up_w.weight,
                u_s,
                u_b,
                gate_w.group_size,
                gate_w.bits,
                None,
            )
        {
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
                if projection_policy == ProjectionBatchPolicy::Shared
                    && fastpath::dense_qmatmul_rms_norm_enabled()
                    && down.is_affine_quantized()
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
        // Qwen prefill 4-bit dual gate/up + SwiGLU Metal (simdgroup MMA).
        // Host-FFI dual_qmm_swiglu stays OFF (875 vs 891).
        if qwen_dense_ffn
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::qwen_prefill_dual_qmm_swiglu_metal_enabled()
            && let Some(ffn_hidden) = qwen_prefill_dual_qmm_swiglu_metal(x, gate_w, up_w)
        {
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
        // Qwen multi-token dual affine qmm + SwiGLU in one C++ call (no
        // compile, no down fuse, no dual-stream). Targets p2048 gate_up
        // 837ms + activation 54ms. Gemma stays on dual_qmm_geglu (OFF).
        if qwen_dense_ffn
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::qwen_dual_qmm_swiglu_enabled()
            && let Some(ffn_hidden) = qwen_dual_qmm_swiglu(x, gate_w, up_w)
        {
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
                if projection_policy == ProjectionBatchPolicy::Shared
                    && fastpath::dense_qmatmul_rms_norm_enabled()
                    && down.is_affine_quantized()
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
        // Multi-token Gemma dual gate/up Metal + fused GEGLU (opt-in only:
        // pure-wall A/B on mbp-m5 measured ~8.5× regression vs MLX dual qmm).
        if cfg.uses_geglu
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && let Some(ffn_hidden) = gemma_dense_ffn_dual_gate_up_geglu_metal(x, gate_w, up_w)
        {
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
                if projection_policy == ProjectionBatchPolicy::Shared
                    && fastpath::dense_qmatmul_rms_norm_enabled()
                    && down.is_affine_quantized()
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
        // Prefer the fused gate/up+SwiGLU Metal matvec kernel on Qwen decode
        // *before* the host-side split-FFN compile path. Compile used to win
        // first and permanently shadow the kernel, costing ~3–4% pure decode
        // on M5 Max Qwen3.5-9B (107 → 110+ tok/s with kernel first).
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
            // Prefer the matching decode matvec Metal path for down_proj so the
            // full FFN stays on the custom 4-bit matvec kernels (gate/up/down).
            // Measured ~111.3 pure tok/s on M5 Max Qwen3.5-9B vs ~110.7 when
            // post-norm fusion shadowed the custom down path.
            if let Some(down_out) = qwen_dense_ffn_down_matvec_metal_impl(&ffn_hidden, down, None) {
                forward_profile_eval_elapsed(
                    profile_decode,
                    profile_prefill,
                    DecodeProfileStage::PostAttnFfnDown,
                    down_started,
                    &[&down_out],
                );
                return match post_norm {
                    Some(norm_w) => rms_norm(&down_out, Some(norm_w), cfg.rms_norm_eps, None),
                    None => down_out,
                };
            }
            if let Some(norm_w) = post_norm {
                if !profile_decode
                    && !profile_prefill
                    && projection_policy == ProjectionBatchPolicy::Shared
                    && fastpath::dense_qmatmul_rms_norm_enabled()
                    && down.is_affine_quantized()
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
        // Fallback when the matvec kernel is off or rejects the shape: compile
        // the full split FFN graph so host encoding is not rebuilt every token.
        // Dense (unquantized) projections must not enter this path: the per-layer
        // compile cache keys only (model, layer_idx), so a prior quantized main
        // layer can reuse a quantized_matmul graph against bf16 weights (MTP
        // sidecar heads). Skip compile when any of gate/up/down lack scales.
        if !cfg.uses_geglu
            && seq == 1
            && leading_elements == 1
            && fastpath::dense_ffn_compile_enabled()
            && gate_w.scales.is_some()
            && up_w.scales.is_some()
            && w.down_proj.as_ref().is_some_and(|d| d.scales.is_some())
            && let Some(down_w) = w.down_proj.as_ref()
            && let Some((inputs, schema)) =
                flatten_split_dense_ffn_inputs(x, gate_w, up_w, down_w, post_norm)
        {
            let input_refs: Vec<&MlxArray> = inputs.iter().collect();
            let eps = cfg.rms_norm_eps;
            let body = move |inputs: &MlxVectorArray| {
                let x = inputs.get(0);
                let (gate_qw, up_qw, down_qw, post_norm_w) = schema.rebuild(inputs);
                let gate = qw_with_policy(&x, &gate_qw, projection_policy);
                let up = qw_with_policy(&x, &up_qw, projection_policy);
                let hidden = silu_mul(&gate, &up, None);
                let out = qw_with_policy(&hidden, &down_qw, projection_policy);
                if let Some(norm_w) = post_norm_w {
                    vec![rms_norm(&out, Some(&norm_w), eps, None)]
                } else {
                    vec![out]
                }
            };
            if let Some(result) = apply_layer_dense_ffn_decode(
                cfg.compile_cache_identity,
                layer_idx,
                &input_refs,
                body,
            )
            .and_then(|r| r.into_iter().next())
            {
                return result;
            }
        }
        // Exact Qwen linear MTP verify is S=2..=4. Decode compile covers S=1
        // only; without a leading=2 graph the verify FFN is re-encoded every
        // layer every step. Same split body as decode/prefill compile.
        if qwen_dense_ffn
            && crate::fastpath::qwen_linear_mtp_exact_enabled()
            && (2..=4).contains(&leading_elements)
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && let Some(ffn_out) = qwen_compiled_split_verify_ffn(
                cfg.compile_cache_identity,
                layer_idx,
                x,
                gate_w,
                up_w,
                w.down_proj.as_ref(),
                post_norm,
                cfg.rms_norm_eps,
                projection_policy,
            )
        {
            return ffn_out;
        }
        // Qwen split **prefill** compile (seq>1): same graph as decode compile
        // (gate + up + SwiGLU + down), shape-specific so qmm stays on the
        // multi-token kernel. Packed Qwen prefill compile stays forbidden.
        if qwen_dense_ffn
            && seq > 1
            && leading_elements >= fastpath::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::qwen_split_ffn_prefill_compile_enabled()
            && let Some(ffn_out) = qwen_compiled_split_prefill_ffn(
                cfg.compile_cache_identity,
                layer_idx,
                x,
                gate_w,
                up_w,
                w.down_proj.as_ref(),
                post_norm,
                cfg.rms_norm_eps,
                projection_policy,
            )
        {
            return ffn_out;
        }
        // mlxcel residual: `compiled_gelu_approx_mlp_forward` for split affine
        // qGELU MLP. gs64/bits=4 + single-token: shapeless compile (#680).
        // AXQ 4-bit (gs!=64) seq=128: shape-specific compile
        // (`AX_MLX_COMPILED_QGELU_AXQ_P128=1`, default OFF after wash).
        // Multi-token non-4bit (flip Gemma MLP bits=8): shape-specific compile
        // (#705 prefill recovery) so prefill qmm is not forced onto decode
        // kernels. Opt-in: AX_MLX_COMPILED_QGELU_AXQ_P128 /
        // AX_MLX_COMPILED_QGELU_PREFILL_SHAPED; kill AX_MLX_COMPILED_QGELU_MLP.
        if cfg.uses_geglu
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && !fastpath::should_gemma4_dual_stream_gate_up_p128(&cfg.model_family, seq)
            && let Some(down_w) = w.down_proj.as_ref()
            && let (Some(g_s), Some(u_s), Some(d_s)) = (
                gate_w.scales.as_ref(),
                up_w.scales.as_ref(),
                down_w.scales.as_ref(),
            )
            && let (Some(g_b), Some(u_b), Some(d_b)) = (
                gate_w.biases.as_ref(),
                up_w.biases.as_ref(),
                down_w.biases.as_ref(),
            )
            && gate_w.group_size > 0
            && gate_w.bits > 0
            && up_w.group_size == gate_w.group_size
            && up_w.bits == gate_w.bits
            && down_w.group_size == gate_w.group_size
            && down_w.bits == gate_w.bits
            && let Some(ffn_out) = compiled_gelu_approx_split_mlp(
                x,
                &gate_w.weight,
                g_s,
                g_b,
                &up_w.weight,
                u_s,
                u_b,
                &down_w.weight,
                d_s,
                d_b,
                gate_w.group_size,
                gate_w.bits,
                None,
            )
        {
            return match post_norm {
                Some(norm_w) => rms_norm(&ffn_out, Some(norm_w), cfg.rms_norm_eps, None),
                None => ffn_out,
            };
        }
        packed_gate_up = None;
        // Profile residual (pure Gemma): gate_up dual qmm dominates multi-token
        // wall. Prefer (in order): shape-compiled dual qmm, single-FFI dual
        // affine qmm (Metal GEGLU kept), portable two-qw. Compile and dual-FFI
        // are opt-in env kill-switches (default OFF after pure rejects).
        // Qwen split prefill uses the *forced* compile (Gemma env stays OFF).
        if qwen_dense_ffn
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && let Some((gate, up)) = qwen_compiled_split_prefill_gate_up(x, gate_w, up_w)
        {
            (gate, up)
        } else if cfg.uses_geglu
            && seq > 1
            && !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && let (Some(g_s), Some(u_s)) = (gate_w.scales.as_ref(), up_w.scales.as_ref())
            && let (Some(g_b), Some(u_b)) = (gate_w.biases.as_ref(), up_w.biases.as_ref())
            && gate_w.group_size > 0
            && gate_w.bits > 0
            && up_w.group_size == gate_w.group_size
            && up_w.bits == gate_w.bits
        {
            if fastpath::should_gemma4_dual_stream_gate_up_p128(&cfg.model_family, seq)
                && let Some((gate, up)) = dual_stream_affine_qmm(
                    x,
                    &gate_w.weight,
                    g_s,
                    g_b,
                    &up_w.weight,
                    u_s,
                    u_b,
                    gate_w.group_size,
                    gate_w.bits,
                    None,
                )
            {
                (gate, up)
            } else if let Some((gate, up)) = compiled_dual_gate_up_qmm(
                x,
                &gate_w.weight,
                g_s,
                g_b,
                &up_w.weight,
                u_s,
                u_b,
                gate_w.group_size,
                gate_w.bits,
                None,
            ) {
                (gate, up)
            } else if let Some((gate, up)) = dual_affine_qmm(
                x,
                &gate_w.weight,
                g_s,
                g_b,
                &up_w.weight,
                u_s,
                u_b,
                gate_w.group_size,
                gate_w.bits,
                None,
            ) {
                (gate, up)
            } else {
                let gate = qw_with_policy(x, gate_w, projection_policy);
                let up = qw_with_policy(x, up_w, projection_policy);
                (gate, up)
            }
        } else if qwen_dense_ffn
            && let Some((gate, up)) =
                qwen_prefill_maybe_dual_affine_gate_up(&cfg.model_family, seq, x, gate_w, up_w)
        {
            (gate, up)
        } else {
            let gate = qw_with_policy(x, gate_w, projection_policy);
            let up = qw_with_policy(x, up_w, projection_policy);
            (gate, up)
        }
    };
    // Opt-in: co-submit dual gate/up before GEGLU (AX_MLX_ASYNC_DUAL_GATE_UP).
    // Profile residual gate_up ~3.26s; mlxcel builds both qmm then activation.
    // Qwen p2048: default-ON async_eval of the pair at seq>=1024 (not dual-stream).
    qwen_prefill_maybe_async_gate_up(&gate_out, &up_out, qwen_dense_ffn, seq);
    if seq > 1
        && (fastpath::async_dual_gate_up_enabled()
            || fastpath::should_gemma4_async_dual_gate_up_p128(&cfg.model_family, seq))
    {
        async_eval(&[&gate_out, &up_out]);
    }
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

    // Multi-token split GEGLU→down fuse (C++ gelu_approx_mul + down qmm).
    // Profile residual after dual gate_up qmm: activation + down (~2.5s force-eval).
    // Opt-in: AX_MLX_DENSE_GEGLU_DOWN_FUSE=1.
    let down_src = w
        .down_proj
        .as_ref()
        .expect("dense FFN layer must have down_proj");
    let down_q2 = if qwen_dense_ffn && fastpath::should_qwen_prefill_q2_down(seq) {
        cached_prefill_q2_down(cfg.compile_cache_identity, layer_idx, down_src)
    } else {
        None
    };
    let down = down_q3
        .as_ref()
        .or(down_gs64.as_ref())
        .or(down_q2.as_ref())
        .or(down_cw.as_ref())
        .unwrap_or(down_src);
    if cfg.uses_geglu
        && seq > 1
        && !profile_decode
        && !profile_prefill
        && projection_policy == ProjectionBatchPolicy::Shared
        && fastpath::dense_geglu_down_fuse_enabled()
        && down.is_affine_quantized()
        && let Some(scales) = down.scales.as_ref()
    {
        let activation_started = Instant::now();
        let fused = gelu_approx_mul_quantized_matmul(
            &gate_out,
            &up_out,
            &down.weight,
            scales,
            down.biases.as_ref(),
            down.group_size,
            down.bits,
            None,
        );
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnActivation,
            activation_started,
            &[&fused],
        );
        let down_started = Instant::now();
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnDown,
            down_started,
            &[&fused],
        );
        return match post_norm {
            Some(norm_w) => rms_norm(&fused, Some(norm_w), cfg.rms_norm_eps, None),
            None => fused,
        };
    }
    // Qwen SwiGLU→down fuse. Same residual as Gemma GEGLU fuse (activation +
    // down after split gate/up). Gemma env stays OFF.
    if qwen_dense_ffn
        && seq > 1
        && !profile_decode
        && !profile_prefill
        && projection_policy == ProjectionBatchPolicy::Shared
        && fastpath::qwen_swiglu_down_fuse_enabled()
        && let Some(fused) = qwen_swiglu_down_fuse(&gate_out, &up_out, down)
    {
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnActivation,
            Instant::now(),
            &[&fused],
        );
        return match post_norm {
            Some(norm_w) => rms_norm(&fused, Some(norm_w), cfg.rms_norm_eps, None),
            None => fused,
        };
    }

    // Gemma4 uses GEGLU with fast-approx GELU gate (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate).
    //
    // Gemma4 uses the direct MLX GeGLU shim. It preserves mlx-lm's activation
    // math without the server-thread stream hazards of the removed compiled
    // closure experiment.
    let activation_started = Instant::now();
    let ffn_hidden = dense_ffn_activation(cfg, &gate_out, &up_out);
    qwen_prefill_maybe_eval_ffn_hidden(&ffn_hidden, qwen_dense_ffn, seq);
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::PostAttnFfnActivation,
        activation_started,
        &[&ffn_hidden],
    );
    let down_started = Instant::now();
    // Standalone down GEMM: flatten [B,S,I] → [B*S,I] so MLX picks the 2-D
    // qmm kernel. Does not fuse silu+down (that path remasured 876 vs 891).
    if qwen_dense_ffn
        && seq > 1
        && !profile_decode
        && !profile_prefill
        && projection_policy == ProjectionBatchPolicy::Shared
        && fastpath::qwen_prefill_down_compile_enabled()
        && let Some(out) =
            qwen_compiled_prefill_down_qmm(cfg.compile_cache_identity, layer_idx, &ffn_hidden, down)
    {
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnDown,
            down_started,
            &[&out],
        );
        return match post_norm {
            Some(norm_w) => rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None),
            None => out,
        };
    }
    if qwen_dense_ffn
        && seq > 1
        && !profile_decode
        && !profile_prefill
        && projection_policy == ProjectionBatchPolicy::Shared
        && fastpath::qwen_prefill_flat_down_qmm_enabled()
        && let Some(out) = qwen_prefill_flat_down_qmm(&ffn_hidden, down)
    {
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnDown,
            down_started,
            &[&out],
        );
        return match post_norm {
            Some(norm_w) => rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None),
            None => out,
        };
    }
    if let Some(norm_w) = post_norm {
        if !profile_decode
            && !profile_prefill
            && projection_policy == ProjectionBatchPolicy::Shared
            && fastpath::dense_qmatmul_rms_norm_enabled()
            && down.is_affine_quantized()
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
        qwen_prefill_maybe_async_down(&out, qwen_dense_ffn, seq);
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
    qwen_prefill_maybe_async_down(&out, qwen_dense_ffn, seq);
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::PostAttnFfnDown,
        down_started,
        &[&out],
    );
    out
}

/// Shape-compiled dual affine qmm for Qwen split **prefill** (seq>1).
/// Gemma stays on env-gated [`compiled_dual_gate_up_qmm`] (default OFF).
fn qwen_compiled_split_prefill_gate_up(
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<(MlxArray, MlxArray)> {
    if !fastpath::qwen_compiled_dual_gate_up_enabled() {
        return None;
    }
    let x_shape = x.shape();
    if x_shape.len() < 2 || x_shape[x_shape.len() - 2] <= 1 {
        return None;
    }
    let (g_s, u_s) = (gate.scales.as_ref()?, up.scales.as_ref()?);
    let (g_b, u_b) = (gate.biases.as_ref()?, up.biases.as_ref()?);
    if gate.group_size <= 0
        || gate.bits <= 0
        || up.group_size != gate.group_size
        || up.bits != gate.bits
    {
        return None;
    }
    compiled_dual_gate_up_qmm_forced(
        x,
        &gate.weight,
        g_s,
        g_b,
        &up.weight,
        u_s,
        u_b,
        gate.group_size,
        gate.bits,
        None,
    )
}

/// One C++ dual steel qmm for Qwen split prefill. Not compile (forced
/// compiled dual stays opt-in / closed wash). C++ also reads this flag.
fn qwen_prefill_maybe_dual_affine_gate_up(
    model_family: &str,
    seq: i32,
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<(MlxArray, MlxArray)> {
    qwen_prefill_maybe_dual_affine_gate_up_for(
        fastpath::qwen_prefill_dual_affine_qmm_enabled(),
        model_family,
        seq,
        x,
        gate,
        up,
    )
}

/// Pure helper for [`qwen_prefill_maybe_dual_affine_gate_up`].
pub(crate) fn qwen_prefill_maybe_dual_affine_gate_up_for(
    enabled: bool,
    model_family: &str,
    seq: i32,
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<(MlxArray, MlxArray)> {
    if !fastpath::should_qwen_prefill_dual_affine_qmm_for(enabled, model_family, seq) {
        return None;
    }
    let (g_s, u_s) = (gate.scales.as_ref()?, up.scales.as_ref()?);
    let (g_b, u_b) = (gate.biases.as_ref()?, up.biases.as_ref()?);
    if gate.group_size <= 0
        || gate.bits <= 0
        || up.group_size != gate.group_size
        || up.bits != gate.bits
    {
        return None;
    }
    dual_affine_qmm_forced(
        x,
        &gate.weight,
        g_s,
        g_b,
        &up.weight,
        u_s,
        u_b,
        gate.group_size,
        gate.bits,
        None,
    )
}

/// Shape-compiled Qwen split FFN for **prefill** (gate + up + SwiGLU + down).
/// Packed Qwen prefill compile stays forbidden; this is the unused split
/// `mx.compile` analog for contract shapes (leading ≥ 128).
#[allow(clippy::too_many_arguments)]
fn qwen_compiled_split_verify_ffn(
    model_identity: u64,
    layer_idx: usize,
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    down: Option<&QuantizedWeight>,
    post_norm: Option<&MlxArray>,
    rms_norm_eps: f32,
    projection_policy: ProjectionBatchPolicy,
) -> Option<MlxArray> {
    let x_shape = x.shape();
    if x_shape.len() < 2 {
        return None;
    }
    let leading_elements: i64 = x_shape[..x_shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, dim| acc.checked_mul(i64::from(*dim)))?;
    if !(2..=4).contains(&leading_elements) {
        return None;
    }
    let down = down?;
    if gate.scales.is_none() || up.scales.is_none() || down.scales.is_none() {
        return None;
    }
    let (inputs, schema) = flatten_split_dense_ffn_inputs(x, gate, up, down, post_norm)?;
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let body = move |inputs: &MlxVectorArray| {
        let x = inputs.get(0);
        let (gate_qw, up_qw, down_qw, post_norm_w) = schema.rebuild(inputs);
        let gate = qw_with_policy(&x, &gate_qw, projection_policy);
        let up = qw_with_policy(&x, &up_qw, projection_policy);
        let hidden = silu_mul(&gate, &up, None);
        let out = qw_with_policy(&hidden, &down_qw, projection_policy);
        if let Some(norm_w) = post_norm_w {
            vec![rms_norm(&out, Some(&norm_w), rms_norm_eps, None)]
        } else {
            vec![out]
        }
    };
    crate::per_layer_compile::apply_layer_dense_ffn_prefill_min(
        model_identity,
        layer_idx,
        leading_elements,
        2,
        &input_refs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

/// Distinct compile-cache salt so residual+FFN graphs never share a key
/// with the FFN-only S=2 verify compile.
const VERIFY_FFN_RESIDUAL_COMPILE_SALT: u64 = 0x5245_5349_4446_464E;

#[derive(Clone, Copy)]
struct CompiledSplitVerifyResidualSchema {
    gate: QuantInputSlot,
    up: QuantInputSlot,
    down: QuantInputSlot,
}

fn flatten_split_verify_ffn_residual_inputs(
    hidden: &MlxArray,
    attn_proj: &MlxArray,
    ffn_norm: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    down: &QuantizedWeight,
) -> Option<(Vec<MlxArray>, CompiledSplitVerifyResidualSchema)> {
    let mut inputs = vec![hidden.clone(), attn_proj.clone(), ffn_norm.clone()];
    let gate_slot = push_quant_inputs(&mut inputs, Some(gate))?;
    let up_slot = push_quant_inputs(&mut inputs, Some(up))?;
    let down_slot = push_quant_inputs(&mut inputs, Some(down))?;
    Some((
        inputs,
        CompiledSplitVerifyResidualSchema {
            gate: gate_slot,
            up: up_slot,
            down: down_slot,
        },
    ))
}

/// Shape-compile `add_rms_norm(hidden, attn) → split FFN → add residual`
/// for exact S=2..=4 verify.
///
/// The portable RMS+SiLU *attention* gate stays outside this closure.
/// Falls back to the imperative residual+FFN path on compile miss.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_compiled_split_verify_ffn_plus_residual(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    attn_proj: &MlxArray,
    _layer_idx: usize,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled() || cfg.uses_geglu {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") {
        return None;
    }
    if w.router_proj.is_some() || w.ffn_post_norm.is_some() {
        return None;
    }
    let shape = hidden.shape();
    if shape.len() < 2 || attn_proj.shape() != shape {
        return None;
    }
    let leading_elements: i64 = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, dim| acc.checked_mul(i64::from(*dim)))?;
    if !(2..=4).contains(&leading_elements) {
        return None;
    }
    let gate = w.gate_proj.as_ref()?;
    let up = w.up_proj.as_ref()?;
    let down = w.down_proj.as_ref()?;
    if gate.scales.is_none() || up.scales.is_none() || down.scales.is_none() {
        return None;
    }
    let (inputs, schema) =
        flatten_split_verify_ffn_residual_inputs(hidden, attn_proj, &w.ffn_norm, gate, up, down)?;
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    let body = move |inputs: &MlxVectorArray| {
        let hidden = inputs.get(0);
        let attn = inputs.get(1);
        let ffn_norm = inputs.get(2);
        let (residual, normed) = add_rms_norm_pair(&hidden, &attn, &ffn_norm, eps, None);
        let gate_qw = schema.gate.rebuild(inputs);
        let up_qw = schema.up.rebuild(inputs);
        let down_qw = schema.down.rebuild(inputs);
        let gate = qw_with_policy(&normed, &gate_qw, ProjectionBatchPolicy::Shared);
        let up = qw_with_policy(&normed, &up_qw, ProjectionBatchPolicy::Shared);
        let act = silu_mul(&gate, &up, None);
        let ffn = qw_with_policy(&act, &down_qw, ProjectionBatchPolicy::Shared);
        vec![add(&residual, &ffn, None)]
    };
    apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity
            ^ VERIFY_FFN_RESIDUAL_COMPILE_SALT
            ^ compile_quant_contract_salt(&[gate, up, down]),
        SHARED_VERIFY_COMPILE_LAYER,
        leading_elements,
        2,
        &input_refs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

const VERIFY_LA_GATE_O_PROJ_COMPILE_SALT: u64 = 0x4C41_4741_5445_4F50;

#[derive(Clone, Copy)]
struct CompiledSplitVerifyLaGateOProjSchema {
    o_proj: QuantInputSlot,
}

/// Shape-compile portable `rms+silu_mul+reshape → o_proj` for exact S=2..=4
/// linear-attention verify. Factory `4d2a9a40` ON=`f4b5490d`; unhooked.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn qwen_compiled_split_verify_la_gate_o_proj(
    cfg: &ModelConfig,
    w: &LayerWeights,
    gd_out: &MlxArray,
    z: &MlxArray,
    layer_idx: usize,
    seq: i32,
    value_dim: i32,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled() || cfg.uses_geglu {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") {
        return None;
    }
    if !(2..=4).contains(&seq) {
        return None;
    }
    if gd_out.shape() != z.shape() {
        return None;
    }
    let linear = w.linear_attn.as_ref()?;
    linear.out_proj.scales.as_ref()?;
    let gd_shape = gd_out.shape();
    if gd_shape.len() < 2 || gd_shape.get(1).copied() != Some(seq) {
        return None;
    }
    let dtype = gd_out.dtype();
    let batch = gd_shape[0];
    let mut inputs = vec![gd_out.clone(), z.clone(), linear.norm.clone()];
    let o_slot = push_quant_inputs(&mut inputs, Some(&linear.out_proj))?;
    let schema = CompiledSplitVerifyLaGateOProjSchema { o_proj: o_slot };
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity ^ VERIFY_LA_GATE_O_PROJ_COMPILE_SALT,
        layer_idx,
        i64::from(seq),
        2,
        &input_refs,
        move |inputs: &MlxVectorArray| {
            let gd = inputs.get(0);
            let z = inputs.get(1);
            let la_norm = inputs.get(2);
            let normed = rms_norm(&gd, Some(&la_norm), eps, None);
            let gate_f32 = astype(&z, MlxDtype::Float32, None);
            let normed_f32 = astype(&normed, MlxDtype::Float32, None);
            let gated = silu_mul(&gate_f32, &normed_f32, None);
            let gated = astype(&gated, dtype, None);
            let flat = reshape(&gated, &[batch, seq, value_dim], None);
            let o_proj = schema.o_proj.rebuild(inputs);
            vec![qw_with_policy(
                &flat,
                &o_proj,
                ProjectionBatchPolicy::Shared,
            )]
        },
    )
    .and_then(|r| r.into_iter().next())
}

const VERIFY_LA_GATE_O_PROJ_FFN_COMPILE_SALT: u64 = 0x4C41_474F_4646_4E32;

#[derive(Clone, Copy)]
struct CompiledSplitVerifyLaGateOProjFfnSchema {
    o_proj: QuantInputSlot,
    gate: QuantInputSlot,
    up: QuantInputSlot,
    down: QuantInputSlot,
}

/// Shape-compile portable gate + o_proj + residual + FFN as **one** exact
/// S=2..=4 closure so `hidden` and the gate graph eval together.
/// Split compiles (gate outside, or gate+o_proj then FFN) were `f4b5490d`.
/// Factory `19bc8f95` ON=`f4b5490d`; unhooked.
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn qwen_compiled_split_verify_la_gate_o_proj_ffn(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    gd_out: &MlxArray,
    z: &MlxArray,
    layer_idx: usize,
    seq: i32,
    value_dim: i32,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled() || cfg.uses_geglu {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") {
        return None;
    }
    if w.router_proj.is_some() || w.ffn_post_norm.is_some() {
        return None;
    }
    if !(2..=4).contains(&seq) {
        return None;
    }
    if gd_out.shape() != z.shape() || gd_out.shape().get(1).copied() != Some(seq) {
        return None;
    }
    let linear = w.linear_attn.as_ref()?;
    let gate = w.gate_proj.as_ref()?;
    let up = w.up_proj.as_ref()?;
    let down = w.down_proj.as_ref()?;
    if linear.out_proj.scales.is_none()
        || gate.scales.is_none()
        || up.scales.is_none()
        || down.scales.is_none()
    {
        return None;
    }
    let dtype = gd_out.dtype();
    let batch = gd_out.shape()[0];
    let mut inputs = vec![
        hidden.clone(),
        gd_out.clone(),
        z.clone(),
        linear.norm.clone(),
        w.ffn_norm.clone(),
    ];
    let o_slot = push_quant_inputs(&mut inputs, Some(&linear.out_proj))?;
    let gate_slot = push_quant_inputs(&mut inputs, Some(gate))?;
    let up_slot = push_quant_inputs(&mut inputs, Some(up))?;
    let down_slot = push_quant_inputs(&mut inputs, Some(down))?;
    let schema = CompiledSplitVerifyLaGateOProjFfnSchema {
        o_proj: o_slot,
        gate: gate_slot,
        up: up_slot,
        down: down_slot,
    };
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity ^ VERIFY_LA_GATE_O_PROJ_FFN_COMPILE_SALT,
        layer_idx,
        i64::from(seq),
        2,
        &input_refs,
        move |inputs: &MlxVectorArray| {
            let hidden = inputs.get(0);
            let gd = inputs.get(1);
            let z = inputs.get(2);
            let la_norm = inputs.get(3);
            let ffn_norm = inputs.get(4);
            let normed = rms_norm(&gd, Some(&la_norm), eps, None);
            let gated = astype(
                &silu_mul(
                    &astype(&z, MlxDtype::Float32, None),
                    &astype(&normed, MlxDtype::Float32, None),
                    None,
                ),
                dtype,
                None,
            );
            let flat = reshape(&gated, &[batch, seq, value_dim], None);
            let o_proj = schema.o_proj.rebuild(inputs);
            let attn = qw_with_policy(&flat, &o_proj, ProjectionBatchPolicy::Shared);
            let (residual, normed) = add_rms_norm_pair(&hidden, &attn, &ffn_norm, eps, None);
            let gate_qw = schema.gate.rebuild(inputs);
            let up_qw = schema.up.rebuild(inputs);
            let down_qw = schema.down.rebuild(inputs);
            let g = qw_with_policy(&normed, &gate_qw, ProjectionBatchPolicy::Shared);
            let u = qw_with_policy(&normed, &up_qw, ProjectionBatchPolicy::Shared);
            let act = silu_mul(&g, &u, None);
            let ffn = qw_with_policy(&act, &down_qw, ProjectionBatchPolicy::Shared);
            vec![add(&residual, &ffn, None)]
        },
    )
    .and_then(|r| r.into_iter().next())
}

const VERIFY_FA_O_PROJ_FFN_COMPILE_SALT: u64 = 0x4641_4F50_4646_4E32;

#[derive(Clone, Copy)]
struct CompiledSplitVerifyFaOProjFfnSchema {
    o_proj: QuantInputSlot,
    gate: QuantInputSlot,
    up: QuantInputSlot,
    down: QuantInputSlot,
}

/// Shape-compile `flatten(SDPA) → o_proj → add_rms_norm → split FFN → add`
/// for exact S=2..=4 **full-attention** verify.
///
/// SDPA is the graph-break. This is not the linear-attention portable
/// RMS+SiLU gate, and not the unhooked LA-out_proj compile that became
/// `f4b5490d`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen_compiled_split_verify_fa_o_proj_ffn(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    attn_sdpa: &MlxArray,
    _layer_idx: usize,
    query_seq: usize,
    n_heads: usize,
    head_dim: usize,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled() || cfg.uses_geglu {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") {
        return None;
    }
    if w.router_proj.is_some() || w.ffn_post_norm.is_some() || w.attn_post_norm.is_some() {
        return None;
    }
    if !(2..=4).contains(&(query_seq as i32)) {
        return None;
    }
    let hidden_shape = hidden.shape();
    if hidden_shape.len() < 2 {
        return None;
    }
    let leading_elements: i64 = hidden_shape[..hidden_shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, dim| acc.checked_mul(i64::from(*dim)))?;
    if leading_elements != i64::from(query_seq as i32) {
        return None;
    }
    let o_proj = w.o_proj.as_ref()?;
    let gate = w.gate_proj.as_ref()?;
    let up = w.up_proj.as_ref()?;
    let down = w.down_proj.as_ref()?;
    if o_proj.scales.is_none()
        || gate.scales.is_none()
        || up.scales.is_none()
        || down.scales.is_none()
    {
        return None;
    }
    let mut inputs = vec![hidden.clone(), attn_sdpa.clone(), w.ffn_norm.clone()];
    let o_slot = push_quant_inputs(&mut inputs, Some(o_proj))?;
    let gate_slot = push_quant_inputs(&mut inputs, Some(gate))?;
    let up_slot = push_quant_inputs(&mut inputs, Some(up))?;
    let down_slot = push_quant_inputs(&mut inputs, Some(down))?;
    let schema = CompiledSplitVerifyFaOProjFfnSchema {
        o_proj: o_slot,
        gate: gate_slot,
        up: up_slot,
        down: down_slot,
    };
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    let batch = hidden_shape[0];
    let seq_i = query_seq as i32;
    let n_heads_i = n_heads as i32;
    let head_dim_i = head_dim as i32;
    let body = move |inputs: &MlxVectorArray| {
        let hidden = inputs.get(0);
        let attn_sdpa = inputs.get(1);
        let ffn_norm = inputs.get(2);
        let flat = {
            let transposed = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
            reshape(&transposed, &[batch, seq_i, n_heads_i * head_dim_i], None)
        };
        let o_proj = schema.o_proj.rebuild(inputs);
        let attn = qw_with_policy(&flat, &o_proj, ProjectionBatchPolicy::Shared);
        let (residual, normed) = add_rms_norm_pair(&hidden, &attn, &ffn_norm, eps, None);
        let gate_qw = schema.gate.rebuild(inputs);
        let up_qw = schema.up.rebuild(inputs);
        let down_qw = schema.down.rebuild(inputs);
        let gate = qw_with_policy(&normed, &gate_qw, ProjectionBatchPolicy::Shared);
        let up = qw_with_policy(&normed, &up_qw, ProjectionBatchPolicy::Shared);
        let act = silu_mul(&gate, &up, None);
        let ffn = qw_with_policy(&act, &down_qw, ProjectionBatchPolicy::Shared);
        vec![add(&residual, &ffn, None)]
    };
    apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity
            ^ VERIFY_FA_O_PROJ_FFN_COMPILE_SALT
            ^ compile_quant_contract_salt(&[o_proj, gate, up, down]),
        SHARED_VERIFY_COMPILE_LAYER,
        leading_elements,
        2,
        &input_refs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

const VERIFY_FA_ATTN_NORM_QKV_COMPILE_SALT: u64 = 0x4641_514B_5652_4D53;

#[derive(Clone, Copy)]
struct CompiledSplitVerifyFaAttnNormQkvSchema {
    q: QuantInputSlot,
    k: QuantInputSlot,
    v: QuantInputSlot,
}

/// Shape-compile `rms_norm → Q/K/V qw` for exact S=2..=4 full-attention
/// verify. Factory `--full` `4419b1fe` kept identity but regressed
/// general-long 1.038 → 1.025; unhooked from `standard.rs`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn qwen_compiled_split_verify_fa_attn_norm_qkv(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    layer_idx: usize,
    seq: usize,
) -> Option<(MlxArray, MlxArray, MlxArray)> {
    if !fastpath::qwen_linear_mtp_exact_enabled() {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") || cfg.attn_output_gate {
        return None;
    }
    if !(2..=4).contains(&(seq as i32)) {
        return None;
    }
    if w.qkv_packed.is_some() {
        return None;
    }
    let hidden_shape = hidden.shape();
    if hidden_shape.len() < 2 {
        return None;
    }
    let leading_elements: i64 = hidden_shape[..hidden_shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, dim| acc.checked_mul(i64::from(*dim)))?;
    if leading_elements != i64::from(seq as i32) {
        return None;
    }
    let q_proj = w.q_proj.as_ref()?;
    let k_proj = w.k_proj.as_ref()?;
    let v_proj = w.v_proj.as_ref()?;
    if q_proj.scales.is_none() || k_proj.scales.is_none() || v_proj.scales.is_none() {
        return None;
    }
    let mut inputs = vec![hidden.clone(), w.attn_norm.clone()];
    let q_slot = push_quant_inputs(&mut inputs, Some(q_proj))?;
    let k_slot = push_quant_inputs(&mut inputs, Some(k_proj))?;
    let v_slot = push_quant_inputs(&mut inputs, Some(v_proj))?;
    let schema = CompiledSplitVerifyFaAttnNormQkvSchema {
        q: q_slot,
        k: k_slot,
        v: v_slot,
    };
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    let outs = apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity ^ VERIFY_FA_ATTN_NORM_QKV_COMPILE_SALT,
        layer_idx,
        leading_elements,
        2,
        &input_refs,
        move |inputs: &MlxVectorArray| {
            let hidden = inputs.get(0);
            let attn_norm = inputs.get(1);
            let normed = rms_norm(&hidden, Some(&attn_norm), eps, None);
            let q_qw = schema.q.rebuild(inputs);
            let k_qw = schema.k.rebuild(inputs);
            let v_qw = schema.v.rebuild(inputs);
            let q = qw_with_policy(&normed, &q_qw, ProjectionBatchPolicy::Shared);
            let k = qw_with_policy(&normed, &k_qw, ProjectionBatchPolicy::Shared);
            let v = qw_with_policy(&normed, &v_qw, ProjectionBatchPolicy::Shared);
            vec![q, k, v]
        },
    )?;
    if outs.len() != 3 {
        return None;
    }
    Some((outs[0].clone(), outs[1].clone(), outs[2].clone()))
}

#[cfg(test)]
const VERIFY_O_PROJ_FFN_COMPILE_SALT: u64 = 0x4F50_524F_4A46_464E;

#[cfg(test)]
#[derive(Clone, Copy)]
struct CompiledSplitVerifyOProjFfnSchema {
    out_proj: QuantInputSlot,
    gate: QuantInputSlot,
    up: QuantInputSlot,
    down: QuantInputSlot,
}

#[cfg(test)]
fn flatten_split_verify_o_proj_ffn_inputs(
    hidden: &MlxArray,
    gated: &MlxArray,
    ffn_norm: &MlxArray,
    out_proj: &QuantizedWeight,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    down: &QuantizedWeight,
) -> Option<(Vec<MlxArray>, CompiledSplitVerifyOProjFfnSchema)> {
    let mut inputs = vec![hidden.clone(), gated.clone(), ffn_norm.clone()];
    let out_slot = push_quant_inputs(&mut inputs, Some(out_proj))?;
    let gate_slot = push_quant_inputs(&mut inputs, Some(gate))?;
    let up_slot = push_quant_inputs(&mut inputs, Some(up))?;
    let down_slot = push_quant_inputs(&mut inputs, Some(down))?;
    Some((
        inputs,
        CompiledSplitVerifyOProjFfnSchema {
            out_proj: out_slot,
            gate: gate_slot,
            up: up_slot,
            down: down_slot,
        },
    ))
}

/// Shape-compile `out_proj(gated) → add_rms_norm → split FFN → add residual`
/// for exact S=2..=4 linear-attention verify.
///
/// `gated` is the portable RMS+SiLU output (`[1, seq, value_dim]`). That gate
/// stays outside the closure.
#[cfg(test)]
pub(crate) fn qwen_compiled_split_verify_o_proj_ffn_plus_residual(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    gated: &MlxArray,
    layer_idx: usize,
) -> Option<MlxArray> {
    if !fastpath::qwen_linear_mtp_exact_enabled() || cfg.uses_geglu {
        return None;
    }
    if !cfg.model_family.starts_with("qwen") {
        return None;
    }
    if w.router_proj.is_some() || w.ffn_post_norm.is_some() {
        return None;
    }
    let hidden_shape = hidden.shape();
    if hidden_shape.len() < 2 {
        return None;
    }
    let leading_elements: i64 = hidden_shape[..hidden_shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, dim| acc.checked_mul(i64::from(*dim)))?;
    if !(2..=4).contains(&leading_elements) {
        return None;
    }
    let out_proj = &w.linear_attn.as_ref()?.out_proj;
    out_proj.scales.as_ref()?;
    let gate = w.gate_proj.as_ref()?;
    let up = w.up_proj.as_ref()?;
    let down = w.down_proj.as_ref()?;
    if gate.scales.is_none() || up.scales.is_none() || down.scales.is_none() {
        return None;
    }
    let (inputs, schema) = flatten_split_verify_o_proj_ffn_inputs(
        hidden,
        gated,
        &w.ffn_norm,
        out_proj,
        gate,
        up,
        down,
    )?;
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let eps = cfg.rms_norm_eps;
    let body = move |inputs: &MlxVectorArray| {
        let hidden = inputs.get(0);
        let gated = inputs.get(1);
        let ffn_norm = inputs.get(2);
        let out_proj = schema.out_proj.rebuild(inputs);
        let attn = qw_with_policy(&gated, &out_proj, ProjectionBatchPolicy::Shared);
        let (residual, normed) = add_rms_norm_pair(&hidden, &attn, &ffn_norm, eps, None);
        let gate_qw = schema.gate.rebuild(inputs);
        let up_qw = schema.up.rebuild(inputs);
        let down_qw = schema.down.rebuild(inputs);
        let gate = qw_with_policy(&normed, &gate_qw, ProjectionBatchPolicy::Shared);
        let up = qw_with_policy(&normed, &up_qw, ProjectionBatchPolicy::Shared);
        let act = silu_mul(&gate, &up, None);
        let ffn = qw_with_policy(&act, &down_qw, ProjectionBatchPolicy::Shared);
        vec![add(&residual, &ffn, None)]
    };
    apply_layer_dense_ffn_prefill_min(
        cfg.compile_cache_identity ^ VERIFY_O_PROJ_FFN_COMPILE_SALT,
        layer_idx,
        leading_elements,
        2,
        &input_refs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

#[allow(clippy::too_many_arguments)]
fn qwen_compiled_split_prefill_ffn(
    model_identity: u64,
    layer_idx: usize,
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    down: Option<&QuantizedWeight>,
    post_norm: Option<&MlxArray>,
    rms_norm_eps: f32,
    projection_policy: ProjectionBatchPolicy,
) -> Option<MlxArray> {
    let x_shape = x.shape();
    if x_shape.len() < 2 || x_shape[x_shape.len() - 2] <= 1 {
        return None;
    }
    let leading_elements: i64 = x_shape
        .iter()
        .rev()
        .skip(1)
        .map(|d| i64::from(*d))
        .product();
    if leading_elements < fastpath::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING {
        return None;
    }
    let down = down?;
    if gate.scales.is_none() || up.scales.is_none() || down.scales.is_none() {
        return None;
    }
    let (inputs, schema) = flatten_split_dense_ffn_inputs(x, gate, up, down, post_norm)?;
    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
    let body = move |inputs: &MlxVectorArray| {
        let x = inputs.get(0);
        let (gate_qw, up_qw, down_qw, post_norm_w) = schema.rebuild(inputs);
        let gate = qw_with_policy(&x, &gate_qw, projection_policy);
        let up = qw_with_policy(&x, &up_qw, projection_policy);
        let hidden = silu_mul(&gate, &up, None);
        let out = qw_with_policy(&hidden, &down_qw, projection_policy);
        if let Some(norm_w) = post_norm_w {
            vec![rms_norm(&out, Some(&norm_w), rms_norm_eps, None)]
        } else {
            vec![out]
        }
    };
    apply_layer_dense_ffn_prefill_min(
        model_identity,
        layer_idx,
        leading_elements,
        fastpath::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING,
        &input_refs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

/// Shape-compile only the Qwen split-prefill **down** qmm.
/// Full split FFN compile and flat-down stay OFF.
fn qwen_compiled_prefill_down_qmm(
    model_identity: u64,
    layer_idx: usize,
    hidden: &MlxArray,
    down: &QuantizedWeight,
) -> Option<MlxArray> {
    qwen_compiled_prefill_down_qmm_for(
        fastpath::qwen_prefill_down_compile_enabled(),
        model_identity,
        layer_idx,
        hidden,
        down,
    )
}

fn qwen_compiled_prefill_down_qmm_for(
    enabled: bool,
    model_identity: u64,
    layer_idx: usize,
    hidden: &MlxArray,
    down: &QuantizedWeight,
) -> Option<MlxArray> {
    let shape = hidden.shape();
    if shape.len() < 2 || shape[shape.len() - 2] <= 1 {
        return None;
    }
    let leading: i64 = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if !fastpath::should_qwen_prefill_down_compile_for(enabled, shape[shape.len() - 2], leading) {
        return None;
    }
    let scales = down.scales.as_ref()?;
    let biases = down.biases.as_ref()?;
    if down.group_size <= 0 || down.bits <= 0 {
        return None;
    }
    let inputs = [hidden, &down.weight, scales, biases];
    let group_size = down.group_size;
    let bits = down.bits;
    let mode = down.mlx_quantization_mode();
    let body = move |inputs: &MlxVectorArray| {
        let x = inputs.get(0);
        let weight = inputs.get(1);
        let scales = inputs.get(2);
        let biases = inputs.get(3);
        vec![quantized_matmul_with_mode(
            &x,
            &weight,
            &scales,
            Some(&biases),
            true,
            Some(group_size),
            Some(bits),
            mode,
            None,
        )]
    };
    apply_layer_dense_ffn_prefill_min(
        model_identity,
        layer_idx ^ 0xD0_00_00,
        leading,
        fastpath::QWEN_SPLIT_FFN_PREFILL_COMPILE_MIN_LEADING,
        &inputs,
        body,
    )
    .and_then(|r| r.into_iter().next())
}

/// Flatten leading dims and run the down affine qmm as a 2-D matmul.
/// Flag is call-site only so tests can drive the body after a wash flip.
/// Flatten `[B,S,H] → [B*S,H]` for Qwen prefill FFN qmm. Shipped by
/// [`ffn_swiglu_with_policy`] when [`fastpath::should_qwen_prefill_flat_ffn`].
pub(crate) fn flatten_qwen_prefill_ffn_activation(x: &MlxArray) -> (MlxArray, [i32; 3]) {
    let shape = x.shape();
    let batch = shape[0];
    let seq = shape[1];
    let hidden = shape[2];
    (
        reshape(x, &[batch * seq, hidden], None),
        [batch, seq, hidden],
    )
}

/// Restore `[B*S,H'] → [B,S,H']` after a flattened Qwen prefill FFN.
pub(crate) fn restore_qwen_prefill_ffn_activation(out: &MlxArray, orig: [i32; 3]) -> MlxArray {
    let out_last = out.shape().last().copied().unwrap_or(orig[2]);
    reshape(out, &[orig[0], orig[1], out_last], None)
}

thread_local! {
    static PREFILL_Q2_DOWN: RefCell<HashMap<(u64, usize), QuantizedWeight>> =
        RefCell::new(HashMap::new());
}

fn cached_prefill_q2_down(
    model_identity: u64,
    layer_idx: usize,
    src: &QuantizedWeight,
) -> Option<QuantizedWeight> {
    let key = (model_identity, layer_idx);
    PREFILL_Q2_DOWN.with(|slot| {
        if let Some(existing) = slot.borrow().get(&key) {
            return Some(existing.clone());
        }
        let made = crate::weights::requant_affine_to_prefill_q2(src)?;
        slot.borrow_mut().insert(key, made.clone());
        Some(made)
    })
}

thread_local! {
    static PREFILL_FFN_GS64: RefCell<HashMap<(u64, usize, u8), QuantizedWeight>> =
        RefCell::new(HashMap::new());
}

const PREFILL_FFN_GS64_PACKED: u8 = 0;
const PREFILL_FFN_GS64_GATE: u8 = 1;
const PREFILL_FFN_GS64_UP: u8 = 2;
const PREFILL_FFN_GS64_DOWN: u8 = 3;

fn cached_prefill_ffn_gs64(
    model_identity: u64,
    layer_idx: usize,
    slot: u8,
    src: &QuantizedWeight,
) -> Option<QuantizedWeight> {
    let key = (model_identity, layer_idx, slot);
    PREFILL_FFN_GS64.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let made = crate::weights::requant_affine_to_prefill_gs64(src)?;
        cache.borrow_mut().insert(key, made.clone());
        Some(made)
    })
}

thread_local! {
    static PREFILL_FFN_Q3: RefCell<HashMap<(u64, usize, u8), QuantizedWeight>> =
        RefCell::new(HashMap::new());
}

fn cached_prefill_ffn_q3(
    model_identity: u64,
    layer_idx: usize,
    slot: u8,
    src: &QuantizedWeight,
) -> Option<QuantizedWeight> {
    let key = (model_identity, layer_idx, slot);
    PREFILL_FFN_Q3.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let made = crate::weights::requant_affine_to_prefill_q3(src)?;
        cache.borrow_mut().insert(key, made.clone());
        Some(made)
    })
}

thread_local! {
    static PREFILL_FFN_CONTIG_W: RefCell<HashMap<(u64, usize, u8), QuantizedWeight>> =
        RefCell::new(HashMap::new());
    static PREFILL_ATTN_CONTIG_W: RefCell<HashMap<usize, QuantizedWeight>> =
        RefCell::new(HashMap::new());
    static PREFILL_SPLIT_PACKED: RefCell<HashMap<usize, (QuantizedWeight, QuantizedWeight)>> =
        RefCell::new(HashMap::new());
}

fn cached_prefill_split_packed_ffn(
    src: &QuantizedWeight,
) -> Option<(QuantizedWeight, QuantizedWeight)> {
    let key = src as *const QuantizedWeight as usize;
    PREFILL_SPLIT_PACKED.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let made = crate::weights::split_packed_ffn_gate_up(src)?;
        cache.borrow_mut().insert(key, made.clone());
        Some(made)
    })
}

/// Cache a contiguous overlay of one attention quantized projection.
pub(crate) fn cached_prefill_attn_contiguous_weight(src: &QuantizedWeight) -> QuantizedWeight {
    let key = src as *const QuantizedWeight as usize;
    PREFILL_ATTN_CONTIG_W.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return existing.clone();
        }
        let made = crate::weights::contiguous_affine_weight(src);
        cache.borrow_mut().insert(key, made.clone());
        made
    })
}

/// Submit FFN down qmm work before residual/next-layer rms is attached.
fn qwen_prefill_maybe_async_down(down_out: &MlxArray, qwen_dense_ffn: bool, seq: i32) {
    qwen_prefill_maybe_async_down_for(
        down_out,
        fastpath::qwen_prefill_async_down_enabled(),
        qwen_dense_ffn,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_down`].
pub(crate) fn qwen_prefill_maybe_async_down_for(
    down_out: &MlxArray,
    enabled: bool,
    qwen_dense_ffn: bool,
    seq: i32,
) {
    if qwen_dense_ffn && fastpath::should_qwen_prefill_async_down_for(enabled, seq) {
        async_eval(&[down_out]);
    }
}

/// Materialize the Qwen SwiGLU activation once before down qmm.
fn qwen_prefill_maybe_eval_ffn_hidden(h: &MlxArray, qwen_dense_ffn: bool, seq: i32) {
    qwen_prefill_maybe_eval_ffn_hidden_for(
        h,
        fastpath::qwen_prefill_eval_ffn_hidden_enabled(),
        qwen_dense_ffn,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_eval_ffn_hidden`].
pub(crate) fn qwen_prefill_maybe_eval_ffn_hidden_for(
    h: &MlxArray,
    enabled: bool,
    qwen_dense_ffn: bool,
    seq: i32,
) {
    if qwen_dense_ffn && fastpath::should_qwen_prefill_eval_ffn_hidden_for(enabled, seq) {
        mlx_sys::eval(&[h]);
    }
}

/// Submit packed gate+up qmm work before SwiGLU/down is attached.
fn qwen_prefill_maybe_async_packed_gate_up(packed: &MlxArray, qwen_dense_ffn: bool, seq: i32) {
    qwen_prefill_maybe_async_packed_gate_up_for(
        packed,
        fastpath::qwen_prefill_async_packed_gate_up_enabled(),
        qwen_dense_ffn,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_async_packed_gate_up`].
pub(crate) fn qwen_prefill_maybe_async_packed_gate_up_for(
    packed: &MlxArray,
    enabled: bool,
    qwen_dense_ffn: bool,
    seq: i32,
) {
    if qwen_dense_ffn && fastpath::should_qwen_prefill_async_packed_gate_up_for(enabled, seq) {
        async_eval(&[packed]);
    }
}

/// Submit gate/up qmm work before SwiGLU/down is attached. Qwen seq>=1024 only.
fn qwen_prefill_maybe_async_gate_up(
    gate: &MlxArray,
    up: &MlxArray,
    qwen_dense_ffn: bool,
    seq: i32,
) {
    if qwen_dense_ffn && fastpath::should_qwen_prefill_async_gate_up(seq) {
        async_eval(&[gate, up]);
    }
}

/// Materialize the Qwen dense FFN activation once before gate/up/packed qmm.
fn qwen_prefill_maybe_eval_ffn_input(x: &MlxArray, qwen_dense_ffn: bool, seq: i32) {
    qwen_prefill_maybe_eval_ffn_input_for(
        x,
        fastpath::qwen_prefill_eval_ffn_input_enabled(),
        qwen_dense_ffn,
        seq,
    );
}

/// Pure helper for [`qwen_prefill_maybe_eval_ffn_input`].
pub(crate) fn qwen_prefill_maybe_eval_ffn_input_for(
    x: &MlxArray,
    enabled: bool,
    qwen_dense_ffn: bool,
    seq: i32,
) {
    if qwen_dense_ffn && fastpath::should_qwen_prefill_eval_ffn_input_for(enabled, seq) {
        mlx_sys::eval(&[x]);
    }
}

fn cached_prefill_ffn_contiguous_weight(
    model_identity: u64,
    layer_idx: usize,
    slot: u8,
    src: &QuantizedWeight,
) -> Option<QuantizedWeight> {
    let key = (model_identity, layer_idx, slot);
    PREFILL_FFN_CONTIG_W.with(|cache| {
        if let Some(existing) = cache.borrow().get(&key) {
            return Some(existing.clone());
        }
        let made = crate::weights::contiguous_affine_weight(src);
        cache.borrow_mut().insert(key, made.clone());
        Some(made)
    })
}

fn qwen_prefill_flat_down_qmm(hidden: &MlxArray, down: &QuantizedWeight) -> Option<MlxArray> {
    let shape = hidden.shape();
    if shape.len() < 2 {
        return None;
    }
    let last = *shape.last()?;
    if last <= 0 || shape[shape.len() - 2] <= 1 {
        return None;
    }
    if down.scales.is_none() || down.group_size <= 0 || down.bits <= 0 {
        return None;
    }
    let leading: i64 = shape[..shape.len() - 1]
        .iter()
        .try_fold(1_i64, |acc, &dim| acc.checked_mul(i64::from(dim)))?;
    if leading <= 1 {
        return None;
    }
    let flat = reshape(hidden, &[leading as i32, last], None);
    let out = qw(&flat, down);
    let mut out_shape = shape;
    *out_shape.last_mut()? = *out.shape().last()?;
    Some(reshape(&out, &out_shape, None))
}

/// One C++ call: `silu(qmm(x,gate)) * qmm(x,up)` for Qwen split prefill.
/// Flag is call-site only so tests can drive the body after a wash flip.
fn qwen_dual_qmm_swiglu(
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
) -> Option<MlxArray> {
    let (g_s, u_s) = (gate.scales.as_ref()?, up.scales.as_ref()?);
    let (g_b, u_b) = (gate.biases.as_ref()?, up.biases.as_ref()?);
    if gate.group_size <= 0
        || gate.bits <= 0
        || up.group_size != gate.group_size
        || up.bits != gate.bits
    {
        return None;
    }
    dual_qmm_swiglu(
        x,
        &gate.weight,
        g_s,
        g_b,
        &up.weight,
        u_s,
        u_b,
        gate.group_size,
        gate.bits,
        None,
    )
}

/// Fuse `silu(gate)*up` into the down affine qmm for Qwen split prefill.
fn qwen_swiglu_down_fuse(
    gate: &MlxArray,
    up: &MlxArray,
    down: &QuantizedWeight,
) -> Option<MlxArray> {
    let scales = down.scales.as_ref()?;
    if !down.is_affine_quantized() || down.group_size <= 0 || down.bits <= 0 {
        return None;
    }
    silu_mul_quantized_matmul(
        gate,
        up,
        &down.weight,
        scales,
        down.biases.as_ref(),
        down.group_size,
        down.bits,
        None,
    )
}

/// Packed prefill FFN is unused on Gemma 4 last-only 1-token rows.
pub(crate) fn use_packed_dense_ffn_prefill(
    prefer_split: bool,
    has_packed: bool,
    skip_last_packed: bool,
) -> bool {
    !prefer_split && has_packed && !skip_last_packed
}

fn prefer_split_dense_ffn_gate_up(
    model_family: &str,
    qwen_dense_ffn: bool,
    seq: i32,
    leading_elements: i64,
    has_split_gate_up: bool,
) -> bool {
    let qwen_speculative_row_exact = fastpath::qwen_linear_mtp_exact_enabled()
        && qwen_dense_ffn
        && leading_elements > 1
        && leading_elements <= 4;
    // Gemma4 long-prefill historically preferred split gate/up (two qmatmuls)
    // over packed fixed-shape. Kill-switch `AX_MLX_GEMMA4_SPLIT_PREFILL_FFN=0`
    // forces packed + prefill-compile for pure thr A/B on M5 (S1 residual).
    // Unified shares the Gemma 4 text backbone; keep split-prefill optim when
    // convert emits `gemma4_unified` (DI-W0-002 family label honesty).
    let gemma4_split_prefill = (model_family == "gemma4" || model_family == "gemma4_unified")
        && seq >= GEMMA4_SPLIT_PREFILL_MIN_SEQ
        && leading_elements >= i64::from(GEMMA4_SPLIT_PREFILL_MIN_SEQ)
        && gemma4_split_prefill_ffn_enabled()
        && !fastpath::should_gemma4_packed_ffn_compile_p128(model_family, seq);
    has_split_gate_up
        && ((qwen_dense_ffn && seq == 1 && leading_elements == 1)
            || qwen_speculative_row_exact
            || gemma4_split_prefill)
}

fn gemma4_split_prefill_ffn_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_GEMMA4_SPLIT_PREFILL_FFN") {
            Ok(raw) => {
                let v = raw.trim();
                !(v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("off"))
            }
            // Default ON: prior 128/512/2048 A/B preferred split gate/up for
            // Gemma4 publication-shape prefill.
            Err(_) => true,
        }
    })
}

fn dense_ffn_prefill_compile_supported(model_family: &str, leading_elements: i64) -> bool {
    if model_family.starts_with("qwen") {
        return fastpath::should_qwen_packed_ffn_prefill_compile(model_family, leading_elements);
    }
    true
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
    let seq = hidden.shape().get(1).copied().unwrap_or(1) as usize;
    // Multi-token: upcast residual for router so expert selection matches
    // singleton pure-direct near-ties (4/6-bit MoE).
    let hidden_r = if seq > 1 && crate::fastpath::multi_token_f32_attention_enabled() {
        astype(hidden, MlxDtype::Float32, None)
    } else {
        hidden.clone()
    };
    let normed = rms_norm(&hidden_r, Some(combined_scale), cfg.rms_norm_eps, None);

    let expert_scores = qw(&normed, router_proj);
    let expert_scores = if seq > 1 && crate::fastpath::multi_token_f32_attention_enabled() {
        astype(&expert_scores, MlxDtype::Float32, None)
    } else {
        expert_scores
    };
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

    // Decode-shaped and eligible: from here on, the call either hits the
    // fused kernel or is a real fallback worth counting as route evidence.
    record_moe_router_fused_attempt();

    let Some((indices, weights)) = moe_router_fused_metal_apply(logits_f32, num_experts, top_k)
    else {
        record_moe_router_fused_fallback();
        return None;
    };

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
        record_moe_router_fused_fallback();
        return None;
    }
    record_moe_router_fused_hit();
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

/// DeepSeek V4 (Flash) router: `sqrt(softplus)` scoring with the standard
/// DeepSeek selection tail, plus hash-table routing on the leading
/// `num_hash_layers` MoE layers.
///
/// Scoring (vLLM `csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu`,
/// llama.cpp `LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` in
/// `llama-graph.cpp:1974-1977`): the gate matmul runs in f32 and
/// `probs = sqrt(softplus(logits))` with the numerically stable softplus form.
///
/// Learned routing reuses the `moe_router_deepseek_v3` tail: correction bias
/// added to a COPY for top-k selection only, optional group masking, optional
/// `norm_topk_prob` renormalisation, `routed_scaling_factor` scaling; gathered
/// weights always come from the unbiased probs.
///
/// Hash routing (layers with a `tid2eid` table instead of a correction bias —
/// `DeepseekV4Config::is_hash_routed_layer`): expert INDICES are gathered
/// from the `[vocab, topk]` table at `token_ids`; expert WEIGHTS still come
/// from the unbiased sqrtsoftplus probs at those indices (llama.cpp
/// `deepseek4.cpp:1331-1337`, `llama-graph.cpp:2028-2033`). `token_ids` is
/// `None` on a hash-routed layer is a contract violation.
pub(crate) fn moe_router_deepseek_v4(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    token_ids: Option<&MlxArray>,
) -> (MlxArray, MlxArray) {
    let router_proj = w
        .router_proj
        .as_ref()
        .expect("DeepSeek V4 MoE layer must have router_proj");
    // Gate matmul in f32 (llama.cpp forces GGML_PREC_F32 for sqrtsoftplus).
    let logits = qw(&astype(x, MlxDtype::Float32, None), router_proj);
    let probs = sqrt_softplus_scores(&logits);
    let last_axis = probs.ndim() as i32 - 1;

    let tid2eid = w.deepseek_v4.as_ref().and_then(|v4| v4.tid2eid.as_ref());
    let top_k_indices = if let Some(tid2eid) = tid2eid {
        // Hash-routed layer: indices come from the token→expert table, never
        // from the scored probs.
        let token_ids = token_ids
            .expect("DeepSeek V4 hash-routed MoE layer requires token_ids for tid2eid lookup");
        let ids = if matches!(token_ids.dtype(), MlxDtype::Int32 | MlxDtype::Uint32) {
            token_ids.clone()
        } else {
            astype(token_ids, MlxDtype::Int32, None)
        };
        let table = if tid2eid.dtype() == MlxDtype::Int32 {
            tid2eid.clone()
        } else {
            astype(tid2eid, MlxDtype::Int32, None)
        };
        // [vocab, topk] gathered at [1, seq] ids → [1, seq, topk].
        take(&table, &ids, 0, None)
    } else {
        // Selection scores: correction bias on a copy, then optional group
        // masking (no-ops when topk_group == n_group).
        let selection_scores = if let Some(bias) = w.router_correction_bias.as_ref() {
            add(&probs, &astype(bias, MlxDtype::Float32, None), None)
        } else {
            probs.clone()
        };
        let selection_scores =
            deepseek_group_expert_mask(cfg, &selection_scores, cfg.moe_n_group, cfg.moe_topk_group);
        let (indices, _) = top_k_by_argpartition(
            &selection_scores,
            cfg.moe_expert_count,
            cfg.moe_experts_per_token,
            false,
        );
        indices
    };

    // Weights always come from the unbiased sqrtsoftplus probs — for hash
    // routing this is exactly the gather at the table indices.
    let top_k_weights = take_along_axis(&probs, &top_k_indices, last_axis, None);

    // Optionally normalise top-k weights to sum to 1 (done in f32 for precision).
    let top_k_weights = if cfg.moe_experts_per_token > 1 && cfg.moe_norm_topk_prob {
        let denominator = sum_axis(&top_k_weights, last_axis, true, None);
        divide(&top_k_weights, &denominator, None)
    } else {
        top_k_weights
    };

    // Scale by routed_scaling_factor — still f32.
    let scaling = cfg.moe_routed_scaling_factor;
    let top_k_weights = if (scaling - 1.0).abs() > 1e-6 {
        scale_hidden(&top_k_weights, scaling)
    } else {
        top_k_weights
    };

    // dtype cast deferred to here — after all f32 arithmetic — matching the V3 router.
    let top_k_weights = astype(&top_k_weights, x.dtype(), None);

    (top_k_indices, top_k_weights)
}

/// `sqrt(softplus(x))` computed in f32 with the numerically stable softplus
/// form `max(x, 0) + log1p(exp(-|x|))` (vLLM `topk_softplus_sqrt_kernels.cu:100`).
fn sqrt_softplus_scores(logits: &MlxArray) -> MlxArray {
    let logits = astype(logits, MlxDtype::Float32, None);
    let zero = mlx_sys::ops::cached_scalar(0.0, MlxDtype::Float32);
    let relu = maximum(&logits, &zero, None);
    let abs = maximum(&logits, &negative(&logits, None), None);
    let exp_neg_abs = exp(&negative(&abs, None), None);
    let softplus = add(&relu, &log1p(&exp_neg_abs, None), None);
    power(
        &softplus,
        &mlx_sys::ops::cached_scalar(0.5, MlxDtype::Float32),
        None,
    )
}

/// GPT-OSS MoE router: top-k on raw logits, then softmax on the selected set.
///
/// Matches mlx-lm `gpt_oss.MLPBlock`:
///   g = router(x)
///   experts, indices = topk(g, k)
///   expert_weights = softmax(experts, precise=True)
///
/// Differs from Qwen3/Gemma4 (softmax-all then top-k, or argpartition then
/// softmax with different pre-norms). Do **not** full-softmax then top-k.
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

    let (top_k_indices, top_k_raw) = top_k_by_argpartition(
        &logits,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        false,
    );
    let top_k_weights = softmax_precise(&top_k_raw, last_axis, None);
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
        deepseek_v4: None,
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
        expert_stream: None,
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
    linear_bias: Option<usize>,
    group_size: i32,
    bits: i32,
    /// Quant mode string pointer length is small; store as fixed for Copy.
    /// Supported: "affine" | "mxfp4" | "mxfp8" | "nvfp4" (default affine).
    mode_tag: u8,
}

impl QuantInputSlot {
    fn mode_tag(mode: &str) -> u8 {
        match mode {
            "mxfp4" => 1,
            "mxfp8" => 2,
            "nvfp4" => 3,
            _ => 0, // affine / unknown
        }
    }

    /// Resolve the compile-time mode tag from the runtime quant contract so
    /// mislabeled affine 4/32 (no group bias) is stored as MXFP4, matching
    /// [`QuantizedWeight::mlx_quantization_mode`].
    fn mode_tag_from_quant(q: &QuantizedWeight) -> u8 {
        match q.mlx_quantization_mode() {
            mlx_sys::MlxQuantizationMode::Mxfp4 => Self::mode_tag("mxfp4"),
            mlx_sys::MlxQuantizationMode::Mxfp8 => Self::mode_tag("mxfp8"),
            mlx_sys::MlxQuantizationMode::Nvfp4 => Self::mode_tag("nvfp4"),
            mlx_sys::MlxQuantizationMode::Affine => Self::mode_tag("affine"),
        }
    }

    fn mode_str(tag: u8) -> &'static str {
        match tag {
            1 => "mxfp4",
            2 => "mxfp8",
            3 => "nvfp4",
            _ => "affine",
        }
    }

    fn rebuild(&self, inputs: &MlxVectorArray) -> QuantizedWeight {
        QuantizedWeight {
            weight: inputs.get(self.weight),
            scales: self.scales.map(|i| inputs.get(i)),
            biases: self.biases.map(|i| inputs.get(i)),
            group_size: self.group_size,
            bits: self.bits,
            mode: Self::mode_str(self.mode_tag).to_string(),
            linear_bias: self.linear_bias.map(|i| inputs.get(i)),
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
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
    let linear_bias = push_optional_input(inputs, q.linear_bias.as_ref());
    Some(QuantInputSlot {
        weight,
        scales,
        biases,
        linear_bias,
        group_size: q.group_size,
        bits: q.bits,
        mode_tag: QuantInputSlot::mode_tag_from_quant(q),
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

/// Index layout for Qwen-style **split** gate/up dense FFN compile.
///
/// Qwen decode prefers split gate/up so the optional matvec Metal kernel can
/// engage, which previously skipped the packed-gate_up compile path entirely.
/// Compiling the split graph (gate + up + SwiGLU + down) cuts per-step host
/// graph encoding for those layers on M5 Max Qwen3.5-9B.
#[derive(Clone, Copy)]
pub(crate) struct CompiledSplitDenseFfnSchema {
    gate: QuantInputSlot,
    up: QuantInputSlot,
    down: QuantInputSlot,
    post_norm: Option<usize>,
}

impl CompiledSplitDenseFfnSchema {
    pub(crate) fn rebuild(
        &self,
        inputs: &MlxVectorArray,
    ) -> (
        QuantizedWeight,
        QuantizedWeight,
        QuantizedWeight,
        Option<MlxArray>,
    ) {
        (
            self.gate.rebuild(inputs),
            self.up.rebuild(inputs),
            self.down.rebuild(inputs),
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

/// Flatten split gate/up/down weights for compiled Qwen decode FFN.
pub(crate) fn flatten_split_dense_ffn_inputs(
    x: &MlxArray,
    gate: &QuantizedWeight,
    up: &QuantizedWeight,
    down: &QuantizedWeight,
    post_norm: Option<&MlxArray>,
) -> Option<(Vec<MlxArray>, CompiledSplitDenseFfnSchema)> {
    let mut inputs: Vec<MlxArray> = vec![x.clone()];
    let gate_slot = push_quant_inputs(&mut inputs, Some(gate))?;
    let up_slot = push_quant_inputs(&mut inputs, Some(up))?;
    let down_slot = push_quant_inputs(&mut inputs, Some(down))?;
    let post_norm_idx = push_optional_input(&mut inputs, post_norm);
    Some((
        inputs,
        CompiledSplitDenseFfnSchema {
            gate: gate_slot,
            up: up_slot,
            down: down_slot,
            post_norm: post_norm_idx,
        },
    ))
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
// Tier 2A: Deep expert-block fusion — decode-only (compositional Metal).
//
// When `AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL` is on, decode runs gather_qmm
// gate_up → Metal fused activation/unsort → gather_qmm down → Metal
// weighted-sum under one outer entry. A true single-dispatch 4-bit
// mega-kernel remains residual.
// ---------------------------------------------------------------------------

/// Attempt deep expert-block fusion for MoE decode.
///
/// Returns `Some(output)` if the compositional Metal path succeeds, `None` to
/// fall back to the standard multi-dispatch path.
///
/// Gated by `AX_MLX_MOE_DEEP_EXPERT_BLOCK_METAL` (default OFF).
fn try_moe_deep_expert_block_metal(
    cfg: &ModelConfig,
    gate_up_exps_packed: Option<&QuantizedWeight>,
    down_exps: Option<&QuantizedWeight>,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
) -> Option<MlxArray> {
    if !fastpath::moe_deep_expert_block_metal_enabled() {
        return None;
    }
    // Compositional deep expert block (decode): gather_qmm gate_up → fused
    // Metal activation/unsort → gather_qmm down → Metal weighted-sum.
    // Single-dispatch 4-bit mega-kernel remains residual for further TTFT
    // gains; this path still uses the shipped Metal kernels end-to-end when
    // eligible and fails closed (None) otherwise.
    let seq = x.shape().get(1).copied().unwrap_or(1) as usize;
    let batch = x.shape().first().copied().unwrap_or(1) as usize;
    if seq != 1 || batch != 1 {
        return None;
    }
    let packed = gate_up_exps_packed?;
    let down_exps = down_exps?;
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let gather_inputs = switch_gather_inputs(&x_exp, top_k_indices);
    if gather_inputs.sorted_indices {
        // Fused activation/unsort path expects unsorted gather.
        return None;
    }
    let gate_up = qw_gather(
        &gather_inputs.x,
        packed,
        &gather_inputs.indices,
        gather_inputs.sorted_indices,
    );
    let half = cfg.moe_expert_intermediate_size as i32;
    let top_k = top_k_indices.shape().last().copied().unwrap_or(0);
    let activated = moe_fused_activation_unsort_metal(
        &gate_up,
        gather_inputs.inv_order.as_ref().unwrap_or(top_k_indices),
        half,
        top_k,
        gate_up.dtype(),
        cfg.uses_geglu,
    )?;
    // Down projection via gather on activated expert tokens.
    let activated_exp = expand_dims_axes(&activated, &[-2], None);
    let down_gather = switch_gather_inputs(&activated_exp, top_k_indices);
    let down_out = qw_gather(
        &down_gather.x,
        down_exps,
        &down_gather.indices,
        down_gather.sorted_indices,
    );
    // Prefer Metal weighted sum when available.
    if let Some(summed) = qwen3_moe_weighted_sum_metal(&down_out, top_k_weights, down_out.dtype()) {
        return Some(summed);
    }
    if let Some(summed) = gemma4_moe_weighted_sum_metal(&down_out, top_k_weights, down_out.dtype())
    {
        return Some(summed);
    }
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
    // Leading batch dim. Two decode-only Metal fast paths below hardcode a
    // single batch lane in their grid/output shape (`moe_fused_activation_unsort_metal`
    // emits `[1, 1, top_k, hidden]`, grid `top_k*hidden`; the deep-expert-block
    // stub's contract is likewise batch=1). They engage on `seq == 1`, which is
    // also true for a `[B, 1, hidden]` batched-decode cohort — so gate them on
    // `batch == 1` too, or lanes 1..B-1 are silently dropped. The remaining MoE
    // path (gather_qmm + packed SwiGLU/GeGLU + portable activation) is
    // batch-general, so `[B, 1, hidden]` decodes correctly through it.
    let batch = x.shape().first().copied().unwrap_or(1) as usize;
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

    // DeepSeek V4's clamped SwiGLU has no fused-kernel equivalent: every
    // activation-fusing Metal fast path must stay off so the clamped split
    // path in `dense_ffn_activation` runs.
    let v4_swiglu_clamp = deepseek_v4_swiglu_limit(cfg).is_some();

    // SSD expert streaming: when the layer's fused expert stack is not
    // resident, page it in here. Every kernel path below then runs unchanged
    // (same gather_qmm) on the returned QuantizedWeight values.
    let paged_experts = if w.gate_up_exps_packed.is_none()
        && w.gate_exps.is_none()
        && w.up_exps.is_none()
        && w.down_exps.is_none()
    {
        w.expert_stream.as_ref().map(|source| {
            source
                .stack()
                .expect("expert stream paging failed for MoE layer")
        })
    } else {
        None
    };
    let gate_up_exps_packed = w.gate_up_exps_packed.as_ref().or_else(|| {
        paged_experts
            .as_ref()
            .and_then(|stack| stack.gate_up_exps_packed.as_ref())
    });
    let gate_exps = w.gate_exps.as_ref().or_else(|| {
        paged_experts
            .as_ref()
            .and_then(|stack| stack.gate_exps.as_ref())
    });
    let up_exps = w.up_exps.as_ref().or_else(|| {
        paged_experts
            .as_ref()
            .and_then(|stack| stack.up_exps.as_ref())
    });
    let down_exps_ref = w.down_exps.as_ref().or_else(|| {
        paged_experts
            .as_ref()
            .and_then(|stack| stack.down_exps.as_ref())
    });

    // Tier 2A: try deep expert-block fusion (decode-only). Fuses gather_qmm
    // gate_up + SwiGLU + gather_qmm down + weighted-sum into one dispatch.
    // Falls back to the standard multi-dispatch path when ineligible.
    if seq == 1
        && batch == 1
        && !v4_swiglu_clamp
        && let Some(out) = try_moe_deep_expert_block_metal(
            cfg,
            gate_up_exps_packed,
            down_exps_ref,
            x,
            top_k_indices,
            top_k_weights,
        )
    {
        return out;
    }

    // Match MLX SwitchGLU: [batch, seq, hidden] → [batch, seq, 1, 1, hidden].
    // The extra singleton before top_k is required by gather_mm/gather_qmm broadcasting.
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let gather_inputs = switch_gather_inputs(&x_exp, top_k_indices);
    let down_exps = down_exps_ref.expect("MoE layer must have down_exps");

    // Phase 1B: when the expert gate_up is packed and the flag is on, try the
    // packed SwiGLU Metal kernel directly on the gather_qmm output, fusing the
    // last-dim split + SiLU + multiply into one dispatch. Decode-only (seq==1):
    // at prefill the tensor is large and bandwidth-bound, where the separate
    // slice+silu_mul ops are faster than the single packed dispatch. Falls back
    // to the split-activation path when the kernel is ineligible or at prefill.
    let hidden = if let Some(packed) = gate_up_exps_packed {
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
        let moe_packed_geglu_ok = cfg.uses_geglu
            && fastpath::moe_geglu_packed_metal_enabled()
            && (seq == 1 || seq <= fastpath::MOE_PACKED_GEGLU_PREFILL_MAX_SEQ);
        let fused = if moe_packed_geglu_ok {
            packed_geglu_metal_impl(&out, half)
        } else if !cfg.uses_geglu
            && seq == 1
            && !v4_swiglu_clamp
            && fastpath::moe_swiglu_packed_metal_enabled()
        {
            packed_swiglu_metal_impl(&out, half)
        } else if seq == 1
            && batch == 1
            && !v4_swiglu_clamp
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
    } else if let Some(gate_exps) = gate_exps {
        let gate_up_started = Instant::now();
        let gate_out = qw_gather(
            &gather_inputs.x,
            gate_exps,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        let up_exps = up_exps.expect("MoE layer must have up_exps");
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
    } else {
        // Nemotron-H ReLU² experts: only up (fc1) + down (fc2), no SwiGLU gate.
        let gate_up_started = Instant::now();
        let up_exps = up_exps.expect("ReLU2 MoE layer must have up_exps");
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
            &[&up_out],
        );
        if profile_moe {
            record_moe_profile_stage(
                MoeProfileStage::ExpertGateUp,
                saturating_profile_us(gate_up_started),
            );
        }
        let activation_started = Instant::now();
        let zero = zeros(&[], up_out.dtype(), None);
        let relu = maximum(&up_out, &zero, None);
        let h = multiply(&relu, &relu, None);
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
/// Multi-token teacher-forced windows are short (depth-2 → S≤3, K=8 → 24
/// selections) but still benefit from expert-id sort so gather_qmm streams
/// the same expert weights contiguously (unique-expert amortization). The
/// final output is unsorted back to original order — bit-identical layout
/// restore after a permutation of independent rows.
const SWITCH_GLU_SORT_THRESHOLD_MT: usize = 8;

pub(crate) fn switch_gather_inputs(
    x_expanded: &MlxArray,
    indices: &MlxArray,
) -> SwitchGatherInputs {
    let indices_shape = indices.shape();
    let selection_count = shape_element_count(&indices_shape);
    let top_k = indices_shape.last().copied().unwrap_or(1).max(1) as usize;
    // Prefer expert-id sort whenever multi-token (seq>1 in the expanded
    // layout) or selection count is large. Multi-token exact path stacks
    // S positions into one gather; sorting amortizes unique-expert weight
    // traffic without changing per-row gather arithmetic after unsort.
    let seq_axis = indices_shape
        .get(indices_shape.len().saturating_sub(2))
        .copied()
        .unwrap_or(1);
    let multi_token = seq_axis > 1;
    let sort_threshold = if multi_token {
        SWITCH_GLU_SORT_THRESHOLD_MT
    } else {
        SWITCH_GLU_SORT_THRESHOLD
    };
    if selection_count < sort_threshold {
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
        MlxQuantizationMode, add, astype, concatenate, eval, quantize, quantized_matmul, slice,
        slice_last_dim,
    };

    #[test]
    fn dense_ffn_prefill_compile_keeps_qwen_imperative_by_default() {
        assert!(!dense_ffn_prefill_compile_supported("qwen3_5", 512));
        assert!(!dense_ffn_prefill_compile_supported("qwen3_next", 128));
        assert!(dense_ffn_prefill_compile_supported("gemma4", 512));
        assert!(
            !dense_ffn_prefill_compile_supported("qwen3_5", 1024),
            "Qwen packed prefill compile stays opt-in after community 3d wash"
        );
        assert!(fastpath::should_qwen_packed_ffn_prefill_compile_for(
            true, "qwen3_5", 1024
        ));
    }

    #[test]
    fn qwen_compiled_split_prefill_gate_up_stays_opt_in_after_wash() {
        let x = array_f32(&vec![0.1; 128], &[1, 2, 64]);
        let w = QuantizedWeight {
            weight: array_f32(&vec![0.0; 64], &[1, 64]),
            scales: Some(array_f32(&[1.0], &[1])),
            biases: Some(array_f32(&[0.0], &[1])),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        assert!(
            qwen_compiled_split_prefill_gate_up(&x, &w, &w).is_none(),
            "dual-qmm compile must stay default-OFF after the 890.96 vs 891 wash"
        );
    }

    #[test]
    fn qwen_prefill_dual_affine_gate_up_matches_two_qmm() {
        let seq = 1024i32;
        let hidden = 64i32;
        let inter = 32i32;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i as f32) - 2048.0) * 0.000_244_140_63)
            .collect();
        let gate_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) - 1024.0) * 0.0005)
            .collect();
        let up_data: Vec<f32> = (0..inter * hidden)
            .map(|i| ((i as f32) - 512.0) * -0.0004)
            .collect();
        let x = array_f32(&x_data, &[1, seq, hidden]);
        let gq = quantize(
            &array_f32(&gate_data, &[inter, hidden]),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let uq = quantize(
            &array_f32(&up_data, &[inter, hidden]),
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let gate_w = QuantizedWeight {
            weight: gq[0].clone(),
            scales: Some(gq[1].clone()),
            biases: Some(gq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let up_w = QuantizedWeight {
            weight: uq[0].clone(),
            scales: Some(uq[1].clone()),
            biases: Some(uq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        assert!(
            qwen_prefill_maybe_dual_affine_gate_up_for(false, "qwen3_5", seq, &x, &gate_w, &up_w)
                .is_none()
        );
        let (g_dual, u_dual) =
            qwen_prefill_maybe_dual_affine_gate_up_for(true, "qwen3_5", seq, &x, &gate_w, &up_w)
                .expect("Qwen dual-affine qmm must engage at the p2048 chunk length");
        let g_ref = qw(&x, &gate_w);
        let u_ref = qw(&x, &up_w);
        eval(&[&g_dual, &u_dual, &g_ref, &u_ref]);
        let a = g_dual.data_f32();
        let b = g_ref.data_f32();
        assert_eq!(a.len(), b.len());
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        for (l, r) in u_dual.data_f32().iter().zip(u_ref.data_f32().iter()) {
            max_abs = max_abs.max((l - r).abs());
        }
        assert!(
            max_abs < 1e-4,
            "dual-affine qmm must match two steel qw, max_abs={max_abs}"
        );
        assert!(
            fastpath::should_qwen_prefill_dual_affine_qmm_for(true, "qwen3_5", 1024),
            "shipped dual-affine gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qwen_compiled_split_prefill_ffn_matches_two_qmm_4bit_gs32() {
        // Contract p128 leading=128. AXQ language FFN is 4-bit gs32.
        let x_data: Vec<f32> = (0..128 * 64)
            .map(|i| ((i as f32) - 4096.0) * 0.000_244_140_63)
            .collect();
        let gate_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0005)
            .collect();
        let up_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 512.0) * -0.0004)
            .collect();
        let down_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let x = array_f32(&x_data, &[1, 128, 64]);
        let gate_w = array_f32(&gate_data, &[32, 64]);
        let up_w = array_f32(&up_data, &[32, 64]);
        let down_w = array_f32(&down_data, &[64, 32]);
        let gq = quantize(
            &gate_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let uq = quantize(
            &up_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gq.len(), 3);
        assert_eq!(uq.len(), 3);
        assert_eq!(dq.len(), 3);
        let qweight = |q: &[MlxArray]| QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let gate = qweight(&gq);
        let up = qweight(&uq);
        let down = qweight(&dq);
        let compiled = qwen_compiled_split_prefill_ffn(
            0x5157_454e_5052_4546,
            0,
            &x,
            &gate,
            &up,
            Some(&down),
            None,
            1e-6,
            ProjectionBatchPolicy::Shared,
        )
        .expect("Qwen 4-bit gs32 split FFN prefill compile should engage at leading=128");
        let p_gate = quantized_matmul(
            &x,
            &gq[0],
            &gq[1],
            Some(&gq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let p_up = quantized_matmul(
            &x,
            &uq[0],
            &uq[1],
            Some(&uq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let hidden = silu_mul(&p_gate, &p_up, None);
        let portable = quantized_matmul(
            &hidden,
            &dq[0],
            &dq[1],
            Some(&dq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        assert_close(compiled.data_f32(), portable.data_f32(), 3.0e-2);
        let decode = array_f32(&x_data[..64], &[1, 1, 64]);
        assert!(
            qwen_compiled_split_prefill_ffn(
                0x5157_454e_5052_4546,
                0,
                &decode,
                &gate,
                &up,
                Some(&down),
                None,
                1e-6,
                ProjectionBatchPolicy::Shared,
            )
            .is_none(),
            "split FFN prefill compile must reject decode seq==1"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_ffn_mxfp4_s2_matches_imperative() {
        let seq = 2i32;
        let hidden = 64i32;
        let intermediate = 32i32;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.01)
            .collect();
        let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.008)
            .collect();
        let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.006)
            .collect();
        let x = array_f32(&x_data, &[1, seq, hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            assert_eq!(q.len(), 2);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let gate = qmx(&array_f32(&gate_data, &[intermediate, hidden]));
        let up = qmx(&array_f32(&up_data, &[intermediate, hidden]));
        let down = qmx(&array_f32(&down_data, &[hidden, intermediate]));
        let compiled = qwen_compiled_split_verify_ffn(
            0x5652_4659_4d58_5034,
            0,
            &x,
            &gate,
            &up,
            Some(&down),
            None,
            1e-6,
            ProjectionBatchPolicy::Shared,
        )
        .expect("exact S=2 MXFP4 split FFN compile must engage");
        let g = qw(&x, &gate);
        let u = qw(&x, &up);
        let hidden_act = silu_mul(&g, &u, None);
        let portable = qw(&hidden_act, &down);
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        let a = compiled.data_f32();
        let b = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled exact S=2 MXFP4 FFN must match imperative, max_abs={max_abs}"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_ffn_plus_residual_s2_matches_imperative() {
        let seq = 2i32;
        let hidden = 64i32;
        let intermediate = 32i32;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let attn_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.0078125)
            .collect();
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.01)
            .collect();
        let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.008)
            .collect();
        let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.006)
            .collect();
        let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
        let attn = array_f32(&attn_data, &[1, seq, hidden]);
        let ffn_norm = array_f32(&norm_data, &[hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x5245_5349_4446_464E;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy, &hidden_x);
        w.router_proj = None;
        w.ffn_norm = ffn_norm.clone();
        w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
        w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
        w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled = qwen_compiled_split_verify_ffn_plus_residual(&cfg, &w, &hidden_x, &attn, 0)
            .expect("exact S=2 residual+FFN compile must engage");
        let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
        let g = qw(&normed, w.gate_proj.as_ref().unwrap());
        let u = qw(&normed, w.up_proj.as_ref().unwrap());
        let act = silu_mul(&g, &u, None);
        let ffn = qw(&act, w.down_proj.as_ref().unwrap());
        let portable = add(&residual, &ffn, None);
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        let a = compiled.data_f32();
        let b = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled residual+FFN must match add_rms+ffn+add, max_abs={max_abs}"
        );
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            qwen_compiled_split_verify_ffn_plus_residual(&cfg, &w, &hidden_x, &attn, 0).is_none(),
            "residual+FFN compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_o_proj_ffn_plus_residual_s2_matches_imperative() {
        let seq = 2i32;
        let hidden = 64i32;
        let value_dim = 64i32;
        let intermediate = 32i32;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let gated_data: Vec<f32> = (0..(seq * value_dim) as usize)
            .map(|i| ((i as f32) - 6.0) * 0.03125)
            .collect();
        let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
            .map(|i| ((i as f32) - 20.0) * 0.004)
            .collect();
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.01)
            .collect();
        let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.008)
            .collect();
        let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.006)
            .collect();
        let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
        let gated = array_f32(&gated_data, &[1, seq, value_dim]);
        let ffn_norm = array_f32(&norm_data, &[hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x4F50_524F_4A46_464E;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy.clone(), &hidden_x);
        w.router_proj = None;
        w.ffn_norm = ffn_norm.clone();
        w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
        w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
        w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
        let out_proj = qmx(&array_f32(&o_data, &[hidden, value_dim]));
        w.linear_attn = Some(crate::weights::LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: None,
            in_proj_ba: None,
            fused_qkvz_ba: None,
            prefill_q2_qkvz: None,
            prefill_q2_ba: None,
            conv1d_dense: dummy.clone(),
            conv1d_bias: None,
            dt_bias: dummy.clone(),
            a_log: dummy,
            d: None,
            norm: ffn_norm.clone(),
            out_proj,
        });
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled =
            qwen_compiled_split_verify_o_proj_ffn_plus_residual(&cfg, &w, &hidden_x, &gated, 0)
                .expect("exact S=2 o_proj+residual+FFN compile must engage");
        let attn = qw(&gated, &w.linear_attn.as_ref().unwrap().out_proj);
        let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
        let g = qw(&normed, w.gate_proj.as_ref().unwrap());
        let u = qw(&normed, w.up_proj.as_ref().unwrap());
        let act = silu_mul(&g, &u, None);
        let ffn = qw(&act, w.down_proj.as_ref().unwrap());
        let portable = add(&residual, &ffn, None);
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        let a = compiled.data_f32();
        let b = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled o_proj+residual+FFN must match qw+add_rms+ffn+add, max_abs={max_abs}"
        );
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            qwen_compiled_split_verify_o_proj_ffn_plus_residual(&cfg, &w, &hidden_x, &gated, 0)
                .is_none(),
            "o_proj+FFN compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_fa_o_proj_ffn_s2_matches_imperative() {
        let seq = 2i32;
        let n_heads = 2usize;
        let head_dim = 32usize;
        let value_dim = (n_heads * head_dim) as i32;
        let hidden = 64i32;
        let intermediate = 32i32;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let sdpa_data: Vec<f32> = (0..(n_heads * seq as usize * head_dim))
            .map(|i| ((i as f32) - 10.0) * 0.0234375)
            .collect();
        let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
            .map(|i| ((i as f32) - 20.0) * 0.004)
            .collect();
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.01)
            .collect();
        let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.008)
            .collect();
        let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.006)
            .collect();
        let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
        let attn_sdpa = array_f32(&sdpa_data, &[1, n_heads as i32, seq, head_dim as i32]);
        let ffn_norm = array_f32(&norm_data, &[hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x4641_4F50_4646_4E32;
        cfg.n_heads = n_heads;
        cfg.head_dim = head_dim;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy, &hidden_x);
        w.router_proj = None;
        w.ffn_norm = ffn_norm.clone();
        w.o_proj = Some(qmx(&array_f32(&o_data, &[hidden, value_dim])));
        w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
        w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
        w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled = qwen_compiled_split_verify_fa_o_proj_ffn(
            &cfg,
            &w,
            &hidden_x,
            &attn_sdpa,
            0,
            seq as usize,
            n_heads,
            head_dim,
        )
        .expect("exact S=2 FA flatten+o_proj+FFN compile must engage");
        let transposed = transpose(&attn_sdpa, &[0, 2, 1, 3], None);
        let flat = reshape(&transposed, &[1, seq, value_dim], None);
        let attn = qw(&flat, w.o_proj.as_ref().unwrap());
        let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
        let g = qw(&normed, w.gate_proj.as_ref().unwrap());
        let u = qw(&normed, w.up_proj.as_ref().unwrap());
        let act = silu_mul(&g, &u, None);
        let ffn = qw(&act, w.down_proj.as_ref().unwrap());
        let portable = add(&residual, &ffn, None);
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        let a = compiled.data_f32();
        let b = portable.data_f32();
        let mut max_abs = 0.0f32;
        for i in 0..a.len() {
            max_abs = max_abs.max((a[i] - b[i]).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled FA flatten+o_proj+FFN must match transpose+qw+add_rms+ffn, max_abs={max_abs}"
        );
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            qwen_compiled_split_verify_fa_o_proj_ffn(
                &cfg,
                &w,
                &hidden_x,
                &attn_sdpa,
                0,
                seq as usize,
                n_heads,
                head_dim,
            )
            .is_none(),
            "FA o_proj+FFN compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_la_gate_o_proj_s2_matches_imperative() {
        let seq = 2i32;
        let hv = 2i32;
        let dv = 32i32;
        let value_dim = hv * dv;
        let n = (seq * hv * dv) as usize;
        let gd_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
        let z_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 8.0) * 0.015625).collect();
        let norm_data: Vec<f32> = (0..dv as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let o_data: Vec<f32> = (0..(value_dim * value_dim) as usize)
            .map(|i| ((i as f32) - 20.0) * 0.004)
            .collect();
        let gd = astype(
            &array_f32(&gd_data, &[1, seq, hv, dv]),
            MlxDtype::Bfloat16,
            None,
        );
        let z = astype(
            &array_f32(&z_data, &[1, seq, hv, dv]),
            MlxDtype::Bfloat16,
            None,
        );
        let la_norm = astype(&array_f32(&norm_data, &[dv]), MlxDtype::Bfloat16, None);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x4C41_4741_5445_4F50;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy.clone(), &array_f32(&[0.0; 2], &[1, 2]));
        w.router_proj = None;
        let out_proj = qmx(&array_f32(&o_data, &[value_dim, value_dim]));
        w.linear_attn = Some(crate::weights::LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: None,
            in_proj_ba: None,
            fused_qkvz_ba: None,
            prefill_q2_qkvz: None,
            prefill_q2_ba: None,
            conv1d_dense: dummy.clone(),
            conv1d_bias: None,
            dt_bias: dummy.clone(),
            a_log: dummy,
            d: None,
            norm: la_norm.clone(),
            out_proj: out_proj.clone(),
        });
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled =
            qwen_compiled_split_verify_la_gate_o_proj(&cfg, &w, &gd, &z, 0, seq, value_dim)
                .expect("exact S=2 LA gate+o_proj compile must engage");
        let normed = rms_norm(&gd, Some(&la_norm), 1e-6, None);
        let gated = astype(
            &silu_mul(
                &astype(&z, MlxDtype::Float32, None),
                &astype(&normed, MlxDtype::Float32, None),
                None,
            ),
            gd.dtype(),
            None,
        );
        let flat = reshape(&gated, &[1, seq, value_dim], None);
        let portable = qw(&flat, &out_proj);
        eval(&[&compiled, &portable]);
        let a = astype(&compiled, MlxDtype::Float32, None);
        let b = astype(&portable, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let mut max_abs = 0.0f32;
        for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
            max_abs = max_abs.max((l - r).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled LA gate+o_proj must match rms+silu+reshape+qw, max_abs={max_abs}"
        );
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            qwen_compiled_split_verify_la_gate_o_proj(&cfg, &w, &gd, &z, 0, seq, value_dim)
                .is_none(),
            "LA gate+o_proj compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_la_gate_o_proj_ffn_s2_matches_imperative() {
        let seq = 2i32;
        let hv = 2i32;
        let dv = 32i32;
        let value_dim = hv * dv;
        let hidden = 64i32;
        let intermediate = 32i32;
        let n = (seq * hv * dv) as usize;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let gd_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 16.0) * 0.03125).collect();
        let z_data: Vec<f32> = (0..n).map(|i| ((i as f32) - 8.0) * 0.015625).collect();
        let la_norm_data: Vec<f32> = (0..dv as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let ffn_norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.8 + (i as f32) * 0.003)
            .collect();
        let o_data: Vec<f32> = (0..(hidden * value_dim) as usize)
            .map(|i| ((i as f32) - 20.0) * 0.004)
            .collect();
        let gate_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 16.0) * 0.01)
            .collect();
        let up_data: Vec<f32> = (0..(intermediate * hidden) as usize)
            .map(|i| ((i as f32) - 8.0) * -0.008)
            .collect();
        let down_data: Vec<f32> = (0..(hidden * intermediate) as usize)
            .map(|i| ((i as f32) - 12.0) * 0.006)
            .collect();
        let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
        let gd = astype(
            &array_f32(&gd_data, &[1, seq, hv, dv]),
            MlxDtype::Bfloat16,
            None,
        );
        let z = astype(
            &array_f32(&z_data, &[1, seq, hv, dv]),
            MlxDtype::Bfloat16,
            None,
        );
        let la_norm = astype(&array_f32(&la_norm_data, &[dv]), MlxDtype::Bfloat16, None);
        let ffn_norm = array_f32(&ffn_norm_data, &[hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x4C41_474F_4646_4E32;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy.clone(), &hidden_x);
        w.router_proj = None;
        w.ffn_norm = ffn_norm.clone();
        w.gate_proj = Some(qmx(&array_f32(&gate_data, &[intermediate, hidden])));
        w.up_proj = Some(qmx(&array_f32(&up_data, &[intermediate, hidden])));
        w.down_proj = Some(qmx(&array_f32(&down_data, &[hidden, intermediate])));
        let out_proj = qmx(&array_f32(&o_data, &[hidden, value_dim]));
        w.linear_attn = Some(crate::weights::LinearAttentionWeights {
            in_proj_qkv: None,
            in_proj_z: None,
            in_proj_a: None,
            in_proj_b: None,
            in_proj_qkvz: None,
            in_proj_ba: None,
            fused_qkvz_ba: None,
            prefill_q2_qkvz: None,
            prefill_q2_ba: None,
            conv1d_dense: dummy.clone(),
            conv1d_bias: None,
            dt_bias: dummy.clone(),
            a_log: dummy,
            d: None,
            norm: la_norm.clone(),
            out_proj: out_proj.clone(),
        });
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let compiled = qwen_compiled_split_verify_la_gate_o_proj_ffn(
            &cfg, &w, &hidden_x, &gd, &z, 0, seq, value_dim,
        )
        .expect("exact S=2 LA gate+o_proj+FFN compile must engage");
        let normed = rms_norm(&gd, Some(&la_norm), 1e-6, None);
        let gated = astype(
            &silu_mul(
                &astype(&z, MlxDtype::Float32, None),
                &astype(&normed, MlxDtype::Float32, None),
                None,
            ),
            gd.dtype(),
            None,
        );
        let flat = reshape(&gated, &[1, seq, value_dim], None);
        let attn = qw(&flat, &out_proj);
        let (residual, normed) = add_rms_norm_pair(&hidden_x, &attn, &ffn_norm, 1e-6, None);
        let g = qw(&normed, w.gate_proj.as_ref().unwrap());
        let u = qw(&normed, w.up_proj.as_ref().unwrap());
        let act = silu_mul(&g, &u, None);
        let ffn = qw(&act, w.down_proj.as_ref().unwrap());
        let portable = add(&residual, &ffn, None);
        eval(&[&compiled, &portable]);
        let a = astype(&compiled, MlxDtype::Float32, None);
        let b = astype(&portable, MlxDtype::Float32, None);
        eval(&[&a, &b]);
        let mut max_abs = 0.0f32;
        for (l, r) in a.data_f32().iter().zip(b.data_f32().iter()) {
            max_abs = max_abs.max((l - r).abs());
        }
        assert!(
            max_abs < 1.0e-5,
            "compiled LA gate+o_proj+FFN must match imperative, max_abs={max_abs}"
        );
    }

    #[test]
    fn qwen_compiled_split_verify_fa_attn_norm_qkv_s2_matches_imperative() {
        let seq = 2i32;
        let hidden = 64i32;
        let q_out = 64i32;
        let kv_out = 32i32;
        let x_data: Vec<f32> = (0..(seq * hidden) as usize)
            .map(|i| ((i as f32) - 32.0) * 0.015625)
            .collect();
        let norm_data: Vec<f32> = (0..hidden as usize)
            .map(|i| 0.75 + (i as f32) * 0.004)
            .collect();
        let q_data: Vec<f32> = (0..(q_out * hidden) as usize)
            .map(|i| ((i as f32) - 10.0) * 0.004)
            .collect();
        let k_data: Vec<f32> = (0..(kv_out * hidden) as usize)
            .map(|i| ((i as f32) - 6.0) * 0.005)
            .collect();
        let v_data: Vec<f32> = (0..(kv_out * hidden) as usize)
            .map(|i| ((i as f32) - 4.0) * -0.003)
            .collect();
        let hidden_x = array_f32(&x_data, &[1, seq, hidden]);
        let attn_norm = array_f32(&norm_data, &[hidden]);
        let qmx = |w: &MlxArray| {
            let q = quantize(w, Some(32), Some(4), MlxQuantizationMode::Mxfp4, None, None);
            QuantizedWeight {
                weight: q[0].clone(),
                scales: Some(q[1].clone()),
                biases: None,
                group_size: 32,
                bits: 4,
                mode: "mxfp4".to_string(),
                linear_bias: None,
                decode_weight_t: None,
                decode_q4_weight: None,
                decode_q4_scales: None,
                decode_q4_biases: None,
            }
        };
        let mut cfg = v4_test_config(1, 1);
        cfg.model_family = "qwen3_5".to_string();
        cfg.compile_cache_identity = 0x4641_514B_5652_4D53;
        cfg.attn_output_gate = false;
        let dummy = array_f32(&[0.0], &[1]);
        let mut w = v4_layer_weights(dummy, &hidden_x);
        w.router_proj = None;
        w.attn_norm = attn_norm.clone();
        w.q_proj = Some(qmx(&array_f32(&q_data, &[q_out, hidden])));
        w.k_proj = Some(qmx(&array_f32(&k_data, &[kv_out, hidden])));
        w.v_proj = Some(qmx(&array_f32(&v_data, &[kv_out, hidden])));
        let _exact = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let (cq, ck, cv) =
            qwen_compiled_split_verify_fa_attn_norm_qkv(&cfg, &w, &hidden_x, 0, seq as usize)
                .expect("exact S=2 FA attn_norm+QKV compile must engage");
        let normed = rms_norm(&hidden_x, Some(&attn_norm), 1e-6, None);
        let pq = qw(&normed, w.q_proj.as_ref().unwrap());
        let pk = qw(&normed, w.k_proj.as_ref().unwrap());
        let pv = qw(&normed, w.v_proj.as_ref().unwrap());
        eval(&[&cq, &ck, &cv, &pq, &pk, &pv]);
        let max_abs = |a: &MlxArray, b: &MlxArray| {
            a.data_f32()
                .iter()
                .zip(b.data_f32().iter())
                .fold(0.0f32, |m, (l, r)| m.max((l - r).abs()))
        };
        assert!(
            max_abs(&cq, &pq) < 1.0e-5 && max_abs(&ck, &pk) < 1.0e-5 && max_abs(&cv, &pv) < 1.0e-5,
            "compiled FA attn_norm+QKV must match rms+qw, q={} k={} v={}",
            max_abs(&cq, &pq),
            max_abs(&ck, &pk),
            max_abs(&cv, &pv)
        );
        let _off = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
        assert!(
            qwen_compiled_split_verify_fa_attn_norm_qkv(&cfg, &w, &hidden_x, 0, seq as usize)
                .is_none(),
            "FA attn_norm+QKV compile must stay off when exact MTP is scoped off"
        );
    }

    #[test]
    fn qwen_compiled_prefill_down_qmm_matches_qw_4bit_gs32() {
        let hidden_data: Vec<f32> = (0..128 * 32)
            .map(|i| ((i as f32) - 2048.0) * 0.00048828125)
            .collect();
        let down_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 128, 32]);
        let down_w = array_f32(&down_data, &[64, 32]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let compiled =
            qwen_compiled_prefill_down_qmm_for(true, 0x444F_574E_5052_4546, 3, &hidden, &down)
                .expect("down-only prefill compile should engage at leading=128");
        let portable = qw(&hidden, &down);
        eval(&[&compiled, &portable]);
        assert_eq!(compiled.shape(), portable.shape());
        assert_close(compiled.data_f32(), portable.data_f32(), 3.0e-2);
        let decode = array_f32(&hidden_data[..32], &[1, 1, 32]);
        assert!(
            qwen_compiled_prefill_down_qmm_for(true, 0x444F_574E_5052_4546, 3, &decode, &down)
                .is_none(),
            "down-only prefill compile must reject decode seq==1"
        );
        assert!(
            qwen_compiled_prefill_down_qmm(0x444F_574E_5052_4546, 3, &hidden, &down).is_none(),
            "default-off compile flag must keep the imperative down qmm"
        );
    }

    #[test]
    fn cached_prefill_q2_down_requants_4bit_and_qws() {
        let hidden_data: Vec<f32> = (0..32 * 32)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 32, 32]);
        let down_w = array_f32(&down_data, &[64, 32]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let q2 = cached_prefill_q2_down(0x5132_444F_574E, 7, &down)
            .expect("4-bit gs32 down must grow a 2-bit overlay");
        assert_eq!(q2.bits, crate::weights::PREFILL_LA_Q2_BITS);
        assert_eq!(q2.group_size, crate::weights::PREFILL_LA_Q2_GROUP_SIZE);
        let again = cached_prefill_q2_down(0x5132_444F_574E, 7, &down)
            .expect("second lookup must hit the overlay cache");
        assert_eq!(again.bits, q2.bits);
        let out = qw(&hidden, &q2);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 32, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "2-bit down qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_q2_down_for(true, 1024),
            "shipped down q2 gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn cached_prefill_ffn_gs64_requants_4bit_gs32_and_qws() {
        let hidden_data: Vec<f32> = (0..64 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..64 * 64)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 64, 64]);
        let down_w = array_f32(&down_data, &[64, 64]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let gs64 = cached_prefill_ffn_gs64(0x4753_3634, 3, PREFILL_FFN_GS64_DOWN, &down)
            .expect("4-bit gs32 down must grow a gs64 overlay");
        assert_eq!(gs64.bits, 4);
        assert_eq!(gs64.group_size, crate::weights::PREFILL_FFN_GS64_GROUP_SIZE);
        let again = cached_prefill_ffn_gs64(0x4753_3634, 3, PREFILL_FFN_GS64_DOWN, &down)
            .expect("second lookup must hit the overlay cache");
        assert_eq!(again.group_size, gs64.group_size);
        let out = qw(&hidden, &gs64);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 64, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "gs64 down qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_ffn_gs64_for(true, 1024),
            "shipped FFN gs64 gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn cached_prefill_ffn_q3_requants_4bit_and_qws() {
        let hidden_data: Vec<f32> = (0..32 * 32)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 32, 32]);
        let down_w = array_f32(&down_data, &[64, 32]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let q3 = cached_prefill_ffn_q3(0x5133_4646_4e51, 5, PREFILL_FFN_GS64_DOWN, &down)
            .expect("4-bit gs32 down must grow a 3-bit overlay");
        assert_eq!(q3.bits, crate::weights::PREFILL_FFN_Q3_BITS);
        assert_eq!(q3.group_size, crate::weights::PREFILL_FFN_Q3_GROUP_SIZE);
        let again = cached_prefill_ffn_q3(0x5133_4646_4e51, 5, PREFILL_FFN_GS64_DOWN, &down)
            .expect("second lookup must hit the overlay cache");
        assert_eq!(again.bits, q3.bits);
        let out = qw(&hidden, &q3);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 32, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "3-bit down qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_q3_ffn_for(true, 1024),
            "shipped FFN q3 gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn cached_prefill_ffn_contiguous_weight_keeps_bits_and_qws() {
        let hidden_data: Vec<f32> = (0..32 * 32)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 32, 32]);
        let down_w = array_f32(&down_data, &[64, 32]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let contig =
            cached_prefill_ffn_contiguous_weight(0x434f_4e54, 2, PREFILL_FFN_GS64_DOWN, &down)
                .expect("affine down must grow a contiguous overlay");
        assert_eq!(contig.bits, 4);
        assert_eq!(contig.group_size, 32);
        let again =
            cached_prefill_ffn_contiguous_weight(0x434f_4e54, 2, PREFILL_FFN_GS64_DOWN, &down)
                .expect("second lookup must hit the overlay cache");
        assert_eq!(again.bits, contig.bits);
        let out = qw(&hidden, &contig);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 32, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "contiguous-weight qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_contiguous_ffn_weights_for(true, 1024),
            "shipped FFN contiguous-weight gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qwen_prefill_maybe_async_gate_up_submits_pair_at_min_seq() {
        let gate_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01).collect();
        let up_data: Vec<f32> = (0..32).map(|i| (i as f32) * -0.01).collect();
        let gate = array_f32(&gate_data, &[1, 32, 1]);
        let up = array_f32(&up_data, &[1, 32, 1]);
        qwen_prefill_maybe_async_gate_up(&gate, &up, true, 1024);
        eval(&[&gate, &up]);
        assert_eq!(gate.shape(), vec![1, 32, 1]);
        assert_eq!(up.shape(), vec![1, 32, 1]);
        assert!(
            fastpath::should_qwen_prefill_async_gate_up_for(true, 1024),
            "shipped async gate/up gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_async_gate_up(&gate, &up, false, 1024);
        qwen_prefill_maybe_async_gate_up(&gate, &up, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_async_packed_gate_up_submits_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let packed = array_f32(&data, &[1, 32, 1]);
        qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, true, 1024);
        eval(&[&packed]);
        assert_eq!(packed.shape(), vec![1, 32, 1]);
        assert!(
            packed.data_f32().iter().all(|v| v.is_finite()),
            "async packed gate/up must leave a finite materialized tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_async_packed_gate_up_for(true, 1024),
            "shipped async packed-gate/up gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_async_packed_gate_up_for(&packed, false, true, 1024);
        qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, false, 1024);
        qwen_prefill_maybe_async_packed_gate_up_for(&packed, true, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_eval_ffn_hidden_materializes_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let h = array_f32(&data, &[1, 32, 1]);
        qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, true, 1024);
        eval(&[&h]);
        assert_eq!(h.shape(), vec![1, 32, 1]);
        assert!(
            h.data_f32().iter().all(|v| v.is_finite()),
            "eval-ffn-hidden must leave a finite materialized activation"
        );
        assert!(
            fastpath::should_qwen_prefill_eval_ffn_hidden_for(true, 1024),
            "shipped FFN hidden-eval gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_eval_ffn_hidden_for(&h, false, true, 1024);
        qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, false, 1024);
        qwen_prefill_maybe_eval_ffn_hidden_for(&h, true, true, 512);
    }

    #[test]
    fn qwen_prefill_maybe_async_down_submits_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let down = array_f32(&data, &[1, 32, 1]);
        qwen_prefill_maybe_async_down_for(&down, true, true, 1024);
        eval(&[&down]);
        assert_eq!(down.shape(), vec![1, 32, 1]);
        assert!(
            down.data_f32().iter().all(|v| v.is_finite()),
            "async down must leave a finite materialized tensor"
        );
        assert!(
            fastpath::should_qwen_prefill_async_down_for(true, 1024),
            "shipped async-down gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_async_down_for(&down, false, true, 1024);
        qwen_prefill_maybe_async_down_for(&down, true, false, 1024);
        qwen_prefill_maybe_async_down_for(&down, true, true, 512);
    }

    #[test]
    fn cached_prefill_attn_contiguous_weight_keeps_bits_and_qws() {
        let hidden_data: Vec<f32> = (0..32 * 32)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let proj_data: Vec<f32> = (0..64 * 32)
            .map(|i| ((i as f32) - 768.0) * 0.0003)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 32, 32]);
        let proj_w = array_f32(&proj_data, &[64, 32]);
        let dq = quantize(
            &proj_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let src = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let contig = cached_prefill_attn_contiguous_weight(&src);
        assert_eq!(contig.bits, 4);
        assert_eq!(contig.group_size, 32);
        let again = cached_prefill_attn_contiguous_weight(&src);
        assert_eq!(again.bits, contig.bits);
        let out = qw(&hidden, &contig);
        eval(&[&out]);
        assert_eq!(out.shape(), vec![1, 32, 64]);
        assert!(
            out.data_f32().iter().all(|v| v.is_finite()),
            "contiguous attn-weight qmm must produce finite values"
        );
        assert!(
            fastpath::should_qwen_prefill_contiguous_attn_weights_for(true, "qwen3_5", 1024),
            "shipped attn contiguous-weight gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qwen_prefill_ffn_f32_input_promotes_bf16_at_min_seq() {
        let data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.125).collect();
        let bf16 = astype(&array_f32(&data, &[1, 8, 1]), MlxDtype::Bfloat16, None);
        eval(&[&bf16]);
        assert_eq!(bf16.dtype(), MlxDtype::Bfloat16);
        let (promoted, restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, true, 1024);
        eval(&[&promoted]);
        assert_eq!(promoted.dtype(), MlxDtype::Float32);
        assert_eq!(restore, Some(MlxDtype::Bfloat16));
        let back = qwen_prefill_ffn_restore_dtype(&promoted, restore);
        eval(&[&back]);
        assert_eq!(back.dtype(), MlxDtype::Bfloat16);
        let (short, short_restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, true, 512);
        assert!(short_restore.is_none());
        assert_eq!(short.dtype(), MlxDtype::Bfloat16);
        let (other, other_restore) = qwen_prefill_ffn_f32_input_for(&bf16, true, false, 1024);
        assert!(other_restore.is_none());
        assert_eq!(other.dtype(), MlxDtype::Bfloat16);
        assert!(
            fastpath::should_qwen_prefill_ffn_f32_input_for(true, 1024),
            "shipped FFN f32-input gate must accept the p2048 chunk length"
        );
    }

    #[test]
    fn qwen_prefill_maybe_eval_ffn_input_materializes_at_min_seq() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03125).collect();
        let x = array_f32(&data, &[1, 32, 1]);
        qwen_prefill_maybe_eval_ffn_input_for(&x, true, true, 1024);
        eval(&[&x]);
        assert_eq!(x.shape(), vec![1, 32, 1]);
        assert!(
            x.data_f32().iter().all(|v| v.is_finite()),
            "eval-ffn-input must leave a finite materialized activation"
        );
        assert!(
            fastpath::should_qwen_prefill_eval_ffn_input_for(true, 1024),
            "shipped FFN input-eval gate must accept the p2048 chunk length"
        );
        qwen_prefill_maybe_eval_ffn_input_for(&x, false, true, 1024);
        qwen_prefill_maybe_eval_ffn_input_for(&x, true, false, 1024);
        qwen_prefill_maybe_eval_ffn_input_for(&x, true, true, 512);
    }

    #[test]
    fn qwen_prefill_dual_qmm_swiglu_metal_matches_two_qmm_silu_mul_4bit_gs32() {
        let x_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let gate_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0005)
            .collect();
        let up_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 512.0) * -0.0004)
            .collect();
        let x = array_f32(&x_data, &[1, 8, 64]);
        let gate_w = array_f32(&gate_data, &[32, 64]);
        let up_w = array_f32(&up_data, &[32, 64]);
        let gq = quantize(
            &gate_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let uq = quantize(
            &up_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gq.len(), 3);
        assert_eq!(uq.len(), 3);
        let qweight = |q: &[MlxArray]| QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let gate = qweight(&gq);
        let up = qweight(&uq);
        let metal = qwen_prefill_dual_qmm_swiglu_metal(&x, &gate, &up)
            .expect("Qwen 4-bit gs32 prefill dual qmm Metal should engage");
        let p_gate = quantized_matmul(
            &x,
            &gq[0],
            &gq[1],
            Some(&gq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let p_up = quantized_matmul(
            &x,
            &uq[0],
            &uq[1],
            Some(&uq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let portable = silu_mul(&p_gate, &p_up, None);
        eval(&[&metal, &portable]);
        assert_eq!(metal.shape(), portable.shape());
        assert_close(metal.data_f32(), portable.data_f32(), 5.0e-2);
        let decode = array_f32(&x_data[..64], &[1, 1, 64]);
        assert!(
            qwen_prefill_dual_qmm_swiglu_metal(&decode, &gate, &up).is_none(),
            "prefill dual qmm Metal must reject decode seq==1"
        );
    }

    #[test]
    fn qwen_attn_norm_qkv_fuse_matches_rms_then_qw_4bit_gs32() {
        // Shipped Qwen full-attn fuse: rms_norm_quantized_matmul vs rms + qw.
        let x_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let w_data: Vec<f32> = (0..96 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let x = array_f32(&x_data, &[1, 8, 64]);
        let weight = array_f32(&w_data, &[96, 64]);
        let qw_q = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let norm_w = array_f32(&vec![1.0f32; 64], &[64]);
        let fused = rms_norm_quantized_matmul(
            &x,
            &norm_w,
            1e-6,
            &qw_q[0],
            &qw_q[1],
            Some(&qw_q[2]),
            32,
            4,
            None,
        );
        let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
        let portable = quantized_matmul(
            &normed,
            &qw_q[0],
            &qw_q[1],
            Some(&qw_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
        assert!(fastpath::should_attn_norm_qkv_fuse_for(
            true, false, false, "qwen3_5", 128
        ));
        assert!(!fastpath::should_attn_norm_qkv_fuse_for(
            true, false, false, "gemma4", 512
        ));
        assert!(
            fastpath::should_gemma4_attn_norm_qkv_fuse_p128_for(true, "gemma4", 128),
            "AXQ p128 packed QKV must take the attn-norm fuse"
        );
    }

    #[test]
    fn gemma4_attn_norm_qkv_fuse_p128_matches_rms_then_qw_4bit_gs32() {
        // Shipped Gemma 4 p128 fuse: same C++ rms_norm_quantized_matmul as
        // the Qwen path, on the contract seq=128 / AXQ gs=32 layout.
        let hidden = 64;
        let seq = 128;
        let x_data: Vec<f32> = (0..seq * hidden)
            .map(|i| ((i as f32) - 4096.0) * 0.0009765625)
            .collect();
        let w_data: Vec<f32> = (0..96 * hidden)
            .map(|i| ((i as f32) - 2048.0) * 0.0004)
            .collect();
        let x = array_f32(&x_data, &[1, seq as i32, hidden as i32]);
        let weight = array_f32(&w_data, &[96, hidden as i32]);
        let qw_q = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let norm_w = array_f32(&vec![1.0f32; hidden], &[hidden as i32]);
        let fused = rms_norm_quantized_matmul(
            &x,
            &norm_w,
            1e-6,
            &qw_q[0],
            &qw_q[1],
            Some(&qw_q[2]),
            32,
            4,
            None,
        );
        let normed = rms_norm(&x, Some(&norm_w), 1e-6, None);
        let portable = quantized_matmul(
            &normed,
            &qw_q[0],
            &qw_q[1],
            Some(&qw_q[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
        assert!(fastpath::should_call_attn_norm_qkv_fuse(
            fastpath::should_attn_norm_qkv_fuse_for(false, false, true, "gemma4", 128),
            true,
            false,
            false,
        ));
    }

    #[test]
    fn qwen_prefill_contiguous_ffn_qw_matches_view_4bit_gs32() {
        // Shipped path: contiguous([B,S,H]) then qw must match qw on the view.
        let full: Vec<f32> = (0..16 * 64)
            .map(|i| ((i as f32) - 512.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let wide = array_f32(&full, &[1, 16, 64]);
        let view = slice(&wide, &[0, 4, 0], &[1, 12, 64], &[1, 1, 1], None);
        assert_eq!(view.shape(), vec![1, 8, 64]);
        let down_w = array_f32(&down_data, &[32, 64]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let packed = qw(&contiguous(&view, None), &down);
        let portable = qw(&view, &down);
        eval(&[&packed, &portable]);
        assert_eq!(packed.shape(), portable.shape());
        assert_close(packed.data_f32(), portable.data_f32(), 3.0e-2);
        assert!(fastpath::should_qwen_prefill_contiguous_ffn_for(
            true, "qwen3_5", 8, 3
        ));
        assert!(!fastpath::should_qwen_prefill_contiguous_ffn_for(
            true, "qwen3_5", 1, 3
        ));
    }

    #[test]
    fn qwen_prefill_flat_ffn_activation_qw_matches_3d_4bit_gs32() {
        // Drives the shipped flatten/restore used by ffn_swiglu_with_policy
        // so gate/up/down qmm see [B*S,H] but the layer output stays [B,S,H'].
        let hidden_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 8, 64]);
        let down_w = array_f32(&down_data, &[32, 64]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let (flat, orig) = flatten_qwen_prefill_ffn_activation(&hidden);
        assert_eq!(flat.shape(), vec![8, 64]);
        assert_eq!(orig, [1, 8, 64]);
        let flat_out = restore_qwen_prefill_ffn_activation(&qw(&flat, &down), orig);
        let portable = qw(&hidden, &down);
        eval(&[&flat_out, &portable]);
        assert_eq!(flat_out.shape(), portable.shape());
        assert_eq!(flat_out.shape(), vec![1, 8, 32]);
        assert_close(flat_out.data_f32(), portable.data_f32(), 3.0e-2);
        assert!(!fastpath::should_qwen_prefill_flat_ffn_for(
            true, "qwen3_5", 1, 3
        ));
    }

    #[test]
    fn qwen_prefill_flat_down_qmm_matches_3d_qw_4bit_gs32() {
        let hidden_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let down_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0004)
            .collect();
        let hidden = array_f32(&hidden_data, &[1, 8, 64]);
        let down_w = array_f32(&down_data, &[32, 64]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(dq.len(), 3);
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let flat = qwen_prefill_flat_down_qmm(&hidden, &down)
            .expect("Qwen 4-bit gs32 flat down qmm should engage");
        let portable = qw(&hidden, &down);
        eval(&[&flat, &portable]);
        assert_eq!(flat.shape(), portable.shape());
        assert_eq!(flat.shape(), vec![1, 8, 32]);
        assert_close(flat.data_f32(), portable.data_f32(), 3.0e-2);
        let decode = array_f32(&hidden_data[..64], &[1, 1, 64]);
        assert!(
            qwen_prefill_flat_down_qmm(&decode, &down).is_none(),
            "flat down qmm must reject decode seq==1"
        );
    }

    #[test]
    fn qwen_dual_qmm_swiglu_matches_two_qmm_silu_mul_4bit_gs32() {
        // AXQ language FFN is 4-bit gs32. Drive the C++ body directly so a
        // later wash flip of the call-site flag does not skip this test.
        let x_data: Vec<f32> = (0..8 * 64)
            .map(|i| ((i as f32) - 256.0) * 0.0009765625)
            .collect();
        let gate_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 1024.0) * 0.0005)
            .collect();
        let up_data: Vec<f32> = (0..32 * 64)
            .map(|i| ((i as f32) - 512.0) * -0.0004)
            .collect();
        let x = array_f32(&x_data, &[1, 8, 64]);
        let gate_w = array_f32(&gate_data, &[32, 64]);
        let up_w = array_f32(&up_data, &[32, 64]);
        let gq = quantize(
            &gate_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let uq = quantize(
            &up_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(gq.len(), 3);
        assert_eq!(uq.len(), 3);
        let qweight = |q: &[MlxArray]| QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let gate = qweight(&gq);
        let up = qweight(&uq);
        let fused = qwen_dual_qmm_swiglu(&x, &gate, &up)
            .expect("Qwen 4-bit gs32 dual qmm + SwiGLU should engage");
        let p_gate = quantized_matmul(
            &x,
            &gq[0],
            &gq[1],
            Some(&gq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let p_up = quantized_matmul(
            &x,
            &uq[0],
            &uq[1],
            Some(&uq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        let portable = silu_mul(&p_gate, &p_up, None);
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
        let mut bad_gate = gate.clone();
        bad_gate.group_size = 0;
        assert!(
            qwen_dual_qmm_swiglu(&x, &bad_gate, &up).is_none(),
            "dual qmm + SwiGLU must reject group_size<=0"
        );
    }

    #[test]
    fn qwen_swiglu_down_fuse_matches_silu_mul_then_qmm_4bit_gs32() {
        let gate_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
        let up_data: Vec<f32> = (0..512).map(|i| ((i as f32) + 1.0) * 0.0078125).collect();
        let down_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0004).collect();
        let gate = array_f32(&gate_data, &[1, 8, 64]);
        let up = array_f32(&up_data, &[1, 8, 64]);
        let down_w = array_f32(&down_data, &[32, 64]);
        let dq = quantize(
            &down_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(dq.len(), 3);
        let down = QuantizedWeight {
            weight: dq[0].clone(),
            scales: Some(dq[1].clone()),
            biases: Some(dq[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let fused =
            qwen_swiglu_down_fuse(&gate, &up, &down).expect("Qwen SwiGLU+down fuse should engage");
        let hidden = silu_mul(&gate, &up, None);
        let portable = quantized_matmul(
            &hidden,
            &dq[0],
            &dq[1],
            Some(&dq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
    }

    #[test]
    fn qwen_la_out_proj_silu_mul_qmm_matches_rms_silu_then_qw_4bit_gs32() {
        // Shipped LA output fuse: rms_norm(hidden) then silu(z)*normed @ out_proj.
        let hidden_data: Vec<f32> = (0..512).map(|i| ((i as f32) - 256.0) * 0.015625).collect();
        let gate_data: Vec<f32> = (0..512).map(|i| ((i as f32) + 1.0) * 0.0078125).collect();
        let proj_data: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 0.0004).collect();
        let hidden = array_f32(&hidden_data, &[1, 8, 64]);
        let gate = array_f32(&gate_data, &[1, 8, 64]);
        let proj_w = array_f32(&proj_data, &[32, 64]);
        let pq = quantize(
            &proj_w,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let norm_w = array_f32(&vec![1.0f32; 64], &[64]);
        let normed = rms_norm(&hidden, Some(&norm_w), 1e-6, None);
        let fused =
            silu_mul_quantized_matmul(&gate, &normed, &pq[0], &pq[1], Some(&pq[2]), 32, 4, None)
                .expect("LA out_proj silu_mul qmm should engage on 4-bit gs32");
        let gated = silu_mul(&gate, &normed, None);
        let portable = quantized_matmul(
            &gated,
            &pq[0],
            &pq[1],
            Some(&pq[2]),
            true,
            Some(32),
            Some(4),
            None,
        );
        eval(&[&fused, &portable]);
        assert_eq!(fused.shape(), portable.shape());
        assert_close(fused.data_f32(), portable.data_f32(), 3.0e-2);
        assert!(fastpath::should_qwen_la_out_proj_silu_mul_qmm_for(
            true, "qwen3_5", 8
        ));
        assert!(!fastpath::should_qwen_la_out_proj_silu_mul_qmm_for(
            true, "qwen3_5", 1
        ));
    }

    #[test]
    fn dense_ffn_split_gate_up_policy_is_shape_and_family_scoped() {
        assert!(prefer_split_dense_ffn_gate_up(
            "gemma4", false, 127, 127, true
        ));
        assert!(!prefer_split_dense_ffn_gate_up("gemma4", false, 1, 1, true));
        assert!(!prefer_split_dense_ffn_gate_up("gemma4", false, 4, 4, true));
        assert!(!prefer_split_dense_ffn_gate_up(
            "gemma4", false, 126, 126, true
        ));
        assert!(prefer_split_dense_ffn_gate_up(
            "qwen3_next",
            true,
            1,
            1,
            true
        ));
        assert!(!prefer_split_dense_ffn_gate_up(
            "qwen3_next",
            true,
            128,
            128,
            true
        ));
        assert!(!prefer_split_dense_ffn_gate_up(
            "gemma4", false, 128, 128, false
        ));
        assert!(
            prefer_split_dense_ffn_gate_up("gemma4", false, 128, 128, true),
            "default-off packed compile leaves p128 on split gate/up"
        );
    }

    #[test]
    fn use_packed_dense_ffn_prefill_skips_last_only_packed() {
        assert!(
            super::use_packed_dense_ffn_prefill(false, true, false),
            "packed prefill stays on when last-only skip is off"
        );
        assert!(
            !super::use_packed_dense_ffn_prefill(false, true, true),
            "last-only 1-token FFN must skip unused packed prefill qmm"
        );
        assert!(!super::use_packed_dense_ffn_prefill(true, true, false));
        assert!(!super::use_packed_dense_ffn_prefill(false, false, false));
        assert!(
            crate::fastpath::should_gemma4_prefill_skip_unused_last_ffn_packed_for(
                true, "gemma4", true, 128
            ),
            "shipped skip-unused-last-ffn-packed must accept contract p128 last layer"
        );
    }

    #[test]
    fn gemma4_split_qkv_policy_is_shape_and_family_scoped() {
        assert!(prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            127,
            true,
        ));
        assert!(prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            511,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            1,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            4,
            true,
        ));
        // Chunk-512 pure: packed is faster than split (mbp-m5 A/B ~1.03×).
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            512,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            2_048,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "qwen3_next",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            128,
            true,
        ));
        assert!(prefer_split_qkv_projection(
            "qwen3_next",
            false,
            ProjectionBatchPolicy::Shared,
            2,
            128,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            true,
            ProjectionBatchPolicy::Shared,
            1,
            128,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::RowExact,
            1,
            128,
            true,
        ));
        assert!(!prefer_split_qkv_projection(
            "gemma4",
            false,
            ProjectionBatchPolicy::Shared,
            1,
            128,
            false,
        ));
    }

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
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let up = QuantizedWeight {
            weight: up_q[0].clone(),
            scales: Some(up_q[1].clone()),
            biases: Some(up_q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };

        let metal = qwen_dense_ffn_gate_up_swiglu_metal_impl(&x, &gate, &up)
            .expect("4-bit affine gate/up SwiGLU matvec should be eligible");
        assert_eq!(metal.shape(), vec![1, 1, 16]);
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
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };
        let batched = mlx_sys::zeros(&[2, 1, 32], MlxDtype::Float32, None);
        let prefill = mlx_sys::zeros(&[1, 2, 32], MlxDtype::Float32, None);

        assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&batched, &weight, &weight).is_none());
        assert!(qwen_dense_ffn_gate_up_swiglu_metal_impl(&prefill, &weight, &weight).is_none());
    }

    #[test]
    fn qwen_dense_ffn_gate_up_matvec_metal_skips_only_regressed_27b_quantization() {
        assert!(qwen_dense_ffn_gate_up_matvec_metal_regresses(
            "qwen3_5", 64, 5120, 17_408, 0, 4, 64
        ));
        assert!(
            !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 32, 4096, 12_288, 0, 4, 64),
            "the measured Qwen3.5-9B win must remain eligible"
        );
        assert!(
            !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 256, 4, 64),
            "the dense-model exception must not mask MoE configurations"
        );
        assert!(
            !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 40, 2048, 0, 256, 4, 64),
            "the Qwen3.6-35B-A3B geometry must remain unaffected"
        );
        assert!(
            !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 0, 6, 64),
            "the unmeasured 6-bit configuration must not inherit the 4-bit exception"
        );
        assert!(
            !qwen_dense_ffn_gate_up_matvec_metal_regresses("qwen3_5", 64, 5120, 17_408, 0, 4, 32),
            "the exception must remain specific to the measured group size"
        );
    }

    #[test]
    fn qwen_dense_ffn_down_matvec_metal_matches_quantized_matmul() {
        // Intermediate → hidden decode matvec (tiled TG path for InputDim > 4096).
        // Use intermediate=64, out=16, group_size=32 to exercise multi-tile packing.
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
        let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
        let x = array_f32(&x_data, &[1, 1, 64]);
        let weight = array_f32(&weight_data, &[16, 64]);
        let q = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        assert_eq!(q.len(), 3);
        let down = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };

        let metal = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
            .expect("4-bit affine down matvec should be eligible");
        let reference =
            quantized_matmul(&x, &q[0], &q[1], Some(&q[2]), true, Some(32), Some(4), None);
        mlx_sys::transforms::try_eval(&[&metal, &reference])
            .expect("Qwen dense FFN down matvec Metal kernel must compile and evaluate");

        assert_eq!(metal.shape(), vec![1, 1, 16]);
        assert_close(metal.data_f32(), reference.data_f32(), 1.0e-4);
    }

    #[test]
    fn qwen_dense_ffn_down_residual_metal_matches_add() {
        let x_data: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.015625).collect();
        let weight_data: Vec<f32> = (0..1024).map(|i| ((i as f32) - 400.0) * 0.00125).collect();
        let residual_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 2.0).collect();
        let x = array_f32(&x_data, &[1, 1, 64]);
        let residual = array_f32(&residual_data, &[1, 1, 16]);
        let weight = array_f32(&weight_data, &[16, 64]);
        let q = quantize(
            &weight,
            Some(32),
            Some(4),
            MlxQuantizationMode::Affine,
            None,
            None,
        );
        let down = QuantizedWeight {
            weight: q[0].clone(),
            scales: Some(q[1].clone()),
            biases: Some(q[2].clone()),
            group_size: 32,
            bits: 4,
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
        };

        let fused = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, Some(&residual))
            .expect("residual down matvec should be eligible");
        let split = qwen_dense_ffn_down_matvec_metal_impl(&x, &down, None)
            .expect("plain down matvec should be eligible");
        let reference = add(&residual, &split, None);
        mlx_sys::transforms::try_eval(&[&fused, &reference])
            .expect("residual down Metal kernel must compile and evaluate");
        assert_eq!(fused.shape(), vec![1, 1, 16]);
        assert_close(fused.data_f32(), reference.data_f32(), 1.0e-4);
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
            mode: "affine".to_string(),
            linear_bias: None,
            decode_weight_t: None,
            decode_q4_weight: None,
            decode_q4_scales: None,
            decode_q4_biases: None,
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

    fn v4_test_config(experts: usize, top_k: usize) -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "deepseek_v4".to_string(),
            layer_count: 1,
            hidden_size: 4,
            intermediate_size: 8,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            vocab_size: 16,
            rope_theta: 10000.0,
            rope_dims: 8,
            attn_output_gate: false,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: experts,
            moe_experts_per_token: top_k,
            moe_expert_intermediate_size: 8,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            protected_prefix_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: true,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            deepseek_v4: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 2.5,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
            generation_kind: ax_engine_core::GenerationKind::Autoregressive,
            kv_cache_quant: vec![None; 1],
        }
    }

    fn v4_layer_weights(router: MlxArray, x: &MlxArray) -> LayerWeights {
        LayerWeights {
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
            deepseek_v4: None,
            ffn_norm: x.clone(),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: Some(QuantizedWeight::new(router, None, None)),
            router_correction_bias: None,
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: None,
            shared_up_proj: None,
            shared_down_proj: None,
            gate_up_exps_packed: None,
            gate_exps: None,
            up_exps: None,
            down_exps: None,
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
            expert_stream: None,
        }
    }

    /// Manual `sqrt(softplus(x))` in plain f64 arithmetic for comparisons.
    fn manual_sqrt_softplus(x: f32) -> f64 {
        let x = x as f64;
        let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
        softplus.sqrt()
    }

    #[test]
    fn sqrt_softplus_scores_matches_manual() {
        let logits = array_f32(&[-20.0, -1.5, 0.0, 0.5, 3.0, 20.0], &[1, 1, 6]);
        let scores = sqrt_softplus_scores(&logits);
        eval(&[&scores]);
        assert_eq!(scores.dtype(), MlxDtype::Float32);
        let actual = scores.data_f32().to_vec();
        let expected: Vec<f32> = [-20.0, -1.5, 0.0, 0.5, 3.0, 20.0]
            .iter()
            .map(|x| manual_sqrt_softplus(*x) as f32)
            .collect();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn moe_router_deepseek_v4_learned_path_matches_manual() {
        // Identity gate: logits == x. top-2 of 4 experts, norm_topk_prob on,
        // routed_scaling_factor 2.5 (from v4_test_config).
        let cfg = v4_test_config(4, 2);
        let router = array_f32(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        );
        let x = array_f32(&[0.5, 3.0, -1.0, 1.5], &[1, 1, 4]);
        let mut w = v4_layer_weights(router, &x);
        // Correction bias flips the selection: expert 0 beats expert 3.
        w.router_correction_bias = Some(array_f32(&[3.0, 0.0, 0.0, 0.0], &[4]));

        let (indices, weights) = moe_router_deepseek_v4(&cfg, &w, &x, None);
        let indices = astype(&indices, MlxDtype::Uint32, None);
        eval(&[&indices, &weights]);
        assert_eq!(indices.shape(), vec![1, 1, 2]);
        assert_eq!(weights.shape(), vec![1, 1, 2]);

        // Selection scores: probs + bias → experts 1 (prob √softplus(3) ≈ 1.74)
        // and 0 (≈0.93 + 3.0) win. Weights come from the UNBIASED probs,
        // renormalised then scaled by 2.5.
        let p = [0.5_f32, 3.0, -1.0, 1.5].map(manual_sqrt_softplus);
        let sel: Vec<f64> = [p[0] + 3.0, p[1], p[2], p[3]].to_vec();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|a, b| sel[*b].total_cmp(&sel[*a]));
        let (e0, e1) = (order[0], order[1]);
        let w0 = p[e0] / (p[e0] + p[e1]) * 2.5;
        let w1 = p[e1] / (p[e0] + p[e1]) * 2.5;

        let idx = indices.data_u32().to_vec();
        let got = weights.data_f32().to_vec();
        let mut got_pairs: Vec<(u32, f32)> = idx.into_iter().zip(got).collect();
        got_pairs.sort_by_key(|(i, _)| *i);
        let mut expect_pairs: Vec<(u32, f32)> = [(e0 as u32, w0 as f32), (e1 as u32, w1 as f32)]
            .into_iter()
            .collect();
        expect_pairs.sort_by_key(|(i, _)| *i);
        assert_eq!(got_pairs.len(), expect_pairs.len());
        for ((gi, gw), (ei, ew)) in got_pairs.iter().zip(expect_pairs.iter()) {
            assert_eq!(gi, ei, "selected expert mismatch");
            assert!((gw - ew).abs() < 1e-4, "weight {gw} vs expected {ew}");
        }
    }

    #[test]
    fn moe_router_deepseek_v4_hash_path_uses_tid2eid_indices() {
        // Hash routing: indices come from the tid2eid table at token_ids;
        // weights still come from the unbiased sqrtsoftplus probs.
        let cfg = v4_test_config(4, 2);
        let router = array_f32(
            &[
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            &[4, 4],
        );
        let x = array_f32(&[0.5, 3.0, -1.0, 1.5], &[1, 1, 4]);
        let mut w = v4_layer_weights(router, &x);
        // [vocab=16, topk=2] table; token 7 routes to experts (2, 0).
        let mut table = vec![0u32; 16 * 2];
        table[7 * 2] = 2;
        table[7 * 2 + 1] = 0;
        let tid2eid = MlxArray::from_raw_data(
            table.as_ptr() as *const u8,
            std::mem::size_of_val(table.as_slice()),
            &[16, 2],
            MlxDtype::Uint32,
        );
        w.deepseek_v4 = Some(crate::weights::DeepseekV4LayerWeights {
            wq_a: QuantizedWeight::new(x.clone(), None, None),
            q_a_norm: x.clone(),
            wq_b: QuantizedWeight::new(x.clone(), None, None),
            wkv: QuantizedWeight::new(x.clone(), None, None),
            kv_norm: x.clone(),
            wo_a: QuantizedWeight::new(x.clone(), None, None),
            wo_b: QuantizedWeight::new(x.clone(), None, None),
            attn_sink: None,
            hc_attn_fn: x.clone(),
            hc_attn_base: x.clone(),
            hc_attn_scale: x.clone(),
            hc_ffn_fn: x.clone(),
            hc_ffn_base: x.clone(),
            hc_ffn_scale: x.clone(),
            compressor: None,
            indexer: None,
            tid2eid: Some(tid2eid),
        });
        // Deliberately WRONG correction bias: hash routing must ignore it.
        w.router_correction_bias = Some(array_f32(&[0.0, 0.0, 100.0, 0.0], &[4]));

        let token_ids = MlxArray::from_raw_data(
            [7u32].as_ptr() as *const u8,
            std::mem::size_of::<u32>(),
            &[1, 1],
            MlxDtype::Uint32,
        );
        let (indices, weights) = moe_router_deepseek_v4(&cfg, &w, &x, Some(&token_ids));
        let indices = astype(&indices, MlxDtype::Uint32, None);
        eval(&[&indices, &weights]);
        assert_eq!(indices.shape(), vec![1, 1, 2]);

        let p = [0.5_f32, 3.0, -1.0, 1.5].map(manual_sqrt_softplus);
        let w2 = p[2] / (p[2] + p[0]) * 2.5;
        let w0 = p[0] / (p[2] + p[0]) * 2.5;

        let idx = indices.data_u32().to_vec();
        let got = weights.data_f32().to_vec();
        let mut got_pairs: Vec<(u32, f32)> = idx.into_iter().zip(got).collect();
        got_pairs.sort_by_key(|(i, _)| *i);
        let expect_pairs: Vec<(u32, f32)> = vec![(0, w0 as f32), (2, w2 as f32)];
        for ((gi, gw), (ei, ew)) in got_pairs.iter().zip(expect_pairs.iter()) {
            assert_eq!(gi, ei, "hash-routed expert mismatch");
            assert!((gw - ew).abs() < 1e-4, "weight {gw} vs expected {ew}");
        }
    }
}
