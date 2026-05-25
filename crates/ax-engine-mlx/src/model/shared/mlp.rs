use mlx_sys::{
    KernelOutputSpec, KernelTemplateArg, MlxArray, MlxClosure, MlxDtype, MlxMetalKernel,
    MlxVectorArray, add, argpartition_axis, argsort_axis, astype, divide, expand_dims,
    expand_dims_axes, gelu_approx_mul, multiply, reshape, rms_norm, silu_mul, slice_last_dim,
    softmax, sum_axis, take, take_along_axis, topk_axis,
};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::fastpath;
use crate::weights::{LayerWeights, QuantizedWeight};

use super::super::config::{GlmRouterConfig, ModelConfig};
use super::super::profile::{
    DecodeProfileStage, decode_profile_enabled, forward_profile_eval_elapsed,
    prefill_profile_enabled,
};
use super::utils::{
    mlx_slice_last_dim, qkv_slices, qw, qw_gather, scalar_like, scale_hidden, shape_element_count,
    squeeze_switch_singleton,
};

pub(crate) fn qkv_project(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    head_dim: usize,
) -> (MlxArray, MlxArray, MlxArray, Option<MlxArray>) {
    let slices = qkv_slices(cfg, head_dim);
    if let Some(packed) = &w.qkv_packed {
        let out = qw(x, packed);
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
                &[1, seq, cfg.n_heads as i32, 2 * head_dim as i32],
                None,
            );
            let q = reshape(
                &slice_last_dim(&qg_heads, 0, head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            let gate = reshape(
                &slice_last_dim(&qg_heads, head_dim as i32, 2 * head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
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
        let q_full = qw(x, w.q_proj.as_ref().unwrap());
        let (q, gate) = if slices.gate.is_some() {
            // attn_output_gate=true: q_proj output is [B, L, n_heads, 2*head_dim] interleaved.
            // Split by reshaping to [B, L, n_heads, 2*head_dim] and slicing last dim,
            // matching mlx-lm's `mx.split(q_proj_out.reshape(B, L, n_heads, -1), 2, axis=-1)`.
            let seq = q_full.shape()[1];
            let q_heads = reshape(
                &q_full,
                &[1, seq, cfg.n_heads as i32, 2 * head_dim as i32],
                None,
            );
            let q = reshape(
                &slice_last_dim(&q_heads, 0, head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            let gate = reshape(
                &slice_last_dim(&q_heads, head_dim as i32, 2 * head_dim as i32, None),
                &[1, seq, (cfg.n_heads * head_dim) as i32],
                None,
            );
            (q, Some(gate))
        } else {
            // q_proj output is exactly [B, L, n_heads * head_dim] — no slice needed.
            (q_full, None)
        };
        let k = qw(x, w.k_proj.as_ref().unwrap());
        let v = w
            .v_proj
            .as_ref()
            .map(|v_proj| qw(x, v_proj))
            .unwrap_or_else(|| k.clone());
        (q, k, v, gate)
    }
}

pub(crate) fn attention_output_projection(
    attn_flat: &MlxArray,
    attn_gate: Option<&MlxArray>,
    o_proj: &QuantizedWeight,
) -> MlxArray {
    let gated = if let Some(gate) = attn_gate {
        multiply(attn_flat, &mlx_sys::ops::sigmoid(gate, None), None)
    } else {
        attn_flat.clone()
    };
    qw(&gated, o_proj)
}

/// Gemma-family GeGLU activation.
///
/// This preserves mlx-lm's `nn.gelu_approx(gate) * x` math while using AX's
/// direct MLX shim to collapse the scalar-heavy activation chain behind one
/// stable FFI call. The older compiled-closure experiment was removed from the
/// production surface because MLX compiled closures are thread/stream-registry
/// sensitive in the Rust server worker model.
pub(crate) fn geglu(gate: &MlxArray, x: &MlxArray) -> MlxArray {
    gelu_approx_mul(gate, x, None)
}

pub(crate) fn per_layer_input_gate(gate: &MlxArray, per_layer_input: &MlxArray) -> MlxArray {
    // mlx-lm keeps this Gemma4DecoderLayer per-layer input gate imperative;
    // the direct shim preserves the same math with one stable FFI call.
    gelu_approx_mul(gate, per_layer_input, None)
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
    let outputs = {
        let mut guard = cache.lock().expect("swiglu compile cache mutex poisoned");
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
static GEMMA4_MOE_WEIGHTED_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();
static GEMMA4_MOE_WEIGHTED_SCALED_SUM_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const PACKED_GEGLU_KERNEL_SOURCE: &str = r#"
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
    float gate_sq = gate_v * gate_v;
    float inner = 0.7978845608028654f * (gate_v + 0.044715f * gate_v * gate_sq);
    float activated = 0.5f * gate_v * (1.0f + tanh(inner));
    out[idx] = static_cast<T>(activated * up_v);
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
        "ax_gemma_packed_geglu_v1",
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

pub(crate) fn ffn_swiglu(cfg: &ModelConfig, w: &LayerWeights, x: &MlxArray) -> MlxArray {
    let seq = x.shape().get(1).copied().unwrap_or(1);
    let profile_decode = seq == 1 && decode_profile_enabled();
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
    let gate_up_started = Instant::now();
    let packed_gate_up: Option<MlxArray>;
    let mut gate_up_profile_recorded = false;
    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_packed {
        let out = qw(x, packed);
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
                let out = qw(
                    &ffn_hidden,
                    w.down_proj
                        .as_ref()
                        .expect("dense FFN layer must have down_proj"),
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
                let out = qw(
                    &ffn_hidden,
                    w.down_proj
                        .as_ref()
                        .expect("dense FFN layer must have down_proj"),
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
        }
        packed_gate_up = Some(out.clone());
        let gate = mlx_slice_last_dim(&out, 0, half);
        let up = mlx_slice_last_dim(&out, half, half * 2);
        (gate, up)
    } else {
        packed_gate_up = None;
        let gate = qw(x, w.gate_proj.as_ref().unwrap());
        let up = qw(x, w.up_proj.as_ref().unwrap());
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
    let out = qw(
        &ffn_hidden,
        w.down_proj
            .as_ref()
            .expect("dense FFN layer must have down_proj"),
    );
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

/// Qwen3 MoE router: proj → softmax → pick top-k by weight value (no rms_norm).
pub(crate) fn moe_router_qwen3(
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
    let weights_all = softmax(&logits, last_axis, None);
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

/// Zero out experts belonging to the worst (n_group - topk_group) groups.
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
    moe_experts_forward_impl(cfg, w, x, top_k_indices, top_k_weights, None)
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
    )
}

fn moe_experts_forward_impl(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    top_k_indices: &MlxArray,
    top_k_weights: &MlxArray,
    top_k_expert_scale: Option<&MlxArray>,
) -> MlxArray {
    // Match MLX SwitchGLU: [batch, seq, hidden] → [batch, seq, 1, 1, hidden].
    // The extra singleton before top_k is required by gather_mm/gather_qmm broadcasting.
    let x_exp = expand_dims_axes(x, &[-2, -3], None);
    let gather_inputs = switch_gather_inputs(&x_exp, top_k_indices);
    let down_exps = w.down_exps.as_ref().expect("MoE layer must have down_exps");

    let (gate_out, up_out) = if let Some(packed) = &w.gate_up_exps_packed {
        let out = qw_gather(
            &gather_inputs.x,
            packed,
            &gather_inputs.indices,
            gather_inputs.sorted_indices,
        );
        let half = cfg.moe_expert_intermediate_size as i32;
        (
            mlx_slice_last_dim(&out, 0, half),
            mlx_slice_last_dim(&out, half, half * 2),
        )
    } else {
        let gate_exps = w.gate_exps.as_ref().expect("MoE layer must have gate_exps");
        let up_exps = w.up_exps.as_ref().expect("MoE layer must have up_exps");
        (
            qw_gather(
                &gather_inputs.x,
                gate_exps,
                &gather_inputs.indices,
                gather_inputs.sorted_indices,
            ),
            qw_gather(
                &gather_inputs.x,
                up_exps,
                &gather_inputs.indices,
                gather_inputs.sorted_indices,
            ),
        )
    };

    // Gemma4 experts use direct GEGLU with fast-approx GELU (matches
    // mlx_lm's `nn.gelu_approx`). Qwen3 uses the SwiGLU helper, which can
    // still use the compiled-closure cache.
    let hidden = dense_ffn_activation(cfg, &gate_out, &up_out);

    // Down projection: [1, seq, top_k, hidden]
    let down_out = squeeze_switch_singleton(&qw_gather(
        &hidden,
        down_exps,
        &gather_inputs.indices,
        gather_inputs.sorted_indices,
    ));
    let down_out = gather_inputs.unsort(down_out);

    // Weighted sum over top_k dimension → [1, seq, hidden]. Gemma4 decode hits
    // this in every layer; fuse multiply + reduction + cast to keep the direct
    // pipeline graph smaller. Other MoE families keep the generic MLX path.
    if cfg.gemma4_moe_router {
        if let Some(expert_scale) = top_k_expert_scale {
            if let Some(out) = gemma4_moe_weighted_scaled_sum_metal(
                &down_out,
                top_k_weights,
                top_k_indices,
                expert_scale,
                x.dtype(),
            ) {
                return out;
            }
        } else if let Some(out) = gemma4_moe_weighted_sum_metal(&down_out, top_k_weights, x.dtype())
        {
            return out;
        }
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
    astype(&out, x.dtype(), None)
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
    use mlx_sys::{concatenate, eval, slice_last_dim};

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
        assert_close(metal.data_f32(), direct.data_f32(), 2.0e-2);
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
}
