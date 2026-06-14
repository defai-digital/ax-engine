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

static GELU_MUL_KERNEL: OnceLock<MlxMetalKernel> = OnceLock::new();

const GELU_MUL_KERNEL_SOURCE: &str = r#"
    uint idx = thread_position_in_grid.x;
    if (idx >= ElementCount) {
        return;
    }

    T gate_v = gate[idx];
    T x_v = x[idx];
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
    if let Some(out) = gelu_approx_mul_metal(gate, x, fastpath::geglu_mul_metal_enabled()) {
        return out;
    }
    gelu_approx_mul(gate, x, None)
}

pub(crate) fn per_layer_input_gate(gate: &MlxArray, per_layer_input: &MlxArray) -> MlxArray {
    if let Some(out) = gelu_approx_mul_metal(
        gate,
        per_layer_input,
        fastpath::gemma4_per_layer_gelu_mul_metal_enabled(),
    ) {
        return out;
    }
    // mlx-lm keeps this Gemma4DecoderLayer per-layer input gate imperative;
    // the direct shim preserves the same math with one stable FFI call.
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
            "ax_gemma_gelu_mul_v1",
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
    if let Some(gated) = gelu_approx_mul_metal(
        gate,
        per_layer_input,
        fastpath::gemma4_per_layer_gelu_mul_metal_enabled(),
    ) {
        return qw(&gated, proj_w);
    }
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

pub(crate) fn ffn_swiglu(
    cfg: &ModelConfig,
    w: &LayerWeights,
    x: &MlxArray,
    post_norm: Option<&MlxArray>,
) -> MlxArray {
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
                let down = w
                    .down_proj
                    .as_ref()
                    .expect("dense FFN layer must have down_proj");
                if let Some(norm_w) = post_norm {
                    if !profile_decode
                        && !profile_prefill
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
                    let out = qw(&ffn_hidden, down);
                    forward_profile_eval_elapsed(
                        profile_decode,
                        profile_prefill,
                        DecodeProfileStage::PostAttnFfnDown,
                        down_started,
                        &[&out],
                    );
                    return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
                }
                let out = qw(&ffn_hidden, down);
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
                    let out = qw(&ffn_hidden, down);
                    forward_profile_eval_elapsed(
                        profile_decode,
                        profile_prefill,
                        DecodeProfileStage::PostAttnFfnDown,
                        down_started,
                        &[&out],
                    );
                    return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
                }
                let out = qw(&ffn_hidden, down);
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
    let down = w
        .down_proj
        .as_ref()
        .expect("dense FFN layer must have down_proj");
    if let Some(norm_w) = post_norm {
        if !profile_decode
            && !profile_prefill
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
        let out = qw(&ffn_hidden, down);
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::PostAttnFfnDown,
            down_started,
            &[&out],
        );
        return rms_norm(&out, Some(norm_w), cfg.rms_norm_eps, None);
    }
    let out = qw(&ffn_hidden, down);
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
    // mlx-lm uses mx.softmax(..., precise=True) for all MoE routers; with bf16 logits and
    // many experts (e.g. 256) the tiny round-off can flip top-k rankings and corrupt output.
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
        let half = cfg.moe_expert_intermediate_size as i32;
        if !cfg.uses_geglu
            && seq == 1
            && fastpath::moe_swiglu_packed_metal_enabled()
            && let Some(fused) = packed_swiglu_metal_impl(&out, half)
        {
            let activation_started = Instant::now();
            forward_profile_eval_elapsed(
                profile_decode,
                profile_prefill,
                DecodeProfileStage::MoeExpertActivation,
                activation_started,
                &[&fused],
            );
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
        let activation_started = Instant::now();
        let h = dense_ffn_activation(cfg, &gate_out, &up_out);
        forward_profile_eval_elapsed(
            profile_decode,
            profile_prefill,
            DecodeProfileStage::MoeExpertActivation,
            activation_started,
            &[&h],
        );
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

    // Fresh timer for the weighted-sum stage so it does not include the down
    // projection time (which is already recorded under MoeExpertDown).
    let weighted_sum_started = Instant::now();

    // Phase 1A: when shared_expert_out is provided, try the fused weighted-sum
    // kernel that adds the shared expert inside the same dispatch. Decode-only
    // (seq==1): at prefill the weighted-sum is bandwidth-bound on a large tensor,
    // where the fused kernel's extra input read costs more than the dispatch it
    // saves. Falls back to the separate `add` in the branches below at prefill.
    if seq == 1
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
