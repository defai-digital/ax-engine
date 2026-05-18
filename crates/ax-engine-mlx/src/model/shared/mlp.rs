use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, add, argpartition_axis, argsort_axis, astype,
    divide, expand_dims, expand_dims_axes, gelu_approx, multiply, reshape, rms_norm,
    slice_last_dim, softmax, sum_axis, take, take_along_axis,
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

/// SPIKE ARTIFACT (W2.a, 2026-05-14) — currently NOT WIRED into the production
/// hot path.
///
/// `geglu(gate, x) = gelu_approx(gate) * x` collapsed into one compiled-closure
/// dispatch via `MlxClosure::compile(shapeless=true)`. Mirrors mlx_lm's
/// `@partial(mx.compile, shapeless=True) def geglu(gate, x)` in
/// `mlx_lm.models.gemma4_text`. Uncompiled this expands to 9 ops from
/// `gelu_approx` plus 1 multiply; compiled, the fused graph dispatches as a
/// single MLX op.
///
/// **Why not wired**: the W2.a spike confirmed bit-exact correctness
/// (`geglu_compiled_matches_imperative`) but production wiring crashed the
/// server with `"There is no Stream(gpu, N) in current thread"` from
/// `mlx_closure_apply` running on tokio worker threads. The follow-up source
/// read found the relevant MLX 0.31 / mlx-c 0.6 contract: default streams,
/// compiled-function cache entries, and Metal command encoders are all
/// thread-local; `mlx_set_default_stream` does not register an existing stream
/// index's encoder on another thread. The helper is kept as a record of the
/// math + the test as regression protection if the stream story is ever solved.
/// See
/// `benchmarks/results/mlx-inference/2026-05-14-w2-stream-contract-source-read/stream-contract.md`.
/// GeGLU compiled helper kept (and unit-tested) as a record of the math;
/// the production gate that wires this into `dense_ffn_activation` is
/// `fastpath::prefill_ffn_compile_enabled()` reading
/// `AX_MLX_PREFILL_FFN_COMPILE`. Pre-W1 this helper was dead code because
/// the original `OnceLock<Option<MlxClosure>>` cache was process-wide:
/// thread A compiled the closure, thread B (a different tokio worker)
/// inherited it and aborted at `mlx_closure_apply` with
/// `"There is no Stream(gpu, N) in current thread"` — MLX 0.31 / mlx-c
/// 0.6 default streams + compiled-function caches + Metal command
/// encoders are thread-local.
///
/// W1 fix: per-thread cache keyed by `ThreadId` in a process-static
/// `Mutex<HashMap<ThreadId, MlxClosure>>`. Each thread compiles its own
/// closure on first use and reuses it on subsequent calls from the same
/// thread. The cache is intentionally NOT a `thread_local!` because
/// thread_local destructors run after the thread's MLX state has
/// torn down, leaving a stale `MlxClosure` that SIGSEGVs at drop. The
/// process-static map holds entries until process exit, mirroring the
/// embedding compile cache in `MlxRunner` (`runner.rs::EmbedCompileKey =
/// (ThreadId, ...)`). Combined with `MlxClosure::try_apply` (which
/// rejects a cross-thread apply before reaching `mlx_closure_apply`),
/// the helper fails closed: on any contract violation we fall back to
/// the imperative `gelu_approx + multiply` path without aborting.
pub(crate) fn geglu(gate: &MlxArray, x: &MlxArray) -> MlxArray {
    use std::collections::HashMap;
    use std::collections::hash_map::Entry;
    use std::thread::ThreadId;

    // The GeGLU compiled closure ABORTS the process inside `mlx_closure_apply`
    // ("There is no Stream(gpu, 0) in current thread" at
    // mlx-c-0.6.0/mlx/c/transforms.cpp:73) under two MLX 0.31 / mlx-c 0.6
    // edge cases that `try_apply`'s cross-thread guard cannot catch:
    //
    //   1. Rank-4 inputs from MoE expert `qw_gather` outputs. Verified on
    //      Gemma 4 26B A4B (mixed dense+MoE GeGLU). The SwiGLU compile wrap
    //      survives the same 4D shape on Qwen 3.6 35B-A3B / Coder Next /
    //      GLM 4.7 Flash, so the issue is specific to the
    //      `gelu_approx + multiply` op tree.
    //
    //   2. Very wide intermediate dimensions. Empirically Gemma 4 31B 4-bit
    //      (`intermediate_size = 21504`, hidden 5376) aborts at the first
    //      apply, while E4B (`intermediate_size = 10240`) is stable. A
    //      threshold of 16K on the last-axis size keeps E2B (6144) and E4B
    //      (10240) inside the compile wrap and routes 31B's enormous FFN
    //      hidden to the imperative path.
    //
    // Both fallback paths produce identical numerics (the unit test asserts
    // bit-exact equality vs `gelu_approx + multiply`) and match the W1
    // pre-K behaviour on these call sites.
    const COMPILE_WRAP_LAST_DIM_CEILING: usize = 16_384;
    let last_dim = *gate.shape().last().unwrap_or(&0) as usize;
    if gate.ndim() > 3 || last_dim > COMPILE_WRAP_LAST_DIM_CEILING {
        return multiply(&gelu_approx(gate, None), x, None);
    }

    static GEGLU_COMPILE_CACHE: OnceLock<Mutex<HashMap<ThreadId, MlxClosure>>> = OnceLock::new();

    let cache = GEGLU_COMPILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();
    let outputs = {
        let mut guard = cache.lock().expect("geglu compile cache mutex poisoned");
        if let Entry::Vacant(slot) = guard.entry(tid)
            && let Ok(compiled) = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
                let gate = inputs.get(0);
                let x = inputs.get(1);
                let activated = gelu_approx(&gate, None);
                vec![multiply(&activated, &x, None)]
            })
            .compile(true)
        {
            slot.insert(compiled);
        }
        guard
            .get(&tid)
            .and_then(|cls| cls.try_apply(&[gate, x]).ok())
    };

    if let Some(mut outputs) = outputs
        && let Some(out) = outputs.pop()
    {
        return out;
    }
    multiply(&gelu_approx(gate, None), x, None)
}

pub(crate) fn per_layer_input_gate(gate: &MlxArray, per_layer_input: &MlxArray) -> MlxArray {
    use std::collections::HashMap;
    use std::collections::hash_map::Entry;
    use std::thread::ThreadId;

    if !fastpath::per_layer_gate_compile_enabled() {
        return multiply(&gelu_approx(gate, None), per_layer_input, None);
    }

    const COMPILE_WRAP_LAST_DIM_CEILING: usize = 16_384;
    let last_dim = *gate.shape().last().unwrap_or(&0) as usize;
    if gate.ndim() > 3 || last_dim > COMPILE_WRAP_LAST_DIM_CEILING {
        return multiply(&gelu_approx(gate, None), per_layer_input, None);
    }

    type PerLayerGateCompileKey = (ThreadId, usize);
    static PER_LAYER_GATE_COMPILE_CACHE: OnceLock<
        Mutex<HashMap<PerLayerGateCompileKey, MlxClosure>>,
    > = OnceLock::new();

    let cache = PER_LAYER_GATE_COMPILE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (std::thread::current().id(), last_dim);
    let outputs = {
        let mut guard = cache
            .lock()
            .expect("per-layer gate compile cache mutex poisoned");
        if let Entry::Vacant(slot) = guard.entry(key)
            && let Ok(compiled) = MlxClosure::new_dyn(|inputs: &MlxVectorArray| {
                let gate = inputs.get(0);
                let per_layer_input = inputs.get(1);
                let activated = gelu_approx(&gate, None);
                vec![multiply(&activated, &per_layer_input, None)]
            })
            .compile(true)
        {
            slot.insert(compiled);
        }
        guard
            .get(&key)
            .and_then(|cls| cls.try_apply(&[gate, per_layer_input]).ok())
    };

    if let Some(mut outputs) = outputs
        && let Some(out) = outputs.pop()
    {
        return out;
    }
    multiply(&gelu_approx(gate, None), per_layer_input, None)
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
                let activated = mlx_sys::ops::silu(&gate, None);
                vec![multiply(&activated, &up, None)]
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
    multiply(&mlx_sys::ops::silu(gate, None), up, None)
}

pub(crate) fn dense_ffn_activation(cfg: &ModelConfig, gate: &MlxArray, up: &MlxArray) -> MlxArray {
    if cfg.uses_geglu {
        if fastpath::prefill_ffn_compile_enabled() {
            geglu(gate, up)
        } else {
            multiply(&gelu_approx(gate, None), up, None)
        }
    } else if fastpath::prefill_ffn_compile_swiglu_enabled() {
        swiglu(gate, up)
    } else {
        multiply(&mlx_sys::ops::silu(gate, None), up, None)
    }
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
        let gate = mlx_slice_last_dim(&out, 0, half);
        let up = mlx_slice_last_dim(&out, half, half * 2);
        (gate, up)
    } else {
        let gate = qw(x, w.gate_proj.as_ref().unwrap());
        let up = qw(x, w.up_proj.as_ref().unwrap());
        (gate, up)
    };
    forward_profile_eval_elapsed(
        profile_decode,
        profile_prefill,
        DecodeProfileStage::PostAttnFfnGateUp,
        gate_up_started,
        &[&gate_out, &up_out],
    );

    // Gemma4 uses GEGLU with fast-approx GELU gate (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate).
    //
    // mlx_lm wraps the gate-activation + multiply in `@partial(mx.compile,
    // shapeless=True) def geglu(gate, x)`. Wiring our equivalent
    // `geglu` helper here was attempted under W2.a but reverted. The source-read
    // follow-up found that MLX default streams, compile cache entries, and Metal
    // command encoders are thread-local; `mlx_set_default_stream` does not
    // register an existing stream index's encoder on another thread. Running the
    // compiled helper from per-request tokio worker threads therefore fails with
    // `"There is no Stream(gpu, N) in current thread"`.
    // See the tracked source-read artifact under
    // `benchmarks/results/mlx-inference/2026-05-14-w2-stream-contract-source-read/`.
    // See `.internal/planning/MLX-DECODE-W2A-GEGLU-FINDINGS.md` for the
    // dead-end record (workarounds attempted + revert rationale). The
    // `geglu` helper is kept (and unit-tested) as a record of the math;
    // the production hot path stays imperative.
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
    let hidden = dense_ffn_activation(cfg, &gate, &up);
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
    let (top_k_indices, mut top_k_weights) = top_k_by_argpartition(
        &expert_scores,
        cfg.moe_expert_count,
        cfg.moe_experts_per_token,
        true,
    );
    // Apply per-expert output scale (initialized to ones; fine-tuned checkpoints may differ).
    if let Some(pes) = &w.router_expert_scale {
        let gathered = take(pes, &top_k_indices, 0, None);
        top_k_weights = multiply(&top_k_weights, &gathered, None);
    }
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
    let (_, group_top2) = top_k_by_argpartition(&grouped, experts_per_group, 2, false);
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
    let (_, group_top2) = top_k_by_argpartition(&grouped, experts_per_group, 2, false);
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

    // Gemma4 experts use GEGLU with fast-approx GELU (matches mlx_lm's `nn.gelu_approx`).
    // Qwen3 uses SwiGLU (SiLU gate). Both routes go through the compile cache when the
    // corresponding `AX_MLX_PREFILL_FFN_COMPILE[_SWIGLU]` env flag is engaged.
    let hidden = dense_ffn_activation(cfg, &gate_out, &up_out);

    // Down projection: [1, seq, top_k, hidden]
    let down_out = squeeze_switch_singleton(&qw_gather(
        &hidden,
        down_exps,
        &gather_inputs.indices,
        gather_inputs.sorted_indices,
    ));
    let down_out = gather_inputs.unsort(down_out);

    // Weighted sum over top_k dimension → [1, seq, hidden]
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
