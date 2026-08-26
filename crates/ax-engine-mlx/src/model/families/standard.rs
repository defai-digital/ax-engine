use mlx_sys::{
    MlxArray, MlxDtype, MlxVectorArray, add, add_rms_norm_pair, astype, rms_norm, rope,
    rope_dynamic, slice, slice_update_dynamic,
};
use std::time::Instant;

/// Env-gated per-stage timing for the batched decode forward
/// (`AX_MLX_BATCHED_PROFILE=1`). Diagnostic only: when enabled it inserts an
/// `eval` barrier at each stage boundary so per-stage GPU wall is attributable
/// (Phase 3.4, docs/performance/batched-decode-ceiling.md). Off by default —
/// the barriers and timers do not run, so the production path is unchanged.
pub(crate) mod batched_profile {
    use std::cell::RefCell;
    use std::sync::OnceLock;

    pub(crate) const STAGES: [&str; 4] = ["pre_attn", "attention", "o_proj", "ffn"];

    thread_local! {
        static ACC: RefCell<[u128; 4]> = const { RefCell::new([0; 4]) };
    }

    pub(crate) fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| {
            matches!(
                std::env::var("AX_MLX_BATCHED_PROFILE").as_deref(),
                Ok("1") | Ok("true") | Ok("yes")
            )
        })
    }

    pub(super) fn record(stage: usize, us: u128) {
        ACC.with(|a| {
            let mut acc = a.borrow_mut();
            acc[stage] = acc[stage].saturating_add(us);
        });
    }

    /// Read and reset the per-stage microsecond accumulators.
    pub fn take() -> [u128; 4] {
        ACC.with(|a| {
            let v = *a.borrow();
            *a.borrow_mut() = [0; 4];
            v
        })
    }
}

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::profile::{
    DecodeProfileStage, Gemma4MoeProfileStage, decode_profile_enabled,
    forward_profile_eval_elapsed, gemma4_moe_profile_enabled, prefill_profile_enabled,
    profile_eval_elapsed, record_gemma4_moe_decode_layer,
};
use super::super::shared::{
    Gemma4PrefillSkipLastFfnPackedGuard, KVConcatBuffer, ProjectionBatchPolicy,
    add_then_multiply_scalar, apply_neox_rope_cos_sin, attention_mask_array,
    attention_output_projection_batched, attention_output_projection_with_post_norm,
    attention_output_projection_with_post_norm_policy, bidirectional_attention,
    direct_qk_norm_rope_route_enabled_for_family, ffn_swiglu, ffn_swiglu_batched,
    ffn_swiglu_plus_residual, ffn_swiglu_row_exact, flatten_attention_output_bhsd,
    flatten_compiled_moe_inputs, flatten_gemma4_dual_path_inputs, full_precision_attention,
    full_precision_attention_with_window, gemma4_prefill_maybe_async_first_kv,
    linear_attention_forward_batched, moe_experts_forward, moe_experts_forward_gemma4,
    moe_experts_forward_with_cloned_weights, moe_experts_forward_with_shared,
    moe_router_deepseek_v3, moe_router_gemma4, moe_router_glm, moe_router_qwen3,
    packed_qkv_kv_head_count, per_layer_input_gate_project, prepare_value_bhsd_from_proj,
    qk_norm_bhsd_from_proj, qk_norm_rope_bhsd_from_proj_with_route, qkv_project,
    qkv_project_batched, qkv_project_last_query, qkv_project_pos0_exact_rest_shared,
    qkv_project_row_exact, qkv_project_with_input_norm, qw,
    qwen_compiled_split_verify_fa_o_proj_ffn, qwen_compiled_split_verify_ffn_plus_residual,
    qwen_prefill_maybe_async_sdpa, qwen_prefill_maybe_last_query_q,
    qwen_prefill_maybe_last_token_bsh, qwen_prefill_maybe_last_token_flat, qwen_prefill_query_seq,
    rms_norm_opt, rope_bhsd_batch_offset_safe, shape_element_count, shared_expert_forward,
};
use crate::attention_mask::{batched_decode_validity_mask_with_window, create_ring_sliding_mask};
use crate::batched_kv_cache::BatchedKvCache;
use crate::batched_linear_state::BatchedLinearState;
use crate::fastpath;
use crate::kv_cache::{MlxAttentionKv, MlxKVCache};
use crate::paged_attention::paged_decode_attention;
use crate::per_layer_compile::{
    apply_layer_gemma4_dual_path_decode, apply_layer_gemma4_dual_path_prefill,
    apply_layer_moe_decode,
};
use crate::weights::LayerWeights;

/// Minimum top-k selection count above which the sort path is taken in Gemma4 MoE.
const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

/// Last-layer residual + pre-FFN RMSNorm for last-position-only prefill.
///
/// When `skip_unused_prefix` is set, slice both residual inputs to the last
/// token *before* add + RMSNorm so the discarded prefix does not pay an
/// add. Otherwise keep the historical add-then-slice-then-rms path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn last_layer_residual_and_ffn_norm(
    hidden: &MlxArray,
    attn_proj: &MlxArray,
    ffn_norm: &MlxArray,
    per_layer_input: Option<&MlxArray>,
    seq: usize,
    hidden_size: usize,
    eps: f32,
    skip_unused_prefix: bool,
) -> (MlxArray, MlxArray, Option<MlxArray>) {
    let hs = hidden_size as i32;
    let last = (seq - 1) as i32;
    let slice_last = |x: &MlxArray, last_dim: i32| {
        if x.shape().get(1).copied().unwrap_or(1) > 1 {
            slice(x, &[0, last, 0], &[1, last + 1, last_dim], &[1, 1, 1], None)
        } else {
            x.clone()
        }
    };
    let sliced_pli = per_layer_input.map(|pli| {
        let pli_last_dim = *pli.shape().last().unwrap_or(&hs);
        slice_last(pli, pli_last_dim)
    });
    if skip_unused_prefix {
        let hidden_last = slice_last(hidden, hs);
        let attn_last = slice_last(attn_proj, hs);
        let (residual, normed) = add_rms_norm_pair(&hidden_last, &attn_last, ffn_norm, eps, None);
        (residual, normed, sliced_pli)
    } else {
        let hidden_for_add = if attn_proj.shape().get(1).copied().unwrap_or(1) == 1
            && hidden.shape().get(1).copied().unwrap_or(1) > 1
        {
            slice_last(hidden, hs)
        } else {
            hidden.clone()
        };
        let residual = add(&hidden_for_add, attn_proj, None);
        let sliced_hidden = slice_last(&residual, hs);
        let normed = rms_norm(&sliced_hidden, Some(ffn_norm), eps, None);
        (sliced_hidden, normed, sliced_pli)
    }
}

// ---------------------------------------------------------------------------
// Post-attention shared pipeline
// ---------------------------------------------------------------------------

/// Shared post-attention pipeline: residual add, optional last-position-only
/// slice, pre-FFN norm, FFN (MoE or dense), residual, and per-layer gating.
///
/// Called by both [`layer_forward`] (causal) and [`layer_forward_bidirectional`]
/// after their respective attention mechanisms produce `attn_proj`.
#[allow(clippy::too_many_arguments)]
fn layer_shell_post_attention(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    attn_proj: &MlxArray,
    seq: usize,
    layer_idx: usize,
    per_layer_input: Option<&MlxArray>,
    last_position_only_after_attention: bool,
    // Cache-only prefill: KV is already written; residual is discarded. Skip
    // the last-layer FFN entirely (even the last-only 1-token FFN).
    skip_post_attention_ffn: bool,
    profile_forward_layer: bool,
    profile_decode_layer: bool,
    profile_prefill_layer: bool,
    profile_gemma4_moe_decode: bool,
    post_attn_started: Option<Instant>,
) -> MlxArray {
    // 15. Residual (+ optional fused pre-FFN RMSNorm).
    let residual_norm_started = profile_forward_layer.then(Instant::now);

    // Cache-only terminal layer: attention already wrote full-seq KV; FFN
    // residual is discarded before the completing decode_step. Return here.
    if skip_post_attention_ffn {
        let hidden = add(hidden, attn_proj, None);
        if let Some(started) = residual_norm_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnResidualNorm,
                started,
                &[&hidden],
            );
        }
        if let Some(started) = post_attn_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttn,
                started,
                &[&hidden],
            );
        }
        return hidden;
    }

    // Direct C++ post-attn FFN composite (Gemma dense packed GEGLU): residual +
    // pre-FFN RMSNorm + gate_up qmatmul + gelu_approx_mul + down qmatmul +
    // residual. One FFI boundary for the whole post-attention dense block.
    // Opt-in via `AX_MLX_DIRECT_CPP_GEMMA4_POST_ATTN_FFN` (prior decode A/B was
    // neutral-to-negative; re-enabled for pure long-prefill thr A/B on M5).
    let last_only_active = last_position_only_after_attention && seq > 1;
    if !last_only_active
        && !profile_forward_layer
        && !profile_decode_layer
        && !profile_prefill_layer
        && !profile_gemma4_moe_decode
        && w.router_proj.is_none()
        && per_layer_input.is_none()
        && w.per_layer_gate.is_none()
        && cfg.uses_geglu
        && fastpath::direct_cpp_gemma4_post_attn_ffn_enabled()
        && let Some(gate_up) = w.gate_up_packed.as_ref()
        && let Some(gu_scales) = gate_up.scales.as_ref()
        && let Some(down) = w.down_proj.as_ref()
        && let Some(down_scales) = down.scales.as_ref()
    {
        let out = mlx_sys::gemma4_post_attn_ffn_block(
            hidden,
            attn_proj,
            &w.ffn_norm,
            w.ffn_post_norm.as_ref(),
            w.layer_scalar.as_ref(),
            &gate_up.weight,
            gu_scales,
            gate_up.biases.as_ref(),
            &down.weight,
            down_scales,
            down.biases.as_ref(),
            gate_up.group_size,
            gate_up.bits,
            cfg.rms_norm_eps,
            None,
        );
        if let Some(started) = residual_norm_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnResidualNorm,
                started,
                &[&out],
            );
        }
        if let Some(started) = post_attn_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttn,
                started,
                &[&out],
            );
        }
        return out;
    }

    // Exact S=2..=4 Qwen dense: compile residual-add + pre-FFN RMS + FFN +
    // residual as one closure. Portable attention RMS+SiLU is not in this
    // graph. Last-only / Gemma / per-layer-input stay on the split path.
    if !last_only_active
        && !profile_forward_layer
        && w.router_proj.is_none()
        && per_layer_input.is_none()
        && w.per_layer_gate.is_none()
        && w.layer_scalar.is_none()
        && let Some(out) =
            qwen_compiled_split_verify_ffn_plus_residual(cfg, w, hidden, attn_proj, layer_idx)
    {
        if let Some(started) = residual_norm_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnResidualNorm,
                started,
                &[&out],
            );
        }
        if let Some(started) = post_attn_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttn,
                started,
                &[&out],
            );
        }
        return out;
    }

    // 15a. Optional last-position-only slice for the terminal prefill layer
    // (slice after residual add, before FFN norm — keeps the FFN single-token).
    // Common multi-token path fuses residual-add + pre-FFN RMSNorm into one
    // compiled composite (mlxcel-style; used every Gemma/Qwen dense layer).
    let (hidden, normed2, per_layer_input_owned) = if last_only_active {
        last_layer_residual_and_ffn_norm(
            hidden,
            attn_proj,
            &w.ffn_norm,
            per_layer_input,
            seq,
            cfg.hidden_size,
            cfg.rms_norm_eps,
            fastpath::should_gemma4_prefill_skip_unused_last_residual(
                &cfg.model_family,
                true,
                seq as i32,
            ),
        )
    } else {
        let (residual, normed2) =
            add_rms_norm_pair(hidden, attn_proj, &w.ffn_norm, cfg.rms_norm_eps, None);
        (residual, normed2, None)
    };
    let per_layer_input: Option<&MlxArray> = if last_only_active {
        per_layer_input_owned.as_ref()
    } else {
        per_layer_input
    };
    // Pre-FFN norm is either fused above or computed on the last-token slice.
    if let Some(started) = residual_norm_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnResidualNorm,
            started,
            &[&normed2],
        );
    }

    // 17. FFN: MoE or dense.
    let _skip_last_ffn_packed = Gemma4PrefillSkipLastFfnPackedGuard::arm(
        last_only_active
            && crate::fastpath::should_gemma4_prefill_skip_unused_last_ffn_packed(
                &cfg.model_family,
                true,
                seq as i32,
            ),
    );
    let ffn_started = profile_forward_layer.then(Instant::now);
    let ffn_out = if w.router_proj.is_some() {
        if cfg.gemma4_moe_router {
            // Try compiled dual-path: shapeless decode, fixed-shape prefill.
            let compiled_result = if fastpath::moe_layer_compile_enabled() {
                flatten_gemma4_dual_path_inputs(&normed2, &hidden, w).and_then(
                    |(inputs, mut schema)| {
                        let cfg_clone = cfg.clone();
                        schema.moe_expert_count = cfg.moe_expert_count;
                        schema.moe_experts_per_token = cfg.moe_experts_per_token;
                        let input_refs: Vec<&MlxArray> = inputs.iter().collect();
                        if seq == 1 {
                            apply_layer_gemma4_dual_path_decode(
                                cfg.compile_cache_identity,
                                layer_idx,
                                &input_refs,
                                move |inputs: &MlxVectorArray| {
                                    vec![schema.forward(inputs, &cfg_clone)]
                                },
                            )
                        } else {
                            // Prefill: fixed-shape graph per leading element count
                            // (chunk size). Host-encoding bound on long S1 shapes.
                            let shape = normed2.shape();
                            let leading = shape[..shape.len().saturating_sub(1)]
                                .iter()
                                .fold(1_i64, |acc, &d| acc.saturating_mul(i64::from(d)));
                            apply_layer_gemma4_dual_path_prefill(
                                cfg.compile_cache_identity,
                                layer_idx,
                                leading,
                                &input_refs,
                                move |inputs: &MlxVectorArray| {
                                    vec![schema.forward(inputs, &cfg_clone)]
                                },
                            )
                        }
                    },
                )
            } else {
                None
            };
            if let Some(result) = compiled_result.and_then(|r| r.into_iter().next()) {
                result
            } else {
                // Gemma4 dual-path: dense sub-block + expert sub-block.
                // Multi-token teacher-forced verify (seq > 1) must match
                // sequential singleton decode for Tier 2 greedy exactness.
                // Batched gather_qmm / router over the seq axis drifts vs
                // pure-direct; process each position as a singleton (depth-2
                // MTP is only seq=3). Prefill-sized chunks still use the
                // batched path below when compile is off and seq is large.
                // Multi-token MoE exact+amort (smokef79):
                // - Dense residual + router: per-pos RowExact
                // - Experts: one gather over S (unique-expert sort for locality)
                // smokef80 Shared dense broke exactness; keep dense per-pos.
                let gemma4_mtp_seq_exact = seq > 1 && seq <= 8;
                if gemma4_mtp_seq_exact {
                    // Dense residual + router per-pos; experts batched over S for amort.
                    // (Full per-pos experts smokef112 still drifted at successive step=18.)
                    use mlx_sys::{concatenate, slice};
                    let hs = cfg.hidden_size as i32;
                    let mut h1_rows = Vec::with_capacity(seq);
                    let mut h2_normed_rows = Vec::with_capacity(seq);
                    let mut idx_rows = Vec::with_capacity(seq);
                    let mut wts_rows = Vec::with_capacity(seq);
                    for t in 0..seq {
                        let n2 = slice(
                            &normed2,
                            &[0, t as i32, 0],
                            &[1, (t + 1) as i32, hs],
                            &[1, 1, 1],
                            None,
                        );
                        let h_res = slice(
                            &hidden,
                            &[0, t as i32, 0],
                            &[1, (t + 1) as i32, hs],
                            &[1, 1, 1],
                            None,
                        );
                        h1_rows.push(ffn_swiglu_row_exact(
                            cfg,
                            w,
                            &n2,
                            w.ffn_post_norm1.as_ref(),
                            layer_idx,
                        ));
                        let h2_norm = w
                            .ffn_norm2
                            .as_ref()
                            .expect("validated Gemma4 MoE layer must include ffn_norm_2");
                        h2_normed_rows.push(rms_norm(
                            &h_res,
                            Some(h2_norm),
                            cfg.rms_norm_eps,
                            None,
                        ));
                        let (top_k_indices, top_k_weights) = moe_router_gemma4(cfg, w, &h_res);
                        idx_rows.push(top_k_indices);
                        wts_rows.push(top_k_weights);
                    }
                    let h2_refs: Vec<&MlxArray> = h2_normed_rows.iter().collect();
                    let h2_normed_all = concatenate(&h2_refs, 1, None);
                    let idx_refs: Vec<&MlxArray> = idx_rows.iter().collect();
                    let idx_all = concatenate(&idx_refs, 1, None);
                    let wts_refs: Vec<&MlxArray> = wts_rows.iter().collect();
                    let wts_all = concatenate(&wts_refs, 1, None);
                    let h2_all =
                        moe_experts_forward_gemma4(cfg, w, &h2_normed_all, &idx_all, &wts_all);
                    let mut rows = Vec::with_capacity(seq);
                    for (t, h1) in h1_rows.iter().enumerate() {
                        let h2 = slice(
                            &h2_all,
                            &[0, t as i32, 0],
                            &[1, (t + 1) as i32, hs],
                            &[1, 1, 1],
                            None,
                        );
                        rows.push(crate::model::shared::combine_gemma4_dual_path_outputs(
                            h1,
                            &h2,
                            w.ffn_post_norm2.as_ref(),
                            w.ffn_post_norm.as_ref(),
                            cfg.rms_norm_eps,
                        ));
                    }
                    let refs: Vec<&MlxArray> = rows.iter().collect();
                    concatenate(&refs, 1, None)
                } else {
                    let dense_started = profile_gemma4_moe_decode.then(Instant::now);
                    let h1 = ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm1.as_ref(), layer_idx);
                    if let Some(started) = dense_started {
                        profile_eval_elapsed(
                            profile_gemma4_moe_decode,
                            Gemma4MoeProfileStage::Dense,
                            started,
                            &[&h1],
                        );
                    }
                    let h2_norm = w
                        .ffn_norm2
                        .as_ref()
                        .expect("validated Gemma4 MoE layer must include ffn_norm_2");
                    let h2_normed = rms_norm(&hidden, Some(h2_norm), cfg.rms_norm_eps, None);
                    let router_started = profile_gemma4_moe_decode.then(Instant::now);
                    let (top_k_indices, top_k_weights) = moe_router_gemma4(cfg, w, &hidden);
                    if let Some(started) = router_started {
                        profile_eval_elapsed(
                            profile_gemma4_moe_decode,
                            Gemma4MoeProfileStage::Router,
                            started,
                            &[&top_k_indices, &top_k_weights],
                        );
                    }
                    if profile_gemma4_moe_decode {
                        let topk_selections = shape_element_count(&top_k_indices.shape());
                        record_gemma4_moe_decode_layer(
                            topk_selections,
                            topk_selections >= SWITCH_GLU_SORT_THRESHOLD,
                        );
                    }
                    let expert_started = profile_gemma4_moe_decode.then(Instant::now);
                    let h2 = moe_experts_forward_gemma4(
                        cfg,
                        w,
                        &h2_normed,
                        &top_k_indices,
                        &top_k_weights,
                    );
                    if let Some(started) = expert_started {
                        profile_eval_elapsed(
                            profile_gemma4_moe_decode,
                            Gemma4MoeProfileStage::Expert,
                            started,
                            &[&h2],
                        );
                    }
                    let post_started = profile_gemma4_moe_decode.then(Instant::now);
                    let out = crate::model::shared::combine_gemma4_dual_path_outputs(
                        &h1,
                        &h2,
                        w.ffn_post_norm2.as_ref(),
                        w.ffn_post_norm.as_ref(),
                        cfg.rms_norm_eps,
                    );
                    if let Some(started) = post_started {
                        profile_eval_elapsed(
                            profile_gemma4_moe_decode,
                            Gemma4MoeProfileStage::Post,
                            started,
                            &[&out],
                        );
                    }
                    out
                }
            }
        } else {
            let router_started = profile_forward_layer.then(Instant::now);
            // Match MTP / family-specialized paths: GLM router, DeepSeek-style
            // sigmoid routing when the manifest sets it, otherwise Qwen softmax.
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, w, &normed2)
            } else if cfg.moe_sigmoid_routing {
                moe_router_deepseek_v3(cfg, w, &normed2)
            } else {
                moe_router_qwen3(cfg, w, &normed2)
            };
            if let Some(started) = router_started {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::MoeRouter,
                    started,
                    &[&top_k_indices, &top_k_weights],
                );
            }
            let shared_started = profile_forward_layer.then(Instant::now);
            let shared_out = if w.shared_gate_proj.is_some() {
                Some(shared_expert_forward(cfg, w, &normed2))
            } else {
                None
            };
            if let Some(started) = shared_started {
                if let Some(shared) = &shared_out {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::MoeSharedExpert,
                        started,
                        &[shared],
                    );
                } else {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::MoeSharedExpert,
                        started,
                        &[],
                    );
                }
            }
            // Try compiled MoE decode closure. SSD-streamed expert layers are
            // excluded: the closure captures the expert weights as graph
            // constants, but streamed layers resolve them at forward time.
            let compiled_result =
                if seq == 1 && w.expert_stream.is_none() && fastpath::moe_layer_compile_enabled() {
                    let cfg_clone = cfg.clone();
                    let (inputs, schema) = flatten_compiled_moe_inputs(
                        &normed2,
                        &top_k_indices,
                        &top_k_weights,
                        w.gate_up_exps_packed.as_ref(),
                        w.gate_exps.as_ref(),
                        w.up_exps.as_ref(),
                        w.down_exps.as_ref(),
                        shared_out.as_ref(),
                    );
                    let input_refs: Vec<&MlxArray> = inputs.iter().collect();
                    apply_layer_moe_decode(
                        cfg.compile_cache_identity,
                        layer_idx,
                        &input_refs,
                        move |inputs: &MlxVectorArray| {
                            let (x, indices, weights, gate_up, gate, up, down, shared) =
                                schema.rebuild(inputs);
                            vec![moe_experts_forward_with_cloned_weights(
                                &cfg_clone, &x, &indices, &weights, gate_up, gate, up, down,
                                shared, None,
                            )]
                        },
                    )
                } else {
                    None
                };
            let out = if let Some(result) = compiled_result.and_then(|r| r.into_iter().next()) {
                result
            } else if let Some(shared) = &shared_out {
                moe_experts_forward_with_shared(
                    cfg,
                    w,
                    &normed2,
                    &top_k_indices,
                    &top_k_weights,
                    shared,
                )
            } else {
                moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights)
            };
            rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
        }
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE layers).
        // 4-bit multi-token short verify: per-position FFN (smokef12 cut 12b4
        // div 3→1; 6-bit keeps batched FFN for formal4 ≥1.20× speed).
        let dense_seq = normed2.shape().get(1).copied().unwrap_or(1) as usize;
        if crate::fastpath::gemma_mt_perpos_ffn_enabled()
            && dense_seq > 1
            && dense_seq <= 8
            && crate::fastpath::multi_token_f32_attention_enabled()
        {
            use mlx_sys::{concatenate, slice};
            let hs = cfg.hidden_size as i32;
            let mut rows = Vec::with_capacity(dense_seq);
            for t in 0..dense_seq {
                let row = slice(
                    &normed2,
                    &[0, t as i32, 0],
                    &[1, (t + 1) as i32, hs],
                    &[1, 1, 1],
                    None,
                );
                rows.push(ffn_swiglu(
                    cfg,
                    w,
                    &row,
                    w.ffn_post_norm.as_ref(),
                    layer_idx,
                ));
            }
            let refs: Vec<&MlxArray> = rows.iter().collect();
            concatenate(&refs, 1, None)
        } else if seq == 1
            && w.ffn_post_norm.is_none()
            && w.per_layer_gate.is_none()
            && w.layer_scalar.is_none()
            && per_layer_input.is_none()
        {
            ffn_swiglu_plus_residual(cfg, w, &normed2, None, layer_idx, &hidden)
        } else {
            ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref(), layer_idx)
        }
    };
    if let Some(started) = ffn_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnFfn,
            started,
            &[&ffn_out],
        );
    }

    // 18-19. Residual + per-layer input gating.
    let residual_gate_started = profile_forward_layer.then(Instant::now);
    let fuse_ffn_residual = seq == 1
        && w.ffn_post_norm.is_none()
        && w.per_layer_gate.is_none()
        && w.layer_scalar.is_none()
        && per_layer_input.is_none()
        && w.router_proj.is_none();
    let out = if fuse_ffn_residual {
        ffn_out
    } else if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(pli)) = (
        w.per_layer_gate.as_ref(),
        w.per_layer_proj_w.as_ref(),
        w.per_layer_post_norm.as_ref(),
        per_layer_input,
    ) {
        let residual = add(&hidden, &ffn_out, None);
        let projected = per_layer_input_gate_project(
            cfg.compile_cache_identity,
            &qw(&residual, gate_w),
            pli,
            proj_w,
        );
        let normed = rms_norm(&projected, Some(post_norm), cfg.rms_norm_eps, None);
        if let Some(scalar) = &w.layer_scalar {
            add_then_multiply_scalar(&residual, &normed, scalar)
        } else {
            add(&residual, &normed, None)
        }
    } else if let Some(scalar) = &w.layer_scalar {
        add_then_multiply_scalar(&hidden, &ffn_out, scalar)
    } else {
        add(&hidden, &ffn_out, None)
    };
    if let Some(started) = residual_gate_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttnResidualGate,
            started,
            &[&out],
        );
    }
    if let Some(started) = post_attn_started {
        forward_profile_eval_elapsed(
            profile_decode_layer,
            profile_prefill_layer,
            DecodeProfileStage::PostAttn,
            started,
            &[&out],
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Causal layer forward
// ---------------------------------------------------------------------------

/// Full layer forward for standard GQA attention families (Gemma4, Gemma3, Qwen3).
///
/// Handles per-head QK norm, KV-sharing (Gemma4), sliding window attention,
/// Gemma4 dual-path MoE, Qwen3 MoE, dense FFN, and per-layer input gating
/// (Gemma4).
///
/// `last_position_only_after_attention`: when `true` and `seq > 1`, the layer
/// slices its attention-residual stream to the last sequence position before
/// running pre-FFN norm + FFN + post-FFN residual. The KV cache writes have
/// already happened inside attention, so the slice is safe; the FFN, gating,
/// and layer-scalar steps then operate on a `[1, 1, hidden]` tensor instead
/// of `[1, seq, hidden]`. This matches the lazy-eval prune that mlx-lm gets
/// for free on the last layer when the model output is discarded.
///
/// `skip_post_attention_ffn`: when `true`, write only the layer's cache side
/// effects (K/V append, or nothing for KV-shared consumers) and return
/// without SDPA / o_proj / residual / FFN. Use only for the **last** layer of
/// a cache-only prefill (`FinalLogitsMode::Skip`) where residual is discarded.
///
/// Callers must only set last-only / skip-FFN for the **last transformer
/// layer** in a prefill pass. Setting either on a non-terminal layer breaks
/// correctness: the next layer would receive a wrong-shaped residual or the
/// residual stream would skip an FFN.
pub(crate) struct FullAttentionVerifyLayerOutput {
    pub hidden: MlxArray,
    pub k: MlxArray,
    pub v: MlxArray,
}

/// Pure dense-Qwen full-attention layer for the whole target-verifier graph.
///
/// Logical K/V and the current RoPE offset are explicit graph inputs. The
/// offset remains array-valued, while the manual NeoX RoPE path avoids the
/// tensor-indexed table slice that MLX shapeless compilation cannot infer.
pub(crate) fn layer_forward_verify_functional(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    layer_idx: usize,
    rope_offset: &MlxArray,
    cached_k: &MlxArray,
    cached_v: &MlxArray,
    rope_cos: &MlxArray,
    rope_sin: &MlxArray,
) -> Option<FullAttentionVerifyLayerOutput> {
    let (head_dim, rope_theta, rope_dims, layer_rope_freqs, sliding_window, kv_source, v_norm) =
        layer_params(cfg, layer_idx);
    if sliding_window.is_some()
        || kv_source.is_some()
        || w.router_proj.is_some()
        || w.ffn_post_norm.is_some()
        || w.per_layer_gate.is_some()
        || w.layer_scalar.is_some()
        || cfg.uses_geglu
    {
        return None;
    }
    let seq = hidden.shape().get(1).copied()? as usize;
    if !(2..=4).contains(&seq) {
        return None;
    }
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let (q_raw, k_raw, v_raw, attn_gate) = qkv_project(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape().get(2).copied()? as usize).checked_div(head_dim)?;
    let v_new =
        prepare_value_bhsd_from_proj(&v_raw, v_norm, kv_heads, head_dim, seq, cfg.rms_norm_eps);
    let q = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let _ = (rope_theta, layer_rope_freqs);
    let q_rope = astype(
        &apply_neox_rope_cos_sin(&q, rope_dims as i32, rope_cos, rope_sin),
        q.dtype(),
        None,
    );
    let k_new = astype(
        &apply_neox_rope_cos_sin(&k, rope_dims as i32, rope_cos, rope_sin),
        k.dtype(),
        None,
    );
    let k_all = slice_update_dynamic(cached_k, &k_new, rope_offset, &[2], None);
    let v_all = slice_update_dynamic(cached_v, &v_new, rope_offset, &[2], None);
    let capacity = k_all.shape().get(2).copied()?;
    let mask = super::super::shared::fixed_capacity_causal_mask(rope_offset, seq as i32, capacity);
    let attn = full_precision_attention(&q_rope, &k_all, &v_all, cfg.query_scale, seq, &Some(mask));
    let flat = flatten_attention_output_bhsd(&attn, seq, cfg.n_heads, head_dim);
    let attn_proj = attention_output_projection_with_post_norm_policy(
        &flat,
        attn_gate.as_ref(),
        w.o_proj.as_ref()?,
        w.attn_post_norm.as_ref(),
        cfg.rms_norm_eps,
        ProjectionBatchPolicy::Shared,
    );
    let (residual, normed2) =
        add_rms_norm_pair(hidden, &attn_proj, &w.ffn_norm, cfg.rms_norm_eps, None);
    let hidden = ffn_swiglu_plus_residual(cfg, w, &normed2, None, layer_idx, &residual);
    Some(FullAttentionVerifyLayerOutput {
        hidden,
        k: k_all,
        v: v_all,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    shared_mask: Option<&Option<MlxArray>>,
    last_position_only_after_attention: bool,
    skip_post_attention_ffn: bool,
) -> MlxArray {
    layer_forward_internal(
        cfg,
        w,
        hidden,
        cache,
        layer_idx,
        token_offset,
        per_layer_input,
        shared_mask,
        last_position_only_after_attention,
        skip_post_attention_ffn,
        None,
    )
}

/// Standard transformer layer with explicit Qwen multimodal rotary factors.
///
/// Linear-attention layers never call this entry point. Full-attention layers
/// share the exact post-attention/FFN path with [`layer_forward`]; only Q/K
/// rotary application differs during the initial visual prefill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward_with_mrope(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    shared_mask: Option<&Option<MlxArray>>,
    last_position_only_after_attention: bool,
    skip_post_attention_ffn: bool,
    mrope: &crate::qwen3_vl::QwenMropeCosSin,
) -> MlxArray {
    layer_forward_internal(
        cfg,
        w,
        hidden,
        cache,
        layer_idx,
        token_offset,
        per_layer_input,
        shared_mask,
        last_position_only_after_attention,
        skip_post_attention_ffn,
        Some(mrope),
    )
}

#[allow(clippy::too_many_arguments)]
fn layer_forward_internal(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    shared_mask: Option<&Option<MlxArray>>,
    last_position_only_after_attention: bool,
    skip_post_attention_ffn: bool,
    mrope: Option<&crate::qwen3_vl::QwenMropeCosSin>,
) -> MlxArray {
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        sliding_window,
        kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);

    let seq = hidden.shape()[1] as usize;
    crate::model::shared::set_qwen_prefill_reuse_rope_active(
        fastpath::should_qwen_prefill_reuse_rope(&cfg.model_family, seq as i32),
    );
    crate::model::shared::set_qwen_prefill_dequant_dense_family(matches!(
        cfg.model_family.to_ascii_lowercase().as_str(),
        "qwen3_5" | "qwen3_next"
    ));
    let protected_prefix_window =
        (cfg.model_family == "unlimited_ocr" && seq == 1 && cache.seq_len() > 0)
            .then_some(cfg.protected_prefix_sliding_window)
            .flatten();

    // Cache-only terminal layer: residual/logits are discarded. Only KV (or
    // linear-state) side effects matter. KV-shared consumers write nothing, so
    // the entire layer is a no-op. Normal layers project K/V (not Q when split
    // weights exist) and append — skipping SDPA, o_proj, residual, and FFN.
    if skip_post_attention_ffn {
        if kv_source.is_some() {
            return hidden.clone();
        }
        let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
        let ring_layout = cache.sliding_ring_layout(sliding_window, seq);
        let (k_raw, v_raw) = if let (Some(k_w), v_w) = (w.k_proj.as_ref(), w.v_proj.as_ref()) {
            // Split K/V: skip the Q matmul entirely (major last-layer saving).
            let k = qw(&normed, k_w);
            let v = v_w.map(|vw| qw(&normed, vw)).unwrap_or_else(|| k.clone());
            (k, v)
        } else {
            // Packed QKV: still pay the packed matmul but drop Q after.
            let (_, k, v, _) = qkv_project(cfg, w, &normed, head_dim);
            (k, v)
        };
        let kv_heads = (k_raw.shape()[2] as usize)
            .checked_div(head_dim)
            .expect("k projection output must divide by head_dim");
        let v = prepare_value_bhsd_from_proj(
            &v_raw,
            v_norm_no_scale,
            kv_heads,
            head_dim,
            seq,
            cfg.rms_norm_eps,
        );
        let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
        let (rope_base, rope_freqs_ref) = rope_freqs
            .map(|f| (None, Some(f)))
            .unwrap_or((Some(rope_theta), None));
        let use_direct_k_rope = mrope.is_none()
            && direct_qk_norm_rope_route_enabled_for_family(
                cfg.model_family.as_str(),
                w.k_norm.as_ref(),
            );
        let k_rope = if use_direct_k_rope {
            qk_norm_rope_bhsd_from_proj_with_route(
                &k_raw,
                w.k_norm.as_ref(),
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
                rope_dims,
                rope_base,
                token_offset,
                rope_freqs_ref,
                true,
            )
        } else {
            let k = qk_norm_bhsd_from_proj(
                &k_raw,
                w.k_norm.as_ref(),
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );
            if let Some(mrope) = mrope {
                crate::qwen3_vl::apply_interleaved_mrope(&k, mrope, rope_dims)
            } else {
                rope_bhsd_batch_offset_safe(
                    &k,
                    rope_dims as i32,
                    rope_base,
                    token_offset as i32,
                    rope_freqs_ref,
                )
            }
        };
        let retained_window = if seq == 1 || ring_layout.is_some() {
            sliding_window
        } else if sliding_window.is_some() && fastpath::multi_token_window_views_enabled() {
            match shared_mask {
                Some(Some(mask)) => mask.shape().last().map(|&len| len as usize),
                _ => sliding_window.map(|window| window + seq - 1),
            }
        } else {
            None
        };
        if let Some(window) = protected_prefix_window {
            let _ = cache
                .append_with_protected_prefix_window_for_attention(layer_idx, k_rope, v, window);
        } else {
            let _ = cache.append_with_retained_window(layer_idx, k_rope, v, retained_window);
        }
        return hidden.clone();
    }

    // 1. Attention norm (may be fused into packed QKV below).
    // Only skip the standalone rms when the fused QKV path will actually
    // run. Exact / moe-mt identity skip the fuse and still need `normed`.
    // The exact contract only constrains decode shapes (seq <= 4); prefill
    // keeps the fused path (fastpath::qwen_linear_mtp_exact_for_seq).
    let skip_qkv_fuse = crate::fastpath::qwen_linear_mtp_exact_for_seq(seq as i32)
        || crate::fastpath::moe_mt_bf16_identity_enabled();
    let fuse_attn_norm_qkv = fastpath::should_call_attn_norm_qkv_fuse(
        fastpath::should_attn_norm_qkv_fuse(&cfg.model_family, seq as i32),
        w.qkv_packed.is_some(),
        kv_source.is_some(),
        skip_qkv_fuse,
    );
    let normed = if fuse_attn_norm_qkv {
        None
    } else {
        Some(rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None))
    };
    if let Some(normed) = normed.as_ref() {
        crate::model::shared::qwen_prefill_maybe_eval_attn_input(
            normed,
            &cfg.model_family,
            seq as i32,
        );
    }

    let ring_layout = cache.sliding_ring_layout(sliding_window, seq);
    let profile_gemma4_moe_decode =
        cfg.gemma4_moe_router && seq == 1 && gemma4_moe_profile_enabled();
    let attention_started = profile_gemma4_moe_decode.then(Instant::now);
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    // Opt-in one-call fused offset-0 prefill attention (mlxcel residual; see
    // `AX_MLX_FUSED_PREFILL_ATTENTION` in fastpath). Strict Phase-1 gate: the
    // fused C++ path implements exactly rms_norm → packed affine QKV →
    // per-head QK rms_norm → rope(theta, offset 0) → maskless-causal SDPA →
    // o-proj, so every feature outside that contract falls through to the
    // portable path below.
    let fused_prefill = 'fused: {
        let dbg = fastpath::prefill_time_debug_env();
        if !fastpath::fused_prefill_attention_should_try_for_seq(&cfg.model_family, seq as i32)
            || seq <= 1
        {
            break 'fused None;
        }
        // Offset chunks (chunked prefill continuation) fuse via the two-stage
        // qkv_rope -> cache append -> sdpa_oproj pair; that pair is only
        // exact for full-attention layers, where bottom-right "causal"
        // matches the portable mask over the whole cached history.
        let offset_chunk = token_offset != 0 || cache.seq_len() != 0;
        let gate_reason = if kv_source.is_some() {
            Some("kv_source")
        } else if last_position_only_after_attention
            && fastpath::should_gemma4_prefill_last_query_p128(&cfg.model_family, true, seq as i32)
        {
            // Last-layer last-query needs the portable Q/SDPA/o_proj slices;
            // fused C++ still emits a full-seq last-layer attention.
            Some("last_query")
        } else if mrope.is_some() {
            Some("mrope")
        } else if !offset_chunk && ring_layout.is_some() {
            Some("ring_layout")
        } else if protected_prefix_window.is_some() {
            Some("protected_prefix")
        } else if !fastpath::fused_prefill_attention_family_supported(&cfg.model_family) {
            Some("family")
        } else if fastpath::fused_prefill_qwen_skip_offset(&cfg.model_family, offset_chunk) {
            Some("qwen_offset_chunk")
        } else if !matches!(head_dim, 64 | 80 | 128 | 256) {
            // Mirrors mlxcel's NAX gate: fast SDPA only has steel kernels
            // for these head dims; anything else (Gemma 4 global layers at
            // 512) hits MLX's slow reference path and loses to the portable
            // route (measured +0.4ms offset-0, +8ms offset chunks on 12B).
            Some("head_dim")
        } else if offset_chunk && token_offset == 0 {
            Some("cache_ahead_of_offset")
        } else if !offset_chunk && sliding_window.is_some_and(|window| seq > window) {
            Some("sliding_gt_window")
        } else {
            None
        };
        if let Some(reason) = gate_reason {
            if dbg {
                eprintln!("AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: {reason}");
            }
            break 'fused None;
        }
        let Some(o_proj) = w.o_proj.as_ref() else {
            break 'fused None;
        };
        // Typed predicate, not a raw `mode` string compare: a mislabeled
        // MXFP4 pack (`mode=="affine"`, 4/32, no group biases) resolves to
        // Mxfp4 via `mlx_quantization_mode()` and must not reach the fused
        // helpers as affine (MLX panics without biases). Genuine scales-only
        // MXFP4 is hosted — the shim infers the mode per projection from the
        // absent bias channel.
        if !o_proj.is_fused_qmm_quantized() {
            break 'fused None;
        }
        let affine_matching = |qw: &crate::weights::QuantizedWeight| {
            qw.is_fused_qmm_quantized()
                && qw.group_size == o_proj.group_size
                && qw.bits == o_proj.bits
        };
        let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
        if offset_chunk {
            let (Some(q_proj), Some(k_proj)) = (w.q_proj.as_ref(), w.k_proj.as_ref()) else {
                if dbg {
                    eprintln!(
                        "AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: offset_packed"
                    );
                }
                break 'fused None;
            };
            // Value-from-key layers (v_proj absent) reuse the K projection
            // weights: identical op on identical input matches the portable
            // `v_raw = k_raw` reuse bit-for-bit.
            let v_proj = w.v_proj.as_ref().unwrap_or(k_proj);
            // Head counts derive from the projection out-features like the
            // portable path (Gemma 4 global layers: wider heads, MQA KV).
            let n_heads_layer = q_proj.weight.shape()[0] as usize / head_dim;
            let kv_heads_layer = k_proj.weight.shape()[0] as usize / head_dim;
            if !affine_matching(q_proj) || !affine_matching(k_proj) || !affine_matching(v_proj) {
                break 'fused None;
            }
            let Some((q_rope, k_rope, v)) = mlx_sys::ops::fused_qkv_rope_split(
                hidden,
                &w.attn_norm,
                cfg.rms_norm_eps,
                (
                    &q_proj.weight,
                    q_proj.scales.as_ref().expect("checked above"),
                    q_proj.biases.as_ref(),
                ),
                (
                    &k_proj.weight,
                    k_proj.scales.as_ref().expect("checked above"),
                    k_proj.biases.as_ref(),
                ),
                (
                    &v_proj.weight,
                    v_proj.scales.as_ref().expect("checked above"),
                    v_proj.biases.as_ref(),
                ),
                w.q_norm.as_ref(),
                w.k_norm.as_ref(),
                cfg.rms_norm_eps,
                v_norm_no_scale,
                w.v_proj.is_none(),
                n_heads_layer as i32,
                kv_heads_layer as i32,
                head_dim as i32,
                rope_dims as i32,
                rope_theta,
                rope_freqs,
                token_offset as i32,
                q_proj.group_size,
                q_proj.bits,
                None,
            ) else {
                if dbg {
                    eprintln!(
                        "AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: shim_error"
                    );
                }
                break 'fused None;
            };
            // Mirror the portable retained-window policy so sliding-window
            // and rotated-ring layers keep their exact cache geometry.
            let retained_window = if ring_layout.is_some() {
                sliding_window
            } else if sliding_window.is_some() && fastpath::multi_token_window_views_enabled() {
                match shared_mask {
                    Some(Some(mask)) => mask.shape().last().map(|&len| len as usize),
                    _ => sliding_window.map(|window| window + seq - 1),
                }
            } else {
                None
            };
            let attention_kv = cache.append_with_retained_window_for_attention(
                layer_idx,
                k_rope,
                v,
                retained_window,
            );
            let key_len = attention_kv.key_len();
            let (k_full, v_full) = attention_kv.into_dense();
            // Mirror the portable mask selection: hoisted shared mask when
            // its key length matches, ring mask when rotated, plain
            // windowed-causal otherwise. `None` means maskless bottom-right
            // causal inside the shim.
            let shared_usable = shared_mask.is_some_and(|m| match m.as_ref() {
                Some(mask) => mask.shape().last().is_some_and(|&k| k as usize == key_len),
                None => true,
            });
            let local_mask: Option<MlxArray> = if shared_usable {
                None
            } else {
                match ring_layout {
                    Some(ring) if ring.needs_mask(seq) && key_len == ring.capacity => Some(
                        create_ring_sliding_mask(seq, ring.window, ring.capacity, ring.write_start),
                    ),
                    _ => attention_mask_array(seq, key_len, sliding_window),
                }
            };
            let sdpa_mask: Option<&MlxArray> = if shared_usable {
                shared_mask.and_then(|m| m.as_ref())
            } else {
                local_mask.as_ref()
            };
            let out = match mlx_sys::ops::fused_sdpa_oproj(
                &q_rope,
                &k_full,
                &v_full,
                cfg.query_scale,
                sdpa_mask,
                &o_proj.weight,
                o_proj.scales.as_ref().expect("checked above"),
                o_proj.biases.as_ref(),
                o_proj.group_size,
                o_proj.bits,
                None,
            ) {
                Some(out) => {
                    if dbg {
                        eprintln!(
                            "AX_PREFILL_TIME_DEBUG fused_prefill_attention engaged layer={layer_idx} offset={token_offset}"
                        );
                    }
                    if let Some(post_norm) = w.attn_post_norm.as_ref() {
                        rms_norm(&out, Some(post_norm), cfg.rms_norm_eps, None)
                    } else {
                        out
                    }
                }
                None => {
                    // K/V are already appended, so the portable block below
                    // must not run; finish with the portable SDPA + o-proj
                    // helpers over the same appended views and mask.
                    let fallback_mask: Option<MlxArray> = sdpa_mask.cloned();
                    let attn = full_precision_attention_with_window(
                        &q_rope,
                        &k_full,
                        &v_full,
                        cfg.query_scale,
                        seq,
                        &fallback_mask,
                        sliding_window.filter(|_| ring_layout.is_none()),
                        ring_layout,
                    );
                    let attn_flat =
                        flatten_attention_output_bhsd(&attn, seq, n_heads_layer, head_dim);
                    attention_output_projection_with_post_norm(
                        &attn_flat,
                        None,
                        o_proj,
                        w.attn_post_norm.as_ref(),
                        cfg.rms_norm_eps,
                    )
                }
            };
            break 'fused Some(out);
        }
        let fused_result = if let Some(packed) = w.qkv_packed.as_ref() {
            if !affine_matching(packed) {
                break 'fused None;
            }
            let packed_rows = packed.weight.shape().first().copied().unwrap_or(0) as usize;
            let Some(kv_heads) = packed_qkv_kv_head_count(cfg, head_dim, packed_rows) else {
                if dbg {
                    eprintln!(
                        "AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: packed_qkv_geometry"
                    );
                }
                break 'fused None;
            };
            mlx_sys::ops::fused_causal_prefill_attention(
                hidden,
                &w.attn_norm,
                cfg.rms_norm_eps,
                &packed.weight,
                packed.scales.as_ref().expect("checked above"),
                packed.biases.as_ref(),
                w.q_norm.as_ref(),
                w.k_norm.as_ref(),
                cfg.rms_norm_eps,
                v_norm_no_scale,
                cfg.n_heads as i32,
                kv_heads as i32,
                head_dim as i32,
                rope_dims as i32,
                rope_theta,
                rope_freqs,
                cfg.query_scale,
                &o_proj.weight,
                o_proj.scales.as_ref().expect("checked above"),
                o_proj.biases.as_ref(),
                packed.group_size,
                packed.bits,
                w.attn_post_norm.as_ref().filter(|_| {
                    fastpath::should_gemma4_fused_prefill_fold_post_norm(
                        &cfg.model_family,
                        seq as i32,
                        true,
                    )
                }),
                None,
            )
        } else if let (Some(q_proj), Some(k_proj)) = (w.q_proj.as_ref(), w.k_proj.as_ref()) {
            // Value-from-key layers (v_proj absent) reuse the K projection
            // weights, matching the portable `v_raw = k_raw` reuse.
            let v_proj = w.v_proj.as_ref().unwrap_or(k_proj);
            if !affine_matching(q_proj) || !affine_matching(k_proj) || !affine_matching(v_proj) {
                break 'fused None;
            }
            // Head counts derive from the projection out-features like the
            // portable path (Gemma 4 global layers: wider heads, MQA KV).
            let n_heads_layer = q_proj.weight.shape()[0] as usize / head_dim;
            let kv_heads_layer = k_proj.weight.shape()[0] as usize / head_dim;
            mlx_sys::ops::fused_causal_prefill_attention_split(
                hidden,
                &w.attn_norm,
                cfg.rms_norm_eps,
                (
                    &q_proj.weight,
                    q_proj.scales.as_ref().expect("checked above"),
                    q_proj.biases.as_ref(),
                ),
                (
                    &k_proj.weight,
                    k_proj.scales.as_ref().expect("checked above"),
                    k_proj.biases.as_ref(),
                ),
                (
                    &v_proj.weight,
                    v_proj.scales.as_ref().expect("checked above"),
                    v_proj.biases.as_ref(),
                ),
                w.q_norm.as_ref(),
                w.k_norm.as_ref(),
                cfg.rms_norm_eps,
                v_norm_no_scale,
                w.v_proj.is_none(),
                n_heads_layer as i32,
                kv_heads_layer as i32,
                head_dim as i32,
                rope_dims as i32,
                rope_theta,
                rope_freqs,
                cfg.query_scale,
                &o_proj.weight,
                o_proj.scales.as_ref().expect("checked above"),
                o_proj.biases.as_ref(),
                q_proj.group_size,
                q_proj.bits,
                w.attn_post_norm.as_ref().filter(|_| {
                    fastpath::should_gemma4_fused_prefill_fold_post_norm(
                        &cfg.model_family,
                        seq as i32,
                        true,
                    )
                }),
                None,
            )
        } else {
            if dbg {
                eprintln!(
                    "AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: no_projections"
                );
            }
            break 'fused None;
        };
        let Some((out, k_rope, v)) = fused_result else {
            if dbg {
                eprintln!("AX_PREFILL_TIME_DEBUG fused_prefill skip layer={layer_idx}: shim_error");
            }
            break 'fused None;
        };
        if fastpath::prefill_time_debug_env() {
            eprintln!("AX_PREFILL_TIME_DEBUG fused_prefill_attention engaged layer={layer_idx}");
        }
        let retained_window =
            if sliding_window.is_some() && fastpath::multi_token_window_views_enabled() {
                match shared_mask {
                    Some(Some(mask)) => mask.shape().last().map(|&len| len as usize),
                    _ => sliding_window.map(|window| window + seq - 1),
                }
            } else {
                None
            };
        gemma4_prefill_maybe_async_first_kv(&k_rope, &v, &cfg.model_family, seq as i32);
        let _ =
            cache.append_with_retained_window_for_attention(layer_idx, k_rope, v, retained_window);
        let folded_post_norm = fastpath::should_gemma4_fused_prefill_fold_post_norm(
            &cfg.model_family,
            seq as i32,
            w.attn_post_norm.is_some(),
        );
        let out = if folded_post_norm {
            out
        } else if let Some(post_norm) = w.attn_post_norm.as_ref() {
            rms_norm(&out, Some(post_norm), cfg.rms_norm_eps, None)
        } else {
            out
        };
        Some(out)
    };

    // 2-7. QKV projections + RoPE + KV cache append + SDPA.
    let post_attn_started;
    let attn_proj = if let Some(fused) = fused_prefill {
        post_attn_started = None;
        fused
    } else {
        let pre_sdpa_started = profile_forward_layer.then(Instant::now);
        let (q_rope, attention_kv, attn_gate) = if let Some(src_layer) = kv_source {
            // KV-shared layer (Gemma4 layers 24-41): compute Q only.
            let normed = normed
                .as_ref()
                .expect("KV-shared path keeps portable attn_norm");
            let q_raw = qw(
                normed,
                w.q_proj.as_ref().expect("KV-shared layer must have q_proj"),
            );
            let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
            let (rope_base, rope_freqs_ref) = rope_freqs
                .map(|f| (None, Some(f)))
                .unwrap_or((Some(rope_theta), None));
            let direct_q_rope = mrope.is_none()
                && direct_qk_norm_rope_route_enabled_for_family(
                    cfg.model_family.as_str(),
                    w.q_norm.as_ref(),
                );
            let q_rope = if direct_q_rope {
                let qk_norm_started = profile_forward_layer.then(Instant::now);
                let q_rope = qk_norm_rope_bhsd_from_proj_with_route(
                    &q_raw,
                    w.q_norm.as_ref(),
                    cfg.n_heads,
                    head_dim,
                    seq,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    token_offset,
                    rope_freqs_ref,
                    direct_q_rope,
                );
                if let Some(started) = qk_norm_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaQkNorm,
                        started,
                        &[&q_rope],
                    );
                }
                q_rope
            } else {
                let qk_norm_started = profile_forward_layer.then(Instant::now);
                let q = qk_norm_bhsd_from_proj(
                    &q_raw,
                    w.q_norm.as_ref(),
                    cfg.n_heads,
                    head_dim,
                    seq,
                    cfg.rms_norm_eps,
                );
                if let Some(started) = qk_norm_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaQkNorm,
                        started,
                        &[&q],
                    );
                }
                let rope_kv_started = profile_forward_layer.then(Instant::now);
                let q_rope = if let Some(mrope) = mrope {
                    crate::qwen3_vl::apply_interleaved_mrope(&q, mrope, rope_dims)
                } else {
                    rope_bhsd_batch_offset_safe(
                        &q,
                        rope_dims as i32,
                        rope_base,
                        token_offset as i32,
                        rope_freqs_ref,
                    )
                };
                if let Some(started) = rope_kv_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaRopeKv,
                        started,
                        &[&q_rope],
                    );
                }
                q_rope
            };
            let (ck, cv) = cache.peek_source_kv(src_layer, seq);
            (q_rope, MlxAttentionKv::Dense { k: ck, v: cv }, None)
        } else {
            // Normal layer: compute Q, K, V from own projections.
            let qkv_proj_started = profile_forward_layer.then(Instant::now);
            // MoE multi-token (smokef99/104): pos0 RowExact + rest Shared QKV.
            // Formal exact at w≈1.016 (smokef99); full Shared breaks exactness.
            let moe_mt_exact = crate::fastpath::moe_mt_bf16_identity_enabled();
            let last_q_proj = fastpath::should_qwen_prefill_last_query_q_proj(
                &cfg.model_family,
                last_position_only_after_attention,
                seq as i32,
            ) || fastpath::should_gemma4_prefill_last_query_p128(
                &cfg.model_family,
                last_position_only_after_attention,
                seq as i32,
            );
            let last_q_x = if last_q_proj {
                normed
                    .as_ref()
                    .and_then(|x| qwen_prefill_maybe_last_token_bsh(x, true))
            } else {
                None
            };
            let (q_raw, k_raw, v_raw, attn_gate_raw) = if fuse_attn_norm_qkv && !skip_qkv_fuse {
                qkv_project_with_input_norm(
                    cfg,
                    w,
                    hidden,
                    head_dim,
                    Some(&w.attn_norm),
                    cfg.rms_norm_eps,
                )
            } else if moe_mt_exact {
                // pos0-RowExact + rest-Shared QKV (smokef99/104 best exact @~1.02).
                // Full Shared gains speed but breaks formal A/B (smokef81/108).
                // Long-context agent: full RowExact — hybrid Shared rest drifts
                // after long prefill (smokef104 agent first_diff@14 @prompt~4k).
                let long_ctx = token_offset >= 512;
                if long_ctx {
                    qkv_project_row_exact(
                        cfg,
                        w,
                        normed
                            .as_ref()
                            .expect("portable path materializes attn_norm"),
                        head_dim,
                    )
                } else {
                    qkv_project_pos0_exact_rest_shared(
                        cfg,
                        w,
                        normed
                            .as_ref()
                            .expect("portable path materializes attn_norm"),
                        head_dim,
                    )
                }
            } else if let Some(ref last_x) = last_q_x {
                qkv_project_last_query(
                    cfg,
                    w,
                    normed
                        .as_ref()
                        .expect("portable path materializes attn_norm"),
                    last_x,
                    head_dim,
                )
            } else {
                qkv_project(
                    cfg,
                    w,
                    normed
                        .as_ref()
                        .expect("portable path materializes attn_norm"),
                    head_dim,
                )
            };
            let kv_heads = (k_raw.shape()[2] as usize)
                .checked_div(head_dim)
                .expect("k projection output must divide by head_dim");
            // Packed QKV ignores last_x and keeps full-seq Q; split last-query
            // shrinks Q to S=1. RoPE offset follows the actual Q length.
            let q_seq = q_raw
                .shape()
                .get(1)
                .copied()
                .filter(|&s| s > 0)
                .map(|s| s as usize)
                .unwrap_or(seq);
            let q_offset = token_offset.saturating_add(seq.saturating_sub(q_seq));
            // Packed / last-query-SDPA-only still has full-seq Q after qkv.
            // Slice before QK-norm so unused prefix tokens skip RMSNorm+RoPE.
            let (q_raw, q_seq, q_offset) = if fastpath::should_qwen_prefill_skip_unused_qk_norm(
                &cfg.model_family,
                last_position_only_after_attention,
                seq as i32,
            ) && let Some(last_q) =
                qwen_prefill_maybe_last_token_bsh(&q_raw, q_seq > 1)
            {
                (
                    last_q,
                    1usize,
                    token_offset.saturating_add(seq.saturating_sub(1)),
                )
            } else {
                (q_raw, q_seq, q_offset)
            };
            if let Some(started) = qkv_proj_started {
                let mut refs: Vec<&MlxArray> = vec![&q_raw, &k_raw, &v_raw];
                if let Some(g) = attn_gate_raw.as_ref() {
                    refs.push(g);
                }
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::PreSdpaQkvProj,
                    started,
                    &refs,
                );
            }

            let v = prepare_value_bhsd_from_proj(
                &v_raw,
                v_norm_no_scale,
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );

            let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
            let (rope_base, rope_freqs_ref) = rope_freqs
                .map(|f| (None, Some(f)))
                .unwrap_or((Some(rope_theta), None));
            let use_direct_q_rope = mrope.is_none()
                && direct_qk_norm_rope_route_enabled_for_family(
                    cfg.model_family.as_str(),
                    w.q_norm.as_ref(),
                );
            let use_direct_k_rope = mrope.is_none()
                && direct_qk_norm_rope_route_enabled_for_family(
                    cfg.model_family.as_str(),
                    w.k_norm.as_ref(),
                );
            let use_direct_qk_rope = use_direct_q_rope || use_direct_k_rope;
            let (q_rope, k_rope) = if use_direct_qk_rope {
                let qk_norm_started = profile_forward_layer.then(Instant::now);
                let q_rope = qk_norm_rope_bhsd_from_proj_with_route(
                    &q_raw,
                    w.q_norm.as_ref(),
                    cfg.n_heads,
                    head_dim,
                    q_seq,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    q_offset,
                    rope_freqs_ref,
                    use_direct_q_rope,
                );
                let k_rope = qk_norm_rope_bhsd_from_proj_with_route(
                    &k_raw,
                    w.k_norm.as_ref(),
                    kv_heads,
                    head_dim,
                    seq,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    token_offset,
                    rope_freqs_ref,
                    use_direct_k_rope,
                );
                if let Some(started) = qk_norm_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaQkNorm,
                        started,
                        &[&q_rope, &k_rope],
                    );
                }
                (q_rope, k_rope)
            } else {
                let qk_norm_started = profile_forward_layer.then(Instant::now);
                let q = qk_norm_bhsd_from_proj(
                    &q_raw,
                    w.q_norm.as_ref(),
                    cfg.n_heads,
                    head_dim,
                    q_seq,
                    cfg.rms_norm_eps,
                );
                let k = qk_norm_bhsd_from_proj(
                    &k_raw,
                    w.k_norm.as_ref(),
                    kv_heads,
                    head_dim,
                    seq,
                    cfg.rms_norm_eps,
                );
                if let Some(started) = qk_norm_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaQkNorm,
                        started,
                        &[&q, &k],
                    );
                }
                let rope_kv_started = profile_forward_layer.then(Instant::now);
                let (q_rope, k_rope) = if let Some(mrope) = mrope {
                    (
                        crate::qwen3_vl::apply_interleaved_mrope(&q, mrope, rope_dims),
                        crate::qwen3_vl::apply_interleaved_mrope(&k, mrope, rope_dims),
                    )
                } else {
                    (
                        rope_bhsd_batch_offset_safe(
                            &q,
                            rope_dims as i32,
                            rope_base,
                            q_offset as i32,
                            rope_freqs_ref,
                        ),
                        rope_bhsd_batch_offset_safe(
                            &k,
                            rope_dims as i32,
                            rope_base,
                            token_offset as i32,
                            rope_freqs_ref,
                        ),
                    )
                };
                if let Some(started) = rope_kv_started {
                    forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaRopeKv,
                        started,
                        &[&q_rope, &k_rope],
                    );
                }
                (q_rope, k_rope)
            };

            let rope_kv_started = profile_forward_layer.then(Instant::now);
            let retained_window = if seq == 1 || ring_layout.is_some() {
                sliding_window
            } else if sliding_window.is_some() && fastpath::multi_token_window_views_enabled() {
                match shared_mask {
                    Some(Some(mask)) => mask.shape().last().map(|&len| len as usize),
                    _ => sliding_window.map(|window| window + seq - 1),
                }
            } else {
                None
            };
            let attention_kv = if let Some(window) = protected_prefix_window {
                cache
                    .append_with_protected_prefix_window_for_attention(layer_idx, k_rope, v, window)
            } else {
                cache.append_with_retained_window_for_attention(
                    layer_idx,
                    k_rope,
                    v,
                    retained_window,
                )
            };
            match &attention_kv {
                MlxAttentionKv::Dense { k, v } => {
                    gemma4_prefill_maybe_async_first_kv(k, v, &cfg.model_family, seq as i32)
                }
                MlxAttentionKv::Paged(_) => {}
            }
            if let Some(started) = rope_kv_started {
                match &attention_kv {
                    MlxAttentionKv::Dense { k, v } => forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaRopeKv,
                        started,
                        &[&q_rope, k, v],
                    ),
                    MlxAttentionKv::Paged(view) => forward_profile_eval_elapsed(
                        profile_decode_layer,
                        profile_prefill_layer,
                        DecodeProfileStage::PreSdpaRopeKv,
                        started,
                        &[&q_rope, &view.k_slab, &view.v_slab, &view.block_table],
                    ),
                }
            }
            (q_rope, attention_kv, attn_gate_raw)
        };
        if let Some(started) = pre_sdpa_started {
            let mut refs: Vec<&MlxArray> = vec![&q_rope];
            match &attention_kv {
                MlxAttentionKv::Dense { k, v } => {
                    refs.push(k);
                    refs.push(v);
                }
                MlxAttentionKv::Paged(view) => {
                    refs.push(&view.k_slab);
                    refs.push(&view.v_slab);
                    refs.push(&view.block_table);
                }
            }
            if let Some(g) = attn_gate.as_ref() {
                refs.push(g);
            }
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PreSdpa,
                started,
                &refs,
            );
        }
        // 8. SDPA. Last-only generate: KV is already written, so the last
        // query is enough. Full-seq shared masks are invalid at Q=1.
        let last_query_sdpa = fastpath::should_qwen_prefill_last_query_sdpa(
            &cfg.model_family,
            last_position_only_after_attention,
            seq as i32,
        ) || fastpath::should_gemma4_prefill_last_query_p128(
            &cfg.model_family,
            last_position_only_after_attention,
            seq as i32,
        );
        let q_rope = match qwen_prefill_maybe_last_query_q(&q_rope, last_query_sdpa) {
            Some(q) => q,
            None => q_rope,
        };
        // Query length is Q's BHSD dim 2. Last-token Q proj implies this
        // last-query-SDPA flag; packed QKV still keeps full-seq Q and slices
        // here. Never pass the full-seq `seq` to SDPA (`f763ca23…` crash).
        let query_seq = qwen_prefill_query_seq(&q_rope, seq);
        let key_len = attention_kv.key_len();
        // Prefer a hoisted/shared mask only when its last dim matches the
        // post-append key length (ring capacity when rotating). Rebuild
        // locally on mismatch so mask and K cannot disagree.
        let shared_usable = query_seq == seq
            && shared_mask.is_some_and(|m| match m.as_ref() {
                Some(mask) => mask.shape().last().is_some_and(|&k| k as usize == key_len),
                None => true,
            });
        let local_mask: Option<MlxArray> = if shared_usable {
            None
        } else {
            match ring_layout {
                Some(ring) if ring.needs_mask(query_seq) && key_len == ring.capacity => {
                    Some(create_ring_sliding_mask(
                        query_seq,
                        ring.window,
                        ring.capacity,
                        ring.write_start,
                    ))
                }
                _ => attention_mask_array(query_seq, key_len, sliding_window),
            }
        };
        let none_mask: Option<MlxArray> = None;
        let mask_opt: &Option<MlxArray> = if shared_usable {
            shared_mask.unwrap_or(&none_mask)
        } else {
            &local_mask
        };
        let sdpa_started = profile_forward_layer.then(Instant::now);
        let attn_sdpa = match attention_kv {
            MlxAttentionKv::Dense { k, v } => full_precision_attention_with_window(
                &q_rope,
                &k,
                &v,
                cfg.query_scale,
                query_seq,
                mask_opt,
                sliding_window.filter(|_| ring_layout.is_none()),
                ring_layout,
            ),
            MlxAttentionKv::Paged(view) => {
                if query_seq == 1
                    && mask_opt.is_none()
                    && let Some(output) = paged_decode_attention(&q_rope, &view, cfg.query_scale)
                {
                    cache.record_paged_attention_result(true);
                    output
                } else {
                    cache.record_paged_attention_result(false);
                    let (k, v) = view.materialize();
                    full_precision_attention_with_window(
                        &q_rope,
                        &k,
                        &v,
                        cfg.query_scale,
                        query_seq,
                        mask_opt,
                        sliding_window.filter(|_| ring_layout.is_none()),
                        ring_layout,
                    )
                }
            }
        };
        if let Some(started) = sdpa_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::Sdpa,
                started,
                &[&attn_sdpa],
            );
        }
        qwen_prefill_maybe_async_sdpa(&attn_sdpa, &cfg.model_family, query_seq as i32);
        post_attn_started = profile_forward_layer.then(Instant::now);
        let output_proj_started = profile_forward_layer.then(Instant::now);

        // Exact S=2..=4 full-attn: compile flatten + o_proj + residual + FFN
        // after SDPA. Not the linear-attention portable RMS+SiLU gate.
        if !last_position_only_after_attention
            && attn_gate.is_none()
            && query_seq == seq
            && let Some(out) = qwen_compiled_split_verify_fa_o_proj_ffn(
                cfg,
                w,
                hidden,
                &attn_sdpa,
                layer_idx,
                query_seq,
                cfg.n_heads,
                head_dim,
            )
        {
            if let Some(started) = output_proj_started {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::PostAttnOutputProj,
                    started,
                    &[&out],
                );
            }
            if let Some(started) = post_attn_started {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::PostAttn,
                    started,
                    &[&out],
                );
            }
            return out;
        }

        // 10-11. Flatten + output projection (+ optional post-attn RMSNorm).
        // Last-only generate prefill: o_proj is position-wise and KV is already
        // written, so slice SDPA to the last token before flatten/o_proj.
        let last_token_o_proj = fastpath::should_qwen_prefill_last_token_o_proj(
            &cfg.model_family,
            last_position_only_after_attention,
            seq as i32,
        ) || fastpath::should_gemma4_prefill_last_query_p128(
            &cfg.model_family,
            last_position_only_after_attention,
            seq as i32,
        );
        let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, query_seq, cfg.n_heads, head_dim);
        let attn_flat = qwen_prefill_maybe_last_token_flat(&attn_flat, last_token_o_proj);
        // RowExact o_proj under moe_mt (smokef99 exact). Shared o_proj broke
        // formal identity (smokef100).
        let o_policy = if crate::fastpath::moe_mt_bf16_identity_enabled() {
            crate::model::shared::utils::ProjectionBatchPolicy::RowExact
        } else {
            crate::model::shared::utils::ProjectionBatchPolicy::Shared
        };
        let attn_proj = attention_output_projection_with_post_norm_policy(
            &attn_flat,
            attn_gate.as_ref(),
            w.o_proj
                .as_ref()
                .expect("full-attention layer must have o_proj"),
            w.attn_post_norm.as_ref(),
            cfg.rms_norm_eps,
            o_policy,
        );
        if let Some(started) = output_proj_started {
            forward_profile_eval_elapsed(
                profile_decode_layer,
                profile_prefill_layer,
                DecodeProfileStage::PostAttnOutputProj,
                started,
                &[&attn_proj],
            );
        }
        attn_proj
    };
    // Static S1 fairness probe: split each standard Gemma4 text-prefill layer
    // after attention output projection, before the residual/FFN graph is
    // encoded. `sublayer` also retains the caller's normal layer-boundary
    // barrier; the default-off path adds no evaluation.
    if fastpath::pipeline_sublayer_eval_should_fire(seq, &cfg.model_family) {
        mlx_sys::eval(&[&attn_proj]);
    }
    if let Some(started) = attention_started {
        profile_eval_elapsed(
            profile_gemma4_moe_decode,
            Gemma4MoeProfileStage::Attention,
            started,
            &[&attn_proj],
        );
    }

    // Delegate to shared post-attention pipeline.
    layer_shell_post_attention(
        cfg,
        w,
        hidden,
        &attn_proj,
        seq,
        layer_idx,
        per_layer_input,
        last_position_only_after_attention,
        skip_post_attention_ffn,
        profile_forward_layer,
        profile_decode_layer,
        profile_prefill_layer,
        profile_gemma4_moe_decode,
        post_attn_started,
    )
}

// ---------------------------------------------------------------------------
// Batched decode layer forward
// ---------------------------------------------------------------------------

/// Batched-decode analog of [`layer_forward`] for full-attention **dense**
/// families (Qwen3, Llama, Mistral) — milestone 2a of batched MLX decode.
///
/// Runs one decode token for each of B rows (`hidden` is `[B, 1, hidden]`)
/// through the same per-layer graph as `layer_forward`. Projection helpers use
/// the row-exact policy so every row preserves the single-sequence quantized
/// reduction graph; the remaining QK norm, RoPE, attention, activation, and
/// residual operations stay batched. It appends to a [`BatchedKvCache`]
/// (whose `append_decode_layer` returns the current-token-inclusive
/// `[B, H, key_len, D]` view) and passes the batched validity `mask`
/// (`[B, 1, 1, key_len]`) to attention.
///
/// Supports **ragged** rows: `offsets[r]` is row `r`'s current decode position,
/// so a continuously-batched cohort at different sequence positions decodes
/// together. Scope (asserted): normal (non-KV-shared) full/sliding-attention
/// layers, Qwen-compatible MoE, and scalar residuals. Per-layer-input gating
/// remains unsupported; the batched runner routes only eligible requests here.
#[allow(clippy::too_many_arguments)]
pub fn layer_forward_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &mut BatchedKvCache,
    layer_idx: usize,
    offsets: &[usize],
    mask: &Option<MlxArray>,
) -> MlxArray {
    debug_assert_eq!(
        offsets.len(),
        cache.batch(),
        "one RoPE offset per batch row"
    );
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        sliding_window,
        kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);
    assert!(
        kv_source.is_none(),
        "batched decode (2a): KV-shared layers unsupported"
    );
    assert!(
        w.per_layer_gate.is_none(),
        "batched decode: per-layer-input gating unsupported"
    );

    let seq = 1usize;
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let (q_raw, k_raw, v_raw, attn_gate) = qkv_project_batched(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");
    let v = prepare_value_bhsd_from_proj(
        &v_raw,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    let q = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    // Per-row RoPE in ONE dispatch: each batch row is at a different decode
    // position (`offsets[r]`), so this used to slice each row, `rope` it with
    // that row's scalar offset, and concatenate — O(batch) slices + ropes +
    // a concat, per q and per k, per layer. That per-row loop was the
    // seq-independent, batch-linear cost that kept batched decode from
    // amortizing (docs/performance/batched-decode-ceiling.md, Phase 3.3).
    // MLX 0.32 `rope_dynamic` accepts a `[batch]` offset array and applies a
    // per-row position — bit-identical to the loop — so one call replaces it.
    let offsets_i32: Vec<i32> = offsets.iter().map(|&o| o as i32).collect();
    let offset_arr = MlxArray::from_raw_data(
        offsets_i32.as_ptr() as *const u8,
        std::mem::size_of_val(offsets_i32.as_slice()),
        &[offsets_i32.len() as i32],
        MlxDtype::Int32,
    );
    let rope_row = |x: &MlxArray| -> MlxArray {
        rope_dynamic(
            x,
            rope_dims as i32,
            false,
            rope_base,
            1.0,
            &offset_arr,
            rope_freqs_ref,
            None,
        )
    };
    let q_rope = rope_row(&q);
    let k_rope = rope_row(&k);
    // Env-gated per-stage barrier timing (off by default → the `if` is false
    // and nothing below changes the production graph).
    let prof = batched_profile::enabled();
    let mut mark_at = Instant::now();
    let mut mark = |stage: usize, outs: &[&MlxArray]| {
        if prof {
            mlx_sys::eval(outs);
            batched_profile::record(stage, mark_at.elapsed().as_micros());
            mark_at = Instant::now();
        }
    };
    mark(0, &[&q_rope, &k_rope, &v]);

    let (cached_k, cached_v) = cache.append_decode_layer(layer_idx, &k_rope, &v);
    let sliding_mask = sliding_window.map(|window| {
        let valid_lengths = offsets
            .iter()
            .map(|offset| offset.saturating_add(1))
            .collect::<Vec<_>>();
        batched_decode_validity_mask_with_window(
            &valid_lengths,
            cached_k.shape()[2] as usize,
            Some(window),
        )
    });
    let layer_mask = if sliding_mask.is_some() {
        &sliding_mask
    } else {
        mask
    };
    let attn_sdpa = full_precision_attention(
        &q_rope,
        &cached_k,
        &cached_v,
        cfg.query_scale,
        seq,
        layer_mask,
    );
    let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
    mark(1, &[&attn_flat]);

    let attn_proj = attention_output_projection_batched(
        &attn_flat,
        attn_gate.as_ref(),
        w.o_proj
            .as_ref()
            .expect("full-attention layer must have o_proj"),
    );
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };
    let hidden = add(hidden, &attn_proj, None);
    mark(2, &[&hidden]);

    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_batched(cfg, w, &normed2, layer_idx);
    let out = if let Some(scalar) = &w.layer_scalar {
        add_then_multiply_scalar(&hidden, &ffn_out, scalar)
    } else {
        add(&hidden, &ffn_out, None)
    };
    mark(3, &[&out]);
    out
}

/// Batched decode FFN — dense SwiGLU or sparse MoE, matching the single-row
/// dispatch (`layer_forward` / `qwen3_linear::layer_forward`) for a
/// `[B, 1, hidden]` cohort.
///
/// Dense path routes through [`ffn_swiglu_batched`] (row-exact / shared
/// projection policy, Phase 3.5). MoE path uses the qwen3 router + experts,
/// which are batch-general for `[B, 1, hidden]`: the router falls back off its
/// batch=1 fused kernel, `gather_qmm` broadcasts the batch dim, and the one
/// batch=1-shaped activation fast path is guarded off (see
/// `moe_experts_forward_impl`). Expert selection therefore stays per-row; only
/// the shared quantized reductions carry the usual batched-kernel bf16 drift,
/// which decode certification bounds to greedy-token parity.
/// Padded batched-prefill layer: the `[B, L, hidden]` sibling of
/// [`layer_forward_batched`]'s `[B, 1, hidden]` decode step. Every row is
/// cold (RoPE position 0), so one scalar-offset rope covers the batch, and
/// attention runs over the in-chunk K/V only — no cache is involved — under
/// the caller's causal+padding mask
/// ([`crate::attention_mask::batched_prefill_causal_mask`]). Returns the
/// layer output plus this layer's K/V (`[B, kv_heads, L, head_dim]`) so the
/// caller can extract each row's `[0..len)` region into its per-request
/// cache.
pub fn layer_forward_batched_prefill(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    layer_idx: usize,
    padded_len: usize,
    mask: &Option<MlxArray>,
) -> (MlxArray, MlxArray, MlxArray) {
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        sliding_window,
        kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);
    // `supports_batched_prefill` refuses these shapes; the asserts keep the
    // invariant local if a new caller skips the gate.
    assert!(
        sliding_window.is_none(),
        "batched prefill: sliding-window layers unsupported"
    );
    assert!(
        kv_source.is_none(),
        "batched prefill: KV-shared layers unsupported"
    );
    assert!(
        w.per_layer_gate.is_none(),
        "batched prefill: per-layer-input gating unsupported"
    );
    assert!(
        w.router_proj.is_none(),
        "batched prefill: MoE layers unsupported"
    );

    let seq = padded_len;
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let (q_raw, k_raw, v_raw, attn_gate) = qkv_project_batched(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");
    let v = prepare_value_bhsd_from_proj(
        &v_raw,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    let q = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let rope_zero = |x: &MlxArray| -> MlxArray {
        mlx_sys::rope(
            x,
            rope_dims as i32,
            false,
            rope_base,
            1.0,
            0,
            rope_freqs_ref,
            None,
        )
    };
    let q_rope = rope_zero(&q);
    let k_rope = rope_zero(&k);

    let attn_sdpa = full_precision_attention(&q_rope, &k_rope, &v, cfg.query_scale, seq, mask);
    let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
    let attn_proj = attention_output_projection_batched(
        &attn_flat,
        attn_gate.as_ref(),
        w.o_proj
            .as_ref()
            .expect("full-attention layer must have o_proj"),
    );
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };
    let hidden = add(hidden, &attn_proj, None);

    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_batched(cfg, w, &normed2, layer_idx);
    let out = if let Some(scalar) = &w.layer_scalar {
        add_then_multiply_scalar(&hidden, &ffn_out, scalar)
    } else {
        add(&hidden, &ffn_out, None)
    };
    (out, k_rope, v)
}

fn ffn_batched(
    cfg: &ModelConfig,
    w: &LayerWeights,
    normed2: &MlxArray,
    layer_idx: usize,
) -> MlxArray {
    if w.router_proj.is_none() {
        return ffn_swiglu_batched(cfg, w, normed2, w.ffn_post_norm.as_ref(), layer_idx);
    }
    assert!(
        !cfg.gemma4_moe_router,
        "batched decode: gemma4 MoE router unsupported"
    );
    // WS-T1 Decision A: per-row MoE for bit-exact B>1 certification.
    // Shared gather_qmm amortization is intentionally uncertified (see
    // docs/performance/batched-hybrid-moe-linear-decode.md).
    if crate::batched_decode_policy::row_exact_moe_enabled(
        &cfg.model_family,
        cfg.moe_expert_count > 0,
    ) {
        return ffn_batched_moe_row_exact(cfg, w, normed2);
    }
    let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
        moe_router_glm(cfg, w, normed2)
    } else if cfg.moe_sigmoid_routing {
        moe_router_deepseek_v3(cfg, w, normed2)
    } else {
        moe_router_qwen3(cfg, w, normed2)
    };
    let out = if w.shared_gate_proj.is_some() {
        let shared = shared_expert_forward(cfg, w, normed2);
        moe_experts_forward_with_shared(cfg, w, normed2, &top_k_indices, &top_k_weights, &shared)
    } else {
        moe_experts_forward(cfg, w, normed2, &top_k_indices, &top_k_weights)
    };
    rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
}

/// Per-row MoE expert path (Decision A RowExact): each batch row is isolated so
/// gather_qmm reduction order cannot introduce cross-row drift.
fn ffn_batched_moe_row_exact(cfg: &ModelConfig, w: &LayerWeights, normed2: &MlxArray) -> MlxArray {
    use mlx_sys::{concatenate, slice};
    let shape = normed2.shape();
    let batch = *shape.first().unwrap_or(&1) as usize;
    if batch <= 1 {
        let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
            moe_router_glm(cfg, w, normed2)
        } else if cfg.moe_sigmoid_routing {
            moe_router_deepseek_v3(cfg, w, normed2)
        } else {
            moe_router_qwen3(cfg, w, normed2)
        };
        let out = if w.shared_gate_proj.is_some() {
            let shared = shared_expert_forward(cfg, w, normed2);
            moe_experts_forward_with_shared(
                cfg,
                w,
                normed2,
                &top_k_indices,
                &top_k_weights,
                &shared,
            )
        } else {
            moe_experts_forward(cfg, w, normed2, &top_k_indices, &top_k_weights)
        };
        return rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps);
    }
    let hidden = cfg.hidden_size as i32;
    let mut rows = Vec::with_capacity(batch);
    for r in 0..batch {
        let row = slice(
            normed2,
            &[r as i32, 0, 0],
            &[(r + 1) as i32, 1, hidden],
            &[1, 1, 1],
            None,
        );
        let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
            moe_router_glm(cfg, w, &row)
        } else if cfg.moe_sigmoid_routing {
            moe_router_deepseek_v3(cfg, w, &row)
        } else {
            moe_router_qwen3(cfg, w, &row)
        };
        let out = if w.shared_gate_proj.is_some() {
            let shared = shared_expert_forward(cfg, w, &row);
            moe_experts_forward_with_shared(cfg, w, &row, &top_k_indices, &top_k_weights, &shared)
        } else {
            moe_experts_forward(cfg, w, &row, &top_k_indices, &top_k_weights)
        };
        let out = rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps);
        rows.push(out);
    }
    let refs: Vec<&MlxArray> = rows.iter().collect();
    concatenate(&refs, 0, None)
}

/// Batched decode forward for a linear-attention (gated-delta) layer — the
/// [`layer_forward_batched`] sibling for Qwen3-Next's linear layers. Mirrors
/// `qwen3_linear::layer_forward` for a `[B, 1, hidden]` cohort: attn-norm →
/// batched gated-delta ([`linear_attention_forward_batched`], reading/writing
/// per-row state in `lin_state`) → residual → ffn-norm → batched FFN (dense or
/// MoE) → residual. No KV cache (the recurrent state carries the history), so
/// no offsets/mask. Decode-only.
pub fn layer_forward_batched_linear(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    lin_state: &mut BatchedLinearState,
    linear_state_idx: usize,
    model_layer_idx: usize,
) -> MlxArray {
    assert!(
        w.per_layer_gate.is_none(),
        "batched decode: per-layer-input gating unsupported"
    );
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);
    let attn_proj = linear_attention_forward_batched(cfg, w, &normed, lin_state, linear_state_idx);
    let hidden = add(hidden, &attn_proj, None);

    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = ffn_batched(cfg, w, &normed2, model_layer_idx);
    if let Some(scalar) = &w.layer_scalar {
        add_then_multiply_scalar(&hidden, &ffn_out, scalar)
    } else {
        add(&hidden, &ffn_out, None)
    }
}

// ---------------------------------------------------------------------------
// Bidirectional layer forward (DiffusionGemma)
// ---------------------------------------------------------------------------

/// Bidirectional layer forward for DiffusionGemma denoiser.
///
/// Same QKV projections, QK-norm, MoE FFN, per-layer gating as [`layer_forward`],
/// but:
/// - **Bidirectional** (non-causal) attention over the canvas.
/// - **Read-only** KV cache: attends to cached prompt KV without writing.
/// - Canvas K/V are computed fresh from `hidden` each denoiser step.
///
/// Post-attention pipeline (residual, FFN, gating) is shared with
/// [`layer_forward`] via [`layer_shell_post_attention`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_forward_bidirectional(
    cfg: &ModelConfig,
    w: &LayerWeights,
    hidden: &MlxArray,
    cache: &MlxKVCache,
    layer_idx: usize,
    token_offset: usize,
    per_layer_input: Option<&MlxArray>,
    kv_buffer: Option<&mut KVConcatBuffer>,
) -> MlxArray {
    let (
        head_dim,
        rope_theta,
        rope_dims,
        layer_rope_freqs,
        sliding_window,
        _kv_source,
        v_norm_no_scale,
    ) = layer_params(cfg, layer_idx);

    let seq = hidden.shape()[1] as usize;

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    // 2-6. QKV projections.
    let (q_raw, k_raw, v_raw, attn_gate_raw) = qkv_project(cfg, w, &normed, head_dim);
    let kv_heads = (k_raw.shape()[2] as usize)
        .checked_div(head_dim)
        .expect("k projection output must divide by head_dim");
    let v = prepare_value_bhsd_from_proj(
        &v_raw,
        v_norm_no_scale,
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    // QK norm (no RoPE yet — apply RoPE after).
    let q = qk_norm_bhsd_from_proj(
        &q_raw,
        w.q_norm.as_ref(),
        cfg.n_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );
    let k = qk_norm_bhsd_from_proj(
        &k_raw,
        w.k_norm.as_ref(),
        kv_heads,
        head_dim,
        seq,
        cfg.rms_norm_eps,
    );

    // Apply RoPE to Q and K using canvas positions.
    let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
    let (rope_base, rope_freqs_ref) = rope_freqs
        .map(|f| (None, Some(f)))
        .unwrap_or((Some(rope_theta), None));
    let q_rope = rope(
        &q,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs_ref,
        None,
    );
    let k_rope = rope(
        &k,
        rope_dims as i32,
        false,
        rope_base,
        1.0,
        token_offset as i32,
        rope_freqs_ref,
        None,
    );

    // Read cached prompt K/V (no mutation).
    let (cached_k_full, cached_v_full) = cache
        .peek_layer_full_kv(layer_idx)
        .expect("bidirectional layer requires cached prompt KV from prefill");

    // Symmetric SWA for bidirectional attention.
    let (cached_k, cached_v, swa_sliced) = if let Some(window) = sliding_window {
        let cached_seq = cached_k_full.shape()[2] as usize;
        let prompt_start = token_offset.saturating_sub(window);
        if prompt_start > 0 && prompt_start < cached_seq {
            let b = cached_k_full.shape()[0];
            let h = cached_k_full.shape()[1];
            let d = cached_k_full.shape()[3];
            let sliced_k = slice(
                &cached_k_full,
                &[0, 0, prompt_start as i32, 0],
                &[b, h, cached_seq as i32, d],
                &[1, 1, 1, 1],
                None,
            );
            let sliced_v = slice(
                &cached_v_full,
                &[0, 0, prompt_start as i32, 0],
                &[b, h, cached_seq as i32, d],
                &[1, 1, 1, 1],
                None,
            );
            (sliced_k, sliced_v, true)
        } else {
            (cached_k_full.clone(), cached_v_full.clone(), false)
        }
    } else {
        (cached_k_full.clone(), cached_v_full.clone(), false)
    };

    // Bidirectional attention: canvas Q attends to cached prompt KV + canvas KV.
    let kv_buf_for_attn = if swa_sliced { None } else { kv_buffer };
    let attn_sdpa = bidirectional_attention(
        &q_rope,
        &cached_k,
        &cached_v,
        &k_rope,
        &v,
        cfg.query_scale,
        sliding_window,
        kv_buf_for_attn,
    );

    let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
    let attn_proj = attention_output_projection_with_post_norm(
        &attn_flat,
        attn_gate_raw.as_ref(),
        w.o_proj
            .as_ref()
            .expect("bidirectional attention layer must have o_proj"),
        w.attn_post_norm.as_ref(),
        cfg.rms_norm_eps,
    );

    // Delegate to shared post-attention pipeline (residual, FFN, gating).
    layer_shell_post_attention(
        cfg,
        w,
        hidden,
        &attn_proj,
        seq,
        layer_idx,
        per_layer_input,
        false, // last_position_only_after_attention
        false, // skip_post_attention_ffn
        false, // profile_forward_layer
        false, // profile_decode_layer
        false, // profile_prefill_layer
        false, // profile_gemma4_moe_decode
        None,  // post_attn_started
    )
}

#[cfg(test)]
mod tests {
    use super::last_layer_residual_and_ffn_norm;
    use mlx_sys::{MlxArray, MlxDtype, add, eval, rms_norm, slice};

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    #[test]
    fn last_layer_residual_and_ffn_norm_skip_prefix_matches_add_then_slice() {
        let seq = 4;
        let hidden_size = 2;
        let hidden_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let attn_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25).collect();
        let norm_data = [1.0_f32, 1.0];
        let hidden = array_f32(&hidden_data, &[1, seq, hidden_size]);
        let attn = array_f32(&attn_data, &[1, seq, hidden_size]);
        let ffn_norm = array_f32(&norm_data, &[hidden_size]);

        let (skip_res, skip_norm, pli) = last_layer_residual_and_ffn_norm(
            &hidden,
            &attn,
            &ffn_norm,
            None,
            seq as usize,
            hidden_size as usize,
            1e-5,
            true,
        );
        assert!(pli.is_none());

        let full_add = add(&hidden, &attn, None);
        let last = seq - 1;
        let sliced = slice(
            &full_add,
            &[0, last, 0],
            &[1, last + 1, hidden_size],
            &[1, 1, 1],
            None,
        );
        let sliced_norm = rms_norm(&sliced, Some(&ffn_norm), 1e-5, None);
        eval(&[&skip_res, &skip_norm, &sliced, &sliced_norm]);
        assert_eq!(skip_res.shape(), vec![1, 1, hidden_size]);
        assert_eq!(skip_norm.shape(), vec![1, 1, hidden_size]);
        let a = skip_res.data_f32();
        let b = sliced.data_f32();
        assert_eq!(a, b, "slice-then-add last row must match add-then-slice");
        let na = skip_norm.data_f32();
        let nb = sliced_norm.data_f32();
        assert_eq!(na.len(), nb.len());
        for (x, y) in na.iter().zip(nb.iter()) {
            assert!(
                (x - y).abs() < 1.0e-5,
                "slice-then-add_rms last row must match add-then-slice-then-rms: {x} vs {y}"
            );
        }
        assert!(
            crate::fastpath::should_gemma4_prefill_skip_unused_last_residual_for(
                true, "gemma4", true, 128
            ),
            "shipped skip-unused-last-residual must accept contract p128 last layer"
        );
    }
}
