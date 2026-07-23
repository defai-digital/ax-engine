use mlx_sys::{MlxArray, MlxDtype, MlxVectorArray, add, rms_norm, rope, rope_dynamic, slice};
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
    KVConcatBuffer, add_then_multiply_scalar, attention_mask_array, attention_output_projection,
    attention_output_projection_batched, bidirectional_attention,
    direct_qk_norm_rope_route_enabled_for_family, ffn_swiglu, ffn_swiglu_batched,
    flatten_attention_output_bhsd, flatten_compiled_moe_inputs, flatten_gemma4_dual_path_inputs,
    full_precision_attention, linear_attention_forward_batched, moe_experts_forward,
    moe_experts_forward_gemma4, moe_experts_forward_with_cloned_weights,
    moe_experts_forward_with_shared, moe_router_deepseek_v3, moe_router_gemma4, moe_router_glm,
    moe_router_qwen3, per_layer_input_gate_project, prepare_value_bhsd_from_proj,
    qk_norm_bhsd_from_proj, qk_norm_rope_bhsd_from_proj_with_route, qkv_project,
    qkv_project_batched, qw, rms_norm_opt, shape_element_count, shared_expert_forward,
};
use crate::attention_mask::{batched_decode_validity_mask_with_window, create_ring_sliding_mask};
use crate::batched_kv_cache::BatchedKvCache;
use crate::batched_linear_state::BatchedLinearState;
use crate::fastpath;
use crate::kv_cache::{MlxAttentionKv, MlxKVCache};
use crate::paged_attention::paged_decode_attention;
use crate::per_layer_compile::{apply_layer_gemma4_dual_path_decode, apply_layer_moe_decode};
use crate::weights::LayerWeights;

/// Minimum top-k selection count above which the sort path is taken in Gemma4 MoE.
const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

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
    // 15. Residual.
    let residual_norm_started = profile_forward_layer.then(Instant::now);
    let last_only_active = last_position_only_after_attention && seq > 1;

    let hidden = add(hidden, attn_proj, None);

    // Cache-only terminal layer: attention already wrote full-seq KV; FFN
    // residual is discarded before the completing decode_step. Return here.
    if skip_post_attention_ffn {
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

    // 15a. Optional last-position-only slice for the terminal prefill layer.
    let (hidden, per_layer_input_owned) = if last_only_active {
        let last = (seq - 1) as i32;
        let hs = cfg.hidden_size as i32;
        let sliced_hidden = slice(&hidden, &[0, last, 0], &[1, last + 1, hs], &[1, 1, 1], None);
        let sliced_pli = per_layer_input.map(|pli| {
            let dims = pli.shape();
            let pli_last_dim = *dims.last().unwrap_or(&hs);
            slice(
                pli,
                &[0, last, 0],
                &[1, last + 1, pli_last_dim],
                &[1, 1, 1],
                None,
            )
        });
        (sliced_hidden, sliced_pli)
    } else {
        (hidden, None)
    };
    let per_layer_input: Option<&MlxArray> = if last_only_active {
        per_layer_input_owned.as_ref()
    } else {
        per_layer_input
    };

    // 16. Pre-FFN norm.
    let normed2 = rms_norm(&hidden, Some(&w.ffn_norm), cfg.rms_norm_eps, None);
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
    let ffn_started = profile_forward_layer.then(Instant::now);
    let ffn_out = if w.router_proj.is_some() {
        if cfg.gemma4_moe_router {
            // Try compiled dual-path decode closure.
            let compiled_result = if seq == 1 && fastpath::moe_layer_compile_enabled() {
                flatten_gemma4_dual_path_inputs(&normed2, &hidden, w).and_then(
                    |(inputs, mut schema)| {
                        let cfg_clone = cfg.clone();
                        schema.moe_expert_count = cfg.moe_expert_count;
                        schema.moe_experts_per_token = cfg.moe_experts_per_token;
                        let input_refs: Vec<&MlxArray> = inputs.iter().collect();
                        apply_layer_gemma4_dual_path_decode(
                            cfg.compile_cache_identity,
                            layer_idx,
                            &input_refs,
                            move |inputs: &MlxVectorArray| vec![schema.forward(inputs, &cfg_clone)],
                        )
                    },
                )
            } else {
                None
            };
            if let Some(result) = compiled_result.and_then(|r| r.into_iter().next()) {
                result
            } else {
                // Gemma4 dual-path: dense sub-block + expert sub-block.
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
                let h2 =
                    moe_experts_forward_gemma4(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
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
            // Try compiled MoE decode closure.
            let compiled_result = if seq == 1 && fastpath::moe_layer_compile_enabled() {
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
                            &cfg_clone, &x, &indices, &weights, gate_up, gate, up, down, shared,
                            None,
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
        ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref(), layer_idx)
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
    let out = if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(pli)) = (
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
        let use_direct_k_rope = direct_qk_norm_rope_route_enabled_for_family(
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
            rope(
                &k,
                rope_dims as i32,
                false,
                rope_base,
                1.0,
                token_offset as i32,
                rope_freqs_ref,
                None,
            )
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
        let _ = cache.append_with_retained_window(layer_idx, k_rope, v, retained_window);
        return hidden.clone();
    }

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    let ring_layout = cache.sliding_ring_layout(sliding_window, seq);
    let profile_gemma4_moe_decode =
        cfg.gemma4_moe_router && seq == 1 && gemma4_moe_profile_enabled();
    let attention_started = profile_gemma4_moe_decode.then(Instant::now);
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;

    // 2-7. QKV projections + RoPE + KV cache append + SDPA.
    let post_attn_started;
    let attn_proj = {
        let pre_sdpa_started = profile_forward_layer.then(Instant::now);
        let (q_rope, attention_kv, attn_gate) = if let Some(src_layer) = kv_source {
            // KV-shared layer (Gemma4 layers 24-41): compute Q only.
            let q_raw = qw(
                &normed,
                w.q_proj.as_ref().expect("KV-shared layer must have q_proj"),
            );
            let rope_freqs = layer_rope_freqs.or(cfg.rope_freqs.as_ref());
            let (rope_base, rope_freqs_ref) = rope_freqs
                .map(|f| (None, Some(f)))
                .unwrap_or((Some(rope_theta), None));
            let direct_q_rope = direct_qk_norm_rope_route_enabled_for_family(
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
            let (q_raw, k_raw, v_raw, attn_gate_raw) = qkv_project(cfg, w, &normed, head_dim);
            let kv_heads = (k_raw.shape()[2] as usize)
                .checked_div(head_dim)
                .expect("k projection output must divide by head_dim");
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
            let use_direct_q_rope = direct_qk_norm_rope_route_enabled_for_family(
                cfg.model_family.as_str(),
                w.q_norm.as_ref(),
            );
            let use_direct_k_rope = direct_qk_norm_rope_route_enabled_for_family(
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
                    seq,
                    cfg.rms_norm_eps,
                    rope_dims,
                    rope_base,
                    token_offset,
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
            let attention_kv = cache.append_with_retained_window_for_attention(
                layer_idx,
                k_rope,
                v,
                retained_window,
            );
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
        // 8. SDPA.
        let key_len = attention_kv.key_len();
        let local_mask: Option<MlxArray>;
        let mask_opt: &Option<MlxArray> = if let Some(m) = shared_mask {
            m
        } else {
            local_mask = match ring_layout {
                Some(ring) if ring.needs_mask(seq) => Some(create_ring_sliding_mask(
                    seq,
                    ring.window,
                    ring.capacity,
                    ring.write_start,
                )),
                Some(_) => None,
                None => attention_mask_array(seq, key_len, sliding_window),
            };
            &local_mask
        };
        let sdpa_started = profile_forward_layer.then(Instant::now);
        let attn_sdpa = match attention_kv {
            MlxAttentionKv::Dense { k, v } => {
                full_precision_attention(&q_rope, &k, &v, cfg.query_scale, seq, mask_opt)
            }
            MlxAttentionKv::Paged(view) => {
                if seq == 1
                    && mask_opt.is_none()
                    && let Some(output) = paged_decode_attention(&q_rope, &view, cfg.query_scale)
                {
                    cache.record_paged_attention_result(true);
                    output
                } else {
                    cache.record_paged_attention_result(false);
                    let (k, v) = view.materialize();
                    full_precision_attention(&q_rope, &k, &v, cfg.query_scale, seq, mask_opt)
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
        post_attn_started = profile_forward_layer.then(Instant::now);
        let output_proj_started = profile_forward_layer.then(Instant::now);

        // 10-11. Flatten + output projection.
        let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);
        let attn_proj = attention_output_projection(
            &attn_flat,
            attn_gate.as_ref(),
            w.o_proj
                .as_ref()
                .expect("full-attention layer must have o_proj"),
        );
        // 14. Optional post-attention layernorm.
        let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
            rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
        } else {
            attn_proj
        };
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
    let attn_proj = attention_output_projection(
        &attn_flat,
        attn_gate_raw.as_ref(),
        w.o_proj
            .as_ref()
            .expect("bidirectional attention layer must have o_proj"),
    );
    let attn_proj = if let Some(post_norm) = &w.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };

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
