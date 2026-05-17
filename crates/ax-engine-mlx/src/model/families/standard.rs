use mlx_sys::{MlxArray, add, gelu_approx, multiply, reshape, rms_norm, rope, transpose};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::profile::{
    DecodeProfileStage, Gemma4MoeProfileStage, decode_profile_enabled,
    forward_profile_eval_elapsed, gemma4_moe_profile_enabled, prefill_profile_enabled,
    profile_eval_elapsed, record_gemma4_moe_decode_layer,
};
use super::super::shared::{
    attention_mask_array, attention_output_projection, ffn_swiglu, full_precision_attention,
    moe_experts_forward, moe_router_gemma4, moe_router_glm, moe_router_qwen3, prepare_value_bhsd,
    qk_norm_bshd, qkv_project, qw, rms_norm_opt, shape_element_count, shared_expert_forward,
    turboquant_decode_attention_experimental,
};
use super::super::turboquant_context::{
    TurboQuantModelDecodeCandidate, TurboQuantModelDecodeCandidateStatus,
    TurboQuantModelDecodeContext,
};
use crate::kv_cache::{MlxKVCache, MlxKvCompressionDecodeOutcome};
use crate::weights::LayerWeights;

/// Minimum top-k selection count above which the sort path is taken in Gemma4 MoE.
const SWITCH_GLU_SORT_THRESHOLD: usize = 64;

/// Full layer forward for standard GQA attention families (Gemma4, Gemma3, Qwen3).
///
/// Handles per-head QK norm, KV-sharing (Gemma4), sliding window attention,
/// Gemma4 dual-path MoE, Qwen3 MoE, dense FFN, per-layer input gating (Gemma4),
/// and the TurboQuant experimental decode path.
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
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let (head_dim, rope_theta, rope_dims, sliding_window, kv_source, v_norm_no_scale) =
        layer_params(cfg, layer_idx);

    // 1. Attention norm.
    let normed = rms_norm(hidden, Some(&w.attn_norm), cfg.rms_norm_eps, None);

    let seq = hidden.shape()[1] as usize;
    let profile_gemma4_moe_decode =
        cfg.gemma4_moe_router && seq == 1 && gemma4_moe_profile_enabled();
    let attention_started = profile_gemma4_moe_decode.then(Instant::now);
    let profile_decode_layer = seq == 1 && decode_profile_enabled();
    let profile_prefill_layer = seq > 1 && prefill_profile_enabled();
    let profile_forward_layer = profile_decode_layer || profile_prefill_layer;
    let post_attn_started;
    let attn_proj = {
        let pre_sdpa_started = profile_forward_layer.then(Instant::now);
        // 2-7. QKV projections + RoPE. KV-shared layers skip K/V and borrow from source.
        let (q_rope, cached_k, cached_v, attn_gate) = if let Some(src_layer) = kv_source {
            // KV-shared layer (Gemma4 layers 24-41): compute Q only.
            let q_raw = qw(
                &normed,
                w.q_proj.as_ref().expect("KV-shared layer must have q_proj"),
            );
            let q = reshape(
                &q_raw,
                &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
                None,
            );
            let qk_norm_started = profile_forward_layer.then(Instant::now);
            let q = qk_norm_bshd(
                q,
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
            let q = transpose(&q, &[0, 2, 1, 3], None);
            let (rope_base, rope_freqs_ref) = cfg
                .rope_freqs
                .as_ref()
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
            let (ck, cv) = cache.peek_source_kv(src_layer, seq);
            if let Some(started) = rope_kv_started {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::PreSdpaRopeKv,
                    started,
                    &[&q_rope, &ck, &cv],
                );
            }
            (q_rope, ck, cv, None)
        } else {
            // Normal layer: compute Q, K, V from own projections.
            let qkv_proj_started = profile_forward_layer.then(Instant::now);
            let (q_raw, k_raw, v_raw, attn_gate_raw) = qkv_project(cfg, w, &normed, head_dim);

            let q = reshape(
                &q_raw,
                &[1, seq as i32, cfg.n_heads as i32, head_dim as i32],
                None,
            );
            let kv_heads = (k_raw.shape()[2] as usize)
                .checked_div(head_dim)
                .expect("k projection output must divide by head_dim");
            let k = reshape(
                &k_raw,
                &[1, seq as i32, kv_heads as i32, head_dim as i32],
                None,
            );
            let v = reshape(
                &v_raw,
                &[1, seq as i32, kv_heads as i32, head_dim as i32],
                None,
            );
            if let Some(started) = qkv_proj_started {
                let mut refs: Vec<&MlxArray> = vec![&q, &k, &v];
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

            let qk_norm_started = profile_forward_layer.then(Instant::now);
            let q = qk_norm_bshd(
                q,
                w.q_norm.as_ref(),
                cfg.n_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );
            let k = qk_norm_bshd(
                k,
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
            let q = transpose(&q, &[0, 2, 1, 3], None);
            let k = transpose(&k, &[0, 2, 1, 3], None);
            let v = prepare_value_bhsd(
                v,
                v_norm_no_scale,
                kv_heads,
                head_dim,
                seq,
                cfg.rms_norm_eps,
            );

            let (rope_base, rope_freqs_ref) = cfg
                .rope_freqs
                .as_ref()
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

            let (ck, cv) = if seq == 1 {
                cache.append_with_retained_window(layer_idx, k_rope, v, sliding_window)
            } else {
                cache.append(layer_idx, k_rope, v)
            };
            if let Some(started) = rope_kv_started {
                forward_profile_eval_elapsed(
                    profile_decode_layer,
                    profile_prefill_layer,
                    DecodeProfileStage::PreSdpaRopeKv,
                    started,
                    &[&q_rope, &ck, &cv],
                );
            }
            (q_rope, ck, cv, attn_gate_raw)
        };
        if let Some(started) = pre_sdpa_started {
            let mut refs: Vec<&MlxArray> = vec![&q_rope, &cached_k, &cached_v];
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
        let turboquant_candidate = turboquant_context
            .map(|context| {
                context.decode_candidate(
                    cfg,
                    cache,
                    layer_idx,
                    seq,
                    head_dim,
                    cached_k.shape()[1] as usize,
                    sliding_window,
                    kv_source,
                    false,
                )
            })
            .unwrap_or_else(TurboQuantModelDecodeCandidate::disabled);
        cache.record_turboquant_decode_candidate(turboquant_candidate.telemetry_status());

        // 8. SDPA (GQA: MLX broadcasts KV heads internally). For decode, Gemma
        // sliding-window layers present only the retained window to SDPA, matching
        // mlx_lm/mlx-swift-lm rotating-cache behavior.
        let key_len = cached_k.shape()[2] as usize;
        let local_mask: Option<MlxArray>;
        let mask_opt: &Option<MlxArray> = if let Some(m) = shared_mask {
            m
        } else {
            local_mask = attention_mask_array(seq, key_len, sliding_window);
            &local_mask
        };
        let sdpa_started = profile_forward_layer.then(Instant::now);
        let attn_sdpa =
            if turboquant_candidate.status == TurboQuantModelDecodeCandidateStatus::Ready {
                let turboquant_out = turboquant_decode_attention_experimental(
                    cache,
                    layer_idx,
                    &q_rope,
                    seq,
                    cfg.n_heads,
                    head_dim,
                    cfg.query_scale,
                );
                let outcome = turboquant_out
                    .as_ref()
                    .map(|output| output.outcome)
                    .unwrap_or(MlxKvCompressionDecodeOutcome::Fallback);
                cache.record_turboquant_fused_decode_attempt(outcome);
                if let Some(output) = turboquant_out.as_ref() {
                    cache.record_turboquant_fused_decode_timing(output.timing);
                }
                turboquant_out
                    .map(|output| output.attention)
                    .unwrap_or_else(|| {
                        full_precision_attention(
                            &q_rope,
                            &cached_k,
                            &cached_v,
                            cfg.query_scale,
                            seq,
                            mask_opt,
                        )
                    })
            } else {
                full_precision_attention(
                    &q_rope,
                    &cached_k,
                    &cached_v,
                    cfg.query_scale,
                    seq,
                    mask_opt,
                )
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

        // 10. Transpose back: [1, n_heads, seq, head_dim] → [1, seq, n_heads, head_dim].
        let attn_out = transpose(&attn_sdpa, &[0, 2, 1, 3], None);

        // 11. Reshape to [1, seq, hidden].
        let attn_flat = reshape(
            &attn_out,
            &[1, seq as i32, (cfg.n_heads * head_dim) as i32],
            None,
        );

        // 12-13. Optional Qwen3.5 output gate, then output projection.
        let attn_proj = attention_output_projection(
            &attn_flat,
            attn_gate.as_ref(),
            w.o_proj
                .as_ref()
                .expect("full-attention layer must have o_proj"),
        );

        // 14. Optional post-attention layernorm (Gemma4): applied BEFORE residual add.
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

    // 15. Residual.
    let residual_norm_started = profile_forward_layer.then(Instant::now);
    let hidden = add(hidden, &attn_proj, None);

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
            // Gemma4 dual-path: dense sub-block + expert sub-block.
            let dense_started = profile_gemma4_moe_decode.then(Instant::now);
            let h1 = ffn_swiglu(cfg, w, &normed2);
            let h1 = rms_norm_opt(&h1, w.ffn_post_norm1.as_ref(), cfg.rms_norm_eps);
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
            // Gemma4 intentionally routes from raw hidden through its own combined-scale
            // RMSNorm, while experts consume the separately normalized h2 input.
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
            let h2 = moe_experts_forward(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
            if let Some(started) = expert_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Expert,
                    started,
                    &[&h2],
                );
            }
            let post_started = profile_gemma4_moe_decode.then(Instant::now);
            let h2 = rms_norm_opt(&h2, w.ffn_post_norm2.as_ref(), cfg.rms_norm_eps);
            let combined = add(&h1, &h2, None);
            let out = rms_norm_opt(&combined, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps);
            if let Some(started) = post_started {
                profile_eval_elapsed(
                    profile_gemma4_moe_decode,
                    Gemma4MoeProfileStage::Post,
                    started,
                    &[&out],
                );
            }
            out
        } else {
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, w, &normed2)
            } else {
                // Qwen3 MoE: router (proj → softmax → top-k) + expert forward.
                moe_router_qwen3(cfg, w, &normed2)
            };
            let mut out = moe_experts_forward(cfg, w, &normed2, &top_k_indices, &top_k_weights);
            if w.shared_gate_proj.is_some() {
                out = add(&out, &shared_expert_forward(cfg, w, &normed2), None);
            }
            rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
        }
    } else {
        // Dense path (Qwen3, Gemma4 non-MoE layers).
        let out = ffn_swiglu(cfg, w, &normed2);
        rms_norm_opt(&out, w.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
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

    // 18. Residual.
    let residual_gate_started = profile_forward_layer.then(Instant::now);
    let mut out = add(&hidden, &ffn_out, None);

    // 19. Per-layer input gating (Gemma4 2B/4B): gate(h) * per_layer_embed → proj → norm + h.
    if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(pli)) = (
        w.per_layer_gate.as_ref(),
        w.per_layer_proj_w.as_ref(),
        w.per_layer_post_norm.as_ref(),
        per_layer_input,
    ) {
        let gate = gelu_approx(&qw(&out, gate_w), None);
        let gated = multiply(&gate, pli, None);
        let projected = qw(&gated, proj_w);
        let normed = rms_norm(&projected, Some(post_norm), cfg.rms_norm_eps, None);
        out = add(&out, &normed, None);
    }

    // 20. Optional layer scalar (Gemma4): h = h * layer_scalar.
    let out = if let Some(scalar) = &w.layer_scalar {
        multiply(&out, scalar, None)
    } else {
        out
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
