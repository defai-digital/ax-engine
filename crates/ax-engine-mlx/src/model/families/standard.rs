use mlx_sys::{MlxArray, add, rms_norm, rope, slice};
use std::time::Instant;

use super::super::ModelConfig;
use super::super::config::layer_params;
use super::super::profile::{
    DecodeProfileStage, Gemma4MoeProfileStage, decode_profile_enabled,
    forward_profile_eval_elapsed, gemma4_moe_profile_enabled, prefill_profile_enabled,
    profile_eval_elapsed, record_gemma4_moe_decode_layer,
};
use super::super::shared::{
    add_then_multiply_scalar, attention_mask_array, attention_output_projection,
    direct_qk_norm_rope_route_enabled_for_family, ffn_swiglu, flatten_attention_output_bhsd,
    full_precision_attention, moe_experts_forward, moe_experts_forward_gemma4,
    moe_experts_forward_with_shared, moe_router_gemma4, moe_router_glm, moe_router_qwen3,
    per_layer_input_gate_project, prepare_value_bhsd_from_proj, qk_norm_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj_with_route, qkv_project, qw, rms_norm_opt, shape_element_count,
    shared_expert_forward, turboquant_decode_attention_experimental,
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
///
/// `last_position_only_after_attention`: when `true` and `seq > 1`, the layer
/// slices its attention-residual stream to the last sequence position before
/// running pre-FFN norm + FFN + post-FFN residual. The KV cache writes have
/// already happened inside attention, so the slice is safe; the FFN, gating,
/// and layer-scalar steps then operate on a `[1, 1, hidden]` tensor instead
/// of `[1, seq, hidden]`. This matches the lazy-eval prune that mlx-lm gets
/// for free on the last layer when the model output is discarded.
///
/// Callers must only set this flag for the **last transformer layer** in a
/// prefill pass, and only when the downstream consumer needs just the
/// last-position output (i.e. argmax/sample for the first decode token).
/// Setting it on a non-terminal layer breaks correctness: the next layer
/// would receive a 1-position hidden but the cache still expects matching
/// sequence positions.
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
    last_position_only_after_attention: bool,
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
            (q_rope, ck, cv, None)
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

        // 10-11. Convert [B, n_heads, seq, head_dim] to [B, seq, hidden].
        // Single-token decode skips the transpose because seq=1 is already
        // head-major in the flattened projection order.
        let attn_flat = flatten_attention_output_bhsd(&attn_sdpa, seq, cfg.n_heads, head_dim);

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
    let last_only_active = last_position_only_after_attention && seq > 1;

    let hidden = add(hidden, &attn_proj, None);

    // 15a. Optional last-position-only slice for the terminal prefill layer.
    // KV cache writes happened inside attention; downstream consumers (final
    // norm + lm_head + argmax) only need the last-position output. Slicing
    // here lets the FFN run on `[1, 1, hidden]` instead of `[1, seq, hidden]`,
    // which is the dominant cost on long prompts. See the function-level
    // doc for the correctness contract.
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
            // Gemma4 dual-path: dense sub-block + expert sub-block.
            let dense_started = profile_gemma4_moe_decode.then(Instant::now);
            let h1 = ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm1.as_ref());
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
            let h2 = moe_experts_forward_gemma4(cfg, w, &h2_normed, &top_k_indices, &top_k_weights);
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
            let router_started = profile_forward_layer.then(Instant::now);
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, w, &normed2)
            } else {
                // Qwen3 MoE: router (proj → softmax → top-k) + expert forward.
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
            // Compute shared expert before the expert forward so the weighted-sum
            // kernel can optionally fuse the shared-expert add (Phase 1A).
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
            let out = if let Some(shared) = &shared_out {
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
        ffn_swiglu(cfg, w, &normed2, w.ffn_post_norm.as_ref())
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

    // 19. Per-layer input gating (Gemma4 2B/4B): gate(h) * per_layer_embed → proj → norm + h.
    //
    // Match mlx_lm's Gemma4DecoderLayer per-layer input gate. The shared
    // helper preserves the same `gelu_approx * per_layer_input` math through
    // AX's direct MLX activation shim.
    let out = if let (Some(gate_w), Some(proj_w), Some(post_norm), Some(pli)) = (
        w.per_layer_gate.as_ref(),
        w.per_layer_proj_w.as_ref(),
        w.per_layer_post_norm.as_ref(),
        per_layer_input,
    ) {
        let residual = add(&hidden, &ffn_out, None);
        let projected = per_layer_input_gate_project(&qw(&residual, gate_w), pli, proj_w);
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
