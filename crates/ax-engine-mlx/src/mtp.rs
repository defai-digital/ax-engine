use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, ScaledDotProductAttentionMask, add, argmax,
    astype, concatenate, eval, multiply, random_categorical, reshape, rms_norm,
    scaled_dot_product_attention_with_mask, sigmoid, slice, softmax, take,
};

use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::model::shared::{
    apply_final_logit_softcap, ffn_swiglu, flatten_attention_output_bhsd,
    glm_mla_attention_forward, moe_experts_forward, moe_router_deepseek_v3, moe_router_glm,
    moe_router_qwen3, prepare_value_bhsd_from_proj, qk_norm_rope_bhsd_from_proj, qw, rms_norm_opt,
    shared_expert_forward,
};
use crate::model::{ModelConfig, embed_tokens_arr};
use crate::sampling::{TokenDistribution, Xorshift64};
use crate::weights::{GlmMtpWeights, ModelWeights, MtpWeights};
use std::sync::OnceLock;

/// Draft sampling mode for MTP speculative decoding.
///
/// `Greedy` uses argmax selection (current default, single GPU eval).
/// `Stochastic` applies top-p/top-k filtering + temperature sampling per depth
/// (requires per-depth GPU sync, but recovers acceptance when MTP head argmax
/// disagrees with the target model).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MtpDraftMode {
    #[default]
    Greedy,
    Stochastic,
}

/// Returns the current MTP draft mode, cached via `OnceLock`.
///
/// Priority: `AX_MLX_MTP_DRAFT_MODE` env → default `Greedy`.
pub fn mtp_draft_mode_from_env() -> MtpDraftMode {
    static CACHED: OnceLock<MtpDraftMode> = OnceLock::new();
    *CACHED.get_or_init(|| {
        match std::env::var("AX_MLX_MTP_DRAFT_MODE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .replace('_', "-")
            .as_str()
        {
            "stochastic" => MtpDraftMode::Stochastic,
            _ => MtpDraftMode::Greedy,
        }
    })
}

/// Minimum MTP-head confidence (probability assigned to the drafted token)
/// required to keep a speculative draft token.
///
/// Without the gate, speculative drafts are produced up to `max_depth` deep
/// regardless of how confident the head is, but deep tokens on hard,
/// fresh-generation inputs are frequently rejected by the target model, which
/// drags the measured accept rate down (e.g. Qwen3.6 27B `python_modules_long`
/// pure-MTP accept fell to ~82%). With the gate, the draft is truncated at the
/// first depth whose head confidence (its true, temperature-1.0 probability)
/// falls below the threshold, so only high-confidence tokens are proposed for
/// verification. This trades a little speculative depth on hard inputs for a
/// much higher accept rate, and is correctness-preserving: truncating a draft
/// never changes the committed output, only how many tokens are verified ahead.
///
/// Read from `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`; valid range `[0.0, 1.0)`.
/// Defaults to [`DEFAULT_MTP_DRAFT_MIN_CONFIDENCE`] (gate on); set the variable
/// to `0` to disable the gate and restore the prior full-depth draft behavior.
pub fn mtp_draft_min_confidence_from_env() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| match std::env::var("AX_MLX_MTP_DRAFT_MIN_CONFIDENCE") {
        Ok(raw) => raw
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|value| value.is_finite() && *value >= 0.0 && *value < 1.0)
            .unwrap_or(DEFAULT_MTP_DRAFT_MIN_CONFIDENCE),
        Err(_) => DEFAULT_MTP_DRAFT_MIN_CONFIDENCE,
    })
}

/// `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE` parsed to `Some(value)` only when set and
/// valid; `None` when unset, so speculation-profile resolution can supply a
/// preset instead.
fn mtp_draft_min_confidence_explicit() -> Option<f32> {
    static CACHED: OnceLock<Option<f32>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("AX_MLX_MTP_DRAFT_MIN_CONFIDENCE")
            .ok()
            .and_then(|raw| {
                raw.trim()
                    .parse::<f32>()
                    .ok()
                    .filter(|value| value.is_finite() && *value >= 0.0 && *value < 1.0)
            })
    })
}

/// Resolve the Qwen fused MTP gate with speculation-profile precedence (ADR-022):
/// explicit env > profile preset > built-in default.
///
/// The Qwen MTP path accepts via rejection sampling (it carries draft log-probs),
/// so it preserves the sampling distribution exactly — it has no greedy-bias
/// concern. Accordingly `coding`/`agentic`/`auto` defer to the validated `0.90`
/// default; only `chatbot` raises the gate, and that is a throughput choice (cut
/// rejection waste at high temperature), not a correctness one. `temperature`
/// drives `auto`.
pub fn resolve_mtp_draft_min_confidence(
    profile: crate::speculation_profile::SpeculationProfile,
    temperature: Option<f32>,
) -> f32 {
    crate::speculation_profile::resolve_gate(
        mtp_draft_min_confidence_explicit(),
        profile.qwen_gate(temperature),
        DEFAULT_MTP_DRAFT_MIN_CONFIDENCE,
    )
    .0
}

/// Default MTP draft confidence gate, tuned for **throughput** (not for the
/// maximum accept rate).
///
/// The prior value (0.98) was calibrated to hold the pure-MTP *accept rate* at
/// 99%+, but that over-truncates: it only proposes near-certain tokens, so the
/// draft is shorter than it needs to be and decode leaves speed on the table. A
/// Qwen3.6 27B (MTP) depth-throughput sweep on the fair-MTP suites
/// (`docs/MTP-DRAFT-GATE-THROUGHPUT.md`) shows 0.90 is the throughput optimum:
/// it proposes slightly longer drafts that are still almost always accepted,
/// while the rare extra rejection costs only one cheap recompute forward (the
/// `fwds/step` count stays near 1.03 even on the hardest suite). Measured
/// wall-clock gains over the 0.98 default: flappy +5%, python_modules_long
/// +4-14%, long_code +13%, with tokens-per-forward up 7-16%.
///
/// Lowering the gate is **correctness-preserving** — truncating fewer draft
/// tokens never changes the committed output (greedy) or its distribution
/// (sampled, via rejection sampling), only how far ahead each step verifies. It
/// does lower the reported *accept rate* (more drafts proposed); that is a speed
/// knob, not a quality change. Override with `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE`;
/// set 0.98 to restore the accept-rate-maximizing behavior, or 0 to disable.
pub const DEFAULT_MTP_DRAFT_MIN_CONFIDENCE: f32 = 0.90;

/// Truncate a draft to the longest leading run whose per-depth head confidence
/// stays at or above `min_confidence` (probability, not log-prob).
///
/// Gating starts at depth 0, so a low-confidence first token yields an empty
/// draft and that step falls back to an ordinary verified decode (it does not
/// enter the speculative accept/draft accounting). This keeps the accept rate
/// bounded below by the gate: only tokens the head is at least `min_confidence`
/// sure of are ever proposed for verification.
fn apply_draft_confidence_gate(
    result: (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]),
    min_confidence: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let (mut tokens, mut log_probs, mut distributions, _added, accept3) = result;
    if min_confidence <= 0.0 || tokens.is_empty() {
        let added = tokens.len();
        return (tokens, log_probs, distributions, added, accept3);
    }
    let ln_threshold = min_confidence.ln();
    // First depth whose head confidence drops below the threshold; keep [0, keep).
    let mut keep = tokens.len();
    for (depth, &log_prob) in log_probs.iter().enumerate() {
        if !log_prob.is_finite() || log_prob < ln_threshold {
            keep = depth;
            break;
        }
    }
    if keep >= tokens.len() {
        let added = tokens.len();
        return (tokens, log_probs, distributions, added, accept3);
    }
    tokens.truncate(keep);
    log_probs.truncate(keep);
    if distributions.len() > keep {
        distributions.truncate(keep);
    }
    (tokens, log_probs, distributions, keep, accept3)
}

/// Lazy argmax over a `[vocab_size]` f32 logit array.
///
/// Returns a lazy `[1]` uint32 array — caller must `eval` it to materialise.
/// This avoids a per-depth GPU sync barrier in the draft loop, allowing
/// all depth levels to build their compute graphs before a single batch eval.
fn lazy_argmax_logits(logits: &MlxArray) -> MlxArray {
    let vocab = logits.shape()[0];
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    argmax(&logits_2d, None)
}

/// Compute `log(softmax(logits / temperature)[token])` on GPU using a lazy
/// `[1]` uint32 token array (e.g. from `lazy_argmax_logits`).
///
/// The `take` index is the lazy argmax result, so the entire softmax → gather
/// → log chain stays lazy and can be fused into a single GPU dispatch.
fn gpu_draft_log_prob_lazy(
    logits: &MlxArray,
    lazy_token: &MlxArray,
    temperature: f32,
    vocab: i32,
) -> MlxArray {
    use mlx_sys::log as mlx_log;
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    let inv_temp = MlxArray::from_f32(1.0 / temperature);
    let scaled = multiply(&logits_2d, &inv_temp, None);
    let probs = softmax(&scaled, -1, None);

    let prob = take(&probs, lazy_token, 1, None);

    let log_prob = mlx_log(&prob, None);
    let floor = MlxArray::from_f32(-30.0f32);
    mlx_sys::maximum(&log_prob, &floor, None)
}

/// GPU-side stochastic sampling: `random_categorical(logits / T)`.
///
/// Returns a lazy `[1]` uint32 array — caller must `eval` to materialise.
/// Uses MLX's internal RNG (not the per-request `Xorshift64`), so results
/// are not reproducible across runs.  Output quality is preserved via
/// rejection sampling against the target model.
fn lazy_random_sample(logits: &MlxArray, temperature: f32, vocab: i32) -> MlxArray {
    let logits_2d = reshape(logits, &[1_i32, vocab], None);
    let inv_temp = MlxArray::from_f32(1.0 / temperature);
    let scaled = multiply(&logits_2d, &inv_temp, None);
    random_categorical(&scaled, None)
}

/// Run one recurrent MTP head forward pass for a single decode step.
///
/// Returns new hidden state `[1, 1, hidden_size]`.  Caller applies
/// `rms_norm(h, mtp_norm) @ lm_head` to get draft logits.
///
/// * `head`           — shared MTP weights (reused across all depth levels).
/// * `main_hidden`    — post-norm hidden from the main model (hidden_variant="post_norm")
///   or output from a preceding MTP head call, shape `[1, 1, hidden_size]`.
/// * `prev_token_arr` — token ID as a GPU uint32 array, shape `[1]`.  May be a lazy
///   argmax result; no CPU sync is required before calling.
/// * `weights`        — main model weights (for the shared token embedding).
/// * `cache`          — shared 1-layer KV cache for this head (grows by 1 per call).
///
/// RoPE offset is `cache.seq_len + cache.rope_offset` (or an explicit
/// `rope_offset_override` when provided).  This matches the mlx-lm
/// `cache.offset` convention while supporting capped warmup where physical
/// KV entries start at buffer position 0 but represent tokens at higher
/// prompt positions.  Callers must NOT pass absolute sequence positions
/// unless using `rope_offset_override`.
pub fn mtp_head_forward(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
) -> MlxArray {
    // Use the explicit RoPE offset when provided (e.g. during capped warmup
    // where KV entries start at buffer position 0 but represent prompt tokens
    // at higher positions).  Otherwise use the MTP KV-cache seq_len + rope_offset
    // as the RoPE offset (matches mlx-lm cache.offset, with rope_offset accounting
    // for physical-vs-logical position differences after capped warmup).
    let token_offset = rope_offset_override.unwrap_or(cache.seq_len + cache.rope_offset);
    // 1. Embed prev_token → [1, 1, hidden_size] in bf16.
    let embed = embed_tokens_arr(prev_token_arr, &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);

    // 2. Combined input: fc(cat([enorm(embed), hnorm(hidden)])).
    //    concat_order = "embedding_hidden" → [enorm, hnorm] along last dim.
    let enormed = rms_norm(
        &embed,
        Some(&head.pre_fc_norm_embedding),
        cfg.rms_norm_eps,
        None,
    );
    let hnormed = rms_norm(
        main_hidden,
        Some(&head.pre_fc_norm_hidden),
        cfg.rms_norm_eps,
        None,
    );
    let combined = concatenate(&[&enormed, &hnormed], -1, None);
    let mut h = qw(&combined, &head.fc);

    // 3. Attention sub-layer (Qwen3NextAttention).
    //
    // q_proj output = [1, 1, n_heads * head_dim * 2] with per-head interleaving:
    //   [h0_query(head_dim), h0_gate(head_dim), h1_query(head_dim), h1_gate(head_dim), ...]
    // This matches mlx-lm: `mx.split(q_proj_out.reshape(B,L,n_heads,-1), 2, axis=-1)`.
    // We must reshape to [1, 1, n_heads, 2*head_dim] and then slice the last dim —
    // NOT a simple first-half / second-half slice (which mixes heads).
    // Output = o_proj(sdpa_out * sigmoid(gate)), then residual.
    let fused_ffn_norm;
    {
        let normed = rms_norm(&h, Some(&head.attn_norm), cfg.rms_norm_eps, None);

        // Reshape q_proj output to expose per-head query/gate layout.
        let n_h = head.n_heads as i32;
        let hd = head.head_dim as i32;
        let q_half = n_h * hd;
        let qg_raw = qw(&normed, &head.q_proj); // [1, 1, n_heads * head_dim * 2]
        let qg_heads = reshape(&qg_raw, &[1_i32, 1, n_h, 2 * hd], None); // [1, 1, n_heads, 2*hd]
        let q_raw = reshape(
            &slice(
                &qg_heads,
                &[0, 0, 0, 0],
                &[1, 1, n_h, hd],
                &[1, 1, 1, 1],
                None,
            ),
            &[1_i32, 1, q_half],
            None,
        );
        let gate = reshape(
            &slice(
                &qg_heads,
                &[0, 0, 0, hd],
                &[1, 1, n_h, 2 * hd],
                &[1, 1, 1, 1],
                None,
            ),
            &[1_i32, 1, q_half],
            None,
        );

        let k_raw = qw(&normed, &head.k_proj);
        let v_raw = qw(&normed, &head.v_proj);

        let v = prepare_value_bhsd_from_proj(
            &v_raw,
            false,
            head.n_kv_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
        );

        let (rope_base, rope_freqs_ref) = if let Some(freqs) = cfg.rope_freqs.as_ref() {
            (None, Some(freqs))
        } else {
            (Some(cfg.rope_theta), None)
        };

        let q_rope = qk_norm_rope_bhsd_from_proj(
            &q_raw,
            head.q_norm.as_ref(),
            head.n_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
            cfg.rope_dims,
            rope_base,
            token_offset,
            rope_freqs_ref,
        );
        let k_rope = qk_norm_rope_bhsd_from_proj(
            &k_raw,
            head.k_norm.as_ref(),
            head.n_kv_heads,
            head.head_dim,
            1,
            cfg.rms_norm_eps,
            cfg.rope_dims,
            rope_base,
            token_offset,
            rope_freqs_ref,
        );

        let (cached_k, cached_v) = cache.append(0, k_rope, v);
        cache.seq_len += 1;

        let query_scale = 1.0 / (head.head_dim as f32).sqrt();
        let attn_out = scaled_dot_product_attention_with_mask(
            &q_rope,
            &cached_k,
            &cached_v,
            query_scale,
            ScaledDotProductAttentionMask::None,
            None,
        );

        // Flatten [1, n_heads, 1, head_dim] → [1, 1, n_heads * head_dim].
        let attn_flat = flatten_attention_output_bhsd(&attn_out, 1, head.n_heads, head.head_dim);

        // Apply sigmoid gating: o_proj(attn_flat * sigmoid(gate)).
        let gated = multiply(&attn_flat, &sigmoid(&gate, None), None);
        let attn_proj = qw(&gated, &head.o_proj);
        // Fuse add(h, attn_proj) + rms_norm(h, ffn_norm) into a single C++ call.
        let (h_new, fnormed) =
            mlx_sys::add_rms_norm_pair(&h, &attn_proj, &head.ffn_norm, cfg.rms_norm_eps, None);
        h = h_new;
        fused_ffn_norm = fnormed;
    }

    // 4. FFN sub-layer (SwiGLU).
    {
        let normed = fused_ffn_norm;
        let ffn_out = if head.ffn_layer.router_proj.is_some() {
            let (top_k_indices, top_k_weights) = if cfg.glm_router.is_some() {
                moe_router_glm(cfg, &head.ffn_layer, &normed)
            } else if cfg.moe_sigmoid_routing {
                moe_router_deepseek_v3(cfg, &head.ffn_layer, &normed)
            } else {
                moe_router_qwen3(cfg, &head.ffn_layer, &normed)
            };
            let mut out = moe_experts_forward(
                cfg,
                &head.ffn_layer,
                &normed,
                &top_k_indices,
                &top_k_weights,
            );
            if head.ffn_layer.shared_gate_proj.is_some() {
                out = add(
                    &out,
                    &shared_expert_forward(cfg, &head.ffn_layer, &normed),
                    None,
                );
            }
            out
        } else {
            ffn_swiglu(cfg, &head.ffn_layer, &normed, None)
        };
        h = add(&h, &ffn_out, None);
    }

    h
}

/// Apply `rms_norm(hidden, mtp_norm) @ main_model.lm_head` to produce draft logits.
///
/// Returns f32 logits `[vocab_size]` ready for argmax / sampling.
pub fn mtp_hidden_to_logits(
    hidden: &MlxArray,
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let normed = mtp_hidden_post_norm(hidden, head, cfg);
    mtp_post_norm_to_logits(&normed, weights, cfg)
}

fn mtp_hidden_post_norm(hidden: &MlxArray, head: &MtpWeights, cfg: &ModelConfig) -> MlxArray {
    rms_norm(hidden, Some(&head.mtp_norm), cfg.rms_norm_eps, None)
}

fn mtp_post_norm_to_logits(
    post_norm_hidden: &MlxArray,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    use mlx_sys::reshape as mlx_reshape;
    let logits = qw(post_norm_hidden, &weights.lm_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    let logits_f32 = apply_final_logit_softcap(cfg, &logits_f32);
    // [1, 1, vocab] → [vocab]
    mlx_reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Prototype primitive: run one MTP head step and return both the post-norm
/// hidden (to chain into the next depth) and the full draft logits `[vocab]`.
///
/// `mtp_draft_tokens` only ever follows the argmax chain, so it cannot expose
/// the per-depth logits a tree drafter needs to branch on (top-k alternatives).
/// This helper drives a single recurrent step explicitly so a caller can pick
/// any token(s) from `logits` and feed a chosen token back as `prev_token` for
/// the next depth, using `post_norm_hidden` as that step's `main_hidden`.
///
/// Appends one entry to `cache` (the head's 1-layer recurrent KV). Clone the
/// cache before stepping a sibling branch so each tree path keeps its own KV.
/// Returns `None` when the model has no MTP head.
pub fn mtp_head_step(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    main_hidden: &MlxArray,
    prev_token: u32,
    cache: &mut MlxKVCache,
) -> Option<(MlxArray, MlxArray)> {
    let head = weights.mtp.as_ref()?;
    let tok = [prev_token];
    let prev_token_arr =
        MlxArray::from_raw_data(tok.as_ptr() as *const u8, 4, &[1_i32], MlxDtype::Uint32);
    let new_hidden = mtp_head_forward(
        head,
        main_hidden,
        &prev_token_arr,
        weights,
        cache,
        cfg,
        None,
    );
    let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
    let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);
    Some((post_norm_hidden, logits))
}

// ---------------------------------------------------------------------------
// Compiled MTP draft head
// ---------------------------------------------------------------------------

/// Build a compiled closure that runs the full multi-depth Qwen MTP draft
/// chain in a single `mlx_compile`-fused dispatch.
///
/// The closure body traces D iterations of `mtp_head_forward` +
/// `mtp_hidden_post_norm` + `mtp_post_norm_to_logits`, chaining hidden state
/// and token (via lazy argmax or `random_categorical`) across depths.  The
/// compiled graph replays the full chain in one dispatch, reducing ~25 × D
/// MLX C-API calls to a single compiled-graph apply.
///
/// `temperature`: when > 0, token chaining uses `random_categorical` (GPU
/// sampling); when ≤ 0, uses lazy argmax (greedy / sampled paths).
///
/// Returns `None` when the kill switch `AX_MTP_COMPILED_HEAD=0` is set or
/// compilation fails.
fn build_compiled_mtp_draft(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    cache: &mut MlxKVCache,
    max_depth: usize,
    temperature: f32,
) -> Option<MlxClosure> {
    if !fastpath::mtp_compiled_head_enabled() {
        return None;
    }
    let cfg_addr = cfg as *const ModelConfig as usize;
    let weights_addr = weights as *const ModelWeights as usize;
    let head_addr = head as *const MtpWeights as usize;
    let cache_addr = cache as *mut MlxKVCache as usize;
    let vocab = cfg.vocab_size as i32;

    let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
        let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
        let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
        let head_ref = unsafe { &*(head_addr as *const MtpWeights) };
        let cache_ref = unsafe { &mut *(cache_addr as *mut MlxKVCache) };

        let mut prev_hidden = inputs.get(0);
        let mut prev_token_arr = inputs.get(1);
        let base_offset = cache_ref.seq_len + cache_ref.rope_offset;

        let mut outputs: Vec<MlxArray> = Vec::with_capacity(max_depth * 2);
        for d in 0..max_depth {
            let new_hidden = mtp_head_forward(
                head_ref,
                &prev_hidden,
                &prev_token_arr,
                weights_ref,
                cache_ref,
                cfg_ref,
                Some(base_offset + d),
            );
            let post_norm = mtp_hidden_post_norm(&new_hidden, head_ref, cfg_ref);
            let logits = mtp_post_norm_to_logits(&post_norm, weights_ref, cfg_ref);

            let tok = if temperature > 0.0 {
                let logits_2d = reshape(&logits, &[1_i32, vocab], None);
                let inv_temp = MlxArray::from_f32(1.0 / temperature);
                let scaled = multiply(&logits_2d, &inv_temp, None);
                random_categorical(&scaled, None)
            } else {
                lazy_argmax_logits(&logits)
            };

            outputs.push(post_norm.clone());
            outputs.push(logits);
            prev_hidden = post_norm;
            prev_token_arr = tok;
        }
        outputs
    });
    closure.compile(false).ok()
}

/// Build a compiled closure for the full multi-depth GLM MTP draft chain.
///
/// Same pattern as [`build_compiled_mtp_draft`] but uses
/// `glm_mtp_head_forward` + `glm_mtp_hidden_to_logits`.
fn build_compiled_glm_mtp_draft(
    head: &GlmMtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    cache: &mut MlxKVCache,
    max_depth: usize,
    temperature: f32,
) -> Option<MlxClosure> {
    if !fastpath::mtp_compiled_head_enabled() {
        return None;
    }
    let cfg_addr = cfg as *const ModelConfig as usize;
    let weights_addr = weights as *const ModelWeights as usize;
    let head_addr = head as *const GlmMtpWeights as usize;
    let cache_addr = cache as *mut MlxKVCache as usize;
    let vocab = cfg.vocab_size as i32;

    let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
        let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
        let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
        let head_ref = unsafe { &*(head_addr as *const GlmMtpWeights) };
        let cache_ref = unsafe { &mut *(cache_addr as *mut MlxKVCache) };

        let mut prev_hidden = inputs.get(0);
        let mut prev_token_arr = inputs.get(1);
        let base_offset = cache_ref.seq_len + cache_ref.rope_offset;

        let mut outputs: Vec<MlxArray> = Vec::with_capacity(max_depth * 2);
        for d in 0..max_depth {
            let new_hidden = glm_mtp_head_forward(
                head_ref,
                &prev_hidden,
                &prev_token_arr,
                weights_ref,
                cache_ref,
                cfg_ref,
                Some(base_offset + d),
            );
            let logits = glm_mtp_hidden_to_logits(&new_hidden, head_ref, cfg_ref);

            let tok = if temperature > 0.0 {
                let logits_2d = reshape(&logits, &[1_i32, vocab], None);
                let inv_temp = MlxArray::from_f32(1.0 / temperature);
                let scaled = multiply(&logits_2d, &inv_temp, None);
                random_categorical(&scaled, None)
            } else {
                lazy_argmax_logits(&logits)
            };

            outputs.push(new_hidden.clone());
            outputs.push(logits);
            prev_hidden = new_hidden;
            prev_token_arr = tok;
        }
        outputs
    });
    closure.compile(false).ok()
}

/// Draft up to `head.max_depth` tokens by applying the MTP head recurrently.
///
/// Returns `(draft_tokens, draft_log_probs, draft_distributions, added, top2_margins)`.
/// Draft log-probs are full-vocab softmax probabilities for rejection-sampling acceptance.
/// `top2_margins` is always `[0.0; 3]` (retained for API compatibility).
///
/// Gracefully handles `weights.mtp = None` by returning empty.
///
/// ## Performance design
///
/// **Greedy mode (temperature == 0):** Chains lazy `argmax` across all depth
/// levels without per-depth GPU sync barriers, then materialises all tokens in
/// a single `eval`.  This eliminates 2–3 synchronous GPU round-trips per draft
/// step compared to the previous depth-by-depth `eval` + `data_u32` pattern,
/// allowing the GPU to execute the full multi-depth graph as one fused batch.
///
/// **Temperature mode (temperature > 0):** CPU-side sampling requires the
/// logits at each depth level (to apply top-k/top-p with a per-request RNG),
/// so a per-depth GPU sync is unavoidable.  However, the draft log-probability
/// is now computed on GPU using the same softmax path as the target-model
/// probability (`softmax(logits / T)` → `take`), eliminating the numerical
/// mismatch between CPU-side `full_vocab_token_logprob` (f32 reduction) and
/// GPU-side target probs that caused ~20 pp lower acceptance rates on complex
/// code prompts.
pub fn mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    mtp_draft_tokens_gated(
        weights,
        cfg,
        first_hidden,
        first_token,
        cache,
        max_depth_cap,
        rng,
        // Speculation-profile resolution (ADR-022): explicit env > profile > 0.90
        // default. Temperature is unavailable at this wrapper, so `auto` keeps the
        // validated default; explicit profiles still apply.
        resolve_mtp_draft_min_confidence(
            crate::speculation_profile::speculation_profile_from_env(),
            None,
        ),
    )
}

/// Like [`mtp_draft_tokens`] but with an explicit draft-confidence gate instead
/// of the process-global `AX_MLX_MTP_DRAFT_MIN_CONFIDENCE` env value.
///
/// This lets a caller vary the gate per request/step (e.g. an adaptive
/// throughput controller that loosens the gate on hard content and tightens it
/// on easy content — see `docs/MTP-DRAFT-GATE-THROUGHPUT.md`). The gate is always
/// correctness-preserving: it only changes how many speculative tokens are
/// proposed for verification, never the committed output.
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_tokens_gated(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    rng: &mut Xorshift64,
    min_confidence: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    }

    let vocab = cfg.vocab_size as i32;
    let draft_mode = mtp_draft_mode_from_env();

    // The confidence gate keys off the head's true (temperature 1.0) probability
    // of each drafted token. The greedy draft path computes exactly that, while
    // the sampled path's temperature-scaled log-probs saturate near 1.0 and lose
    // gating resolution. So whenever the gate is active we draft greedily (argmax)
    // unless stochastic drafting was explicitly requested via AX_MLX_MTP_DRAFT_MODE.
    let gate_forces_greedy = min_confidence > 0.0 && draft_mode != MtpDraftMode::Stochastic;
    let result = if gate_forces_greedy {
        mtp_draft_tokens_greedy(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
        )
    } else {
        match draft_mode {
            MtpDraftMode::Stochastic => mtp_draft_tokens_stochastic(
                head,
                weights,
                cfg,
                first_hidden,
                first_token,
                cache,
                max_depth,
                vocab,
                rng,
            ),
            MtpDraftMode::Greedy => {
                let use_temperature = head.draft_sampling.temperature > 0.0;
                if use_temperature {
                    mtp_draft_tokens_sampled(
                        head,
                        weights,
                        cfg,
                        first_hidden,
                        first_token,
                        cache,
                        max_depth,
                        vocab,
                        rng,
                    )
                } else {
                    mtp_draft_tokens_greedy(
                        head,
                        weights,
                        cfg,
                        first_hidden,
                        first_token,
                        cache,
                        max_depth,
                        vocab,
                    )
                }
            }
        }
    };

    // `result.3` is the number of MTP KV entries the draft path physically
    // appended to `cache` (one per head forward). The confidence gate can drop
    // the low-confidence tail of the draft; those tail forwards already wrote
    // their entries, so dropping them without trimming leaves stale rows that
    // inflate `cache.seq_len` above the returned `added`. That breaks the
    // invariant the decode loop relies on (MTP `cache.seq_len` == the running
    // `mtp_decode_count`): a fully-accepted gated draft (post-verify
    // `rejected_count == 0`, so the rollback trims nothing) would leave those
    // rows for the next step's MTP head to attend over at an inflated RoPE
    // offset, silently degrading draft acceptance. Trim the gated-out tail so
    // the cache always matches `added`. Output is unaffected either way (every
    // draft is verified against the target model); this preserves the
    // speculative acceptance rate.
    let appended = result.3;
    let gated = apply_draft_confidence_gate(result, min_confidence);
    let dropped = appended.saturating_sub(gated.3);
    if dropped > 0 {
        let target = cache.seq_len.saturating_sub(dropped);
        let _ = cache.trim_to(target);
    }
    gated
}

/// Advance the MTP recurrent state through caller-supplied prefix tokens, then
/// draft up to `max_tail_depth` extra MTP tokens after that prefix.
///
/// This supports hybrid n-gram + MTP speculation: an n-gram provider can fill
/// the high-confidence prefix, while the MTP head fills the remaining draft
/// slots. `added` in the return value includes both forced-prefix MTP forwards
/// and sampled/greedy tail forwards so cache rollback can trim by rejected draft
/// count.
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_tokens_after_forced_prefix(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    forced_prefix: &[u32],
    cache: &mut MlxKVCache,
    max_tail_depth: usize,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    if forced_prefix.is_empty() {
        return mtp_draft_tokens(
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            Some(max_tail_depth),
            rng,
        );
    }

    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for &forced_token in forced_prefix {
        let new_hidden = mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        prev_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let tok_data = [forced_token];
        prev_token_arr = MlxArray::from_raw_data(
            tok_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
    }

    if max_tail_depth == 0 {
        let kv_refs = cache.collect_eval_refs();
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&prev_hidden);
        targets.extend(kv_refs);
        eval(&targets);
        return (vec![], vec![], vec![], forced_prefix.len(), [0.0f32; 3]);
    }

    let last_forced = forced_prefix.last().copied().unwrap_or(first_token);
    let (draft, log_probs, distributions, tail_added, top2_margins) = mtp_draft_tokens(
        weights,
        cfg,
        &prev_hidden,
        last_forced,
        cache,
        Some(max_tail_depth),
        rng,
    );

    (
        draft,
        log_probs,
        distributions,
        forced_prefix.len().saturating_add(tail_added),
        top2_margins,
    )
}

/// Greedy draft path: build full lazy graph across all depths, eval once.
///
/// Eliminates per-depth GPU sync barriers by passing lazy `argmax` results
/// directly as the next depth's token input.  `mtp_head_forward` already
/// supports lazy `prev_token_arr` (embedding lookup works on unevaluated
/// arrays), so the entire multi-depth computation builds a single fused graph
/// that MLX can execute in one GPU dispatch batch.
#[allow(clippy::too_many_arguments)]
fn mtp_draft_tokens_greedy(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // ── Compiled path ───────────────────────────────────────────────
    if let Some(compiled) = build_compiled_mtp_draft(head, weights, cfg, cache, max_depth, 0.0) {
        let seq_len_before = cache.seq_len;
        let first_token_data = [first_token];
        let first_token_arr = MlxArray::from_raw_data(
            first_token_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        match compiled.try_apply(&[first_hidden, &first_token_arr]) {
            Ok(closure_outputs) => {
                // closure_outputs = [post_norm_0, logits_0, post_norm_1, logits_1, ...]
                let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
                let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
                for d in 0..max_depth {
                    let logits = &closure_outputs[d * 2 + 1];
                    let lazy_tok = lazy_argmax_logits(logits);
                    let lazy_lp = gpu_draft_log_prob_lazy(logits, &lazy_tok, 1.0, vocab);
                    lazy_tokens.push(lazy_tok);
                    lazy_log_probs.push(lazy_lp);
                }
                let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
                for t in &lazy_tokens {
                    all_refs.push(t);
                }
                for lp in &lazy_log_probs {
                    all_refs.push(lp);
                }
                eval(&all_refs);
                let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
                let draft_log_probs: Vec<f32> =
                    lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
                let added = draft_tokens.len();
                cache.seq_len = seq_len_before + added;
                return (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]);
            }
            Err(_) => {
                cache.seq_len = seq_len_before;
            }
        }
    }

    // ── Imperative fallback ─────────────────────────────────────────
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    // Build the full multi-depth lazy graph: no GPU syncs.
    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        // Lazy argmax — NOT evaluated yet.
        let lazy_tok = lazy_argmax_logits(&logits);
        // Compute draft log-prob at T=1.0 (model’s own confidence in its argmax
        // choice), staying in the lazy graph alongside the token selection.
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, 1.0, vocab);
        lazy_tokens.push(lazy_tok.clone());
        lazy_log_probs.push(lazy_lp);

        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }

    // Single batch eval for all depth levels at once — tokens and log-probs together.
    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for lp in &lazy_log_probs {
        all_refs.push(lp);
    }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();

    let added = draft_tokens.len();
    (
        draft_tokens,
        draft_log_probs,
        vec![], // no distributions for greedy
        added,
        [0.0f32; 3],
    )
}

/// Fused lazy draft path: argmax selection + GPU log-probs in a single eval.
///
/// Uses lazy `argmax` for token selection (like the greedy path) but also
/// computes `log(softmax(logits/T)[token])` on GPU for rejection-sampling
/// acceptance.  The entire multi-depth graph is built lazily and materialised
/// in a single `eval`, eliminating the per-depth GPU sync barriers that made
/// the previous sampled path 3–4× slower.
///
/// The argmax-selected tokens are deterministic for a given hidden state,
/// matching how MTPLX and Lightning-MLX produce MTP drafts.  Rejection
/// sampling acceptance (min(1, p_target/p_draft)) still provides stochastic
/// quality control over the output.
#[allow(clippy::too_many_arguments)]
fn mtp_draft_tokens_sampled(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    _rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let temperature = head.draft_sampling.temperature;

    // ── Compiled path ───────────────────────────────────────────────
    // Sampled uses argmax for token selection (T=0 in closure), then
    // temperature-scaled log-prob outside.
    if let Some(compiled) = build_compiled_mtp_draft(head, weights, cfg, cache, max_depth, 0.0) {
        let seq_len_before = cache.seq_len;
        let first_token_data = [first_token];
        let first_token_arr = MlxArray::from_raw_data(
            first_token_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        match compiled.try_apply(&[first_hidden, &first_token_arr]) {
            Ok(closure_outputs) => {
                let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
                let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
                for d in 0..max_depth {
                    let logits = &closure_outputs[d * 2 + 1];
                    let lazy_tok = lazy_argmax_logits(logits);
                    let lazy_lp = gpu_draft_log_prob_lazy(logits, &lazy_tok, temperature, vocab);
                    lazy_tokens.push(lazy_tok);
                    lazy_log_probs.push(lazy_lp);
                }
                let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
                for t in &lazy_tokens {
                    all_refs.push(t);
                }
                for lp in &lazy_log_probs {
                    all_refs.push(lp);
                }
                eval(&all_refs);
                let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
                let draft_log_probs: Vec<f32> =
                    lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
                let added = draft_tokens.len();
                cache.seq_len = seq_len_before + added;
                return (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]);
            }
            Err(_) => {
                cache.seq_len = seq_len_before;
            }
        }
    }

    // ── Imperative fallback ─────────────────────────────────────────
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        let lazy_tok = lazy_argmax_logits(&logits);
        lazy_tokens.push(lazy_tok.clone());

        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, temperature, vocab);
        lazy_log_probs.push(lazy_lp);

        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }

    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for lp in &lazy_log_probs {
        all_refs.push(lp);
    }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();

    let added = draft_tokens.len();
    (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3])
}

/// Stochastic MTP draft path: GPU-side `random_categorical` sampling in a fused
/// lazy graph.
///
/// Each depth samples a token via `random_categorical(logits / temperature)` on
/// GPU, then chains the lazy token array into the next depth's MTP head forward
/// (same pattern as `mtp_draft_tokens_sampled` but with true stochastic sampling
/// instead of argmax).  The entire multi-depth graph is built lazily and
/// materialised in a single `eval`, eliminating the per-depth GPU sync barriers
/// that previously made this path 3–4× slower than greedy.
///
/// Uses MLX's internal RNG state (not the per-request `Xorshift64`), so results
/// are not bit-reproducible across runs.  Output quality is preserved because
/// the target model's verify step rejection-samples against the true distribution.
///
/// Falls back to argmax when `temperature <= 0` (greedy is deterministic).
#[allow(clippy::too_many_arguments)]
fn mtp_draft_tokens_stochastic(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    _rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let temperature = head.draft_sampling.temperature;

    // ── Compiled path ───────────────────────────────────────────────
    // Stochastic uses random_categorical inside the closure (temperature > 0).
    if let Some(compiled) =
        build_compiled_mtp_draft(head, weights, cfg, cache, max_depth, temperature)
    {
        let seq_len_before = cache.seq_len;
        let first_token_data = [first_token];
        let first_token_arr = MlxArray::from_raw_data(
            first_token_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        match compiled.try_apply(&[first_hidden, &first_token_arr]) {
            Ok(closure_outputs) => {
                // closure_outputs has logits at odd indices; we need to
                // re-derive tokens from logits via lazy_random_sample for
                // log-prob consistency with the imperative path.
                let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
                let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
                for d in 0..max_depth {
                    let logits = &closure_outputs[d * 2 + 1];
                    let lazy_tok = if temperature > 0.0 {
                        lazy_random_sample(logits, temperature, vocab)
                    } else {
                        lazy_argmax_logits(logits)
                    };
                    let lazy_lp =
                        gpu_draft_log_prob_lazy(logits, &lazy_tok, temperature.max(1.0), vocab);
                    lazy_tokens.push(lazy_tok);
                    lazy_log_probs.push(lazy_lp);
                }
                let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
                for t in &lazy_tokens {
                    all_refs.push(t);
                }
                for lp in &lazy_log_probs {
                    all_refs.push(lp);
                }
                eval(&all_refs);
                let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
                let draft_log_probs: Vec<f32> =
                    lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
                let added = draft_tokens.len();
                cache.seq_len = seq_len_before + added;
                return (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]);
            }
            Err(_) => {
                cache.seq_len = seq_len_before;
            }
        }
    }

    // ── Imperative fallback ─────────────────────────────────────────
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);

    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for _ in 0..max_depth {
        let new_hidden = mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let post_norm_hidden = mtp_hidden_post_norm(&new_hidden, head, cfg);
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, weights, cfg);

        // GPU-side stochastic sampling (lazy — no CPU sync).
        let lazy_tok = if temperature > 0.0 {
            lazy_random_sample(&logits, temperature, vocab)
        } else {
            lazy_argmax_logits(&logits)
        };
        lazy_tokens.push(lazy_tok.clone());

        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, temperature.max(1.0), vocab);
        lazy_log_probs.push(lazy_lp);

        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }

    // Single batch eval for all depth levels — tokens and log-probs together.
    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for lp in &lazy_log_probs {
        all_refs.push(lp);
    }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();

    let added = draft_tokens.len();
    (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3])
}

// -------------------------------------------------------------------------
// GLM 4.7 Flash MTP forward
// -------------------------------------------------------------------------

/// Run one recurrent GLM MTP head forward pass for a single decode step.
///
/// Returns new hidden state `[1, 1, hidden_size]`.
///
/// * `head`           — GLM MTP weights.
/// * `main_hidden`    — post-norm hidden from the main model (shape `[1, 1, hidden_size]`).
/// * `prev_token_arr` — token ID as a GPU uint32 array, shape `[1]`.
/// * `weights`        — main model weights (for the shared token embedding).
/// * `cache`          — 1-layer GLM MLA KV cache for this head.
/// * `cfg`            — main model config (provides rms_norm_eps, rope_theta, mla_attention, etc.).
/// * `rope_offset_override` — explicit RoPE offset (capped warmup); `None` to use `cache.seq_len`.
pub fn glm_mtp_head_forward(
    head: &GlmMtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
) -> MlxArray {
    let token_offset = rope_offset_override.unwrap_or(cache.seq_len + cache.rope_offset);

    // 1. embed prev_token → [1, 1, hidden_size] in bf16.
    let embed = embed_tokens_arr(prev_token_arr, &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);

    // 2. Fused input: eh_proj(cat([enorm(embed), hnorm(main_hidden)]))
    let enormed = rms_norm(&embed, Some(&head.enorm), cfg.rms_norm_eps, None);
    let hnormed = rms_norm(main_hidden, Some(&head.hnorm), cfg.rms_norm_eps, None);
    let combined = concatenate(&[&enormed, &hnormed], -1, None);
    let h = qw(&combined, &head.eh_proj);

    // 3. GLM transformer layer (MLA attention + MoE FFN), same pattern as glm4_moe_lite::layer_forward.
    let normed = rms_norm(&h, Some(&head.layer.attn_norm), cfg.rms_norm_eps, None);
    let attn_proj = glm_mla_attention_forward(cfg, &head.layer, &normed, cache, 0, token_offset);
    let attn_proj = if let Some(post_norm) = &head.layer.attn_post_norm {
        rms_norm(&attn_proj, Some(post_norm), cfg.rms_norm_eps, None)
    } else {
        attn_proj
    };
    cache.seq_len += 1;
    let hidden = add(&h, &attn_proj, None);

    let normed2 = rms_norm(&hidden, Some(&head.layer.ffn_norm), cfg.rms_norm_eps, None);
    let ffn_out = if head.layer.router_proj.is_some() {
        let (top_k_indices, top_k_weights) = moe_router_glm(cfg, &head.layer, &normed2);
        let mut out =
            moe_experts_forward(cfg, &head.layer, &normed2, &top_k_indices, &top_k_weights);
        if head.layer.shared_gate_proj.is_some() {
            out = add(
                &out,
                &shared_expert_forward(cfg, &head.layer, &normed2),
                None,
            );
        }
        rms_norm_opt(&out, head.layer.ffn_post_norm.as_ref(), cfg.rms_norm_eps)
    } else {
        ffn_swiglu(
            cfg,
            &head.layer,
            &normed2,
            head.layer.ffn_post_norm.as_ref(),
        )
    };

    add(&hidden, &ffn_out, None)
}

/// Apply `shared_head.head(rms_norm(hidden, shared_head_norm))` to produce draft logits.
///
/// Returns f32 logits `[vocab_size]` ready for argmax / sampling.
pub fn glm_mtp_hidden_to_logits(
    hidden: &MlxArray,
    head: &GlmMtpWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let normed = rms_norm(hidden, Some(&head.shared_head_norm), cfg.rms_norm_eps, None);
    let logits = qw(&normed, &head.shared_head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    // [1, 1, vocab] → [vocab]
    reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Draft up to `head.max_depth` tokens using the GLM MTP head.
///
/// Returns `(draft_tokens, draft_log_probs, draft_distributions, added, top2_margins)`.
/// Mirrors `mtp_draft_tokens` but calls `glm_mtp_head_forward` + `glm_mtp_hidden_to_logits`.
/// Returns empty when `weights.glm_mtp` is `None`.
#[allow(clippy::too_many_arguments)]
pub fn glm_mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    glm_mtp_draft_tokens_gated(
        weights,
        cfg,
        first_hidden,
        first_token,
        cache,
        max_depth_cap,
        rng,
        resolve_mtp_draft_min_confidence(
            crate::speculation_profile::speculation_profile_from_env(),
            None,
        ),
    )
}

/// Like [`glm_mtp_draft_tokens`] but with an explicit draft-confidence gate.
#[allow(clippy::too_many_arguments)]
pub fn glm_mtp_draft_tokens_gated(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    _rng: &mut Xorshift64,
    min_confidence: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.glm_mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    }

    let vocab = cfg.vocab_size as i32;
    let draft_mode = mtp_draft_mode_from_env();
    let gate_forces_greedy = min_confidence > 0.0 && draft_mode != MtpDraftMode::Stochastic;

    let result = if gate_forces_greedy || draft_mode == MtpDraftMode::Greedy {
        glm_mtp_draft_tokens_greedy(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
        )
    } else {
        // Stochastic path.
        glm_mtp_draft_tokens_stochastic(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
        )
    };

    let appended = result.3;
    let gated = apply_draft_confidence_gate(result, min_confidence);
    let dropped = appended.saturating_sub(gated.3);
    if dropped > 0 {
        let target = cache.seq_len.saturating_sub(dropped);
        let _ = cache.trim_to(target);
    }
    gated
}

/// Greedy GLM MTP draft: lazy argmax across all depths, single batch eval.
#[allow(clippy::too_many_arguments)]
fn glm_mtp_draft_tokens_greedy(
    head: &GlmMtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // ── Compiled path ───────────────────────────────────────────────
    if let Some(compiled) = build_compiled_glm_mtp_draft(head, weights, cfg, cache, max_depth, 0.0)
    {
        let seq_len_before = cache.seq_len;
        let first_token_data = [first_token];
        let first_token_arr = MlxArray::from_raw_data(
            first_token_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        match compiled.try_apply(&[first_hidden, &first_token_arr]) {
            Ok(closure_outputs) => {
                let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
                let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
                for d in 0..max_depth {
                    let logits = &closure_outputs[d * 2 + 1];
                    let lazy_tok = lazy_argmax_logits(logits);
                    let lazy_lp = gpu_draft_log_prob_lazy(logits, &lazy_tok, 1.0, vocab);
                    lazy_tokens.push(lazy_tok);
                    lazy_log_probs.push(lazy_lp);
                }
                let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
                for t in &lazy_tokens {
                    all_refs.push(t);
                }
                for lp in &lazy_log_probs {
                    all_refs.push(lp);
                }
                eval(&all_refs);
                let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
                let draft_log_probs: Vec<f32> =
                    lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
                let added = draft_tokens.len();
                cache.seq_len = seq_len_before + added;
                return (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]);
            }
            Err(_) => {
                cache.seq_len = seq_len_before;
            }
        }
    }

    // ── Imperative fallback ─────────────────────────────────────────
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for _ in 0..max_depth {
        let new_hidden = glm_mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let logits = glm_mtp_hidden_to_logits(&new_hidden, head, cfg);
        let lazy_tok = lazy_argmax_logits(&logits);
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, 1.0, vocab);
        lazy_tokens.push(lazy_tok.clone());
        lazy_log_probs.push(lazy_lp);
        prev_hidden = new_hidden;
        prev_token_arr = lazy_tok;
    }

    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for lp in &lazy_log_probs {
        all_refs.push(lp);
    }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
    let added = draft_tokens.len();
    (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3])
}

/// Stochastic GLM MTP draft: GPU-side `random_categorical` sampling.
#[allow(clippy::too_many_arguments)]
fn glm_mtp_draft_tokens_stochastic(
    head: &GlmMtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let temperature = head.draft_sampling.temperature;

    // ── Compiled path ───────────────────────────────────────────────
    if let Some(compiled) =
        build_compiled_glm_mtp_draft(head, weights, cfg, cache, max_depth, temperature)
    {
        let seq_len_before = cache.seq_len;
        let first_token_data = [first_token];
        let first_token_arr = MlxArray::from_raw_data(
            first_token_data.as_ptr() as *const u8,
            4,
            &[1_i32],
            MlxDtype::Uint32,
        );
        match compiled.try_apply(&[first_hidden, &first_token_arr]) {
            Ok(closure_outputs) => {
                let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
                let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
                for d in 0..max_depth {
                    let logits = &closure_outputs[d * 2 + 1];
                    let lazy_tok = if temperature > 0.0 {
                        lazy_random_sample(logits, temperature, vocab)
                    } else {
                        lazy_argmax_logits(logits)
                    };
                    let lazy_lp =
                        gpu_draft_log_prob_lazy(logits, &lazy_tok, temperature.max(1.0), vocab);
                    lazy_tokens.push(lazy_tok);
                    lazy_log_probs.push(lazy_lp);
                }
                let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
                for t in &lazy_tokens {
                    all_refs.push(t);
                }
                for lp in &lazy_log_probs {
                    all_refs.push(lp);
                }
                eval(&all_refs);
                let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
                let draft_log_probs: Vec<f32> =
                    lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
                let added = draft_tokens.len();
                cache.seq_len = seq_len_before + added;
                return (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]);
            }
            Err(_) => {
                cache.seq_len = seq_len_before;
            }
        }
    }

    // ── Imperative fallback ─────────────────────────────────────────
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut prev_hidden = first_hidden.clone();
    let first_token_data = [first_token];
    let mut prev_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );

    for _ in 0..max_depth {
        let new_hidden = glm_mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let logits = glm_mtp_hidden_to_logits(&new_hidden, head, cfg);
        let lazy_tok = if temperature > 0.0 {
            lazy_random_sample(&logits, temperature, vocab)
        } else {
            lazy_argmax_logits(&logits)
        };
        lazy_tokens.push(lazy_tok.clone());
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, temperature.max(1.0), vocab);
        lazy_log_probs.push(lazy_lp);
        prev_hidden = new_hidden;
        prev_token_arr = lazy_tok;
    }

    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2);
    for t in &lazy_tokens {
        all_refs.push(t);
    }
    for lp in &lazy_log_probs {
        all_refs.push(lp);
    }
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
    let added = draft_tokens.len();
    (draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3])
}

#[cfg(test)]
mod confidence_gate_tests {
    use super::*;

    /// Run the gate over draft `tokens` with per-depth head `probs`, returning
    /// the surviving tokens and the reported `added` count.
    fn gate(tokens: Vec<u32>, probs: Vec<f32>, min_conf: f32) -> (Vec<u32>, usize) {
        let log_probs: Vec<f32> = probs.iter().map(|p| p.ln()).collect();
        let (toks, _lp, _dist, added, _a3) =
            apply_draft_confidence_gate((tokens, log_probs, vec![], 0, [0.0; 3]), min_conf);
        (toks, added)
    }

    #[test]
    fn disabled_gate_keeps_full_draft() {
        let (toks, added) = gate(vec![1, 2, 3], vec![0.99, 0.10, 0.95], 0.0);
        assert_eq!(toks, vec![1, 2, 3]);
        assert_eq!(added, 3);
    }

    #[test]
    fn all_confident_unchanged() {
        let (toks, added) = gate(vec![1, 2, 3], vec![0.99, 0.97, 0.96], 0.90);
        assert_eq!(toks, vec![1, 2, 3]);
        assert_eq!(added, 3);
    }

    #[test]
    fn truncates_at_first_low_confidence_depth() {
        // depth 1 (0.50) is below the 0.90 gate -> keep only depth 0.
        let (toks, added) = gate(vec![1, 2, 3], vec![0.97, 0.50, 0.99], 0.90);
        assert_eq!(toks, vec![1]);
        assert_eq!(added, 1);
    }

    #[test]
    fn low_confidence_first_token_empties_draft() {
        let (toks, added) = gate(vec![1, 2, 3], vec![0.40, 0.99, 0.99], 0.90);
        assert!(toks.is_empty());
        assert_eq!(added, 0);
    }

    #[test]
    fn non_finite_log_prob_truncates() {
        let log_probs = vec![(0.99_f32).ln(), f32::NEG_INFINITY, (0.99_f32).ln()];
        let (toks, _lp, _dist, added, _a3) =
            apply_draft_confidence_gate((vec![1, 2, 3], log_probs, vec![], 0, [0.0; 3]), 0.90);
        assert_eq!(toks, vec![1]);
        assert_eq!(added, 1);
    }
}
