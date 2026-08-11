use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, ScaledDotProductAttentionMask, add, argmax,
    astype, broadcast_to, concatenate, eval, multiply, random_categorical, reshape, rms_norm,
    rope_dynamic, scaled_dot_product_attention_with_mask, sigmoid, slice, softmax, take,
};

use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::model::shared::{
    apply_final_logit_softcap, ffn_swiglu, flatten_attention_output_bhsd,
    glm_mla_attention_forward, moe_experts_forward, moe_router_deepseek_v3, moe_router_glm,
    moe_router_qwen3, prepare_value_bhsd_from_proj, qk_norm_bhsd_from_proj,
    qk_norm_rope_bhsd_from_proj, qw, rms_norm_opt, shared_expert_forward,
};
use crate::model::{ModelConfig, deepseek_v4_family, embed_tokens_arr};
use crate::sampling::{TokenDistribution, Xorshift64};
use crate::weights::{DeepseekV4NextnWeights, GlmMtpWeights, ModelWeights, MtpWeights};
use std::sync::OnceLock;

/// Draft sampling mode for MTP speculative decoding.
///
/// `Greedy` uses argmax selection (current default, single GPU eval).
/// `Stochastic` applies temperature sampling per depth via
/// `random_categorical(logits / T)` (no top-p/top-k truncation). Requires
/// per-depth GPU sync, but recovers acceptance when MTP head argmax disagrees
/// with the target model. Rejection-sampling exactness holds for any covering
/// proposal distribution; the temperature lock between sample and log-prob is
/// what matters for accept/reject.
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
pub fn mtp_draft_min_confidence_env_value() -> Option<f32> {
    mtp_draft_min_confidence_explicit()
}

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
/// concern. Accordingly no profile raises the gate above the validated `0.90`
/// default: `coding`/`agentic` lower it for their suites, and `chatbot`/high-T
/// `auto` defer (the former 0.99 diversity pin was a throughput regression —
/// −6..−26% MTP decode in the 2026-07-28 6-bit refresh — with nothing to
/// protect on an exact path). `temperature` drives `auto`.
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
/// RoPE offset is `cache.seq_len() + cache.rope_offset` (or an explicit
/// `rope_offset_override` when provided).  This matches the mlx-lm
/// `cache.offset` convention while supporting capped warmup where physical
/// KV entries start at buffer position 0 but represent tokens at higher
/// prompt positions.  Callers must NOT pass absolute sequence positions
/// unless using `rope_offset_override`.
/// How a single MTP head step sources and advances its K/V context.
///
/// The imperative path uses [`MtpKvStep::Cache`], which appends into the real
/// KV cache and bumps `seq_len` (a side effect).  The **compiled** path uses
/// [`MtpKvStep::Threaded`], which carries the running K/V purely as MLX arrays
/// and concatenates the new token — no cache mutation — so the closure body is
/// pure and safe to `mlx_compile`.  The caller seeds `Threaded` with the
/// cache's existing logical K/V (passed as explicit closure inputs) and reads
/// the final K/V back out after the chain to commit to the cache.
enum MtpKvStep<'a> {
    Cache(&'a mut MlxKVCache),
    Threaded { k: MlxArray, v: MlxArray },
}

impl MtpKvStep<'_> {
    /// RoPE base offset used when no explicit override is supplied.  Only valid
    /// for `Cache`; `Threaded` callers always pass `rope_offset_override`.
    fn rope_base_offset(&self) -> usize {
        match self {
            MtpKvStep::Cache(cache) => cache.seq_len() + cache.rope_offset,
            MtpKvStep::Threaded { .. } => {
                unreachable!("threaded MTP step requires an explicit rope_offset_override")
            }
        }
    }

    /// Append this step's `k_rope`/`v` and return the K/V to feed SDPA
    /// (the full context including the new token).
    fn append(&mut self, k_rope: MlxArray, v: MlxArray) -> (MlxArray, MlxArray) {
        match self {
            MtpKvStep::Cache(cache) => {
                let cached = cache.append(0, k_rope, v);
                cache.advance(1);
                cached
            }
            MtpKvStep::Threaded { k, v: running_v } => {
                // Sequence axis is dim 2 of [1, n_kv_heads, S, head_dim].
                let new_k = concatenate(&[k, &k_rope], 2, None);
                let new_v = concatenate(&[running_v, &v], 2, None);
                *k = new_k.clone();
                *running_v = new_v.clone();
                (new_k, new_v)
            }
        }
    }
}

pub fn mtp_head_forward(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
) -> MlxArray {
    let mut kv = MtpKvStep::Cache(cache);
    mtp_head_forward_inner(
        head,
        main_hidden,
        prev_token_arr,
        weights,
        &mut kv,
        cfg,
        rope_offset_override,
        None,
    )
}

pub fn mtp_warmup_cache_kv_batched(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_tokens: &[u32],
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset: usize,
) {
    if prev_tokens.is_empty() {
        return;
    }
    let seq_len = prev_tokens.len();
    let prev_token_arr = MlxArray::from_raw_data(
        prev_tokens.as_ptr() as *const u8,
        std::mem::size_of_val(prev_tokens),
        &[seq_len as i32],
        MlxDtype::Uint32,
    );
    let embed = embed_tokens_arr(&prev_token_arr, &weights.token_embedding, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);
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
    let h = qw(&combined, &head.fc);
    let normed = rms_norm(&h, Some(&head.attn_norm), cfg.rms_norm_eps, None);

    let k_raw = qw(&normed, &head.k_proj);
    let v_raw = qw(&normed, &head.v_proj);
    let v = prepare_value_bhsd_from_proj(
        &v_raw,
        false,
        head.n_kv_heads,
        head.head_dim,
        seq_len,
        cfg.rms_norm_eps,
    );

    let (rope_base, rope_freqs_ref) = if let Some(freqs) = cfg.rope_freqs.as_ref() {
        (None, Some(freqs))
    } else {
        (Some(cfg.rope_theta), None)
    };
    let k_rope = qk_norm_rope_bhsd_from_proj(
        &k_raw,
        head.k_norm.as_ref(),
        head.n_kv_heads,
        head.head_dim,
        seq_len,
        cfg.rms_norm_eps,
        cfg.rope_dims,
        rope_base,
        rope_offset,
        rope_freqs_ref,
    );
    cache.set_layer_kv_logical(0, k_rope, v, seq_len);
}

#[allow(clippy::too_many_arguments)]
fn mtp_head_forward_inner(
    head: &MtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    kv: &mut MtpKvStep,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
    rope_offset_arr: Option<&MlxArray>,
) -> MlxArray {
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

        let (q_rope, k_rope) = if let Some(offset_arr) = rope_offset_arr {
            // Dynamic RoPE: offset flows through the computation graph as an
            // array node, enabling mx.compile closure reuse across steps.
            let q_normed = qk_norm_bhsd_from_proj(
                &q_raw,
                head.q_norm.as_ref(),
                head.n_heads,
                head.head_dim,
                1,
                cfg.rms_norm_eps,
            );
            let k_normed = qk_norm_bhsd_from_proj(
                &k_raw,
                head.k_norm.as_ref(),
                head.n_kv_heads,
                head.head_dim,
                1,
                cfg.rms_norm_eps,
            );
            let q_r = rope_dynamic(
                &q_normed,
                cfg.rope_dims as i32,
                false,
                rope_base,
                1.0,
                offset_arr,
                rope_freqs_ref,
                None,
            );
            let k_r = rope_dynamic(
                &k_normed,
                cfg.rope_dims as i32,
                false,
                rope_base,
                1.0,
                offset_arr,
                rope_freqs_ref,
                None,
            );
            (q_r, k_r)
        } else {
            // Static RoPE: offset baked as a scalar constant.
            // Use the explicit RoPE offset when provided (e.g. during capped
            // warmup where KV entries start at buffer position 0 but represent
            // prompt tokens at higher positions).  Otherwise use the MTP
            // KV-cache seq_len + rope_offset (matches mlx-lm cache.offset).
            let token_offset = rope_offset_override.unwrap_or_else(|| kv.rope_base_offset());
            let q_r = qk_norm_rope_bhsd_from_proj(
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
            let k_r = qk_norm_rope_bhsd_from_proj(
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
            (q_r, k_r)
        };

        let (cached_k, cached_v) = kv.append(k_rope, v);

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
            // Reserved compile-cache slot: main model layers use 0..N-1.
            // Sharing layer 0 with the main stack reuses a quantized_matmul
            // graph against bf16 MTP sidecar weights and panics at load.
            ffn_swiglu(cfg, &head.ffn_layer, &normed, None, usize::MAX)
        };
        h = add(&h, &ffn_out, None);
    }

    h
}

/// Apply `rms_norm(hidden, mtp_norm) @ draft_lm_head` to produce draft logits.
///
/// Returns f32 logits `[vocab_size]` ready for argmax / sampling.
pub fn mtp_hidden_to_logits(
    hidden: &MlxArray,
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let normed = mtp_hidden_post_norm(hidden, head, cfg);
    mtp_post_norm_to_logits(&normed, head, weights, cfg)
}

fn mtp_hidden_post_norm(hidden: &MlxArray, head: &MtpWeights, cfg: &ModelConfig) -> MlxArray {
    rms_norm(hidden, Some(&head.mtp_norm), cfg.rms_norm_eps, None)
}

fn mtp_post_norm_to_logits(
    post_norm_hidden: &MlxArray,
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    use mlx_sys::reshape as mlx_reshape;
    let lm_head = head.draft_lm_head.as_ref().unwrap_or(&weights.lm_head);
    let logits = qw(post_norm_hidden, lm_head);
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
    let logits = mtp_post_norm_to_logits(&post_norm_hidden, head, weights, cfg);
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
/// **Pure:** the closure does NOT capture or mutate the KV cache.  The existing
/// context is supplied as the explicit inputs `init_k` / `init_v`, and the new
/// per-depth K/V is threaded functionally via [`MtpKvStep::Threaded`] (concat,
/// no cache write).  This satisfies `mlx_compile`'s pure-function contract; the
/// earlier impure version aborted decode with `[eval] Attempting to eval an
/// array without a primitive` because the captured lazy KV entered the trace as
/// an un-passed constant.  The RoPE base offset is passed as an array input
/// (input 4) and flows through the computation graph via `rope_dynamic`, so
/// the compiled closure is reused across decode steps without recompilation.
///
/// Inputs:  `[first_hidden, first_token, init_k, init_v, base_offset_arr]`.
/// Outputs: `[hidden_0, logits_0, tok_0, …, hidden_{D-1}, logits_{D-1},
/// tok_{D-1}, final_k, final_v]` — 3 arrays per depth plus the final threaded
/// K/V (so the caller can commit it to the cache).  Callers use the `tok`
/// output directly rather than re-sampling, so the output token always matches
/// the chained token.
///
/// Returns `None` when the kill switch `AX_MTP_COMPILED_HEAD=0` is set or
/// compilation fails.
fn build_compiled_mtp_draft(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    max_depth: usize,
    temperature: f32,
) -> Option<MlxClosure> {
    if !fastpath::mtp_compiled_head_enabled() {
        return None;
    }
    let cfg_addr = cfg as *const ModelConfig as usize;
    let weights_addr = weights as *const ModelWeights as usize;
    let head_addr = head as *const MtpWeights as usize;
    let vocab = cfg.vocab_size as i32;

    // SAFETY: The closure captures raw pointers to `cfg`, `weights`, and `head`
    // because `MlxClosure::new_dyn()` requires `'static` captures. The referenced
    // objects outlive every invocation: the sole caller (`run_compiled_mtp_draft`)
    // borrows all three for its whole scope, applies the returned closure
    // synchronously, and drops it before returning.
    let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
        let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
        let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
        let head_ref = unsafe { &*(head_addr as *const MtpWeights) };

        let mut prev_hidden = inputs.get(0);
        let mut prev_token_arr = inputs.get(1);
        let mut kv = MtpKvStep::Threaded {
            k: inputs.get(2),
            v: inputs.get(3),
        };
        // Input 4: RoPE base offset as an array scalar (int32).  Flows through
        // the computation graph so the compiled closure is reusable across decode
        // steps with different positions (eliminates per-step recompilation).
        let base_offset_arr = inputs.get(4);

        let mut outputs: Vec<MlxArray> = Vec::with_capacity(max_depth * 3 + 2);
        for d in 0..max_depth {
            let depth_offset = if d == 0 {
                base_offset_arr.clone()
            } else {
                let d_val = d as i32;
                let d_arr = MlxArray::from_raw_data(
                    &d_val as *const i32 as *const u8,
                    4,
                    &[1_i32],
                    MlxDtype::Int32,
                );
                add(&base_offset_arr, &d_arr, None)
            };
            let new_hidden = mtp_head_forward_inner(
                head_ref,
                &prev_hidden,
                &prev_token_arr,
                weights_ref,
                &mut kv,
                cfg_ref,
                None,
                Some(&depth_offset),
            );
            let post_norm = mtp_hidden_post_norm(&new_hidden, head_ref, cfg_ref);
            let logits = mtp_post_norm_to_logits(&post_norm, head_ref, weights_ref, cfg_ref);

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
            outputs.push(tok.clone());
            prev_hidden = post_norm;
            prev_token_arr = tok;
        }
        // Emit the final threaded K/V so the caller can commit it to the cache
        // (the closure never touched the cache itself).
        if let MtpKvStep::Threaded { k, v } = kv {
            outputs.push(k);
            outputs.push(v);
        }
        outputs
    });
    closure.compile(false).ok()
}

/// Standard draft result: `(draft_tokens, draft_log_probs, draft_distributions,
/// added, top2_margins)`.  `top2_margins` is always `[0.0; 3]` on the compiled
/// path (retained for API compatibility with the imperative path).
type DraftTokens = (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]);

/// Build and apply the pure compiled Qwen MTP draft closure for one draft call,
/// then commit the threaded K/V back to the cache.
///
/// Shared path for the three Qwen compiled entry points (greedy / sampled /
/// stochastic).  Returns `None` — falling back to the imperative path — when:
/// the kill switch is off / compile fails; the MTP layer has no existing KV yet
/// (the first decode step of a sequence, handled imperatively so the closure
/// always receives a non-empty `init_k`/`init_v`); or the apply errors.
///
/// `logprob_temperature` is the temperature applied to the draft log-prob (the
/// in-closure token-sampling temperature is baked in at build time): `1.0` for
/// greedy, and the same sampling temperature used for the token draw for
/// sampled/stochastic (must match so rejection sampling sees true `q(token)`).
#[allow(clippy::too_many_arguments)]
fn run_compiled_mtp_draft(
    head: &MtpWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    sample_temperature: f32,
    logprob_temperature: f32,
    compute_log_probs: bool,
) -> Option<DraftTokens> {
    // A depth-one draft has no recurrent chain to fuse. Building/applying the
    // compiled closure only adds transform bookkeeping over the same lazy
    // graph and is measurably slower on Apple Silicon; keep the imperative
    // single-eval path for this exact-depth production profile.
    if max_depth <= 1 || !fastpath::mtp_compiled_head_enabled() {
        return None;
    }
    // Seed the closure with the existing logical KV as explicit inputs.  When
    // the layer is empty (first step) fall back to the imperative path so the
    // pure closure never has to special-case a zero-length context.
    let (init_k, init_v) = cache.logical_layer_kv(0)?;
    let seq_len_before = cache.seq_len();
    let base_offset = seq_len_before + cache.rope_offset;
    let compiled = build_compiled_mtp_draft(head, weights, cfg, max_depth, sample_temperature)?;

    let first_token_data = [first_token];
    let first_token_arr = MlxArray::from_raw_data(
        first_token_data.as_ptr() as *const u8,
        4,
        &[1_i32],
        MlxDtype::Uint32,
    );
    // RoPE base offset as an array scalar so the compiled closure is reusable
    // across decode steps without recompilation.
    let offset_val = base_offset as i32;
    let offset_arr = MlxArray::from_raw_data(
        &offset_val as *const i32 as *const u8,
        4,
        &[1_i32],
        MlxDtype::Int32,
    );
    let outputs = compiled
        .try_apply(&[
            first_hidden,
            &first_token_arr,
            &init_k,
            &init_v,
            &offset_arr,
        ])
        .ok()?;

    // outputs = [hidden_d, logits_d, tok_d, ...]*D, final_k, final_v.
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
    let mut lazy_log_probs: Vec<MlxArray> = Vec::with_capacity(max_depth);
    for d in 0..max_depth {
        let logits = &outputs[d * 3 + 1];
        let lazy_tok = outputs[d * 3 + 2].clone();
        if compute_log_probs {
            lazy_log_probs.push(gpu_draft_log_prob_lazy(
                logits,
                &lazy_tok,
                logprob_temperature,
                vocab,
            ));
        }
        lazy_tokens.push(lazy_tok);
    }
    let final_k = &outputs[max_depth * 3];
    let final_v = &outputs[max_depth * 3 + 1];

    let mut all_refs: Vec<&MlxArray> = Vec::with_capacity(max_depth * 2 + 2);
    all_refs.extend(lazy_tokens.iter());
    all_refs.extend(lazy_log_probs.iter());
    all_refs.push(final_k);
    all_refs.push(final_v);
    eval(&all_refs);

    let draft_tokens: Vec<u32> = lazy_tokens.iter().map(|a| a.data_u32()[0]).collect();
    let draft_log_probs: Vec<f32> = lazy_log_probs.iter().map(|a| a.data_f32()[0]).collect();
    let added = draft_tokens.len();
    // Commit the threaded K/V + advanced seq_len into the cache so the verify
    // step and subsequent imperative appends see the correct state.
    cache.set_layer_kv_logical(0, final_k.clone(), final_v.clone(), seq_len_before + added);
    Some((draft_tokens, draft_log_probs, vec![], added, [0.0f32; 3]))
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
    // Prefer temperature-aware resolution when available via process profile.
    // Callers with request temperature should use `mtp_draft_tokens_gated` +
    // `mtp_adaptive_gate::resolve_mtp_gate_from_env`.
    mtp_draft_tokens_gated(
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

fn greedy_draft_needs_temperature_log_probs(
    draft_temperature: f32,
    min_confidence: f32,
    qwen_exact_profile: bool,
) -> bool {
    draft_temperature > 0.0 && !(qwen_exact_profile && min_confidence == 0.0)
}

/// Temperature at which Qwen MTP draft log-probs are recorded by
/// [`mtp_draft_tokens_gated`] / hybrid forced-prefix tails.
///
/// Accept-path rejection sampling must use this same T for `q(token)` (see
/// runner `mtp_pending_draft_log_prob_temperature`). Using the head's
/// `draft_sampling.temperature` (often 0.7) while the greedy draft path wrote
/// log-probs at T=1.0 breaks exactness — the common Qwen3.6 linear exact /
/// confidence-gated path.
///
/// Mirrors the branch structure of [`mtp_draft_tokens_gated`]:
/// - confidence gate force-greedy → log-probs at **1.0**
/// - stochastic mode → head draft temperature (or 1.0 if unset)
/// - greedy mode with temperature log-probs → head draft temperature
/// - pure greedy (exact profile, gate 0) → **1.0** when log-probs exist
pub fn qwen_mtp_draft_log_prob_temperature(
    mode: MtpDraftMode,
    draft_head_temperature: f32,
    min_confidence: f32,
    qwen_exact_profile: bool,
) -> f32 {
    let head_t = if draft_head_temperature > 0.0 {
        draft_head_temperature
    } else {
        1.0
    };
    let gate_forces_greedy = min_confidence > 0.0 && mode != MtpDraftMode::Stochastic;
    if gate_forces_greedy {
        return 1.0;
    }
    match mode {
        MtpDraftMode::Stochastic => head_t,
        MtpDraftMode::Greedy => {
            if greedy_draft_needs_temperature_log_probs(
                draft_head_temperature,
                min_confidence,
                qwen_exact_profile,
            ) {
                head_t
            } else {
                1.0
            }
        }
    }
}

/// Process-env draft mode + head T / gate / exact profile → recorded log-prob T.
pub fn qwen_mtp_draft_log_prob_temperature_from_env(
    draft_head_temperature: f32,
    min_confidence: f32,
    qwen_exact_profile: bool,
) -> f32 {
    qwen_mtp_draft_log_prob_temperature(
        mtp_draft_mode_from_env(),
        draft_head_temperature,
        min_confidence,
        qwen_exact_profile,
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
            min_confidence > 0.0,
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
                // The exact Qwen depth-one profile uses a deterministic-delta
                // proposal law. With the confidence gate disabled, the
                // temperature-scaled draft log-prob is neither used for
                // gating nor acceptance, so avoid building its full-vocabulary
                // softmax. Preserve the established path for every other
                // runtime profile.
                let use_temperature = greedy_draft_needs_temperature_log_probs(
                    head.draft_sampling.temperature,
                    min_confidence,
                    fastpath::qwen_linear_mtp_exact_enabled(),
                );
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
                        min_confidence > 0.0,
                    )
                }
            }
        }
    };

    // `result.3` is the number of MTP KV entries the draft path physically
    // appended to `cache` (one per head forward). The confidence gate can drop
    // the low-confidence tail of the draft; those tail forwards already wrote
    // their entries, so dropping them without trimming leaves stale rows that
    // inflate `cache.seq_len()` above the returned `added`. That breaks the
    // invariant the decode loop relies on (MTP `cache.seq_len()` == the running
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
        let target = cache.seq_len().saturating_sub(dropped);
        if !cache.trim_to(target) {
            // The `cache.seq_len() == added` invariant breaks if the trim is
            // refused; the next step's draft head then attends over gated-out
            // rows at inflated offsets. Output stays correct (every draft is
            // verified), so warn rather than fail.
            tracing::warn!(target, "MTP confidence-gate trim refused");
        }
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
    // Same gate as pure MTP for this request; `None` → global profile default.
    min_confidence: Option<f32>,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let min_confidence = min_confidence.unwrap_or_else(|| {
        resolve_mtp_draft_min_confidence(
            crate::speculation_profile::speculation_profile_from_env(),
            None,
        )
    });
    if forced_prefix.is_empty() {
        return mtp_draft_tokens_gated(
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            Some(max_tail_depth),
            rng,
            min_confidence,
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
    let (draft, log_probs, distributions, tail_added, top2_margins) = mtp_draft_tokens_gated(
        weights,
        cfg,
        &prev_hidden,
        last_forced,
        cache,
        Some(max_tail_depth),
        rng,
        min_confidence,
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
/// Greedy MTP draft tokens scheduled with `async_eval` but not yet extracted.
///
/// The caller stores the arrays across the decode-cycle boundary and extracts
/// host values at the start of the next cycle, overlapping the draft head's
/// GPU forward with per-token host work (detokenization, stream emission).
pub struct MtpLazyDraft {
    /// One `[1]` u32 lazy argmax array per drafted depth level.
    pub tokens: Vec<MlxArray>,
}

/// Build and asynchronously schedule the greedy zero-gate MTP draft graph.
///
/// This is the imperative lazy body of [`mtp_draft_tokens_greedy`] with the
/// terminal `eval` replaced by `async_eval` and host extraction deferred to
/// the caller. It is exactness-preserving — the identical lazy graph is
/// evaluated, only the synchronization point moves — and is only legal in the
/// regime where the synchronous greedy path computes no log-probs or
/// distributions (confidence gate disabled, non-stochastic drafting).
pub fn mtp_draft_tokens_greedy_async(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
) -> Option<MtpLazyDraft> {
    let head = weights.mtp.as_ref()?;
    let max_depth = max_depth_cap.unwrap_or(head.max_depth).min(head.max_depth);
    if max_depth == 0 {
        return None;
    }
    let mut lazy_tokens: Vec<MlxArray> = Vec::with_capacity(max_depth);
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
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, head, weights, cfg);
        let lazy_tok = lazy_argmax_logits(&logits);
        lazy_tokens.push(lazy_tok.clone());
        prev_hidden = post_norm_hidden;
        prev_token_arr = lazy_tok;
    }
    let refs: Vec<&MlxArray> = lazy_tokens.iter().collect();
    mlx_sys::async_eval(&refs);
    Some(MtpLazyDraft {
        tokens: lazy_tokens,
    })
}

/// Extract host token values from an async-scheduled draft.
///
/// Blocks only if the scheduled GPU work has not yet completed.
pub fn mtp_lazy_draft_extract(lazy: &MtpLazyDraft) -> Vec<u32> {
    let refs: Vec<&MlxArray> = lazy.tokens.iter().collect();
    eval(&refs);
    lazy.tokens.iter().map(|a| a.data_u32()[0]).collect()
}

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
    compute_log_probs: bool,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // ── Compiled path ───────────────────────────────────────────────
    if let Some(result) = run_compiled_mtp_draft(
        head,
        weights,
        cfg,
        first_hidden,
        first_token,
        cache,
        max_depth,
        vocab,
        0.0,
        1.0,
        compute_log_probs,
    ) {
        return result;
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
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, head, weights, cfg);

        // Lazy argmax — NOT evaluated yet.
        let lazy_tok = lazy_argmax_logits(&logits);
        // Compute draft log-prob at T=1.0 (model’s own confidence in its argmax
        // choice), staying in the lazy graph alongside the token selection.
        if compute_log_probs {
            lazy_log_probs.push(gpu_draft_log_prob_lazy(&logits, &lazy_tok, 1.0, vocab));
        }
        lazy_tokens.push(lazy_tok.clone());

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
    if let Some(result) = run_compiled_mtp_draft(
        head,
        weights,
        cfg,
        first_hidden,
        first_token,
        cache,
        max_depth,
        vocab,
        0.0,
        temperature,
        true,
    ) {
        return result;
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
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, head, weights, cfg);

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
    // Stochastic uses random_categorical inside the closure (temperature > 0);
    // the chained tok output is used directly so the output token matches the
    // token used to compute subsequent depths (avoid a second random draw).
    if let Some(result) = run_compiled_mtp_draft(
        head,
        weights,
        cfg,
        first_hidden,
        first_token,
        cache,
        max_depth,
        vocab,
        temperature,
        // Log-prob T must match sample T so rejection sampling sees true q(token).
        if temperature > 0.0 { temperature } else { 1.0 },
        true,
    ) {
        return result;
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
        let logits = mtp_post_norm_to_logits(&post_norm_hidden, head, weights, cfg);

        // GPU-side stochastic sampling (lazy — no CPU sync).
        let lazy_tok = if temperature > 0.0 {
            lazy_random_sample(&logits, temperature, vocab)
        } else {
            lazy_argmax_logits(&logits)
        };
        lazy_tokens.push(lazy_tok.clone());

        let log_prob_t = if temperature > 0.0 { temperature } else { 1.0 };
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, log_prob_t, vocab);
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
/// * `rope_offset_override` — explicit RoPE offset (capped warmup); `None` to use `cache.seq_len()`.
pub fn glm_mtp_head_forward(
    head: &GlmMtpWeights,
    main_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
) -> MlxArray {
    let token_offset = rope_offset_override.unwrap_or(cache.seq_len() + cache.rope_offset);

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
    cache.advance(1);
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
            usize::MAX,
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

/// Like [`glm_mtp_draft_tokens`], but first threads the GLM MTP head through
/// `forced_prefix` (real cache appends, correctly incrementing RoPE offsets)
/// before drafting the tail. Mirrors [`mtp_draft_tokens_after_forced_prefix`]
/// for the GLM MTP head: used in the hybrid n-gram+MTP path so the MTP tail's
/// RoPE offset and `state.mtp_decode_count` correctly account for the
/// n-gram-drafted prefix tokens, instead of drafting the tail as if the
/// prefix never happened (which desyncs RoPE and causes `mtp_decode_count`
/// under-accounting to over-trim `state.mtp_cache` on partial n-gram-prefix
/// rejection).
#[allow(clippy::too_many_arguments)]
pub fn glm_mtp_draft_tokens_after_forced_prefix(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    forced_prefix: &[u32],
    cache: &mut MlxKVCache,
    max_tail_depth: usize,
    rng: &mut Xorshift64,
    // Same gate as pure MTP for this request; `None` → global profile default.
    min_confidence: Option<f32>,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(head) = weights.glm_mtp.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let min_confidence = min_confidence.unwrap_or_else(|| {
        resolve_mtp_draft_min_confidence(
            crate::speculation_profile::speculation_profile_from_env(),
            None,
        )
    });
    if forced_prefix.is_empty() {
        return glm_mtp_draft_tokens_gated(
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            Some(max_tail_depth),
            rng,
            min_confidence,
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
        prev_hidden = glm_mtp_head_forward(
            head,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
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
    let (draft, log_probs, distributions, tail_added, top2_margins) = glm_mtp_draft_tokens_gated(
        weights,
        cfg,
        &prev_hidden,
        last_forced,
        cache,
        Some(max_tail_depth),
        rng,
        min_confidence,
    );

    (
        draft,
        log_probs,
        distributions,
        forced_prefix.len().saturating_add(tail_added),
        top2_margins,
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
        // Token selection is always argmax here (matching Qwen's naming),
        // but the log-prob fed into rejection-sampling accept/reject math
        // must still be computed at the configured draft-sampling
        // temperature when one is set — mirrors Qwen's
        // `mtp_draft_tokens_sampled` vs `mtp_draft_tokens_greedy` split
        // (see `mtp_draft_tokens_gated`'s `use_temperature` check above).
        let log_prob_temperature = if head.draft_sampling.temperature > 0.0 {
            head.draft_sampling.temperature
        } else {
            1.0
        };
        glm_mtp_draft_tokens_greedy(
            head,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
            log_prob_temperature,
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
        let target = cache.seq_len().saturating_sub(dropped);
        if !cache.trim_to(target) {
            // The `cache.seq_len() == added` invariant breaks if the trim is
            // refused; the next step's draft head then attends over gated-out
            // rows at inflated offsets. Output stays correct (every draft is
            // verified), so warn rather than fail.
            tracing::warn!(target, "MTP confidence-gate trim refused");
        }
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
    log_prob_temperature: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // GLM intentionally runs the imperative path (no compiled head).  GLM uses
    // the MLA latent KV cache (glm_mla_layers, two latent tensors), and the
    // compiled head's only payoff is fusing the tiny MTP-head dispatches — which
    // the Qwen A/B showed wins nothing on these large, memory-bandwidth-bound
    // models (±1%).  A pure threaded MLA variant would add real complexity and
    // risk for ~0% gain, so it is deliberately not implemented, not a TODO.
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
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, log_prob_temperature, vocab);
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

    // GLM intentionally runs the imperative path (no compiled head).  GLM uses
    // the MLA latent KV cache (glm_mla_layers, two latent tensors), and the
    // compiled head's only payoff is fusing the tiny MTP-head dispatches — which
    // the Qwen A/B showed wins nothing on these large, memory-bandwidth-bound
    // models (±1%).  A pure threaded MLA variant would add real complexity and
    // risk for ~0% gain, so it is deliberately not implemented, not a TODO.
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
        // Match sample temperature (not max(T,1.0)) so q(token) is exact.
        let log_prob_t = if temperature > 0.0 { temperature } else { 1.0 };
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, log_prob_t, vocab);
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

// -------------------------------------------------------------------------
// DeepSeek V4 nextn (MTP) forward
// -------------------------------------------------------------------------

/// Default draft temperature for the V4 nextn head's stochastic path — the
/// AXQ artifact carries no runtime sampler config, so this matches the GLM
/// sidecar default.
pub const DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE: f32 = 0.7;

/// Temperature used when writing DeepSeek V4 draft log-probs for `mode`
/// (and thus the T the runner must use for rejection-sampling rescale).
///
/// Must match [`deepseek_v4_mtp_draft_tokens_gated`]:
/// - **Greedy** (default): log-probs at T=1.0
/// - **Stochastic** (`AX_MLX_MTP_DRAFT_MODE=stochastic`): sample + log-prob at
///   [`DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE`] (0.7)
///
/// Hard-coding 0.7 for every nextn attach made greedy accepts rescale as if
/// drafts were sampled at 0.7 while log_p was recorded at 1.0.
pub fn deepseek_v4_mtp_draft_log_prob_temperature_for_mode(mode: MtpDraftMode) -> f32 {
    match mode {
        MtpDraftMode::Stochastic => DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE,
        MtpDraftMode::Greedy => 1.0,
    }
}

/// Process-env mode → draft log-prob temperature (see
/// [`deepseek_v4_mtp_draft_log_prob_temperature_for_mode`]).
pub fn deepseek_v4_mtp_draft_log_prob_temperature() -> f32 {
    deepseek_v4_mtp_draft_log_prob_temperature_for_mode(mtp_draft_mode_from_env())
}

/// Think-aware V4 draft temperature. Inside an open think block the target
/// model usually samples at temperature 1.0 (DeepSeek thinking defaults), so
/// a 0.7 draft is systematically sharper than the target and loses
/// acceptance; match the target there. Outside think (or for sharper target
/// sampling) keep the tuned default.
pub fn deepseek_v4_mtp_effective_draft_temperature(in_think: bool, target_temperature: f32) -> f32 {
    if in_think && target_temperature >= 1.0 {
        target_temperature.min(1.0)
    } else {
        DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE
    }
}

/// Temperature used for **both** DeepSeek V4 draft sampling and draft log-probs
/// (and therefore accept-path rejection rescale).
///
/// - **Greedy** (default): always 1.0 — greedy drafts record log-probs at T=1.0
///   regardless of think/target temperature.
/// - **Stochastic**: think-aware sample temperature via
///   [`deepseek_v4_mtp_effective_draft_temperature`].
///
/// DI-DS-MTP: accept rescale previously used mode-only 0.7 while stochastic
/// think drafts sampled at 1.0, breaking rejection-sampling exactness.
pub fn deepseek_v4_mtp_sample_and_log_temperature(
    mode: MtpDraftMode,
    in_think: bool,
    target_temperature: f32,
) -> f32 {
    match mode {
        MtpDraftMode::Greedy => 1.0,
        MtpDraftMode::Stochastic => {
            deepseek_v4_mtp_effective_draft_temperature(in_think, target_temperature)
        }
    }
}

/// Process-env draft mode + request think/target → sample/log temperature.
pub fn deepseek_v4_mtp_sample_and_log_temperature_from_env(
    in_think: bool,
    target_temperature: f32,
) -> f32 {
    deepseek_v4_mtp_sample_and_log_temperature(
        mtp_draft_mode_from_env(),
        in_think,
        target_temperature,
    )
}

/// KV-cache slot count for the dedicated nextn cache: llama.cpp places the
/// MTP block at `il = n_layer + nextn_layer_offset`, so the block appends its
/// raw-path latent K at slot `layer_count` and the cache needs one slot past
/// the main stack.
pub fn deepseek_v4_mtp_cache_layer_count(cfg: &ModelConfig) -> usize {
    cfg.layer_count + 1
}

/// Depth supported by the V4 nextn head: exactly one predictor block
/// (llama.cpp asserts `n_layer_nextn == 1`). Returns 0 — disabling MTP
/// drafts — when the config carries no nextn layer; a multi-block stack is
/// rejected loudly (once) the same way.
fn deepseek_v4_mtp_max_depth(cfg: &ModelConfig) -> usize {
    let Some(v4_cfg) = cfg.deepseek_v4.as_ref() else {
        return 0;
    };
    match v4_cfg.num_nextn_predict_layers {
        1 => 1,
        n => {
            if n > 1 {
                static WARNED: OnceLock<()> = OnceLock::new();
                WARNED.get_or_init(|| {
                    tracing::warn!(
                        num_nextn_predict_layers = n,
                        "DeepSeek V4 MTP supports exactly one nextn block; disabling MTP drafts"
                    );
                });
            }
            0
        }
    }
}

/// Run the DeepSeek V4 nextn (MTP) block for a single decode step.
///
/// Returns the block's packed output hidden `[1, 1, hc*hidden]` — re-fed as
/// `packed_hidden` when chaining draft steps (llama.cpp `t_h_nextn`).
///
/// * `nextn`          — nextn weights (block + sidecar tensors).
/// * `packed_hidden`  — packed pre-collapse residual `[1, 1, hc*hidden]` from
///   the main model (`deepseek_v4_forward_all_positions_with_packed`) or a
///   previous nextn block output.
/// * `prev_token_arr` — token ID as a GPU uint32 array, shape `[1]`.
/// * `weights`        — main model weights (shared token-embedding fallback).
/// * `cache`          — dedicated nextn KV cache
///   ([`deepseek_v4_mtp_cache_layer_count`] slots).
/// * `cfg`            — main model config.
/// * `rope_offset_override` — explicit RoPE offset (capped warmup); `None` to
///   use `cache.seq_len() + cache.rope_offset` (the GLM head's convention).
pub fn deepseek_v4_mtp_head_forward(
    nextn: &DeepseekV4NextnWeights,
    packed_hidden: &MlxArray,
    prev_token_arr: &MlxArray,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset_override: Option<usize>,
) -> MlxArray {
    let v4_cfg = cfg.deepseek_v4.as_ref().expect("DeepSeek V4 config");
    let layer = nextn
        .layer
        .as_deref()
        .expect("DeepSeek V4 nextn block layer weights");
    debug_assert!(
        layer
            .deepseek_v4
            .as_ref()
            .is_some_and(|v4_w| v4_w.tid2eid.is_none()),
        "DeepSeek V4 nextn block must never hash-route (learned sqrtsoftplus router only)"
    );
    let token_offset = rope_offset_override.unwrap_or(cache.seq_len() + cache.rope_offset);
    let hc = v4_cfg.hc_mult as i32;
    let hidden = cfg.hidden_size as i32;

    // 1. Embed the previous token (nextn-specific table when present) →
    // enorm; reshape the packed hidden to per-stream layout → hnorm per
    // stream (llama.cpp graph_mtp, deepseek4.cpp:1438-1457).
    let embed_table = nextn
        .embed_tokens
        .as_ref()
        .unwrap_or(&weights.token_embedding);
    let embed = embed_tokens_arr(prev_token_arr, embed_table, cfg.hidden_size);
    let embed = astype(&embed, MlxDtype::Bfloat16, None);
    let e_normed = rms_norm(
        &embed,
        Some(nextn.enorm.as_ref().expect("DeepSeek V4 nextn enorm")),
        cfg.rms_norm_eps,
        None,
    );
    let h_streams = reshape(packed_hidden, &[1, 1, hc, hidden], None);
    let h_normed = rms_norm(
        &h_streams,
        Some(nextn.hnorm.as_ref().expect("DeepSeek V4 nextn hnorm")),
        cfg.rms_norm_eps,
        None,
    );

    // 2. Input projection into the packed stream: fused
    // eh_proj(cat([enorm(e) tiled, hnorm(h)])) per stream (llama.cpp GGUF
    // layout), or the separate h_proj(h) + e_proj(e) broadcast sum (raw HF
    // layout, vLLM `DeepSeekV4MultiTokenPredictorLayer`).
    let packed = if let Some(eh_proj) = nextn.eh_proj.as_ref() {
        let e_tiled = broadcast_to(
            &reshape(&e_normed, &[1, 1, 1, hidden], None),
            &[1, 1, hc, hidden],
            None,
        );
        let combined = concatenate(&[&e_tiled, &h_normed], -1, None);
        let combined = reshape(&combined, &[hc, 2 * hidden], None);
        reshape(&qw(&combined, eh_proj), &[1, 1, hc * hidden], None)
    } else {
        let e_proj = nextn.e_proj.as_ref().expect("DeepSeek V4 nextn e_proj");
        let h_proj = nextn.h_proj.as_ref().expect("DeepSeek V4 nextn h_proj");
        let e_part = reshape(&qw(&e_normed, e_proj), &[1, hidden], None);
        let h_part = qw(&reshape(&h_normed, &[hc, hidden], None), h_proj);
        let summed = add(&h_part, &e_part, None);
        reshape(&summed, &[1, 1, hc * hidden], None)
    };

    // 3. One full V4 block at layer index `num_hidden_layers`: hc pre/post,
    // raw-path attention (compress_ratio is 0 out of range, so the
    // compressor/indexer never engage), learned-router MoE — `token_ids` is
    // `None` and the block carries no tid2eid table, so the hash path is
    // structurally unreachable (llama.cpp asserts the same).
    let packed = deepseek_v4_family::layer_forward(
        cfg,
        layer,
        &packed,
        cache,
        cfg.layer_count,
        token_offset,
        None,
        None,
    );
    cache.advance(1);
    packed
}

/// Collapse the nextn block's packed output and apply the shared head:
/// `hc_head` → shared-head RMSNorm (nextn-specific or the root final norm) →
/// shared LM head (nextn-specific or root) — llama.cpp deepseek4.cpp:1528-1544.
///
/// Prefer the **MTP-layer** `hc_head_*` when present (vLLM
/// `DeepSeekV4MultiTokenPredictorLayer`); fall back to the target root head
/// only for legacy packs that omit it.
///
/// Returns f32 logits `[vocab_size]` ready for argmax / sampling.
pub fn deepseek_v4_mtp_hidden_to_logits(
    packed_hidden: &MlxArray,
    nextn: &DeepseekV4NextnWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> MlxArray {
    let head_w = nextn.hc_head.as_ref().unwrap_or_else(|| {
        static WARNED: OnceLock<()> = OnceLock::new();
        WARNED.get_or_init(|| {
            tracing::warn!(
                "DeepSeek V4 MTP draft using target root hc_head_* — mtp.N.hc_head_* missing; acceptance may be degraded"
            );
        });
        weights
            .deepseek_v4_head
            .as_ref()
            .expect("DeepSeek V4 head weights (hc_head_*)")
    });
    let hidden = deepseek_v4_family::collapse_for_head(cfg, head_w, packed_hidden);
    let norm_w = nextn
        .shared_head_norm
        .as_ref()
        .unwrap_or(&weights.final_norm);
    let normed = rms_norm(&hidden, Some(norm_w), cfg.rms_norm_eps, None);
    let head = nextn.shared_head_head.as_ref().unwrap_or(&weights.lm_head);
    let logits = qw(&normed, head);
    let logits_f32 = astype(&logits, MlxDtype::Float32, None);
    reshape(&logits_f32, &[cfg.vocab_size as i32], None)
}

/// Draft up to `max_depth` (≤ `num_nextn_predict_layers` = 1) tokens using the
/// DeepSeek V4 nextn block.
///
/// Returns `(draft_tokens, draft_log_probs, draft_distributions, added, top2_margins)`.
/// Mirrors [`glm_mtp_draft_tokens`]; returns empty when
/// `weights.deepseek_v4_nextn` is `None` or the block layer is absent.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_mtp_draft_tokens(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    rng: &mut Xorshift64,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // Legacy convenience: process-env draft mode + no think context (T=0.7
    // stochastic / T=1.0 greedy). Prefer the gated form with
    // [`deepseek_v4_mtp_sample_and_log_temperature`] from the runner.
    let draft_temperature = deepseek_v4_mtp_sample_and_log_temperature_from_env(false, 0.0);
    deepseek_v4_mtp_draft_tokens_gated(
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
        draft_temperature,
    )
}

/// Like [`deepseek_v4_mtp_draft_tokens`], but first threads the nextn block
/// through `forced_prefix` (real cache appends, correctly incrementing RoPE
/// offsets) before drafting the tail. Mirrors
/// [`glm_mtp_draft_tokens_after_forced_prefix`] for the hybrid n-gram+MTP path.
///
/// `draft_temperature` must match the temperature used for accept-path
/// log-prob rescale (see [`deepseek_v4_mtp_sample_and_log_temperature`]).
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_mtp_draft_tokens_after_forced_prefix(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    forced_prefix: &[u32],
    cache: &mut MlxKVCache,
    max_tail_depth: usize,
    rng: &mut Xorshift64,
    // Same gate the pure-MTP path resolved for this request. `None` falls
    // back to the process-global profile resolver (tests / legacy callers).
    min_confidence: Option<f32>,
    draft_temperature: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(nextn) = weights.deepseek_v4_nextn.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    let min_confidence = min_confidence.unwrap_or_else(|| {
        resolve_mtp_draft_min_confidence(
            crate::speculation_profile::speculation_profile_from_env(),
            None,
        )
    });
    if forced_prefix.is_empty() {
        return deepseek_v4_mtp_draft_tokens_gated(
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            Some(max_tail_depth),
            rng,
            min_confidence,
            draft_temperature,
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
        prev_hidden = deepseek_v4_mtp_head_forward(
            nextn,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
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
    let (draft, log_probs, distributions, tail_added, top2_margins) =
        deepseek_v4_mtp_draft_tokens_gated(
            weights,
            cfg,
            &prev_hidden,
            last_forced,
            cache,
            Some(max_tail_depth),
            rng,
            min_confidence,
            draft_temperature,
        );

    (
        draft,
        log_probs,
        distributions,
        forced_prefix.len().saturating_add(tail_added),
        top2_margins,
    )
}

/// Like [`deepseek_v4_mtp_draft_tokens`] but with an explicit draft-confidence gate.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_mtp_draft_tokens_gated(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth_cap: Option<usize>,
    _rng: &mut Xorshift64,
    min_confidence: f32,
    draft_temperature: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    let Some(nextn) = weights.deepseek_v4_nextn.as_ref() else {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    };
    if nextn.layer.is_none() {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    }
    let max_depth = deepseek_v4_mtp_max_depth(cfg).min(max_depth_cap.unwrap_or(usize::MAX));
    if max_depth == 0 {
        return (vec![], vec![], vec![], 0, [0.0; 3]);
    }

    let vocab = cfg.vocab_size as i32;
    let draft_mode = mtp_draft_mode_from_env();
    let gate_forces_greedy = min_confidence > 0.0 && draft_mode != MtpDraftMode::Stochastic;

    let result = if gate_forces_greedy || draft_mode == MtpDraftMode::Greedy {
        // Greedy argmax; log-probs at T=1.0 (no draft-sampler config ships
        // with the V4 artifact — see the GLM gated dispatcher for the split).
        deepseek_v4_mtp_draft_tokens_greedy(
            nextn,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
            1.0,
        )
    } else {
        deepseek_v4_mtp_draft_tokens_stochastic(
            nextn,
            weights,
            cfg,
            first_hidden,
            first_token,
            cache,
            max_depth,
            vocab,
            draft_temperature,
        )
    };

    let appended = result.3;
    let gated = apply_draft_confidence_gate(result, min_confidence);
    let dropped = appended.saturating_sub(gated.3);
    if dropped > 0 {
        let target = cache.seq_len().saturating_sub(dropped);
        if !cache.trim_to(target) {
            // Same contract as the GLM path: output stays correct (every
            // draft is verified), so warn rather than fail.
            tracing::warn!(target, "MTP confidence-gate trim refused");
        }
    }
    gated
}

/// Greedy V4 nextn draft: lazy argmax across all depths, single batch eval.
#[allow(clippy::too_many_arguments)]
fn deepseek_v4_mtp_draft_tokens_greedy(
    nextn: &DeepseekV4NextnWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    log_prob_temperature: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
    // Imperative path like GLM: a single nextn block at depth ≤1 — a compiled
    // head would fuse nothing of consequence on these bandwidth-bound models.
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
        let new_hidden = deepseek_v4_mtp_head_forward(
            nextn,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let logits = deepseek_v4_mtp_hidden_to_logits(&new_hidden, nextn, weights, cfg);
        let lazy_tok = lazy_argmax_logits(&logits);
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, log_prob_temperature, vocab);
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

#[cfg(test)]
mod qwen_mtp_log_prob_temperature_tests {
    use super::{
        MtpDraftMode, greedy_draft_needs_temperature_log_probs, qwen_mtp_draft_log_prob_temperature,
    };

    #[test]
    fn exact_profile_zero_gate_records_t1_not_head_0_7() {
        // Qwen3.6 linear exact production path: greedy drafts, no confidence
        // gate → pure greedy log-probs at T=1.0. Accept must not use head 0.7.
        assert!(!greedy_draft_needs_temperature_log_probs(0.7, 0.0, true));
        assert_eq!(
            qwen_mtp_draft_log_prob_temperature(MtpDraftMode::Greedy, 0.7, 0.0, true),
            1.0
        );
    }

    #[test]
    fn confidence_gate_force_greedy_records_t1() {
        assert_eq!(
            qwen_mtp_draft_log_prob_temperature(MtpDraftMode::Greedy, 0.7, 0.5, false),
            1.0
        );
    }

    #[test]
    fn stochastic_records_head_temperature() {
        assert_eq!(
            qwen_mtp_draft_log_prob_temperature(MtpDraftMode::Stochastic, 0.7, 0.0, true),
            0.7
        );
    }

    #[test]
    fn non_exact_greedy_with_head_t_uses_temperature_log_probs() {
        assert!(greedy_draft_needs_temperature_log_probs(0.7, 0.0, false));
        assert_eq!(
            qwen_mtp_draft_log_prob_temperature(MtpDraftMode::Greedy, 0.7, 0.0, false),
            0.7
        );
    }
}

#[cfg(test)]
mod deepseek_v4_think_gate_tests {
    use super::{
        DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE, MtpDraftMode, deepseek_v4_mtp_effective_draft_temperature,
        deepseek_v4_mtp_sample_and_log_temperature,
    };

    #[test]
    fn inside_think_matches_target_temperature() {
        assert_eq!(deepseek_v4_mtp_effective_draft_temperature(true, 1.0), 1.0);
    }

    #[test]
    fn outside_think_keeps_tuned_default() {
        assert_eq!(
            deepseek_v4_mtp_effective_draft_temperature(false, 1.0),
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE
        );
    }

    #[test]
    fn sharp_target_sampling_keeps_default_inside_think() {
        assert_eq!(
            deepseek_v4_mtp_effective_draft_temperature(true, 0.6),
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE
        );
    }

    /// Simulate a think-boundary step: pre-result was inside think, result
    /// closed the block. Next-draft sample+log T must drop to the tuned
    /// default; accept for the *current* pending batch keeps the stored T.
    #[test]
    fn sample_and_log_temperature_tracks_think_boundary() {
        let target = 1.0;
        let pre_result_in_think = true;
        let post_result_in_think = false; // result contained </think>
        let draft_t_before = deepseek_v4_mtp_sample_and_log_temperature(
            MtpDraftMode::Stochastic,
            pre_result_in_think,
            target,
        );
        let draft_t_after = deepseek_v4_mtp_sample_and_log_temperature(
            MtpDraftMode::Stochastic,
            post_result_in_think,
            target,
        );
        assert_eq!(draft_t_before, 1.0, "inside think: match target");
        assert_eq!(
            draft_t_after, DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE,
            "after </think>: next draft must leave think temperature"
        );
        // Greedy is always 1.0 on either side of the boundary.
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(
                MtpDraftMode::Greedy,
                post_result_in_think,
                target
            ),
            1.0
        );
    }
}

/// Stochastic V4 nextn draft: GPU-side `random_categorical` sampling.
#[allow(clippy::too_many_arguments)]
fn deepseek_v4_mtp_draft_tokens_stochastic(
    nextn: &DeepseekV4NextnWeights,
    weights: &ModelWeights,
    cfg: &ModelConfig,
    first_hidden: &MlxArray,
    first_token: u32,
    cache: &mut MlxKVCache,
    max_depth: usize,
    vocab: i32,
    temperature: f32,
) -> (Vec<u32>, Vec<f32>, Vec<TokenDistribution>, usize, [f32; 3]) {
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
        let new_hidden = deepseek_v4_mtp_head_forward(
            nextn,
            &prev_hidden,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            None,
        );
        let logits = deepseek_v4_mtp_hidden_to_logits(&new_hidden, nextn, weights, cfg);
        let lazy_tok = if temperature > 0.0 {
            lazy_random_sample(&logits, temperature, vocab)
        } else {
            lazy_argmax_logits(&logits)
        };
        lazy_tokens.push(lazy_tok.clone());
        // Log-prob must use the same temperature as sampling so q(token) matches
        // the proposal distribution used for rejection sampling (was max(T,1.0)).
        let log_prob_t = if temperature > 0.0 { temperature } else { 1.0 };
        let lazy_lp = gpu_draft_log_prob_lazy(&logits, &lazy_tok, log_prob_t, vocab);
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

/// Result of sequential greedy DeepSeek V4 MTP verification.
///
/// Production KV advances by `1 + accept_count` (primary always commits; same
/// plan as the runner's `deepseek_v4_mtp_committed_verify_len`) and matches
/// pure single-token greedy decode for the committed prefix.
#[derive(Clone, Debug)]
pub struct SequentialGreedyDeepseekV4MtpVerify {
    /// Leading draft tokens equal to sequential greedy production.
    pub accept_count: usize,
    /// Correction (partial reject) or bonus (full accept) token.
    pub correction_token: u32,
    /// Packed residual `[1, 1, hc*hidden]` at the last committed position —
    /// seeds the next nextn draft step.
    pub draft_hidden: MlxArray,
    /// Per-position greedy predictions (length `accept_count + 1`).
    pub predicted: Vec<u32>,
    /// Last committed position logits `[vocab]` (greedy tail uses
    /// `correction_token` directly; shape matches `forward_argmax`).
    pub last_logits: MlxArray,
}

/// Verify DeepSeek V4 MTP drafts with **singleton** target forwards.
///
/// Multi-token teacher-forced verify (`deepseek_v4_forward_all_positions_with_packed`
/// over `[last] ++ drafts`) can disagree with pure single-token greedy on
/// compressor / latent-K sliding state — enough to flip an argmax and break
/// Tier-2 exactness after a full draft accept. This path mirrors direct decode:
/// one token at a time, same production cache, packed residual captured for
/// the next nextn draft.
///
/// Token decisions come from singleton
/// [`deepseek_v4_forward_all_positions_with_packed`](crate::model::deepseek_v4_forward_all_positions_with_packed)
/// (same trunk as production [`forward_argmax`](crate::model::forward_argmax);
/// bf16→f32 cast is exact and softcap is monotonic, so argmax matches). The
/// packed residual is the nextn `h` input that `forward_argmax` alone cannot
/// provide.
///
/// On entry `cache.seq_len()` must equal `token_offset`. On exit the cache has
/// advanced by `1 + accept_count`.
pub fn sequential_greedy_deepseek_v4_mtp_verify(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    last_token: u32,
    drafts: &[u32],
    token_offset: usize,
    draft_hidden_width: usize,
) -> SequentialGreedyDeepseekV4MtpVerify {
    use crate::model::deepseek_v4_forward_all_positions_with_packed;

    let mut predicted: Vec<u32> = Vec::with_capacity(drafts.len().saturating_add(1));
    let mut accept_count = 0usize;

    let (logits, packed) = deepseek_v4_forward_all_positions_with_packed(
        cfg,
        weights,
        &[last_token],
        cache,
        token_offset,
    );
    cache.advance(1);
    let pred_arr = argmax(&logits, None);
    {
        let kv_refs = cache.collect_eval_refs();
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
        targets.push(&pred_arr);
        targets.push(&packed);
        targets.extend(kv_refs);
        eval(&targets);
    }
    let mut next_tok = pred_arr.data_u32().first().copied().unwrap_or(0);
    predicted.push(next_tok);
    let mut draft_hidden = slice_packed_hidden_row(&packed, 0, draft_hidden_width);
    let mut last_logits = reshape_singleton_vocab_logits(&logits, cfg.vocab_size);

    for (index, &draft) in drafts.iter().enumerate() {
        if next_tok != draft {
            break;
        }
        accept_count += 1;
        let (logits, packed) = deepseek_v4_forward_all_positions_with_packed(
            cfg,
            weights,
            &[draft],
            cache,
            token_offset + 1 + index,
        );
        cache.advance(1);
        let pred_arr = argmax(&logits, None);
        {
            let kv_refs = cache.collect_eval_refs();
            let mut targets: Vec<&MlxArray> = Vec::with_capacity(2 + kv_refs.len());
            targets.push(&pred_arr);
            targets.push(&packed);
            targets.extend(kv_refs);
            eval(&targets);
        }
        next_tok = pred_arr.data_u32().first().copied().unwrap_or(0);
        predicted.push(next_tok);
        draft_hidden = slice_packed_hidden_row(&packed, 0, draft_hidden_width);
        last_logits = reshape_singleton_vocab_logits(&logits, cfg.vocab_size);
    }

    SequentialGreedyDeepseekV4MtpVerify {
        accept_count,
        correction_token: next_tok,
        draft_hidden,
        predicted,
        last_logits,
    }
}

/// Slice one packed residual row to `[1, 1, width]` (DeepSeek nextn draft input).
fn slice_packed_hidden_row(packed: &MlxArray, pos: usize, width: usize) -> MlxArray {
    let p = pos as i32;
    let w = width as i32;
    let shape = packed.shape();
    // Single-token forward returns `[1, 1, hc*hidden]`; multi-row history is
    // `[1, seq, hc*hidden]`. Both share the last axis width.
    if shape.len() == 3 && shape[1] == 1 {
        if shape[2] == w {
            return packed.clone();
        }
        return reshape(packed, &[1, 1, w], None);
    }
    let row = slice(packed, &[0, p, 0], &[1, p + 1, w], &[1, 1, 1], None);
    reshape(&row, &[1, 1, w], None)
}

/// Normalize packed-path logits `[1, vocab]` (or already `[vocab]`) to
/// production `[vocab]` shape expected by the shared MTP tail sampler.
fn reshape_singleton_vocab_logits(logits: &MlxArray, vocab_size: usize) -> MlxArray {
    let shape = logits.shape();
    if shape == [vocab_size as i32] {
        return logits.clone();
    }
    reshape(logits, &[vocab_size as i32], None)
}

/// Warm the DeepSeek V4 nextn KV cache from prompt-side packed residuals.
///
/// Mirrors [`mtp_warmup_cache_kv_batched`] for the Qwen head: without this the
/// nextn attention starts decode with almost no prompt history and acceptance
/// collapses. `packed_hidden_seq` is `[1, seq, hc*hidden]` aligned with
/// `prev_tokens` (token that *follows* each packed row, same contract as Qwen).
pub fn deepseek_v4_mtp_warmup_cache(
    nextn: &DeepseekV4NextnWeights,
    packed_hidden_seq: &MlxArray,
    prev_tokens: &[u32],
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    cfg: &ModelConfig,
    rope_offset: usize,
) {
    if prev_tokens.is_empty() || nextn.layer.is_none() {
        return;
    }
    let seq = prev_tokens.len();
    let shape = packed_hidden_seq.shape();
    let avail = shape.get(1).copied().unwrap_or(0).max(0) as usize;
    let n = seq.min(avail);
    if n == 0 {
        return;
    }
    let width = shape.get(2).copied().unwrap_or(0);
    for (i, &token) in prev_tokens.iter().enumerate().take(n) {
        let packed_row = slice(
            packed_hidden_seq,
            &[0, i as i32, 0],
            &[1, (i + 1) as i32, width],
            &[1, 1, 1],
            None,
        );
        let packed_row = reshape(&packed_row, &[1, 1, width], None);
        let tok = [token];
        let prev_token_arr =
            MlxArray::from_raw_data(tok.as_ptr() as *const u8, 4, &[1_i32], MlxDtype::Uint32);
        let _ = deepseek_v4_mtp_head_forward(
            nextn,
            &packed_row,
            &prev_token_arr,
            weights,
            cache,
            cfg,
            Some(rope_offset + i),
        );
    }
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
    fn disabled_gate_does_not_require_draft_log_probs() {
        let (tokens, log_probs, _distributions, added, _accept3) =
            apply_draft_confidence_gate((vec![1, 2], vec![], vec![], 0, [0.0; 3]), 0.0);
        assert_eq!(tokens, vec![1, 2]);
        assert!(log_probs.is_empty());
        assert_eq!(added, 2);
    }

    #[test]
    fn exact_profile_skips_unused_temperature_log_probs_only_with_gate_disabled() {
        assert!(!greedy_draft_needs_temperature_log_probs(0.7, 0.0, true));
        assert!(greedy_draft_needs_temperature_log_probs(0.7, 0.1, true));
        assert!(greedy_draft_needs_temperature_log_probs(0.7, 0.0, false));
        assert!(!greedy_draft_needs_temperature_log_probs(0.0, 0.0, false));
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

// -------------------------------------------------------------------------
// DeepSeek V4 nextn (MTP) tests — tiny synthetic block, compile + shape/value
// checks only (run on hardware elsewhere).
// -------------------------------------------------------------------------
#[cfg(test)]
mod deepseek_v4_mtp_tests {
    use super::*;
    use crate::weights::{
        DeepseekV4HeadWeights, DeepseekV4LayerWeights, LayerWeights, QuantizedWeight,
    };
    use mlx_sys::eval;

    #[test]
    fn draft_log_prob_temperature_is_mode_aware_not_always_0_7() {
        // Regression (skeptic): accept-path must not hard-code 0.7 for greedy.
        // Greedy drafts record log-probs at T=1.0 (deepseek_v4_mtp_draft_tokens_greedy).
        assert_eq!(
            deepseek_v4_mtp_draft_log_prob_temperature_for_mode(MtpDraftMode::Greedy),
            1.0,
            "greedy nextn log-probs are at T=1.0; accept rescale must match"
        );
        assert_eq!(
            deepseek_v4_mtp_draft_log_prob_temperature_for_mode(MtpDraftMode::Stochastic),
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE,
            "stochastic nextn samples and logs at DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE"
        );
        assert!(
            (DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE - 0.7).abs() < 1e-6,
            "stochastic constant must stay 0.7"
        );
        // Process env default is greedy → helper must return 1.0 under default mode.
        assert_eq!(
            deepseek_v4_mtp_draft_log_prob_temperature_for_mode(mtp_draft_mode_from_env()),
            deepseek_v4_mtp_draft_log_prob_temperature()
        );
    }

    /// DI-DS-MTP: draft sampling T and accept log-prob T must stay locked.
    #[test]
    fn sample_and_log_temperature_matches_draft_and_accept() {
        // Greedy always 1.0 even when think would raise effective sample T.
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(MtpDraftMode::Greedy, true, 1.0),
            1.0
        );
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(MtpDraftMode::Greedy, false, 0.7),
            1.0
        );
        // Stochastic outside think: tuned 0.7 for both sample and log.
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(MtpDraftMode::Stochastic, false, 1.0),
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE
        );
        // Stochastic inside think at T>=1: sample and log both 1.0 (not 0.7).
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(MtpDraftMode::Stochastic, true, 1.0),
            1.0,
            "think-mode stochastic must not accept-rescale as if drafts were at 0.7"
        );
        // Stochastic think with cooler target keeps tuned default.
        assert_eq!(
            deepseek_v4_mtp_sample_and_log_temperature(MtpDraftMode::Stochastic, true, 0.6),
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE
        );
    }

    // Tiny synthetic dims: E=8, D=4, H=1, G=1, R_o=2, rot=2, R_q=4, HC=4.
    const E: usize = 8;
    const D: usize = 4;
    const H: usize = 1;
    const G: usize = 1;
    const R_O: usize = 2;
    const ROT: usize = 2;
    const R_Q: usize = 4;
    const HC: usize = 4;
    const N_EXPERTS: usize = 2;
    const TOP_K: usize = 1;
    const INTER: usize = 4;
    const VOCAB: usize = 8;

    fn array_f32(data: &[f32], shape: &[i32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            shape,
            MlxDtype::Float32,
        )
    }

    /// Deterministic pseudo-random fill (no external deps).
    fn fill(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5)
            .collect()
    }

    fn token_arr(token: u32) -> MlxArray {
        let data = [token];
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data),
            &[1_i32],
            MlxDtype::Uint32,
        )
    }

    fn test_v4_config() -> crate::model::DeepseekV4Config {
        crate::model::DeepseekV4Config {
            head_dim: D,
            qk_rope_head_dim: ROT,
            q_lora_rank: R_Q,
            o_lora_rank: R_O,
            o_groups: G,
            index_topk: 8,
            index_n_heads: 1,
            index_head_dim: 4,
            compress_rope_theta: 50000.0,
            compress_rope_scaling: None,
            has_attn_sinks: true,
            compress_ratios: vec![0],
            hc_mult: HC,
            hc_sinkhorn_iters: 3,
            hc_eps: 1e-5,
            // The nextn block must sit beyond the hash layers; the test block
            // carries the learned router only.
            num_hash_layers: 0,
            num_nextn_predict_layers: 1,
            scoring_func: Some("sqrtsoftplus".to_string()),
            swiglu_limit: 7.0,
        }
    }

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            compile_cache_identity: 1,
            model_family: "deepseek_v4".to_string(),
            layer_count: 1,
            hidden_size: E,
            intermediate_size: INTER,
            n_heads: H,
            n_kv_heads: 1,
            head_dim: D,
            vocab_size: VOCAB,
            rope_theta: 10000.0,
            rope_dims: ROT,
            attn_output_gate: false,
            query_scale: 1.0 / (D as f32).sqrt(),
            final_logit_softcapping: None,
            moe_expert_count: N_EXPERTS,
            moe_experts_per_token: TOP_K,
            moe_expert_intermediate_size: INTER,
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
            deepseek_v4: Some(test_v4_config()),
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            rope_mscale: 1.0,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 1,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
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

    fn dense_weight(rows: usize, cols: usize, seed: f32) -> QuantizedWeight {
        QuantizedWeight::new(
            array_f32(&fill(rows * cols, seed), &[rows as i32, cols as i32]),
            None,
            None,
        )
    }

    fn hc_branch_weights(mixes: usize, seed: f32) -> (MlxArray, MlxArray, MlxArray) {
        (
            array_f32(
                &fill(mixes * HC * E, seed),
                &[mixes as i32, (HC * E) as i32],
            ),
            array_f32(&fill(mixes, seed + 0.1), &[mixes as i32]),
            array_f32(&[1.0, 1.0, 1.0], &[3]),
        )
    }

    /// Nextn block layer: learned sqrtsoftplus router (correction bias), never
    /// a tid2eid hash table.
    fn test_nextn_layer_weights() -> LayerWeights {
        let mixes = 2 * HC + HC * HC;
        let (hc_attn_fn, hc_attn_base, hc_attn_scale) = hc_branch_weights(mixes, 0.31);
        let (hc_ffn_fn, hc_ffn_base, hc_ffn_scale) = hc_branch_weights(mixes, 0.47);
        LayerWeights {
            attn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
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
            deepseek_v4: Some(DeepseekV4LayerWeights {
                wq_a: dense_weight(R_Q, E, 0.11),
                q_a_norm: array_f32(&fill(R_Q, 0.8), &[R_Q as i32]),
                wq_b: dense_weight(H * D, R_Q, 0.13),
                wkv: dense_weight(D, E, 0.17),
                kv_norm: array_f32(&fill(D, 0.8), &[D as i32]),
                wo_a: dense_weight(G * R_O, H * D / G, 0.19),
                wo_b: dense_weight(E, G * R_O, 0.23),
                attn_sink: Some(array_f32(&[-1.0], &[H as i32])),
                hc_attn_fn,
                hc_attn_base,
                hc_attn_scale,
                hc_ffn_fn,
                hc_ffn_base,
                hc_ffn_scale,
                compressor: None,
                indexer: None,
                tid2eid: None,
            }),
            ffn_norm: array_f32(&fill(E, 0.9), &[E as i32]),
            ffn_post_norm: None,
            gate_proj: None,
            up_proj: None,
            gate_up_packed: None,
            down_proj: None,
            ffn_norm2: None,
            ffn_post_norm1: None,
            ffn_post_norm2: None,
            router_proj: Some(dense_weight(N_EXPERTS, E, 0.29)),
            router_correction_bias: Some(array_f32(&fill(N_EXPERTS, 0.05), &[N_EXPERTS as i32])),
            router_scale: None,
            router_combined_scale: None,
            router_expert_scale: None,
            layer_scalar: None,
            per_layer_gate: None,
            per_layer_proj_w: None,
            per_layer_post_norm: None,
            shared_expert_gate: None,
            shared_gate_up_proj: None,
            shared_gate_proj: Some(dense_weight(INTER, E, 0.37)),
            shared_up_proj: Some(dense_weight(INTER, E, 0.41)),
            shared_down_proj: Some(dense_weight(E, INTER, 0.43)),
            gate_up_exps_packed: None,
            gate_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * INTER * E, 0.53),
                    &[N_EXPERTS as i32, INTER as i32, E as i32],
                ),
                None,
                None,
            )),
            up_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * INTER * E, 0.59),
                    &[N_EXPERTS as i32, INTER as i32, E as i32],
                ),
                None,
                None,
            )),
            down_exps: Some(QuantizedWeight::new(
                array_f32(
                    &fill(N_EXPERTS * E * INTER, 0.61),
                    &[N_EXPERTS as i32, E as i32, INTER as i32],
                ),
                None,
                None,
            )),
            mxfp4_gate_up_exps: None,
            mxfp4_down_exps: None,
            attn_sink: None,
            rotation_smoothing_inverse: None,
        }
    }

    fn test_nextn_weights() -> DeepseekV4NextnWeights {
        DeepseekV4NextnWeights {
            e_proj: None,
            h_proj: None,
            eh_proj: Some(dense_weight(E, 2 * E, 0.71)),
            enorm: Some(array_f32(&fill(E, 0.73), &[E as i32])),
            hnorm: Some(array_f32(&fill(E, 0.79), &[E as i32])),
            // Exercise the root fallbacks (final norm / lm_head / shared
            // embedding table) rather than nextn-specific tensors.
            shared_head_norm: None,
            embed_tokens: None,
            shared_head_head: None,
            // Dedicated MTP HC head (distinct from target root head).
            hc_head: Some(crate::weights::DeepseekV4HeadWeights {
                hc_head_fn: array_f32(&fill(HC * HC * E, 0.31), &[HC as i32, (HC * E) as i32]),
                hc_head_base: array_f32(&fill(HC, 0.32), &[HC as i32]),
                hc_head_scale: array_f32(&[1.0], &[1]),
            }),
            layer: Some(Box::new(test_nextn_layer_weights())),
        }
    }

    fn test_model_weights(nextn: Option<DeepseekV4NextnWeights>) -> ModelWeights {
        ModelWeights {
            token_embedding: dense_weight(VOCAB, E, 0.83),
            final_norm: array_f32(&fill(E, 0.89), &[E as i32]),
            lm_head: dense_weight(VOCAB, E, 0.97),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            deepseek_v4_head: Some(DeepseekV4HeadWeights {
                hc_head_fn: array_f32(&fill(HC * HC * E, 0.21), &[HC as i32, (HC * E) as i32]),
                hc_head_base: array_f32(&fill(HC, 0.22), &[HC as i32]),
                hc_head_scale: array_f32(&[1.0], &[1]),
            }),
            deepseek_v4_nextn: nextn,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            gemma4_vl_vision: None,
            diffusion_self_conditioning: None,
            unlimited_ocr_vision: None,
            qwen3_vl_vision: None,
            minicpm_v46_vision: None,
            nemotron_omni: None,
        }
    }

    fn packed_hidden(seed: f32) -> MlxArray {
        array_f32(&fill(HC * E, seed), &[1, 1, (HC * E) as i32])
    }

    #[test]
    fn head_forward_shapes_and_two_step_chaining() {
        let cfg = test_model_config();
        let weights = test_model_weights(Some(test_nextn_weights()));
        let nextn = weights.deepseek_v4_nextn.as_ref().expect("nextn");
        let mut cache = MlxKVCache::new(deepseek_v4_mtp_cache_layer_count(&cfg));

        let h0 = packed_hidden(0.67);
        let out1 = deepseek_v4_mtp_head_forward(
            nextn,
            &h0,
            &token_arr(3),
            &weights,
            &mut cache,
            &cfg,
            None,
        );
        eval(&[&out1]);
        assert_eq!(out1.shape(), vec![1, 1, (HC * E) as i32]);
        assert!(out1.data_f32().iter().all(|v| v.is_finite()));
        assert_eq!(cache.seq_len(), 1);

        // Chain: the block's own packed output is the next step's `h` input
        // (llama.cpp `t_h_nextn`).
        let out2 = deepseek_v4_mtp_head_forward(
            nextn,
            &out1,
            &token_arr(5),
            &weights,
            &mut cache,
            &cfg,
            None,
        );
        eval(&[&out2]);
        assert_eq!(out2.shape(), vec![1, 1, (HC * E) as i32]);
        assert!(out2.data_f32().iter().all(|v| v.is_finite()));
        assert_eq!(cache.seq_len(), 2);
    }

    fn test_target_weights() -> ModelWeights {
        let mut weights = test_model_weights(Some(test_nextn_weights()));
        // Main stack: one raw-path V4 layer (compress_ratios[0] == 0).
        // Nextn-only fixtures leave layers empty and cannot exercise the
        // singleton target trunk used by production greedy decode.
        weights.layers = vec![test_nextn_layer_weights()];
        weights
    }

    fn prefill_prompt(cfg: &ModelConfig, weights: &ModelWeights, prompt: &[u32]) -> MlxKVCache {
        use crate::model::forward_argmax;
        let mut cache = MlxKVCache::new(cfg.layer_count);
        for (i, &tok) in prompt.iter().enumerate() {
            let logits = forward_argmax(cfg, weights, &[tok], &mut cache, i);
            cache.advance(1);
            eval(&[&logits]);
        }
        cache
    }

    /// Singleton packed verify path must match production `forward_argmax`
    /// token decisions while also returning the packed residual nextn needs.
    ///
    /// Sequential greedy MTP verify then matches pure single-token greedy at
    /// the verify boundary (accept + bonus / reject + correction). This is
    /// the correctness contract that multi-token teacher-forced clone-adopt
    /// violated on real compressor/indexer stacks.
    #[test]
    fn sequential_greedy_verify_matches_singleton_direct_decode() {
        use crate::model::{deepseek_v4_forward_all_positions_with_packed, forward_argmax};

        let cfg = test_model_config();
        let weights = test_target_weights();
        assert_eq!(
            weights.layers.len(),
            cfg.layer_count,
            "regression must exercise the real target layer stack"
        );
        let prompt: Vec<u32> = vec![1, 2, 3, 4];
        let width = HC * E;
        // Runner commit plan: primary always lands in KV (same as
        // deepseek_v4_mtp_committed_verify_len in runner/mod.rs).
        let commit_len = |accept_count: usize, draft_len: usize| 1 + accept_count.min(draft_len);

        // --- (1) Singleton packed argmax == forward_argmax token decision ---
        let mut pack_cache = prefill_prompt(&cfg, &weights, &prompt);
        let mut argmax_cache = prefill_prompt(&cfg, &weights, &prompt);
        let probe = 5u32 % VOCAB as u32;
        let offset = pack_cache.seq_len();
        let (packed_logits, packed_h) = deepseek_v4_forward_all_positions_with_packed(
            &cfg,
            &weights,
            &[probe],
            &mut pack_cache,
            offset,
        );
        let direct_logits = forward_argmax(&cfg, &weights, &[probe], &mut argmax_cache, offset);
        let pack_pred = argmax(&packed_logits, None);
        let direct_pred = argmax(&direct_logits, None);
        eval(&[&pack_pred, &direct_pred, &packed_h]);
        assert_eq!(
            pack_pred.data_u32()[0],
            direct_pred.data_u32()[0],
            "singleton packed verify logits must match production forward_argmax"
        );
        assert_eq!(
            packed_h.shape(),
            vec![1, 1, width as i32],
            "packed residual must keep nextn width hc*hidden"
        );

        // --- (2) Pure direct stream after feeding first_gen ---
        let mut pure = prefill_prompt(&cfg, &weights, &prompt);
        let first_gen = probe;
        let mut pure_stream = Vec::new();
        let mut last = first_gen;
        for _ in 0..3 {
            let off = pure.seq_len();
            let logits = forward_argmax(&cfg, &weights, &[last], &mut pure, off);
            pure.advance(1);
            let pred = argmax(&logits, None);
            eval(&[&pred]);
            let tok = pred.data_u32()[0];
            pure_stream.push(tok);
            last = tok;
        }
        // pure_stream = [t1, t2, t3] after feeding first_gen, t1, t2.

        // --- (3) Sequential MTP accept path: draft = [t1] → accept, bonus = t2 ---
        let mut mtp = prefill_prompt(&cfg, &weights, &prompt);
        let token_offset = mtp.seq_len();
        let draft = vec![pure_stream[0]];
        let seq = sequential_greedy_deepseek_v4_mtp_verify(
            &cfg,
            &weights,
            &mut mtp,
            first_gen,
            &draft,
            token_offset,
            width,
        );
        assert_eq!(seq.accept_count, 1, "oracle draft must fully accept");
        assert_eq!(
            mtp.seq_len(),
            token_offset + commit_len(1, 1),
            "full accept commits primary + accepted draft"
        );
        let mut emitted = draft[..seq.accept_count].to_vec();
        emitted.push(seq.correction_token);
        assert_eq!(
            emitted,
            pure_stream[..2],
            "accepted draft + bonus must equal pure singleton greedy"
        );
        assert_eq!(seq.draft_hidden.shape(), vec![1, 1, width as i32]);
        assert_eq!(
            seq.last_logits.shape(),
            vec![VOCAB as i32],
            "last_logits must be production [vocab] shape"
        );
        assert_eq!(seq.predicted, pure_stream[..2]);

        // --- (4) Old multi-token teacher-forced oracle at the same boundary ---
        // Accept decisions from multi-token predicted rows. On this raw-path
        // synthetic stack they typically match sequential; the sequential
        // path is still required on real CSA/HCA layers where multi-token
        // adopt diverges. Pin that sequential remains ground truth vs pure.
        let mut mt_cache = prefill_prompt(&cfg, &weights, &prompt);
        let verify_input = vec![first_gen, pure_stream[0]];
        let (mt_logits, _mt_packed) = deepseek_v4_forward_all_positions_with_packed(
            &cfg,
            &weights,
            &verify_input,
            &mut mt_cache,
            token_offset,
        );
        let mt_pred = argmax(&mt_logits, None);
        eval(&[&mt_pred]);
        let mt_predicted = mt_pred.data_u32().to_vec();
        let mt_accept = crate::ngram_accel::greedy_draft_target_accept_count(&draft, &mt_predicted);
        // Sequential always matches pure; multi-token may or may not.
        assert_eq!(
            seq.accept_count,
            crate::ngram_accel::greedy_draft_target_accept_count(&draft, &seq.predicted),
            "sequential accept must follow its own singleton predictions"
        );
        if mt_accept != seq.accept_count || mt_predicted.first() != Some(&pure_stream[0]) {
            // When multi-token disagrees, sequential is the correct oracle.
            assert_eq!(
                seq.accept_count, 1,
                "when multi-token diverges, sequential still accepts oracle draft"
            );
            assert_eq!(seq.correction_token, pure_stream[1]);
        }

        // --- (5) Reject boundary: wrong draft commits primary only ---
        let mut reject = prefill_prompt(&cfg, &weights, &prompt);
        let offset = reject.seq_len();
        let mut wrong = (pure_stream[0] + 1) % VOCAB as u32;
        if wrong == pure_stream[0] {
            wrong = (wrong + 1) % VOCAB as u32;
        }
        let seq_rej = sequential_greedy_deepseek_v4_mtp_verify(
            &cfg,
            &weights,
            &mut reject,
            first_gen,
            &[wrong],
            offset,
            width,
        );
        assert_eq!(seq_rej.accept_count, 0, "wrong draft must fully reject");
        assert_eq!(
            reject.seq_len(),
            offset + commit_len(0, 1),
            "full reject still commits primary last_token"
        );
        assert_eq!(
            seq_rej.correction_token, pure_stream[0],
            "correction must equal singleton greedy at the reject position"
        );
        assert_eq!(seq_rej.predicted, vec![pure_stream[0]]);
    }

    #[test]
    fn head_forward_is_deterministic() {
        let cfg = test_model_config();
        let weights = test_model_weights(Some(test_nextn_weights()));
        let nextn = weights.deepseek_v4_nextn.as_ref().expect("nextn");
        let run = || {
            let mut cache = MlxKVCache::new(deepseek_v4_mtp_cache_layer_count(&cfg));
            let out = deepseek_v4_mtp_head_forward(
                nextn,
                &packed_hidden(0.67),
                &token_arr(3),
                &weights,
                &mut cache,
                &cfg,
                None,
            );
            eval(&[&out]);
            out.data_f32().to_vec()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn hidden_to_logits_vocab_shape() {
        let cfg = test_model_config();
        let weights = test_model_weights(Some(test_nextn_weights()));
        let nextn = weights.deepseek_v4_nextn.as_ref().expect("nextn");
        let logits = deepseek_v4_mtp_hidden_to_logits(&packed_hidden(0.67), nextn, &weights, &cfg);
        eval(&[&logits]);
        assert_eq!(logits.shape(), vec![VOCAB as i32]);
        assert!(logits.data_f32().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn draft_tokens_empty_without_nextn() {
        let cfg = test_model_config();
        let weights = test_model_weights(None);
        let mut cache = MlxKVCache::new(deepseek_v4_mtp_cache_layer_count(&cfg));
        let mut rng = Xorshift64::new(1);
        let (draft, log_probs, dist, added, margins) = deepseek_v4_mtp_draft_tokens(
            &weights,
            &cfg,
            &packed_hidden(0.67),
            3,
            &mut cache,
            None,
            &mut rng,
        );
        assert!(draft.is_empty());
        assert!(log_probs.is_empty());
        assert!(dist.is_empty());
        assert_eq!(added, 0);
        assert_eq!(margins, [0.0; 3]);
    }

    #[test]
    fn draft_tokens_depth_one_then_trim_and_redraft() {
        let cfg = test_model_config();
        let weights = test_model_weights(Some(test_nextn_weights()));
        let mut cache = MlxKVCache::new(deepseek_v4_mtp_cache_layer_count(&cfg));
        let mut rng = Xorshift64::new(1);

        let (draft, log_probs, _dist, added, _m) = deepseek_v4_mtp_draft_tokens_gated(
            &weights,
            &cfg,
            &packed_hidden(0.67),
            3,
            &mut cache,
            None,
            &mut rng,
            0.0,
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE,
        );
        assert_eq!(draft.len(), 1);
        assert_eq!(log_probs.len(), 1);
        assert_eq!(added, 1);
        assert!(draft[0] < VOCAB as u32);
        assert!(log_probs[0].is_finite());
        assert_eq!(cache.seq_len(), 1);

        // Simulate a rejected draft: the cache trims back and the next step
        // re-drafts from the new committed hidden.
        assert!(cache.trim_to(0));
        let (draft2, _lp, _d, added2, _m2) = deepseek_v4_mtp_draft_tokens_gated(
            &weights,
            &cfg,
            &packed_hidden(0.71),
            4,
            &mut cache,
            None,
            &mut rng,
            0.0,
            DEEPSEEK_V4_MTP_DRAFT_TEMPERATURE,
        );
        assert_eq!(draft2.len(), 1);
        assert_eq!(added2, 1);
        assert_eq!(cache.seq_len(), 1);
    }
}
