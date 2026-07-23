use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

use ax_engine_core::gemma4_unified::Gemma4UnifiedRuntimeInputs;
use mlx_sys::{MlxArray, argmax, async_eval, clear_cache, eval};

use crate::gemma4_unified::build_chunk_embeddings;
use crate::kv_cache::MlxKVCache;
use crate::linear_attention_ops::GATED_DELTA_THREADGROUP_CACHE_CAPACITY;
use crate::model::{
    FinalLogitsMode, ModelConfig, forward, forward_all_positions_post_norm_last_lm_head,
    forward_all_positions_with_final_hidden, forward_argmax, forward_cache_only,
    forward_lazy_single_argmax, forward_with_initial_hidden_and_media_ranges,
};
use crate::sampling::{
    MlxSamplingParams, MlxSamplingRequest, Xorshift64, sample_categorical_gpu,
    sample_categorical_into, sample_categorical_with_topk_gpu, sample_categorical_with_topp_gpu,
};
use crate::unlimited_ocr::{UnlimitedOcrImageViews, build_embeddings_with_image};
use crate::weights::ModelWeights;

/// Default chunk size for chunked prefill, matching mlx-lm's default
/// `prefill_step_size` and the long GatedDelta Metal kernel specialization.
pub const DEFAULT_PREFILL_CHUNK: usize = GATED_DELTA_THREADGROUP_CACHE_CAPACITY;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectPipelineTimings {
    pub forward_wall_us: u32,
    /// Subset of `forward_wall_us`: time spent inside the 64-layer
    /// `layer_forward` loop. Zero unless
    /// `AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1` is set. The residual
    /// `forward_wall_us - layer_loop_wall_us - head_wall_us` covers
    /// embed-tokens + per-layer-input + dtype-cast + scale-hidden graph-build
    /// cost.
    pub forward_layer_loop_wall_us: u32,
    /// Subset of `forward_wall_us`: time spent in the post-layer RMSNorm,
    /// lm-head qmatmul, and logit finalize chain. Zero unless
    /// `AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE=1` is set.
    pub forward_head_wall_us: u32,
    pub argmax_wall_us: u32,
    pub async_eval_wall_us: u32,
    /// Diagnostic-only: time spent in the synchronous `eval(next_token)` barrier
    /// inserted right after `async_eval`. Zero unless `AX_MLX_DIRECT_PIPELINE_BARRIER`
    /// is set; enabling it breaks the double-buffer overlap and turns
    /// `async_eval_wall_us` into a near-pure submit cost while this bucket
    /// captures the per-step GPU-completion time.
    pub next_complete_wall_us: u32,
    pub pending_eval_wall_us: u32,
    pub pending_read_wall_us: u32,
    /// Total mlx-sys FFI op count dispatched across all Qwen linear-attention
    /// layers in the most recent forward. Divide by
    /// `linear_attention_layer_count` to get ops/layer. Zero for models without
    /// linear-attention layers.
    pub linear_attention_layer_ops: u64,
    pub linear_attention_layer_count: u32,
    /// Total mlx-sys FFI op count dispatched across all standard-SDPA layers
    /// (Gemma 4, full-attention slices of Qwen 3.6, Llama 3, etc.) in the most
    /// recent forward. Divide by `full_attention_layer_count` to get ops/layer.
    pub full_attention_layer_ops: u64,
    pub full_attention_layer_count: u32,
}

/// Per-forward thread-local accumulator that `advance_direct_pipeline_*`
/// resets-before / takes-after each direct-pipeline step. Values flow out via
/// `DirectPipelineTimings`; this struct is an internal carrier and intentionally
/// has no public accessors (a previous draft exposed
/// `forward_stage_timings_snapshot()` for trace harnesses, which was withdrawn
/// once the same data shipped through `DirectPipelineTimings`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ForwardStageTimings {
    layer_loop_wall_us: u32,
    head_wall_us: u32,
    linear_attention_layer_ops: u64,
    linear_attention_layer_count: u32,
    full_attention_layer_ops: u64,
    full_attention_layer_count: u32,
}

thread_local! {
    static FORWARD_STAGE_TIMINGS: Cell<ForwardStageTimings> = const {
        Cell::new(ForwardStageTimings {
            layer_loop_wall_us: 0,
            head_wall_us: 0,
            linear_attention_layer_ops: 0,
            linear_attention_layer_count: 0,
            full_attention_layer_ops: 0,
            full_attention_layer_count: 0,
        })
    };
}

fn reset_forward_stage_timings() {
    FORWARD_STAGE_TIMINGS.with(|cell| cell.set(ForwardStageTimings::default()));
}

fn take_forward_stage_timings() -> ForwardStageTimings {
    FORWARD_STAGE_TIMINGS.with(|cell| cell.replace(ForwardStageTimings::default()))
}

/// Record the wall time spent inside the per-layer forward loop. Called once
/// per profiled decode forward by `forward_lazy_single_*`.
pub(crate) fn record_forward_layer_loop_wall_us(us: u32) {
    FORWARD_STAGE_TIMINGS.with(|cell| {
        let mut current = cell.get();
        current.layer_loop_wall_us = current.layer_loop_wall_us.saturating_add(us);
        cell.set(current);
    });
}

/// Record the wall time spent inside the post-layer RMSNorm + lm-head chain.
pub(crate) fn record_forward_head_wall_us(us: u32) {
    FORWARD_STAGE_TIMINGS.with(|cell| {
        let mut current = cell.get();
        current.head_wall_us = current.head_wall_us.saturating_add(us);
        cell.set(current);
    });
}

/// Record the number of mlx-sys FFI ops dispatched for one layer's forward pass,
/// classified by whether the layer uses the Qwen linear-attention recurrence or
/// the standard full-attention SDPA. Counts plus invocations are aggregated per
/// forward and read by `advance_direct_pipeline_*` so direct-cpp shim targeting
/// can be driven from real per-layer-kind op distributions instead of guesswork.
pub(crate) fn record_layer_ops(is_linear_attention: bool, op_delta: u64) {
    FORWARD_STAGE_TIMINGS.with(|cell| {
        let mut current = cell.get();
        if is_linear_attention {
            current.linear_attention_layer_ops =
                current.linear_attention_layer_ops.saturating_add(op_delta);
            current.linear_attention_layer_count =
                current.linear_attention_layer_count.saturating_add(1);
        } else {
            current.full_attention_layer_ops =
                current.full_attention_layer_ops.saturating_add(op_delta);
            current.full_attention_layer_count =
                current.full_attention_layer_count.saturating_add(1);
        }
        cell.set(current);
    });
}

pub struct DirectPipelineAdvance {
    pub token: u32,
    pub next_pending: MlxArray,
    pub timings: DirectPipelineTimings,
}

fn elapsed_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

static DIRECT_PIPELINE_BARRIER_ENABLED: OnceLock<bool> = OnceLock::new();
static DIRECT_PIPELINE_STAGE_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

fn direct_pipeline_barrier_enabled() -> bool {
    *DIRECT_PIPELINE_BARRIER_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_DIRECT_PIPELINE_BARRIER").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

pub(crate) fn direct_pipeline_stage_profile_enabled() -> bool {
    *DIRECT_PIPELINE_STAGE_PROFILE_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_DIRECT_PIPELINE_STAGE_PROFILE").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Process the full prompt in chunks of `chunk_size` tokens.
///
/// Returns the sampled next-token ID from the last-token logits.
///
/// # Prefill boundary
///
/// For deterministic argmax, mirror `mlx_lm.generate_step`: prefill every
/// prompt token except the final one as
/// cache-state-only work, then run the final prompt token through the normal
/// single-token step to produce the first generated token. Evaluating only KV
/// refs lets MLX's lazy graph prune the final logits path for long prompt
/// chunks. Non-greedy sampling keeps the historical full-logits path until the
/// sampling contract is audited.
/// GPU-first sampled-token selection shared by the chunked-prefill variants:
/// exact GPU top-k sampling, then the GPU top-p candidate path, then the exact
/// CPU categorical fallback with reusable buffers. `before_cpu_fallback` runs
/// only when both GPU paths decline, and must materialise whatever the caller
/// needs evaluated before `logits.data_f32()` is read on the CPU.
#[allow(clippy::too_many_arguments)]
fn sample_prefill_token_gpu_first(
    logits: &MlxArray,
    sampling: MlxSamplingParams,
    repetition_tokens: &[u32],
    rng: &mut Xorshift64,
    before_cpu_fallback: impl FnOnce(),
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> u32 {
    if let Some(tok) = sample_categorical_with_topk_gpu(logits, sampling, repetition_tokens, rng)
        .or_else(|| sample_categorical_with_topp_gpu(logits, sampling, repetition_tokens, rng))
    {
        return tok;
    }
    before_cpu_fallback();
    let logits_data = logits.data_f32();
    sample_categorical_into(
        logits_data,
        sampling,
        repetition_tokens,
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
    )
}

pub fn chunked_prefill(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
) -> u32 {
    let mut sampling_probs_buf = Vec::new();
    let mut sampling_logits_buf = Vec::new();
    let mut sampling_candidates_buf = Vec::new();
    chunked_prefill_with_sampling_buffers(
        cfg,
        weights,
        prompt_tokens,
        cache,
        chunk_size,
        sampling_request,
        rng,
        &mut sampling_probs_buf,
        &mut sampling_logits_buf,
        &mut sampling_candidates_buf,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_prefill_with_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> u32 {
    let sampling = sampling_request.params;
    let chunk_size = chunk_size.max(1);
    let total = prompt_tokens.len();

    let cache_only_prefix_len = mlx_lm_style_cache_only_prefix_len(total, sampling);
    if cache_only_prefix_len > 0 {
        let mut offset = 0;
        while offset < cache_only_prefix_len {
            let end = (offset + chunk_size).min(cache_only_prefix_len);
            let chunk = &prompt_tokens[offset..end];
            // Cache-only: skip lm_head projection (hidden×vocab_size).
            let _hidden = forward_cache_only(cfg, weights, chunk, cache, cache.seq_len());
            cache.advance(chunk.len());
            // Materialise KV only after the *last* cache-only chunk. Intermediate
            // barriers force a full layer-stack eval for every sub-chunk (Qwen
            // linear-attention clamps to 1024, so p=2048 paid two full barriers).
            // Lazy slice_update chains stay short for ≤2–3 chunks and the final
            // barrier still matches mlx_lm's post-prefix eval before decode.
            if end == cache_only_prefix_len {
                eval_kv_refs(cache);
            }
            offset = end;
        }

        let tok = decode_step_with_sampling_buffers(
            cfg,
            weights,
            prompt_tokens[cache_only_prefix_len],
            cache,
            sampling_request,
            rng,
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        );
        // E2B/E4B defer cleanup past the first-token boundary; request
        // completion and the decode cadence still bound cache growth.
        if clear_cache_after_split_prefill(cfg.hidden_size_per_layer_input) {
            clear_cache();
        }
        return tok;
    }

    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(total);
        let chunk = &prompt_tokens[offset..end];
        let is_final_chunk = end == total;
        let needs_full_logits =
            is_final_chunk && (sampling.temperature > 0.0 || sampling.uses_logits_processors());
        let logits = if needs_full_logits {
            forward(cfg, weights, chunk, cache, cache.seq_len())
        } else if is_final_chunk {
            // Final chunk, greedy: need argmax but not full f32 logits.
            forward_argmax(cfg, weights, chunk, cache, cache.seq_len())
        } else {
            // Non-final chunk: skip lm_head projection entirely.
            forward_cache_only(cfg, weights, chunk, cache, cache.seq_len())
        };
        cache.advance(chunk.len());
        offset = end;

        if offset == total {
            let tok = if sampling.temperature > 0.0 || sampling.uses_logits_processors() {
                sample_prefill_token_gpu_first(
                    &logits,
                    sampling,
                    sampling_request.repetition_tokens,
                    rng,
                    || eval_with_kv_refs(&logits, cache),
                    sampling_probs_buf,
                    sampling_logits_buf,
                    sampling_candidates_buf,
                )
            } else {
                // GPU argmax over [vocab] logits -> token ID.
                let token_arr = argmax(&logits, None);
                eval_with_kv_refs(&token_arr, cache);
                token_arr.first_u32_unchecked()
            };
            // Free MLX's graph/intermediate-array cache after prefill.
            // SwiftLM does the same (MLX.Memory.clearCache()) to reclaim GPU
            // memory consumed by the prefill computation graph.
            clear_cache();
            return tok;
        } else {
            // Drain GPU queue. hidden depends on all transformer layers
            // (KV cache writes via attention); eval_with_kv_refs explicitly
            // materialises the KV cache arrays alongside the hidden state to
            // prevent O(N²) lazy-graph growth across chunks.
            eval_with_kv_refs(&logits, cache);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_prefill_gemma4_unified_with_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    inputs: &Gemma4UnifiedRuntimeInputs,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> Result<u32, String> {
    let (tok, _, _) = chunked_prefill_gemma4_unified_with_mtp_history_and_sampling_buffers(
        cfg,
        weights,
        prompt_tokens,
        cache,
        inputs,
        sampling_request,
        rng,
        sampling_probs_buf,
        sampling_logits_buf,
        sampling_candidates_buf,
        false,
    )?;
    Ok(tok)
}

/// Gemma4 unified multimodal prefill with optional MTP post-norm capture (WS-M5).
///
/// When `capture_mtp_history` is true (and MTP weights exist), returns post-norm
/// hidden for every prompt position so `initialize_generation_state` can warm
/// the MTP head. When false, still uses the full-seq path for consistent numerics.
#[allow(clippy::too_many_arguments)]
pub fn chunked_prefill_gemma4_unified_with_mtp_history_and_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    inputs: &Gemma4UnifiedRuntimeInputs,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
    capture_mtp_history: bool,
) -> Result<(u32, Option<MlxArray>, Vec<u32>), String> {
    use crate::model::forward_with_initial_hidden_media_post_norm_last_lm_head;

    let sampling = sampling_request.params;
    // gemma4_vl reuses unified connector weights but fail-closes on missing towers.
    let chunk = if crate::gemma4_vl::is_gemma4_vl_family(&cfg.model_family) {
        crate::gemma4_vl::build_vl_prefill_embeddings(cfg, weights, prompt_tokens, inputs)
            .map_err(|e| e.to_string())?
    } else {
        build_chunk_embeddings(cfg, weights, prompt_tokens, 0, inputs).map_err(|e| e.to_string())?
    };
    let media_ranges: Vec<(usize, usize)> = chunk
        .media_ranges
        .iter()
        .map(|range| (range.start, range.end_inclusive))
        .collect();
    let (logits, post_norm) = forward_with_initial_hidden_media_post_norm_last_lm_head(
        cfg,
        weights,
        prompt_tokens,
        chunk.hidden,
        &media_ranges,
        cache,
        0,
    );
    cache.advance(prompt_tokens.len());

    let tok = if sampling.temperature > 0.0 || sampling.uses_logits_processors() {
        sample_prefill_token_gpu_first(
            &logits,
            sampling,
            sampling_request.repetition_tokens,
            rng,
            || eval_with_kv_refs(&logits, cache),
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        )
    } else {
        let token_arr = argmax(&logits, None);
        eval_with_kv_refs(&token_arr, cache);
        token_arr.first_u32_unchecked()
    };

    let mut history_tokens = Vec::new();
    let mtp_hidden = if capture_mtp_history && weights.mtp.is_some() {
        // History tokens: prompt[1..] + sampled first decode token (matches text path).
        if prompt_tokens.len() > 1 {
            history_tokens.extend_from_slice(&prompt_tokens[1..]);
        }
        history_tokens.push(tok);
        eval(&[&post_norm]);
        Some(post_norm)
    } else {
        None
    };
    clear_cache();
    Ok((tok, mtp_hidden, history_tokens))
}

/// Prefill Unlimited-OCR with dual-vision image features injected at `<image>`
/// soft-token positions. Single full-prompt prefill (no chunking) — vision
/// features are dense and typically a few hundred tokens.
#[allow(clippy::too_many_arguments)]
pub fn chunked_prefill_unlimited_ocr_with_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    image_views: &UnlimitedOcrImageViews,
    cache: &mut MlxKVCache,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> Result<u32, String> {
    let sampling = sampling_request.params;
    let hidden = build_embeddings_with_image(cfg, weights, prompt_tokens, image_views)
        .map_err(|e| e.to_string())?;
    // Unlimited-OCR uses ordinary causal attention for the complete prefill.
    // Its R-SWA cache only starts replacing generated decode entries after the
    // full prompt has been retained.  In particular, image soft tokens are not
    // a bidirectional PrefixLM block (that behaviour is specific to Gemma4).
    let media_ranges = [];
    let logits = forward_with_initial_hidden_and_media_ranges(
        cfg,
        weights,
        prompt_tokens,
        hidden,
        &media_ranges,
        cache,
        0,
        FinalLogitsMode::Full,
    );
    cache.advance(prompt_tokens.len());

    let tok = if sampling.temperature > 0.0 || sampling.uses_logits_processors() {
        sample_prefill_token_gpu_first(
            &logits,
            sampling,
            sampling_request.repetition_tokens,
            rng,
            || eval_with_kv_refs(&logits, cache),
            sampling_probs_buf,
            sampling_logits_buf,
            sampling_candidates_buf,
        )
    } else {
        let token_arr = argmax(&logits, None);
        eval_with_kv_refs(&token_arr, cache);
        token_arr.first_u32_unchecked()
    };
    clear_cache();
    Ok(tok)
}

/// Like `chunked_prefill` but also returns the pre-norm hidden at the last
/// prompt position.  Used by the MTP warmup path to prime the MTP head's KV
/// cache with one entry from the prefill context before decode starts.
///
/// The final chunk is always processed with `forward_all_positions_with_final_hidden`
/// so the hidden is available even when sampling is greedy.  The remainder of
/// the implementation is identical to `chunked_prefill`.
pub fn chunked_prefill_with_final_hidden(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
) -> (u32, MlxArray) {
    let (tok, hidden, _) = chunked_prefill_with_mtp_history(
        cfg,
        weights,
        prompt_tokens,
        cache,
        chunk_size,
        sampling_request,
        rng,
    );
    (tok, hidden)
}

/// Like `chunked_prefill_with_final_hidden`, but returns the full post-norm
/// hidden sequence for the final prefill chunk plus the committed token IDs
/// that each hidden row predicts. MTP uses this to seed its recurrent cache
/// with prompt/history transitions instead of only the final prefill token.
pub fn chunked_prefill_with_mtp_history(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
) -> (u32, MlxArray, Vec<u32>) {
    let mut sampling_probs_buf = Vec::new();
    let mut sampling_logits_buf = Vec::new();
    let mut sampling_candidates_buf = Vec::new();
    chunked_prefill_with_mtp_history_and_sampling_buffers(
        cfg,
        weights,
        prompt_tokens,
        cache,
        chunk_size,
        sampling_request,
        rng,
        &mut sampling_probs_buf,
        &mut sampling_logits_buf,
        &mut sampling_candidates_buf,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_prefill_with_mtp_history_and_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> (u32, MlxArray, Vec<u32>) {
    use mlx_sys::MlxDtype;
    let sampling = sampling_request.params;
    let chunk_size = chunk_size.max(1);
    let total = prompt_tokens.len();

    // cache_only_prefix_len fast path: no final-hidden support (short path).
    // Fall back to the normal chunked_prefill for this case and return a
    // placeholder hidden (zero-shaped).  MTP warmup skips None returns;
    // this case is rare for MTP-benchmark runs (all tokens processed as a
    // single mlx-lm-style batch).
    let cache_only_prefix_len = mlx_lm_style_cache_only_prefix_len(total, sampling);
    if cache_only_prefix_len > 0 {
        let mut offset = 0;
        while offset < cache_only_prefix_len {
            let end = (offset + chunk_size).min(cache_only_prefix_len);
            let chunk = &prompt_tokens[offset..end];
            // Cache-only: skip lm_head projection (hidden×vocab_size).
            let _hidden = forward_cache_only(cfg, weights, chunk, cache, cache.seq_len());
            cache.advance(chunk.len());
            if end == cache_only_prefix_len {
                eval_kv_refs(cache);
            }
            offset = end;
        }
        // Completing step: full forward for the last prompt token so MTP gets
        // post-norm hidden. Sample when temperature>0; argmax when greedy.
        let last_tok = prompt_tokens[cache_only_prefix_len];
        let last_offset = cache.seq_len();
        let (logits_all, final_hidden) =
            forward_all_positions_with_final_hidden(cfg, weights, &[last_tok], cache, last_offset);
        cache.advance(1);
        let logits_row = {
            use mlx_sys::{astype, reshape, slice};
            let lv = slice(
                &logits_all,
                &[0, 0],
                &[1, cfg.vocab_size as i32],
                &[1, 1],
                None,
            );
            let lv = astype(&lv, MlxDtype::Float32, None);
            reshape(&lv, &[cfg.vocab_size as i32], None)
        };
        let tok = if sampling.temperature > 0.0 {
            eval_with_kv_refs(&logits_row, cache);
            sample_prefill_token_gpu_first(
                &logits_row,
                sampling,
                sampling_request.repetition_tokens,
                rng,
                || eval(&[&logits_row, &final_hidden]),
                sampling_probs_buf,
                sampling_logits_buf,
                sampling_candidates_buf,
            )
        } else {
            let token_arr = argmax(&logits_row, None);
            eval_kv_refs(cache);
            eval(&[&token_arr, &final_hidden]);
            token_arr.data_u32()[0]
        };
        if sampling.temperature > 0.0 {
            eval(&[&final_hidden]);
        }
        clear_cache();
        return (tok, final_hidden, vec![tok]);
    }

    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(total);
        let chunk = &prompt_tokens[offset..end];
        let is_final_chunk = end == total;
        let chunk_offset = cache.seq_len();

        if is_final_chunk {
            // Use last-position-only lm_head to avoid seq×vocab matmul on
            // all positions; returns logits as [vocab] directly.
            let (last_logits, post_norm_all) = forward_all_positions_post_norm_last_lm_head(
                cfg,
                weights,
                chunk,
                cache,
                chunk_offset,
            );
            cache.advance(chunk.len());

            let tok = if sampling.temperature > 0.0 || sampling.uses_logits_processors() {
                eval_with_kv_refs(&last_logits, cache);
                sample_prefill_token_gpu_first(
                    &last_logits,
                    sampling,
                    sampling_request.repetition_tokens,
                    rng,
                    || eval(&[&last_logits]),
                    sampling_probs_buf,
                    sampling_logits_buf,
                    sampling_candidates_buf,
                )
            } else {
                let token_arr = argmax(&last_logits, None);
                eval_with_kv_refs(&token_arr, cache);
                token_arr.data_u32()[0]
            };
            // Materialize post_norm_all before clear_cache() so the MTP warmup
            // consumer can safely slice into it after the pool is cleared.
            let mut history_tokens = Vec::with_capacity(chunk.len());
            if chunk.len() > 1 {
                history_tokens.extend_from_slice(&chunk[1..]);
            }
            history_tokens.push(tok);
            eval(&[&post_norm_all]);
            clear_cache();
            return (tok, post_norm_all, history_tokens);
        } else {
            // Non-final chunk: skip lm_head projection (hidden×vocab_size).
            // Defer materialisation until the final chunk's eval (same contract
            // as the greedy cache-only multi-chunk path): intermediate
            // barriers force a full layer-stack eval per sub-chunk and hurt
            // MTP long-prompt TTFT (Qwen linear clamp → multiple chunks).
            // KV writes stay on the lazy graph and are pulled by the final
            // `eval_with_kv_refs` / `eval_kv_refs` on the completing step.
            let _hidden = forward_cache_only(cfg, weights, chunk, cache, chunk_offset);
            cache.advance(chunk.len());
            offset = end;
        }
    }
}

fn mlx_lm_style_cache_only_prefix_len(total_tokens: usize, sampling: MlxSamplingParams) -> usize {
    // Process n−1 tokens with FinalLogitsMode::Skip when the completing step
    // only needs the last prompt token's logits. Repetition penalty is
    // excluded (needs the full residual stream for the completing step).
    //
    // Greedy: always n−1 (direct-mode high-water path).
    // Sampling (MTP publication rows use temp>0): only when the prompt is long
    // enough that last-layer skip amortizes the extra completing-step barrier
    // (short sampled prompts measured slightly slower under dual-barrier).
    const SAMPLING_CACHE_ONLY_MIN_TOKENS: usize = 512;
    if total_tokens <= 1 || sampling.uses_logits_processors() {
        return 0;
    }
    if sampling.temperature <= 0.0 || total_tokens >= SAMPLING_CACHE_ONLY_MIN_TOKENS {
        total_tokens - 1
    } else {
        0
    }
}

fn clear_cache_after_split_prefill(hidden_size_per_layer_input: usize) -> bool {
    hidden_size_per_layer_input == 0
}

fn eval_kv_refs(cache: &MlxKVCache) {
    let kv_refs = cache.collect_eval_refs();
    if !kv_refs.is_empty() {
        eval(&kv_refs);
    }
}

fn eval_with_kv_refs(output: &MlxArray, cache: &MlxKVCache) {
    let kv_refs = cache.collect_eval_refs();
    if kv_refs.is_empty() {
        eval(&[output]);
    } else {
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(output);
        targets.extend(kv_refs);
        eval(&targets);
    }
}

/// Start the double-buffer direct decode pipeline from a known (materialised) token.
///
/// Runs one forward pass, submits the result to the GPU via `async_eval`, and
/// returns the **lazy** token array.  The caller stores this and passes it to
/// `advance_direct_pipeline` on the next decode step.
///
/// Only valid for deterministic argmax decoding (temperature = 0).  For sampling paths use `decode_step`.
pub fn start_direct_pipeline(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
) -> MlxArray {
    let token_offset = cache.seq_len();
    let logits = forward_argmax(cfg, weights, &[last_token], cache, token_offset);
    cache.advance(1);
    // KV cache is in token_arr's computation graph; no extra refs needed.
    let token_arr = argmax(&logits, None);
    async_eval(&[&token_arr]);
    token_arr
}

/// Advance the double-buffer direct pipeline by one step.
///
/// This mirrors mlx_lm's `_step(y)` → `mx.async_eval(next_y)` → `mx.eval(y)` pattern:
///
/// 1. Build step N+1's compute graph using the pending lazy token (no GPU sync).
/// 2. Submit step N+1 to the GPU via `async_eval`.
/// 3. Materialise the pending (step N) token — GPU should already have it.
/// 4. Return `(step_N_token_u32, step_N+1_lazy_token)`.
///
/// The caller stores the returned lazy token as the next pending value.
pub fn advance_direct_pipeline(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    pending: &MlxArray, // lazy token from previous `start_direct_pipeline` / `advance_direct_pipeline`
    cache: &mut MlxKVCache,
) -> (u32, MlxArray) {
    let advanced = advance_direct_pipeline_with_timings(cfg, weights, pending, cache);
    (advanced.token, advanced.next_pending)
}

pub fn advance_direct_pipeline_with_timings(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    pending: &MlxArray, // lazy token from previous `start_direct_pipeline` / `advance_direct_pipeline`
    cache: &mut MlxKVCache,
) -> DirectPipelineAdvance {
    // Build next step's graph using the lazy pending token.
    // forward_lazy_single accepts an unevaluated MlxArray, so this runs entirely
    // on the CPU without waiting for `pending` to be materialised.
    let token_offset = cache.seq_len();
    let stage_profile = direct_pipeline_stage_profile_enabled();
    if stage_profile {
        reset_forward_stage_timings();
    }
    let forward_started = Instant::now();
    let logits = forward_lazy_single_argmax(cfg, weights, pending, cache, token_offset);
    let forward_wall_us = elapsed_us(forward_started);
    let forward_stage = if stage_profile {
        take_forward_stage_timings()
    } else {
        ForwardStageTimings::default()
    };
    cache.advance(1);
    let argmax_started = Instant::now();
    let next_token_arr = argmax(&logits, None);
    let argmax_wall_us = elapsed_us(argmax_started);
    // Submit step N+1 to the GPU before waiting for step N.
    // KV cache is in next_token_arr's computation graph (via SDPA), so no extra
    // refs needed — they would only add one GPU command buffer per layer (≈85µs each).
    let async_eval_started = Instant::now();
    async_eval(&[&next_token_arr]);
    let async_eval_wall_us = elapsed_us(async_eval_started);

    // Diagnostic barrier: force step N+1 GPU completion before measuring the
    // pending (step N) wait. Splits `async_eval` cost into "pure submit" vs
    // "GPU-completion wait" by removing the double-buffer overlap.
    let next_complete_wall_us = if direct_pipeline_barrier_enabled() {
        let started = Instant::now();
        eval(&[&next_token_arr]);
        elapsed_us(started)
    } else {
        0
    };

    // Materialise the pending (step N) token.  Because `async_eval` was called
    // in the previous `start_direct_pipeline` / `advance_direct_pipeline`, the GPU
    // has been working on this token the entire time the CPU was building N+1's
    // graph above — so `eval` is typically a no-op barrier.
    let pending_eval_started = Instant::now();
    eval(&[pending]);
    let pending_eval_wall_us = elapsed_us(pending_eval_started);
    let pending_read_started = Instant::now();
    let tok = pending.first_u32_unchecked();
    let pending_read_wall_us = elapsed_us(pending_read_started);

    DirectPipelineAdvance {
        token: tok,
        next_pending: next_token_arr,
        timings: DirectPipelineTimings {
            forward_wall_us,
            forward_layer_loop_wall_us: forward_stage.layer_loop_wall_us,
            forward_head_wall_us: forward_stage.head_wall_us,
            argmax_wall_us,
            async_eval_wall_us,
            next_complete_wall_us,
            pending_eval_wall_us,
            pending_read_wall_us,
            linear_attention_layer_ops: forward_stage.linear_attention_layer_ops,
            linear_attention_layer_count: forward_stage.linear_attention_layer_count,
            full_attention_layer_ops: forward_stage.full_attention_layer_ops,
            full_attention_layer_count: forward_stage.full_attention_layer_count,
        },
    }
}

/// Decode one token: forward pass for a single token and return sampled ID.
///
/// When `temperature` is 0.0, uses GPU argmax.  When > 0.0, evals
/// logits to CPU and samples from the temperature-scaled categorical
/// distribution.  The caller must pass a per-request `rng` for reproducibility.
pub fn decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
) -> u32 {
    let mut sampling_probs_buf = Vec::new();
    let mut sampling_logits_buf = Vec::new();
    let mut sampling_candidates_buf = Vec::new();
    decode_step_with_sampling_buffers(
        cfg,
        weights,
        last_token,
        cache,
        sampling_request,
        rng,
        &mut sampling_probs_buf,
        &mut sampling_logits_buf,
        &mut sampling_candidates_buf,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decode_step_with_sampling_buffers(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    sampling_probs_buf: &mut Vec<f32>,
    sampling_logits_buf: &mut Vec<f32>,
    sampling_candidates_buf: &mut Vec<(usize, f32)>,
) -> u32 {
    let sampling = sampling_request.params;
    let token_offset = cache.seq_len();
    let deterministic_argmax = sampling.temperature <= 0.0 && !sampling.uses_logits_processors();
    let logits = if deterministic_argmax {
        forward_argmax(cfg, weights, &[last_token], cache, token_offset)
    } else {
        forward(cfg, weights, &[last_token], cache, token_offset)
    };
    cache.advance(1);

    if sampling.temperature > 0.0
        && !sampling.uses_logits_processors()
        && sampling.top_k == 0
        && sampling.top_p >= 1.0
    {
        // GPU-side sampling: no logits transfer to CPU.
        sample_categorical_gpu(&logits, sampling.temperature)
    } else if sampling.temperature > 0.0 || sampling.uses_logits_processors() {
        if let Some(tok) = sample_categorical_with_topk_gpu(
            &logits,
            sampling,
            sampling_request.repetition_tokens,
            rng,
        )
        .or_else(|| {
            sample_categorical_with_topp_gpu(
                &logits,
                sampling,
                sampling_request.repetition_tokens,
                rng,
            )
        }) {
            tok
        } else {
            eval_with_kv_refs(&logits, cache);
            let logits_data = logits.data_f32();
            sample_categorical_into(
                logits_data,
                sampling,
                sampling_request.repetition_tokens,
                rng,
                sampling_probs_buf,
                sampling_logits_buf,
                sampling_candidates_buf,
            )
        }
    } else {
        // Deterministic argmax path: GPU argmax, no CPU data movement.
        let token_arr = argmax(&logits, None);
        eval_with_kv_refs(&token_arr, cache);
        token_arr.first_u32_unchecked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_lm_style_prefill_leaves_final_prompt_token_for_step() {
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(2048, MlxSamplingParams::greedy()),
            2047
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(513, MlxSamplingParams::greedy()),
            512
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(512, MlxSamplingParams::greedy()),
            511
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(2, MlxSamplingParams::greedy()),
            1
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(1, MlxSamplingParams::greedy()),
            0
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(0, MlxSamplingParams::greedy()),
            0
        );
    }

    #[test]
    fn mlx_lm_style_prefill_allows_long_sampling_without_repetition_penalty() {
        // Long sampled MTP / direct: n−1 cache-only, last token sampled.
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(2048, MlxSamplingParams::new(0.7, 1.0, 0)),
            2047
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(512, MlxSamplingParams::new(0.6, 0.95, 20)),
            511
        );
        // Short sampled prompts stay on the single-pass path (barrier amortisation).
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(2, MlxSamplingParams::new(0.6, 0.95, 20)),
            0
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(511, MlxSamplingParams::new(0.6, 0.95, 20)),
            0
        );
        // Repetition penalty still needs the full residual path.
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(
                2048,
                MlxSamplingParams::greedy().with_repetition_penalty(1.1, None)
            ),
            0
        );
    }

    #[test]
    fn e_series_prefill_defers_cache_cleanup() {
        assert!(!clear_cache_after_split_prefill(256));
        assert!(clear_cache_after_split_prefill(0));
    }
}
