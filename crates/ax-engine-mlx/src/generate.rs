use std::cell::Cell;
use std::sync::OnceLock;
use std::time::Instant;

use mlx_sys::{MlxArray, argmax, async_eval, clear_cache, eval};

use crate::kv_cache::MlxKVCache;
use crate::linear_attention_ops::GATED_DELTA_THREADGROUP_CACHE_CAPACITY;
use crate::model::{
    ModelConfig, TurboQuantModelDecodeContext, forward, forward_argmax_with_turboquant_context,
    forward_lazy_single_argmax_with_turboquant_context, forward_with_turboquant_context,
};
use crate::sampling::{
    MlxSamplingParams, MlxSamplingRequest, Xorshift64, sample_categorical, sample_categorical_gpu,
};
use crate::weights::ModelWeights;

/// Default chunk size for chunked prefill, matching mlx-lm's default
/// `prefill_step_size` and the long GatedDelta Metal kernel specialization.
pub const DEFAULT_PREFILL_CHUNK: usize = GATED_DELTA_THREADGROUP_CACHE_CAPACITY;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectPipelineTimings {
    pub forward_wall_us: u32,
    /// Subset of `forward_wall_us`: time spent inside the 64-layer
    /// `layer_forward_with_turboquant_context` loop. Always recorded (no eval
    /// barrier inserted). The residual `forward_wall_us - layer_loop_wall_us -
    /// head_wall_us` covers embed-tokens + per-layer-input + dtype-cast +
    /// scale-hidden graph-build cost.
    pub forward_layer_loop_wall_us: u32,
    /// Subset of `forward_wall_us`: time spent in the post-layer final RMSNorm
    /// + lm-head qmatmul + logit finalize chain. Always recorded.
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForwardStageTimings {
    pub layer_loop_wall_us: u32,
    pub head_wall_us: u32,
    pub linear_attention_layer_ops: u64,
    pub linear_attention_layer_count: u32,
    pub full_attention_layer_ops: u64,
    pub full_attention_layer_count: u32,
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
/// per decode forward by `forward_lazy_single_*`. Always-on; uses a thread-local
/// `Cell` to avoid any locking on the hot path.
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

fn direct_pipeline_barrier_enabled() -> bool {
    *DIRECT_PIPELINE_BARRIER_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("AX_MLX_DIRECT_PIPELINE_BARRIER").as_deref(),
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
/// For deterministic argmax on prompts longer than 512 tokens, mirror
/// `mlx_lm.generate_step`: prefill every prompt token except the final one as
/// cache-state-only work, then run the final prompt token through the normal
/// single-token step to produce the first generated token. Evaluating only KV
/// refs lets MLX's lazy graph prune the final logits path for long prompt
/// chunks. Non-greedy sampling and short prompts keep the historical full-logits
/// path until the sampling and small-prompt performance contracts are audited.
pub fn chunked_prefill(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
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
            let _logits = forward_argmax_with_turboquant_context(
                cfg,
                weights,
                chunk,
                cache,
                cache.seq_len,
                None,
            );
            cache.seq_len += chunk.len();
            eval_kv_refs(cache);
            clear_cache();
            offset = end;
        }

        let tok = decode_step(
            cfg,
            weights,
            prompt_tokens[cache_only_prefix_len],
            cache,
            sampling_request,
            rng,
        );
        clear_cache();
        return tok;
    }

    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(total);
        let chunk = &prompt_tokens[offset..end];
        let is_final_chunk = end == total;
        let needs_full_logits =
            is_final_chunk && (sampling.temperature > 0.0 || sampling.uses_repetition_penalty());
        let logits = if needs_full_logits {
            forward(cfg, weights, chunk, cache, cache.seq_len)
        } else {
            forward_argmax_with_turboquant_context(cfg, weights, chunk, cache, cache.seq_len, None)
        };
        cache.seq_len += chunk.len();
        offset = end;

        if offset == total {
            let tok = if sampling.temperature > 0.0 || sampling.uses_repetition_penalty() {
                eval_with_kv_refs(&logits, cache);
                let logits_data = logits.data_f32();
                sample_categorical(
                    logits_data,
                    sampling,
                    sampling_request.repetition_tokens,
                    rng,
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
            // Drain GPU queue asynchronously. logits depends on the KV cache
            // transitively (via SDPA), so evaluating logits materialises the
            // appended KV slice_update nodes and prevents O(N²) graph growth.
            async_eval(&[&logits]);
        }
    }
}

fn mlx_lm_style_cache_only_prefix_len(total_tokens: usize, sampling: MlxSamplingParams) -> usize {
    if total_tokens > 512 && sampling.temperature <= 0.0 && !sampling.uses_repetition_penalty() {
        total_tokens - 1
    } else {
        0
    }
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
    start_direct_pipeline_with_turboquant_context(cfg, weights, last_token, cache, None)
}

pub fn start_direct_pipeline_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> MlxArray {
    let token_offset = cache.seq_len;
    let logits = forward_argmax_with_turboquant_context(
        cfg,
        weights,
        &[last_token],
        cache,
        token_offset,
        turboquant_context,
    );
    cache.seq_len += 1;
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
    advance_direct_pipeline_with_turboquant_context(cfg, weights, pending, cache, None)
}

pub fn advance_direct_pipeline_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    pending: &MlxArray, // lazy token from previous `start_direct_pipeline` / `advance_direct_pipeline`
    cache: &mut MlxKVCache,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> (u32, MlxArray) {
    let advanced = advance_direct_pipeline_with_timings_and_turboquant_context(
        cfg,
        weights,
        pending,
        cache,
        turboquant_context,
    );
    (advanced.token, advanced.next_pending)
}

pub fn advance_direct_pipeline_with_timings_and_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    pending: &MlxArray, // lazy token from previous `start_direct_pipeline` / `advance_direct_pipeline`
    cache: &mut MlxKVCache,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> DirectPipelineAdvance {
    // Build next step's graph using the lazy pending token.
    // forward_lazy_single accepts an unevaluated MlxArray, so this runs entirely
    // on the CPU without waiting for `pending` to be materialised.
    let token_offset = cache.seq_len;
    reset_forward_stage_timings();
    let forward_started = Instant::now();
    let logits = forward_lazy_single_argmax_with_turboquant_context(
        cfg,
        weights,
        pending,
        cache,
        token_offset,
        turboquant_context,
    );
    let forward_wall_us = elapsed_us(forward_started);
    let forward_stage = take_forward_stage_timings();
    cache.seq_len += 1;
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
    decode_step_with_turboquant_context(
        cfg,
        weights,
        last_token,
        cache,
        sampling_request,
        rng,
        None,
    )
}

pub fn decode_step_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    sampling_request: MlxSamplingRequest<'_>,
    rng: &mut Xorshift64,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> u32 {
    let sampling = sampling_request.params;
    let token_offset = cache.seq_len;
    let deterministic_argmax = sampling.temperature <= 0.0 && !sampling.uses_repetition_penalty();
    let logits = if deterministic_argmax {
        forward_argmax_with_turboquant_context(
            cfg,
            weights,
            &[last_token],
            cache,
            token_offset,
            turboquant_context,
        )
    } else {
        forward_with_turboquant_context(
            cfg,
            weights,
            &[last_token],
            cache,
            token_offset,
            turboquant_context,
        )
    };
    cache.seq_len += 1;

    if sampling.temperature > 0.0
        && !sampling.uses_repetition_penalty()
        && sampling.top_k == 0
        && sampling.top_p >= 1.0
    {
        // GPU-side sampling: no logits transfer to CPU.
        sample_categorical_gpu(&logits, sampling.temperature)
    } else if sampling.temperature > 0.0 || sampling.uses_repetition_penalty() {
        eval_with_kv_refs(&logits, cache);
        let logits_data = logits.data_f32();
        sample_categorical(
            logits_data,
            sampling,
            sampling_request.repetition_tokens,
            rng,
        )
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
    }

    #[test]
    fn mlx_lm_style_prefill_keeps_short_prompt_on_historical_path() {
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(512, MlxSamplingParams::greedy()),
            0
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
    fn mlx_lm_style_prefill_split_is_greedy_only_for_now() {
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(2048, MlxSamplingParams::new(0.7, 1.0, 0)),
            0
        );
        assert_eq!(
            mlx_lm_style_cache_only_prefix_len(
                2048,
                MlxSamplingParams::greedy().with_repetition_penalty(1.1, None)
            ),
            0
        );
    }
}
