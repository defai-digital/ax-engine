use std::time::Instant;

use mlx_sys::{MlxArray, argmax, async_eval, clear_cache, eval};

use crate::kv_cache::MlxKVCache;
use crate::linear_attention::GATED_DELTA_THREADGROUP_CACHE_CAPACITY;
use crate::model::{
    ModelConfig, TurboQuantModelDecodeContext, forward,
    forward_lazy_single_with_turboquant_context, forward_with_turboquant_context,
};
use crate::sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64, sample_categorical};
use crate::weights::ModelWeights;

/// Default chunk size for chunked prefill, matching SwiftLM's default and the
/// GatedDelta Metal kernel's threadgroup cache capacity.
pub const DEFAULT_PREFILL_CHUNK: usize = GATED_DELTA_THREADGROUP_CACHE_CAPACITY;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectPipelineTimings {
    pub forward_wall_us: u32,
    pub async_eval_wall_us: u32,
    pub pending_eval_wall_us: u32,
    pub pending_read_wall_us: u32,
}

pub struct DirectPipelineAdvance {
    pub token: u32,
    pub next_pending: MlxArray,
    pub timings: DirectPipelineTimings,
}

fn elapsed_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

/// Process the full prompt in chunks of `chunk_size` tokens.
///
/// Returns the sampled next-token ID from the last-token logits.
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

    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(total);
        let chunk = &prompt_tokens[offset..end];
        let logits = forward(cfg, weights, chunk, cache, cache.seq_len);
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
                token_arr.data_u32().first().copied().unwrap_or(0)
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
    let logits = forward_with_turboquant_context(
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
    let forward_started = Instant::now();
    let logits = forward_lazy_single_with_turboquant_context(
        cfg,
        weights,
        pending,
        cache,
        token_offset,
        turboquant_context,
    );
    let forward_wall_us = elapsed_us(forward_started);
    cache.seq_len += 1;
    let async_eval_started = Instant::now();
    let next_token_arr = argmax(&logits, None);
    // Submit step N+1 to the GPU before waiting for step N.
    // KV cache is in next_token_arr's computation graph (via SDPA), so no extra
    // refs needed — they would only add one GPU command buffer per layer (≈85µs each).
    async_eval(&[&next_token_arr]);
    let async_eval_wall_us = elapsed_us(async_eval_started);

    // Materialise the pending (step N) token.  Because `async_eval` was called
    // in the previous `start_direct_pipeline` / `advance_direct_pipeline`, the GPU
    // has been working on this token the entire time the CPU was building N+1's
    // graph above — so `eval` is typically a no-op barrier.
    let pending_eval_started = Instant::now();
    eval(&[pending]);
    let pending_eval_wall_us = elapsed_us(pending_eval_started);
    let pending_read_started = Instant::now();
    let tok = pending.data_u32().first().copied().unwrap_or(0);
    let pending_read_wall_us = elapsed_us(pending_read_started);

    DirectPipelineAdvance {
        token: tok,
        next_pending: next_token_arr,
        timings: DirectPipelineTimings {
            forward_wall_us,
            async_eval_wall_us,
            pending_eval_wall_us,
            pending_read_wall_us,
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
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
) -> u32 {
    decode_step_with_turboquant_context(cfg, weights, last_token, cache, sampling, rng, None)
}

pub fn decode_step_with_turboquant_context(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    sampling: MlxSamplingParams,
    rng: &mut Xorshift64,
    turboquant_context: Option<&TurboQuantModelDecodeContext<'_>>,
) -> u32 {
    let token_offset = cache.seq_len;
    let logits = forward_with_turboquant_context(
        cfg,
        weights,
        &[last_token],
        cache,
        token_offset,
        turboquant_context,
    );
    cache.seq_len += 1;

    if sampling.temperature > 0.0 {
        eval_with_kv_refs(&logits, cache);
        let logits_data = logits.data_f32();
        sample_categorical(logits_data, sampling, &[last_token], rng)
    } else if sampling.uses_repetition_penalty() {
        eval_with_kv_refs(&logits, cache);
        sample_categorical(logits.data_f32(), sampling, &[last_token], rng)
    } else {
        // Deterministic argmax path: GPU argmax, no CPU data movement.
        let token_arr = argmax(&logits, None);
        eval_with_kv_refs(&token_arr, cache);
        token_arr.data_u32().first().copied().unwrap_or(0)
    }
}
