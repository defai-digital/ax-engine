use mlx_sys::{MlxArray, argmax, async_eval, clear_cache, eval};

use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, forward, forward_lazy_single};
use crate::sampling::{Xorshift64, sample_categorical};
use crate::weights::ModelWeights;

/// Default chunk size for chunked prefill, matching SwiftLM's default.
pub const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Process the full prompt in chunks of `chunk_size` tokens.
///
/// Returns the sampled next-token ID (GPU argmax on last-token logits).
pub fn chunked_prefill(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt_tokens: &[u32],
    cache: &mut MlxKVCache,
    chunk_size: usize,
) -> u32 {
    let chunk_size = chunk_size.max(1);
    let total = prompt_tokens.len();

    let mut offset = 0;
    loop {
        let end = (offset + chunk_size).min(total);
        let chunk = &prompt_tokens[offset..end];
        let logits = forward(cfg, weights, chunk, cache, offset);
        cache.seq_len += chunk.len();
        offset = end;

        if offset == total {
            // GPU argmax over [vocab] logits → token ID.
            let token_arr = argmax(&logits, None);
            eval(&[&token_arr]);
            // Free MLX's graph/intermediate-array cache after prefill.
            // SwiftLM does the same (MLX.Memory.clearCache()) to reclaim GPU
            // memory consumed by the prefill computation graph.
            clear_cache();
            return token_arr.data_u32().first().copied().unwrap_or(0);
        } else {
            // Drain GPU queue asynchronously; don't block on intermediate chunks.
            async_eval(&[&logits]);
        }
    }
}

/// Start the double-buffer greedy decode pipeline from a known (materialised) token.
///
/// Runs one forward pass, submits the result to the GPU via `async_eval`, and
/// returns the **lazy** token array.  The caller stores this and passes it to
/// `advance_greedy_pipeline` on the next decode step.
///
/// Only valid for temperature = 0 (greedy).  For sampling paths use `decode_step`.
pub fn start_greedy_pipeline(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
) -> MlxArray {
    let token_offset = cache.seq_len;
    let logits = forward(cfg, weights, &[last_token], cache, token_offset);
    cache.seq_len += 1;
    let token_arr = argmax(&logits, None);
    let kv_refs = cache.collect_eval_refs();
    let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
    targets.push(&token_arr);
    targets.extend(kv_refs);
    async_eval(&targets);
    token_arr
}

/// Advance the double-buffer greedy pipeline by one step.
///
/// This mirrors mlx_lm's `_step(y)` → `mx.async_eval(next_y)` → `mx.eval(y)` pattern:
///
/// 1. Build step N+1's compute graph using the pending lazy token (no GPU sync).
/// 2. Submit step N+1 to the GPU via `async_eval`.
/// 3. Materialise the pending (step N) token — GPU should already have it.
/// 4. Return `(step_N_token_u32, step_N+1_lazy_token)`.
///
/// The caller stores the returned lazy token as the next pending value.
pub fn advance_greedy_pipeline(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    pending: &MlxArray, // lazy token from previous `start_greedy_pipeline` / `advance_greedy_pipeline`
    cache: &mut MlxKVCache,
) -> (u32, MlxArray) {
    // Build next step's graph using the lazy pending token.
    // forward_lazy_single accepts an unevaluated MlxArray, so this runs entirely
    // on the CPU without waiting for `pending` to be materialised.
    let token_offset = cache.seq_len;
    let logits = forward_lazy_single(cfg, weights, pending, cache, token_offset);
    cache.seq_len += 1;
    let next_token_arr = argmax(&logits, None);
    let kv_refs = cache.collect_eval_refs();
    let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
    targets.push(&next_token_arr);
    targets.extend(kv_refs);
    // Submit step N+1 to the GPU before waiting for step N.
    // The GPU will sequence correctly: N+1 starts as soon as N's results are ready.
    async_eval(&targets);

    // Materialise the pending (step N) token.  Because `async_eval` was called
    // in the previous `start_greedy_pipeline` / `advance_greedy_pipeline`, the GPU
    // has been working on this token the entire time the CPU was building N+1's
    // graph above — so `eval` is typically a no-op barrier.
    eval(&[pending]);
    let tok = pending.data_u32().first().copied().unwrap_or(0);

    (tok, next_token_arr)
}

/// Decode one token: forward pass for a single token and return sampled ID.
///
/// When `temperature` is 0.0, uses GPU argmax (greedy).  When > 0.0, evals
/// logits to CPU and samples from the temperature-scaled categorical
/// distribution.  The caller must pass a per-request `rng` for reproducibility.
pub fn decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
    temperature: f32,
    rng: &mut Xorshift64,
) -> u32 {
    let token_offset = cache.seq_len;
    let logits = forward(cfg, weights, &[last_token], cache, token_offset);
    cache.seq_len += 1;

    let kv_refs = cache.collect_eval_refs();

    if temperature > 0.0 {
        // Eval logits + KV buffers to CPU, then sample with temperature.
        // Each append() adds a slice_update graph node to every layer's K/V buffer;
        // materialising here prevents O(N²) graph traversal over the decode loop.
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&logits);
        targets.extend(kv_refs);
        eval(&targets);
        let logits_data = logits.data_f32();
        sample_categorical(logits_data, temperature, rng)
    } else {
        // Greedy path: GPU argmax, no CPU data movement.
        let token_arr = argmax(&logits, None);
        let mut targets: Vec<&MlxArray> = Vec::with_capacity(1 + kv_refs.len());
        targets.push(&token_arr);
        targets.extend(kv_refs);
        eval(&targets);
        token_arr.data_u32().first().copied().unwrap_or(0)
    }
}
