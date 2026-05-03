use mlx_sys::{argmax, async_eval, clear_cache, eval};

use crate::kv_cache::MlxKVCache;
use crate::model::{ModelConfig, forward};
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

/// Decode one token: forward pass for a single token and return sampled ID.
pub fn decode_step(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    last_token: u32,
    cache: &mut MlxKVCache,
) -> u32 {
    let token_offset = cache.seq_len;
    let logits = forward(cfg, weights, &[last_token], cache, token_offset);
    cache.seq_len += 1;

    let token_arr = argmax(&logits, None);
    eval(&[&token_arr]);
    token_arr.data_u32().first().copied().unwrap_or(0)
}
