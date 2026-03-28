//! Forward pass evaluation and logits access.

use ax_engine_core::model::WeightStore;

use crate::types::*;

/// Run the forward pass (evaluate tokens).
///
/// Processes `n_tokens` tokens starting at position `n_past` in the context.
/// Returns 0 on success, negative on error.
#[unsafe(no_mangle)]
pub extern "C" fn llama_eval(
    ctx: *mut LlamaContext,
    tokens: *const LlamaToken,
    n_tokens: i32,
    n_past: i32,
) -> i32 {
    if ctx.is_null() || tokens.is_null() || n_tokens <= 0 || n_past < 0 {
        return -1;
    }

    let ctx = unsafe { &mut *ctx };
    if ctx.model.is_null() {
        return -1;
    }

    let token_count = n_tokens as usize;
    let n_past = n_past as usize;
    let token_slice = unsafe { std::slice::from_raw_parts(tokens, token_count) };

    // Keep shim state synchronized with explicit caller-provided history length.
    // - n_past == current_seq: continue without rewinding.
    // - 0 <= n_past < current_seq: trim/rewind to that position.
    // - n_past > current_seq: invalid continuation state.
    let current_pos = ctx.kv.seq_len();
    if n_past > current_pos {
        tracing::error!(
            "llama_eval: invalid n_past={n_past}, context has only {current_pos} tokens"
        );
        return -1;
    }
    if n_past != current_pos {
        ctx.kv.truncate_to(n_past);
    }

    let position_after = match n_past.checked_add(token_count) {
        Some(v) => v,
        None => {
            tracing::error!("llama_eval: n_past + n_tokens overflow");
            return -1;
        }
    };
    if position_after > ctx.n_ctx as usize {
        tracing::error!(
            "llama_eval: token window exceeds context: n_past={n_past}, n_tokens={token_count}, n_ctx={}",
            ctx.n_ctx
        );
        return -1;
    }

    for &token in token_slice {
        if token < 0 {
            tracing::error!("llama_eval: negative token ID: {token}");
            return -1;
        }
    }

    let model_ref = unsafe { &*ctx.model };
    let weights = WeightStore::new(&model_ref.mapped);
    let tokens_u32: Vec<u32> = token_slice.iter().map(|&t| t as u32).collect();
    ctx.logits.fill(0.0);
    if let Err(e) =
        model_ref
            .model
            .forward_batch(&tokens_u32, &mut ctx.kv, &weights, &mut ctx.logits)
    {
        tracing::error!("llama_eval: batched forward pass failed: {e}");
        return -1;
    }

    // `forward_batch` advances the shared KV cache; use it as truth for position.
    ctx.position = ctx.kv.seq_len();
    if ctx.position != position_after {
        tracing::error!(
            "llama_eval: unexpected KV position after eval: expected {position_after}, got {}",
            ctx.position
        );
        return -1;
    }
    0
}

/// Get logits output from the last eval.
///
/// Returns pointer to float array of size n_vocab.
/// The pointer is valid until the next call to llama_eval or llama_free.
#[unsafe(no_mangle)]
pub extern "C" fn llama_get_logits(ctx: *mut LlamaContext) -> *mut f32 {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*ctx).logits.as_mut_ptr() }
}
