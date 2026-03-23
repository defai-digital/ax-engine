//! Forward pass evaluation and logits access.

use ax_core::model::WeightStore;

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
    let model_ref = unsafe { &*ctx.model };
    let token_slice = unsafe { std::slice::from_raw_parts(tokens, n_tokens as usize) };
    let weights = WeightStore::new(&model_ref.mapped);

    for &token in token_slice {
        if token < 0 {
            tracing::error!("llama_eval: negative token ID: {token}");
            return -1;
        }
    }

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

    let position = n_past as usize + token_slice.len();
    ctx.position = position;
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
