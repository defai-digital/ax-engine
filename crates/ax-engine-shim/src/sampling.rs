//! Sampling functions: temperature, top-k, top-p, repetition penalty, and token selection.

use ax_engine_core::sampling::{self as core_sampling};

use crate::types::*;

fn is_valid_repetition_penalty(penalty: f32) -> bool {
    penalty.is_finite() && penalty > 0.0
}

/// Apply temperature scaling to logits.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_temperature(
    _ctx: *mut LlamaContext,
    logits: *mut f32,
    n_vocab: i32,
    temp: f32,
) {
    if logits.is_null() || n_vocab <= 0 || !temp.is_finite() || temp <= 0.0 {
        return;
    }
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, n_vocab as usize) };
    core_sampling::temperature::apply_temperature(logits, temp);
}

/// Apply top-k filtering to logits.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_top_k(
    _ctx: *mut LlamaContext,
    logits: *mut f32,
    n_vocab: i32,
    k: i32,
) {
    if logits.is_null() || n_vocab <= 0 {
        return;
    }
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, n_vocab as usize) };
    core_sampling::top_k::apply_top_k(logits, k);
}

/// Apply top-p (nucleus) filtering to logits.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_top_p(
    _ctx: *mut LlamaContext,
    logits: *mut f32,
    n_vocab: i32,
    p: f32,
) {
    if logits.is_null() || n_vocab <= 0 {
        return;
    }
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, n_vocab as usize) };
    core_sampling::top_p::apply_top_p(logits, p);
}

/// Apply repetition penalty to logits.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_repetition_penalty(
    _ctx: *mut LlamaContext,
    logits: *mut f32,
    n_vocab: i32,
    last_tokens: *const LlamaToken,
    last_n: i32,
    penalty: f32,
) {
    if logits.is_null()
        || n_vocab <= 0
        || last_tokens.is_null()
        || last_n <= 0
        || !is_valid_repetition_penalty(penalty)
    {
        return;
    }
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, n_vocab as usize) };
    let tokens_i32 = unsafe { std::slice::from_raw_parts(last_tokens, last_n as usize) };
    // Convert i32 tokens to u32 for the core API
    let tokens_u32: Vec<u32> = tokens_i32
        .iter()
        .filter(|&&t| t >= 0)
        .map(|&t| t as u32)
        .collect();
    core_sampling::repetition::apply_repetition_penalty(logits, &tokens_u32, penalty);
}

/// Greedy sampling: return the token with the highest logit.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_token_greedy(
    _ctx: *mut LlamaContext,
    logits: *const f32,
    n_vocab: i32,
) -> LlamaToken {
    if logits.is_null() || n_vocab <= 0 {
        return -1;
    }
    let logits = unsafe { std::slice::from_raw_parts(logits, n_vocab as usize) };
    core_sampling::argmax(logits) as LlamaToken
}

/// Sample a token from logits using softmax + weighted random selection.
///
/// Call this after applying temperature, top-k, top-p, etc.
/// Uses the context's RNG state for reproducibility.
#[unsafe(no_mangle)]
pub extern "C" fn llama_sample_token(
    ctx: *mut LlamaContext,
    logits: *mut f32,
    n_vocab: i32,
) -> LlamaToken {
    if ctx.is_null() || logits.is_null() || n_vocab <= 0 {
        return -1;
    }

    let ctx = unsafe { &mut *ctx };
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, n_vocab as usize) };

    // Softmax to convert logits to probabilities
    ax_engine_core::compute::softmax::softmax(logits);

    // Categorical sampling using context's RNG
    let r = xorshift64(&mut ctx.rng_state);
    let threshold = (r >> 11) as f64 / (1u64 << 53) as f64;

    let mut cumsum = 0.0f64;
    for (i, &p) in logits.iter().enumerate() {
        cumsum += p as f64;
        if cumsum > threshold {
            return i as LlamaToken;
        }
    }

    // Fallback: return last token (rounding errors)
    (n_vocab - 1) as LlamaToken
}

/// Xorshift64 PRNG.
fn xorshift64(state: &mut u64) -> u64 {
    if *state == 0 {
        *state = 1;
    }
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_repetition_penalty_rejects_zero() {
        assert!(!is_valid_repetition_penalty(0.0));
    }

    #[test]
    fn test_is_valid_repetition_penalty_rejects_non_finite_values() {
        assert!(!is_valid_repetition_penalty(f32::NAN));
        assert!(!is_valid_repetition_penalty(f32::INFINITY));
    }

    #[test]
    fn test_is_valid_repetition_penalty_accepts_positive_values() {
        assert!(is_valid_repetition_penalty(1.1));
    }
}
