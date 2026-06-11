//! Per-layer decode compile POC (Track C2).
//!
//! This module provides infrastructure for compiling per-layer decode closures
//! that include KV-cache mutation. The compiled closure can be reused across
//! decode steps without per-step re-tracing.
//!
//! ## Design
//!
//! The POC answers these questions from the tech spec:
//!
//! 1. Can the compiled closure accept the current token hidden state and
//!    KV-cache inputs and return updated cache arrays without per-step re-tracing?
//! 2. Does `shapeless=true` compilation preserve correctness across Gemma
//!    sliding/full attention and Qwen linear/full attention decode steps?
//! 3. Does the compiled graph reuse across decode steps without tracing prompt
//!    length into a unique graph every step?
//! 4. Are there aliasing or mutation-safety issues with returned KV-cache arrays?
//!
//! ## Usage
//!
//! The compiled closure is cached per-thread. On first use at a given layer,
//! the closure is compiled with `shapeless=true`. Subsequent decode steps
//! reuse the compiled graph.
//!
//! ## Safety
//!
//! KV-cache arrays are returned as new arrays (not mutated in place). The
//! caller is responsible for updating the cache state with the returned arrays.
//! This avoids aliasing issues with compiled graphs.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;

use mlx_sys::{MlxArray, MlxClosure, MlxVectorArray};

/// Per-layer decode closure cache.
///
/// Maps `(layer_index, thread_id)` to a compiled closure that performs
/// a single transformer layer's decode step including KV-cache update.
static LAYER_DECODE_CACHE: OnceLock<Mutex<HashMap<(usize, ThreadId), MlxClosure>>> =
    OnceLock::new();

/// Apply a per-layer decode closure, compiling it on first use.
///
/// The closure takes:
/// - `inputs[0]`: hidden state `[1, 1, hidden_dim]`
/// - `inputs[1]`: KV cache key `[1, seq_len, kv_heads, kv_dim]`
/// - `inputs[2]`: KV cache value `[1, seq_len, kv_heads, kv_dim]`
///
/// And returns:
/// - `outputs[0]`: updated hidden state `[1, 1, hidden_dim]`
/// - `outputs[1]`: updated KV cache key `[1, seq_len+1, kv_heads, kv_dim]`
/// - `outputs[2]`: updated KV cache value `[1, seq_len+1, kv_heads, kv_dim]`
///
/// The compiled closure is cached per `(layer_index, thread_id)` and reused
/// across decode steps. `shapeless=true` allows the closure to accept
/// different sequence lengths without recompilation.
pub fn apply_layer_decode(
    layer_index: usize,
    hidden: &MlxArray,
    kv_key: &MlxArray,
    kv_value: &MlxArray,
    layer_fn: impl Fn(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
) -> Option<Vec<MlxArray>> {
    let cache = LAYER_DECODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();

    let guard = cache.lock().ok()?;
    if let Some(closure) = guard.get(&(layer_index, tid)) {
        return closure.try_apply(&[hidden, kv_key, kv_value]).ok();
    }
    drop(guard);

    let mut guard = cache.lock().ok()?;
    if let std::collections::hash_map::Entry::Vacant(slot) = guard.entry((layer_index, tid)) {
        let closure = MlxClosure::new_dyn(layer_fn);
        if let Ok(compiled) = closure.compile(true) {
            let result = compiled.try_apply(&[hidden, kv_key, kv_value]).ok();
            slot.insert(compiled);
            return result;
        }
    }

    None
}

/// Clear the per-layer decode closure cache.
///
/// This is useful for testing or when switching models.
pub fn clear_layer_decode_cache() {
    if let Some(cache) = LAYER_DECODE_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_cache_does_not_panic() {
        clear_layer_decode_cache();
    }
}
