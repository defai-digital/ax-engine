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
use std::panic::AssertUnwindSafe;
use std::sync::{LazyLock, Mutex, OnceLock};
use std::thread::ThreadId;

use mlx_sys::{MlxArray, MlxClosure, MlxVectorArray};

/// Try applying a compiled closure, returning `Some(outputs)` on success.
///
/// This wraps `try_apply` with two layers of failure detection:
///
/// 1. **`catch_unwind`** (debug builds only): catches Rust panics from MLX
///    stream-registry invalidation. Ineffective under `panic = "abort"`.
/// 2. **MLX error slot check** (abort-safe): after a successful `try_apply`,
///    drains the MLX error capture slot. If MLX recorded a lazy error during
///    the apply (e.g. Metal command buffer failure that did not surface as
///    a non-zero FFI status), the result is treated as a failure so the
///    caller falls back to the imperative path.
///
/// The `catch_unwind` layer is kept because it provides defense-in-depth in
/// debug/test profiles. The error-slot check is the production safety net.
fn try_apply_with_abort_safety(
    closure: &MlxClosure,
    inputs: &[&MlxArray],
) -> Option<Vec<MlxArray>> {
    // Drain any stale error from the slot so we only see errors from *this* apply.
    let _ = mlx_sys::take_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| closure.try_apply(inputs).ok()))
        .ok()
        .flatten();
    // Abort-safe: filter out results where MLX recorded a lazy error during
    // the apply (e.g. Metal command buffer failure that did not surface as
    // a non-zero FFI status), so the caller falls back to the imperative path.
    result.filter(|_| mlx_sys::take_last_error().is_none())
}

/// Per-layer MoE decode closure cache.
///
/// Maps `(layer_index, thread_id)` to a compiled closure that performs
/// a single MoE layer's decode step (router output → expert forward →
/// weighted sum). Inputs are `(hidden, top_k_indices, top_k_weights)`.
// Value is `Option<MlxClosure>`: `Some` is a working compiled closure reused
// across decode steps; `None` records a layer whose compiled MoE closure failed
// to compile or apply, so we fall back to the imperative path permanently
// instead of retrying (and re-flooding MLX errors) on every step.
type MoeDecodeCache = OnceLock<Mutex<HashMap<(usize, ThreadId), Option<MlxClosure>>>>;
static LAYER_MOE_DECODE_CACHE: MoeDecodeCache = OnceLock::new();

/// Per-layer dense FFN decode closure cache.
///
/// Maps `(layer_index, thread_id)` to a compiled closure that performs
/// a single dense FFN layer's decode step (gate_up → split → activation →
/// down → optional post-norm). All weight tensors are passed as explicit
/// inputs to satisfy MLX's no-uncaptured-inputs contract.
// Value is `Option<MlxClosure>`: `Some` is a working compiled closure reused
// across decode steps; `None` records a layer whose compiled dense FFN
// closure failed to compile or apply, so we fall back to the imperative
// path permanently instead of retrying (and re-flooding MLX errors) on
// every step.
/// Per-layer dense FFN compiled closure cache.
///
/// Maps `(layer_index, thread_id)` to an optional compiled closure. `Some`
/// is a working closure; `None` marks a layer that permanently fell back
/// to the imperative path.
type DenseFfnCache = Mutex<HashMap<(usize, ThreadId), (Option<MlxClosure>, u64)>>;

static LAYER_DENSE_FFN_DECODE_CACHE: OnceLock<DenseFfnCache> = OnceLock::new();

/// Number of successful compilations before the cache entry is evicted and
/// recompiled. Prevents stale MLX stream-registry references from
/// accumulating in long-running processes.
///
/// Override via `AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD=<n>`.
static COMPILE_CACHE_REFRESH_THRESHOLD: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("AX_MLX_COMPILE_CACHE_REFRESH_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(10_000)
});

/// Per-layer Gemma4 dual-path (dense + expert) decode closure cache.
///
/// Maps `(layer_index, thread_id)` to a compiled closure that performs the
/// entire dual-path MoE block: dense sub-block + expert sub-block + combine.
#[allow(clippy::type_complexity)]
static LAYER_GEMMA4_DUAL_PATH_CACHE: OnceLock<
    Mutex<HashMap<(usize, ThreadId), Option<MlxClosure>>>,
> = OnceLock::new();

/// Apply a compiled MoE decode closure for a single layer.
///
/// `inputs` is the full positional input vector the compiled function depends
/// on. By contract `inputs[0..=2]` are `(hidden, top_k_indices, top_k_weights)`
/// and the remaining entries are the expert weights and optional shared-expert
/// output (see `flatten_compiled_moe_inputs`). Passing every dependency
/// explicitly is required: MLX-C 0.6.0 rejects compiling a function that
/// references arrays captured from the closure environment ("uncaptured
/// inputs is not allowed"), which previously made this path abort on the first
/// decode step.
///
/// The closure returns `outputs[0]`: the MoE block output `[1, 1, hidden_dim]`.
///
/// The compiled closure is cached per `(layer_index, thread_id)` and reused
/// across decode steps. `shapeless=true` is safe because hidden dim and top_k
/// are constant per model; the only varying dimension (seq) is always 1 in
/// decode.
///
/// Gated by `AX_MLX_MOE_LAYER_COMPILE` (default ON). Returns `None` when the
/// flag is off, compilation fails, or the closure cannot be applied.
pub fn apply_layer_moe_decode(
    layer_index: usize,
    inputs: &[&MlxArray],
    moe_fn: impl Fn(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
) -> Option<Vec<MlxArray>> {
    if !crate::fastpath::moe_layer_compile_enabled() {
        return None;
    }
    let cache = LAYER_MOE_DECODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();

    let mut guard = cache.lock().ok()?;
    // Scope the read so the immutable borrow is released before
    // we potentially call `guard.remove()`.
    let apply_result: Option<Option<Vec<MlxArray>>> =
        if let Some(entry) = guard.get(&(layer_index, tid)) {
            let closure_ref: Option<&MlxClosure> = match entry {
                Some(c) => Some(c),
                None => None,
            };
            if let Some(c) = closure_ref {
                Some(try_apply_with_abort_safety(c, inputs))
            } else {
                // Known-incompatible: return None directly.
                return None;
            }
        } else {
            None
        };
    if let Some(result) = apply_result {
        if result.is_none() {
            tracing::warn!(
                target = "ax_engine_mlx",
                path = "moe_decode",
                layer = layer_index,
                "compiled_closure_apply_failed; removing entry for recompilation"
            );
            guard.remove(&(layer_index, tid));
        }
        return result;
    }
    drop(guard);

    let mut guard = cache.lock().ok()?;
    if let std::collections::hash_map::Entry::Vacant(slot) = guard.entry((layer_index, tid)) {
        let closure = MlxClosure::new_dyn(moe_fn);
        if let Ok(compiled) = closure.compile(true) {
            // Use catch_unwind for the first apply as well — compilation
            // tracing may leave MLX in a state that panics on first apply.
            let result = try_apply_with_abort_safety(&compiled, inputs);
            // Cache the compiled closure only if its first apply produced
            // output. If it failed (e.g. a model whose MoE graph the compiled
            // path cannot shape-infer), record `None` so later steps fall back
            // to the imperative path without retrying and re-flooding errors.
            if result.is_none() {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "moe_decode",
                    layer = layer_index,
                    "compiled_closure_fallback"
                );
            }
            slot.insert(result.is_some().then_some(compiled));
            return result;
        }
        // Compilation itself failed: record incompatible so we do not recompile
        // on every decode step.
        tracing::warn!(
            target = "ax_engine_mlx",
            path = "moe_decode",
            layer = layer_index,
            "compiled_closure_fallback"
        );
        slot.insert(None);
    }

    None
}

/// Apply a compiled dense FFN decode closure for a single layer.
///
/// `inputs` is the full positional input vector the compiled function
/// depends on. By contract `inputs[0]` is the post-norm hidden state and
/// the remaining entries are the FFN weight tensors (gate_up, down,
/// optional post-norm) plus any captured scales/biases. Passing every
/// dependency explicitly is required: MLX rejects compiling a function
/// that references arrays captured from the closure environment.
///
/// The closure returns `outputs[0]`: the dense FFN output
/// `[1, 1, hidden_dim]`.
///
/// The compiled closure is cached per `(layer_index, thread_id)` and
/// reused across decode steps. `shapeless=true` is safe because hidden
/// dim is constant per model; the only varying dimension (seq) is always
/// 1 in decode.
///
/// Gated by `AX_MLX_DENSE_FFN_COMPILE` (default ON, kill-switch). Returns
/// `None` when the flag is off, compilation fails, or the closure cannot be
/// applied.
pub fn apply_layer_dense_ffn_decode(
    layer_index: usize,
    inputs: &[&MlxArray],
    ffn_fn: impl Fn(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
) -> Option<Vec<MlxArray>> {
    if !crate::fastpath::dense_ffn_compile_enabled() {
        return None;
    }
    let cache = LAYER_DENSE_FFN_DECODE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();
    let threshold = *COMPILE_CACHE_REFRESH_THRESHOLD;

    let mut guard = cache.lock().ok()?;
    // Scope the mutable borrow so it is released before we potentially
    // call `guard.remove()`. A present entry with a `None` closure means
    // permanently incompatible and is never retried, so eviction for
    // recompilation must remove the key rather than clear the closure
    // in place.
    let apply_result: Option<(Option<Vec<MlxArray>>, bool)> =
        if let Some((entry, generation)) = guard.get_mut(&(layer_index, tid)) {
            let closure_ref: Option<&MlxClosure> = match &*entry {
                Some(c) => Some(c),
                None => None,
            };
            if let Some(c) = closure_ref {
                let result = try_apply_with_abort_safety(c, inputs);
                let evict = if result.is_some() {
                    let new_generation = generation.wrapping_add(1);
                    *generation = new_generation;
                    // Threshold reached — evict the key so the next call
                    // recompiles.
                    new_generation.is_multiple_of(threshold)
                } else {
                    // Failed apply — evict the key so the next call recompiles.
                    true
                };
                Some((result, evict))
            } else {
                // Known-incompatible for this layer: skip straight to the
                // imperative fallback instead of re-attempting every decode
                // step.
                return None;
            }
        } else {
            None
        };
    if let Some((result, evict)) = apply_result {
        if result.is_none() {
            tracing::warn!(
                target = "ax_engine_mlx",
                path = "dense_ffn_decode",
                layer = layer_index,
                "compiled_closure_apply_failed; removing entry for recompilation"
            );
        }
        if evict {
            guard.remove(&(layer_index, tid));
        }
        return result;
    }
    drop(guard);

    let mut guard = cache.lock().ok()?;
    if let std::collections::hash_map::Entry::Vacant(slot) = guard.entry((layer_index, tid)) {
        let closure = MlxClosure::new_dyn(ffn_fn);
        if let Ok(compiled) = closure.compile(true) {
            let result = try_apply_with_abort_safety(&compiled, inputs);
            if result.is_none() {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "dense_ffn_decode",
                    layer = layer_index,
                    "compiled_closure_fallback"
                );
            }
            if result.is_some() {
                let generation: u64 = 1;
                // Leave the key vacant when the per-key generation threshold
                // is already reached so the next call recompiles; inserting a
                // `None` closure would mark the layer permanently
                // incompatible.
                if !generation.is_multiple_of(threshold) {
                    slot.insert((Some(compiled), generation));
                }
            } else {
                slot.insert((None, 0));
            }
            return result;
        }
        // Compilation itself failed: record incompatible so we do not
        // recompile on every decode step.
        tracing::warn!(
            target = "ax_engine_mlx",
            path = "dense_ffn_decode",
            layer = layer_index,
            "compiled_closure_fallback"
        );
        slot.insert((None, 0));
    }

    None
}

/// Clear the per-layer dense FFN decode closure cache.
pub fn clear_layer_dense_ffn_decode_cache() {
    if let Some(cache) = LAYER_DENSE_FFN_DECODE_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}

/// Clear the per-layer MoE decode closure cache.
pub fn clear_layer_moe_decode_cache() {
    if let Some(cache) = LAYER_MOE_DECODE_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}

/// Apply a compiled Gemma4 dual-path decode closure for a single layer.
///
/// `inputs` is the full positional input vector the compiled function depends
/// on: `[normed2, hidden, ...weights]`. Passing every dependency explicitly is
/// required: MLX rejects compiling a function that references arrays captured
/// from the closure environment.
///
/// The closure returns `outputs[0]`: the combined dual-path output
/// `[1, 1, hidden_dim]`.
///
/// Gated by `AX_MLX_MOE_LAYER_COMPILE` (reuses the MoE compile flag since the
/// optimization is analogous). Returns `None` when the flag is off, compilation
/// fails, or the closure cannot be applied.
pub fn apply_layer_gemma4_dual_path_decode(
    layer_index: usize,
    inputs: &[&MlxArray],
    dual_path_fn: impl Fn(&MlxVectorArray) -> Vec<MlxArray> + Send + 'static,
) -> Option<Vec<MlxArray>> {
    if !crate::fastpath::moe_layer_compile_enabled() {
        return None;
    }
    let cache = LAYER_GEMMA4_DUAL_PATH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let tid = std::thread::current().id();

    let mut guard = cache.lock().ok()?;
    // Scope the read so the immutable borrow is released before
    // we potentially call `guard.remove()`.
    let apply_result: Option<Option<Vec<MlxArray>>> =
        if let Some(entry) = guard.get(&(layer_index, tid)) {
            let closure_ref: Option<&MlxClosure> = match entry {
                Some(c) => Some(c),
                None => None,
            };
            if let Some(c) = closure_ref {
                Some(try_apply_with_abort_safety(c, inputs))
            } else {
                return None;
            }
        } else {
            None
        };
    if let Some(result) = apply_result {
        if result.is_none() {
            tracing::warn!(
                target = "ax_engine_mlx",
                path = "gemma4_dual_path",
                layer = layer_index,
                "compiled_closure_apply_failed; removing entry for recompilation"
            );
            guard.remove(&(layer_index, tid));
        }
        return result;
    }
    drop(guard);

    let mut guard = cache.lock().ok()?;
    if let std::collections::hash_map::Entry::Vacant(slot) = guard.entry((layer_index, tid)) {
        let closure = MlxClosure::new_dyn(dual_path_fn);
        if let Ok(compiled) = closure.compile(true) {
            let result = try_apply_with_abort_safety(&compiled, inputs);
            if result.is_none() {
                tracing::warn!(
                    target = "ax_engine_mlx",
                    path = "gemma4_dual_path",
                    layer = layer_index,
                    "compiled_closure_fallback"
                );
            }
            slot.insert(result.is_some().then_some(compiled));
            return result;
        }
        tracing::warn!(
            target = "ax_engine_mlx",
            path = "gemma4_dual_path",
            layer = layer_index,
            "compiled_closure_fallback"
        );
        slot.insert(None);
    }

    None
}

/// Clear the per-layer Gemma4 dual-path decode closure cache.
pub fn clear_layer_gemma4_dual_path_cache() {
    if let Some(cache) = LAYER_GEMMA4_DUAL_PATH_CACHE.get()
        && let Ok(mut guard) = cache.lock()
    {
        guard.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clear_moe_cache_does_not_panic() {
        clear_layer_moe_decode_cache();
    }

    #[test]
    fn test_clear_dense_ffn_cache_does_not_panic() {
        clear_layer_dense_ffn_decode_cache();
    }

    #[test]
    fn test_clear_gemma4_dual_path_cache_does_not_panic() {
        clear_layer_gemma4_dual_path_cache();
    }

    #[test]
    fn dense_ffn_refresh_threshold_evicts_key_for_recompilation() {
        use mlx_sys::MlxDtype;

        // Layer index far above any real model's layer count so this test
        // never collides with entries from other tests on the same thread.
        const LAYER: usize = 900_001;
        if !crate::fastpath::dense_ffn_compile_enabled() {
            return;
        }
        let tid = std::thread::current().id();
        let threshold = *COMPILE_CACHE_REFRESH_THRESHOLD;

        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let x = MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(&data),
            &[1, 1, 4],
            MlxDtype::Float32,
        );
        let ffn =
            |inputs: &MlxVectorArray| vec![mlx_sys::add(&inputs.get(0), &inputs.get(0), None)];

        let first = apply_layer_dense_ffn_decode(LAYER, &[&x], ffn);
        assert!(first.is_some(), "first apply should compile and succeed");

        // Fast-forward the per-key generation to one below the refresh
        // threshold so the next successful apply triggers the refresh.
        {
            let cache = LAYER_DENSE_FFN_DECODE_CACHE.get().expect("cache exists");
            let mut guard = cache.lock().expect("cache lock");
            let entry = guard
                .get_mut(&(LAYER, tid))
                .expect("entry cached after first apply");
            assert!(entry.0.is_some(), "cached closure should be live");
            entry.1 = threshold - 1;
        }

        let at_threshold = apply_layer_dense_ffn_decode(LAYER, &[&x], ffn);
        assert!(
            at_threshold.is_some(),
            "apply at refresh threshold should still succeed"
        );

        // The refresh must REMOVE the key: a present entry whose closure is
        // `None` means permanently incompatible and is never retried, which
        // would silently disable the compiled path for the process lifetime.
        {
            let cache = LAYER_DENSE_FFN_DECODE_CACHE.get().expect("cache exists");
            let guard = cache.lock().expect("cache lock");
            assert!(
                !guard.contains_key(&(LAYER, tid)),
                "refresh threshold must evict the key so the next call recompiles"
            );
        }

        let recompiled = apply_layer_dense_ffn_decode(LAYER, &[&x], ffn);
        assert!(
            recompiled.is_some(),
            "post-refresh apply should recompile, not fall back permanently"
        );
        {
            let cache = LAYER_DENSE_FFN_DECODE_CACHE.get().expect("cache exists");
            let mut guard = cache.lock().expect("cache lock");
            let entry = guard
                .get(&(LAYER, tid))
                .expect("recompiled entry should be cached");
            assert!(entry.0.is_some(), "recompiled closure should be live");
            guard.remove(&(LAYER, tid));
        }
    }
}
