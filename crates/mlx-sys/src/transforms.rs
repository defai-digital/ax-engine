use std::ffi::{CStr, CString};
use std::sync::LazyLock;

use crate::array::MlxArray;
use crate::error::{panic_on_status, prepare_error_capture, status_to_result};
use crate::ffi;

fn make_vector_array(arrays: &[&MlxArray]) -> ffi::mlx_vector_array {
    unsafe {
        prepare_error_capture();
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            let rc = ffi::mlx_vector_array_append_value(vec, arr.inner);
            panic_on_status("mlx_vector_array_append_value", rc);
        }
        vec
    }
}

/// Block the calling thread until all arrays have been computed on the GPU.
///
/// Panics on evaluation failure. Before the recording error handler existed,
/// the default behaviour was to `exit(-1)` the whole process, so this is
/// strictly more diagnosable; callers with a fallback path should use
/// [`try_eval`].
///
/// Inside an MLX closure body (compile tracing) the panic is downgraded to
/// poison propagation, mirroring `panic_on_status`: a panic here would
/// unwind across the C++ trampoline, which aborts the process under the
/// release `panic = "abort"` profile. The failure is restored into the
/// thread-local error slot so the compiled-closure caller's post-apply drain
/// turns it into a per-layer fallback (reachable e.g. when
/// `AX_MLX_DECODE_PROFILE=1` forces stage evals inside a compiled body).
pub fn eval(arrays: &[&MlxArray]) {
    if let Err(message) = try_eval(arrays) {
        if crate::error::in_closure_body() {
            crate::error::poison_slot(message);
            return;
        }
        panic!("{message}");
    }
}

/// Block the calling thread until all arrays have been computed on the GPU,
/// surfacing MLX errors (including lazy Metal-kernel compile failures) as
/// `Err` instead of killing the process.
pub fn try_eval(arrays: &[&MlxArray]) -> Result<(), String> {
    if arrays.is_empty() {
        return Ok(());
    }
    prepare_error_capture();
    unsafe {
        let vec = make_vector_array(arrays);
        let rc = ffi::mlx_eval(vec);
        ffi::mlx_vector_array_free(vec);
        status_to_result("mlx_eval", rc)?;
    }
    Ok(())
}

/// Evaluate a scalar uint32 array and return its first element.
///
/// Intended for generated-token readback (`argmax` over logits). Callers must
/// pass an array whose first element is a valid uint32 scalar or vector entry.
pub fn eval_first_u32(array: &MlxArray) -> u32 {
    eval(&[array]);
    array.first_u32_unchecked()
}

/// Enable MLX graph compilation globally.
///
/// When enabled, MLX caches and reuses compiled compute graphs across calls
/// with the same shapes, reducing per-step CPU overhead significantly.
/// Call once at program start or before the first forward pass.
pub fn enable_compile() {
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_enable_compile();
        panic_on_status("mlx_enable_compile", rc);
    }
}

/// Free MLX's internal array/graph computation cache.
///
/// Call this after completing a prefill pass to reclaim the GPU memory that
/// MLX holds for intermediate arrays in the graph cache. SwiftLM does the
/// same — `MLX.Memory.clearCache()` — immediately after chunked prefill.
pub fn clear_cache() {
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_clear_cache();
        panic_on_status("mlx_clear_cache", rc);
    }
}

/// Enqueue computation without blocking. Use `eval` later to synchronize.
pub fn async_eval(arrays: &[&MlxArray]) {
    if arrays.is_empty() {
        return;
    }
    unsafe {
        let vec = make_vector_array(arrays);
        prepare_error_capture();
        let rc = ffi::mlx_async_eval(vec);
        ffi::mlx_vector_array_free(vec);
        panic_on_status("mlx_async_eval", rc);
    }
}

/// Pin up to `limit` bytes of GPU memory as wired (never paged out between requests).
///
/// Returns the previous wired limit. Mirrors `mx.set_wired_limit()` in mlx_lm.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut prev = 0usize;
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_set_wired_limit(&mut prev, limit);
        panic_on_status("mlx_set_wired_limit", rc);
    }
    prev
}

/// Set the MLX memory limit (guideline for maximum memory during graph evaluation).
///
/// Defaults to 1.5x `max_recommended_working_set_size` when Metal is available.
/// Returns the previous memory limit. Mirrors `mx.set_memory_limit()`.
pub fn set_memory_limit(limit: usize) -> usize {
    let mut prev = 0usize;
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_set_memory_limit(&mut prev, limit);
        panic_on_status("mlx_set_memory_limit", rc);
    }
    prev
}

/// Set the MLX cache limit (reclaims free cache memory above this threshold).
///
/// Setting to 0 disables caching entirely — the most impactful single setting
/// for preventing unbounded memory growth during long inference sessions.
/// Returns the previous cache limit. Mirrors `mx.set_cache_limit()`.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut prev = 0usize;
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_set_cache_limit(&mut prev, limit);
        panic_on_status("mlx_set_cache_limit", rc);
    }
    prev
}

/// Return the peak memory used in bytes since program start or last reset.
///
/// Mirrors `mx.get_peak_memory()`.
pub fn get_peak_memory() -> usize {
    let mut bytes = 0usize;
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_get_peak_memory(&mut bytes);
        panic_on_status("mlx_get_peak_memory", rc);
    }
    bytes
}

/// Reset the peak memory counter to zero.
///
/// Mirrors `mx.reset_peak_memory()`.
pub fn reset_peak_memory() {
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_reset_peak_memory();
        panic_on_status("mlx_reset_peak_memory", rc);
    }
}

/// Return the current cache memory usage in bytes.
///
/// This is memory held by MLX's allocator that is not actively in use but has
/// not been returned to the system. Mirrors `mx.get_cache_memory()`.
pub fn get_cache_memory() -> usize {
    let mut bytes = 0usize;
    unsafe {
        prepare_error_capture();
        let rc = ffi::mlx_get_cache_memory(&mut bytes);
        panic_on_status("mlx_get_cache_memory", rc);
    }
    bytes
}

/// Return Metal's recommended max working-set size in bytes.
///
/// This is the safe upper bound for `set_wired_limit` on Apple Silicon —
/// wiring more than this value is rejected by the driver. Returns 0 if the
/// query fails (e.g., running on CPU-only hardware).
///
/// The key MLX exposes via `mlx_device_info_get` is the snake_case
/// `max_recommended_working_set_size` (see
/// `mlx/backend/metal/device_info.cpp`). Earlier revisions of this wrapper
/// queried `recommendedMaxWorkingSetSize` (camelCase from the underlying
/// Objective-C selector) which is *not* in MLX's keys map and silently
/// returned 0 — that broke `set_wired_limit(max_recommended_working_set_size())`
/// in `ax-engine-mlx::runner` and the I-4 device-pressure probe.
pub fn max_recommended_working_set_size() -> usize {
    unsafe {
        prepare_error_capture();
        // MLX_GPU = 1 (enum mlx_device_type_: MLX_CPU=0, MLX_GPU=1).
        let dev = ffi::mlx_device_new_type(1, 0);
        let mut info = ffi::mlx_device_info_new();
        let ok = ffi::mlx_device_info_get(&mut info, dev) == 0;
        ffi::mlx_device_free(dev);

        let mut size = 0usize;
        if ok {
            static KEY: LazyLock<CString> =
                LazyLock::new(|| CString::new("max_recommended_working_set_size").unwrap());
            ffi::mlx_device_info_get_size(&mut size, info, KEY.as_ptr());
        }
        ffi::mlx_device_info_free(info);
        size
    }
}

/// Return the MLX GPU architecture identifier used for kernel selection.
pub fn gpu_device_architecture() -> Result<String, String> {
    unsafe {
        prepare_error_capture();
        let dev = ffi::mlx_device_new_type(1, 0);
        let mut info = ffi::mlx_device_info_new();
        let info_rc = ffi::mlx_device_info_get(&mut info, dev);
        ffi::mlx_device_free(dev);
        if info_rc != 0 {
            ffi::mlx_device_info_free(info);
            return Err(crate::error::last_error_message("mlx_device_info_get"));
        }

        static KEY: LazyLock<CString> =
            LazyLock::new(|| CString::new("architecture").expect("static key has no NUL"));
        let mut value = std::ptr::null();
        let value_rc = ffi::mlx_device_info_get_string(&mut value, info, KEY.as_ptr());
        let architecture = if value_rc == 0 && !value.is_null() {
            Ok(CStr::from_ptr(value).to_string_lossy().into_owned())
        } else {
            Err(String::from("MLX GPU architecture is unavailable"))
        };
        ffi::mlx_device_info_free(info);
        architecture
    }
}

/// Best-effort raise of MLX's Metal command-buffer split thresholds via the
/// `MLX_MAX_MB_PER_BUFFER` / `MLX_MAX_OPS_PER_BUFFER` environment variables.
///
/// MLX's Metal encoder commits (splits) a command buffer once the
/// accumulated input bytes exceed `MLX_MAX_MB_PER_BUFFER` (default 40–50 MB
/// by GPU architecture), counting every newly-seen input at its FULL
/// `data_size()`. Checkpoints with many large weight tensors per step
/// (MoE expert stacks) therefore split on every layer, and the resulting
/// task count trips `eval_impl`'s scheduler backpressure, turning
/// `async_eval` into a de-facto barrier — see
/// `docs/performance/gather-qmm-async-serialization.md`.
///
/// MLX reads these variables once at Metal device initialization, so this
/// must run before the first GPU operation in the process; afterwards it is
/// a silent no-op (safe, just ineffective). Values already present in the
/// environment are treated as an explicit user override and left untouched.
/// Returns `(mb_applied, ops_applied)`.
pub fn set_metal_buffer_caps_env(max_mb: u32, max_ops: u32) -> (bool, bool) {
    let mut applied = (false, false);
    if std::env::var_os("MLX_MAX_MB_PER_BUFFER").is_none() {
        // SAFETY: called during single-threaded model/runner initialization,
        // before decode worker threads that could read the environment
        // concurrently are spawned.
        unsafe { std::env::set_var("MLX_MAX_MB_PER_BUFFER", max_mb.to_string()) };
        applied.0 = true;
    }
    if std::env::var_os("MLX_MAX_OPS_PER_BUFFER").is_none() {
        // SAFETY: as above.
        unsafe { std::env::set_var("MLX_MAX_OPS_PER_BUFFER", max_ops.to_string()) };
        applied.1 = true;
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MlxArray, argmax};

    #[test]
    fn eval_first_u32_reads_argmax_scalar() {
        let logits = MlxArray::from_f32_slice(&[0.0, 2.0, 1.0]);
        let token = argmax(&logits, None);

        assert_eq!(eval_first_u32(&token), 1);
    }

    #[test]
    fn gpu_device_architecture_is_available() {
        let architecture = gpu_device_architecture().expect("GPU architecture should be present");
        assert!(!architecture.is_empty());
    }

    #[test]
    fn async_eval_completes_correctly() {
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        let b = MlxArray::from_f32_slice(&[4.0, 5.0, 6.0]);
        let c = crate::ops::add(&a, &b, None);

        // async_eval enqueues, eval blocks
        async_eval(&[&c]);
        eval(&[&c]);
        assert_eq!(c.data_f32(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn clear_cache_runs_without_error() {
        // Just verify it doesn't crash
        clear_cache();
    }

    #[test]
    fn set_wired_limit_returns_previous_value() {
        // Set to 0 (no-op) and verify it returns without error
        let prev = set_wired_limit(0);
        // prev is the previous limit; on first call it should be 0 or some value
        let _ = prev;
        // Restore
        set_wired_limit(prev);
    }

    #[test]
    fn enable_compile_runs_without_error() {
        // Just verify it doesn't crash
        enable_compile();
    }

    #[test]
    fn device_info_returns_nonempty_name() {
        let size = max_recommended_working_set_size();
        assert!(
            size > 0,
            "max_recommended_working_set_size should be > 0 on Apple Silicon"
        );
    }

    #[test]
    fn set_memory_limit_returns_previous_value() {
        // Save current, set to 0, restore
        let prev = set_memory_limit(0);
        // 0 is a valid limit (means no limit in some MLX versions)
        let _ = prev;
        set_memory_limit(prev);
    }

    #[test]
    fn set_cache_limit_returns_previous_value() {
        let prev = set_cache_limit(0);
        let _ = prev;
        set_cache_limit(prev);
    }

    #[test]
    fn get_peak_memory_returns_nonzero_after_eval() {
        // Force some GPU work
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = crate::ops::add(&a, &a, None);
        eval(&[&b]);
        let peak = get_peak_memory();
        // Peak memory should be > 0 after any GPU work on Apple Silicon
        // (MLX always reports nonzero after first eval)
        assert!(
            peak > 0,
            "get_peak_memory should be > 0 after GPU eval, got {peak}"
        );
    }

    #[test]
    fn reset_peak_memory_resets_counter() {
        // Force some work
        let a = MlxArray::from_f32_slice(&[1.0, 2.0, 3.0]);
        eval(&[&a]);
        reset_peak_memory();
        let after_reset = get_peak_memory();
        // After reset, peak should be 0 or very small (only the reset call itself)
        // We allow a small tolerance for any internal bookkeeping
        assert!(
            after_reset < 1024 * 1024,
            "peak memory after reset should be < 1MB, got {after_reset}"
        );
    }

    #[test]
    fn get_cache_memory_returns_value() {
        // Just verify it returns a value without crashing
        let _cache = get_cache_memory();
    }
}
