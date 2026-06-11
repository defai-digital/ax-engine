use std::ffi::CString;

use crate::array::MlxArray;
use crate::error::{install_recoverable_error_handler, last_error_message, take_last_error};
use crate::ffi;

fn make_vector_array(arrays: &[&MlxArray]) -> ffi::mlx_vector_array {
    unsafe {
        let vec = ffi::mlx_vector_array_new();
        for arr in arrays {
            ffi::mlx_vector_array_append_value(vec, arr.inner);
        }
        vec
    }
}

/// Block the calling thread until all arrays have been computed on the GPU.
///
/// Panics on evaluation failure. Before the recording error handler existed,
/// mlx-c's default handler would `exit(-1)` instead, so this is strictly more
/// diagnosable; callers with a fallback path should use [`try_eval`].
pub fn eval(arrays: &[&MlxArray]) {
    if let Err(message) = try_eval(arrays) {
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
    install_recoverable_error_handler();
    let _ = take_last_error();
    unsafe {
        let vec = make_vector_array(arrays);
        let rc = ffi::mlx_eval(vec);
        ffi::mlx_vector_array_free(vec);
        if rc != 0 {
            return Err(last_error_message("mlx_eval"));
        }
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
        ffi::mlx_enable_compile();
    }
}

/// Free MLX's internal array/graph computation cache.
///
/// Call this after completing a prefill pass to reclaim the GPU memory that
/// MLX holds for intermediate arrays in the graph cache. SwiftLM does the
/// same — `MLX.Memory.clearCache()` — immediately after chunked prefill.
pub fn clear_cache() {
    unsafe {
        ffi::mlx_clear_cache();
    }
}

/// Enqueue computation without blocking. Use `eval` later to synchronize.
pub fn async_eval(arrays: &[&MlxArray]) {
    if arrays.is_empty() {
        return;
    }
    unsafe {
        let vec = make_vector_array(arrays);
        ffi::mlx_async_eval(vec);
        ffi::mlx_vector_array_free(vec);
    }
}

/// Pin up to `limit` bytes of GPU memory as wired (never paged out between requests).
///
/// Returns the previous wired limit. Mirrors `mx.set_wired_limit()` in mlx_lm.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut prev = 0usize;
    unsafe {
        ffi::mlx_set_wired_limit(&mut prev, limit);
    }
    prev
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
        // MLX_GPU = 1 (enum mlx_device_type_: MLX_CPU=0, MLX_GPU=1).
        let dev = ffi::mlx_device_new_type(1, 0);
        let mut info = ffi::mlx_device_info_new();
        let ok = ffi::mlx_device_info_get(&mut info, dev) == 0;
        ffi::mlx_device_free(dev);

        let mut size = 0usize;
        if ok {
            let key = CString::new("max_recommended_working_set_size").unwrap();
            ffi::mlx_device_info_get_size(&mut size, info, key.as_ptr());
        }
        ffi::mlx_device_info_free(info);
        size
    }
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
}
