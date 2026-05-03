use std::ffi::CString;

use crate::array::MlxArray;
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
pub fn eval(arrays: &[&MlxArray]) {
    if arrays.is_empty() {
        return;
    }
    unsafe {
        let vec = make_vector_array(arrays);
        ffi::mlx_eval(vec);
        ffi::mlx_vector_array_free(vec);
    }
}

/// Enable MLX graph compilation globally.
///
/// When enabled, MLX caches and reuses compiled compute graphs across calls
/// with the same shapes, reducing per-step CPU overhead significantly.
/// Call once at program start or before the first forward pass.
pub fn enable_compile() {
    unsafe { ffi::mlx_enable_compile(); }
}

/// Free MLX's internal array/graph computation cache.
///
/// Call this after completing a prefill pass to reclaim the GPU memory that
/// MLX holds for intermediate arrays in the graph cache. SwiftLM does the
/// same — `MLX.Memory.clearCache()` — immediately after chunked prefill.
pub fn clear_cache() {
    unsafe { ffi::mlx_clear_cache(); }
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
    unsafe { ffi::mlx_set_wired_limit(&mut prev, limit); }
    prev
}

/// Return Metal's `recommendedMaxWorkingSetSize` in bytes.
///
/// This is the safe upper bound for `set_wired_limit` on Apple Silicon —
/// wiring more than this value is rejected by the driver. Returns 0 if the
/// query fails (e.g., running on CPU-only hardware).
pub fn max_recommended_working_set_size() -> usize {
    unsafe {
        // MLX_GPU = 1 (enum mlx_device_type_: MLX_CPU=0, MLX_GPU=1).
        let dev = ffi::mlx_device_new_type(1, 0);
        let mut info = ffi::mlx_device_info_new();
        let ok = ffi::mlx_device_info_get(&mut info, dev) == 0;
        ffi::mlx_device_free(dev);

        let mut size = 0usize;
        if ok {
            let key = CString::new("recommendedMaxWorkingSetSize").unwrap();
            ffi::mlx_device_info_get_size(&mut size, info, key.as_ptr());
        }
        ffi::mlx_device_info_free(info);
        size
    }
}
