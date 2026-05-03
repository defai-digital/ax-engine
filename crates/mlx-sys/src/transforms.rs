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
