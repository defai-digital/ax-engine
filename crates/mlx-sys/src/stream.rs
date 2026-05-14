use std::ptr;

use crate::ffi;

/// Get the current thread's default GPU stream without taking ownership.
///
/// MLX 0.31 keeps one default stream per device per thread. If the current
/// thread has no GPU default yet, this call lazily creates and registers one
/// for this thread.
pub(crate) fn default_gpu_raw() -> ffi::mlx_stream {
    unsafe { ffi::mlx_default_gpu_stream_new() }
}

/// A GPU compute stream.
///
/// Wraps `mlx_stream`. Operations dispatched to the same stream execute in order.
pub struct MlxStream {
    pub(crate) inner: ffi::mlx_stream,
    owns: bool,
}

unsafe impl Send for MlxStream {}
unsafe impl Sync for MlxStream {}

impl MlxStream {
    /// The default GPU stream (used by all ops when no stream is specified).
    pub fn default_gpu() -> Self {
        Self {
            inner: default_gpu_raw(),
            owns: false,
        }
    }

    /// Create a new dedicated GPU stream.
    ///
    /// Unlike `default_gpu()`, this allocates a fresh stream that is not shared
    /// with other callers. Setting it as default via `set_as_default()` avoids
    /// implicit cross-stream synchronization that the shared default stream can
    /// incur — this mirrors `mx.new_stream(mx.default_device())` in mlx_lm.
    ///
    /// Upstream MLX registers the Metal command encoder for a newly created GPU
    /// stream only in the thread that created the stream. Passing the stream to
    /// another thread and calling `set_as_default()` there does not register that
    /// stream index's encoder in MLX 0.31 / mlx-c 0.6.
    pub fn new_gpu() -> Self {
        unsafe {
            // MLX_GPU = 1 (enum mlx_device_type_: MLX_CPU=0, MLX_GPU=1)
            let dev = ffi::mlx_device_new_type(1, 0);
            let stream = ffi::mlx_stream_new_device(dev);
            ffi::mlx_device_free(dev);
            Self {
                inner: stream,
                owns: true,
            }
        }
    }

    /// Set this stream as the current thread's default for MLX operations.
    ///
    /// This is thread-local in upstream MLX. It does not make the stream usable
    /// from every worker thread, and it does not register a Metal command encoder
    /// for an existing stream index on a different thread.
    pub fn set_as_default(&self) {
        unsafe {
            ffi::mlx_set_default_stream(self.inner);
        }
    }
}

impl Drop for MlxStream {
    fn drop(&mut self) {
        if self.owns && !self.inner.ctx.is_null() {
            unsafe { ffi::mlx_stream_free(self.inner) };
            self.inner.ctx = ptr::null_mut();
        }
    }
}

impl Default for MlxStream {
    fn default() -> Self {
        Self::default_gpu()
    }
}
