use std::ptr;

use crate::ffi;

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
        unsafe {
            Self {
                inner: ffi::mlx_default_gpu_stream_new(),
                owns: false,
            }
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
