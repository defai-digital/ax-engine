//! MLX GPU stream wrapper.
//!
//! # Ownership and thread-local behavior (invariant I-3)
//!
//! Upstream MLX 0.31 (`mlx-c` 0.6) keeps **one default GPU stream per device
//! per OS thread**. The stream index returned by `mlx_default_gpu_stream_new`
//! is the index of the thread-local default; calling it from a different
//! thread lazily creates a fresh default for *that* thread. Two threads
//! never share the same default stream — they cannot, because MLX's Metal
//! command encoder for a stream index is registered on the thread that
//! first observed that index.
//!
//! [`MlxStream`] currently declares `Send + Sync` so call sites can pass
//! stream handles through Rust-side channels. **This is a risk boundary,
//! not a guarantee.** The wrapper is safe to send/share at the type level
//! because all FFI operations that read the stream's `inner` field are
//! `unsafe` and the caller is responsible for ensuring they run on a
//! thread that has registered an encoder for that stream index. Concretely:
//!
//! 1. **`set_as_default()` is thread-local.** Calling it on thread A only
//!    sets thread A's default. Thread B remains on its own default.
//! 2. **Newly created streams (`new_gpu()`) register their Metal command
//!    encoder only on the creating thread.** Passing the handle to another
//!    thread and calling `set_as_default()` there does *not* register that
//!    stream's encoder on the new thread; subsequent MLX ops on the new
//!    thread will fall back to that thread's own default, silently
//!    bypassing the dedicated stream's ordering guarantees.
//! 3. **`mlx_eval`, `mlx_clear_cache`, and host buffer reads inherit the
//!    calling thread's default stream.** They do *not* take a stream
//!    argument; the active default is implicit. Mixing eval calls across
//!    threads without explicit stream pinning produces undefined ordering
//!    relative to in-flight ops.
//!
//! # What this module guarantees
//!
//! - Drop reclaims an owned (`new_gpu`-created) stream's MLX handle exactly
//!   once. Default-stream handles (`default_gpu`) are non-owning and do not
//!   call `mlx_stream_free` on drop.
//! - `Send`/`Sync` are correct in the narrow technical sense: the wrapper
//!   is a pointer pair (`ffi::mlx_stream`, `bool owns`) with no
//!   thread-affine Rust state. The thread-affinity lives inside MLX, not
//!   inside this struct.
//!
//! # What this module does NOT guarantee
//!
//! - That a `&MlxStream` shared with another thread will dispatch ops
//!   through that stream on the other thread. It will not (see point 2
//!   above).
//! - That `clear_cache` on a worker thread affects another thread's
//!   in-flight evaluation.
//! - That a stream created on thread A and freed on thread B is safe in
//!   every MLX release. mlx-c 0.6 tolerates this for the GPU stream type
//!   used here, but stricter releases may change that.
//!
//! # Migration to `!Send` is a separate ADR
//!
//! A type-level migration (for example, an `MlxStream<'_, Token>`
//! borrow-checker model that pins each stream to its creating thread)
//! would close the gap structurally. ADR-007 explicitly defers that
//! migration: the audit lives in this module + `docs/MLX-BACKEND.md`,
//! and a follow-up ADR can propose the type change once we have AX-
//! specific evidence of a regression that the audit alone cannot catch.
//!
//! See `.internal/audit/PHASE3-MLX-STREAM-OWNERSHIP-AUDIT.md` for the
//! call-site inventory and blast-radius estimate.

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
/// Wraps `mlx_stream`. Operations dispatched to the same stream execute in
/// order. **See the module-level docs for the thread-affinity contract that
/// the `Send + Sync` impls below do *not* enforce structurally.** Callers
/// passing a handle across threads are responsible for ensuring the
/// receiving thread has registered an encoder for the stream's index.
pub struct MlxStream {
    pub(crate) inner: ffi::mlx_stream,
    owns: bool,
}

// SAFETY: The struct is a pointer pair with no Rust-side thread-affine
// state. The thread affinity lives inside MLX's command-encoder registry,
// not inside this struct. See module-level docs for the call-site contract.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Invariant I-3 regression probe: MLX maintains one default GPU stream
    /// per OS thread. If two threads share the same default stream
    /// (`inner.ctx` pointer equal), then `set_as_default` semantics no longer
    /// hold and the audit at
    /// `.internal/audit/PHASE3-MLX-STREAM-OWNERSHIP-AUDIT.md` must be
    /// revisited before any code relies on cross-thread stream pinning.
    #[test]
    fn cross_thread_default_streams_are_distinct() {
        let main_stream = MlxStream::default_gpu();
        let main_ctx = main_stream.inner.ctx as usize;

        let worker_ctx: usize = thread::spawn(|| {
            let worker_stream = MlxStream::default_gpu();
            worker_stream.inner.ctx as usize
        })
        .join()
        .expect("worker thread must complete cleanly");

        assert_ne!(
            main_ctx, worker_ctx,
            "MLX default GPU stream pointers were equal across threads. \
             Upstream MLX 0.31 guarantees one default-per-thread; if this \
             changes, the I-3 audit and Send/Sync impls must be revisited."
        );
    }

    /// Newly created streams (`new_gpu`) own their handle; default streams
    /// do not. This is a structural property that Drop relies on; if it
    /// regresses, the wrong stream might be freed on drop.
    #[test]
    fn new_gpu_owns_its_handle_default_gpu_does_not() {
        let default = MlxStream::default_gpu();
        let owned = MlxStream::new_gpu();
        assert!(!default.owns, "default_gpu must not claim ownership");
        assert!(owned.owns, "new_gpu must claim ownership");
        // Pointers should also differ: a dedicated stream is a separate
        // index from the thread's default.
        assert_ne!(
            default.inner.ctx as usize, owned.inner.ctx as usize,
            "new_gpu must allocate a distinct stream from the thread default"
        );
    }
}
