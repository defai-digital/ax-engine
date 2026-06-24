//! MLX GPU stream wrapper.
//!
//! # Ownership and thread-local behavior (invariant I-3)
//!
//! Upstream MLX 0.31 keeps **one default GPU stream per device
//! per OS thread**. The stream returned by `mlx_default_gpu_stream_new`
//! is the thread-local default; calling it from a different
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
//! - Drop reclaims an owned stream wrapper exactly once. The ax_shim
//!   layer represents even default streams as heap-allocated wrapper objects
//!   around the scheduler-owned `mlx::core::Stream`, so wrapper ownership is
//!   separate from stream-index ownership.
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
//!   every MLX release. The current shim tolerates this for the GPU stream type
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

use std::cell::RefCell;
use std::ptr;

use crate::error::{last_error_message, panic_on_status, prepare_error_capture};
use crate::ffi;

struct CachedDefaultGpuStream {
    inner: ffi::mlx_stream,
}

impl CachedDefaultGpuStream {
    fn empty() -> Self {
        Self {
            inner: ffi::mlx_stream {
                ctx: ptr::null_mut(),
            },
        }
    }

    fn get_or_init(&mut self) -> ffi::mlx_stream {
        if self.inner.ctx.is_null() {
            prepare_error_capture();
            self.inner = unsafe { ffi::mlx_default_gpu_stream_new() };
            panic_on_null_stream("mlx_default_gpu_stream_new", self.inner);
        }
        self.inner
    }

    fn refresh(&mut self) -> ffi::mlx_stream {
        self.free();
        prepare_error_capture();
        self.inner = unsafe { ffi::mlx_default_gpu_stream_new() };
        panic_on_null_stream("mlx_default_gpu_stream_new", self.inner);
        self.inner
    }

    fn free(&mut self) {
        if !self.inner.ctx.is_null() {
            unsafe { ffi::mlx_stream_free(self.inner) };
            self.inner.ctx = ptr::null_mut();
        }
    }
}

impl Drop for CachedDefaultGpuStream {
    fn drop(&mut self) {
        self.free();
    }
}

thread_local! {
    static DEFAULT_GPU_STREAM: RefCell<CachedDefaultGpuStream> =
        RefCell::new(CachedDefaultGpuStream::empty());
}

fn fresh_default_gpu_stream() -> ffi::mlx_stream {
    prepare_error_capture();
    let stream = unsafe { ffi::mlx_default_gpu_stream_new() };
    panic_on_null_stream("mlx_default_gpu_stream_new", stream);
    stream
}

fn refresh_default_gpu_raw() -> ffi::mlx_stream {
    DEFAULT_GPU_STREAM.with(|stream| stream.borrow_mut().refresh())
}

/// Get the current thread's default GPU stream without taking ownership.
///
/// MLX 0.31 keeps one default stream per device per thread. If the current
/// thread has no GPU default yet, this call lazily creates and registers one
/// for this thread.
///
/// ax_shim allocates a wrapper object even for default-stream handles (see
/// `.internal/reference/SwiftLM/mlx-swift/Source/Cmlx/mlx-c/mlx/c/private/stream.h`).
/// The hot path calls this helper for every MLX op, so cache one wrapper per
/// OS thread instead of allocating and leaking a wrapper on every dispatch.
pub(crate) fn default_gpu_raw() -> ffi::mlx_stream {
    DEFAULT_GPU_STREAM.with(|stream| stream.borrow_mut().get_or_init())
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
            inner: fresh_default_gpu_stream(),
            owns: true,
        }
    }

    /// The default CPU stream. Owns the ax_shim wrapper and frees it on drop.
    ///
    /// Use this instead of calling `ffi::mlx_default_cpu_stream_new()` directly
    /// to avoid leaking the wrapper object. ax_shim allocates a heap wrapper even
    /// for default streams; callers are responsible for freeing it.
    pub fn default_cpu() -> Self {
        prepare_error_capture();
        let inner = unsafe { ffi::mlx_default_cpu_stream_new() };
        panic_on_null_stream("mlx_default_cpu_stream_new", inner);
        Self { inner, owns: true }
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
    /// stream index's encoder in MLX 0.31.
    pub fn new_gpu() -> Self {
        unsafe {
            // MLX_GPU = 1 (enum mlx_device_type_: MLX_CPU=0, MLX_GPU=1)
            prepare_error_capture();
            let dev = ffi::mlx_device_new_type(1, 0);
            if dev.ctx.is_null() {
                panic!("{}", last_error_message("mlx_device_new_type"));
            }
            let stream = ffi::mlx_stream_new_device(dev);
            ffi::mlx_device_free(dev);
            panic_on_null_stream("mlx_stream_new_device", stream);
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
            prepare_error_capture();
            let rc = ffi::mlx_set_default_stream(self.inner);
            panic_on_status("mlx_set_default_stream", rc);
        }
        refresh_default_gpu_raw();
    }
}

fn panic_on_null_stream(operation: &str, stream: ffi::mlx_stream) {
    if stream.ctx.is_null() {
        panic!("{}", last_error_message(operation));
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

    /// The cached raw default handle is thread-local. `inner.ctx` is the
    /// ax_shim wrapper pointer, not the upstream stream identity; this test
    /// protects the wrapper cache from accidentally becoming process-global.
    #[test]
    fn cached_default_stream_wrappers_are_thread_local() {
        let main_stream = default_gpu_raw();
        let main_ctx = main_stream.ctx as usize;

        let worker_ctx: usize = thread::spawn(|| {
            let worker_stream = default_gpu_raw();
            worker_stream.ctx as usize
        })
        .join()
        .expect("worker thread must complete cleanly");

        assert_ne!(
            main_ctx, worker_ctx,
            "thread-local default stream wrappers must not be shared across OS threads"
        );
    }

    /// ax_shim allocates wrapper objects for both default and dedicated streams.
    /// Dropping either Rust wrapper must free only that wrapper object, not the
    /// scheduler-owned default stream itself.
    #[test]
    fn stream_wrappers_own_their_handles() {
        let default = MlxStream::default_gpu();
        let owned = MlxStream::new_gpu();
        assert!(default.owns, "default_gpu must own its ax_shim wrapper");
        assert!(owned.owns, "new_gpu must claim ownership");
        // Pointers should differ because each handle is a distinct ax_shim wrapper.
        assert_ne!(
            default.inner.ctx as usize, owned.inner.ctx as usize,
            "new_gpu must allocate a distinct stream wrapper from the thread default"
        );
    }

    #[test]
    fn default_gpu_raw_reuses_one_wrapper_per_thread() {
        let first = default_gpu_raw();
        let second = default_gpu_raw();
        assert_eq!(
            first.ctx as usize, second.ctx as usize,
            "hot-path default_gpu_raw should reuse the thread-local ax_shim wrapper"
        );
    }

    #[test]
    fn set_as_default_refreshes_cached_default_stream_value() {
        let dedicated = MlxStream::new_gpu();
        dedicated.set_as_default();
        let after = default_gpu_raw();
        assert!(
            unsafe { ffi::mlx_stream_equal(after, dedicated.inner) },
            "changing the thread default must refresh the cached default stream value"
        );
    }
}
