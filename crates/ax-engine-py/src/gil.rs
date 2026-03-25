use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use pyo3::ffi;
use pyo3::prelude::*;

/// Release the Python GIL while running non-Python Rust code on the current
/// OS thread, then reacquire it before returning.
///
/// This exists because the engine's Metal-backed session state is not
/// `Send`/`Sync`, so it cannot use PyO3's safe `allow_threads(...)` API, which
/// requires an `Ungil` closure. The closure passed here must not touch Python
/// objects or call the Python C API while the GIL is released.
pub fn allow_threads_unsend<T>(_py: Python<'_>, f: impl FnOnce() -> T) -> T {
    // SAFETY: `PyEval_SaveThread` releases the GIL for the current thread and
    // returns the thread state token required by `PyEval_RestoreThread`. The
    // closure only runs Rust engine code; it must not access Python APIs. We
    // restore the exact saved thread state before returning, including on panic.
    let thread_state = unsafe { ffi::PyEval_SaveThread() };
    let result = catch_unwind(AssertUnwindSafe(f));
    unsafe {
        ffi::PyEval_RestoreThread(thread_state);
    }
    match result {
        Ok(value) => value,
        Err(payload) => resume_unwind(payload),
    }
}
