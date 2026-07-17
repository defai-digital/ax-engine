//! Recoverable MLX error capture.
//!
//! The ax_shim C++ layer uses a process-wide error handler that records into
//! this Rust-side slot. Without the recording handler installed here, errors
//! are silently discarded. Installing it keeps the message available for
//! retrieval via [`take_last_error`] and lets errors surface through the status
//! codes that wrappers such as [`crate::transforms::try_eval`] and
//! [`crate::metal::MlxMetalKernel::try_apply_with_template`] check.

use std::cell::RefCell;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::Once;

use crate::ffi;

static INSTALL: Once = Once::new();

// Per-thread error slot. The ax_shim C++ layer stores its last error in a
// `thread_local std::string` and invokes the handler synchronously on the
// calling thread, so the Rust slot must be thread-local too. A process-global
// slot would let a concurrent failing op on another thread overwrite or clear
// this thread's not-yet-read error message.
thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

unsafe extern "C" fn recording_error_handler(msg: *const c_char, _data: *mut c_void) {
    // This callback runs inside C++ frames; it must never unwind (unwinding
    // out of an `extern "C"` fn aborts the process). `eprintln!` panics on a
    // broken stderr and `LocalKey::with` panics during thread teardown, so
    // both use the non-panicking variants.
    let message = if msg.is_null() {
        String::from("unknown MLX error")
    } else {
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    // Keep the message visible like the default handler did, minus the exit.
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "mlx error: {message}");
    let _ = LAST_ERROR.try_with(|slot| *slot.borrow_mut() = Some(message));
}

/// Install the recording error handler. Idempotent.
pub fn install_recoverable_error_handler() {
    INSTALL.call_once(|| unsafe {
        ffi::mlx_set_error_handler(Some(recording_error_handler), std::ptr::null_mut(), None);
        // The shim binds to the mlx::core C++ ABI directly; running against a
        // different libmlx than the headers it was compiled with can fail as
        // silent memory corruption. Fail fast with one clear diagnostic
        // instead (the check compares compile-time MLX_VERSION_* against the
        // runtime mx::version()).
        if ffi::ax_shim_check_mlx_version() != 0 {
            let msg = take_last_error()
                .unwrap_or_else(|| String::from("MLX version check failed with no detail"));
            panic!("{msg}");
        }
    });
}

/// Return the version reported by the loaded MLX runtime.
pub fn runtime_version() -> Result<String, String> {
    install_recoverable_error_handler();
    let version = unsafe { ffi::ax_shim_mlx_version() };
    if version.is_null() {
        return Err(String::from("libmlx returned a null version string"));
    }
    Ok(unsafe { CStr::from_ptr(version) }
        .to_string_lossy()
        .into_owned())
}

/// Ensure the recording error handler is installed, *without* clearing the slot.
///
/// Hot per-op call sites (the `ops`/`fast` FFI macros) use this instead of
/// [`prepare_error_capture`]. The error slot is read-and-cleared (`take`) on
/// every non-zero status, and the C++ handler writes it synchronously on
/// failure, so a success-path op never reads the slot — the pre-clear in
/// `prepare_error_capture` is redundant there. Skipping it removes a
/// thread-local `RefCell` access from the decode hot path (one per op, and a
/// decode token issues hundreds of ops).
#[inline]
pub(crate) fn ensure_error_handler() {
    install_recoverable_error_handler();
}

/// Prepare the current thread to capture the next recoverable MLX error,
/// clearing any stale message first.
///
/// Used by coarse-grained, latency-insensitive call sites (eval, weight load,
/// kernel compile, closures, stream setup) where a clean capture window is
/// worth the thread-local clear. Hot per-op paths use [`ensure_error_handler`].
pub(crate) fn prepare_error_capture() {
    install_recoverable_error_handler();
    let _ = take_last_error();
}

/// Drain any message the handler recorded for an error the caller is
/// deliberately swallowing (a fused-shim attempt falling back to the portable
/// composition, or a probe returning `None`). Leaving the slot populated
/// would misattribute the stale message to the next failing call on this
/// thread — or panic a later `dtype()` on a genuine Bool array, which uses
/// the slot to disambiguate the MLX_BOOL error sentinel.
pub(crate) fn clear_stale_error() {
    let _ = take_last_error();
}

/// Take the most recent MLX error message, clearing the slot.
///
/// The ax_shim error handler is thread-local, so callers should take the
/// error immediately after observing a non-zero status code on the same
/// thread.
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Format a failure message for a named MLX operation from the captured
/// error, falling back to a generic message when the handler saw nothing.
pub fn last_error_message(operation: &str) -> String {
    match take_last_error() {
        Some(message) => format!("{operation} failed: {message}"),
        None => format!("{operation} failed with an unreported MLX error"),
    }
}

pub(crate) fn status_to_result(operation: &str, rc: libc::c_int) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(last_error_message(operation))
    }
}

thread_local! {
    /// Depth of Rust closure bodies currently executing under an MLX
    /// closure trampoline on this thread (compile tracing re-enters).
    static CLOSURE_BODY_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// RAII marker for "this thread is inside an MLX closure body".
///
/// While active, `panic_on_status` switches from panicking to
/// poison-propagation: the release profile builds with `panic = "abort"`,
/// so a panic raised inside the closure trampoline can never be caught —
/// any transient MLX error during compile tracing would kill the whole
/// process (the failure mode that forced `AX_MLX_MOE_LAYER_COMPILE` back
/// to opt-in). Instead the failing op leaves its error in the thread-local
/// slot and returns a null-handle array; downstream shim calls reject the
/// null handle cleanly ("expected a non-empty ..."), and the compiled-
/// closure caller's post-apply slot drain (`try_apply_with_abort_safety`)
/// turns the whole apply into a graceful per-layer fallback.
pub struct ClosureBodyGuard(());

impl ClosureBodyGuard {
    pub fn enter() -> Self {
        CLOSURE_BODY_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        ClosureBodyGuard(())
    }
}

impl Drop for ClosureBodyGuard {
    fn drop(&mut self) {
        CLOSURE_BODY_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

pub(crate) fn in_closure_body() -> bool {
    CLOSURE_BODY_DEPTH.with(|d| d.get() > 0)
}

/// Ensure the error slot is non-empty after an in-body op failure, without
/// overwriting a richer message the C++ handler already recorded.
fn ensure_error_slot(operation: &str) {
    LAST_ERROR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(format!(
                "{operation} failed inside a compiled-closure body (poison mode)"
            ));
        }
    });
}

pub(crate) fn panic_on_status(operation: &str, rc: libc::c_int) {
    if rc == 0 {
        return;
    }
    if in_closure_body() {
        ensure_error_slot(operation);
        return;
    }
    if let Err(message) = status_to_result(operation, rc) {
        panic!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_last_error_clears_slot() {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some("synthetic".to_string()));
        assert_eq!(take_last_error().as_deref(), Some("synthetic"));
        assert_eq!(take_last_error(), None);
    }

    #[test]
    fn last_error_message_reports_operation_without_captured_error() {
        let _ = take_last_error();
        assert_eq!(
            last_error_message("mlx_eval"),
            "mlx_eval failed with an unreported MLX error"
        );
    }

    #[test]
    fn status_to_result_reports_operation() {
        let _ = take_last_error();
        assert_eq!(
            status_to_result("mlx_bad", 1).unwrap_err(),
            "mlx_bad failed with an unreported MLX error"
        );
        assert!(status_to_result("mlx_ok", 0).is_ok());
    }

    #[test]
    fn runtime_version_is_available() {
        let version = runtime_version().expect("loaded MLX should report its version");
        let components = version.split('.').collect::<Vec<_>>();
        assert!(components.len() >= 3, "unexpected MLX version: {version}");
        for component in components.iter().take(3) {
            assert!(
                component.parse::<u64>().is_ok(),
                "unexpected MLX version: {version}"
            );
        }
    }

    #[test]
    fn mlx_remains_usable_after_error() {
        install_recoverable_error_handler();
        let _ = take_last_error();

        // Trigger a Metal kernel compile error (same pattern as metal.rs test).
        let kernel = crate::metal::MlxMetalKernel::new(
            "ax_shim_err_recovery_test",
            &["input"],
            &["output"],
            "this is not valid metal source;",
            "",
            true,
        );
        let input = crate::MlxArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = kernel.try_apply_with_template(
            &[&input],
            &[crate::metal::KernelOutputSpec {
                shape: vec![4],
                dtype: crate::array::MlxDtype::Float32,
            }],
            &[],
            (4, 1, 1),
            (4, 1, 1),
            None,
        );

        // Failure may surface at apply or eval time
        let eval_result = match result {
            Err(msg) => Err(msg),
            Ok(outputs) => {
                let refs: Vec<_> = outputs.iter().collect();
                crate::transforms::try_eval(&refs)
            }
        };
        assert!(eval_result.is_err(), "invalid Metal source should fail");
        let _ = take_last_error();

        // MLX must remain usable: a normal op should still work.
        let ok = crate::MlxArray::from_f32(42.0);
        crate::transforms::eval(&[&ok]);
        assert_eq!(ok.data_f32(), &[42.0]);
    }
}
