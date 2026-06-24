//! Recoverable MLX error capture.
//!
//! The ax_shim C++ layer uses a process-wide error handler that records into
//! this Rust-side slot. Without the recording handler installed here, errors
//! are silently discarded. Installing it keeps the message available for
//! retrieval via [`take_last_error`] and lets errors surface through the status
//! codes that wrappers such as [`crate::transforms::try_eval`] and
//! [`crate::metal::MlxMetalKernel::try_apply_with_template`] check.

use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::{Mutex, Once};

use crate::ffi;

static INSTALL: Once = Once::new();
static LAST_ERROR: Mutex<Option<String>> = Mutex::new(None);

unsafe extern "C" fn recording_error_handler(msg: *const c_char, _data: *mut c_void) {
    // This callback runs inside C++ frames; it must never unwind.
    let message = if msg.is_null() {
        String::from("unknown MLX error")
    } else {
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    // Keep the message visible like the default handler did, minus the exit.
    eprintln!("mlx error: {message}");
    if let Ok(mut slot) = LAST_ERROR.lock() {
        *slot = Some(message);
    }
}

/// Install the recording error handler. Idempotent.
pub fn install_recoverable_error_handler() {
    INSTALL.call_once(|| unsafe {
        ffi::mlx_set_error_handler(Some(recording_error_handler), std::ptr::null_mut(), None);
    });
}

/// Prepare the current thread to capture the next recoverable MLX error.
pub(crate) fn prepare_error_capture() {
    install_recoverable_error_handler();
    let _ = take_last_error();
}

/// Take the most recent MLX error message, clearing the slot.
///
/// The ax_shim error handler is thread-local, so callers should take the
/// error immediately after observing a non-zero status code on the same
/// thread.
pub fn take_last_error() -> Option<String> {
    LAST_ERROR.lock().ok().and_then(|mut slot| slot.take())
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

pub(crate) fn panic_on_status(operation: &str, rc: libc::c_int) {
    if let Err(message) = status_to_result(operation, rc) {
        panic!("{message}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_last_error_clears_slot() {
        if let Ok(mut slot) = LAST_ERROR.lock() {
            *slot = Some("synthetic".to_string());
        }
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
