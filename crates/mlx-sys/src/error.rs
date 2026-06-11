//! Recoverable MLX error capture.
//!
//! mlx-c's default error handler prints the message and calls `exit(-1)`,
//! which kills the process before the non-zero status code a failing call
//! returns can ever be observed. Installing the recording handler here keeps
//! the process alive: the message is captured for retrieval via
//! [`take_last_error`] and errors surface through the status codes that
//! wrappers such as [`crate::transforms::try_eval`] and
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

/// Replace mlx-c's process-killing default error handler with the recording
/// handler. Idempotent; mlx-c stores the handler in non-atomic global state,
/// so the [`Once`] guard also serializes installation.
pub fn install_recoverable_error_handler() {
    INSTALL.call_once(|| unsafe {
        ffi::mlx_set_error_handler(Some(recording_error_handler), std::ptr::null_mut(), None);
    });
}

/// Take the most recent MLX error message, clearing the slot.
///
/// The slot is global (mlx-c reports errors through one process-wide
/// handler), so callers should take it immediately after observing a
/// non-zero status code from an mlx-c call.
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
}
