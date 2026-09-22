//! Last engine step and the panic line written before a `panic=abort` exit.
//!
//! `release` builds abort the process on an MLX panic, so `catch_unwind` in
//! the generation worker never runs. This snapshot is the step the worker
//! finished before that panic. The hook writes one line to stderr and flushes
//! it. `release-server` keeps unwind and reports the same snapshot from the
//! worker's catch path instead.

use std::io::{Write, stderr};
use std::panic::PanicHookInfo;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

static HAS_STEP: AtomicBool = AtomicBool::new(false);
static STEP_ID_PRESENT: AtomicBool = AtomicBool::new(false);
static STEP_ID: AtomicU64 = AtomicU64::new(0);
static KV_USAGE_BLOCKS: AtomicU32 = AtomicU32::new(0);

pub(crate) fn note_completed_step(step_id: Option<u64>, kv_usage_blocks: u32) {
    if let Some(step_id) = step_id {
        STEP_ID.store(step_id, Ordering::Release);
        STEP_ID_PRESENT.store(true, Ordering::Release);
    }
    KV_USAGE_BLOCKS.store(kv_usage_blocks, Ordering::Release);
    HAS_STEP.store(true, Ordering::Release);
}

pub(crate) fn last_step_snapshot() -> (Option<u64>, Option<u32>) {
    if !HAS_STEP.load(Ordering::Acquire) {
        return (None, None);
    }
    let step_id = STEP_ID_PRESENT
        .load(Ordering::Acquire)
        .then(|| STEP_ID.load(Ordering::Acquire));
    (step_id, Some(KV_USAGE_BLOCKS.load(Ordering::Acquire)))
}

pub(crate) fn install_abort_panic_hook() {
    if !cfg!(panic = "abort") {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let (step_id, kv_usage_blocks) = last_step_snapshot();
        let line = format_engine_panic_line(
            &panic_message(info),
            info.location().map(|location| location.to_string()),
            step_id,
            kv_usage_blocks,
        );
        let mut err = stderr().lock();
        let _ = err.write_all(line.as_bytes());
        let _ = err.flush();
        previous(info);
    }));
}

pub(crate) const ABORT_BUILD_NOTICE: &str = "\
ax-engine-server: this binary uses panic=abort, so an MLX failure exits the process. \
Build with --profile release-server (panic=unwind) to keep the process up and fail the request. \
A panic still writes one stderr line: error_code=engine_panic step_id=<id> kv_usage_blocks=<n>.";

pub(crate) fn format_engine_panic_line(
    message: &str,
    location: Option<String>,
    step_id: Option<u64>,
    kv_usage_blocks: Option<u32>,
) -> String {
    let step = step_id
        .map(|id| id.to_string())
        .unwrap_or_else(|| "none".to_string());
    let blocks = kv_usage_blocks
        .map(|blocks| blocks.to_string())
        .unwrap_or_else(|| "none".to_string());
    let location = location.unwrap_or_else(|| "unknown".to_string());
    format!(
        "ax-engine-server fatal error_code=engine_panic step_id={step} kv_usage_blocks={blocks} location={location} message={}\n",
        one_line(message)
    )
}

fn panic_message(info: &PanicHookInfo<'_>) -> String {
    if let Some(message) = info.payload().downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = info.payload().downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn one_line(message: &str) -> String {
    message
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_panic_line_carries_the_last_step() {
        let line = format_engine_panic_line(
            "mlx_eval failed\nwith detail",
            Some("runner.rs:10:4".to_string()),
            Some(15),
            Some(15),
        );
        assert_eq!(
            line,
            "ax-engine-server fatal error_code=engine_panic step_id=15 kv_usage_blocks=15 location=runner.rs:10:4 message=mlx_eval failed with detail\n"
        );
    }

    #[test]
    fn engine_panic_line_marks_a_missing_step() {
        let line = format_engine_panic_line("load failed", None, None, None);
        assert!(line.contains("step_id=none"));
        assert!(line.contains("kv_usage_blocks=none"));
        assert!(line.contains("location=unknown"));
        assert!(line.ends_with('\n'));
        assert_eq!(line.lines().count(), 1);
    }
}
