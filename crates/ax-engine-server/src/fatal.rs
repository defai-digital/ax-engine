//! Last engine step and the panic line written before a `panic=abort` exit.
//!
//! `release` builds abort the process on an MLX panic, so `catch_unwind` in
//! the generation worker never runs. This snapshot is the step the worker
//! finished before that panic. The hook writes one line to stderr and flushes
//! it. `release-server` keeps unwind and reports the same snapshot from the
//! worker's catch path instead.
//!
//! The snapshot is thread-local and only armed inside a generation worker:
//! the panic hook and the worker's `catch_unwind` both run on the panicking
//! thread, so a panic in one model's worker can never report a sibling
//! model's step, and a panic on an HTTP / gRPC / runtime thread is not
//! labelled `engine_panic`.

use std::cell::Cell;
use std::io::{Write, stderr};
use std::panic::PanicHookInfo;

#[derive(Clone, Copy)]
struct LastStep {
    step_id: Option<u64>,
    kv_usage_blocks: u32,
}

thread_local! {
    /// `Some` while the current thread runs a generation worker.
    static WORKER_LAST_STEP: Cell<Option<Option<LastStep>>> = const { Cell::new(None) };
}

/// Marks the current thread as a generation worker until dropped. Created at
/// the top of the worker body so that load-time panics are attributed too.
pub(crate) struct WorkerScope(());

impl WorkerScope {
    pub(crate) fn enter() -> Self {
        WORKER_LAST_STEP.with(|slot| slot.set(Some(None)));
        WorkerScope(())
    }
}

impl Drop for WorkerScope {
    fn drop(&mut self) {
        let _ = WORKER_LAST_STEP.try_with(|slot| slot.set(None));
    }
}

pub(crate) fn note_completed_step(step_id: Option<u64>, kv_usage_blocks: u32) {
    let _ = WORKER_LAST_STEP.try_with(|slot| {
        if slot.get().is_some() {
            slot.set(Some(Some(LastStep {
                step_id,
                kv_usage_blocks,
            })));
        }
    });
}

/// `None` outside a generation worker.
fn worker_snapshot() -> Option<Option<LastStep>> {
    WORKER_LAST_STEP.try_with(|slot| slot.get()).ok().flatten()
}

pub(crate) fn last_step_snapshot() -> (Option<u64>, Option<u32>) {
    match worker_snapshot() {
        Some(Some(step)) => (step.step_id, Some(step.kv_usage_blocks)),
        _ => (None, None),
    }
}

pub(crate) fn install_abort_panic_hook() {
    if !cfg!(panic = "abort") {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(step) = worker_snapshot() {
            let line = format_engine_panic_line(
                &panic_message(info),
                info.location().map(|location| location.to_string()),
                step.and_then(|step| step.step_id),
                step.map(|step| step.kv_usage_blocks),
            );
            {
                let mut err = stderr().lock();
                let _ = err.write_all(line.as_bytes());
                let _ = err.flush();
            }
        }
        previous(info);
    }));
}

pub(crate) const ABORT_BUILD_NOTICE: &str = "\
ax-engine-server: this binary uses panic=abort, so an MLX failure exits the process. \
Build with --profile release-server (panic=unwind) to keep the process up and fail the request. \
A generation-worker panic still writes one stderr line: error_code=engine_panic step_id=<id> kv_usage_blocks=<n>.";

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
#[allow(clippy::expect_used)]
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

    #[test]
    fn snapshot_is_per_worker_thread_and_only_inside_a_worker_scope() {
        // Outside a worker scope nothing is recorded.
        note_completed_step(Some(7), 3);
        assert_eq!(last_step_snapshot(), (None, None));

        let scope = WorkerScope::enter();
        assert_eq!(last_step_snapshot(), (None, None));
        note_completed_step(Some(15), 8);
        assert_eq!(last_step_snapshot(), (Some(15), Some(8)));
        // A later step without an id does not keep the previous id.
        note_completed_step(None, 3);
        assert_eq!(last_step_snapshot(), (None, Some(3)));
        note_completed_step(Some(16), 4);

        // A sibling worker on another thread sees only its own steps.
        let sibling = std::thread::spawn(|| {
            let _scope = WorkerScope::enter();
            note_completed_step(Some(42), 100);
            last_step_snapshot()
        })
        .join()
        .expect("sibling worker");
        assert_eq!(sibling, (Some(42), Some(100)));
        assert_eq!(last_step_snapshot(), (Some(16), Some(4)));

        drop(scope);
        assert_eq!(last_step_snapshot(), (None, None));
    }
}
