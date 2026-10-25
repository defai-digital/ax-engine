//! Test-only helpers shared by `#[cfg(test)]` modules in this crate.
//!
//! Most `mlx-sys` tests are independent, but some state is process-global and
//! therefore unsafe to touch from concurrent test threads:
//!
//! * The MLX memory limits and counters (`mlx_set_wired_limit`,
//!   `mlx_set_memory_limit`, `mlx_set_cache_limit`, `mlx_get_peak_memory`,
//!   `mlx_reset_peak_memory`, `mlx_get_cache_memory`, `mlx_get_active_memory`).
//!   A test that reads the previous limit and restores it can interleave with
//!   another test's set/restore and leave a wrong value behind; a reset or a
//!   cache flush in one test changes what another test reads.
//! * The `AX_MLX_COMPILED_*` kill switches. The C++ shims read them with
//!   `getenv` on every call, so a test that sets or clears one changes the
//!   behaviour another test observes.
//!
//! Peak-counter assertions additionally run in a subprocess: a mutex cannot
//! stop unrelated GPU work in other tests from raising the peak, so see
//! [`with_isolated_peak_memory`].

#![allow(clippy::expect_used)]

use std::ffi::OsString;
use std::sync::{Mutex, MutexGuard};

/// Serializes every test that touches process-global MLX state: the memory
/// limits and counters, and the environment variables consulted by the
/// compiled-op kill switches.
static PROCESS_GLOBAL_LOCK: Mutex<()> = Mutex::new(());

fn lock_ignoring_poison(lock: &'static Mutex<()>) -> MutexGuard<'static, ()> {
    // A panicking test must not poison the lock for every test that follows.
    lock.lock().unwrap_or_else(|error| error.into_inner())
}

/// Hold this guard for the whole read -> exercise -> restore sequence of any
/// process-global MLX state: limits and memory counters, or a kill-switch
/// environment variable (pair the latter with [`ScopedEnvVar`], which restores
/// the previous value when dropped).
pub(crate) fn lock_process_global() -> MutexGuard<'static, ()> {
    lock_ignoring_poison(&PROCESS_GLOBAL_LOCK)
}

/// Temporarily sets or unsets an environment variable, restoring the previous
/// value on drop.
///
/// Callers must hold [`lock_process_global`] for the guard's whole lifetime so
/// concurrent tests cannot interleave environment changes, and so a panic in
/// the test body still restores the variable before the next test runs.
pub(crate) struct ScopedEnvVar {
    name: &'static str,
    previous: Option<OsString>,
}

impl ScopedEnvVar {
    /// Set `name=value` for the guard's lifetime.
    pub(crate) fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        // SAFETY: the caller holds `lock_process_global` for the guard's
        // lifetime, so no other thread reads or writes the process environment
        // concurrently.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }

    /// Unset `name` for the guard's lifetime, even if it was set externally.
    pub(crate) fn unset(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        // SAFETY: as in `set`.
        unsafe { std::env::remove_var(name) };
        Self { name, previous }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        match self.previous.take() {
            // SAFETY: the caller held `lock_process_global` for the guard's
            // lifetime.
            Some(value) => unsafe { std::env::set_var(self.name, value) },
            // SAFETY: as above.
            None => unsafe { std::env::remove_var(self.name) },
        }
    }
}

/// Run `measure` in a fresh process so the process-global MLX peak counter
/// cannot see an allocation or a reset from another test thread. The parent
/// asserts that the child actually executed this test.
///
/// This is deliberately stronger than [`lock_process_global`]: the lock only
/// orders tests that take it, while another test's GPU work can still raise
/// the peak in this process.
pub(crate) fn with_isolated_peak_memory(measure: impl FnOnce()) {
    const FLAG: &str = "AX_MLX_SYS_PEAK_TEST_ISOLATED";
    if std::env::var_os(FLAG).is_some() {
        measure();
        return;
    }
    let name = std::thread::current()
        .name()
        .expect("libtest names the test thread")
        .to_owned();
    let exe = std::env::current_exe().expect("test executable");
    let output = std::process::Command::new(exe)
        .args(["--exact", &name, "--test-threads=1", "--nocapture"])
        .env(FLAG, "1")
        .output()
        .expect("spawn isolated peak measurement");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated peak measurement failed for {name}\n{stdout}\n{stderr}"
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("test result: ok. 1 passed;"),
        "isolated peak measurement did not run {name}\n{combined}"
    );
}
