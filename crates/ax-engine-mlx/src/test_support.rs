//! Test-only helpers shared by `#[cfg(test)]` modules in this crate.

#![allow(clippy::expect_used)]

/// Run `measure` in a fresh process so the process-global MLX peak counter
/// cannot see allocations from other test threads. The parent asserts that
/// the child actually executed this test.
pub(crate) fn with_isolated_peak_memory(measure: impl FnOnce()) {
    const FLAG: &str = "AX_MLX_PEAK_TEST_ISOLATED";
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
        combined.contains("1 passed"),
        "isolated peak measurement did not run {name}\n{combined}"
    );
}
