use std::env;

use tracing_subscriber::EnvFilter;

pub(crate) fn init_tracing() {
    let filter = env::var("AX_BENCH_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("RUST_LOG")
                .ok()
                .filter(|value| !value.trim().is_empty())
        });

    let Some(filter) = filter else {
        return;
    };
    let Ok(env_filter) = EnvFilter::try_new(filter) else {
        return;
    };

    // Reserve stdout for command output, including JSON and streamed events.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_ansi(false)
        .compact()
        .try_init();
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    #[test]
    fn tracing_preserves_json_stdout() {
        const CHILD: &str = "AX_BENCH_LOGGING_TEST_CHILD";
        const MESSAGE: &str = "benchmark trace stream probe";
        if std::env::var_os(CHILD).is_some() {
            super::init_tracing();
            tracing::info!("{MESSAGE}");
            // The test harness prints the case name without a trailing newline.
            println!("\n{}", serde_json::json!({"status": "ok"}));
            return;
        }

        let output =
            std::process::Command::new(std::env::current_exe().expect("benchmark test executable"))
                .args([
                    "--exact",
                    "logging::tests::tracing_preserves_json_stdout",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("AX_BENCH_LOG", "info")
                .output()
                .expect("run isolated logging check");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "{stdout}\n{stderr}");
        assert!(stdout.contains("test result: ok. 1 passed;"), "{stdout}");
        assert!(
            stdout.lines().any(|line| line == r#"{"status":"ok"}"#),
            "{stdout}"
        );
        assert!(!stdout.contains(MESSAGE), "{stdout}");
        assert!(stderr.contains(MESSAGE), "{stderr}");
    }
}
