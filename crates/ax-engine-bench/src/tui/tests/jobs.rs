//! Download jobs: progress events, ETA formatting, cancellation, and the
//! log-line parsers that recover output paths.
use super::super::catalog::{self};
use super::super::jobs::{Job, format_eta, parse_output_path_from_log, parse_progress_event};
use super::{new_app, test_task};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Jobs / downloads
// ---------------------------------------------------------------------------

#[test]
fn progress_event_lines_parse() {
    assert_eq!(
        parse_progress_event(r#"{"event":"progress","done":85,"total":100,"file":"snapshot"}"#),
        Some((85, 100, "snapshot".into()))
    );
    assert_eq!(parse_progress_event(r#"{"event":"other"}"#), None);
    assert_eq!(parse_progress_event("plain text"), None);
    assert_eq!(
        parse_progress_event(r#"{"schema_version":"ax.download_model.v1"}"#),
        None
    );
}

#[test]
fn eta_formatting() {
    assert_eq!(format_eta(42), "42s");
    assert_eq!(format_eta(90), "1m30s");
    assert_eq!(format_eta(3720), "1h02m");
}

#[test]
fn queued_download_can_be_cancelled_before_spawn() {
    let mut task = test_task(None);
    task.dest = None;
    assert_eq!(task.status_label(), "queued");
    assert!(task.is_queued());
    task.cancel();
    assert_eq!(task.status_label(), "cancelled");
    assert!(!task.is_queued());
}

#[cfg(unix)]
#[test]
fn cancelling_job_stops_long_lived_descendant() {
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after Unix epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "ax-tui-cancel-process-group-{}-{unique}",
        process::id()
    ));
    let heartbeat = root.join("heartbeat");
    let descendant_pid = root.join("descendant.pid");
    std::fs::create_dir_all(&root).expect("create process-group test directory");

    // Clean up the explicitly recorded descendant even if an assertion
    // catches a cancellation regression, so the test itself never leaks it.
    struct DescendantGuard {
        pid_file: PathBuf,
        root: PathBuf,
    }
    impl Drop for DescendantGuard {
        fn drop(&mut self) {
            if let Ok(pid) = std::fs::read_to_string(&self.pid_file) {
                let _ = process::Command::new("/bin/kill")
                    .args(["-s", "KILL", "--"])
                    .arg(pid.trim())
                    .stdout(process::Stdio::null())
                    .stderr(process::Stdio::null())
                    .status();
            }
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
    let _guard = DescendantGuard {
        pid_file: descendant_pid.clone(),
        root: root.clone(),
    };

    let mut command = process::Command::new("sh");
    command
        .arg("-c")
        .arg(
            r#"(while :; do printf . >> "$1"; /bin/sleep 0.01; done) &
printf '%s\n' "$!" > "$2"
wait"#,
        )
        .arg("ax-tui-process-group-test")
        .arg(&heartbeat)
        .arg(&descendant_pid);
    let mut job = Job::spawn(command, None).expect("spawn parent with heartbeat descendant");

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let heartbeat_started =
            std::fs::metadata(&heartbeat).is_ok_and(|metadata| metadata.len() >= 2);
        let pid_recorded = std::fs::read_to_string(&descendant_pid)
            .is_ok_and(|pid| pid.trim().parse::<u32>().is_ok());
        if heartbeat_started && pid_recorded {
            break;
        }
        if Instant::now() >= deadline {
            job.cancel();
            panic!("descendant did not start its heartbeat before the deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    job.cancel();
    assert_eq!(job.done, Some(-130));

    // Allow any write already in progress at signal delivery to land, then
    // prove the descendant can no longer advance its heartbeat.
    std::thread::sleep(Duration::from_millis(100));
    let settled_len = std::fs::metadata(&heartbeat)
        .expect("heartbeat remains readable")
        .len();
    std::thread::sleep(Duration::from_millis(150));
    let final_len = std::fs::metadata(&heartbeat)
        .expect("heartbeat remains readable")
        .len();
    assert_eq!(
        final_len, settled_len,
        "cancelled job's descendant continued writing its heartbeat"
    );
}

#[test]
fn progress_ratio_and_eta_need_totals() {
    let mut task = test_task(None);
    task.total_bytes = None;
    assert_eq!(task.progress_ratio(), None);
    assert_eq!(task.eta_seconds(), None);
    task.total_bytes = Some(100);
    // No job yet: 0 bytes downloaded.
    assert_eq!(task.progress_ratio(), Some(0.0));
}

#[test]
fn log_parser_prefers_structured_summary_dest() {
    let lines = vec![
        "{\"event\":\"progress\",\"done\":1,\"total\":100,\"file\":\"x\"}".to_string(),
        "Path: /tmp/scraped".to_string(),
        "{\"schema_version\":\"ax.download_model.v1\",\"repo_id\":\"owner/repo\",\"dest\":\"/tmp/structured\",\"status\":\"ready\"}".to_string(),
    ];
    assert_eq!(
        parse_output_path_from_log(&lines).as_deref(),
        Some(Path::new("/tmp/structured"))
    );
}

#[test]
fn log_parser_finds_download_output_paths() {
    assert_eq!(
        parse_output_path_from_log(&["Path: /tmp/direct".to_string()]).as_deref(),
        Some(Path::new("/tmp/direct"))
    );
    assert_eq!(
        parse_output_path_from_log(&["Output dir: /tmp/mtp".to_string()]).as_deref(),
        Some(Path::new("/tmp/mtp"))
    );
    assert_eq!(
        parse_output_path_from_log(&["Output dir:      /tmp/gemma-mtp".to_string()]).as_deref(),
        Some(Path::new("/tmp/gemma-mtp"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Sidecar ready at:".to_string(),
            "  /tmp/sidecar".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/sidecar"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Gemma 4 assistant MTP package ready at:".to_string(),
            "  /tmp/gemma-package".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/gemma-package"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Next:".to_string(),
            "  ax-engine serve /tmp/from-next --port 31418".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/from-next"))
    );
}

#[test]
fn artifact_dir_usable_requires_real_model_files() {
    let root = std::env::temp_dir().join(format!("ax-tui-artifact-usable-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir root");

    let empty = root.join("empty");
    std::fs::create_dir_all(&empty).expect("mkdir");
    assert!(!catalog::artifact_dir_usable(&empty));

    let with_config = root.join("cfg");
    std::fs::create_dir_all(&with_config).expect("mkdir");
    std::fs::write(with_config.join("config.json"), "{}").expect("write");
    assert!(catalog::artifact_dir_usable(&with_config));

    let with_weight = root.join("wt");
    std::fs::create_dir_all(&with_weight).expect("mkdir");
    std::fs::write(with_weight.join("model.safetensors"), b"x").expect("write");
    assert!(catalog::artifact_dir_usable(&with_weight));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn serve_without_resolved_package_path_fails_closed() {
    let mut app = new_app();
    let mut task = test_task(Some(Job::failed("ok".into())));
    // Hermetic: a repo that can never be in this machine's HF cache, so the
    // snapshot-dir fallback cannot resolve a real path. AutomatosX pack
    // tasks carry no preset either — without a package path there is no
    // legitimate way to serve.
    task.repo_id = "ax-tests/does-not-exist-anywhere".to_string();
    task.preset = None;
    task.dest = None;
    task.resolved_path = None;
    if let Some(job) = &mut task.job {
        job.done = Some(0);
        job.log.clear(); // no Path: / Output dir: lines
    }
    app.downloads.push(task);
    app.start_server_for_download(0);
    let job = app.server.expect("server job should be set");
    assert_eq!(
        job.done,
        Some(-1),
        "must not spawn real server without path"
    );
    assert!(
        job.log
            .iter()
            .any(|line| line.contains("no server artifact path could be resolved")),
        "error should explain the unresolved artifact path: {:?}",
        job.log
    );
}
