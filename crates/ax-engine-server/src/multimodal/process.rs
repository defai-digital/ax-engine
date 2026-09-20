use std::process::{Command, Output};

use super::MediaError;
use crate::tasks::BlockingTaskControl;

/// Drain both pipes without reader threads, which could outlive a cancelled
/// request if another process inherits a write descriptor.
#[cfg(unix)]
pub(super) fn run_bounded(
    command: &mut Command,
    control: &BlockingTaskControl,
    stdout_limit: usize,
    stderr_limit: usize,
) -> Result<Output, MediaError> {
    use std::io::{ErrorKind, Read};
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    use std::process::{Child, Stdio};

    struct ReapOnDrop(Child);
    impl Drop for ReapOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn io_error(error: std::io::Error) -> MediaError {
        MediaError::Decode(format!("ffmpeg video decoder I/O failed: {error}"))
    }

    fn drain(
        pipe: &mut UnixStream,
        bytes: &mut Vec<u8>,
        limit: usize,
        label: &str,
        eof: &mut bool,
    ) -> Result<bool, MediaError> {
        if *eof {
            return Ok(false);
        }
        // Read one chunk per iteration so flooding either pipe cannot starve
        // the other pipe, child status, or cancellation checks.
        let mut buffer = [0; 16 * 1024];
        match pipe.read(&mut buffer) {
            Ok(0) => {
                *eof = true;
                Ok(false)
            }
            Ok(count) => {
                if count > limit.saturating_sub(bytes.len()) {
                    return Err(MediaError::Decode(format!(
                        "ffmpeg {label} exceeded its {limit}-byte limit"
                    )));
                }
                bytes.extend_from_slice(&buffer[..count]);
                Ok(true)
            }
            Err(error) if error.kind() == ErrorKind::WouldBlock => Ok(false),
            Err(error) if error.kind() == ErrorKind::Interrupted => Ok(true),
            Err(error) => Err(io_error(error)),
        }
    }

    control
        .check()
        .map_err(|message| MediaError::Interrupted(message.to_string()))?;
    let (mut stdout_pipe, stdout_child) = UnixStream::pair().map_err(io_error)?;
    let (mut stderr_pipe, stderr_child) = UnixStream::pair().map_err(io_error)?;
    stdout_pipe.set_nonblocking(true).map_err(io_error)?;
    stderr_pipe.set_nonblocking(true).map_err(io_error)?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::from(OwnedFd::from(stdout_child)))
        .stderr(Stdio::from(OwnedFd::from(stderr_child)));
    let spawned = command.spawn();
    // Command retains its configured descriptors after spawn. Close our
    // copies so EOF reflects child ownership, including on spawn failure.
    command.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = ReapOnDrop(spawned.map_err(|error| {
        if error.kind() == ErrorKind::NotFound {
            MediaError::Unsupported(
                "inline MP4/WebM video requires ffmpeg on PATH to extract frames; install ffmpeg or send pre-extracted frame tensors via /v1/generate".to_string(),
            )
        } else {
            io_error(error)
        }
    })?);
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let (mut stdout_eof, mut stderr_eof) = (false, false);
    let mut status = None;
    loop {
        control
            .check()
            .map_err(|message| MediaError::Interrupted(message.to_string()))?;
        let out = drain(
            &mut stdout_pipe,
            &mut stdout,
            stdout_limit,
            "stdout",
            &mut stdout_eof,
        )?;
        let err = drain(
            &mut stderr_pipe,
            &mut stderr,
            stderr_limit,
            "stderr",
            &mut stderr_eof,
        )?;
        if status.is_none() {
            status = child.0.try_wait().map_err(io_error)?;
        }
        if stdout_eof
            && stderr_eof
            && let Some(status) = status
        {
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        if !out && !err {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}

#[cfg(not(unix))]
pub(super) fn run_bounded(
    _command: &mut Command,
    _control: &BlockingTaskControl,
    _stdout_limit: usize,
    _stderr_limit: usize,
) -> Result<Output, MediaError> {
    Err(MediaError::Unsupported(
        "inline ffmpeg video decoding requires Unix".to_string(),
    ))
}

#[cfg(all(test, unix))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn shell(script: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        command.args(["-c", script]);
        command
    }

    #[test]
    fn drains_both_pipes_and_preserves_exit_status() {
        let output = run_bounded(
            &mut shell("printf frames; printf timestamps >&2; exit 7"),
            &BlockingTaskControl::default(),
            128,
            128,
        )
        .unwrap();
        assert_eq!(output.stdout, b"frames");
        assert_eq!(output.stderr, b"timestamps");
        assert_eq!(output.status.code(), Some(7));
    }

    #[test]
    fn enforces_each_output_limit_without_waiting_for_exit() {
        for (script, label) in [
            ("while :; do printf 0123456789; done", "stdout"),
            ("while :; do printf 0123456789 >&2; done", "stderr"),
        ] {
            let started = Instant::now();
            let error = run_bounded(
                &mut shell(script),
                &BlockingTaskControl::new(Duration::from_secs(2)),
                32,
                32,
            )
            .unwrap_err();
            assert!(error.to_string().contains(label), "{error}");
            assert!(started.elapsed() < Duration::from_secs(2));
        }
    }

    #[test]
    fn timeout_kills_and_reaps_the_decoder() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let pid_path =
            std::env::temp_dir().join(format!("ax-decoder-{}-{unique}.pid", std::process::id()));
        let mut command = shell("printf '%s' $$ > \"$1\"; exec sleep 60");
        command.arg("decoder").arg(&pid_path);
        let started = Instant::now();
        let error = run_bounded(
            &mut command,
            &BlockingTaskControl::new(Duration::from_millis(200)),
            128,
            128,
        )
        .unwrap_err();
        assert!(matches!(error, MediaError::Interrupted(_)));
        assert!(started.elapsed() < Duration::from_secs(2));
        let pid = std::fs::read_to_string(&pid_path).unwrap();
        std::fs::remove_file(&pid_path).unwrap();
        let status = Command::new("/bin/kill")
            .args(["-0", pid.trim()])
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(!status.success(), "decoder must be reaped before returning");
    }

    #[test]
    fn inherited_output_descriptors_cannot_extend_the_deadline() {
        let started = Instant::now();
        let error = run_bounded(
            &mut shell("sleep 1 & exit 0"),
            &BlockingTaskControl::new(Duration::from_millis(100)),
            128,
            128,
        )
        .unwrap_err();
        assert!(matches!(error, MediaError::Interrupted(_)));
        assert!(started.elapsed() < Duration::from_millis(800));
    }

    #[test]
    fn cancellation_stops_a_decoder_before_its_deadline() {
        let control = BlockingTaskControl::new(Duration::from_secs(60));
        let worker_control = control.clone();
        let worker = std::thread::spawn(move || {
            run_bounded(&mut shell("exec sleep 60"), &worker_control, 128, 128)
        });
        std::thread::sleep(Duration::from_millis(50));
        let started = Instant::now();
        control.cancel();
        assert!(matches!(
            worker.join().unwrap(),
            Err(MediaError::Interrupted(_))
        ));
        assert!(started.elapsed() < Duration::from_secs(2));
    }
}
