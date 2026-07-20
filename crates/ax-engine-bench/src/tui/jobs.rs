//! Background jobs (downloads, server, chat requests) streamed from child
//! `ax-engine` / `ax-engine-server` / `curl` processes over an mpsc channel,
//! so the UI thread never blocks on child I/O.

use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Instant;

use super::catalog;

pub(super) enum JobMsg {
    Line(String),
}

/// Result of advancing a background job by one poll cycle.
pub(super) struct JobTick {
    /// Lines that arrived this tick (for progress/SSE parsers).
    pub fresh: Vec<String>,
    /// True when log, exit status, or download byte counters moved.
    /// Spinner-only advances leave this false so off-screen jobs stay quiet.
    pub material: bool,
}

pub(super) const LOG_CAP: usize = 1000;
pub(super) const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const SPEED_HISTORY_CAP: usize = 120;

pub(super) struct Job {
    rx: Receiver<JobMsg>,
    child: Option<Child>,
    pub log: Vec<String>,
    pub done: Option<i32>,
    /// When set, polled each tick for the live byte counter (downloads only).
    watch_dir: Option<PathBuf>,
    pub bytes: u64,
    pub speed: f64,
    /// Recent `speed` samples (downloads only), newest last, capped at `SPEED_HISTORY_CAP`.
    pub speed_history: Vec<u64>,
    last_poll: Option<(Instant, u64)>,
    pub spinner: usize,
}

impl Job {
    pub fn spawn(cmd: Command, watch_dir: Option<PathBuf>) -> io::Result<Job> {
        Self::spawn_with_stdin(cmd, None, watch_dir)
    }

    /// Spawn with an optional payload written to the child's stdin from a
    /// helper thread (the pipe is closed afterwards so the child sees EOF).
    pub fn spawn_with_stdin(
        mut cmd: Command,
        stdin_payload: Option<String>,
        watch_dir: Option<PathBuf>,
    ) -> io::Result<Job> {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        if stdin_payload.is_some() {
            cmd.stdin(Stdio::piped());
        }
        let mut child = cmd.spawn()?;
        if let (Some(payload), Some(mut stdin)) = (stdin_payload, child.stdin.take()) {
            thread::spawn(move || {
                let _ = stdin.write_all(payload.as_bytes());
            });
        }
        let (tx, rx) = mpsc::channel();
        for pipe in [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn io::Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    if tx.send(JobMsg::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }
        Ok(Job {
            rx,
            child: Some(child),
            log: Vec::new(),
            done: None,
            watch_dir,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        })
    }

    /// A finished, processless job that just surfaces a launch error.
    pub fn failed(message: String) -> Job {
        let (_tx, rx) = mpsc::channel();
        Job {
            rx,
            child: None,
            log: vec![message],
            done: Some(-1),
            watch_dir: None,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        }
    }

    /// A still-running, processless job carrying a fixed log (test-only).
    #[cfg(test)]
    pub fn running_with_log(log: Vec<String>) -> Job {
        let (_tx, rx) = mpsc::channel();
        Job {
            rx,
            child: None,
            log,
            done: None,
            watch_dir: None,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        }
    }

    /// A finished, processless job with a fixed exit code (test-only).
    #[cfg(test)]
    pub fn exited(code: i32) -> Job {
        let (_tx, rx) = mpsc::channel();
        Job {
            rx,
            child: None,
            log: Vec::new(),
            done: Some(code),
            watch_dir: None,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        }
    }

    /// Drain pending lines into `log` and refresh liveness/byte counters.
    ///
    /// `fresh` is the lines that arrived this tick so callers can parse them
    /// (progress events, SSE chunks) without re-scanning the whole log.
    /// `material` is true when anything a non-spinner UI cares about changed
    /// (new lines, exit status, or a byte resample) — spinner alone does not
    /// count, so off-screen jobs do not force 10 Hz full-frame repaints.
    pub fn tick(&mut self) -> JobTick {
        let mut fresh = Vec::new();
        let mut material = false;
        while let Ok(JobMsg::Line(line)) = self.rx.try_recv() {
            self.log.push(line.clone());
            fresh.push(line);
            material = true;
            if self.log.len() > LOG_CAP {
                let overflow = self.log.len() - LOG_CAP;
                self.log.drain(0..overflow);
            }
        }
        if self.done.is_none()
            && let Some(child) = &mut self.child
            && let Ok(Some(status)) = child.try_wait()
        {
            self.done = Some(status.code().unwrap_or(-1));
            material = true;
        }
        if let Some(dir) = &self.watch_dir {
            let now = Instant::now();
            // dir_size walks the whole cache tree; at the 10 Hz tick rate that
            // recursive walk is the main cost of a download, so resample at
            // ~1 Hz and reuse the last byte count (and speed) in between.
            let resample = self
                .last_poll
                .is_none_or(|(last_t, _)| now.duration_since(last_t).as_secs_f64() >= 1.0);
            if resample {
                let bytes = catalog::dir_size(dir);
                if let Some((last_t, last_b)) = self.last_poll {
                    let dt = now.duration_since(last_t).as_secs_f64();
                    if dt > 0.0 {
                        let inst = bytes.saturating_sub(last_b) as f64 / dt;
                        self.speed = if self.speed == 0.0 {
                            inst
                        } else {
                            0.6 * self.speed + 0.4 * inst
                        };
                    }
                }
                self.last_poll = Some((now, bytes));
                self.bytes = bytes;
                self.speed_history.push(self.speed as u64);
                if self.speed_history.len() > SPEED_HISTORY_CAP {
                    let overflow = self.speed_history.len() - SPEED_HISTORY_CAP;
                    self.speed_history.drain(0..overflow);
                }
                // Progress gauge / tab-bar percent updates at sample rate.
                material = true;
            }
        }
        self.spinner = (self.spinner + 1) % SPINNER.len();
        JobTick { fresh, material }
    }

    pub fn is_running(&self) -> bool {
        self.done.is_none() && self.child.is_some()
    }

    pub fn cancel(&mut self) {
        if self.done.is_none()
            && let Some(child) = &mut self.child
        {
            let _ = child.kill();
            let _ = child.wait();
            self.done = Some(-130);
        }
    }
}

// ---------------------------------------------------------------------------
// Download tasks
// ---------------------------------------------------------------------------

/// Result of advancing a download task by one tick.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DownloadOutcome {
    /// Nothing new: still queued/running, or already in a terminal state.
    Pending,
    /// The child just exited successfully (done transitioned to Some(0)).
    Finished,
    /// The child just exited non-zero without a user cancel.
    Failed,
}

/// Coarse lifecycle state of a download, for icon/style mapping.
/// `status_label` stays the label-only formatter over this (it adds the
/// exit code for failures).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DownloadStatus {
    Queued,
    Running,
    Ready,
    Failed,
    Cancelled,
}

pub(super) struct DownloadTask {
    pub label: String,
    pub repo_id: &'static str,
    pub preset: Option<&'static str>,
    pub target: String,
    pub dest: Option<PathBuf>,
    pub watch_dir: PathBuf,
    /// Resolved serve path captured when the child finishes (Path: / Output dir: /
    /// package-ready lines). Prefer this over re-parsing the log.
    pub resolved_path: Option<PathBuf>,
    /// Static catalog estimate of the total download, for the gauge.
    pub total_bytes: Option<u64>,
    /// Latest phase message from `--progress-json` events (resolve, snapshot
    /// done, manifest, ...), shown next to the gauge so the bar never looks
    /// stuck during non-byte phases.
    pub phase: Option<String>,
    pub job: Option<Job>,
    pub cancelled: bool,
}

/// One `{"event":"progress","done":..,"total":..,"file":".."}` line from the
/// download helper, forwarded through `ax-engine download --progress-json`.
pub(super) fn parse_progress_event(line: &str) -> Option<(u64, u64, String)> {
    let value: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if value.get("event")?.as_str()? != "progress" {
        return None;
    }
    Some((
        value.get("done")?.as_u64()?,
        value.get("total")?.as_u64()?,
        value.get("file")?.as_str()?.to_string(),
    ))
}

impl DownloadTask {
    /// Lifecycle state derived from the cancel flag + child exit code.
    pub fn status(&self) -> DownloadStatus {
        if self.cancelled {
            return DownloadStatus::Cancelled;
        }
        match self.job.as_ref().and_then(|job| job.done) {
            None if self.job.is_none() => DownloadStatus::Queued,
            Some(0) => DownloadStatus::Ready,
            Some(_) => DownloadStatus::Failed,
            None => DownloadStatus::Running,
        }
    }

    /// Queue label for the status word; failures carry the exit code.
    pub fn status_label(&self) -> String {
        match self.status() {
            DownloadStatus::Queued => "queued".into(),
            DownloadStatus::Running => "running".into(),
            DownloadStatus::Ready => "ready".into(),
            DownloadStatus::Cancelled => "cancelled".into(),
            DownloadStatus::Failed => match self.job.as_ref().and_then(|job| job.done) {
                Some(code) => format!("failed ({code})"),
                None => "failed".into(),
            },
        }
    }

    pub fn output_path(&self) -> Option<PathBuf> {
        if let Some(path) = &self.resolved_path {
            return Some(path.clone());
        }
        if let Some(dest) = &self.dest {
            return Some(dest.clone());
        }
        self.job
            .as_ref()
            .and_then(|job| parse_output_path_from_log(&job.log))
    }

    pub fn is_queued(&self) -> bool {
        !self.cancelled && self.job.is_none()
    }

    pub fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.done.is_none())
    }

    pub fn is_ready(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.done == Some(0))
    }

    pub fn is_failed(&self) -> bool {
        !self.cancelled
            && self
                .job
                .as_ref()
                .and_then(|job| job.done)
                .is_some_and(|code| code != 0)
    }

    /// Finished, failed, or cancelled — safe to remove from the queue list.
    pub fn is_done(&self) -> bool {
        self.cancelled || self.job.as_ref().is_some_and(|job| job.done.is_some())
    }

    /// Reset a failed/cancelled task so it can be spawned again.
    pub fn requeue(&mut self) {
        self.cancelled = false;
        self.job = None;
        self.phase = None;
        self.resolved_path = None;
    }

    /// Fraction complete (0..=1) from watched bytes vs. the static total.
    pub fn progress_ratio(&self) -> Option<f64> {
        let total = self.total_bytes?;
        if total == 0 {
            return None;
        }
        let bytes = self.job.as_ref().map(|job| job.bytes).unwrap_or(0);
        Some((bytes as f64 / total as f64).clamp(0.0, 1.0))
    }

    /// Seconds until done at the current speed, when both are known.
    pub fn eta_seconds(&self) -> Option<u64> {
        let total = self.total_bytes?;
        let job = self.job.as_ref()?;
        if job.speed <= 0.0 {
            return None;
        }
        Some((total.saturating_sub(job.bytes) as f64 / job.speed) as u64)
    }

    pub fn spawn(&mut self) {
        if !self.is_queued() {
            return;
        }
        let mut cmd_result = std::env::current_exe().map(Command::new);
        if let Ok(cmd) = &mut cmd_result {
            cmd.arg("download").arg(&self.target).arg("--progress-json");
            if let Some(dest) = &self.dest {
                cmd.arg("--dest").arg(dest);
            }
        }
        self.job = Some(match cmd_result {
            Ok(cmd) => Job::spawn(cmd, Some(self.watch_dir.clone()))
                .unwrap_or_else(|err| Job::failed(format!("failed to launch download: {err}"))),
            Err(err) => Job::failed(format!("failed to resolve ax-engine executable: {err}")),
        });
    }

    /// Advance the child job; reports the edge when it just finished or failed
    /// and whether anything visible (besides the spinner) changed.
    pub fn tick(&mut self) -> (DownloadOutcome, bool) {
        let Some(job) = &mut self.job else {
            return (DownloadOutcome::Pending, false);
        };
        let before = job.done;
        let JobTick { fresh, material } = job.tick();
        let mut changed = material;
        for line in fresh {
            if let Some((_done, _total, message)) = parse_progress_event(&line) {
                self.phase = Some(message);
                changed = true;
            }
        }
        let finished_ok = before.is_none() && job.done == Some(0);
        if finished_ok && self.resolved_path.is_none() {
            self.resolved_path = parse_output_path_from_log(&job.log)
                .or_else(|| catalog::repo_snapshot_dir(self.repo_id));
        }
        let outcome = match (before, job.done) {
            (None, Some(0)) => DownloadOutcome::Finished,
            // User-cancelled jobs already got their toast from the cancel flow.
            (None, Some(_)) if !self.cancelled => DownloadOutcome::Failed,
            _ => DownloadOutcome::Pending,
        };
        (outcome, changed)
    }

    pub fn cancel(&mut self) {
        match &mut self.job {
            // Running: kill the child and mark it user-cancelled so the queue
            // label reads "cancelled" instead of "failed (-130)".
            Some(job) if job.done.is_none() => {
                job.cancel();
                self.cancelled = true;
            }
            // Queued: nothing to kill, just mark it.
            None => self.cancelled = true,
            // Already finished: keep the ready/failed label as-is.
            Some(_) => {}
        }
    }
}

pub(super) fn parse_output_path_from_log(lines: &[String]) -> Option<PathBuf> {
    for line in lines.iter().rev() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Path:") {
            let path = rest.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
        // prepare_mtp_sidecar / prepare_gemma4 print "Output dir:   <path>".
        if let Some(rest) = trimmed.strip_prefix("Output dir:") {
            let path = rest.trim();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
        // "Next: ax-engine serve <path> --port …" (single-line form).
        if let Some(idx) = trimmed.find("ax-engine serve ") {
            let rest = trimmed[idx + "ax-engine serve ".len()..].trim();
            let path = rest.split_whitespace().next().unwrap_or("");
            if !path.is_empty() && path.starts_with('/') {
                return Some(PathBuf::from(path));
            }
        }
    }
    let mut next_is_path = false;
    for line in lines {
        if next_is_path {
            let value = line.trim();
            if !value.is_empty() {
                return Some(PathBuf::from(value));
            }
        }
        let trimmed = line.trim();
        // Qwen sidecar: "Sidecar ready at:"; Gemma: "… package ready at:".
        next_is_path = trimmed == "Sidecar ready at:"
            || trimmed.ends_with("package ready at:")
            || trimmed.ends_with("package ready at");
    }
    None
}

pub(super) fn format_eta(seconds: u64) -> String {
    if seconds >= 3600 {
        format!("{}h{:02}m", seconds / 3600, (seconds % 3600) / 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}
