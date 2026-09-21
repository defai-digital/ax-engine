//! Serving for the TUI `App`: starting `ax-engine-server`, tracking when it is
//! actually ready to serve, stopping it again, and the queue hand-off that
//! launches the next pending download.
//!
//! Split out of the `tui` root, which had grown into a single ~2.2k-line
//! `impl App`. `App`'s fields stay private to `tui`; sibling modules inside the
//! TUI can still reach them, exactly as `screens::chat` already does. Methods
//! the sibling screens call are `pub(super)` for the same reason.

use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{self, TryRecvError};
use std::thread;
use std::time::Instant;

use super::App;
use super::catalog;
use super::jobs::{DownloadTask, Job};
use super::server_probe::{
    SERVER_HEALTH_PROBE_INTERVAL, ServerHealth, format_http_base_url, probe_server_health,
    server_log_indicates_ready,
};

impl App {
    /// Base URL for the configured Serve host/port (defaults applied).
    pub(crate) fn configured_server_url(&self) -> Option<String> {
        if self.host_error().is_some() || self.port_error().is_some() {
            return None;
        }
        let host = if self.host.trim().is_empty() {
            "127.0.0.1"
        } else {
            self.host.trim()
        };
        let port = if self.port.trim().is_empty() {
            "31418"
        } else {
            self.port.trim()
        };
        // Empty fields are defaults; non-empty invalid values already rejected.
        port.parse::<u16>().ok().filter(|&p| p > 0)?;
        Some(format_http_base_url(host, port))
    }

    /// True when the given catalog variant is the model the running server
    /// was started with. `server_model` is the display label for
    /// download-served models and the profile label for direct serves —
    /// check both (same rule as the delete modal's in-use warning).
    pub(crate) fn is_served_variant(&self, family_idx: usize, variant_idx: usize) -> bool {
        if !self.server_running() {
            return false;
        }
        let Some(variant) = self
            .families
            .get(family_idx)
            .and_then(|f| f.variants.get(variant_idx))
        else {
            return false;
        };
        let label = format!(
            "{} {}",
            self.families[family_idx].display_name(),
            variant.precision()
        );
        self.server_model
            .as_deref()
            .is_some_and(|m| m == label || m == variant.profile.label)
    }

    /// Returns whether a server job was spawned. Callers that arm follow-up
    /// state (auto-chat, wizard reset) must key on it: a refused start
    /// (invalid host/port, server already running, unknown variant) leaves
    /// the wizard where it was.
    pub(super) fn serve_installed(&mut self, family_idx: usize, variant_idx: usize) -> bool {
        if let Some(err) = self.host_error() {
            self.toast_error(err);
            return false;
        }
        if let Some(err) = self.port_error() {
            self.toast_error(err);
            return false;
        }
        if self.server_running() {
            self.toast_warn("stop the running server first (x on Serve)");
            return false;
        }
        let Some(variant) = self
            .families
            .get(family_idx)
            .and_then(|f| f.variants.get(variant_idx))
        else {
            return false;
        };
        let profile = variant.profile;
        let artifacts_dir = catalog::repo_snapshot_dir(profile.repo_id);
        self.spawn_server(profile.preset, artifacts_dir, profile.label)
    }

    /// See [`Self::serve_installed`] for the return contract.
    pub(super) fn start_server_for_download(&mut self, download_idx: usize) -> bool {
        if let Some(err) = self.host_error() {
            self.toast_error(err);
            return false;
        }
        if let Some(err) = self.port_error() {
            self.toast_error(err);
            return false;
        }
        if self.server_running() {
            self.toast_warn("stop the running server first (x on Serve)");
            return false;
        }
        let Some(task) = self.downloads.get(download_idx) else {
            return false;
        };
        if !task.is_ready() {
            return false;
        }
        // Prefer the path reported by the direct Hub download, then resolve
        // its usable cache snapshot. AutomatosX MTP packs are self-contained.
        let artifacts_dir = task
            .output_path()
            .or_else(|| catalog::repo_snapshot_dir(&task.repo_id));
        let label = task.label.clone();
        self.spawn_server(task.preset, artifacts_dir, &label)
    }

    /// Returns whether a child was launched. Any previous job is cancelled
    /// first so a replaced entry never leaves an orphan holding the port.
    fn spawn_server(
        &mut self,
        preset: Option<&str>,
        artifacts_dir: Option<PathBuf>,
        model_label: &str,
    ) -> bool {
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        self.server_ready = false;
        self.server_ready_scan = 0;
        self.external_server = false;
        self.server_probe = None;
        self.last_server_probe = None;
        self.server_probe_url = None;
        // Every path through here replaces the server job — re-pin its log.
        self.serve_log_scroll.pin_to_bottom();
        let host = if self.host.trim().is_empty() {
            "127.0.0.1".to_string()
        } else {
            self.host.trim().to_string()
        };
        let port = if self.port.trim().is_empty() {
            "31418".to_string()
        } else {
            self.port.trim().to_string()
        };

        let server_bin = crate::find_executable("ax-engine-server");
        let mut cmd = Command::new(server_bin);
        cmd.arg("--host")
            .arg(&host)
            .arg("--port")
            .arg(&port)
            .arg("--mlx");
        if let Some(preset) = preset {
            cmd.arg("--preset").arg(preset);
        }
        match &artifacts_dir {
            Some(dir) => {
                cmd.arg("--mlx-model-artifacts-dir").arg(dir);
            }
            // Direct alias installs: resolve the HF snapshot by preset when we
            // do not yet have an explicit snapshot path (rare after a clean
            // download; common when re-serving a known alias).
            None if preset.is_some() => {
                cmd.arg("--resolve-model-artifacts").arg("hf-cache");
            }
            None => {
                let reason = "no server artifact path could be resolved for this download";
                self.server = Some(Job::failed(reason.into()));
                self.server_url = None;
                self.server_model = None;
                self.toast_error(reason);
                return false;
            }
        }
        // Record the resolved binary in the log so Serve failures are diagnosable
        // when PATH / sibling resolution picks the wrong install.
        let bin_display = cmd.get_program().to_string_lossy().into_owned();
        match Job::spawn(cmd, None) {
            Ok(mut job) => {
                job.log
                    .push(format!("spawning {bin_display} for {model_label}"));
                self.server = Some(job);
                self.server_url = Some(format_http_base_url(&host, &port));
                self.server_model = Some(model_label.to_string());
                true
            }
            Err(err) => {
                self.server = Some(Job::failed(format!(
                    "failed to launch server ({bin_display}): {err}"
                )));
                self.server_url = None;
                self.server_model = None;
                self.toast_error(format!("failed to launch server: {err}"));
                false
            }
        }
    }

    /// URL a health probe should target: the spawn-time URL while our own
    /// child is alive (editing the Serve fields must not re-point the probe
    /// at some other listener), otherwise whatever the fields configure.
    fn probe_target_url(&self) -> Option<String> {
        if self.managed_server_alive()
            && let Some(url) = &self.server_url
        {
            return Some(url.clone());
        }
        self.configured_server_url()
    }

    /// Track the managed server job in both directions: flip `server_ready` on
    /// once the log confirms the bind, and back off if the process has since
    /// exited so Chat stops accepting input. External (probe-only) readiness
    /// is handled by [`Self::tick_server_health_probe`].
    pub fn update_server_ready(&mut self) {
        let Some(job) = &self.server else {
            return;
        };
        if job.done.is_some() {
            // Managed child exited. Drop managed readiness; leave external
            // discovery to the health probe (another process may still hold
            // the port, but that is rare after our kill).
            if self.server_ready && !self.external_server {
                self.server_ready = false;
                self.toast_warn("server stopped — restart it on Serve");
            }
            if !self.external_server {
                // The label and URL described the dead child; keep the job
                // (its log and exit code explain the failure).
                self.server_model = None;
                self.server_url = None;
            }
            return;
        }
        if self.server_ready {
            // Managed job still alive and already ready (log or health).
            self.external_server = false;
            return;
        }
        // Only scan lines that arrived since the last check. The cursor is a
        // stable position (`log_dropped + index`), so a LOG_CAP front drain
        // cannot hide lines that landed below the old absolute index.
        let start = self
            .server_ready_scan
            .saturating_sub(job.log_dropped)
            .min(job.log.len());
        for line in &job.log[start..] {
            if server_log_indicates_ready(line) {
                self.server_ready = true;
                self.external_server = false;
                break;
            }
        }
        self.server_ready_scan = job.log_dropped + job.log.len();
    }

    /// Apply a completed `/health` result. Pure state transition used by the
    /// tick loop and unit tests (no I/O).
    pub(crate) fn apply_server_health(&mut self, health: Option<ServerHealth>) -> bool {
        let was_ready = self.server_ready;
        let was_external = self.external_server;
        let managed_alive = self.managed_server_alive();
        match health {
            Some(health) => {
                if let Some(url) = self.probe_target_url() {
                    self.server_url = Some(url);
                }
                self.server_ready = true;
                if managed_alive {
                    // Our child is the listener (or shares the port); treat as
                    // managed so Stop kills the job rather than only detaching.
                    self.external_server = false;
                } else {
                    self.external_server = true;
                    // A finished (failed or exited) job no longer describes
                    // the listener; drop it so its error line stops showing.
                    self.server = None;
                    if let Some(model_id) = health.model_id {
                        self.server_model = Some(model_id);
                    }
                }
            }
            None => {
                if self.external_server && !managed_alive {
                    // External listener went away.
                    self.external_server = false;
                    self.server_ready = false;
                    self.server_url = None;
                    self.server_model = None;
                }
                // Managed-but-not-ready: keep waiting on the child log. Managed
                // ready is not cleared by a failed probe (health can 503 while
                // the worker is still usable for chat, or curl can flap).
            }
        }
        self.server_ready != was_ready || self.external_server != was_external
    }

    /// Non-blocking `/health` poll. Discovers servers started outside the TUI
    /// and backstops log-line readiness for managed children.
    pub(super) fn tick_server_health_probe(&mut self) -> bool {
        // Drain a finished probe first.
        if let Some(rx) = &self.server_probe {
            match rx.try_recv() {
                Ok(result) => {
                    self.server_probe = None;
                    // A probe answers for the URL it was launched with. If the
                    // Serve fields changed (or became invalid) meanwhile, that
                    // answer says nothing about the configured server: drop
                    // it and let the re-probe below run against the new URL.
                    if self.server_probe_url == self.probe_target_url() {
                        return self.apply_server_health(result);
                    }
                    self.server_probe_url = None;
                }
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => {
                    self.server_probe = None;
                }
            }
        }

        // Managed child already green via its log: job liveness is enough.
        // Otherwise keep probing — that covers external discovery, log-line
        // backstop while starting, and re-attach after a managed crash if
        // something else still holds the port.
        if self.server_ready && !self.external_server && self.managed_server_alive() {
            return false;
        }

        let Some(url) = self.probe_target_url() else {
            return false;
        };
        let url_changed = self.server_probe_url.as_deref() != Some(url.as_str());
        let due = self
            .last_server_probe
            .is_none_or(|t| t.elapsed() >= SERVER_HEALTH_PROBE_INTERVAL);
        if !due && !url_changed {
            return false;
        }

        self.last_server_probe = Some(Instant::now());
        self.server_probe_url = Some(url.clone());
        let (tx, rx) = mpsc::channel();
        self.server_probe = Some(rx);
        thread::spawn(move || {
            let _ = tx.send(probe_server_health(&url));
        });
        false
    }

    pub(super) fn stop_server(&mut self) {
        let had_managed = self.managed_server_alive();
        let had_external = self.external_server;
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        self.server = None;
        self.server_url = None;
        self.server_ready = false;
        self.server_ready_scan = 0;
        self.server_model = None;
        self.external_server = false;
        self.server_probe = None;
        self.last_server_probe = None;
        self.server_probe_url = None;
        self.serve_log_scroll.pin_to_bottom();
        if !had_managed && had_external {
            // External-only: we cannot kill the process; detach so the UI
            // stops claiming it. User stops the real process outside.
            self.toast_warn("detached external server — stop the process outside the TUI");
        }
    }

    /// Most recent non-empty server log line, surfaced when startup fails.
    /// Prefer real startup failures over trailing MLX kernel noise (`mlx error:`).
    pub fn server_error_line(&self) -> Option<String> {
        let job = self.server.as_ref()?;
        job.done?;
        let is_hard_error = |line: &str| {
            let t = line.trim();
            t.starts_with("Error:")
                || t.contains("ERROR")
                || t.contains("panic")
                || t.contains("could not")
                || t.contains("InvalidInput")
                || t.contains("failed to launch")
                || t.contains("Address already in use")
        };
        let is_soft_error = |line: &str| {
            let t = line.trim();
            !t.is_empty()
                && !t.to_ascii_lowercase().starts_with("mlx error:")
                && (t.contains("error") || t.contains("failed"))
        };
        job.log
            .iter()
            .rev()
            .find(|line| is_hard_error(line))
            .or_else(|| job.log.iter().rev().find(|line| is_soft_error(line)))
            .or_else(|| job.log.iter().rev().find(|line| !line.trim().is_empty()))
            .cloned()
    }

    // -- download queue ---------------------------------------------------------

    pub fn start_next_queued_download(&mut self) {
        if self.downloads.iter().any(DownloadTask::is_running) {
            return;
        }
        // A launch-time failure (e.g. executable not found) sets `done` via
        // Job::failed before the first tick, so tick() never sees a
        // None → Some edge and never toasts — surface it here instead.
        let launch_error =
            if let Some(task) = self.downloads.iter_mut().find(|task| task.is_queued()) {
                task.spawn();
                task.job
                    .as_ref()
                    .filter(|job| job.done.is_some())
                    .and_then(|job| job.log.first().cloned())
            } else {
                None
            };
        if let Some(message) = launch_error {
            self.toast_error(message);
        }
    }
}
