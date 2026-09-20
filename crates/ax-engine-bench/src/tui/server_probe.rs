//! "Is a server there yet" detection for the TUI.
//!
//! Kept out of the `tui` root so the log-line heuristic, the `/health` body
//! parser, and the blocking probe sit together as one concern instead of being
//! interleaved with the `App` state machine.

use std::process::Command;
use std::time::Duration;

/// How often to probe `/health` when looking for a ready listener (or confirming
/// an external one is still up). Kept low so Chat flips green quickly after the
/// user starts `ax-engine-server` outside the TUI.
pub(super) const SERVER_HEALTH_PROBE_INTERVAL: Duration = Duration::from_secs(1);

/// True when a captured `ax-engine-server` log line means the HTTP listener is up.
///
/// Handles both the stable operator line (no tracing) and the structured
/// `tracing` form used when `RUST_LOG` / `AX_ENGINE_SERVER_LOG` is set:
/// - `ax-engine-server preview listening on http://127.0.0.1:8080 ...`
/// - `INFO ax-engine-server preview listening bind_address=127.0.0.1:8080 ...`
pub(super) fn server_log_indicates_ready(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.contains("listening on http://") {
        return true;
    }
    // Structured tracing: message is "ax-engine-server preview listening" plus
    // bind_address=... fields (no "on http://" substring).
    if lower.contains("preview listening")
        && (lower.contains("bind_address=") || lower.contains("listening on"))
    {
        return true;
    }
    false
}

/// Parsed `/health` body from a live `ax-engine-server`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ServerHealth {
    pub(super) model_id: Option<String>,
}

/// Parse a successful `GET /health` JSON body. Requires `"status":"ok"`.
pub(super) fn parse_health_body(body: &str) -> Option<ServerHealth> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    if value.get("status").and_then(|s| s.as_str()) != Some("ok") {
        return None;
    }
    // Prefer the ax-engine identity field when present; older/test stubs may
    // omit it as long as status is ok.
    if let Some(service) = value.get("service").and_then(|s| s.as_str())
        && service != "ax-engine-server"
    {
        return None;
    }
    let model_id = value
        .get("model_id")
        .and_then(|m| m.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_string);
    Some(ServerHealth { model_id })
}

/// `http://host:port`, bracketing bare IPv6 hosts for URL parsers.
pub(super) fn format_http_base_url(host: &str, port: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("http://[{host}]:{port}")
    } else {
        format!("http://{host}:{port}")
    }
}

/// Blocking one-shot health probe used off the UI thread.
pub(super) fn probe_server_health(base_url: &str) -> Option<ServerHealth> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    let output = Command::new("curl")
        .args([
            "-sS",
            "-m",
            "1",
            "--connect-timeout",
            "1",
            "-H",
            "Accept: application/json",
            &url,
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&output.stdout);
    parse_health_body(&body)
}
