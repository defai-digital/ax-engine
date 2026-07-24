use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;
use std::fmt;
use std::io::{Cursor, Read};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Default connect timeout for delegated local/remote HTTP backends.
pub const DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS: u64 = 30;
/// Default read/write timeout for delegated HTTP I/O. Reads intentionally share
/// the longer I/O timeout because streaming completions can stay open for the
/// full generation window.
pub const DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS: u64 = 300;
const DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS: usize = 2;
const DELEGATED_HTTP_TRANSPORT_RETRY_BACKOFF: Duration = Duration::from_millis(25);
pub const DEFAULT_DELEGATED_HTTP_MAX_ERROR_BODY_BYTES: usize = 8 * 1024;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DelegatedHttpRetryPolicy {
    Never,
    Idempotent {
        max_attempts: NonZeroUsize,
        initial_backoff: Duration,
        max_backoff: Duration,
    },
}

impl DelegatedHttpRetryPolicy {
    pub fn readiness_default() -> Self {
        Self::Idempotent {
            max_attempts: NonZeroUsize::new(3).expect("three is non-zero"),
            initial_backoff: Duration::from_millis(50),
            max_backoff: Duration::from_millis(500),
        }
    }

    fn max_attempts(&self) -> usize {
        match self {
            Self::Never => 1,
            Self::Idempotent { max_attempts, .. } => max_attempts.get(),
        }
    }

    fn backoff(&self, completed_attempts: usize) -> Duration {
        let Self::Idempotent {
            initial_backoff,
            max_backoff,
            ..
        } = self
        else {
            return Duration::ZERO;
        };

        let exponent = completed_attempts.saturating_sub(1).min(16) as u32;
        let scaled = initial_backoff
            .checked_mul(2_u32.saturating_pow(exponent))
            .unwrap_or(*max_backoff)
            .min(*max_backoff);
        // A small process-local jitter prevents synchronized readiness probes
        // without adding a random-number dependency or affecting retry bounds.
        let jitter_cap_ms = (scaled.as_millis() / 4).min(u128::from(u64::MAX)) as u64;
        if jitter_cap_ms == 0 {
            return scaled;
        }
        let jitter_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.subsec_nanos() as u64 % (jitter_cap_ms + 1))
            .unwrap_or(0);
        scaled.saturating_add(Duration::from_millis(jitter_ms))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum DelegatedProxyPolicy {
    /// Preserve the historical delegated-client behavior.
    Environment,
    /// Ignore proxy environment variables for a co-located inference worker.
    #[default]
    Disabled,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum DelegatedRedirectPolicy {
    /// Preserve the historical ureq redirect limit for existing providers.
    FollowDefault,
    /// Do not follow redirects, so credentials cannot cross origins.
    #[default]
    Disabled,
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
pub enum DelegatedTlsPolicy {
    SystemRoots,
    SystemRootsWithCustomCa {
        ca_pem: Vec<u8>,
        sha256: String,
    },
}

impl DelegatedTlsPolicy {
    pub fn system_roots_with_ca_file(path: impl AsRef<Path>) -> Result<Self, DelegatedHttpConfigError> {
        let path = path.as_ref();
        let ca_pem = std::fs::read(path).map_err(|source| DelegatedHttpConfigError::ReadCaFile {
            path: path.display().to_string(),
            source,
        })?;
        validate_ca_pem(&ca_pem)?;
        let sha256 = format!("{:x}", Sha256::digest(&ca_pem));
        Ok(Self::SystemRootsWithCustomCa { ca_pem, sha256 })
    }

    pub fn trust_fingerprint(&self) -> &str {
        match self {
            Self::SystemRoots => "system-roots",
            Self::SystemRootsWithCustomCa { sha256, .. } => sha256,
        }
    }
}

impl Default for DelegatedTlsPolicy {
    fn default() -> Self {
        Self::SystemRoots
    }
}

impl fmt::Debug for DelegatedTlsPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SystemRoots => formatter.write_str("SystemRoots"),
            Self::SystemRootsWithCustomCa { sha256, .. } => formatter
                .debug_struct("SystemRootsWithCustomCa")
                .field("sha256", sha256)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct DelegatedBearerCredential(String);

impl DelegatedBearerCredential {
    pub fn new(value: impl Into<String>) -> Result<Self, DelegatedHttpConfigError> {
        let value = value.into();
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(DelegatedHttpConfigError::EmptyBearerCredential);
        }
        if trimmed.contains(['\r', '\n']) {
            return Err(DelegatedHttpConfigError::InvalidHeaderValue {
                header: "authorization",
            });
        }
        Ok(Self(trimmed.to_string()))
    }

    fn authorization_value(&self) -> String {
        format!("Bearer {}", self.0)
    }
}

impl fmt::Debug for DelegatedBearerCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("DelegatedBearerCredential([REDACTED])")
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DelegatedHttpHeaders {
    pub accept: Option<String>,
    pub authorization: Option<DelegatedBearerCredential>,
    pub user_agent: Option<String>,
    pub request_id: Option<String>,
}

impl DelegatedHttpHeaders {
    pub fn with_accept(mut self, value: impl Into<String>) -> Self {
        self.accept = Some(value.into());
        self
    }

    pub fn with_bearer(mut self, credential: Option<DelegatedBearerCredential>) -> Self {
        self.authorization = credential;
        self
    }

    pub fn with_user_agent(mut self, value: impl Into<String>) -> Self {
        self.user_agent = Some(value.into());
        self
    }

    pub fn with_request_id(mut self, value: impl Into<String>) -> Self {
        self.request_id = Some(value.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DelegatedHttpRequestOptions {
    pub timeouts: DelegatedHttpTimeouts,
    pub retry: DelegatedHttpRetryPolicy,
    pub headers: DelegatedHttpHeaders,
    pub tls: DelegatedTlsPolicy,
    pub proxy: DelegatedProxyPolicy,
    pub redirects: DelegatedRedirectPolicy,
    pub max_error_body_bytes: usize,
}

impl Default for DelegatedHttpRequestOptions {
    fn default() -> Self {
        Self {
            timeouts: DelegatedHttpTimeouts::default(),
            retry: DelegatedHttpRetryPolicy::Never,
            headers: DelegatedHttpHeaders::default(),
            tls: DelegatedTlsPolicy::default(),
            proxy: DelegatedProxyPolicy::Disabled,
            redirects: DelegatedRedirectPolicy::Disabled,
            max_error_body_bytes: DEFAULT_DELEGATED_HTTP_MAX_ERROR_BODY_BYTES,
        }
    }
}

#[derive(Debug, Error)]
pub enum DelegatedHttpConfigError {
    #[error("delegated bearer credential must not be empty")]
    EmptyBearerCredential,
    #[error("delegated HTTP header {header} contains an invalid newline")]
    InvalidHeaderValue { header: &'static str },
    #[error("failed to read delegated custom CA file {path}: {source}")]
    ReadCaFile {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("delegated custom CA PEM did not contain a valid certificate")]
    EmptyCaFile,
    #[error("delegated custom CA PEM is invalid: {0}")]
    InvalidCaFile(String),
    #[error("delegated max_error_body_bytes must be greater than zero")]
    InvalidMaxErrorBodyBytes,
}

#[derive(Debug)]
pub(crate) enum DelegatedHttpRequestError {
    Serialize(serde_json::Error),
    Status {
        status: u16,
        body: String,
        truncated: bool,
    },
    Request(Box<ureq::Error>),
    Config(DelegatedHttpConfigError),
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DelegatedHttpAgentKey {
    timeouts: DelegatedHttpTimeouts,
    tls: DelegatedTlsPolicy,
    proxy: DelegatedProxyPolicy,
    redirects: DelegatedRedirectPolicy,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DelegatedHttpTimeouts {
    pub connect: Duration,
    pub read: Duration,
    pub write: Duration,
}

impl DelegatedHttpTimeouts {
    pub fn from_secs(connect_secs: u64, read_secs: u64, write_secs: u64) -> Self {
        Self {
            connect: Duration::from_secs(connect_secs),
            read: Duration::from_secs(read_secs),
            write: Duration::from_secs(write_secs),
        }
    }

    pub fn default_connect_secs() -> u64 {
        DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS
    }

    pub fn default_io_secs() -> u64 {
        DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS
    }

    pub(crate) fn agent(self) -> ureq::Agent {
        // A panic elsewhere while holding this lock must not permanently
        // poison the shared agent cache for every later request; recover the
        // last-known-good map instead of propagating the poison.
        let mut agents = delegated_http_agents()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        agents
            .entry(self)
            .or_insert_with(|| self.build_agent())
            .clone()
    }

    fn build_agent(self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_connect(self.connect)
            .timeout_read(self.read)
            .timeout_write(self.write)
            .build()
    }
}

fn delegated_http_agents() -> &'static Mutex<BTreeMap<DelegatedHttpTimeouts, ureq::Agent>> {
    static AGENTS: OnceLock<Mutex<BTreeMap<DelegatedHttpTimeouts, ureq::Agent>>> = OnceLock::new();
    AGENTS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[derive(Debug)]
pub(crate) enum DelegatedHttpPostError {
    Serialize(serde_json::Error),
    Status { status: u16, body: String },
    Request(Box<ureq::Error>),
}

pub(crate) fn send_json_post_with_retry<T>(
    endpoint: &str,
    payload: &T,
    timeouts: DelegatedHttpTimeouts,
    accept: Option<&str>,
) -> Result<ureq::Response, DelegatedHttpPostError>
where
    T: Serialize + ?Sized,
{
    let body = serde_json::to_vec(payload).map_err(DelegatedHttpPostError::Serialize)?;
    let agent = timeouts.agent();

    for attempt in 1..=DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS {
        let mut request = agent.post(endpoint).set("Content-Type", "application/json");
        if let Some(accept) = accept {
            request = request.set("Accept", accept);
        }

        match request.send_bytes(&body) {
            Ok(response) => return Ok(response),
            Err(ureq::Error::Status(status, response)) => {
                let body = response
                    .into_string()
                    .unwrap_or_else(|_| "<failed to read response body>".to_string());
                return Err(DelegatedHttpPostError::Status {
                    status,
                    body: body.trim().to_string(),
                });
            }
            Err(source)
                if attempt < DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS
                    && is_retryable_transport_error(&source) =>
            {
                tracing::warn!(
                    endpoint,
                    attempt,
                    max_attempts = DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS,
                    error = %source,
                    "delegated HTTP transport request failed; retrying once"
                );
                std::thread::sleep(DELEGATED_HTTP_TRANSPORT_RETRY_BACKOFF);
            }
            Err(source) => return Err(DelegatedHttpPostError::Request(Box::new(source))),
        }
    }

    unreachable!("delegated HTTP retry loop always returns from its final attempt")
}

pub(crate) fn send_json_post_request<T, E, F>(
    endpoint: &str,
    payload: &T,
    accept: Option<&str>,
    timeouts: DelegatedHttpTimeouts,
    map_error: F,
) -> Result<ureq::Response, E>
where
    T: Serialize + ?Sized,
    F: FnOnce(DelegatedHttpPostError) -> E,
{
    send_json_post_with_retry(endpoint, payload, timeouts, accept).map_err(map_error)
}

pub(crate) fn parse_json_response<T, E, F>(response: ureq::Response, map_error: F) -> Result<T, E>
where
    T: DeserializeOwned,
    F: FnOnce(serde_json::Error) -> E,
{
    serde_json::from_reader(response.into_reader()).map_err(map_error)
}

pub(crate) fn normalize_base_url(mut value: String) -> String {
    while value.ends_with('/') {
        value.pop();
    }
    value
}

fn is_retryable_transport_error(error: &ureq::Error) -> bool {
    matches!(
        error,
        ureq::Error::Transport(transport)
            if matches!(
                transport.kind(),
                ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns | ureq::ErrorKind::Io
            )
    )
}

impl Default for DelegatedHttpTimeouts {
    fn default() -> Self {
        Self::from_secs(
            DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS,
            DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
            DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;

    #[test]
    fn delegated_http_timeouts_reuses_cached_agent_for_same_timeouts() {
        let timeouts = DelegatedHttpTimeouts::from_secs(97, 98, 99);

        let _ = timeouts.agent();
        assert!(
            delegated_http_agents()
                .lock()
                .expect("agent cache should lock")
                .contains_key(&timeouts)
        );

        let _ = timeouts.agent();
        assert!(
            delegated_http_agents()
                .lock()
                .expect("agent cache should lock")
                .contains_key(&timeouts)
        );
    }

    #[test]
    fn send_json_post_with_retry_retries_one_transport_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener should bind");
        let endpoint = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let attempts = Arc::new(AtomicUsize::new(0));
        let server_attempts = Arc::clone(&attempts);

        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("first connection should arrive");
            server_attempts.fetch_add(1, Ordering::SeqCst);
            drop(stream);

            let (mut stream, _) = listener.accept().expect("retry connection should arrive");
            server_attempts.fetch_add(1, Ordering::SeqCst);
            let mut buf = Vec::new();
            let mut tmp = [0_u8; 256];
            loop {
                let n = stream.read(&mut tmp).expect("request should read");
                buf.extend_from_slice(&tmp[..n]);
                let s = String::from_utf8_lossy(&buf);
                if s.contains(r#"{"prompt":"hello"}"#) || n == 0 {
                    break;
                }
            }
            let request = String::from_utf8_lossy(&buf);
            assert!(request.starts_with("POST /v1/completions HTTP/1.1"));
            assert!(request.contains("Content-Type: application/json"));
            assert!(request.contains(r#"{"prompt":"hello"}"#));

            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\n\
                      Content-Type: application/json\r\n\
                      Content-Length: 11\r\n\
                      Connection: close\r\n\
                      \r\n\
                      {\"ok\":true}",
                )
                .expect("response should write");
        });

        let response = send_json_post_with_retry(
            &endpoint,
            &json!({"prompt": "hello"}),
            DelegatedHttpTimeouts::from_secs(1, 1, 1),
            None,
        )
        .expect("request should succeed after one retry");

        assert_eq!(response.status(), 200);
        handle.join().expect("server thread should finish");
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }
}
