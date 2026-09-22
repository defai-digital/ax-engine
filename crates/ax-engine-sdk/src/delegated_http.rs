use serde::{Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// Default connect timeout for delegated local/remote HTTP backends.
pub const DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS: u64 = 30;
/// Default read/write timeout for delegated HTTP streaming I/O. This bounds a
/// single socket read/write while an SSE stream is open, so it only needs to
/// cover inter-chunk gaps, not the full generation window. ureq surfaces a
/// violated read deadline as a plain `ErrorKind::Io` transport error, which is
/// exactly the error class the retry loop must NOT re-send (see
/// [`is_retryable_transport_error`]).
pub const DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS: u64 = 300;
/// Default floor for the read timeout of blocking (non-streaming) delegated
/// generation calls. A blocking generation produces no response bytes until
/// the whole completion is finished, so its read timeout must cover the full
/// generation window; the shorter streaming I/O timeout above would abort any
/// generation longer than 300 s. One hour by default; configure a longer one
/// by raising `read` in [`DelegatedHttpTimeouts`] (the blocking floor is only
/// ever raised, never lowered, by [`DelegatedHttpTimeouts::for_blocking_generation`]).
pub const DEFAULT_DELEGATED_HTTP_BLOCKING_READ_TIMEOUT_SECS: u64 = 3600;
const DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS: usize = 2;
const DELEGATED_HTTP_TRANSPORT_RETRY_BACKOFF: Duration = Duration::from_millis(25);

type DelegatedHttpAgentCache =
    Mutex<BTreeMap<(DelegatedHttpTimeouts, Option<String>), ureq::Agent>>;

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

    /// Timeouts for blocking (non-streaming) delegated generation calls: the
    /// same connect and write limits, with `read` raised to at least
    /// [`DEFAULT_DELEGATED_HTTP_BLOCKING_READ_TIMEOUT_SECS`]. A blocking
    /// generation sends no response bytes until it finishes, so a streaming
    /// I/O-scale read deadline would abort long generations. The floor only
    /// raises a configured `read`; a value above it is preserved as-is.
    pub fn for_blocking_generation(self) -> Self {
        Self {
            read: self.read.max(Duration::from_secs(
                DEFAULT_DELEGATED_HTTP_BLOCKING_READ_TIMEOUT_SECS,
            )),
            ..self
        }
    }

    pub(crate) fn agent(self, accept: Option<&str>) -> ureq::Agent {
        // A panic elsewhere while holding this lock must not permanently
        // poison the shared agent cache for every later request; recover the
        // last-known-good map instead of propagating the poison.
        let key = (self, accept.map(str::to_string));
        let mut agents = delegated_http_agents()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        agents
            .entry(key)
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

fn delegated_http_agents() -> &'static DelegatedHttpAgentCache {
    static AGENTS: OnceLock<DelegatedHttpAgentCache> = OnceLock::new();
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
    // Streaming readers intentionally stop at the OpenAI [DONE] sentinel.
    // Some upstreams leave an SSE separator buffered after that sentinel, so
    // ureq may close rather than recycle that connection. Keep streaming and
    // non-streaming pools separate so a completed stream cannot evict the
    // low-latency JSON connection used by the next request.
    let agent = timeouts.agent(accept);

    post_with_retry(endpoint, move || {
        let mut request = agent.post(endpoint).set("Content-Type", "application/json");
        if let Some(accept) = accept {
            request = request.set("Accept", accept);
        }
        request.send_bytes(&body).map_err(Box::new)
    })
}

/// Retry driver for one delegated HTTP POST. A failed attempt is re-sent only
/// when the error proves the request never reached the server (see
/// [`is_retryable_transport_error`]); anything that may have happened after
/// the request body was written - including read timeouts on blocking
/// generations - fails the call so the server can never generate twice.
/// The attempt closure boxes its error to keep the `Err` payload small.
fn post_with_retry<F>(
    endpoint: &str,
    mut send_attempt: F,
) -> Result<ureq::Response, DelegatedHttpPostError>
where
    F: FnMut() -> Result<ureq::Response, Box<ureq::Error>>,
{
    for attempt in 1..=DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS {
        match send_attempt() {
            Ok(response) => return Ok(response),
            Err(source) => {
                if attempt < DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS
                    && is_retryable_transport_error(&source)
                {
                    tracing::warn!(
                        endpoint,
                        attempt,
                        max_attempts = DELEGATED_HTTP_TRANSPORT_MAX_ATTEMPTS,
                        error = %source,
                        "delegated HTTP transport request failed before reaching the server; retrying once"
                    );
                    std::thread::sleep(DELEGATED_HTTP_TRANSPORT_RETRY_BACKOFF);
                    continue;
                }

                match *source {
                    ureq::Error::Status(status, response) => {
                        let body = response
                            .into_string()
                            .unwrap_or_else(|_| "<failed to read response body>".to_string());
                        return Err(DelegatedHttpPostError::Status {
                            status,
                            body: body.trim().to_string(),
                        });
                    }
                    source => return Err(DelegatedHttpPostError::Request(Box::new(source))),
                }
            }
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

/// True only for transport errors that prove the request never reached the
/// server, so a single retry cannot make the backend do the work twice.
/// In ureq 2.x, `Dns` and `ConnectionFailed` are produced exclusively while
/// opening the connection (DNS resolution, refused, connect timeout) - before
/// any request byte is written. Everything after that (write failures, read
/// timeouts, EOF/reset mid-response) surfaces as `ErrorKind::Io`, so Io must
/// NOT be retried: a blocking generation slower than the read timeout also
/// fails with Io after the server already received the full request.
fn is_retryable_transport_error(error: &ureq::Error) -> bool {
    matches!(
        error,
        ureq::Error::Transport(transport)
            if matches!(
                transport.kind(),
                ureq::ErrorKind::ConnectionFailed | ureq::ErrorKind::Dns
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
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::Read;
    use std::net::TcpListener;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Instant;

    #[test]
    fn delegated_http_timeouts_reuses_cached_agent_for_same_timeouts_and_accept() {
        let timeouts = DelegatedHttpTimeouts::from_secs(97, 98, 99);
        let key = (timeouts, Some("application/json".to_string()));

        let _ = timeouts.agent(Some("application/json"));
        assert!(
            delegated_http_agents()
                .lock()
                .expect("agent cache should lock")
                .contains_key(&key)
        );

        let _ = timeouts.agent(Some("application/json"));
        assert!(
            delegated_http_agents()
                .lock()
                .expect("agent cache should lock")
                .contains_key(&key)
        );
    }

    #[test]
    fn delegated_http_agents_partition_json_and_streaming_connections() {
        let timeouts = DelegatedHttpTimeouts::from_secs(94, 95, 96);
        let json_key = (timeouts, Some("application/json".to_string()));
        let stream_key = (timeouts, Some("text/event-stream".to_string()));

        let _ = timeouts.agent(Some("application/json"));
        let _ = timeouts.agent(Some("text/event-stream"));

        let agents = delegated_http_agents()
            .lock()
            .expect("agent cache should lock");
        assert!(agents.contains_key(&json_key));
        assert!(agents.contains_key(&stream_key));
    }

    #[test]
    fn send_json_post_with_retry_retries_connection_refused() {
        // A connection-refused error happens before any request byte is
        // written, so re-sending cannot duplicate a generation: the fake
        // transport below replays a real refused error through the real
        // retry loop and must be called exactly twice.
        let refused = connection_refused_error();
        assert_eq!(refused.kind(), ureq::ErrorKind::ConnectionFailed);

        let attempts = Arc::new(AtomicUsize::new(0));
        let mut first_error = Some(Box::new(refused));

        let response = {
            let attempts = Arc::clone(&attempts);
            post_with_retry("http://127.0.0.1:9/v1/completions", move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                match first_error.take() {
                    Some(error) => Err(error),
                    None => Ok(ureq::Response::new(200, "OK", r#"{"ok":true}"#)
                        .expect("fake response should build")),
                }
            })
        }
        .expect("refused connection should be retried once and then succeed");

        assert_eq!(response.status(), 200);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn post_with_retry_does_not_retry_io_error_after_body_sent() {
        // Read timeouts (and every other post-connect failure) surface as
        // ErrorKind::Io after the request body was already written, so they
        // must fail the call instead of re-sending the generation request.
        let io_error = read_timeout_error_after_body_sent();
        assert_eq!(io_error.kind(), ureq::ErrorKind::Io);
        assert!(!is_retryable_transport_error(&io_error));

        let attempts = Arc::new(AtomicUsize::new(0));

        let error = {
            let attempts = Arc::clone(&attempts);
            post_with_retry("http://127.0.0.1:9/v1/completions", move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(Box::new(io_timeout_transport_error()))
            })
        }
        .expect_err("an Io error must fail the request");

        assert!(
            matches!(&error, DelegatedHttpPostError::Request(source) if source.kind() == ureq::ErrorKind::Io),
            "expected an Io transport error, got {error:?}"
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn send_json_post_with_retry_does_not_re_send_after_body_reached_server() {
        // End-to-end variant of the double-generation bug: the scripted
        // server accepts, reads the complete request (the generation was
        // received), then stalls past the client's read timeout. The client
        // must surface the timeout and must NOT open a second connection.
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener should bind");
        let endpoint = format!(
            "http://{}/v1/completions",
            listener.local_addr().expect("listener should have address")
        );
        let connections = Arc::new(AtomicUsize::new(0));
        let server_connections = Arc::clone(&connections);

        let handle = thread::spawn(move || {
            listener
                .set_nonblocking(true)
                .expect("listener should go nonblocking");
            let deadline = Instant::now() + Duration::from_millis(900);
            while Instant::now() < deadline {
                match listener.accept() {
                    Ok((stream, _)) => {
                        // Accepted sockets can inherit the listener's
                        // nonblocking mode on some platforms; the request
                        // read below needs blocking reads.
                        stream
                            .set_nonblocking(false)
                            .expect("accepted stream should block");
                        server_connections.fetch_add(1, Ordering::SeqCst);
                        read_request_until_body(&stream);
                        // Hold the connection open without responding; the
                        // client read timeout fires while we "generate".
                        thread::sleep(Duration::from_millis(600));
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(10));
                    }
                    Err(error) => panic!("accept should not fail: {error}"),
                }
            }
        });

        let error = send_json_post_with_retry(
            &endpoint,
            &json!({"prompt": "hello"}),
            DelegatedHttpTimeouts {
                read: Duration::from_millis(150),
                ..DelegatedHttpTimeouts::from_secs(2, 1, 2)
            },
            None,
        )
        .expect_err("a stalled blocking generation must surface the read timeout");

        assert!(
            matches!(&error, DelegatedHttpPostError::Request(source) if source.kind() == ureq::ErrorKind::Io),
            "expected an Io transport error, got {error:?}"
        );
        handle.join().expect("server thread should finish");
        assert_eq!(
            connections.load(Ordering::SeqCst),
            1,
            "a read timeout after the body reached the server must not be retried"
        );
    }

    #[test]
    fn blocking_generation_timeouts_raise_the_read_deadline() {
        let default_blocking = DelegatedHttpTimeouts::default().for_blocking_generation();
        assert_eq!(
            default_blocking.read,
            Duration::from_secs(DEFAULT_DELEGATED_HTTP_BLOCKING_READ_TIMEOUT_SECS)
        );
        assert_eq!(
            default_blocking.connect,
            Duration::from_secs(DEFAULT_DELEGATED_HTTP_CONNECT_TIMEOUT_SECS)
        );
        assert_eq!(
            default_blocking.write,
            Duration::from_secs(DEFAULT_DELEGATED_HTTP_IO_TIMEOUT_SECS)
        );

        // A configured read timeout above the floor is preserved; a shorter
        // one (streaming-scale) is raised so long generations are not cut
        // off mid-generation.
        let long = DelegatedHttpTimeouts::from_secs(30, 7200, 300).for_blocking_generation();
        assert_eq!(long.read, Duration::from_secs(7200));
        let short = DelegatedHttpTimeouts::from_secs(30, 5, 300).for_blocking_generation();
        assert_eq!(
            short.read,
            Duration::from_secs(DEFAULT_DELEGATED_HTTP_BLOCKING_READ_TIMEOUT_SECS)
        );
    }

    /// Capture a real connection-refused error by sending to a loopback port
    /// that has nothing listening. ureq classifies this (and every other
    /// connect-phase failure) as `ErrorKind::ConnectionFailed`.
    fn connection_refused_error() -> ureq::Error {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        drop(listener);

        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            .build()
            .post(format!("http://{address}/v1/completions").as_str())
            .send_bytes(br#"{"prompt":"hello"}"#)
            .expect_err("connecting to a closed loopback port must fail")
    }

    /// Capture a real `ErrorKind::Io` error raised after the full request
    /// body was sent: the scripted server reads the request, then stalls
    /// until the client's read deadline fires mid-"generation".
    fn read_timeout_error_after_body_sent() -> ureq::Error {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener should bind");
        let address = listener.local_addr().expect("listener should have address");
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("request should arrive");
            read_request_until_body(&stream);
            thread::sleep(Duration::from_millis(700));
            drop(stream);
        });

        let error = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(2))
            .timeout_read(Duration::from_millis(200))
            .build()
            .post(format!("http://{address}/v1/completions").as_str())
            .send_bytes(br#"{"prompt":"hello"}"#)
            .expect_err("a stalled response must hit the read timeout");
        handle.join().expect("server thread should finish");
        error
    }

    /// Build the `Io` transport error shape used by the fake-transport test
    /// without paying for a second scripted stall: same kind as a read
    /// timeout, verified against a real one in the test above.
    fn io_timeout_transport_error() -> ureq::Error {
        let io_error =
            std::io::Error::new(std::io::ErrorKind::WouldBlock, "timed out reading response");
        // ureq wraps every post-connect io failure (including read timeouts)
        // as an ErrorKind::Io transport error; `From<io::Error>` reproduces
        // exactly that classification.
        io_error.into()
    }

    /// Read a full HTTP request (headers plus the fixed JSON body used by
    /// these tests) so the caller knows the body reached the server.
    fn read_request_until_body(mut stream: &std::net::TcpStream) {
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("server read timeout should apply");
        let mut request = Vec::new();
        let mut buffer = [0_u8; 256];
        loop {
            let read = stream
                .read(&mut buffer)
                .expect("request bytes should read until the body arrives");
            assert!(read > 0, "client closed before sending the full body");
            request.extend_from_slice(&buffer[..read]);
            if String::from_utf8_lossy(&request).contains(r#"{"prompt":"hello"}"#) {
                return;
            }
        }
    }
}
