use crate::contracts::{
    ContractError, DoctorReport, HealthReport, ModelsReport, RuntimeInfoReport, parse_doctor_json,
    parse_health_json, parse_models_json, parse_runtime_info_json,
};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JobError {
    #[error("{0}")]
    Contract(#[from] ContractError),
    #[error("doctor command failed with status {status}: {stderr}")]
    DoctorFailed { status: String, stderr: String },
    #[error("failed to run doctor command: {0}")]
    DoctorIo(#[from] std::io::Error),
    #[error("server URL must be local http://host:port, got {0}")]
    InvalidServerUrl(String),
    #[error("server request failed for {url}: {source}")]
    ServerRequest { url: String, source: std::io::Error },
    #[error("server response for {url} was malformed: {message}")]
    ServerResponse { url: String, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DoctorCommand {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
}

impl DoctorCommand {
    pub fn from_cwd(cwd: &Path, model_dir: Option<&Path>) -> Self {
        if let Some(root) = find_source_checkout_root(cwd) {
            let mut args = vec![
                "run".to_string(),
                "--quiet".to_string(),
                "-p".to_string(),
                "ax-engine-bench".to_string(),
                "--".to_string(),
                "doctor".to_string(),
                "--json".to_string(),
            ];
            if let Some(model_dir) = model_dir {
                args.push("--mlx-model-artifacts-dir".to_string());
                args.push(model_dir.display().to_string());
            }
            Self {
                program: "cargo".to_string(),
                args,
                cwd: Some(root),
            }
        } else {
            let mut args = vec!["doctor".to_string(), "--json".to_string()];
            if let Some(model_dir) = model_dir {
                args.push("--mlx-model-artifacts-dir".to_string());
                args.push(model_dir.display().to_string());
            }
            Self {
                program: "ax-engine-bench".to_string(),
                args,
                cwd: None,
            }
        }
    }
}

pub fn run_doctor(command: &DoctorCommand) -> Result<DoctorReport, JobError> {
    let mut process = Command::new(&command.program);
    process.args(&command.args);
    if let Some(cwd) = command.cwd.as_deref() {
        process.current_dir(cwd);
    }
    let output = process.output()?;
    if !output.status.success() {
        return Err(JobError::DoctorFailed {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_doctor_json(&stdout)?)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServerSnapshot {
    pub health: HealthReport,
    pub runtime: RuntimeInfoReport,
    pub models: ModelsReport,
}

pub fn fetch_server_snapshot(base_url: &str) -> Result<ServerSnapshot, JobError> {
    let base_url = base_url.trim_end_matches('/');
    Ok(ServerSnapshot {
        health: fetch_contract(base_url, "/health", parse_health_json)?,
        runtime: fetch_contract(base_url, "/v1/runtime", parse_runtime_info_json)?,
        models: fetch_contract(base_url, "/v1/models", parse_models_json)?,
    })
}

fn fetch_contract<T>(
    base_url: &str,
    path: &str,
    parse: fn(&str) -> Result<T, ContractError>,
) -> Result<T, JobError> {
    let mut last_error = None;
    for _ in 0..3 {
        let result = fetch_local_http_text(base_url, path)
            .and_then(|text| parse(&text).map_err(JobError::from));
        match result {
            Ok(value) => return Ok(value),
            Err(error @ (JobError::ServerRequest { .. } | JobError::ServerResponse { .. })) => {
                last_error = Some(error);
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.expect("retryable server polling error should be captured"))
}

fn fetch_local_http_text(base_url: &str, path: &str) -> Result<String, JobError> {
    let endpoint = parse_local_http_base_url(base_url)?;
    let url = format!("{base_url}{path}");
    let mut stream =
        TcpStream::connect(&endpoint.address).map_err(|source| JobError::ServerRequest {
            url: url.clone(),
            source,
        })?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(3)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(3)));
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
        endpoint.host_header
    )
    .map_err(|source| JobError::ServerRequest {
        url: url.clone(),
        source,
    })?;
    let mut response = String::new();
    match stream.read_to_string(&mut response) {
        Ok(_) => {}
        Err(source) if response.is_empty() => {
            return Err(JobError::ServerRequest {
                url: url.clone(),
                source,
            });
        }
        Err(_) => {}
    }
    let Some((head, body)) = response.split_once("\r\n\r\n") else {
        return Err(JobError::ServerResponse {
            url,
            message: "missing HTTP header separator".to_string(),
        });
    };
    let status_ok = head
        .lines()
        .next()
        .is_some_and(|status| status.contains(" 200 "));
    if !status_ok {
        return Err(JobError::ServerResponse {
            url,
            message: head.lines().next().unwrap_or("unknown status").to_string(),
        });
    }
    Ok(body.to_string())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct LocalHttpEndpoint {
    address: String,
    host_header: String,
}

fn parse_local_http_base_url(base_url: &str) -> Result<LocalHttpEndpoint, JobError> {
    let Some(rest) = base_url.strip_prefix("http://") else {
        return Err(JobError::InvalidServerUrl(base_url.to_string()));
    };
    let authority = rest.split('/').next().unwrap_or(rest);
    let Some(host) = parse_authority_host(authority) else {
        return Err(JobError::InvalidServerUrl(base_url.to_string()));
    };
    if !is_loopback_host(host) {
        return Err(JobError::InvalidServerUrl(base_url.to_string()));
    }
    Ok(LocalHttpEndpoint {
        address: authority.to_string(),
        host_header: authority.to_string(),
    })
}

fn parse_authority_host(authority: &str) -> Option<&str> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after_host) = rest.split_once(']')?;
        return after_host.strip_prefix(':').map(|_| host);
    }
    let (host, port) = authority.rsplit_once(':')?;
    if host.is_empty() || port.is_empty() {
        return None;
    }
    Some(host)
}

fn is_loopback_host(host: &str) -> bool {
    host == "localhost" || host == "127.0.0.1" || host == "::1"
}

fn find_source_checkout_root(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .find(|candidate| is_source_checkout_root(candidate))
        .map(Path::to_path_buf)
}

fn is_source_checkout_root(path: &Path) -> bool {
    path.join("Cargo.toml").is_file()
        && path.join("scripts/download_model.py").is_file()
        && path.join("crates/ax-engine-bench/Cargo.toml").is_file()
        && path.join("crates/ax-engine-server/Cargo.toml").is_file()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;
    use tempfile::tempdir;

    #[test]
    fn doctor_command_uses_source_checkout_when_available() {
        let root = tempdir().expect("tempdir should create");
        let nested = root.path().join("crates/ax-engine-tui");
        std::fs::create_dir_all(root.path().join("scripts")).expect("scripts dir should create");
        std::fs::create_dir_all(root.path().join("crates/ax-engine-bench"))
            .expect("bench dir should create");
        std::fs::create_dir_all(root.path().join("crates/ax-engine-server"))
            .expect("server dir should create");
        std::fs::create_dir_all(&nested).expect("nested dir should create");
        std::fs::write(root.path().join("Cargo.toml"), "[workspace]\n")
            .expect("workspace toml should write");
        std::fs::write(root.path().join("scripts/download_model.py"), "")
            .expect("download script should write");
        std::fs::write(root.path().join("crates/ax-engine-bench/Cargo.toml"), "")
            .expect("bench toml should write");
        std::fs::write(root.path().join("crates/ax-engine-server/Cargo.toml"), "")
            .expect("server toml should write");

        let command = DoctorCommand::from_cwd(&nested, Some(Path::new("/tmp/model")));

        assert_eq!(command.program, "cargo");
        assert_eq!(command.cwd, Some(root.path().to_path_buf()));
        assert!(command.args.contains(&"doctor".to_string()));
        assert!(
            command
                .args
                .contains(&"--mlx-model-artifacts-dir".to_string())
        );
    }

    #[test]
    fn fetch_server_snapshot_reads_local_metadata_endpoints() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("listener should bind");
        let addr = listener.local_addr().expect("listener should have addr");
        let handle = thread::spawn(move || {
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().expect("connection should accept");
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .expect("read timeout should set");
                let mut request = Vec::new();
                loop {
                    let mut chunk = [0_u8; 256];
                    match stream.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(size) => {
                            request.extend_from_slice(&chunk[..size]);
                            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(error)
                            if matches!(
                                error.kind(),
                                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                            ) =>
                        {
                            break;
                        }
                        Err(error) => panic!("request should read: {error}"),
                    }
                }
                let request = String::from_utf8_lossy(&request);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");
                let body = match path {
                    "/health" => {
                        r#"{"status":"ok","service":"ax-engine-server","model_id":"qwen3_dense","runtime":{"selected_backend":"llama_cpp"}}"#
                    }
                    "/v1/runtime" => {
                        r#"{"service":"ax-engine-server","model_id":"qwen3_dense","deterministic":false,"max_batch_tokens":2048,"block_size_tokens":16,"runtime":{"selected_backend":"llama_cpp"}}"#
                    }
                    "/v1/models" => {
                        r#"{"object":"list","data":[{"id":"qwen3_dense","object":"model","owned_by":"ax-engine-v4","runtime":{"selected_backend":"llama_cpp"}}]}"#
                    }
                    _ => r#"{"error":"not found"}"#,
                };
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .expect("response should write");
            }
        });

        let snapshot =
            fetch_server_snapshot(&format!("http://{addr}")).expect("server snapshot should fetch");

        assert_eq!(snapshot.health.status, "ok");
        assert_eq!(snapshot.runtime.model_id, "qwen3_dense");
        assert_eq!(snapshot.models.data[0].id, "qwen3_dense");
        handle.join().expect("server thread should finish");
    }

    #[test]
    fn server_url_must_be_loopback_http() {
        assert!(parse_local_http_base_url("http://127.0.0.1:8080").is_ok());
        assert!(parse_local_http_base_url("http://localhost:8080").is_ok());
        assert!(parse_local_http_base_url("http://[::1]:8080").is_ok());
        assert!(parse_local_http_base_url("https://127.0.0.1:8080").is_err());
        assert!(parse_local_http_base_url("http://example.com:8080").is_err());
        assert!(parse_local_http_base_url("http://127.0.0.1").is_err());
    }
}
