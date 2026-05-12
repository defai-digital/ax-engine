use ax_engine_tui::app::{AppState, LoadState, MODEL_CATALOG};
use ax_engine_tui::contracts::{
    DoctorReport, WorkflowCommand, read_benchmark_artifact_json, read_doctor_json, scan_artifacts,
};
use ax_engine_tui::jobs::plan::{
    CommandInvocation, EvidenceClass, JobDisplaySummary, JobKind, JobPlan, JobSpec,
};
use ax_engine_tui::jobs::runner::{RunningJob, run_to_completion};
use ax_engine_tui::jobs::{DoctorCommand, fetch_server_snapshot, run_doctor};
use ax_engine_tui::profiles::{ManagerProfile, default_profile_root, read_profile, write_profile};
use ax_engine_tui::support::write_support_bundle;
use ax_engine_tui::web;
use serde_json::{Value, json};
use std::env;
use std::io::{self, BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
enum ManagerError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Options {
    check: bool,
    phase2_check: bool,
    doctor_json: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    server_url: Option<String>,
    benchmark_json: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    profile_dir: Option<PathBuf>,
    support_bundle: Option<PathBuf>,
    web_host: String,
    web_port: u16,
    no_open: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            check: false,
            phase2_check: false,
            doctor_json: None,
            model_dir: None,
            server_url: None,
            benchmark_json: None,
            artifact_root: None,
            profile_dir: None,
            support_bundle: None,
            web_host: "127.0.0.1".to_string(),
            web_port: 8765,
            no_open: false,
        }
    }
}

fn main() -> Result<(), ManagerError> {
    let options = parse_args(env::args().skip(1))?;
    // Check and bundle modes need the full state synchronously before printing results.
    if options.check || options.phase2_check || options.support_bundle.is_some() {
        let state = build_state(&options);
        if let Some(path) = options.support_bundle.as_deref() {
            let bundle_path = write_support_bundle(path, &state)
                .map_err(|error| ManagerError::Message(error.to_string()))?;
            println!("support_bundle={}", bundle_path.display());
            return Ok(());
        }
        if options.phase2_check {
            run_phase2_check(&options, &state)?;
            return Ok(());
        }
        print_check_summary(&state);
        return Ok(());
    }
    // Web mode: bind and open browser immediately; run slow doctor checks in the background.
    run_web_manager(&options)
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Options, ManagerError> {
    let mut options = Options::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => options.check = true,
            "--phase2-check" => options.phase2_check = true,
            "--doctor-json" => options.doctor_json = Some(next_path(&mut args, "--doctor-json")?),
            "--model-dir" => options.model_dir = Some(next_path(&mut args, "--model-dir")?),
            "--server-url" => options.server_url = Some(next_value(&mut args, "--server-url")?),
            "--benchmark-json" => {
                options.benchmark_json = Some(next_path(&mut args, "--benchmark-json")?)
            }
            "--artifact-root" => {
                options.artifact_root = Some(next_path(&mut args, "--artifact-root")?)
            }
            "--profile-dir" => options.profile_dir = Some(next_path(&mut args, "--profile-dir")?),
            "--support-bundle" => {
                options.support_bundle = Some(next_path(&mut args, "--support-bundle")?)
            }
            "--web-host" => options.web_host = next_value(&mut args, "--web-host")?,
            "--web-port" => {
                let value = next_value(&mut args, "--web-port")?;
                options.web_port = value
                    .parse()
                    .map_err(|_| ManagerError::Message(format!("invalid --web-port: {value}")))?;
            }
            "--no-open" => options.no_open = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => {
                return Err(ManagerError::Message(format!(
                    "unknown ax-engine-manager option: {other}"
                )));
            }
        }
    }
    Ok(options)
}

fn next_path(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<PathBuf, ManagerError> {
    Ok(PathBuf::from(next_value(args, flag)?))
}

fn next_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, ManagerError> {
    args.next()
        .ok_or_else(|| ManagerError::Message(format!("missing value for {flag}")))
}

fn print_help() {
    println!(
        "AX Engine Manager\n\n\
         Usage: ax-engine-manager [--check] [--phase2-check] [--doctor-json <path>] [--model-dir <path>] \\\n           [--server-url <url>] [--benchmark-json <path>] [--artifact-root <path>] [--profile-dir <path>] \\\n           [--support-bundle <dir>] [--web-host <host>] [--web-port <port>] [--no-open]\n\n\
         Interactive mode starts a local web manager at http://127.0.0.1:8765 by default.\n\
         The web manager provides text model family/size selection, guarded downloads, server port controls,\n\
         endpoint URLs, readiness state, and job status in a browser UI.\n\
         Check mode is read-only and does not start downloads, benchmarks, or servers.\n\
         Support bundles write redacted diagnostics without model weights or secrets."
    );
}

fn build_state(options: &Options) -> AppState {
    let mut state = AppState::empty();

    state.doctor = if let Some(path) = options.doctor_json.as_deref() {
        read_doctor_json(path)
            .map(LoadState::Ready)
            .unwrap_or_else(|error| LoadState::unavailable(error.to_string()))
    } else {
        match env::current_dir() {
            Ok(cwd) => {
                let command = DoctorCommand::from_cwd(&cwd, options.model_dir.as_deref());
                run_doctor(&command)
                    .map(LoadState::Ready)
                    .unwrap_or_else(|error| LoadState::unavailable(error.to_string()))
            }
            Err(error) => LoadState::unavailable(format!("failed to resolve cwd: {error}")),
        }
    };

    if let Some(server_url) = options.server_url.as_ref() {
        state.server.base_url = Some(server_url.trim_end_matches('/').to_string());
        match fetch_server_snapshot(server_url) {
            Ok(snapshot) => {
                state.server.health = LoadState::Ready(snapshot.health);
                state.server.runtime = LoadState::Ready(snapshot.runtime);
                state.server.models = LoadState::Ready(snapshot.models);
            }
            Err(error) => {
                let message = error.to_string();
                state.server.health = LoadState::unavailable(message.clone());
                state.server.runtime = LoadState::unavailable(message.clone());
                state.server.models = LoadState::unavailable(message);
            }
        }
    }

    if let Some(path) = options.benchmark_json.as_deref() {
        state.benchmark_summary = read_benchmark_artifact_json(path)
            .map(LoadState::Ready)
            .unwrap_or_else(|error| LoadState::unavailable(error.to_string()));
    }

    if let Some(root) = options.artifact_root.as_deref() {
        state.artifacts_root = Some(root.display().to_string());
        state.artifacts = scan_artifacts(root)
            .map(LoadState::Ready)
            .unwrap_or_else(|error| LoadState::unavailable(error.to_string()));
    }

    state
}

fn print_check_summary(state: &AppState) {
    let mut stdout = io::stdout();
    write_check_summary(&mut stdout, state).expect("stdout should be writable");
}

fn write_check_summary(mut writer: impl io::Write, state: &AppState) -> io::Result<()> {
    writeln!(writer, "ax-engine-manager check")?;
    match &state.doctor {
        LoadState::Ready(report) => {
            writeln!(writer, "doctor=ready status={}", report.status)?;
            writeln!(writer, "workflow={}", report.workflow.mode)?;
            writeln!(writer, "model_artifacts={}", report.model_artifacts.status)?;
        }
        LoadState::Unavailable(message) => {
            writeln!(writer, "doctor=unavailable reason={message}")?;
        }
        LoadState::NotLoaded(message) => writeln!(writer, "doctor=not_loaded reason={message}")?,
    }
    match &state.server.health {
        LoadState::Ready(health) => writeln!(writer, "server=ready status={}", health.status)?,
        LoadState::Unavailable(message) => writeln!(writer, "server=unavailable reason={message}")?,
        LoadState::NotLoaded(message) => writeln!(writer, "server=not_loaded reason={message}")?,
    }
    match &state.benchmark_summary {
        LoadState::Ready(summary) => {
            writeln!(writer, "benchmark=ready status={}", summary.status)?;
            writeln!(writer, "benchmark_result_dir={}", summary.result_dir)?;
        }
        LoadState::Unavailable(message) => {
            writeln!(writer, "benchmark=unavailable reason={message}")?;
        }
        LoadState::NotLoaded(message) => {
            writeln!(writer, "benchmark=not_loaded reason={message}")?;
        }
    }
    match &state.artifacts {
        LoadState::Ready(entries) => writeln!(writer, "artifacts=ready count={}", entries.len())?,
        LoadState::Unavailable(message) => {
            writeln!(writer, "artifacts=unavailable reason={message}")?;
        }
        LoadState::NotLoaded(message) => {
            writeln!(writer, "artifacts=not_loaded reason={message}")?;
        }
    }
    Ok(())
}

fn run_phase2_check(options: &Options, state: &AppState) -> Result<(), ManagerError> {
    let LoadState::Ready(doctor) = &state.doctor else {
        return Err(ManagerError::Message(
            "phase2 check requires a readable doctor contract".to_string(),
        ));
    };
    let plan = JobPlan::from_doctor(doctor).map_err(|error| {
        ManagerError::Message(format!("failed to build phase2 job plan: {error}"))
    })?;

    let profile_root = options
        .profile_dir
        .clone()
        .map(Ok)
        .unwrap_or_else(default_profile_root)
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    let mut profile = ManagerProfile::new("phase2-check");
    profile.model_dir = doctor.model_artifacts.path.clone();
    profile.server_url = state.server.base_url.clone();
    profile.artifact_root = state.artifacts_root.clone();
    let profile_path = write_profile(&profile_root, &profile)
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    read_profile(&profile_path).map_err(|error| ManagerError::Message(error.to_string()))?;

    let completed = run_to_completion(fake_job("phase2-fake-smoke", "printf 'smoke-ok\\n'"))
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    let mut fake_server = RunningJob::start(fake_sleep_job("phase2-fake-server", "30"))
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    let startup_observed = fake_server
        .wait_for_startup(Duration::from_millis(50))
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    let canceled = fake_server
        .cancel()
        .map_err(|error| ManagerError::Message(error.to_string()))?;
    let benchmark_guard = plan
        .by_kind(JobKind::BenchmarkScenario)
        .and_then(|job| JobDisplaySummary::from_completed_job(job, "succeeded").err())
        .is_some();

    println!("ax-engine-manager phase2-check");
    println!("jobs={}", plan.jobs.len());
    for job in &plan.jobs {
        println!(
            "job={} kind={} evidence={} owns_process={}",
            job.id,
            job.kind.as_str(),
            job.evidence_class.as_str(),
            job.owns_process
        );
    }
    println!("profile=ready path={}", profile_path.display());
    println!(
        "fake_job={} log_tail={}",
        completed.status.as_str(),
        completed.log_tail.len()
    );
    println!(
        "fake_server={} startup_observed={startup_observed}",
        canceled.status.as_str()
    );
    println!("benchmark_display_guard={benchmark_guard}");
    Ok(())
}

fn fake_job(id: &str, script: &str) -> JobSpec {
    JobSpec::new(
        id,
        id,
        JobKind::ServerSmoke,
        EvidenceClass::RouteContract,
        CommandInvocation::new("sh", vec!["-c".to_string(), script.to_string()], None),
    )
}

fn fake_sleep_job(id: &str, seconds: &str) -> JobSpec {
    JobSpec::new(
        id,
        id,
        JobKind::ServerLaunch,
        EvidenceClass::RouteContract,
        CommandInvocation::new("sleep", vec![seconds.to_string()], None),
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DownloadJobSnapshot {
    id: String,
    repo_id: String,
    status: String,
    model_dir: Option<String>,
    message: Option<String>,
    progress: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CachedModelSnapshot {
    repo_id: String,
    path: String,
}

struct ManagedServer {
    child: Child,
    port: u16,
    repo_id: String,
    model_id: String,
    model_dir: String,
    engine: ManagerEngine,
    ready: bool,
    stderr_file: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ManagerEngine {
    AxEngine,
    AxEngineNgram,
}

impl ManagerEngine {
    fn parse(value: Option<&str>) -> Result<Self, ManagerError> {
        match value.unwrap_or("ax-engine-ngram") {
            "ax-engine" => Ok(Self::AxEngine),
            "ax-engine-ngram" => Ok(Self::AxEngineNgram),
            "mlx-lm" => Err(ManagerError::Message(
                "manager engine mlx-lm is not startable yet; start mlx_lm.server separately and use ax-engine-server delegated mode".to_string(),
            )),
            "mlx-swift" => Err(ManagerError::Message(
                "manager engine mlx-swift is not startable yet; use the benchmark adapter outside the manager for now".to_string(),
            )),
            other => Err(ManagerError::Message(format!(
                "unknown manager engine: {other}"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::AxEngine => "ax-engine",
            Self::AxEngineNgram => "ax-engine-ngram",
        }
    }
}

struct WebRuntime {
    state: AppState,
    next_job_id: u64,
    download_jobs: Vec<DownloadJobSnapshot>,
    hf_cache_models: Option<Vec<CachedModelSnapshot>>,
    server: Option<ManagedServer>,
    status_message: String,
}

impl WebRuntime {
    fn new(state: AppState) -> Self {
        Self {
            state,
            next_job_id: 1,
            download_jobs: Vec::new(),
            hf_cache_models: None,
            server: None,
            status_message: "Web manager ready".to_string(),
        }
    }

    fn cleanup_server(&mut self) {
        let Some(server) = self.server.as_mut() else {
            return;
        };
        if let Ok(Some(status)) = server.child.try_wait() {
            // Try to read the last few lines of stderr for a user-facing hint.
            let hint = server
                .stderr_file
                .as_deref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .map(|s| {
                    s.lines()
                        .rfind(|l| !l.trim().is_empty())
                        .unwrap_or("")
                        .trim()
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .map(|s| format!(": {s}"))
                .unwrap_or_default();
            self.status_message = format!("Server exited ({status}){hint}");
            self.server = None;
        }
    }

    fn server_port(&self) -> u16 {
        self.server
            .as_ref()
            .map(|server| server.port)
            .unwrap_or(self.state.server_control.port)
    }

    fn server_model_dir(&self) -> Option<String> {
        self.server.as_ref().map(|s| s.model_dir.clone())
    }

    fn server_model_id(&self) -> Option<String> {
        self.server.as_ref().map(|s| s.model_id.clone())
    }
}

type SharedRuntime = Arc<Mutex<WebRuntime>>;

fn run_web_manager(options: &Options) -> Result<(), ManagerError> {
    let address = format!("{}:{}", options.web_host, options.web_port);
    let listener = TcpListener::bind(&address)?;

    // Start with empty state so the UI is reachable immediately.
    let mut web_runtime = WebRuntime::new(AppState::empty());
    web_runtime.status_message = "Checking environment…".to_string();
    let runtime = Arc::new(Mutex::new(web_runtime));

    // Run the slow doctor + option-driven state setup in the background.
    {
        let rt = Arc::clone(&runtime);
        let opts = options.clone();
        std::thread::spawn(move || {
            let state = build_state(&opts);
            if let Ok(mut guard) = rt.lock() {
                guard.state = state;
                if guard.status_message == "Checking environment…" {
                    guard.status_message = "Ready.".to_string();
                }
            }
        });
    }

    let url = format!("http://{address}");
    println!("ax-engine-manager web={url}");
    println!("Press Ctrl+C to stop.");
    if !options.no_open {
        open_browser(&url);
    }
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let runtime = Arc::clone(&runtime);
                std::thread::spawn(move || {
                    if let Err(error) = handle_client(stream, runtime) {
                        // Suppress read-timeout errors from keep-alive / speculative
                        // connections that the browser opens but never sends data on.
                        if let ManagerError::Io(ref e) = error
                            && matches!(
                                e.kind(),
                                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                            )
                        {
                            return;
                        }
                        eprintln!("manager request failed: {error}");
                    }
                });
            }
            Err(error) => eprintln!("manager connection failed: {error}"),
        }
    }
    Ok(())
}

fn open_browser(url: &str) {
    let _ = Command::new("open")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
}

fn handle_client(mut stream: TcpStream, runtime: SharedRuntime) -> Result<(), ManagerError> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let request = read_http_request(&mut stream)?;

    // Chat proxy: writes directly to stream (streaming SSE response).
    if request.method == "POST" && request.path == "/api/proxy/chat" {
        return proxy_chat(&mut stream, &runtime, &request.body);
    }

    let response = route_request(request, runtime).unwrap_or_else(|error| {
        json_response("400 Bad Request", &json!({"error": error.to_string()}))
            .unwrap_or_else(|_| http_response("500 Internal Server Error", "text/plain", "error"))
    });
    stream.write_all(response.as_bytes())?;
    Ok(())
}

/// Forward a chat request to the inference server.
///
/// Uses `/v1/generate/stream` (native AX endpoint that works with all
/// backends including native MLX), converting OpenAI-format messages to
/// a ChatML prompt and re-emitting native SSE as OpenAI-compatible chunks.
fn proxy_chat(
    client: &mut TcpStream,
    runtime: &SharedRuntime,
    body: &str,
) -> Result<(), ManagerError> {
    let (port, model_dir, managed_model_id) = {
        let mut rt = runtime
            .lock()
            .map_err(|_| ManagerError::Message("runtime lock poisoned".to_string()))?;
        rt.cleanup_server();
        let Some(server) = rt.server.as_ref() else {
            return write_json_error(
                client,
                "503 Service Unavailable",
                "server is not running. Start the selected model first.",
            );
        };
        if !server.ready {
            return write_json_error(
                client,
                "503 Service Unavailable",
                "server is still starting. Try again after the status changes to online.",
            );
        }
        (server.port, rt.server_model_dir(), rt.server_model_id())
    };

    // Prefer the manager's pre-computed model_id (set at server-spawn time)
    // to avoid a 5-second /v1/models network round-trip on every chat request.
    // Only fall back to querying the server for external servers where no
    // managed_model_id is available.
    let model_id = managed_model_id
        .or_else(|| fetch_server_model_id(port))
        .unwrap_or_else(|| "default".to_string());

    // Convert OpenAI messages payload to a native GenerateRequest.
    let native_body = match openai_to_generate_request(body, &model_id, model_dir.as_deref()) {
        Ok(b) => b,
        Err(e) => {
            return write_json_error(client, "400 Bad Request", &format!("bad request: {e}"));
        }
    };

    // Connect to inference server.
    let mut upstream = match std::net::TcpStream::connect(format!("127.0.0.1:{port}")) {
        Ok(s) => s,
        Err(e) => {
            if let Ok(mut rt) = runtime.lock() {
                rt.cleanup_server();
            }
            return write_json_error(
                client,
                "503 Service Unavailable",
                &format!("server is not ready on port {port}: {e}"),
            );
        }
    };
    upstream.set_read_timeout(Some(Duration::from_secs(120)))?;

    // Native MLX streaming never populates delta_text in step events — tokens
    // are generated but not decoded server-side.  Use the blocking endpoint
    // when we own the model dir (always true for managed servers) so we can
    // decode output_tokens ourselves via the Python tokenizer.
    if let Some(ref dir) = model_dir {
        return proxy_chat_blocking(client, &mut upstream, port, &native_body, &model_id, dir);
    }

    // Fallback for external/unknown servers: forward to the native SSE endpoint.
    proxy_chat_streaming(client, &mut upstream, port, &native_body)
}

/// POST to /v1/generate (blocking), decode output_tokens with Python, return as SSE.
/// Used for native MLX servers where streaming never emits delta_text.
fn proxy_chat_blocking(
    client: &mut TcpStream,
    upstream: &mut TcpStream,
    port: u16,
    native_body: &str,
    model_id: &str,
    model_dir: &str,
) -> Result<(), ManagerError> {
    let forward = format!(
        "POST /v1/generate HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        native_body.len(),
        native_body
    );
    // Time from when the server starts working to when we have the full body.
    // Measured before write so network latency to localhost is included but negligible.
    let server_start = std::time::Instant::now();
    upstream.write_all(forward.as_bytes())?;

    // Read the full response (headers + body).
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match upstream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
        if buf.len() > 8 * 1024 * 1024 {
            break;
        }
    }
    // Capture server time before the Python tokenizer subprocess.
    let server_elapsed = server_start.elapsed();

    let header_end = find_header_end(&buf).unwrap_or(0);
    let header_str = String::from_utf8_lossy(&buf[..header_end]);
    let status_ok = header_str
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .map(|s| s.starts_with('2'))
        .unwrap_or(false);

    let body_start = (header_end + 4).min(buf.len());
    let body_bytes = &buf[body_start..];

    if !status_ok {
        let err_text = String::from_utf8_lossy(body_bytes);
        let msg = serde_json::from_str::<serde_json::Value>(&err_text)
            .ok()
            .and_then(|v| {
                let e = v.get("error")?;
                if e.is_string() {
                    e.as_str().map(str::to_string)
                } else {
                    e.get("message")?.as_str().map(str::to_string)
                }
            })
            .unwrap_or_else(|| err_text.trim().to_string());
        return write_json_error(client, "502 Bad Gateway", &msg);
    }

    // Parse response JSON once; extract all needed fields.
    let response_json = serde_json::from_slice::<serde_json::Value>(body_bytes).ok();

    let output_tokens: Vec<u32> = response_json
        .as_ref()
        .and_then(|v| v["output_tokens"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as u32))
        .collect();

    // Token counts for performance stats.  output_token_count / prompt_token_count are
    // populated by from_snapshot() in the SDK; fall back to measured lengths if absent.
    let prompt_token_count = response_json
        .as_ref()
        .and_then(|v| v["prompt_token_count"].as_u64())
        .unwrap_or(0) as u32;
    let output_token_count = response_json
        .as_ref()
        .and_then(|v| v["output_token_count"].as_u64())
        .map(|n| n as u32)
        .unwrap_or(output_tokens.len() as u32);

    // Decode token IDs to text using the model's tokenizer.
    let raw_text = decode_tokens(model_dir, &output_tokens)?;
    let text = strip_generation_artifacts(&raw_text, model_id, model_dir);

    // generation_tps = output tokens / server time (prefill + decode combined).
    // For typical chat responses decode dominates, so this approximates decode throughput.
    let generation_tps = if output_token_count > 0 && server_elapsed.as_millis() > 0 {
        output_token_count as f64 / server_elapsed.as_secs_f64()
    } else {
        0.0
    };

    // Write OpenAI-format SSE to the JS client.
    client.write_all(
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n",
    )?;
    if !text.is_empty() {
        let delta = format!(
            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":null,\"index\":0}}]}}\n\n",
            serde_json::to_string(&text).unwrap_or_else(|_| "\"\"".to_string())
        );
        let _ = client.write_all(delta.as_bytes());
    }
    // Final chunk: finish_reason + usage stats so the client can display perf metrics.
    let finish_reason = response_json
        .as_ref()
        .and_then(|v| v["finish_reason"].as_str())
        .map(openai_finish_reason)
        .unwrap_or("stop");
    let finish_chunk = format!(
        "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":{},\"index\":0}}],\
         \"usage\":{{\"prompt_tokens\":{prompt_token_count},\
         \"completion_tokens\":{output_token_count},\
         \"generation_tps\":{:.1}}}}}\n\ndata: [DONE]\n\n",
        serde_json::to_string(finish_reason).unwrap_or_else(|_| "\"stop\"".to_string()),
        generation_tps
    );
    let _ = client.write_all(finish_chunk.as_bytes());
    client.flush().ok();
    Ok(())
}

/// Fallback streaming path for external/OpenAI-compatible servers.
fn proxy_chat_streaming(
    client: &mut TcpStream,
    upstream: &mut TcpStream,
    port: u16,
    native_body: &str,
) -> Result<(), ManagerError> {
    let forward = format!(
        "POST /v1/generate/stream HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        native_body.len(),
        native_body
    );
    upstream.write_all(forward.as_bytes())?;

    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = upstream.read(&mut tmp)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if find_header_end(&buf).is_some() {
            break;
        }
        if buf.len() > 64 * 1024 {
            return Err(ManagerError::Message(
                "upstream headers too large".to_string(),
            ));
        }
    }

    let header_end = find_header_end(&buf).unwrap_or(buf.len());
    let header_str = String::from_utf8_lossy(&buf[..header_end]);
    let is_sse = header_str
        .to_ascii_lowercase()
        .contains("text/event-stream");
    let status_ok = header_str
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .map(|s| s.starts_with('2'))
        .unwrap_or(false);

    if !is_sse || !status_ok {
        let body_start = (header_end + 4).min(buf.len());
        let mut err_bytes = buf[body_start..].to_vec();
        loop {
            match upstream.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => err_bytes.extend_from_slice(&tmp[..n]),
            }
            if err_bytes.len() > 64 * 1024 {
                break;
            }
        }
        let err_text = String::from_utf8_lossy(&err_bytes);
        let msg = serde_json::from_str::<serde_json::Value>(&err_text)
            .ok()
            .and_then(|v| {
                let e = v.get("error")?;
                if e.is_string() {
                    e.as_str().map(str::to_string)
                } else {
                    e.get("message")?.as_str().map(str::to_string)
                }
            })
            .unwrap_or_else(|| err_text.trim().to_string());
        let out = format!(
            r#"{{"error":{}}}"#,
            serde_json::to_string(&msg).unwrap_or_else(|_| "\"\"".to_string())
        );
        let resp = format!(
            "HTTP/1.1 502 Bad Gateway\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            out.len(),
            out
        );
        client.write_all(resp.as_bytes())?;
        return Ok(());
    }

    client.write_all(
        b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n",
    )?;
    let body_start = (header_end + 4).min(buf.len());
    stream_native_as_openai(client, upstream, &buf[body_start..])?;
    Ok(())
}

fn write_json_error(
    client: &mut TcpStream,
    status: &str,
    message: &str,
) -> Result<(), ManagerError> {
    let body = serde_json::to_string(&json!({ "error": message }))?;
    let resp = format!(
        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    client.write_all(resp.as_bytes())?;
    Ok(())
}

/// Query `/v1/models` and return the first model id the server reports.
fn fetch_server_model_id(port: u16) -> Option<String> {
    use std::io::Write;
    let mut conn = std::net::TcpStream::connect(format!("127.0.0.1:{port}")).ok()?;
    conn.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    let req =
        format!("GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    conn.write_all(req.as_bytes()).ok()?;

    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match conn.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
        }
        if buf.len() > 32 * 1024 {
            break;
        }
    }
    let header_end = find_header_end(&buf)?;
    let body = String::from_utf8_lossy(&buf[header_end + 4..]);
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    v["data"]
        .as_array()?
        .first()?
        .get("id")?
        .as_str()
        .map(str::to_string)
}

/// Detect the prompt template family from a model ID or snapshot directory name.
fn detect_prompt_family(model_id: &str, model_dir: Option<&str>) -> &'static str {
    let classify = |s: &str| -> &'static str {
        let lo = s.to_lowercase();
        if lo.contains("gemma-4") || lo.contains("gemma4") {
            "gemma4"
        } else if lo.contains("gemma") {
            "gemma"
        } else if lo.contains("glm") {
            "glm"
        } else {
            "chatml"
        }
    };
    let from_id = classify(model_id);
    if from_id != "chatml" {
        return from_id;
    }
    // Walk all path segments from the end — the model name typically appears in a
    // middle segment (e.g. "models--google--gemma-4-e2b-it-5bit") while the last
    // segment is a commit hash and wouldn't match.
    model_dir
        .and_then(|dir| {
            dir.trim_end_matches('/').split('/').rev().find_map(|part| {
                let f = classify(part);
                if f != "chatml" { Some(f) } else { None }
            })
        })
        .unwrap_or("chatml")
}

/// Convert an OpenAI chat-completions request body into a native GenerateRequest JSON string.
///
/// Prefer the real tokenizer chat template, matching mlx-engine/mlx-lm. The
/// built-in prompt renderer is only a fallback when `transformers` is missing.
fn openai_to_generate_request(
    body: &str,
    model_id: &str,
    model_dir: Option<&str>,
) -> Result<String, ManagerError> {
    let v: serde_json::Value = serde_json::from_str(body)?;
    let messages = v["messages"]
        .as_array()
        .ok_or_else(|| ManagerError::Message("messages field required".to_string()))?;

    let max_tokens = v["max_tokens"].as_u64().unwrap_or(128) as u32;
    let temperature = v["temperature"].as_f64().unwrap_or(0.0) as f32;

    // Native MLX backend requires pre-tokenized input_tokens.
    let input_tokens = model_dir
        .map(|dir| tokenize_chat_messages(dir, model_id, messages))
        .transpose()?
        .unwrap_or_default();

    let native = json!({
        "model_id": model_id,
        "input_tokens": input_tokens,
        "max_output_tokens": max_tokens,
        "sampling": {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 0,
            "seed": 0,
            "repetition_penalty": 1.15
        },
        "stop_sequences": manager_chat_stop_sequences(model_id, model_dir)
    });

    Ok(native.to_string())
}

fn manager_chat_stop_sequences(model_id: &str, model_dir: Option<&str>) -> Vec<String> {
    match detect_prompt_family(model_id, model_dir) {
        "glm" => vec![
            "<|endoftext|>".to_string(),
            "<|user|>".to_string(),
            "<|observation|>".to_string(),
        ],
        "gemma4" => vec!["<turn|>".to_string()],
        "gemma" => vec!["<end_of_turn>".to_string()],
        _ => vec!["<|im_end|>".to_string()],
    }
}

fn normalize_chat_messages(messages: &[Value]) -> Result<Vec<Value>, ManagerError> {
    messages
        .iter()
        .map(|message| {
            let raw_role = message
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("user")
                .trim();
            let role = if raw_role == "function" {
                "tool"
            } else {
                raw_role
            };
            let content = render_chat_content(message.get("content").unwrap_or(&Value::Null))?;
            Ok(json!({
                "role": role,
                "content": content,
            }))
        })
        .collect()
}

fn render_chat_content(content: &Value) -> Result<String, ManagerError> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut rendered = String::new();
            for part in parts {
                if part.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(ManagerError::Message(
                        "manager chat proxy currently accepts text-only content parts".to_string(),
                    ));
                }
                rendered.push_str(part.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            Ok(rendered)
        }
        Value::Null => Ok(String::new()),
        _ => Err(ManagerError::Message(
            "manager chat proxy content must be a string or text parts".to_string(),
        )),
    }
}

fn tokenize_chat_messages(
    model_dir: &str,
    model_id: &str,
    messages: &[Value],
) -> Result<Vec<u32>, ManagerError> {
    let normalized = normalize_chat_messages(messages)?;
    let messages_json = serde_json::to_string(&normalized)?;
    let disable_thinking = disables_thinking_chat_template(model_id, model_dir);
    let script = r#"
import json
import sys

model_dir = sys.argv[1]
model_id = sys.argv[2].lower()
messages = json.loads(sys.argv[3])
disable_thinking = sys.argv[4] == "1"

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    kwargs = {}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    if getattr(tok, "chat_template", None) is not None:
        try:
            ids = tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        except Exception:
            # Some model templates reject the system role (e.g. older Gemma variants).
            # Retry without the leading system message so the user turn is still formatted
            # correctly rather than falling all the way back to the plain-text renderer.
            msgs_no_sys = [m for m in messages if m.get("role") != "system"]
            ids = tok.apply_chat_template(
                msgs_no_sys,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
    else:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"
        ids = tok.encode(prompt, add_special_tokens=False)
    print(json.dumps([int(i) for i in ids]))
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(2)
"#
    .to_string();

    run_python_tokenizer_script(vec![
        "-c".to_string(),
        script,
        model_dir.to_string(),
        model_id.to_string(),
        messages_json,
        if disable_thinking { "1" } else { "0" }.to_string(),
    ])
    .or_else(|template_error| {
        let prompt = render_fallback_chat_prompt(model_id, Some(model_dir), &normalized);
        tokenize_prompt(model_dir, &prompt).map_err(|fallback_error| {
            ManagerError::Message(format!(
                "tokenizer chat_template failed ({template_error}); fallback tokenizer failed ({fallback_error})"
            ))
        })
    })
}

fn disables_thinking_chat_template(model_id: &str, model_dir: &str) -> bool {
    let model_id = model_id.to_ascii_lowercase();
    let model_dir = model_dir.to_ascii_lowercase();
    model_id.contains("qwen")
        || model_dir.contains("qwen")
        || model_id.contains("glm")
        || model_dir.contains("glm")
}

/// Tokenize `text` using the model's tokenizer.json via the Python `tokenizers` package.
/// Returns token IDs suitable for the native MLX generate endpoint.
fn tokenize_prompt(model_dir: &str, text: &str) -> Result<Vec<u32>, ManagerError> {
    // The `tokenizers` package is a dependency of mlx-lm so it is always present.
    let script = "import sys, json\n\
        from tokenizers import Tokenizer\n\
        t = Tokenizer.from_file(sys.argv[1] + '/tokenizer.json')\n\
        t.no_truncation()\n\
        enc = t.encode(sys.argv[2])\n\
        print(json.dumps(enc.ids))\n"
        .to_string();
    run_python_tokenizer_script(vec![
        "-c".to_string(),
        script,
        model_dir.to_string(),
        text.to_string(),
    ])
}

/// Remove role-loop artifacts that appear after decoding with skip_special_tokens.
///
/// Native MLX servers may not stop exactly at every EOT boundary; extra turn-prefix
/// tokens then get decoded as plain text ("model\n" for Gemma, role tags for GLM).
/// Runaway repetition from absent repetition-penalty is stripped for all models.
fn strip_generation_artifacts(text: &str, model_id: &str, model_dir: &str) -> String {
    let family = detect_prompt_family(model_id, Some(model_dir));
    let cleaned = match family {
        "gemma4" => strip_gemma_turn_artifacts(text, &["<|turn>", "<turn|>"]),
        "gemma" => strip_gemma_turn_artifacts(text, &["<start_of_turn>", "<end_of_turn>"]),
        "glm" => {
            // Strip any role tags that survived decoding.
            let mut t = text.trim_end();
            for marker in &["<|user|>", "<|assistant|>", "<|system|>", "<|endoftext|>"] {
                if let Some(idx) = t.find(marker) {
                    t = t[..idx].trim_end();
                }
            }
            t.to_string()
        }
        _ => text.trim_end().to_string(),
    };
    strip_runaway_repetition(&cleaned)
}

fn strip_gemma_turn_artifacts(text: &str, tokens: &[&str]) -> String {
    let mut cleaned = text.to_string();
    for token in tokens {
        cleaned = cleaned.replace(token, "");
    }

    let mut t = cleaned.trim_start_matches('\n').trim_start();
    while let Some(rest) = strip_leading_gemma_role(t) {
        t = rest.trim_start();
    }

    truncate_at_gemma_turn_role(t).trim_end().to_string()
}

fn strip_leading_gemma_role(text: &str) -> Option<&str> {
    for role in ["model", "assistant"] {
        if let Some(rest) = text.strip_prefix(role).and_then(|rest| {
            rest.strip_prefix('\n')
                .or_else(|| rest.strip_prefix(": "))
                .or_else(|| rest.strip_prefix(':'))
        }) {
            return Some(rest);
        }
    }
    None
}

fn truncate_at_gemma_turn_role(text: &str) -> &str {
    let mut cutoff = text.len();
    for marker in [
        "\nuser\n",
        "\nmodel\n",
        "\nassistant\n",
        "\nsystem\n",
        "\nuser:",
        "\nmodel:",
        "\nassistant:",
        "\nsystem:",
    ] {
        if let Some(idx) = text.find(marker) {
            cutoff = cutoff.min(idx);
        }
    }
    &text[..cutoff]
}

/// Detect and remove a repetitively-looping suffix (e.g. "5555…" or "9, 9, 9.").
///
/// Tries pattern lengths 1–16 and end-alignments 0..pat_len so that a trailing
/// punctuation character (e.g. the final "." in "9, 9, 9.") doesn't prevent the
/// "9, " pattern from being recognised.  Requires ≥ 10 consecutive repetitions.
fn strip_runaway_repetition(text: &str) -> String {
    let bytes = text.as_bytes();
    let n = bytes.len();
    'outer: for pat_len in 1..=16usize {
        if n < pat_len * 10 {
            continue;
        }
        // Try all end-offsets 0..pat_len so the pattern doesn't have to align
        // perfectly with the very last byte (handles trailing "." etc.).
        for end_offset in 0..pat_len {
            let effective_end = match n.checked_sub(end_offset) {
                Some(e) if e >= pat_len * 2 => e,
                _ => continue,
            };
            if !text.is_char_boundary(effective_end) {
                continue;
            }
            let pat = &bytes[effective_end - pat_len..effective_end];
            let mut count = 0usize;
            let mut pos = effective_end;
            loop {
                if pos < pat_len {
                    break;
                }
                if &bytes[pos - pat_len..pos] == pat {
                    count += 1;
                    pos -= pat_len;
                } else {
                    break;
                }
            }
            if count >= 10 {
                if !text.is_char_boundary(pos) {
                    continue 'outer;
                }
                return text[..pos].trim_end().to_string();
            }
        }
    }
    text.to_string()
}

/// Decode a list of token IDs to a UTF-8 string using the model's tokenizer.
fn decode_tokens(model_dir: &str, tokens: &[u32]) -> Result<String, ManagerError> {
    if tokens.is_empty() {
        return Ok(String::new());
    }
    let ids_json = serde_json::to_string(tokens)?;
    let script = r#"
import json, sys
model_dir = sys.argv[1]
ids = json.loads(sys.argv[2])
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    text = tok.decode(ids, skip_special_tokens=True)
except Exception:
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(model_dir + '/tokenizer.json')
    t.no_truncation()
    text = t.decode(ids, skip_special_tokens=True)
print(text, end='')
"#;
    run_python_text_script(vec![
        "-c".to_string(),
        script.to_string(),
        model_dir.to_string(),
        ids_json,
    ])
}

fn run_python_text_script(args: Vec<String>) -> Result<String, ManagerError> {
    let output =
        run_subprocess_output_with_timeout("python3", args, Duration::from_secs(15), "decode")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ManagerError::Message(format!(
            "decode error: {}",
            stderr.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn run_python_tokenizer_script(args: Vec<String>) -> Result<Vec<u32>, ManagerError> {
    let output =
        run_subprocess_output_with_timeout("python3", args, Duration::from_secs(15), "tokenizer")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ManagerError::Message(format!(
            "tokenizer error: {}",
            stderr.trim()
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let ids: Vec<u64> = serde_json::from_str(stdout.trim())
        .map_err(|e| ManagerError::Message(format!("tokenizer output parse error: {e}")))?;
    Ok(ids.into_iter().map(|id| id as u32).collect())
}

fn run_subprocess_output_with_timeout(
    program: &str,
    args: Vec<String>,
    timeout: Duration,
    label: &str,
) -> Result<Output, ManagerError> {
    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| ManagerError::Message(format!("{label} subprocess failed: {e}")))?;
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    let _ = out.read_to_end(&mut stdout);
                }
                let mut stderr = Vec::new();
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_end(&mut stderr);
                }
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) if started.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ManagerError::Message(format!(
                    "{label} timed out after {}s",
                    timeout.as_secs()
                )));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(25)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(ManagerError::Message(format!(
                    "{label} subprocess failed: {e}"
                )));
            }
        }
    }
}

fn render_fallback_chat_prompt(
    model_id: &str,
    model_dir: Option<&str>,
    messages: &[Value],
) -> String {
    let family = detect_prompt_family(model_id, model_dir);
    let mut prompt = String::new();
    match family {
        "gemma4" => {
            prompt.push_str("<bos>");
            for msg in messages {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");
                let turn = if role == "assistant" { "model" } else { role };
                prompt.push_str(&format!("<|turn>{turn}\n{content}<turn|>\n"));
            }
            prompt.push_str("<|turn>model\n");
        }
        "gemma" => {
            for msg in messages {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");
                let turn = if role == "assistant" { "model" } else { role };
                prompt.push_str(&format!("<start_of_turn>{turn}\n{content}<end_of_turn>\n"));
            }
            prompt.push_str("<start_of_turn>model\n");
        }
        "glm" => {
            prompt.push_str("[gMASK]<sop>");
            for msg in messages {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");
                if matches!(role, "tool" | "function") {
                    prompt.push_str("<|observation|><tool_response>");
                    prompt.push_str(content);
                    prompt.push_str("</tool_response>");
                } else {
                    let tag = match role {
                        "assistant" => "<|assistant|>",
                        "system" => "<|system|>",
                        _ => "<|user|>",
                    };
                    prompt.push_str(tag);
                    if role == "assistant" {
                        prompt.push_str("</think>");
                        prompt.push_str(content.trim());
                    } else {
                        prompt.push_str(content);
                    }
                }
            }
            prompt.push_str("<|assistant|></think>");
        }
        _ => {
            for msg in messages {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
            prompt.push_str("<|im_start|>assistant\n");
            if model_id.to_ascii_lowercase().contains("qwen")
                || model_dir.is_some_and(|dir| dir.to_ascii_lowercase().contains("qwen"))
            {
                prompt.push_str("<think>\n\n</think>\n\n");
            }
        }
    }
    prompt
}

/// Read native AX Engine SSE from `upstream` (and any already-buffered `initial` bytes)
/// and re-emit as OpenAI-compatible SSE to `client`.
///
/// Native format per event:
///   event: step
///   data: {"delta_text":"hello",...}
///
/// OpenAI format emitted:
///   data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null,"index":0}]}
fn stream_native_as_openai<W: Write, R: Read>(
    client: &mut W,
    upstream: &mut R,
    initial: &[u8],
) -> Result<(), ManagerError> {
    let mut raw = Vec::from(initial);
    let mut tmp = [0u8; 4096];
    let mut current_event = String::new();
    let mut emitted_text = String::new();
    let mut done_sent = false;

    loop {
        // Process all complete lines already in the buffer.
        while let Some(nl) = raw.iter().position(|&b| b == b'\n') {
            let line_bytes = raw.drain(..=nl).collect::<Vec<_>>();
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim_end_matches(['\r', '\n']);

            if let Some(ev) = line.strip_prefix("event: ") {
                current_event = ev.to_string();
            } else if let Some(data) = line.strip_prefix("data: ") {
                match current_event.as_str() {
                    "step" => {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
                            && let Some(delta) = v["delta_text"].as_str()
                            && !delta.is_empty()
                        {
                            let chunk = format!(
                                "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":null,\"index\":0}}]}}\n\n",
                                serde_json::to_string(delta).unwrap_or_else(|_| "\"\"".to_string())
                            );
                            if client.write_all(chunk.as_bytes()).is_err() {
                                return Ok(());
                            }
                            emitted_text.push_str(delta);
                            client.flush().ok();
                        }
                    }
                    "response" => {
                        let mut finish_reason = "stop";
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                            if let Some(reason) = v["response"]["finish_reason"].as_str() {
                                finish_reason = openai_finish_reason(reason);
                            }
                            if let Some(output_text) = v["response"]["output_text"].as_str() {
                                let missing = if emitted_text.is_empty() {
                                    output_text
                                } else {
                                    output_text.strip_prefix(&emitted_text).unwrap_or("")
                                };
                                if !missing.is_empty() {
                                    let chunk = format!(
                                        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":null,\"index\":0}}]}}\n\n",
                                        serde_json::to_string(missing)
                                            .unwrap_or_else(|_| "\"\"".to_string())
                                    );
                                    if client.write_all(chunk.as_bytes()).is_err() {
                                        return Ok(());
                                    }
                                    emitted_text.push_str(missing);
                                }
                            }
                        }
                        let final_chunk = format!(
                            "data: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":{},\"index\":0}}]}}\n\n\
                              data: [DONE]\n\n",
                            serde_json::to_string(finish_reason)
                                .unwrap_or_else(|_| "\"stop\"".to_string())
                        );
                        let _ = client.write_all(final_chunk.as_bytes());
                        client.flush().ok();
                        done_sent = true;
                    }
                    "error" => {
                        // Server-side error event — surface the message in the chat.
                        let msg = serde_json::from_str::<serde_json::Value>(data)
                            .ok()
                            .and_then(|v| {
                                v["error"]["message"]
                                    .as_str()
                                    .or_else(|| v["error"].as_str())
                                    .map(str::to_string)
                            })
                            .unwrap_or_else(|| data.to_string());
                        let label = format!("[server error: {msg}]");
                        let chunk = format!(
                            "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}},\"finish_reason\":\"stop\",\"index\":0}}]}}\n\n\
                             data: [DONE]\n\n",
                            serde_json::to_string(&label).unwrap_or_else(|_| "\"\"".to_string())
                        );
                        let _ = client.write_all(chunk.as_bytes());
                        client.flush().ok();
                        done_sent = true;
                    }
                    _ => {}
                }
            }
        }

        if done_sent {
            break;
        }

        // Read more bytes from upstream.
        match upstream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => raw.extend_from_slice(&tmp[..n]),
        }
    }

    if !done_sent {
        let _ = client.write_all(b"data: [DONE]\n\n");
        client.flush().ok();
    }

    Ok(())
}

fn openai_finish_reason(value: &str) -> &'static str {
    match value {
        "length" | "max_output_tokens" => "length",
        "content_filter" => "content_filter",
        _ => "stop",
    }
}

fn route_request(request: HttpRequest, runtime: SharedRuntime) -> Result<String, ManagerError> {
    let path = request.path.split('?').next().unwrap_or("/");
    Ok(match (request.method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => {
            http_response("200 OK", "text/html; charset=utf-8", &web::index_html())
        }
        ("GET", "/assets/manager.css") => {
            http_response("200 OK", "text/css; charset=utf-8", web::manager_css())
        }
        ("GET", "/assets/manager.js") => http_response(
            "200 OK",
            "application/javascript; charset=utf-8",
            web::manager_js(),
        ),
        ("GET", "/api/state") => {
            let value = runtime_state_json(&runtime)?;
            json_response("200 OK", &value)?
        }
        ("GET", "/api/system") => json_response("200 OK", &system_metrics_json())?,
        ("POST", "/api/download") => {
            let value = start_download_job(&runtime, &request.body)?;
            json_response("202 Accepted", &value)?
        }
        ("POST", "/api/server/start") => {
            let value = start_server(&runtime, &request.body)?;
            json_response("200 OK", &value)?
        }
        ("POST", "/api/server/stop") => {
            let value = stop_server(&runtime)?;
            json_response("200 OK", &value)?
        }
        ("POST", "/api/server/restart") => {
            let value = restart_server(&runtime, &request.body)?;
            json_response("200 OK", &value)?
        }
        ("GET", path) if path.starts_with("/api/jobs/") => {
            let id = path.trim_start_matches("/api/jobs/");
            let value = download_job_json(&runtime, id)?;
            json_response("200 OK", &value)?
        }
        _ => json_response("404 Not Found", &json!({"error": "not found"}))?,
    })
}

fn system_metrics_json() -> Value {
    json!({
        "timestamp_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0),
        "refresh_interval_ms": 3000,
        "cpu": metric_json(cpu_utilization_percent(), "ps"),
        "gpu": metric_json(gpu_utilization_percent(), "ioreg"),
        "ram": metric_json(ram_utilization_percent(), "vm_stat"),
    })
}

fn metric_json(percent: Option<f64>, source: &str) -> Value {
    match percent {
        Some(value) => json!({
            "percent": round_percent(value),
            "available": true,
            "source": source,
        }),
        None => json!({
            "percent": Value::Null,
            "available": false,
            "source": source,
        }),
    }
}

fn cpu_utilization_percent() -> Option<f64> {
    let ps = command_stdout("ps", &["-A", "-o", "%cpu="], Duration::from_secs(2))?;
    let logical_cpu = command_stdout("sysctl", &["-n", "hw.logicalcpu"], Duration::from_secs(2))?
        .trim()
        .parse::<f64>()
        .ok()?;
    let total_cpu = ps
        .lines()
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .sum::<f64>();
    if logical_cpu <= 0.0 {
        return None;
    }
    Some(clamp_percent(total_cpu / logical_cpu))
}

fn ram_utilization_percent() -> Option<f64> {
    let total_bytes = command_stdout("sysctl", &["-n", "hw.memsize"], Duration::from_secs(2))?
        .trim()
        .parse::<u64>()
        .ok()?;
    let vm_stat = command_stdout("vm_stat", &[], Duration::from_secs(2))?;
    ram_utilization_percent_from_vm_stat(&vm_stat, total_bytes)
}

fn gpu_utilization_percent() -> Option<f64> {
    const AGX_CLASSES: &[&str] = &[
        "AGXAcceleratorG17X",
        "AGXAcceleratorG16X",
        "AGXAcceleratorG16G",
        "AGXAcceleratorG15X",
        "AGXAcceleratorG15G",
        "AGXAcceleratorG14X",
        "AGXAcceleratorG14S",
        "AGXAcceleratorG14G",
        "AGXAcceleratorG13X",
        "AGXAcceleratorG13G",
    ];
    for class_name in AGX_CLASSES {
        if let Some(output) = command_stdout(
            "ioreg",
            &["-r", "-c", class_name, "-d", "1"],
            Duration::from_secs(2),
        ) && let Some(percent) = gpu_utilization_percent_from_ioreg(&output)
        {
            return Some(percent);
        }
    }
    None
}

fn command_stdout(program: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let output = run_subprocess_output_with_timeout(
        program,
        args.iter().map(|arg| (*arg).to_string()).collect(),
        timeout,
        program,
    )
    .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

fn ram_utilization_percent_from_vm_stat(vm_stat: &str, total_bytes: u64) -> Option<f64> {
    if total_bytes == 0 {
        return None;
    }
    let page_size = parse_page_size(vm_stat)?;
    let active_pages = parse_vm_stat_pages(vm_stat, "Pages active:");
    let wired_pages = parse_vm_stat_pages(vm_stat, "Pages wired down:");
    let compressed_pages = parse_vm_stat_pages(vm_stat, "Pages occupied by compressor:");
    if active_pages.is_some() || wired_pages.is_some() || compressed_pages.is_some() {
        let used_pages = active_pages
            .unwrap_or(0)
            .saturating_add(wired_pages.unwrap_or(0))
            .saturating_add(compressed_pages.unwrap_or(0));
        let used_bytes = used_pages.saturating_mul(page_size);
        return Some(clamp_percent(
            used_bytes as f64 * 100.0 / total_bytes as f64,
        ));
    }

    // Fallback for shortened or older vm_stat output. This includes inactive
    // cache pages, so it is less useful for spotting model process release.
    let free_pages = parse_vm_stat_pages(vm_stat, "Pages free:").unwrap_or(0);
    let speculative_pages = parse_vm_stat_pages(vm_stat, "Pages speculative:").unwrap_or(0);
    let available_bytes = free_pages
        .saturating_add(speculative_pages)
        .saturating_mul(page_size);
    let used_bytes = total_bytes.saturating_sub(available_bytes);
    Some(clamp_percent(
        used_bytes as f64 * 100.0 / total_bytes as f64,
    ))
}

fn gpu_utilization_percent_from_ioreg(output: &str) -> Option<f64> {
    parse_number_after_marker(output, "\"Device Utilization %\"")
        .or_else(|| parse_number_after_marker(output, "\"Renderer Utilization %\""))
        .map(clamp_percent)
}

fn parse_page_size(vm_stat: &str) -> Option<u64> {
    let start = vm_stat.find("(page size of ")? + "(page size of ".len();
    let rest = &vm_stat[start..];
    let end = rest.find(" bytes)")?;
    rest[..end].trim().parse::<u64>().ok()
}

fn parse_vm_stat_pages(vm_stat: &str, label: &str) -> Option<u64> {
    vm_stat.lines().find_map(|line| {
        let value = line
            .trim()
            .strip_prefix(label)?
            .trim()
            .trim_end_matches('.');
        value.parse::<u64>().ok()
    })
}

fn parse_number_after_marker(output: &str, marker: &str) -> Option<f64> {
    let start = output.find(marker)? + marker.len();
    let rest = &output[start..];
    let equals = rest.find('=')? + 1;
    let value = rest[equals..].trim_start();
    let end = value
        .find(|ch: char| !(ch.is_ascii_digit() || ch == '.'))
        .unwrap_or(value.len());
    value[..end].parse::<f64>().ok()
}

fn clamp_percent(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 100.0)
    } else {
        0.0
    }
}

fn round_percent(value: f64) -> f64 {
    (clamp_percent(value) * 10.0).round() / 10.0
}

/// Return the local snapshot path for a HuggingFace model if it is already
/// in the Hub cache used by mlx-lm / huggingface_hub.
///
/// Layout: models--{org}--{repo}/refs/main  →  commit hash
///         models--{org}--{repo}/snapshots/{hash}/
fn hf_cache_path(repo_id: &str) -> Option<String> {
    let dashed = repo_id.replace('/', "--");
    let model_cache = hf_hub_cache_root()?.join(format!("models--{dashed}"));

    // Read the commit hash from refs/main.
    let refs_main = model_cache.join("refs/main");
    let hash = std::fs::read_to_string(&refs_main).ok()?;
    let hash = hash.trim();
    if hash.is_empty() {
        return None;
    }

    let snapshot = model_cache.join("snapshots").join(hash);
    // Require config.json AND at least one .safetensors weight file.
    // A directory with only config files is an incomplete download.
    if snapshot.join("config.json").is_file()
        && std::fs::read_dir(&snapshot)
            .ok()?
            .flatten()
            .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("safetensors"))
    {
        Some(snapshot.display().to_string())
    } else {
        None
    }
}

fn hf_hub_cache_root() -> Option<PathBuf> {
    if let Ok(path) = env::var("HF_HUB_CACHE")
        && !path.trim().is_empty()
    {
        return Some(expand_home_path(&path));
    }
    if let Ok(path) = env::var("HF_HOME")
        && !path.trim().is_empty()
    {
        return Some(expand_home_path(&path).join("hub"));
    }
    if let Ok(path) = env::var("XDG_CACHE_HOME")
        && !path.trim().is_empty()
    {
        return Some(expand_home_path(&path).join("huggingface/hub"));
    }
    let home = env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".cache/huggingface/hub"))
}

fn expand_home_path(path: &str) -> PathBuf {
    if path == "~" {
        return env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(path));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return env::var("HOME")
            .map(|home| PathBuf::from(home).join(rest))
            .unwrap_or_else(|_| PathBuf::from(path));
    }
    PathBuf::from(path)
}

fn scan_hf_cache_models() -> Vec<CachedModelSnapshot> {
    MODEL_CATALOG
        .iter()
        .filter_map(|entry| {
            hf_cache_path(entry.repo_id).map(|path| CachedModelSnapshot {
                repo_id: entry.repo_id.to_string(),
                path,
            })
        })
        .collect()
}

fn runtime_state_json(runtime: &SharedRuntime) -> Result<Value, ManagerError> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    runtime.cleanup_server();
    if runtime.hf_cache_models.is_none() {
        runtime.hf_cache_models = Some(scan_hf_cache_models());
    }
    let hf_cache_models = runtime.hf_cache_models.clone().unwrap_or_default();
    let port = runtime.server_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let server_status = match runtime.server.as_ref() {
        Some(server) if server.ready => {
            format!(
                "Running on port {} ({})",
                server.port,
                short_model_label(&server.repo_id)
            )
        }
        Some(server) => {
            format!(
                "Starting on port {} ({})",
                server.port,
                short_model_label(&server.repo_id)
            )
        }
        None => {
            // If the last status_message is a crash hint, surface it here too.
            let msg = &runtime.status_message;
            if msg.starts_with("Server exited") {
                msg.clone()
            } else {
                "Stopped".to_string()
            }
        }
    };
    let selected_repo_id = runtime.state.model_download.selected_entry().repo_id;
    let download_status = runtime
        .download_jobs
        .last()
        .map(|job| format!("{} {}", job.status, job.repo_id))
        .unwrap_or_else(|| "Idle".to_string());
    let downloaded_models: Vec<Value> = {
        let mut seen = std::collections::HashSet::new();
        let mut seen_paths = std::collections::HashSet::new();
        let mut models: Vec<Value> = Vec::new();
        // 1. Successful download jobs from this session.
        for job in runtime.download_jobs.iter().rev() {
            if job.status == "succeeded"
                && let Some(ref path) = job.model_dir
                && seen.insert(job.repo_id.clone())
                && seen_paths.insert(path.clone())
            {
                models.push(json!({ "repo_id": job.repo_id, "path": path }));
            }
        }
        // 2. Doctor-reported pre-existing model (if not already listed).
        if let Some(path) = web::current_model_dir(&runtime.state)
            && seen_paths.insert(path.clone())
            && seen.insert("__doctor__".to_string())
        {
            models.push(json!({ "repo_id": "local", "path": path }));
        }
        // 3. Cached HuggingFace cache scan for catalog models (pre-downloaded outside manager).
        for model in hf_cache_models.iter() {
            if seen.contains(model.repo_id.as_str()) {
                continue;
            }
            if seen_paths.insert(model.path.clone()) && seen.insert(model.repo_id.clone()) {
                models
                    .push(json!({ "repo_id": model.repo_id.clone(), "path": model.path.clone() }));
            }
        }
        models
    };

    Ok(json!({
        "status": runtime.status_message,
        "catalog": web::catalog_json(),
        "selected_repo_id": selected_repo_id,
        "model_dir": latest_model_dir(&runtime).or_else(|| web::current_model_dir(&runtime.state)),
        "download_status": download_status,
        "downloaded_models": downloaded_models,
        "readiness": web::readiness_json(&runtime.state),
        "server": {
            "status": server_status,
            "running": runtime.server.as_ref().is_some_and(|server| server.ready),
            "starting": runtime.server.as_ref().is_some_and(|server| !server.ready),
            "port": port,
            "base_url": base_url,
            "model_id": runtime.server.as_ref().map(|server| server.model_id.clone()),
            "repo_id": runtime.server.as_ref().map(|server| server.repo_id.clone()),
            "model_dir": runtime.server.as_ref().map(|server| server.model_dir.clone()),
            "engine": runtime.server.as_ref().map(|server| server.engine.as_str()),
            "endpoints": web::server_endpoints(&base_url),
        },
        "jobs": runtime.download_jobs.iter().map(download_job_value).collect::<Vec<_>>(),
    }))
}

fn start_download_job(runtime: &SharedRuntime, body: &str) -> Result<Value, ManagerError> {
    let payload: Value = serde_json::from_str(body)?;
    let repo_id = payload
        .get("repo_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ManagerError::Message("repo_id is required".to_string()))?
        .to_string();
    validate_catalog_repo_id(&repo_id)?;

    let (job_id, invocation) = {
        let mut runtime = runtime
            .lock()
            .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
        let invocation = model_download_invocation(&runtime.state, &repo_id)?;
        let job_id = format!("download-{}", runtime.next_job_id);
        runtime.next_job_id += 1;
        runtime.download_jobs.push(DownloadJobSnapshot {
            id: job_id.clone(),
            repo_id: repo_id.clone(),
            status: "running".to_string(),
            model_dir: None,
            message: Some("Starting mlx-lm download...".to_string()),
            progress: 1,
        });
        runtime.status_message = format!("Downloading {repo_id}");
        (job_id, invocation)
    };

    let runtime_for_job = Arc::clone(runtime);
    let job_id_for_thread = job_id.clone();
    std::thread::spawn(move || {
        run_download_with_progress(&runtime_for_job, &job_id_for_thread, &invocation);
    });

    Ok(json!({"id": job_id, "status": "running"}))
}

fn validate_catalog_repo_id(repo_id: &str) -> Result<(), ManagerError> {
    if MODEL_CATALOG.iter().any(|entry| entry.repo_id == repo_id) {
        return Ok(());
    }
    Err(ManagerError::Message(format!(
        "unknown manager catalog repo_id: {repo_id}"
    )))
}

fn download_job_json(runtime: &SharedRuntime, id: &str) -> Result<Value, ManagerError> {
    let runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    runtime
        .download_jobs
        .iter()
        .find(|job| job.id == id)
        .map(download_job_value)
        .ok_or_else(|| ManagerError::Message(format!("unknown job id: {id}")))
}

fn finish_download_job(
    runtime: &SharedRuntime,
    job_id: &str,
    result: Result<String, DownloadRunError>,
) {
    let refresh_model_dir = {
        let mut runtime = match runtime.lock() {
            Ok(runtime) => runtime,
            Err(_) => return,
        };
        let Some(index) = runtime
            .download_jobs
            .iter()
            .position(|job| job.id == job_id)
        else {
            return;
        };
        match result {
            Ok(model_dir) => {
                runtime.download_jobs[index].status = "succeeded".to_string();
                runtime.download_jobs[index].model_dir = Some(model_dir.clone());
                runtime.download_jobs[index].message = None;
                runtime.download_jobs[index].progress = 100;
                runtime.hf_cache_models = None;
                runtime.status_message =
                    format!("Downloaded {}", runtime.download_jobs[index].repo_id);
                Some(model_dir)
            }
            Err(error) => {
                let error_text = error.to_string();
                runtime.download_jobs[index].status = "failed".to_string();
                runtime.download_jobs[index].message = Some(error_text.clone());
                runtime.status_message = format!(
                    "Download failed for {}: {}",
                    runtime.download_jobs[index].repo_id, error_text
                );
                None
            }
        }
    };

    if let Some(model_dir) = refresh_model_dir {
        let refreshed = refresh_doctor_after_download(Path::new(&model_dir));
        if let Ok(mut runtime) = runtime.lock() {
            runtime.state.doctor = refreshed;
        }
    }
}

fn start_server(runtime: &SharedRuntime, body: &str) -> Result<Value, ManagerError> {
    let payload: Value = serde_json::from_str(body)?;
    let port = payload.get("port").and_then(Value::as_u64).unwrap_or(8080);
    if port == 0 || port > u16::MAX as u64 {
        return Err(ManagerError::Message(format!(
            "invalid server port: {port}"
        )));
    }
    let requested_model_dir = payload
        .get("model_dir")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let requested_repo_id = payload
        .get("repo_id")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let manual_model_dir = payload
        .get("manual_model_dir")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let engine = ManagerEngine::parse(payload.get("engine").and_then(Value::as_str))?;

    // Phase 1: resolve model dir before touching any currently running server.
    let (repo_id, model_dir) = {
        let mut rt = runtime
            .lock()
            .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
        let repo_id = requested_repo_id.unwrap_or_else(|| "local".to_string());
        let model_dir = resolve_start_model_dir(
            &rt,
            if repo_id == "local" {
                None
            } else {
                Some(repo_id.as_str())
            },
            requested_model_dir.as_deref(),
            manual_model_dir,
        )?;
        rt.status_message = format!("Preparing {}…", short_model_label(&repo_id));
        (repo_id, model_dir)
    }; // lock released

    // Phase 2: validate the launch plan before doing manifest work. Unsupported
    // manager models should fail closed without running expensive converters or
    // touching an existing server.
    let launch = server_launch_plan(&repo_id, &model_dir, port as u16, engine)?;

    // Phase 3: generate model-manifest.json if it is missing (outside the lock).
    let manifest_path = Path::new(&model_dir).join("model-manifest.json");
    if !manifest_path.is_file() {
        if let Ok(mut rt) = runtime.lock() {
            rt.status_message = format!("Generating manifest for {}…", short_model_label(&repo_id));
        }
        generate_model_manifest(&model_dir)?;
    }

    // Phase 4: spawn the server (re-acquire lock).
    let server_bin = which_server_binary();
    let stderr_path = std::env::temp_dir().join(format!("ax-engine-server-{port}.log"));
    let stderr_file = std::fs::File::create(&stderr_path).ok();

    let stopped_existing = {
        let mut rt = runtime
            .lock()
            .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
        let stopped = stop_server_locked(&mut rt);
        rt.status_message = format!(
            "Launching server on port {port} with {}",
            short_model_label(&repo_id)
        );
        stopped
    };
    if stopped_existing {
        std::thread::sleep(Duration::from_millis(300));
    }

    let child = Command::new(&server_bin)
        .args(&launch.args)
        .stdout(Stdio::null())
        .stderr(match stderr_file {
            Some(f) => Stdio::from(f),
            None => Stdio::null(),
        })
        .spawn()
        .map_err(|e| {
            ManagerError::Message(format!(
                "failed to launch ax-engine-server ({server_bin}): {e}"
            ))
        })?;

    let response = {
        let mut rt = runtime
            .lock()
            .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
        rt.server = Some(ManagedServer {
            child,
            port: port as u16,
            repo_id: repo_id.clone(),
            model_id: launch.model_id.clone(),
            model_dir: model_dir.clone(),
            engine,
            ready: false,
            stderr_file: Some(stderr_path),
        });
        rt.state.server_control.port = port as u16;
        rt.status_message = format!(
            "Server starting on port {port} with {}",
            short_model_label(&repo_id)
        );
        json!({
            "status": "starting",
            "port": port,
            "repo_id": repo_id,
            "model_id": launch.model_id,
            "model_dir": model_dir,
            "engine": engine.as_str()
        })
    }; // lock released before spawning the poll thread

    // Poll /health in a background thread so we never block the mutex.
    // Sets server.ready=true as soon as the server accepts HTTP traffic.
    {
        let rt = Arc::clone(runtime);
        let port_u16 = port as u16;
        let repo = repo_id.clone();
        let model_id = launch.model_id.clone();
        let poll_model_dir = model_dir.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(500));
                if server_health_ok(port_u16) {
                    if let Ok(mut guard) = rt.lock()
                        && let Some(server) = guard.server.as_mut()
                        && server_matches_launch(
                            server,
                            port_u16,
                            &repo,
                            &model_id,
                            &poll_model_dir,
                            engine,
                        )
                        && !server.ready
                    {
                        server.ready = true;
                        guard.status_message = format!(
                            "Server ready on port {} with {}",
                            port_u16,
                            short_model_label(&repo)
                        );
                    }
                    break;
                }
                // Stop polling if the server was stopped or replaced.
                let still_ours = rt
                    .lock()
                    .ok()
                    .and_then(|mut g| {
                        g.cleanup_server();
                        g.server.as_ref().map(|server| {
                            server_matches_launch(
                                server,
                                port_u16,
                                &repo,
                                &model_id,
                                &poll_model_dir,
                                engine,
                            )
                        })
                    })
                    .unwrap_or(false);
                if !still_ours {
                    break;
                }
            }
        });
    }

    Ok(response)
}

fn server_matches_launch(
    server: &ManagedServer,
    port: u16,
    repo_id: &str,
    model_id: &str,
    model_dir: &str,
    engine: ManagerEngine,
) -> bool {
    server.port == port
        && server.repo_id == repo_id
        && server.model_id == model_id
        && server.model_dir == model_dir
        && server.engine == engine
}

/// Run `ax-engine-bench generate-manifest <model_dir>` to produce model-manifest.json.
fn generate_model_manifest(model_dir: &str) -> Result<(), ManagerError> {
    let bench_bin = which_bench_binary();
    let output = Command::new(&bench_bin)
        .args(["generate-manifest", model_dir])
        .output()
        .map_err(|e| {
            ManagerError::Message(format!(
                "failed to run ax-engine-bench generate-manifest ({bench_bin}): {e}"
            ))
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ManagerError::Message(format!(
            "generate-manifest failed: {}",
            stderr.trim()
        )));
    }
    Ok(())
}

fn which_bench_binary() -> String {
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.with_file_name("ax-engine-bench");
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let candidate = Path::new(dir).join("ax-engine-bench");
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    "ax-engine-bench".to_string()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ServerLaunchPlan {
    args: Vec<String>,
    model_id: String,
}

fn server_launch_plan(
    repo_id: &str,
    model_dir: &str,
    port: u16,
    engine: ManagerEngine,
) -> Result<ServerLaunchPlan, ManagerError> {
    let port_arg = port.to_string();
    let repo_lower = repo_id.to_ascii_lowercase();
    if qwen3_coder_next_requires_sanitized_artifacts(&repo_lower, model_dir) {
        return Err(ManagerError::Message(format!(
            "manager cannot start {repo_id}: Qwen3-Coder-Next requires sanitized Qwen3 Next linear-attention weights. Public cache snapshots can fail the AX sanitized-weight check; convert or validate the artifact first, for example: pip install mlx-lm && mlx_lm.convert --hf-path <source> --mlx-path <dest>."
        )));
    }
    if let Some((preset, model_id)) = server_preset_for_repo(&repo_lower) {
        let mut args = vec![
            "--preset".to_string(),
            preset.to_string(),
            "--mlx-model-artifacts-dir".to_string(),
            model_dir.to_string(),
            "--port".to_string(),
            port_arg,
        ];
        if engine == ManagerEngine::AxEngine {
            args.push("--disable-ngram-acceleration".to_string());
        }
        return Ok(ServerLaunchPlan {
            args,
            model_id: model_id.to_string(),
        });
    }
    if repo_lower.contains("gemma-4") || repo_lower.contains("gemma4") {
        return Err(ManagerError::Message(format!(
            "manager cannot start {repo_id}: no ax-engine-server preset is available yet"
        )));
    }

    let mut args = vec![
        "--mlx".to_string(),
        "--mlx-model-artifacts-dir".to_string(),
        model_dir.to_string(),
        "--model-id".to_string(),
        repo_id.to_string(),
        "--port".to_string(),
        port_arg,
    ];
    if engine == ManagerEngine::AxEngine {
        args.push("--disable-ngram-acceleration".to_string());
    }

    Ok(ServerLaunchPlan {
        args,
        model_id: repo_id.to_string(),
    })
}

fn qwen3_coder_next_requires_sanitized_artifacts(repo_lower: &str, model_dir: &str) -> bool {
    let dir_lower = model_dir.to_ascii_lowercase();
    repo_lower.contains("qwen3-coder-next")
        || repo_lower.contains("qwen3_coder_next")
        || dir_lower.contains("qwen3-coder-next")
        || dir_lower.contains("qwen3_coder_next")
}

fn server_preset_for_repo(repo_lower: &str) -> Option<(&'static str, &'static str)> {
    if repo_lower.contains("gemma-4-e2b") || repo_lower.contains("gemma4-e2b") {
        Some(("gemma4-e2b", "gemma4-e2b"))
    } else if repo_lower.contains("gemma-4-31b") || repo_lower.contains("gemma4-31b") {
        Some(("gemma4-31b", "gemma4-31b"))
    } else if repo_lower.contains("glm-4.7")
        || repo_lower.contains("glm-4-7")
        || repo_lower.contains("glm4.7")
        || repo_lower.contains("glm47")
    {
        Some(("glm4.7-flash-4bit", "glm4_moe_lite"))
    } else if repo_lower.contains("qwen3.6-35b")
        || repo_lower.contains("qwen3-6-35b")
        || repo_lower.contains("qwen36-35b")
    {
        Some(("qwen3.6-35b", "qwen3.6-35b"))
    } else {
        None
    }
}

fn resolve_start_model_dir(
    runtime: &WebRuntime,
    repo_id: Option<&str>,
    requested_model_dir: Option<&str>,
    manual_model_dir: bool,
) -> Result<String, ManagerError> {
    if manual_model_dir {
        return requested_model_dir
            .map(str::to_string)
            .ok_or_else(|| ManagerError::Message("manual model_dir is empty".to_string()));
    }

    if let Some(repo_id) = repo_id {
        if let Some(path) = model_dir_for_repo(runtime, repo_id) {
            return Ok(path);
        }
        return Err(ManagerError::Message(format!(
            "selected model is not downloaded: {repo_id}"
        )));
    }

    requested_model_dir.map(str::to_string).ok_or_else(|| {
        ManagerError::Message(
            "model_dir is required — select a downloaded model and enter its path".to_string(),
        )
    })
}

fn model_dir_for_repo(runtime: &WebRuntime, repo_id: &str) -> Option<String> {
    runtime
        .download_jobs
        .iter()
        .rev()
        .find(|job| job.status == "succeeded" && job.repo_id == repo_id)
        .and_then(|job| job.model_dir.clone())
        .or_else(|| hf_cache_path(repo_id))
}

fn short_model_label(repo_id: &str) -> &str {
    repo_id.rsplit('/').next().unwrap_or(repo_id)
}

fn stop_server(runtime: &SharedRuntime) -> Result<Value, ManagerError> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    runtime.cleanup_server();
    if runtime.server.is_none() && runtime.status_message.starts_with("Server exited") {
        return Ok(json!({"status": runtime.status_message}));
    }
    let stopped = stop_server_locked(&mut runtime);
    runtime.status_message = if stopped {
        "Server stopped".to_string()
    } else {
        "Server was not running".to_string()
    };
    Ok(json!({"status": runtime.status_message}))
}

fn restart_server(runtime: &SharedRuntime, body: &str) -> Result<Value, ManagerError> {
    start_server(runtime, body)
}

fn stop_server_locked(runtime: &mut WebRuntime) -> bool {
    let Some(mut server) = runtime.server.take() else {
        return false;
    };
    let _ = server.child.kill();
    let _ = server.child.wait();
    true
}

fn server_health_ok(port: u16) -> bool {
    use std::io::{Read, Write};
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
    let req = format!("GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n");
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf).unwrap_or(0);
    let response = std::str::from_utf8(&buf[..n]).unwrap_or("");
    response.starts_with("HTTP/1.") && response.contains(" 200 ")
}

/// Return the absolute path of `ax-engine-server`, falling back to the name
/// itself (let the OS resolve from PATH) if which-lookup fails.
fn which_server_binary() -> String {
    // Prefer the binary next to the current executable (same release tree).
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.with_file_name("ax-engine-server");
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    // Scan PATH entries explicitly so background threads don't need the shell PATH.
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let candidate = Path::new(dir).join("ax-engine-server");
        if candidate.is_file() {
            return candidate.display().to_string();
        }
    }
    "ax-engine-server".to_string()
}

fn latest_model_dir(runtime: &WebRuntime) -> Option<String> {
    runtime
        .download_jobs
        .iter()
        .rev()
        .find_map(|job| job.model_dir.clone())
}

fn download_job_value(job: &DownloadJobSnapshot) -> Value {
    json!({
        "id": job.id,
        "repo_id": job.repo_id,
        "status": job.status,
        "model_dir": job.model_dir,
        "message": job.message,
        "progress": job.progress,
    })
}

fn model_download_invocation(
    state: &AppState,
    repo_id: &str,
) -> Result<CommandInvocation, ManagerError> {
    let cwd = env::current_dir()?;
    model_download_invocation_for_cwd(state, repo_id, &cwd)
}

fn model_download_invocation_for_cwd(
    state: &AppState,
    repo_id: &str,
    cwd: &Path,
) -> Result<CommandInvocation, ManagerError> {
    if let LoadState::Ready(doctor) = &state.doctor
        && let Some(command) = doctor.workflow.download_model.as_ref()
    {
        return download_invocation_from_workflow(command, repo_id);
    }

    if let Some(root) = source_checkout_root(cwd) {
        return Ok(CommandInvocation::new(
            "python3",
            vec![
                "scripts/download_model.py".to_string(),
                repo_id.to_string(),
                "--json".to_string(),
                "--progress-json".to_string(),
            ],
            Some(root),
        ));
    }

    Ok(installed_python_download_invocation(repo_id))
}

fn download_invocation_from_workflow(
    command: &WorkflowCommand,
    repo_id: &str,
) -> Result<CommandInvocation, ManagerError> {
    let Some((program, args)) = command.argv.split_first() else {
        return Err(ManagerError::Message(
            "download_model workflow command is empty".to_string(),
        ));
    };
    let mut resolved_args = args
        .iter()
        .map(|arg| {
            if arg == "<repo-id>" {
                repo_id.to_string()
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>();
    if resolved_args.iter().any(|arg| arg == "--json")
        && resolved_args
            .iter()
            .any(|arg| arg.ends_with("scripts/download_model.py") || arg == "download_model.py")
        && !resolved_args.iter().any(|arg| arg == "--progress-json")
    {
        resolved_args.push("--progress-json".to_string());
    }
    Ok(CommandInvocation::new(
        program.clone(),
        resolved_args,
        command.cwd.as_ref().map(PathBuf::from),
    ))
}

fn source_checkout_root(cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .find(|path| {
            path.join("Cargo.toml").is_file() && path.join("scripts/download_model.py").is_file()
        })
        .map(Path::to_path_buf)
}

fn installed_python_download_invocation(repo_id: &str) -> CommandInvocation {
    CommandInvocation::new(
        "python3",
        vec![
            "-c".to_string(),
            INSTALLED_DOWNLOAD_PYTHON.to_string(),
            repo_id.to_string(),
            installed_bench_program(),
        ],
        None,
    )
}

fn installed_bench_program() -> String {
    env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("ax-engine-bench")))
        .filter(|path| path.is_file())
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| "ax-engine-bench".to_string())
}

const INSTALLED_DOWNLOAD_PYTHON: &str = r#"
import json, os, subprocess, sys, threading, time
from pathlib import Path

MANIFEST = "model-manifest.json"
repo_id = sys.argv[1]
bench = sys.argv[2] if len(sys.argv) > 2 else "ax-engine-bench"
cache_base = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo_id.replace("/", "--"))
if os.environ.get("HF_HUB_CACHE"):
    cache_base = Path(os.environ["HF_HUB_CACHE"]).expanduser() / ("models--" + repo_id.replace("/", "--"))
elif os.environ.get("HF_HOME"):
    cache_base = Path(os.environ["HF_HOME"]).expanduser() / "hub" / ("models--" + repo_id.replace("/", "--"))
elif os.environ.get("XDG_CACHE_HOME"):
    cache_base = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub" / ("models--" + repo_id.replace("/", "--"))

def emit_progress(done, total, file=""):
    print(json.dumps({"event": "progress", "done": done, "total": total, "file": file}), flush=True)

def summary(dest, status, errors=None):
    dest = Path(dest)
    manifest = dest / MANIFEST
    return {
        "schema_version": "ax.download_model.v1",
        "repo_id": repo_id,
        "dest": str(dest),
        "manifest_path": str(manifest),
        "manifest_present": manifest.exists(),
        "safetensors_count": len(list(dest.glob("*.safetensors"))) if dest.exists() else 0,
        "config_present": (dest / "config.json").exists(),
        "status": status,
        "errors": errors or [],
        "server_command": ["ax-engine-server", "--mlx", "--mlx-model-artifacts-dir", str(dest), "--port", "8080"],
    }

def print_summary(dest, status, errors=None):
    print(json.dumps(summary(dest, status, errors), sort_keys=True))

def latest_snapshot():
    refs_main = cache_base / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        if revision:
            snapshot = cache_base / "snapshots" / revision
            if snapshot.is_dir():
                return snapshot
    snapshots = cache_base / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)

def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "estimating"
    seconds = int(seconds)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"

def snapshot_weight_progress(snapshot):
    index_path = snapshot / "model.safetensors.index.json"
    total = 0
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            total = int(index.get("metadata", {}).get("total_size") or 0)
        except Exception:
            total = 0
    downloaded = 0
    for path in snapshot.glob("*.safetensors"):
        try:
            downloaded += path.stat().st_size
        except OSError:
            pass
    if total <= 0 and downloaded > 0:
        total = downloaded
    if total <= 0:
        return None
    return min(downloaded, total), total

def download_progress_message(started_at):
    elapsed = time.monotonic() - started_at
    snapshot = latest_snapshot()
    if snapshot is not None:
        progress = snapshot_weight_progress(snapshot)
        if progress is not None:
            downloaded, total = progress
            ratio = 0.0 if total == 0 else downloaded / total
            eta = elapsed * (1.0 - ratio) / ratio if ratio > 0 else None
            gib = 1024 ** 3
            return (
                5 + int(min(ratio, 1.0) * 80),
                f"Downloading weights ({downloaded / gib:.1f}/{total / gib:.1f} GiB, "
                f"elapsed {format_duration(elapsed)}, ETA {format_duration(eta)})",
            )
    synthetic = min(25, 5 + int(elapsed // 20))
    return synthetic, f"Downloading with mlx-lm (elapsed {format_duration(elapsed)}, ETA estimating)"

def run_mlx_lm_generate(command, env):
    started_at = time.monotonic()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    output_parts = []

    def drain(stream):
        if stream is None:
            return
        for line in stream:
            output_parts.append(line)

    stdout_thread = threading.Thread(target=drain, args=(proc.stdout,), daemon=True)
    stderr_thread = threading.Thread(target=drain, args=(proc.stderr,), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    while proc.poll() is None:
        done, message = download_progress_message(started_at)
        emit_progress(done, 100, message)
        time.sleep(2)
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    return subprocess.CompletedProcess(command, proc.returncode, "".join(output_parts), "")

emit_progress(0, 3, "Starting mlx-lm download…")
try:
    dest = latest_snapshot()
    if dest is not None and list(dest.glob("*.safetensors")) and (dest / "config.json").exists():
        emit_progress(1, 3, "Using existing mlx-lm cache snapshot…")
    else:
        dest = None
except Exception:
    dest = None

try:
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if dest is None:
        command = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", repo_id,
            "--prompt", "x",
            "--max-tokens", "1",
        ]
        result = run_mlx_lm_generate(command, env)
        if result.returncode != 0:
            output = "\n".join(part for part in [result.stderr.strip(), result.stdout.strip()] if part)
            print_summary(cache_base, "download_failed",
                ["mlx-lm is required. Run: python3 -m pip install mlx-lm", output])
            raise SystemExit(1)
        emit_progress(1, 3, "Resolving mlx-lm cache snapshot…")
        dest = latest_snapshot()
    if dest is None:
        print_summary(cache_base, "download_failed",
            [f"mlx-lm completed but no cache snapshot was found for {repo_id}"])
        raise SystemExit(1)
    dest = Path(dest)
except Exception as exc:
    print_summary(cache_base, "download_failed", [str(exc)])
    raise SystemExit(1)

errors = []
if not list(dest.glob("*.safetensors")):
    errors.append(f"no .safetensors files found in {dest}")
if not (dest / "config.json").exists():
    errors.append(f"config.json missing in {dest}")
if errors:
    print_summary(dest, "invalid", errors)
    raise SystemExit(1)

emit_progress(2, 3, "Generating manifest…")
if not (dest / MANIFEST).exists():
    result = subprocess.run([bench, "generate-manifest", str(dest), "--json"], capture_output=True, text=True)
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or f"{bench} exited with {result.returncode}"
        print_summary(dest, "manifest_missing", [error])
        raise SystemExit(1)

emit_progress(3, 3, "Ready")
print_summary(dest, "ready")
"#;

fn run_download_with_progress(
    runtime: &SharedRuntime,
    job_id: &str,
    invocation: &CommandInvocation,
) {
    let mut command = Command::new(&invocation.program);
    command.args(&invocation.args);
    if let Some(cwd) = invocation.cwd.as_deref() {
        command.current_dir(cwd);
    }
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            finish_download_job(runtime, job_id, Err(DownloadRunError::Io(e)));
            return;
        }
    };

    // Drain stderr in a background thread so a full pipe never blocks the child.
    let stderr_thread = child.stderr.take().map(|stderr| {
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = std::io::BufReader::new(stderr).read_to_string(&mut buf);
            buf.trim().to_string()
        })
    });

    // Stream stdout: progress events update the job snapshot; final summaries
    // may be newline-delimited JSON or pretty multi-line JSON, depending on
    // whether the workflow command includes --progress-json.
    let mut stdout_text = String::new();
    let mut last_summary = String::new();
    if let Some(stdout) = child.stdout.take() {
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            stdout_text.push_str(&line);
            stdout_text.push('\n');
            let trimmed = line.trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(json) = serde_json::from_str::<Value>(&trimmed) {
                if json.get("event").and_then(Value::as_str) == Some("progress") {
                    let done = json.get("done").and_then(Value::as_u64).unwrap_or(0);
                    let total = json
                        .get("total")
                        .and_then(Value::as_u64)
                        .unwrap_or(1)
                        .max(1);
                    let file = json
                        .get("file")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let pct = ((done as f64 / total as f64) * 100.0).min(99.0) as u8;
                    update_download_progress(runtime, job_id, pct, &file);
                } else if is_download_summary_json(&json) {
                    last_summary = trimmed;
                }
            }
        }
    }

    let exit_ok = child.wait().is_ok_and(|s| s.success());
    let stderr = stderr_thread
        .and_then(|t| t.join().ok())
        .unwrap_or_default();

    let result = if last_summary.is_empty() {
        parse_download_stdout(&stdout_text, exit_ok, &stderr)
    } else {
        parse_download_summary(&last_summary, exit_ok, &stderr)
    };
    finish_download_job(runtime, job_id, result);
}

fn update_download_progress(runtime: &SharedRuntime, job_id: &str, pct: u8, file: &str) {
    if let Ok(mut rt) = runtime.lock()
        && let Some(job) = rt.download_jobs.iter_mut().find(|j| j.id == job_id)
    {
        job.progress = pct;
        if !file.is_empty() {
            job.message = Some(format!("{file}  {pct}%"));
        }
    }
}

fn parse_download_summary(
    line: &str,
    exit_ok: bool,
    stderr: &str,
) -> Result<String, DownloadRunError> {
    let summary: Value = serde_json::from_str(line).map_err(|e| {
        if exit_ok || stderr.is_empty() {
            DownloadRunError::InvalidSummary(e.to_string())
        } else {
            DownloadRunError::Failed(stderr.to_string())
        }
    })?;
    let status = summary
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let dest = summary
        .get("dest")
        .and_then(Value::as_str)
        .ok_or_else(|| DownloadRunError::InvalidSummary("missing dest".to_string()))?;
    if exit_ok && status == "ready" {
        return Ok(dest.to_string());
    }
    let errors = summary
        .get("errors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        })
        .filter(|m| !m.is_empty())
        .or_else(|| {
            if stderr.is_empty() {
                None
            } else {
                Some(stderr.to_string())
            }
        })
        .unwrap_or_else(|| format!("status={status}"));
    Err(DownloadRunError::Failed(errors))
}

fn parse_download_stdout(
    stdout: &str,
    exit_ok: bool,
    stderr: &str,
) -> Result<String, DownloadRunError> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(DownloadRunError::Failed(if stderr.is_empty() {
            "download process produced no output".to_string()
        } else {
            stderr.to_string()
        }));
    }

    let mut last_summary = None;
    for line in trimmed.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(json) = serde_json::from_str::<Value>(line)
            && is_download_summary_json(&json)
        {
            last_summary = Some(line.to_string());
        }
    }
    if let Some(summary) = last_summary {
        return parse_download_summary(&summary, exit_ok, stderr);
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(json) if is_download_summary_json(&json) => {
            parse_download_summary(trimmed, exit_ok, stderr)
        }
        Ok(_) if !stderr.is_empty() && !exit_ok => {
            Err(DownloadRunError::Failed(stderr.to_string()))
        }
        Ok(_) => Err(DownloadRunError::InvalidSummary(
            "missing download summary".to_string(),
        )),
        Err(error) if exit_ok || stderr.is_empty() => {
            Err(DownloadRunError::InvalidSummary(error.to_string()))
        }
        Err(_) => Err(DownloadRunError::Failed(stderr.to_string())),
    }
}

fn is_download_summary_json(json: &Value) -> bool {
    json.get("schema_version").and_then(Value::as_str) == Some("ax.download_model.v1")
        || (json.get("status").is_some() && json.get("dest").is_some())
}

#[derive(Debug, Error)]
enum DownloadRunError {
    #[error("download command failed: {0}")]
    Io(io::Error),
    #[error("download command exited unsuccessfully: {0}")]
    Failed(String),
    #[error("download summary was invalid: {0}")]
    InvalidSummary(String),
}

#[allow(dead_code)] // used in tests
fn parse_download_output(output: Output) -> Result<String, DownloadRunError> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let summary: Value = serde_json::from_str(&stdout).map_err(|error| {
        if output.status.success() || stderr.is_empty() {
            DownloadRunError::InvalidSummary(error.to_string())
        } else {
            DownloadRunError::Failed(stderr.clone())
        }
    })?;
    let status = summary
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let dest = summary
        .get("dest")
        .and_then(Value::as_str)
        .ok_or_else(|| DownloadRunError::InvalidSummary("missing dest".to_string()))?;
    if output.status.success() && status == "ready" {
        return Ok(dest.to_string());
    }
    let errors = summary
        .get("errors")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join("; ")
        })
        .filter(|message| !message.is_empty())
        .or({
            if stderr.is_empty() {
                None
            } else {
                Some(stderr)
            }
        })
        .unwrap_or_else(|| format!("status={status}"));
    Err(DownloadRunError::Failed(errors))
}

fn refresh_doctor_after_download(model_dir: &Path) -> LoadState<DoctorReport> {
    match env::current_dir() {
        Ok(cwd) => {
            let command = DoctorCommand::from_cwd(&cwd, Some(model_dir));
            run_doctor(&command)
                .map(LoadState::Ready)
                .unwrap_or_else(|error| LoadState::unavailable(error.to_string()))
        }
        Err(error) => LoadState::unavailable(format!("failed to refresh doctor: {error}")),
    }
}

#[derive(Debug, Eq, PartialEq)]
struct HttpRequest {
    method: String,
    path: String,
    body: String,
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, ManagerError> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        let size = stream.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..size]);
        if request_complete(&bytes) {
            break;
        }
        if bytes.len() > 1024 * 1024 {
            return Err(ManagerError::Message("request too large".to_string()));
        }
    }
    parse_http_request_bytes(&bytes)
}

fn request_complete(bytes: &[u8]) -> bool {
    let Some(header_end) = find_header_end(bytes) else {
        return false;
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let content_length = content_length(&headers).unwrap_or(0);
    request_body_end(header_end, content_length).is_some_and(|end| bytes.len() >= end)
}

fn parse_http_request_bytes(bytes: &[u8]) -> Result<HttpRequest, ManagerError> {
    let header_end = find_header_end(bytes)
        .ok_or_else(|| ManagerError::Message("malformed HTTP request".to_string()))?;
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| ManagerError::Message("missing request line".to_string()))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| ManagerError::Message("missing method".to_string()))?;
    let path = parts
        .next()
        .ok_or_else(|| ManagerError::Message("missing path".to_string()))?;
    let body_start = header_end + 4;
    let length = content_length(&headers).unwrap_or(0);
    let body_end = request_body_end(header_end, length)
        .ok_or_else(|| ManagerError::Message("content-length is too large".to_string()))?;
    if bytes.len() < body_end {
        return Err(ManagerError::Message(format!(
            "incomplete HTTP request body: expected {length} bytes, got {}",
            bytes.len().saturating_sub(body_start)
        )));
    }
    let body = String::from_utf8_lossy(&bytes[body_start..body_end]).to_string();
    Ok(HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        body,
    })
}

fn request_body_end(header_end: usize, content_length: usize) -> Option<usize> {
    header_end.checked_add(4)?.checked_add(content_length)
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn content_length(headers: &str) -> Option<usize> {
    headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

fn json_response(status: &str, value: &Value) -> Result<String, ManagerError> {
    let body = serde_json::to_string(value)?;
    Ok(http_response(status, "application/json", &body))
}

fn http_response(status: &str, content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn parse_args_accepts_web_options() {
        let options = parse_args(
            [
                "--check",
                "--phase2-check",
                "--doctor-json",
                "doctor.json",
                "--server-url",
                "http://127.0.0.1:8080",
                "--artifact-root",
                "benchmarks/results",
                "--profile-dir",
                "profiles",
                "--support-bundle",
                "bundle",
                "--web-host",
                "127.0.0.1",
                "--web-port",
                "9876",
                "--no-open",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .expect("args should parse");

        assert!(options.check);
        assert!(options.phase2_check);
        assert_eq!(options.doctor_json, Some(PathBuf::from("doctor.json")));
        assert_eq!(options.server_url.as_deref(), Some("http://127.0.0.1:8080"));
        assert_eq!(
            options.artifact_root,
            Some(PathBuf::from("benchmarks/results"))
        );
        assert_eq!(options.profile_dir, Some(PathBuf::from("profiles")));
        assert_eq!(options.support_bundle, Some(PathBuf::from("bundle")));
        assert_eq!(options.web_port, 9876);
        assert!(options.no_open);
    }

    #[test]
    fn check_summary_reports_all_surfaces() {
        let mut state = AppState::empty();
        state.benchmark_summary = LoadState::unavailable("missing benchmark artifact");
        state.artifacts = LoadState::Ready(Vec::new());

        let mut summary = Vec::new();
        write_check_summary(&mut summary, &state).expect("summary should write");
        let summary = String::from_utf8(summary).expect("summary should be utf8");

        assert!(summary.contains("doctor=not_loaded"));
        assert!(summary.contains("server=not_loaded"));
        assert!(summary.contains("benchmark=unavailable reason=missing benchmark artifact"));
        assert!(summary.contains("artifacts=ready count=0"));
    }

    #[test]
    fn fallback_qwen_prompt_matches_enable_thinking_false_shape() {
        let messages = vec![json!({"role":"user","content":"hello"})];

        let prompt = render_fallback_chat_prompt("mlx-community/Qwen3-4B-4bit", None, &messages);

        assert_eq!(
            prompt,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn fallback_gemma4_prompt_matches_turn_template_shape() {
        let messages = vec![
            json!({"role":"system","content":"You are helpful."}),
            json!({"role":"user","content":"hello"}),
        ];

        let prompt =
            render_fallback_chat_prompt("mlx-community/gemma-4-e2b-it-4bit", None, &messages);

        assert_eq!(
            prompt,
            "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn fallback_glm_prompt_matches_tokenizer_template_shape() {
        let messages = vec![
            json!({"role":"system","content":"Be concise."}),
            json!({"role":"user","content":"hello"}),
        ];

        let prompt =
            render_fallback_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", None, &messages);

        assert_eq!(
            prompt,
            "[gMASK]<sop><|system|>Be concise.<|user|>hello<|assistant|></think>"
        );
    }

    #[test]
    fn fallback_glm_prompt_preserves_assistant_history_shape() {
        let messages = vec![
            json!({"role":"user","content":"hello"}),
            json!({"role":"assistant","content":"hi there"}),
            json!({"role":"user","content":"next?"}),
        ];

        let prompt =
            render_fallback_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", None, &messages);

        assert_eq!(
            prompt,
            "[gMASK]<sop><|user|>hello<|assistant|></think>hi there<|user|>next?<|assistant|></think>"
        );
    }

    #[test]
    fn fallback_glm_prompt_preserves_tool_observation_shape() {
        let messages = vec![
            json!({"role":"user","content":"call tool"}),
            json!({"role":"assistant","content":"<tool_call>x</tool_call>"}),
            json!({"role":"tool","content":"tool result"}),
            json!({"role":"user","content":"continue"}),
        ];

        let prompt =
            render_fallback_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", None, &messages);

        assert_eq!(
            prompt,
            "[gMASK]<sop><|user|>call tool<|assistant|></think><tool_call>x</tool_call><|observation|><tool_response>tool result</tool_response><|user|>continue<|assistant|></think>"
        );
    }

    #[test]
    fn manager_tokenizer_kwargs_disable_thinking_for_qwen_and_glm() {
        assert!(disables_thinking_chat_template(
            "mlx-community/Qwen3-4B-4bit",
            "/models/qwen"
        ));
        assert!(disables_thinking_chat_template(
            "mlx-community/GLM-4.7-Flash-4bit",
            "/models/glm"
        ));
        assert!(!disables_thinking_chat_template(
            "mlx-community/gemma-4-e2b-it-4bit",
            "/models/gemma"
        ));
    }

    #[test]
    fn normalize_chat_messages_maps_legacy_function_role_to_tool() {
        let messages = vec![json!({"role":"function","content":"tool result"})];

        let normalized = normalize_chat_messages(&messages).expect("messages should normalize");

        assert_eq!(normalized[0]["role"], "tool");
        assert_eq!(normalized[0]["content"], "tool result");
    }

    #[test]
    fn strip_gemma4_turn_artifacts_truncates_next_turn_loop() {
        let text = "model\nhello there<turn|>\n<|turn>user\nrepeat?<turn|>\n<|turn>model\n";

        let cleaned =
            strip_generation_artifacts(text, "mlx-community/gemma-4-e2b-it-4bit", "/models/gemma");

        assert_eq!(cleaned, "hello there");
    }

    #[test]
    fn strip_legacy_gemma_turn_artifacts_truncates_next_turn_loop() {
        let text =
            "<start_of_turn>model\nhello<end_of_turn>\n<start_of_turn>user\nagain?<end_of_turn>";

        let cleaned = strip_generation_artifacts(text, "gemma-3n-e4b-it", "/models/gemma-3n");

        assert_eq!(cleaned, "hello");
    }

    #[test]
    fn chat_content_parts_are_normalized_for_tokenizer_template() {
        let messages = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ]
        })];

        let normalized = normalize_chat_messages(&messages).expect("messages should normalize");

        assert_eq!(normalized[0]["role"], "user");
        assert_eq!(normalized[0]["content"], "hello world");
    }

    #[test]
    fn manager_chat_request_uses_bounded_deterministic_defaults_and_family_stops() {
        let body = json!({
            "messages": [{"role": "user", "content": "hello"}]
        })
        .to_string();

        let cases = [
            ("qwen3_dense", None, json!(["<|im_end|>"])),
            ("gemma4-e2b", None, json!(["<turn|>"])),
            (
                "glm4_moe_lite",
                None,
                json!(["<|endoftext|>", "<|user|>", "<|observation|>"]),
            ),
        ];

        for (model_id, model_dir, expected_stops) in cases {
            let native = openai_to_generate_request(&body, model_id, model_dir)
                .expect("request should convert");
            let value: Value = serde_json::from_str(&native).expect("native body should be json");

            assert_eq!(value["max_output_tokens"], 128);
            assert_eq!(value["sampling"]["temperature"], 0.0);
            assert_eq!(value["stop_sequences"], expected_stops);
        }

        assert_eq!(
            manager_chat_stop_sequences(
                "local",
                Some("/models/mlx-community--gemma-4-e2b-it-4bit")
            ),
            vec!["<turn|>".to_string()]
        );
    }

    #[test]
    fn native_sse_proxy_emits_final_response_output_text_when_steps_are_empty() {
        let initial = br#"event: response
data: {"response":{"output_text":"hello final","finish_reason":"max_output_tokens"}}

"#;
        let mut output = Vec::new();
        let mut upstream = std::io::Cursor::new(Vec::<u8>::new());

        stream_native_as_openai(&mut output, &mut upstream, initial).expect("sse should convert");

        let response = String::from_utf8(output).expect("response should be utf8");
        assert!(response.contains("\"content\":\"hello final\""));
        assert!(response.contains("\"finish_reason\":\"length\""));
        assert!(response.contains("data: [DONE]"));
    }

    #[test]
    fn native_sse_proxy_emits_only_missing_final_output_tail() {
        let initial = br#"event: step
data: {"delta_text":"hello"}

event: response
data: {"response":{"output_text":"hello world","finish_reason":"stop"}}

"#;
        let mut output = Vec::new();
        let mut upstream = std::io::Cursor::new(Vec::<u8>::new());

        stream_native_as_openai(&mut output, &mut upstream, initial).expect("sse should convert");

        let response = String::from_utf8(output).expect("response should be utf8");
        assert!(response.contains("\"content\":\"hello\""));
        assert!(response.contains("\"content\":\" world\""));
        assert!(response.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn download_workflow_replaces_repo_placeholder() {
        let command = WorkflowCommand {
            argv: vec![
                "python3".to_string(),
                "scripts/download_model.py".to_string(),
                "<repo-id>".to_string(),
                "--json".to_string(),
            ],
            cwd: Some("/repo".to_string()),
        };

        let invocation = download_invocation_from_workflow(&command, "mlx-community/Qwen3-4B-4bit")
            .expect("download invocation should build");

        assert_eq!(invocation.program, "python3");
        assert_eq!(
            invocation.args,
            vec![
                "scripts/download_model.py",
                "mlx-community/Qwen3-4B-4bit",
                "--json",
                "--progress-json"
            ]
        );
        assert_eq!(invocation.cwd, Some(PathBuf::from("/repo")));
    }

    #[test]
    fn download_invocation_uses_source_checkout_when_available() {
        let root = tempfile::tempdir().expect("tempdir should create");
        let scripts = root.path().join("scripts");
        std::fs::create_dir_all(&scripts).expect("scripts dir should create");
        std::fs::write(root.path().join("Cargo.toml"), "[workspace]\n")
            .expect("cargo toml should write");
        std::fs::write(scripts.join("download_model.py"), "")
            .expect("download helper should write");

        let nested = root.path().join("nested");
        std::fs::create_dir_all(&nested).expect("nested dir should create");
        let invocation = model_download_invocation_for_cwd(
            &AppState::empty(),
            "mlx-community/Qwen3-4B-4bit",
            &nested,
        )
        .expect("source checkout invocation should build");

        assert_eq!(invocation.program, "python3");
        assert_eq!(
            invocation.args,
            vec![
                "scripts/download_model.py",
                "mlx-community/Qwen3-4B-4bit",
                "--json",
                "--progress-json",
            ]
        );
        assert_eq!(invocation.cwd, Some(root.path().to_path_buf()));
    }

    #[test]
    fn download_invocation_falls_back_to_installed_python_outside_checkout() {
        let root = tempfile::tempdir().expect("tempdir should create");
        let invocation = model_download_invocation_for_cwd(
            &AppState::empty(),
            "mlx-community/Qwen3-4B-4bit",
            root.path(),
        )
        .expect("installed invocation should build");

        assert_eq!(invocation.program, "python3");
        assert_eq!(invocation.args[0], "-c");
        assert!(invocation.args[1].contains("\"mlx_lm\", \"generate\""));
        assert!(invocation.args[1].contains("Downloading weights"));
        assert!(invocation.args[1].contains("ETA"));
        assert!(invocation.args[1].contains("XDG_CACHE_HOME"));
        assert!(invocation.args[1].contains("Using existing mlx-lm cache snapshot"));
        assert_eq!(invocation.args[2], "mlx-community/Qwen3-4B-4bit");
        assert!(invocation.args[3].contains("ax-engine-bench"));
        assert_eq!(invocation.cwd, None);
    }

    #[test]
    fn start_download_rejects_non_catalog_repo_without_creating_job() {
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));

        let error = start_download_job(
            &runtime,
            r#"{"repo_id":"mlx-community/Not-In-Manager-Catalog"}"#,
        )
        .expect_err("non-catalog repo should be rejected");

        assert_eq!(
            error.to_string(),
            "unknown manager catalog repo_id: mlx-community/Not-In-Manager-Catalog"
        );
        let guard = runtime.lock().expect("runtime lock should work");
        assert!(guard.download_jobs.is_empty());
        assert_eq!(guard.next_job_id, 1);
    }

    #[test]
    fn catalog_repo_validation_accepts_catalog_id() {
        validate_catalog_repo_id("mlx-community/Qwen3-4B-4bit")
            .expect("catalog repo should validate");
    }

    #[test]
    fn subprocess_timeout_returns_without_waiting_for_child_exit() {
        let started = Instant::now();

        let error = run_subprocess_output_with_timeout(
            "sleep",
            vec!["2".to_string()],
            Duration::from_millis(50),
            "test subprocess",
        )
        .expect_err("sleep should time out");

        assert!(error.to_string().contains("test subprocess timed out"));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "timeout should kill the subprocess instead of waiting for natural exit"
        );
    }

    #[test]
    fn parses_ready_download_output() {
        let output = Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: br#"{"schema_version":"ax.download_model.v1","status":"ready","dest":"/models/qwen","errors":[]}"#.to_vec(),
            stderr: Vec::new(),
        };

        assert_eq!(
            parse_download_output(output).expect("download should parse"),
            "/models/qwen"
        );
    }

    #[test]
    fn parses_progress_json_download_stdout_without_treating_events_as_summary() {
        let stdout = r#"
{"event":"progress","done":0,"total":100,"file":"Starting mlx-lm download"}
{"event":"progress","done":85,"total":100,"file":"Using existing mlx-lm cache snapshot"}
{"schema_version":"ax.download_model.v1","status":"ready","dest":"/models/glm","errors":[]}
"#;

        assert_eq!(
            parse_download_stdout(stdout, true, "").expect("download should parse"),
            "/models/glm"
        );
    }

    #[test]
    fn parses_pretty_json_download_stdout_from_workflow_command() {
        let stdout = r#"{
  "schema_version": "ax.download_model.v1",
  "status": "ready",
  "dest": "/models/glm",
  "errors": []
}"#;

        assert_eq!(
            parse_download_stdout(stdout, true, "").expect("download should parse"),
            "/models/glm"
        );
    }

    #[test]
    fn ignores_progress_only_json_when_reporting_failed_download() {
        let stdout = r#"{"event":"progress","done":5,"total":100,"file":"Downloading"}"#;

        let error = parse_download_stdout(stdout, false, "mlx-lm failed")
            .expect_err("progress-only output should not be treated as a summary");

        assert_eq!(
            error.to_string(),
            "download command exited unsuccessfully: mlx-lm failed"
        );
    }

    #[test]
    fn failed_non_json_download_output_reports_stderr() {
        let output = Output {
            status: std::process::ExitStatus::from_raw(1),
            stdout: b"not json".to_vec(),
            stderr: b"missing dependency".to_vec(),
        };

        let error = parse_download_output(output).expect_err("download should fail");
        assert_eq!(
            error.to_string(),
            "download command exited unsuccessfully: missing dependency"
        );
    }

    #[test]
    fn start_model_resolution_uses_selected_repo_not_stale_path() {
        let mut runtime = WebRuntime::new(AppState::empty());
        runtime.download_jobs.push(DownloadJobSnapshot {
            id: "download-1".to_string(),
            repo_id: "mlx-community/gemma-4-e2b-it-4bit".to_string(),
            status: "succeeded".to_string(),
            model_dir: Some("/models/gemma".to_string()),
            message: None,
            progress: 0,
        });

        let resolved = resolve_start_model_dir(
            &runtime,
            Some("mlx-community/gemma-4-e2b-it-4bit"),
            Some("/models/qwen3"),
            false,
        )
        .expect("selected repo should resolve to its own model dir");

        assert_eq!(resolved, "/models/gemma");
    }

    #[test]
    fn start_model_resolution_rejects_missing_selected_repo() {
        let runtime = WebRuntime::new(AppState::empty());
        let error = resolve_start_model_dir(
            &runtime,
            Some("local-test/missing-selected-model"),
            Some("/models/qwen3"),
            false,
        )
        .expect_err("missing selected repo should not fall back to stale path");

        assert_eq!(
            error.to_string(),
            "selected model is not downloaded: local-test/missing-selected-model"
        );
    }

    #[test]
    fn start_model_resolution_allows_manual_path_override() {
        let runtime = WebRuntime::new(AppState::empty());
        let resolved = resolve_start_model_dir(
            &runtime,
            Some("mlx-community/gemma-4-e2b-it-4bit"),
            Some("/manual/model"),
            true,
        )
        .expect("manual path should be accepted");

        assert_eq!(resolved, "/manual/model");
    }

    #[test]
    fn cleanup_server_removes_exited_child_and_reports_stderr_hint() {
        let stderr_path = tempfile::NamedTempFile::new()
            .expect("stderr tempfile should create")
            .into_temp_path();
        let stderr_file = std::fs::File::create(&stderr_path).expect("stderr file should open");
        let child = Command::new("sh")
            .arg("-c")
            .arg("echo startup failed >&2; exit 7")
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .expect("exiting child should start");
        let mut runtime = WebRuntime::new(AppState::empty());
        runtime.server = Some(ManagedServer {
            child,
            port: 32130,
            repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
            model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
            model_dir: "/models/qwen".to_string(),
            engine: ManagerEngine::AxEngineNgram,
            ready: false,
            stderr_file: Some(stderr_path.to_path_buf()),
        });

        for _ in 0..20 {
            runtime.cleanup_server();
            if runtime.server.is_none() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        assert!(runtime.server.is_none());
        assert!(runtime.status_message.contains("Server exited"));
        assert!(runtime.status_message.contains("startup failed"));
    }

    #[test]
    fn stop_server_preserves_crash_hint_for_exited_child() {
        let stderr_path = tempfile::NamedTempFile::new()
            .expect("stderr tempfile should create")
            .into_temp_path();
        let stderr_file = std::fs::File::create(&stderr_path).expect("stderr file should open");
        let child = Command::new("sh")
            .arg("-c")
            .arg("echo launch failed >&2; exit 9")
            .stderr(Stdio::from(stderr_file))
            .spawn()
            .expect("exiting child should start");
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.server = Some(ManagedServer {
                child,
                port: 32131,
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_dir: "/models/qwen".to_string(),
                engine: ManagerEngine::AxEngineNgram,
                ready: false,
                stderr_file: Some(stderr_path.to_path_buf()),
            });
        }

        std::thread::sleep(Duration::from_millis(100));
        let response = stop_server(&runtime).expect("stop should return status");
        let status = response["status"]
            .as_str()
            .expect("stop response should include status");
        assert!(
            status.starts_with("Server exited"),
            "stop should preserve the crash status instead of reporting a clean stop: {status}"
        );
        assert!(
            status.contains("launch failed"),
            "stop should preserve the stderr crash hint: {status}"
        );
        let guard = runtime.lock().expect("runtime lock should work");
        assert!(
            guard.server.is_none(),
            "exited server should be removed after stop"
        );
    }

    #[test]
    fn start_server_keeps_running_server_when_model_resolution_fails() {
        let child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("fake server should start");
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.server = Some(ManagedServer {
                child,
                port: 32123,
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_dir: "/models/qwen".to_string(),
                engine: ManagerEngine::AxEngineNgram,
                ready: true,
                stderr_file: None,
            });
        }

        let error = start_server(
            &runtime,
            r#"{"repo_id":"local-test/missing-selected-model","model_dir":"/stale/model","manual_model_dir":false,"port":32124}"#,
        )
        .expect_err("missing selected model should fail");

        assert_eq!(
            error.to_string(),
            "selected model is not downloaded: local-test/missing-selected-model"
        );
        let mut guard = runtime.lock().expect("runtime lock should work");
        let still_running = guard
            .server
            .as_mut()
            .expect("old server should still be tracked")
            .child
            .try_wait()
            .expect("fake server status should be readable")
            .is_none();
        assert!(
            still_running,
            "old server should not be killed by validation failure"
        );
        stop_server_locked(&mut guard);
    }

    #[test]
    fn restart_server_keeps_running_server_when_model_resolution_fails() {
        let child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("fake server should start");
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.server = Some(ManagedServer {
                child,
                port: 32125,
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_dir: "/models/qwen".to_string(),
                engine: ManagerEngine::AxEngineNgram,
                ready: true,
                stderr_file: None,
            });
        }

        let error = restart_server(
            &runtime,
            r#"{"repo_id":"local-test/missing-selected-model","model_dir":"/stale/model","manual_model_dir":false,"port":32126}"#,
        )
        .expect_err("missing selected model should fail");

        assert_eq!(
            error.to_string(),
            "selected model is not downloaded: local-test/missing-selected-model"
        );
        let mut guard = runtime.lock().expect("runtime lock should work");
        let still_running = guard
            .server
            .as_mut()
            .expect("old server should still be tracked")
            .child
            .try_wait()
            .expect("fake server status should be readable")
            .is_none();
        assert!(
            still_running,
            "old server should not be killed by restart validation failure"
        );
        stop_server_locked(&mut guard);
    }

    #[test]
    fn start_server_rejects_unsupported_model_before_manifest_generation() {
        let child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("fake server should start");
        let model_dir = tempfile::tempdir().expect("model dir should create");
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.download_jobs.push(DownloadJobSnapshot {
                id: "download-unsupported".to_string(),
                repo_id: "mlx-community/gemma-4-e4b-it-4bit".to_string(),
                status: "succeeded".to_string(),
                model_dir: Some(model_dir.path().display().to_string()),
                message: None,
                progress: 100,
            });
            guard.server = Some(ManagedServer {
                child,
                port: 32128,
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                model_dir: "/models/qwen".to_string(),
                engine: ManagerEngine::AxEngineNgram,
                ready: true,
                stderr_file: None,
            });
        }

        let body = json!({
            "repo_id": "mlx-community/gemma-4-e4b-it-4bit",
            "model_dir": "/stale/model",
            "manual_model_dir": false,
            "port": 32129,
            "engine": "ax-engine-ngram"
        })
        .to_string();
        let error = start_server(&runtime, &body)
            .expect_err("unsupported model should fail before manifest generation");

        assert_eq!(
            error.to_string(),
            "manager cannot start mlx-community/gemma-4-e4b-it-4bit: no ax-engine-server preset is available yet"
        );
        assert!(
            !model_dir.path().join("model-manifest.json").exists(),
            "unsupported launch should not generate a manifest"
        );
        let mut guard = runtime.lock().expect("runtime lock should work");
        let still_running = guard
            .server
            .as_mut()
            .expect("old server should still be tracked")
            .child
            .try_wait()
            .expect("fake server status should be readable")
            .is_none();
        assert!(
            still_running,
            "old server should not be killed by unsupported launch validation"
        );
        stop_server_locked(&mut guard);
    }

    #[test]
    fn server_health_poll_identity_rejects_reused_port_with_different_model() {
        let child = Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("fake server should start");
        let mut server = ManagedServer {
            child,
            port: 32127,
            repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
            model_id: "mlx-community/Qwen3-4B-4bit".to_string(),
            model_dir: "/models/qwen".to_string(),
            engine: ManagerEngine::AxEngineNgram,
            ready: false,
            stderr_file: None,
        };

        assert!(server_matches_launch(
            &server,
            32127,
            "mlx-community/Qwen3-4B-4bit",
            "mlx-community/Qwen3-4B-4bit",
            "/models/qwen",
            ManagerEngine::AxEngineNgram,
        ));
        assert!(!server_matches_launch(
            &server,
            32127,
            "mlx-community/Gemma-4-E2B-it-4bit",
            "gemma4-e2b",
            "/models/gemma",
            ManagerEngine::AxEngineNgram,
        ));

        let _ = server.child.kill();
        let _ = server.child.wait();
    }

    #[test]
    fn server_launch_plan_uses_glm_preset() {
        let plan = server_launch_plan(
            "mlx-community/GLM-4.7-Flash-4bit",
            "/models/glm",
            9090,
            ManagerEngine::AxEngineNgram,
        )
        .expect("glm preset should be startable");

        assert_eq!(plan.model_id, "glm4_moe_lite");
        assert_eq!(
            plan.args,
            vec![
                "--preset",
                "glm4.7-flash-4bit",
                "--mlx-model-artifacts-dir",
                "/models/glm",
                "--port",
                "9090"
            ]
        );
        assert!(!plan.args.iter().any(|arg| arg == "--model-id"));
    }

    #[test]
    fn server_launch_plan_uses_gemma_preset() {
        let plan = server_launch_plan(
            "mlx-community/gemma-4-e2b-it-4bit",
            "/models/gemma",
            8081,
            ManagerEngine::AxEngineNgram,
        )
        .expect("gemma e2b preset should be startable");

        assert_eq!(plan.model_id, "gemma4-e2b");
        assert_eq!(plan.args[0], "--preset");
        assert_eq!(plan.args[1], "gemma4-e2b");
        assert!(!plan.args.iter().any(|arg| arg == "--model-id"));
    }

    #[test]
    fn server_launch_plan_rejects_gemma_catalog_entries_without_server_preset() {
        let error = server_launch_plan(
            "mlx-community/gemma-4-e4b-it-4bit",
            "/models/gemma-e4b",
            8082,
            ManagerEngine::AxEngineNgram,
        )
        .expect_err("unmapped gemma catalog entry should fail closed");

        assert_eq!(
            error.to_string(),
            "manager cannot start mlx-community/gemma-4-e4b-it-4bit: no ax-engine-server preset is available yet"
        );
    }

    #[test]
    fn server_launch_plan_keeps_generic_mlx_model_id_for_qwen() {
        let repo = "mlx-community/Qwen3-4B-4bit";
        let plan = server_launch_plan(repo, "/models/qwen", 8080, ManagerEngine::AxEngineNgram)
            .expect("qwen should use generic mlx route");

        assert_eq!(plan.model_id, repo);
        assert!(plan.args.iter().any(|arg| arg == "--mlx"));
        assert!(
            plan.args
                .windows(2)
                .any(|pair| pair == ["--model-id", repo])
        );
    }

    #[test]
    fn manager_catalog_omits_qwen3_coder_next_until_sanitized_download_exists() {
        assert!(
            !MODEL_CATALOG
                .iter()
                .any(|entry| entry.repo_id == "mlx-community/Qwen3-Coder-Next-4bit"),
            "manager catalog should contain only directly downloadable and startable models"
        );
    }

    #[test]
    fn server_launch_plan_rejects_qwen3_coder_next_public_snapshot() {
        let error = server_launch_plan(
            "mlx-community/Qwen3-Coder-Next-4bit",
            "/models/qwen3-coder-next",
            8080,
            ManagerEngine::AxEngineNgram,
        )
        .expect_err("public qwen3 coder next snapshot should fail closed");

        let message = error.to_string();
        assert!(message.contains("sanitized Qwen3 Next"));
        assert!(message.contains("mlx_lm.convert"));
    }

    #[test]
    fn server_launch_plan_rejects_qwen3_coder_next_manual_path() {
        let error = server_launch_plan(
            "local",
            "/models/Qwen3_Coder_Next_4bit",
            8080,
            ManagerEngine::AxEngine,
        )
        .expect_err("manual qwen3 coder next paths should fail closed");

        assert!(error.to_string().contains("sanitized Qwen3 Next"));
    }

    #[test]
    fn server_launch_plan_respects_direct_ax_engine_selection() {
        let repo = "mlx-community/Qwen3-4B-4bit";
        let plan = server_launch_plan(repo, "/models/qwen", 8080, ManagerEngine::AxEngine)
            .expect("qwen should use generic mlx route");

        assert!(
            plan.args
                .iter()
                .any(|arg| arg == "--disable-ngram-acceleration"),
            "plain ax-engine selection should disable the accelerated n-gram path"
        );
    }

    #[test]
    fn server_launch_plan_respects_ngram_engine_selection() {
        let repo = "mlx-community/Qwen3-4B-4bit";
        let plan = server_launch_plan(repo, "/models/qwen", 8080, ManagerEngine::AxEngineNgram)
            .expect("qwen should use generic mlx route");

        assert!(
            !plan
                .args
                .iter()
                .any(|arg| arg == "--disable-ngram-acceleration"),
            "ax-engine-ngram selection should leave n-gram acceleration enabled"
        );
    }

    #[test]
    fn manager_engine_defaults_to_ngram_selection() {
        assert_eq!(
            ManagerEngine::parse(None).expect("default engine should parse"),
            ManagerEngine::AxEngineNgram
        );
    }

    #[test]
    fn manager_engine_rejects_unmanaged_launcher_choices() {
        let error = ManagerEngine::parse(Some("mlx-lm")).expect_err("mlx-lm should fail closed");

        assert!(error.to_string().contains("not startable yet"));
    }

    #[test]
    fn parses_http_request_with_json_body() {
        let bytes =
            b"POST /api/download HTTP/1.1\r\ncontent-length: 18\r\n\r\n{\"repo_id\":\"qwen\"}";
        let request = parse_http_request_bytes(bytes).expect("request should parse");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/download");
        assert_eq!(request.body, "{\"repo_id\":\"qwen\"}");
    }

    #[test]
    fn parse_http_request_rejects_incomplete_body_without_panicking() {
        let bytes = b"POST /api/download HTTP/1.1\r\ncontent-length: 18\r\n\r\n{\"repo_id\"";

        let error = parse_http_request_bytes(bytes).expect_err("body is incomplete");

        assert!(error.to_string().contains("incomplete HTTP request body"));
    }

    #[test]
    fn request_complete_rejects_overflowing_content_length() {
        let bytes =
            b"POST /api/download HTTP/1.1\r\ncontent-length: 18446744073709551615\r\n\r\n{}";

        assert!(!request_complete(bytes));
        let error = parse_http_request_bytes(bytes).expect_err("content length should overflow");
        assert_eq!(error.to_string(), "content-length is too large");
    }

    #[test]
    fn parses_system_metric_sources() {
        let vm_stat = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
            Pages free: 100.\n\
            Pages speculative: 50.\n\
            Pages active: 20.\n\
            Pages wired down: 10.\n\
            Pages occupied by compressor: 5.\n";
        let ram = ram_utilization_percent_from_vm_stat(vm_stat, 1_638_400)
            .expect("ram percent should parse");
        assert_eq!(ram, 35.0);

        let ioreg =
            r#""PerformanceStatistics" = {"Device Utilization %"=45,"Renderer Utilization %"=44}"#;
        assert_eq!(gpu_utilization_percent_from_ioreg(ioreg), Some(45.0));
        assert_eq!(
            parse_number_after_marker("\"Device Utilization %\" = 12", "\"Device Utilization %\""),
            Some(12.0)
        );
    }

    #[test]
    fn system_metrics_json_has_cpu_gpu_ram_shape() {
        let metrics = system_metrics_json();

        assert!(metrics["timestamp_ms"].as_u64().is_some());
        assert_eq!(metrics["refresh_interval_ms"], 3000);
        assert!(metrics["cpu"].get("available").is_some());
        assert!(metrics["gpu"].get("available").is_some());
        assert!(metrics["ram"].get("available").is_some());
    }

    #[test]
    fn state_json_uses_cached_hf_cache_models() {
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.hf_cache_models = Some(vec![CachedModelSnapshot {
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                path: "/cached/qwen".to_string(),
            }]);
        }

        let state = runtime_state_json(&runtime).expect("state should render");
        let downloaded = state["downloaded_models"]
            .as_array()
            .expect("downloaded models should be an array");

        assert!(downloaded.iter().any(|model| {
            model["repo_id"] == "mlx-community/Qwen3-4B-4bit" && model["path"] == "/cached/qwen"
        }));
    }

    #[test]
    fn state_json_deduplicates_downloaded_models_by_path() {
        let mut app_state = AppState::empty();
        app_state.doctor = LoadState::Ready(doctor_with_model_path("/cached/qwen"));
        let runtime = Arc::new(Mutex::new(WebRuntime::new(app_state)));
        {
            let mut guard = runtime.lock().expect("runtime lock should work");
            guard.hf_cache_models = Some(vec![CachedModelSnapshot {
                repo_id: "mlx-community/Qwen3-4B-4bit".to_string(),
                path: "/cached/qwen".to_string(),
            }]);
        }

        let state = runtime_state_json(&runtime).expect("state should render");
        let downloaded = state["downloaded_models"]
            .as_array()
            .expect("downloaded models should be an array");

        assert_eq!(
            downloaded
                .iter()
                .filter(|model| model["path"] == "/cached/qwen")
                .count(),
            1
        );
    }

    #[test]
    fn state_json_includes_catalog_and_endpoints() {
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        let state = runtime_state_json(&runtime).expect("state should render");
        let endpoints = state["server"]["endpoints"]
            .as_array()
            .expect("endpoints array");

        assert!(state["catalog"].as_array().expect("catalog array").len() > 3);
        assert_eq!(state["server"]["port"], 8080);
        assert!(
            endpoints
                .iter()
                .any(|endpoint| endpoint["url"] == "http://127.0.0.1:8080/health")
        );
        assert!(
            endpoints
                .iter()
                .any(|endpoint| endpoint["url"] == "http://127.0.0.1:8080/v1/chat/completions")
        );
        assert!(
            !endpoints
                .iter()
                .any(|endpoint| endpoint["url"] == "http://127.0.0.1:8080/v1/embeddings"),
            "manager should not advertise embedding endpoints in the text LLM workflow"
        );
    }

    fn doctor_with_model_path(path: &str) -> DoctorReport {
        serde_json::from_value(json!({
            "schema_version": "ax.engine_bench.doctor.v1",
            "status": "ready",
            "mlx_runtime_ready": true,
            "bringup_allowed": true,
            "workflow": {
                "mode": "source_checkout",
                "cwd": "/repo",
                "source_root": "/repo",
                "doctor": {"argv": ["doctor"], "cwd": null},
                "server": {"argv": ["server"], "cwd": null},
                "generate_manifest": {"argv": ["generate-manifest"], "cwd": null},
                "benchmark": {"argv": ["scenario"], "cwd": null},
                "download_model": {"argv": ["python3", "scripts/download_model.py", "<repo-id>", "--json"], "cwd": "/repo"}
            },
            "model_artifacts": {
                "selected": true,
                "status": "ready",
                "path": path,
                "exists": true,
                "is_dir": true,
                "config_present": true,
                "manifest_present": true,
                "safetensors_present": true,
                "model_type": "qwen3",
                "quantization": null,
                "issues": []
            },
            "issues": [],
            "notes": [],
            "performance_advice": []
        }))
        .expect("doctor report should deserialize")
    }
}
