use ax_engine_tui::app::{AppState, LoadState};
use ax_engine_tui::contracts::{
    WorkflowCommand, read_benchmark_artifact_json, read_doctor_json, scan_artifacts,
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
use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
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
    if options.check {
        print_check_summary(&state);
        return Ok(());
    }
    run_web_manager(state, &options)
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
         The web manager provides model type/family/size selection, guarded downloads, server port controls,\n\
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
}

struct ManagedServer {
    child: Child,
    port: u16,
    model_dir: String,
}

struct WebRuntime {
    state: AppState,
    next_job_id: u64,
    download_jobs: Vec<DownloadJobSnapshot>,
    server: Option<ManagedServer>,
    status_message: String,
}

impl WebRuntime {
    fn new(state: AppState) -> Self {
        Self {
            state,
            next_job_id: 1,
            download_jobs: Vec::new(),
            server: None,
            status_message: "Web manager ready".to_string(),
        }
    }

    fn cleanup_server(&mut self) {
        let Some(server) = self.server.as_mut() else {
            return;
        };
        if let Ok(Some(status)) = server.child.try_wait() {
            self.status_message = format!("Server exited with {status}");
            self.server = None;
        }
    }

    fn server_port(&self) -> u16 {
        self.server
            .as_ref()
            .map(|server| server.port)
            .unwrap_or(self.state.server_control.port)
    }
}

type SharedRuntime = Arc<Mutex<WebRuntime>>;

fn run_web_manager(state: AppState, options: &Options) -> Result<(), ManagerError> {
    let address = format!("{}:{}", options.web_host, options.web_port);
    let listener = TcpListener::bind(&address)?;
    let runtime = Arc::new(Mutex::new(WebRuntime::new(state)));
    let url = format!("http://{address}");
    println!("ax-engine-manager web={url}");
    if !options.no_open {
        open_browser(&url);
    }
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let runtime = Arc::clone(&runtime);
                std::thread::spawn(move || {
                    if let Err(error) = handle_client(stream, runtime) {
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

/// Forward a chat-completions request to the running inference server and
/// stream the response (SSE or JSON) back to the browser client.
fn proxy_chat(
    client: &mut TcpStream,
    runtime: &SharedRuntime,
    body: &str,
) -> Result<(), ManagerError> {
    use std::io::Write;

    let port = runtime
        .lock()
        .map_err(|_| ManagerError::Message("runtime lock poisoned".to_string()))?
        .server_port();

    // Connect to inference server.
    let mut upstream =
        match std::net::TcpStream::connect(format!("127.0.0.1:{port}")) {
            Ok(s) => s,
            Err(e) => {
                let err_body = format!(r#"{{"error":"server not reachable: {e}"}}"#);
                let resp = format!(
                    "HTTP/1.1 503 Service Unavailable\r\n\
                     content-type: application/json\r\n\
                     content-length: {}\r\n\
                     connection: close\r\n\r\n{}",
                    err_body.len(),
                    err_body
                );
                client.write_all(resp.as_bytes())?;
                return Ok(());
            }
        };
    upstream.set_read_timeout(Some(Duration::from_secs(120)))?;

    // Forward request to upstream.
    let forward = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\r\n{}",
        body.len(),
        body
    );
    upstream.write_all(forward.as_bytes())?;

    // Read upstream response headers.
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
            return Err(ManagerError::Message("upstream headers too large".to_string()));
        }
    }

    let header_end = find_header_end(&buf).unwrap_or(buf.len());
    let header_str = String::from_utf8_lossy(&buf[..header_end]);
    let status_code = header_str
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .unwrap_or("200");
    let content_type = if header_str.to_ascii_lowercase().contains("text/event-stream") {
        "text/event-stream"
    } else {
        "application/json"
    };

    // Write response headers to browser client.
    let client_headers = format!(
        "HTTP/1.1 {status_code} OK\r\n\
         content-type: {content_type}\r\n\
         cache-control: no-cache\r\n\
         connection: close\r\n\r\n"
    );
    client.write_all(client_headers.as_bytes())?;

    // Forward body already buffered from upstream.
    let body_start = header_end + 4;
    if body_start < buf.len() {
        client.write_all(&buf[body_start..])?;
        client.flush()?;
    }

    // Stream remaining upstream data to client.
    loop {
        match upstream.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => {
                if client.write_all(&tmp[..n]).is_err() {
                    break;
                }
                client.flush().ok();
            }
            Err(_) => break,
        }
    }

    Ok(())
}

fn route_request(request: HttpRequest, runtime: SharedRuntime) -> Result<String, ManagerError> {
    let path = request.path.split('?').next().unwrap_or("/");
    Ok(match (request.method.as_str(), path) {
        ("GET", "/") | ("GET", "/index.html") => {
            http_response("200 OK", "text/html; charset=utf-8", web::index_html())
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

fn runtime_state_json(runtime: &SharedRuntime) -> Result<Value, ManagerError> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    runtime.cleanup_server();
    let port = runtime.server_port();
    let base_url = format!("http://127.0.0.1:{port}");
    let server_status = match runtime.server.as_ref() {
        Some(server) => format!("Running on port {} ({})", server.port, server.model_dir),
        None => "Stopped".to_string(),
    };
    let selected_repo_id = runtime.state.model_download.selected_entry().repo_id;
    let download_status = runtime
        .download_jobs
        .last()
        .map(|job| format!("{} {}", job.status, job.repo_id))
        .unwrap_or_else(|| "Idle".to_string());
    let downloaded_models: Vec<Value> = {
        let mut seen = std::collections::HashSet::new();
        let mut models: Vec<Value> = Vec::new();
        // From successful download jobs this session.
        for job in runtime.download_jobs.iter().rev() {
            if job.status == "succeeded" {
                if let Some(ref path) = job.model_dir {
                    if seen.insert(job.repo_id.clone()) {
                        models.push(json!({ "repo_id": job.repo_id, "path": path }));
                    }
                }
            }
        }
        // From the doctor report (pre-existing model).
        if let Some(path) = web::current_model_dir(&runtime.state) {
            if let Some(repo) = &runtime
                .download_jobs
                .iter()
                .find(|j| j.model_dir.as_deref() == Some(path.as_str()))
                .map(|j| j.repo_id.clone())
            {
                let _ = repo; // already covered above
            } else {
                models.push(json!({ "repo_id": "local", "path": path }));
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
            "running": runtime.server.is_some(),
            "port": port,
            "base_url": base_url,
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
        .ok_or_else(|| ManagerError::Message("repo_id is required".to_string()))?
        .to_string();

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
            message: None,
        });
        runtime.status_message = format!("Downloading {repo_id}");
        (job_id, invocation)
    };

    let runtime_for_job = Arc::clone(runtime);
    let job_id_for_thread = job_id.clone();
    std::thread::spawn(move || {
        let result = run_download_invocation(&invocation);
        finish_download_job(&runtime_for_job, &job_id_for_thread, result);
    });

    Ok(json!({"id": job_id, "status": "running"}))
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
            runtime.status_message = format!("Downloaded {}", runtime.download_jobs[index].repo_id);
            refresh_doctor_after_download(&mut runtime.state, Path::new(&model_dir));
        }
        Err(error) => {
            runtime.download_jobs[index].status = "failed".to_string();
            runtime.download_jobs[index].message = Some(error.to_string());
            runtime.status_message = format!(
                "Download failed for {}",
                runtime.download_jobs[index].repo_id
            );
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

    let mut runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    stop_server_locked(&mut runtime);
    let model_dir = requested_model_dir
        .or_else(|| latest_model_dir(&runtime))
        .or_else(|| web::current_model_dir(&runtime.state))
        .ok_or_else(|| {
            ManagerError::Message("model_dir is required to start server".to_string())
        })?;
    let child = Command::new("ax-engine-server")
        .arg("--mlx")
        .arg("--mlx-model-artifacts-dir")
        .arg(&model_dir)
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    runtime.server = Some(ManagedServer {
        child,
        port: port as u16,
        model_dir: model_dir.clone(),
    });
    runtime.state.server_control.port = port as u16;
    runtime.status_message = format!("Server starting on port {port}");
    Ok(json!({"status": "starting", "port": port, "model_dir": model_dir}))
}

fn stop_server(runtime: &SharedRuntime) -> Result<Value, ManagerError> {
    let mut runtime = runtime
        .lock()
        .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
    let stopped = stop_server_locked(&mut runtime);
    runtime.status_message = if stopped {
        "Server stopped".to_string()
    } else {
        "Server was not running".to_string()
    };
    Ok(json!({"status": runtime.status_message}))
}

fn restart_server(runtime: &SharedRuntime, body: &str) -> Result<Value, ManagerError> {
    {
        let mut rt = runtime
            .lock()
            .map_err(|_| ManagerError::Message("web runtime lock poisoned".to_string()))?;
        stop_server_locked(&mut rt);
        rt.status_message = "Server restarting…".to_string();
    }
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
    Ok(CommandInvocation::new(
        program.clone(),
        args.iter()
            .map(|arg| {
                if arg == "<repo-id>" {
                    repo_id.to_string()
                } else {
                    arg.clone()
                }
            })
            .collect(),
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
import json
import subprocess
import sys
from pathlib import Path

IGNORE_PATTERNS = ["*.bin", "*.pt", "*.gguf", "*.msgpack", "flax_model*"]
MANIFEST = "model-manifest.json"
repo_id = sys.argv[1]
bench = sys.argv[2] if len(sys.argv) > 2 else "ax-engine-bench"

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
        "server_command": [
            "ax-engine-server",
            "--mlx",
            "--mlx-model-artifacts-dir",
            str(dest),
            "--port",
            "8080",
        ],
    }

def print_summary(dest, status, errors=None):
    print(json.dumps(summary(dest, status, errors), sort_keys=True))

try:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
except Exception as exc:
    print_summary(
        Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo_id.replace("/", "--")),
        "download_failed",
        ["huggingface_hub is required for installed manager downloads. Run: python3 -m pip install huggingface_hub", str(exc)],
    )
    raise SystemExit(1)

try:
    dest = Path(snapshot_download(repo_id=repo_id, ignore_patterns=IGNORE_PATTERNS))
except Exception as exc:
    print_summary(
        Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo_id.replace("/", "--")),
        "download_failed",
        [str(exc)],
    )
    raise SystemExit(1)

errors = []
if not list(dest.glob("*.safetensors")):
    errors.append(f"no .safetensors files found in {dest}")
if not (dest / "config.json").exists():
    errors.append(f"config.json missing in {dest}")
if errors:
    print_summary(dest, "invalid", errors)
    raise SystemExit(1)

if not (dest / MANIFEST).exists():
    result = subprocess.run(
        [bench, "generate-manifest", str(dest), "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or f"{bench} exited with {result.returncode}"
        print_summary(dest, "manifest_missing", [error])
        raise SystemExit(1)

print_summary(dest, "ready")
"#;

fn run_download_invocation(invocation: &CommandInvocation) -> Result<String, DownloadRunError> {
    let mut command = Command::new(&invocation.program);
    command.args(&invocation.args);
    if let Some(cwd) = invocation.cwd.as_deref() {
        command.current_dir(cwd);
    }
    let output = command.output().map_err(DownloadRunError::Io)?;
    parse_download_output(output)
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

fn refresh_doctor_after_download(state: &mut AppState, model_dir: &Path) {
    match env::current_dir() {
        Ok(cwd) => {
            let command = DoctorCommand::from_cwd(&cwd, Some(model_dir));
            state.doctor = run_doctor(&command)
                .map(LoadState::Ready)
                .unwrap_or_else(|error| LoadState::unavailable(error.to_string()));
        }
        Err(error) => {
            state.doctor = LoadState::unavailable(format!("failed to refresh doctor: {error}"));
        }
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
    bytes.len() >= header_end + 4 + content_length
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
    let body = String::from_utf8_lossy(&bytes[body_start..body_start + length]).to_string();
    Ok(HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        body,
    })
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
                "--json"
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
                "--json"
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
        assert!(invocation.args[1].contains("snapshot_download"));
        assert_eq!(invocation.args[2], "mlx-community/Qwen3-4B-4bit");
        assert!(invocation.args[3].contains("ax-engine-bench"));
        assert_eq!(invocation.cwd, None);
    }

    #[test]
    fn parses_ready_download_output() {
        let output = Output {
            status: std::process::ExitStatus::from_raw(0),
            stdout: br#"{"status":"ready","dest":"/models/qwen","errors":[]}"#.to_vec(),
            stderr: Vec::new(),
        };

        assert_eq!(
            parse_download_output(output).expect("download should parse"),
            "/models/qwen"
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
    fn parses_http_request_with_json_body() {
        let bytes =
            b"POST /api/download HTTP/1.1\r\ncontent-length: 18\r\n\r\n{\"repo_id\":\"qwen\"}";
        let request = parse_http_request_bytes(bytes).expect("request should parse");

        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/api/download");
        assert_eq!(request.body, "{\"repo_id\":\"qwen\"}");
    }

    #[test]
    fn state_json_includes_catalog_and_endpoints() {
        let runtime = Arc::new(Mutex::new(WebRuntime::new(AppState::empty())));
        let state = runtime_state_json(&runtime).expect("state should render");

        assert!(state["catalog"].as_array().expect("catalog array").len() > 3);
        assert_eq!(state["server"]["port"], 8080);
        assert!(
            state["server"]["endpoints"]
                .as_array()
                .expect("endpoints array")
                .iter()
                .any(|endpoint| endpoint["url"] == "http://127.0.0.1:8080/health")
        );
    }
}
