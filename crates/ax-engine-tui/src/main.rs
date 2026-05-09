use ax_engine_tui::action::{Action, reduce};
use ax_engine_tui::app::{
    AppState, AppTab, LoadState, ModelFamily, ModelSelectorTarget, ServerControlState,
};
use ax_engine_tui::contracts::{
    WorkflowCommand, read_benchmark_artifact_json, read_doctor_json, scan_artifacts,
};
use ax_engine_tui::jobs::plan::{
    CommandInvocation, EvidenceClass, JobDisplaySummary, JobKind, JobPlan, JobSpec,
};
use ax_engine_tui::jobs::runner::{JobOutput, JobStatus, RunningJob, run_to_completion};
use ax_engine_tui::jobs::{DoctorCommand, fetch_server_snapshot, run_doctor};
use ax_engine_tui::profiles::{ManagerProfile, default_profile_root, read_profile, write_profile};
use ax_engine_tui::support::write_support_bundle;
use ax_engine_tui::ui;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, MouseButton, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
enum ManagerError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
    run_terminal(state)
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
         Usage: ax-engine-manager [--check] [--phase2-check] [--doctor-json <path>] [--model-dir <path>] \\\n           [--server-url <url>] [--benchmark-json <path>] [--artifact-root <path>] [--profile-dir <path>] \\\n           [--support-bundle <dir>]\n\n\
         Check mode is read-only. It may run doctor, read JSON contracts, poll server metadata,\n\
         and render local artifacts without starting downloads, benchmarks, or servers.\n\
         Phase 2 adds guarded local job planning, fake-process cancellation checks, and profiles.\n\
         Phase 3 support bundles write redacted diagnostics without model weights or secrets.\n\
         The Models tab can run the guarded download helper for the selected repo id."
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
        if let Some(control) = ServerControlState::from_base_url(server_url) {
            state.server_control = control;
        }
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

fn run_terminal(mut state: AppState) -> Result<(), ManagerError> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_loop(&mut terminal, &mut state);
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    result
}

fn run_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
) -> Result<(), ManagerError> {
    loop {
        terminal.draw(|frame| ui::render(frame, state))?;
        if state.should_quit {
            return Ok(());
        }
        if !event::poll(Duration::from_millis(250))? {
            continue;
        }
        let size = terminal.size()?;
        let area = Rect::new(0, 0, size.width, size.height);
        if let Some(action) = action_for_event(event::read()?, area, state) {
            handle_action(terminal, state, action)?;
        }
    }
}

fn handle_action<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
    action: Action,
) -> Result<(), ManagerError> {
    if action == Action::DownloadSelectedModel {
        run_selected_model_download(terminal, state)?;
    } else {
        reduce(state, action);
    }
    Ok(())
}

fn action_for_event(event: Event, area: Rect, state: &AppState) -> Option<Action> {
    match event {
        Event::Key(key) => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
            KeyCode::Tab | KeyCode::Right => Some(Action::NextTab),
            KeyCode::BackTab | KeyCode::Left => Some(Action::PreviousTab),
            KeyCode::Char('f') if state.selected_tab == AppTab::Models => Some(
                Action::SelectModelFamily(next_model_family(state.model_download.family)),
            ),
            KeyCode::Char('s') if state.selected_tab == AppTab::Models => {
                Some(Action::SelectModelSize(next_model_size(state)))
            }
            KeyCode::Char('d') | KeyCode::Enter if state.selected_tab == AppTab::Models => {
                Some(Action::DownloadSelectedModel)
            }
            _ => None,
        },
        Event::Mouse(mouse) if mouse.kind == MouseEventKind::Down(MouseButton::Left) => {
            if let Some(tab) = ui::tab_at_position(area, mouse.column, mouse.row) {
                return Some(Action::SelectTab(tab));
            }
            if state.selected_tab == AppTab::Models {
                return ui::model_selector_at_position(area, state, mouse.column, mouse.row).map(
                    |target| match target {
                        ModelSelectorTarget::Family(family) => Action::SelectModelFamily(family),
                        ModelSelectorTarget::Size(index) => Action::SelectModelSize(index),
                        ModelSelectorTarget::Download => Action::DownloadSelectedModel,
                    },
                );
            }
            if state.selected_tab == AppTab::Server {
                return ui::server_control_at_position(area, mouse.column, mouse.row)
                    .map(Action::SelectServerControl);
            }
            None
        }
        _ => None,
    }
}

fn next_model_family(current: ModelFamily) -> ModelFamily {
    let index = ModelFamily::ALL
        .iter()
        .position(|family| *family == current)
        .unwrap_or(0);
    ModelFamily::ALL[(index + 1) % ModelFamily::ALL.len()]
}

fn next_model_size(state: &AppState) -> usize {
    let count = state.model_download.entries().len().max(1);
    (state.model_download.size_index + 1) % count
}

fn run_selected_model_download<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    state: &mut AppState,
) -> Result<(), ManagerError> {
    state.mark_model_download_running();
    terminal.draw(|frame| ui::render(frame, state))?;

    let repo_id = state.model_download.selected_entry().repo_id.to_string();
    let expected_dest = default_model_destination(&repo_id);
    let spec = model_download_job_spec(state, &repo_id)?;
    match run_to_completion(spec) {
        Ok(output) if output.status == JobStatus::Succeeded => {
            let model_dir = model_dir_from_download_output(&output).unwrap_or(expected_dest);
            state.mark_model_download_succeeded(model_dir.clone());
            refresh_doctor_after_download(state, Path::new(&model_dir));
        }
        Ok(output) => {
            state.mark_model_download_failed(download_failure_message(&output));
        }
        Err(error) => {
            state.mark_model_download_failed(error.to_string());
        }
    }
    Ok(())
}

fn model_download_job_spec(state: &AppState, repo_id: &str) -> Result<JobSpec, ManagerError> {
    if let LoadState::Ready(doctor) = &state.doctor
        && let Some(command) = doctor.workflow.download_model.as_ref()
    {
        return Ok(JobSpec::new(
            "download-model",
            "Download selected model",
            JobKind::DownloadModel,
            EvidenceClass::Readiness,
            download_invocation_from_workflow(command, repo_id)?,
        ));
    }

    let cwd = env::current_dir()?;
    let root = source_checkout_root(&cwd).ok_or_else(|| {
        ManagerError::Message(
            "download requires a source checkout or doctor workflow with download_model"
                .to_string(),
        )
    })?;
    Ok(JobSpec::new(
        "download-model",
        "Download selected model",
        JobKind::DownloadModel,
        EvidenceClass::Readiness,
        CommandInvocation::new(
            "python3",
            vec![
                "scripts/download_model.py".to_string(),
                repo_id.to_string(),
                "--json".to_string(),
            ],
            Some(root),
        ),
    ))
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

fn default_model_destination(repo_id: &str) -> String {
    let slug = repo_id.replace('/', "--");
    match env::var("HOME") {
        Ok(home) => format!("{home}/.cache/ax-engine/models/{slug}"),
        Err(_) => format!("~/.cache/ax-engine/models/{slug}"),
    }
}

fn model_dir_from_download_output(output: &JobOutput) -> Option<String> {
    let json = output
        .log_tail
        .iter()
        .filter_map(|line| line.strip_prefix("stdout: "))
        .collect::<Vec<_>>()
        .join("\n");
    serde_json::from_str::<serde_json::Value>(&json)
        .ok()
        .and_then(|value| value.get("dest")?.as_str().map(str::to_string))
}

fn download_failure_message(output: &JobOutput) -> String {
    let tail = output
        .log_tail
        .iter()
        .rev()
        .take(4)
        .cloned()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>()
        .join(" | ");
    if tail.is_empty() {
        format!("job exited with {:?}", output.exit_code)
    } else {
        tail
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_accepts_phase1_read_only_inputs() {
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
    }

    #[test]
    fn check_summary_reports_all_phase1_surfaces() {
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
    fn left_mouse_click_selects_header_tab() {
        let area = Rect::new(0, 0, 80, 24);
        let state = AppState::empty();
        let event = Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 30,
            row: 1,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });

        assert_eq!(
            action_for_event(event, area, &state),
            Some(Action::SelectTab(ax_engine_tui::AppTab::Benchmarks))
        );
    }

    #[test]
    fn mouse_click_outside_header_has_no_action() {
        let area = Rect::new(0, 0, 80, 24);
        let state = AppState::empty();
        let event = Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 30,
            row: 4,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });

        assert_eq!(action_for_event(event, area, &state), None);
    }

    #[test]
    fn model_keyboard_shortcuts_select_and_download() {
        let area = Rect::new(0, 0, 100, 32);
        let mut state = AppState::empty();
        state.selected_tab = ax_engine_tui::AppTab::Models;

        let family = Event::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('f'),
            crossterm::event::KeyModifiers::empty(),
        ));
        assert_eq!(
            action_for_event(family, area, &state),
            Some(Action::SelectModelFamily(ModelFamily::Gemma))
        );

        let size = Event::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('s'),
            crossterm::event::KeyModifiers::empty(),
        ));
        assert_eq!(
            action_for_event(size, area, &state),
            Some(Action::SelectModelSize(1))
        );

        let download = Event::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('d'),
            crossterm::event::KeyModifiers::empty(),
        ));
        assert_eq!(
            action_for_event(download, area, &state),
            Some(Action::DownloadSelectedModel)
        );
    }

    #[test]
    fn left_mouse_click_selects_model_size_and_download() {
        let area = Rect::new(0, 0, 100, 32);
        let mut state = AppState::empty();
        state.selected_tab = ax_engine_tui::AppTab::Models;

        let size = Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 4,
            row: 9,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });
        assert_eq!(
            action_for_event(size, area, &state),
            Some(Action::SelectModelSize(1))
        );

        let download = Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 4,
            row: 18,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });
        assert_eq!(
            action_for_event(download, area, &state),
            Some(Action::DownloadSelectedModel)
        );
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
    fn parses_model_dir_from_download_json_log_tail() {
        let output = JobOutput {
            job_id: "download-model".to_string(),
            status: JobStatus::Succeeded,
            exit_code: Some(0),
            log_tail: vec![
                "stdout: {".to_string(),
                "stdout:   \"dest\": \"/models/qwen\",".to_string(),
                "stdout:   \"status\": \"ready\"".to_string(),
                "stdout: }".to_string(),
            ],
        };

        assert_eq!(
            model_dir_from_download_output(&output).as_deref(),
            Some("/models/qwen")
        );
    }

    #[test]
    fn left_mouse_click_selects_server_button_when_server_tab_is_active() {
        let area = Rect::new(0, 0, 100, 32);
        let mut state = AppState::empty();
        state.selected_tab = ax_engine_tui::AppTab::Server;
        let event = Event::Mouse(crossterm::event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: 11,
            row: 7,
            modifiers: crossterm::event::KeyModifiers::empty(),
        });

        assert_eq!(
            action_for_event(event, area, &state),
            Some(Action::SelectServerControl(
                ax_engine_tui::app::ServerControlSelection::Button(
                    ax_engine_tui::app::ServerControlButton::Start
                )
            ))
        );
    }
}
