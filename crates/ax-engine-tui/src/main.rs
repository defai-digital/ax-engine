use ax_engine_tui::action::{Action, reduce};
use ax_engine_tui::app::{AppState, LoadState};
use ax_engine_tui::contracts::{read_benchmark_artifact_json, read_doctor_json, scan_artifacts};
use ax_engine_tui::jobs::{DoctorCommand, fetch_server_snapshot, run_doctor};
use ax_engine_tui::ui;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use std::env;
use std::io;
use std::path::PathBuf;
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
    doctor_json: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    server_url: Option<String>,
    benchmark_json: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
}

fn main() -> Result<(), ManagerError> {
    let options = parse_args(env::args().skip(1))?;
    let state = build_state(&options);
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
            "--doctor-json" => options.doctor_json = Some(next_path(&mut args, "--doctor-json")?),
            "--model-dir" => options.model_dir = Some(next_path(&mut args, "--model-dir")?),
            "--server-url" => options.server_url = Some(next_value(&mut args, "--server-url")?),
            "--benchmark-json" => {
                options.benchmark_json = Some(next_path(&mut args, "--benchmark-json")?)
            }
            "--artifact-root" => {
                options.artifact_root = Some(next_path(&mut args, "--artifact-root")?)
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
         Usage: ax-engine-manager [--check] [--doctor-json <path>] [--model-dir <path>] \\\n           [--server-url <url>] [--benchmark-json <path>] [--artifact-root <path>]\n\n\
         Phase 1 is read-only. It may run doctor, read JSON contracts, poll server metadata,\n\
         and render local artifacts, but it does not start downloads, benchmarks, or servers."
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
        state.server.base_url = Some(server_url.clone());
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
    println!("ax-engine-manager check");
    match &state.doctor {
        LoadState::Ready(report) => {
            println!("doctor=ready status={}", report.status);
            println!("workflow={}", report.workflow.mode);
            println!("model_artifacts={}", report.model_artifacts.status);
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            println!("doctor=unavailable reason={message}");
        }
    }
    match &state.server.health {
        LoadState::Ready(health) => println!("server=ready status={}", health.status),
        LoadState::Unavailable(message) => println!("server=unavailable reason={message}"),
        LoadState::NotLoaded(message) => println!("server=not_loaded reason={message}"),
    }
}

fn run_terminal(mut state: AppState) -> Result<(), ManagerError> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_loop(&mut terminal, &mut state);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
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
        if let Event::Key(key) = event::read()? {
            let action = match key.code {
                KeyCode::Char('q') | KeyCode::Esc => Some(Action::Quit),
                KeyCode::Tab | KeyCode::Right => Some(Action::NextTab),
                KeyCode::BackTab | KeyCode::Left => Some(Action::PreviousTab),
                _ => None,
            };
            if let Some(action) = action {
                reduce(state, action);
            }
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
                "--doctor-json",
                "doctor.json",
                "--server-url",
                "http://127.0.0.1:8080",
                "--artifact-root",
                "benchmarks/results",
            ]
            .into_iter()
            .map(str::to_string),
        )
        .expect("args should parse");

        assert!(options.check);
        assert_eq!(options.doctor_json, Some(PathBuf::from("doctor.json")));
        assert_eq!(options.server_url.as_deref(), Some("http://127.0.0.1:8080"));
        assert_eq!(
            options.artifact_root,
            Some(PathBuf::from("benchmarks/results"))
        );
    }
}
