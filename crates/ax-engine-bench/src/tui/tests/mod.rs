//! TUI state-machine and rendering tests. Test-only module: unwrap/expect/panic
//! are fine per the workspace lint convention (see [workspace.lints.clippy]).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod catalog;
mod chat;
mod downloads_serve;
mod jobs;
mod misc;
mod navigation;
mod ux;
mod wizard;

use super::hardware::HardwareInfo;
use super::jobs::{DownloadTask, Job};
use super::{App, Screen};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::event::{MouseEvent, MouseEventKind};
use std::path::PathBuf;

fn new_app() -> App {
    // Hermetic fixture: App::with_hardware_for_tests builds the catalog
    // shape without touching the real HF cache on disk, so this never
    // depends on (or pays the cost of scanning) whatever models happen to
    // be installed on the developer machine. Tests that need install state
    // set it explicitly.
    App::with_hardware_for_tests(HardwareInfo::for_tests())
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: ratatui::crossterm::event::KeyEventState::empty(),
    }
}

fn ctrl_key(c: char) -> KeyEvent {
    KeyEvent {
        code: KeyCode::Char(c),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: ratatui::crossterm::event::KeyEventState::empty(),
    }
}

fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::empty(),
    }
}

/// Render the app to an off-screen buffer and flatten it to text.
fn render(app: &App) -> String {
    render_sized(app, 120, 40)
}

fn render_sized(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

fn family_index(app: &App, key: &str) -> usize {
    app.families.iter().position(|f| f.key == key).unwrap()
}

fn test_task(job: Option<Job>) -> DownloadTask {
    DownloadTask {
        label: "gemma4-e2b 4-bit".into(),
        repo_id: "mlx-community/gemma-4-e2b-it-4bit".into(),
        preset: Some("gemma4-e2b".to_string()),
        target: "gemma4-e2b".into(),
        dest: Some(PathBuf::from("/tmp/gemma4-e2b")),
        watch_dir: PathBuf::from("/tmp/gemma4-e2b"),
        resolved_path: None,
        total_bytes: Some(3_583_088_661),
        phase: None,
        job,
        cancelled: false,
    }
}

/// App with a "ready" server (points at the discard port; spawned curls fail
/// fast with connection-refused rather than talking to anything real).
fn chat_ready_app() -> App {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:9".into());
    app.screen = Screen::Chat;
    app
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
}
