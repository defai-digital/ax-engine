//! ratatui terminal UI for `ax-engine tui`.
//!
//! A guided launcher over the existing CLI subcommands: pick a model in a
//! four-step wizard (family -> precision -> options -> confirm), watch the
//! download queue with real progress, serve an installed model, and talk to it
//! on the Chat screen.  All work runs as background child processes
//! (`ax-engine download`, `ax-engine-server`, `curl`) streamed into the UI, so
//! browsing never blocks and quitting is an explicit, confirmed action while
//! jobs are live.
//!
//! It is a child module of the `ax-engine` binary, so it reuses that binary's
//! private catalog and helpers directly via `crate::` (`MODEL_PROFILES`,
//! `default_hf_cache_root`, `find_executable`, ...).

mod catalog;
mod hardware;
mod jobs;
mod screens;
mod theme;
mod widgets;

#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::ffi::OsString;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};

use catalog::{Family, build_families, installed_variants};
use hardware::HardwareInfo;
use jobs::{DownloadMode, DownloadTask, Job};
use screens::chat::ChatState;
use widgets::{DirectoryPicker, Toast};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn cmd_tui(args: &[OsString]) -> Result<u8, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "ax-engine tui — guided model downloader, server launcher, and chat.\n\n\
             Screens: 1 Home · 2 Models · 3 Downloads · 4 Serve · 5 Chat.\n\
             Home shows your hardware and a Quick start action.  The Models\n\
             wizard walks family -> precision (with size and RAM-fit info) ->\n\
             optional MTP speed-up -> a confirm summary before anything is\n\
             downloaded.  Downloads run in a background queue with live\n\
             progress; Serve launches ax-engine-server; Chat streams replies\n\
             from the running server.\n\
             Keys: 1-5 screens · ↑↓ move · Enter select · Esc back · ? help · q quit\n\
             (quitting asks first while downloads or the server are running)."
        );
        return Ok(0);
    }
    if !io::stdout().is_terminal() {
        return Err("ax-engine tui needs an interactive terminal".into());
    }
    let mut terminal = ratatui::init();
    // ratatui::init() does not enable mouse reporting; turn it on so clicks and
    // scroll reach us as Event::Mouse.
    let _ = ratatui::crossterm::execute!(io::stdout(), EnableMouseCapture);
    let result = App::new().run(&mut terminal);
    let _ = ratatui::crossterm::execute!(io::stdout(), DisableMouseCapture);
    ratatui::restore();
    match result {
        Ok(()) => Ok(0),
        Err(err) => Err(format!("tui error: {err}")),
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Screen {
    Home,
    Models,
    Downloads,
    Serve,
    Chat,
}

const SCREENS: [(&str, &str, Screen); 5] = [
    ("⌂", "Home", Screen::Home),
    ("◈", "Models", Screen::Models),
    ("↓", "Downloads", Screen::Downloads),
    ("▶", "Serve", Screen::Serve),
    ("💬", "Chat", Screen::Chat),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WizardStage {
    Families,
    Precision,
    Options,
    Confirm,
}

#[derive(Clone, Copy, PartialEq)]
enum ServeFocus {
    List,
    Host,
    Port,
}

#[derive(Clone, Copy)]
struct PendingDownload {
    family_idx: usize,
    precision_idx: usize,
    with_mtp: bool,
}

/// Overlay that captures all input while open.  Exactly one may be open.
enum Modal {
    Quit {
        downloads: usize,
        server: bool,
    },
    /// From Downloads: a ready item was chosen — serve it?
    ServeReady {
        download_idx: usize,
    },
    /// From the wizard confirm step when the variant is already installed.
    ServeInstalled {
        family_idx: usize,
        variant_idx: usize,
    },
    CancelDownload {
        download_idx: usize,
    },
    /// Destructive: requires typing the word `delete` to arm Enter.
    DeleteModel {
        family_idx: usize,
        variant_idx: usize,
        typed: String,
    },
    StopServer,
    /// Custom destination for the wizard confirm step.
    DestPicker(DirectoryPicker),
}

struct App {
    quit: bool,
    screen: Screen,
    hardware: HardwareInfo,
    families: Vec<Family>,
    modal: Option<Modal>,
    toasts: Vec<Toast>,
    show_help: bool,

    // Home
    home_idx: usize,

    // Models wizard
    stage: WizardStage,
    family_idx: usize,
    precision_idx: usize,
    mtp_idx: usize, // 0 = yes, 1 = no
    pending: Option<PendingDownload>,
    /// Custom destination chosen on the confirm step (None = shared HF cache).
    confirm_dest: Option<PathBuf>,
    /// Case-insensitive substring filter over the family list (`/` to edit).
    filter: String,
    filtering: bool,

    // Downloads
    downloads: Vec<DownloadTask>,
    download_idx: usize,

    // Serve
    serve_focus: ServeFocus,
    serve_idx: usize,
    host: String,
    port: String,
    server: Option<Job>,
    server_url: Option<String>,
    /// Set once the server job's log confirms it actually bound (not just spawned).
    server_ready: bool,
    /// Label of the model the running server was started with (chat request body).
    server_model: Option<String>,

    // Chat
    chat: ChatState,

    // Click-target rects recorded during the last draw (immediate-mode hit-testing).
    sidebar_rect: Cell<Rect>,
    content_list_rect: Cell<Rect>,
    /// Rect of the wizard step header row (for breadcrumb clicks).
    step_header_rect: Cell<Rect>,
}

impl App {
    fn new() -> App {
        App::with_hardware(HardwareInfo::probe())
    }

    fn with_hardware(hardware: HardwareInfo) -> App {
        App {
            quit: false,
            screen: Screen::Home,
            hardware,
            families: build_families(),
            modal: None,
            toasts: Vec::new(),
            show_help: false,
            home_idx: 0,
            stage: WizardStage::Families,
            family_idx: 0,
            precision_idx: 0,
            mtp_idx: 0,
            pending: None,
            confirm_dest: None,
            filter: String::new(),
            filtering: false,
            downloads: Vec::new(),
            download_idx: 0,
            serve_focus: ServeFocus::List,
            serve_idx: 0,
            host: "127.0.0.1".into(),
            port: "8080".into(),
            server: None,
            server_url: None,
            server_ready: false,
            server_model: None,
            chat: ChatState::new(),
            sidebar_rect: Cell::new(Rect::default()),
            content_list_rect: Cell::new(Rect::default()),
            step_header_rect: Cell::new(Rect::default()),
        }
    }

    fn reload_families(&mut self) {
        self.families = build_families();
    }

    fn toast(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::info(text.into()));
    }

    fn toast_success(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::success(text.into()));
    }

    fn toast_warn(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::warning(text.into()));
    }

    fn toast_error(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::error(text.into()));
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.quit {
            terminal.draw(|frame| self.draw(frame))?;
            if event::poll(Duration::from_millis(100))? {
                match event::read()? {
                    Event::Key(key) if key.kind == KeyEventKind::Press => self.on_key(key),
                    Event::Mouse(mouse) => self.on_mouse(mouse),
                    _ => {}
                }
            }
            self.tick();
        }
        for task in &mut self.downloads {
            task.cancel();
        }
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        self.chat.cancel();
        Ok(())
    }

    /// Advance all background jobs and time-based UI state by one poll cycle.
    fn tick(&mut self) {
        let mut finished: Vec<String> = Vec::new();
        for task in &mut self.downloads {
            if task.tick() {
                finished.push(task.label.clone());
            }
        }
        for label in finished {
            self.toast_success(format!("{label} ready"));
            self.reload_families();
        }
        self.start_next_queued_download();
        if let Some(job) = &mut self.server {
            job.tick();
        }
        self.update_server_ready();
        self.chat.tick();
        widgets::expire_toasts(&mut self.toasts);
    }

    // -- input ----------------------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_quit();
            return;
        }
        if self.modal.is_some() {
            self.on_key_modal(key.code);
            return;
        }
        if self.show_help {
            // Any key dismisses help — an overlay that only Esc can close
            // reads as a stuck screen.
            self.show_help = false;
            return;
        }
        if !self.typing() {
            match key.code {
                KeyCode::Char('?') => {
                    self.show_help = true;
                    return;
                }
                KeyCode::Char('q') => {
                    self.request_quit();
                    return;
                }
                KeyCode::Char(c @ '1'..='5') => {
                    self.screen = SCREENS[(c as usize) - ('1' as usize)].2;
                    return;
                }
                _ => {}
            }
        }
        match self.screen {
            Screen::Home => self.on_key_home(key.code),
            Screen::Models => self.on_key_models(key.code),
            Screen::Downloads => self.on_key_downloads(key.code),
            Screen::Serve => self.on_key_serve(key.code),
            Screen::Chat => self.on_key_chat(key),
        }
    }

    /// True while a screen is consuming plain characters (filter, host/port,
    /// chat input), so global single-letter shortcuts must stay inert.
    fn typing(&self) -> bool {
        match self.screen {
            Screen::Models => self.stage == WizardStage::Families && self.filtering,
            Screen::Serve => matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port),
            // Without a ready server, Chat is a static hint screen — capturing
            // keys there would trap the user (no input box is even visible).
            Screen::Chat => self.server_ready,
            _ => false,
        }
    }

    /// `q`/Ctrl-C: quit immediately when idle, confirm when jobs are live.
    fn request_quit(&mut self) {
        let downloads = self
            .downloads
            .iter()
            .filter(|t| t.is_running() || t.is_queued())
            .count();
        let server = self.server.as_ref().is_some_and(Job::is_running);
        if downloads == 0 && !server {
            self.quit = true;
        } else {
            self.modal = Some(Modal::Quit { downloads, server });
        }
    }

    fn on_key_modal(&mut self, code: KeyCode) {
        let Some(modal) = self.modal.take() else {
            return;
        };
        match modal {
            Modal::Quit { .. } => match code {
                KeyCode::Enter | KeyCode::Char('y') => self.quit = true,
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ServeReady { download_idx } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    self.start_server_for_download(download_idx);
                    self.screen = Screen::Serve;
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    self.serve_installed(family_idx, variant_idx);
                    self.screen = Screen::Serve;
                    self.stage = WizardStage::Families;
                    self.pending = None;
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::CancelDownload { download_idx } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    if let Some(task) = self.downloads.get_mut(download_idx) {
                        task.cancel();
                        let label = task.label.clone();
                        self.toast_warn(format!("{label} cancelled"));
                    }
                    self.start_next_queued_download();
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::DeleteModel {
                family_idx,
                variant_idx,
                mut typed,
            } => match code {
                KeyCode::Esc | KeyCode::Left => {}
                KeyCode::Enter if typed == "delete" => {
                    self.delete_installed_variant(family_idx, variant_idx);
                }
                KeyCode::Char(c) => {
                    typed.push(c);
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
                KeyCode::Backspace => {
                    typed.pop();
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
                _ => {
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
            },
            Modal::StopServer => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    self.stop_server();
                    self.toast_warn("server stopped");
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::DestPicker(mut picker) => match code {
                KeyCode::Esc => {}
                KeyCode::Left | KeyCode::Char('h') => {
                    if let Some(parent) = picker.current.parent().map(std::path::Path::to_path_buf)
                    {
                        picker.set_current(parent);
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    picker.selected = picker.selected.saturating_sub(1);
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if picker.selected + 1 < picker.entries.len() {
                        picker.selected += 1;
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    picker.enter_selected();
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Char('~') => {
                    if let Some(home) = widgets::home_dir() {
                        picker.set_current(home);
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Char('d') => {
                    self.confirm_dest = None;
                    self.toast("using the shared HF cache");
                }
                KeyCode::Char('s') => match widgets::validate_writable_parent(&picker.current) {
                    Ok(()) => {
                        self.confirm_dest = Some(picker.current.clone());
                    }
                    Err(err) => {
                        picker.error = Some(err);
                        self.modal = Some(Modal::DestPicker(picker));
                    }
                },
                _ => self.modal = Some(Modal::DestPicker(picker)),
            },
        }
    }

    fn on_mouse(&mut self, mouse: MouseEvent) {
        if self.modal.is_some() {
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollDown => self.scroll(KeyCode::Down),
            MouseEventKind::ScrollUp => self.scroll(KeyCode::Up),
            MouseEventKind::Down(MouseButton::Left) => self.on_click(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Wheel scroll routes to the active screen's existing up/down handler.
    fn scroll(&mut self, code: KeyCode) {
        match self.screen {
            Screen::Home => self.on_key_home(code),
            Screen::Models => self.on_key_models(code),
            Screen::Downloads => self.on_key_downloads(code),
            Screen::Serve => self.on_key_serve(code),
            Screen::Chat => self.scroll_chat(code),
        }
    }

    fn on_click(&mut self, col: u16, row: u16) {
        // Sidebar click switches screens directly.
        if let Some(idx) = widgets::row_in_rect(self.sidebar_rect.get(), col, row) {
            if let Some(&(_, _, screen)) = SCREENS.get(idx) {
                self.screen = screen;
            }
            return;
        }
        // Step header click (breadcrumb navigation) on the Models screen.
        if self.screen == Screen::Models {
            let hdr = self.step_header_rect.get();
            if hdr.height > 0 && row == hdr.y && col >= hdr.x && col < hdr.x + hdr.width {
                let offset = (col - hdr.x) as usize;
                self.on_step_header_click(offset);
                return;
            }
        }
        // Content-list click selects the row (and drills in for the wizard).
        if let Some(idx) = widgets::row_in_rect(self.content_list_rect.get(), col, row) {
            match self.screen {
                Screen::Home => {
                    if idx < self.home_actions().len() {
                        self.home_idx = idx;
                        self.on_key_home(KeyCode::Enter);
                    }
                }
                Screen::Models => self.on_click_models(idx),
                Screen::Downloads => {
                    if idx < self.downloads.len() {
                        self.download_idx = idx;
                    }
                }
                Screen::Serve => {
                    if idx < installed_variants(&self.families).len() {
                        self.serve_focus = ServeFocus::List;
                        self.serve_idx = idx;
                    }
                }
                Screen::Chat => {}
            }
        }
    }

    // -- server lifecycle -------------------------------------------------------

    /// Validation message for the port field, if it holds non-empty, non-numeric, or out-of-range text.
    fn port_error(&self) -> Option<&'static str> {
        let trimmed = self.port.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<u16>() {
            Ok(0) | Err(_) => Some("port must be 1-65535"),
            Ok(_) => None,
        }
    }

    fn server_running(&self) -> bool {
        self.server.as_ref().is_some_and(|j| j.done.is_none())
    }

    fn serve_installed(&mut self, family_idx: usize, variant_idx: usize) {
        if self.port_error().is_some() || self.server_running() {
            return;
        }
        let Some(variant) = self
            .families
            .get(family_idx)
            .and_then(|f| f.variants.get(variant_idx))
        else {
            return;
        };
        let profile = variant.profile;
        // Prefer the exact snapshot directory over the preset+hf-cache scan: with
        // more than one precision of a model family installed, the scan can't
        // tell them apart (they share the preset's alias substring) and errors
        // out with "multiple Hugging Face cache candidates". Knowing the repo id
        // here means there's no ambiguity to resolve.
        let artifacts_dir = catalog::repo_snapshot_dir(profile.repo_id);
        self.spawn_server(profile.preset, artifacts_dir, profile.label);
    }

    fn start_server_for_download(&mut self, download_idx: usize) {
        if self.port_error().is_some() || self.server_running() {
            return;
        }
        let Some(task) = self.downloads.get(download_idx) else {
            return;
        };
        if !task.is_ready() {
            return;
        }
        let artifacts_dir = task.output_path().or_else(|| {
            if task.mode == DownloadMode::Direct && task.preset.is_none() {
                catalog::repo_snapshot_dir(task.repo_id)
            } else {
                None
            }
        });
        let label = task.label.clone();
        self.spawn_server(task.preset, artifacts_dir, &label);
    }

    fn spawn_server(
        &mut self,
        preset: Option<&str>,
        artifacts_dir: Option<PathBuf>,
        model_label: &str,
    ) {
        self.server_ready = false;
        let host = if self.host.trim().is_empty() {
            "127.0.0.1".to_string()
        } else {
            self.host.trim().to_string()
        };
        let port = if self.port.trim().is_empty() {
            "8080".to_string()
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
        // `--preset` (when known) carries model_id/support_tier metadata and is
        // not mutually exclusive with an explicit artifacts dir. The dir always
        // wins for *locating* the weights: it's unambiguous by construction,
        // whereas `--resolve-model-artifacts hf-cache` scans the whole HF cache
        // by alias substring and errors out the moment more than one installed
        // precision/snapshot matches. Only fall back to that scan when the exact
        // directory genuinely couldn't be resolved.
        if let Some(preset) = preset {
            cmd.arg("--preset").arg(preset);
        }
        match &artifacts_dir {
            Some(dir) => {
                cmd.arg("--mlx-model-artifacts-dir").arg(dir);
            }
            None if preset.is_some() => {
                cmd.arg("--resolve-model-artifacts").arg("hf-cache");
            }
            None => {
                self.server = Some(Job::failed(
                    "no server artifact path could be resolved for this download".into(),
                ));
                self.server_url = None;
                return;
            }
        }
        match Job::spawn(cmd, None) {
            Ok(job) => {
                self.server = Some(job);
                self.server_url = Some(format!("http://{host}:{port}"));
                self.server_model = Some(model_label.to_string());
            }
            Err(err) => {
                self.server = Some(Job::failed(format!("failed to launch server: {err}")));
                self.server_url = None;
            }
        }
    }

    /// Flip `server_ready` once the server job's log confirms it actually bound.
    fn update_server_ready(&mut self) {
        if self.server_ready {
            return;
        }
        if let Some(job) = &self.server
            && job
                .log
                .iter()
                .any(|line| line.contains("listening on http://"))
        {
            self.server_ready = true;
        }
    }

    fn stop_server(&mut self) {
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        self.server = None;
        self.server_url = None;
        self.server_ready = false;
        self.server_model = None;
    }

    /// Most recent non-empty server log line, surfaced when startup fails.
    fn server_error_line(&self) -> Option<String> {
        let job = self.server.as_ref()?;
        job.done?;
        job.log
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty())
            .cloned()
    }

    // -- download queue ---------------------------------------------------------

    fn start_next_queued_download(&mut self) {
        if self.downloads.iter().any(DownloadTask::is_running) {
            return;
        }
        if let Some(task) = self.downloads.iter_mut().find(|task| task.is_queued()) {
            task.spawn();
        }
    }

    fn active_download_summary(&self) -> Option<(String, Color)> {
        if let Some(task) = self.downloads.iter().find(|t| t.is_running()) {
            let pct = task
                .progress_ratio()
                .map(|r| format!(" {:.0}%", r * 100.0))
                .unwrap_or_default();
            let speed = task
                .job
                .as_ref()
                .filter(|job| job.speed > 0.0)
                .map(|job| format!(" · {}/s", catalog::format_bytes(job.speed as u64)))
                .unwrap_or_default();
            return Some((format!("{}{pct}{speed}", task.label), Color::Cyan));
        }
        let queued = self.downloads.iter().filter(|t| t.is_queued()).count();
        if queued > 0 {
            return Some((format!("{queued} queued"), Color::Yellow));
        }
        None
    }

    fn server_summary(&self) -> (String, Color) {
        match (&self.server_url, &self.server) {
            (Some(url), Some(job)) if job.done.is_none() && self.server_ready => (
                format!(
                    "{} @ {url}",
                    self.server_model.as_deref().unwrap_or("model")
                ),
                Color::Green,
            ),
            (Some(_), Some(job)) if job.done.is_none() => ("starting…".into(), Color::Yellow),
            (_, Some(job)) if job.done.is_some() => ("failed (see Serve log)".into(), Color::Red),
            _ => ("stopped".into(), Color::DarkGray),
        }
    }

    // -- rendering ------------------------------------------------------------

    fn draw(&self, frame: &mut Frame) {
        // Cleared each frame; the active content list re-records it, so screens
        // without a list leave no stale click target.
        self.content_list_rect.set(Rect::default());
        let outer = Layout::vertical([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(frame.area());
        let body = Layout::horizontal([Constraint::Length(18), Constraint::Min(0)]).split(outer[0]);
        self.draw_sidebar(frame, body[0]);
        match self.screen {
            Screen::Home => self.draw_home(frame, body[1]),
            Screen::Models => self.draw_models(frame, body[1]),
            Screen::Downloads => self.draw_downloads(frame, body[1]),
            Screen::Serve => self.draw_serve(frame, body[1]),
            Screen::Chat => self.draw_chat(frame, body[1]),
        }
        widgets::draw_status_strip(
            frame,
            outer[1],
            self.server_summary(),
            self.active_download_summary(),
        );
        frame.render_widget(
            Paragraph::new(self.footer()).style(Style::default().fg(Color::DarkGray)),
            outer[2],
        );
        widgets::draw_toasts(frame, frame.area(), &self.toasts);
        if self.show_help {
            self.draw_help(frame, frame.area());
        }
        if let Some(modal) = &self.modal {
            self.draw_modal(frame, frame.area(), modal);
        }
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        self.sidebar_rect.set(area);
        let running_downloads = self.downloads.iter().filter(|t| t.is_running()).count();
        let items = SCREENS.iter().map(|(icon, name, screen)| {
            let selected = self.screen == *screen;
            let badge = match screen {
                Screen::Downloads if running_downloads > 0 => {
                    format!(" {}", theme::spinner_dot(running_downloads))
                }
                Screen::Serve if self.server_ready => format!(" {}", theme::OK_DOT),
                _ => String::new(),
            };
            let label = format!(
                " {} {} {}{}",
                if selected { "▸" } else { " " },
                icon,
                name,
                badge
            );
            let style = if selected {
                Style::default()
                    .fg(theme::ACCENT)
                    .bg(Color::Rgb(20, 40, 50))
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(label)).style(style)
        });
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme::MUTED))
            .title(Span::styled(
                " AX Engine ",
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        frame.render_widget(List::new(items.collect::<Vec<_>>()).block(block), area);
    }

    fn footer(&self) -> Line<'static> {
        use theme::{key_chip, key_chip_dim, key_sep};
        let spans: Vec<Span> = if self.modal.is_some() {
            vec![
                key_chip("Esc"),
                key_sep(),
                Span::styled("close dialog", Style::default().fg(theme::DIM)),
            ]
        } else {
            match self.screen {
                Screen::Home => vec![
                    key_chip("↑↓"),
                    Span::styled(" move", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip("Enter"),
                    Span::styled(" select", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("?"),
                    Span::styled(" help", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("q"),
                    Span::styled(" quit", Style::default().fg(theme::DIM)),
                ],
                Screen::Models => match self.stage {
                    WizardStage::Families if self.filtering => vec![
                        Span::styled("type to filter", Style::default().fg(theme::ACCENT)),
                        key_sep(),
                        key_chip("Enter"),
                        Span::styled(" apply", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("Esc"),
                        Span::styled(" cancel", Style::default().fg(theme::DIM)),
                    ],
                    WizardStage::Families => vec![
                        key_chip("↑↓"),
                        Span::styled(" move", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip("Enter"),
                        Span::styled(" next", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("/"),
                        Span::styled(" filter", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("Esc"),
                        Span::styled(" home", Style::default().fg(theme::DIM)),
                    ],
                    WizardStage::Precision => vec![
                        key_chip("↑↓"),
                        Span::styled(" move", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip("Enter"),
                        Span::styled(" next", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("x"),
                        Span::styled(" delete", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("Esc"),
                        Span::styled(" back", Style::default().fg(theme::DIM)),
                    ],
                    WizardStage::Options => vec![
                        key_chip("y"),
                        Span::styled(" yes", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip("n"),
                        Span::styled(" no", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("Esc"),
                        Span::styled(" back", Style::default().fg(theme::DIM)),
                    ],
                    WizardStage::Confirm => vec![
                        key_chip("Enter"),
                        Span::styled(" download", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("c"),
                        Span::styled(" folder", Style::default().fg(theme::DIM)),
                        key_sep(),
                        key_chip_dim("Esc"),
                        Span::styled(" back", Style::default().fg(theme::DIM)),
                    ],
                },
                Screen::Downloads => vec![
                    key_chip("↑↓"),
                    Span::styled(" move", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip("Enter"),
                    Span::styled(" serve", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("x"),
                    Span::styled(" cancel", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("Esc"),
                    Span::styled(" home", Style::default().fg(theme::DIM)),
                ],
                Screen::Serve => vec![
                    key_chip("Enter"),
                    Span::styled(" start", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("x"),
                    Span::styled(" stop", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("c"),
                    Span::styled(" copy", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("t"),
                    Span::styled(" chat", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("Tab"),
                    Span::styled(" fields", Style::default().fg(theme::DIM)),
                ],
                Screen::Chat if !self.server_ready => vec![
                    key_chip("4"),
                    Span::styled(" serve screen", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("Esc"),
                    Span::styled(" home", Style::default().fg(theme::DIM)),
                ],
                Screen::Chat => vec![
                    key_chip("Enter"),
                    Span::styled(" send", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("PgUp/Dn"),
                    Span::styled(" scroll", Style::default().fg(theme::DIM)),
                    key_sep(),
                    key_chip_dim("Esc"),
                    Span::styled(" home", Style::default().fg(theme::DIM)),
                ],
            }
        };
        let mut out = vec![Span::raw("  ")];
        out.extend(spans);
        Line::from(out)
    }

    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let popup = widgets::centered_rect(74, 24, area);
        let lines = vec![
            Line::from(Span::styled(
                "AX Engine TUI",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            Line::raw("Screens: press 1-5 (or click the sidebar) at any time."),
            Line::raw("  1 Home       hardware summary and quick actions"),
            Line::raw("  2 Models     wizard: family -> precision -> options -> confirm"),
            Line::raw("  3 Downloads  background queue with live progress"),
            Line::raw("  4 Serve      launch/stop ax-engine-server for an installed model"),
            Line::raw("  5 Chat       talk to the running server"),
            Line::raw(""),
            Line::raw("Models"),
            Line::raw("  Sizes are estimates; the fit badge compares them to this"),
            Line::raw("  machine's RAM. Nothing downloads until you confirm the summary."),
            Line::raw("  x on an installed precision deletes it (asks you to type 'delete')."),
            Line::raw(""),
            Line::raw("Downloads keep running while you browse; they are cancelled only"),
            Line::raw("when you confirm quitting. Partial downloads resume next time."),
            Line::raw(""),
            Line::raw("Serve: Enter starts, x stops (confirmed), c copies the URL,"),
            Line::raw("t opens Chat. Host/port are editable via Tab."),
            Line::raw(""),
            Line::from(Span::styled(
                "Press any key to close this help. Esc elsewhere steps back, never quits.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        frame.render_widget(ratatui::widgets::Clear, popup);
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(" Help "))
                .wrap(Wrap { trim: false }),
            popup,
        );
    }

    fn draw_modal(&self, frame: &mut Frame, area: Rect, modal: &Modal) {
        match modal {
            Modal::Quit { downloads, server } => {
                let mut lines = Vec::new();
                if *downloads > 0 {
                    lines.push(Line::raw(format!(
                        "{downloads} download{} still in progress — partial data resumes on the next run.",
                        if *downloads == 1 { "" } else { "s" }
                    )));
                }
                if *server {
                    lines.push(Line::raw("The server is running and will be stopped."));
                }
                lines.push(Line::raw(""));
                lines.push(Line::from(Span::styled(
                    "Quit AX Engine?",
                    Style::default().add_modifier(Modifier::BOLD),
                )));
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Quit",
                    lines,
                    vec![
                        theme::key_chip_danger("y quit"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc stay"),
                    ],
                    theme::WARN,
                );
            }
            Modal::ServeReady { download_idx } => {
                let label = self
                    .downloads
                    .get(*download_idx)
                    .map(|t| t.label.clone())
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "✓ Serve model",
                    vec![Line::raw(format!("Start the server with {label}?"))],
                    vec![
                        theme::key_chip("y serve"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc not now"),
                    ],
                    theme::OK,
                );
            }
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => {
                let label = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| format!("{} {}", self.families[*family_idx].key, v.precision()))
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "✓ Already installed",
                    vec![
                        Line::raw(format!("{label} is already downloaded.")),
                        Line::raw("Serve it now instead?"),
                    ],
                    vec![
                        theme::key_chip("y serve"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc back"),
                    ],
                    theme::OK,
                );
            }
            Modal::CancelDownload { download_idx } => {
                let label = self
                    .downloads
                    .get(*download_idx)
                    .map(|t| t.label.clone())
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Cancel download",
                    vec![Line::raw(format!(
                        "Stop downloading {label}? Partial data resumes if you retry later."
                    ))],
                    vec![
                        theme::key_chip_danger("y cancel"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc keep"),
                    ],
                    theme::WARN,
                );
            }
            Modal::DeleteModel {
                family_idx,
                variant_idx,
                typed,
            } => {
                let (label, path, size) = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        (
                            format!("{} {}", self.families[*family_idx].key, v.precision()),
                            catalog::repo_cache_dir(v.profile.repo_id)
                                .display()
                                .to_string(),
                            catalog::format_bytes(v.size),
                        )
                    })
                    .unwrap_or_default();
                let armed = typed == "delete";
                let typed_style = if armed {
                    Style::default().fg(theme::OK)
                } else {
                    Style::default().fg(theme::WARN)
                };
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Delete model",
                    vec![
                        Line::raw(format!("Remove {label} ({size}) from disk?")),
                        Line::from(Span::styled(path, Style::default().fg(theme::MUTED))),
                        Line::raw(""),
                        Line::from(vec![
                            Span::raw("Type 'delete' to confirm: "),
                            Span::styled(format!("{typed}_"), typed_style),
                        ]),
                    ],
                    if armed {
                        vec![
                            theme::key_chip_danger("Enter delete"),
                            theme::key_sep(),
                            theme::key_chip_dim("Esc keep"),
                        ]
                    } else {
                        vec![theme::key_chip_dim("Esc keep")]
                    },
                    theme::DANGER,
                );
            }
            Modal::StopServer => {
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Stop server",
                    vec![Line::raw("Stop the running server?")],
                    vec![
                        theme::key_chip_danger("y stop"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc keep"),
                    ],
                    theme::WARN,
                );
            }
            Modal::DestPicker(picker) => self.draw_dest_picker(frame, area, picker),
        }
    }
}
