//! ratatui terminal UI for `ax-engine tui`.
//!
//! A guided launcher over the existing CLI subcommands: pick a model in a
//! split-panel wizard (family -> precision -> options -> confirm), watch the
//! download queue with real progress, serve an installed model, and talk to it
//! on the Chat screen.  All work runs as background child processes
//! (`ax-engine download`, `ax-engine-server`, `curl`) streamed into the UI, so
//! browsing never blocks and quitting is an explicit, confirmed action while
//! jobs are live.
//!
//! It is a child module of the `ax-engine` binary, so it reuses that binary's
//! private catalog and helpers directly via `crate::`.

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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

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
pub(super) enum Screen {
    Home,
    Models,
    Downloads,
    Serve,
    Chat,
}

/// Tab definitions for the horizontal tab bar.
const TABS: [widgets::TabDef; 5] = [
    widgets::TabDef {
        num: '1',
        label: "Home",
    },
    widgets::TabDef {
        num: '2',
        label: "Models",
    },
    widgets::TabDef {
        num: '3',
        label: "Downloads",
    },
    widgets::TabDef {
        num: '4',
        label: "Serve",
    },
    widgets::TabDef {
        num: '5',
        label: "Chat",
    },
];

const SCREENS: [Screen; 5] = [
    Screen::Home,
    Screen::Models,
    Screen::Downloads,
    Screen::Serve,
    Screen::Chat,
];

fn screen_index(screen: Screen) -> usize {
    SCREENS.iter().position(|s| *s == screen).unwrap_or(0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum WizardStage {
    Families,
    Precision,
    Options,
    Confirm,
}

#[derive(Clone, Copy, PartialEq)]
pub(super) enum ServeFocus {
    List,
    Host,
    Port,
}

#[derive(Clone, Copy)]
pub(super) struct PendingDownload {
    pub family_idx: usize,
    pub precision_idx: usize,
    pub with_mtp: bool,
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
    pub quit: bool,
    pub screen: Screen,
    pub hardware: HardwareInfo,
    pub families: Vec<Family>,
    pub modal: Option<Modal>,
    pub toasts: Vec<Toast>,
    pub show_help: bool,

    // Home
    pub home_idx: usize,

    // Models wizard
    pub stage: WizardStage,
    pub family_idx: usize,
    pub precision_idx: usize,
    pub mtp_idx: usize, // 0 = yes, 1 = no
    pub pending: Option<PendingDownload>,
    /// Custom destination chosen on the confirm step (None = shared HF cache).
    pub confirm_dest: Option<PathBuf>,
    /// Case-insensitive substring filter over the family list (`/` to edit).
    pub filter: String,
    pub filtering: bool,

    // Downloads
    pub downloads: Vec<DownloadTask>,
    pub download_idx: usize,

    // Serve
    pub serve_focus: ServeFocus,
    pub serve_idx: usize,
    pub host: String,
    pub port: String,
    pub server: Option<Job>,
    pub server_url: Option<String>,
    /// Set once the server job's log confirms it actually bound (not just spawned).
    pub server_ready: bool,
    /// Label of the model the running server was started with (chat request body).
    pub server_model: Option<String>,

    // Chat
    pub chat: ChatState,

    // Click-target rects recorded during the last draw (immediate-mode hit-testing).
    tab_hits: Cell<Vec<(Rect, usize)>>,
    pub content_list_rect: Cell<Rect>,
    /// Rect of the wizard step header row (for breadcrumb clicks).
    pub step_header_rect: Cell<Rect>,
}

impl App {
    pub fn new() -> App {
        App::with_hardware(HardwareInfo::probe())
    }

    pub fn with_hardware(hardware: HardwareInfo) -> App {
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
            tab_hits: Cell::new(Vec::new()),
            content_list_rect: Cell::new(Rect::default()),
            step_header_rect: Cell::new(Rect::default()),
        }
    }

    pub fn reload_families(&mut self) {
        self.families = build_families();
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::info(text.into()));
    }

    pub fn toast_success(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::success(text.into()));
    }

    pub fn toast_warn(&mut self, text: impl Into<String>) {
        self.toasts.push(widgets::Toast::warning(text.into()));
    }

    pub fn toast_error(&mut self, text: impl Into<String>) {
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

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_quit();
            return;
        }
        if self.modal.is_some() {
            self.on_key_modal(key.code);
            return;
        }
        if self.show_help {
            // Any key dismisses help.
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
                    self.screen = SCREENS[(c as usize) - ('1' as usize)];
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

    pub fn on_click(&mut self, col: u16, row: u16) {
        // Tab bar click switches screens.
        let tab_hits = self.tab_hits.take();
        let clicked_tab = tab_hits.iter().find_map(|(rect, idx)| {
            (col >= rect.x
                && col < rect.x + rect.width
                && row >= rect.y
                && row < rect.y + rect.height)
                .then_some(*idx)
        });
        self.tab_hits.set(tab_hits);
        if let Some(idx) = clicked_tab {
            self.screen = SCREENS[idx];
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
    pub fn port_error(&self) -> Option<&'static str> {
        let trimmed = self.port.trim();
        if trimmed.is_empty() {
            return None;
        }
        match trimmed.parse::<u16>() {
            Ok(0) | Err(_) => Some("port must be 1-65535"),
            Ok(_) => None,
        }
    }

    pub fn server_running(&self) -> bool {
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
    pub fn update_server_ready(&mut self) {
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
    pub fn server_error_line(&self) -> Option<String> {
        let job = self.server.as_ref()?;
        job.done?;
        job.log
            .iter()
            .rev()
            .find(|line| !line.trim().is_empty())
            .cloned()
    }

    // -- download queue ---------------------------------------------------------

    pub fn start_next_queued_download(&mut self) {
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

    pub fn draw(&self, frame: &mut Frame) {
        // Cleared each frame; the active content list re-records it.
        self.content_list_rect.set(Rect::default());
        let outer = Layout::vertical([
            Constraint::Length(2), // tab bar + separator
            Constraint::Min(0),    // content
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

        // Build compact status spans for the tab bar.
        let status_spans = self.build_status_spans();
        let hits = widgets::draw_tab_bar(
            frame,
            outer[0],
            &TABS,
            screen_index(self.screen),
            status_spans,
        );
        self.tab_hits.set(hits);

        match self.screen {
            Screen::Home => self.draw_home(frame, outer[1]),
            Screen::Models => self.draw_models(frame, outer[1]),
            Screen::Downloads => self.draw_downloads(frame, outer[1]),
            Screen::Serve => self.draw_serve(frame, outer[1]),
            Screen::Chat => self.draw_chat(frame, outer[1]),
        }

        frame.render_widget(
            Paragraph::new(self.footer_line()).style(Style::default().fg(theme::DIM)),
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

    /// Build compact status spans for the right side of the tab bar.
    fn build_status_spans(&self) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let (_server_text, server_color) = self.server_summary();
        let short_server = match server_color {
            c if c == Color::Green => "● running".to_string(),
            c if c == Color::Yellow => "◌ starting".to_string(),
            c if c == Color::Red => "✗ failed".to_string(),
            _ => "○ stopped".to_string(),
        };
        spans.push(Span::styled(
            short_server,
            Style::default().fg(server_color),
        ));
        if let Some((dl_text, dl_color)) = self.active_download_summary() {
            let short_dl = if dl_text.contains('%') {
                format!("  ↓ {}", dl_text.split('%').next().unwrap_or("").trim())
            } else {
                format!("  ↓ {dl_text}")
            };
            spans.push(Span::raw("  "));
            spans.push(Span::styled(short_dl, Style::default().fg(dl_color)));
        }
        spans
    }

    fn footer_line(&self) -> Line<'static> {
        use theme::{key_hint, key_label, key_sep};
        let hints: Vec<Span> = if self.modal.is_some() {
            vec![key_hint("Esc"), key_label(" close")]
        } else {
            match self.screen {
                Screen::Home => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" select"),
                    key_sep(),
                    key_hint("?"),
                    key_label(" help"),
                    key_sep(),
                    key_hint("q"),
                    key_label(" quit"),
                ],
                Screen::Models => match self.stage {
                    WizardStage::Families if self.filtering => vec![
                        Span::styled("type to filter", Style::default().fg(theme::ACCENT)),
                        key_sep(),
                        key_hint("Enter"),
                        key_label(" apply"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" cancel"),
                    ],
                    WizardStage::Families => vec![
                        key_hint("↑↓"),
                        key_label(" move"),
                        key_sep(),
                        key_hint("Enter"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("/"),
                        key_label(" filter"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" home"),
                    ],
                    WizardStage::Precision => vec![
                        key_hint("↑↓"),
                        key_label(" move"),
                        key_sep(),
                        key_hint("Enter"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("x"),
                        key_label(" delete"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" back"),
                    ],
                    WizardStage::Options => vec![
                        key_hint("y"),
                        key_label(" yes"),
                        key_sep(),
                        key_hint("n"),
                        key_label(" no"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" back"),
                    ],
                    WizardStage::Confirm => vec![
                        key_hint("Enter"),
                        key_label(" download"),
                        key_sep(),
                        key_hint("c"),
                        key_label(" folder"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" back"),
                    ],
                },
                Screen::Downloads => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" serve"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" cancel"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" home"),
                ],
                Screen::Serve => vec![
                    key_hint("Enter"),
                    key_label(" start"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" stop"),
                    key_sep(),
                    key_hint("c"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("t"),
                    key_label(" chat"),
                    key_sep(),
                    key_hint("Tab"),
                    key_label(" fields"),
                ],
                Screen::Chat if !self.server_ready => vec![
                    key_hint("4"),
                    key_label(" serve screen"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" home"),
                ],
                Screen::Chat => vec![
                    key_hint("Enter"),
                    key_label(" send"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" scroll"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" home"),
                ],
            }
        };
        let mut out = vec![Span::raw("  ")];
        out.extend(hints);
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
            Line::raw("Screens: press 1-5 (or click the tab bar) at any time."),
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
