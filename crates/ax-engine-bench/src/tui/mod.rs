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
mod markdown;
mod metrics;
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
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags, MouseButton,
    MouseEvent, MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use catalog::{Family, build_families, installed_variants};
use hardware::HardwareInfo;
use jobs::{DownloadMode, DownloadOutcome, DownloadTask, Job};
use metrics::LiveMetrics;
use screens::chat::ChatState;
use widgets::{DirectoryPicker, Toast};

// ---------------------------------------------------------------------------
// Server ready detection
// ---------------------------------------------------------------------------

/// True when a captured `ax-engine-server` log line means the HTTP listener is up.
///
/// Handles both the stable operator line (no tracing) and the structured
/// `tracing` form used when `RUST_LOG` / `AX_ENGINE_SERVER_LOG` is set:
/// - `ax-engine-server preview listening on http://127.0.0.1:8080 ...`
/// - `INFO ax-engine-server preview listening bind_address=127.0.0.1:8080 ...`
pub(crate) fn server_log_indicates_ready(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    if lower.contains("listening on http://") {
        return true;
    }
    // Structured tracing: message is "ax-engine-server preview listening" plus
    // bind_address=... fields (no "on http://" substring).
    if lower.contains("preview listening")
        && (lower.contains("bind_address=") || lower.contains("listening on"))
    {
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn cmd_tui(args: &[OsString]) -> Result<u8, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "ax-engine tui — guided model downloader, server launcher, and chat.\n\n\
             Screens: 1 Home · 2 Models · 3 Downloads · 4 Serve · 5 Chat.\n\
             Home shows your hardware and a Quick start action.  The Models\n\
             wizard walks family -> size (with RAM-fit info) -> optional\n\
             speed-up -> a confirm summary before anything is downloaded.\n\
             Downloads run in a background queue with live progress; Serve\n\
             launches ax-engine-server; Chat streams replies with markdown,\n\
             live tok/s stats, prompt history (↑), and /clear /copy /retry.\n\
             Keys: 1-5 screens (Ctrl+1-5 while typing) · ↑↓ move · Enter select\n\
             · b next step · Esc back one level · ? help · q quit\n\
             (quitting asks first while jobs are running).\n\
             Theme: AX_TUI_THEME=light|mono · NO_COLOR disables color."
        );
        return Ok(0);
    }
    if !io::stdout().is_terminal() {
        return Err("ax-engine tui needs an interactive terminal".into());
    }
    // Resolve palette (dark / AX_TUI_THEME=light / NO_COLOR mono) and glyph
    // set (Unicode / ASCII fallback) before anything draws.
    theme::init();
    let mut terminal = ratatui::init();
    // ratatui::init() does not enable mouse reporting or bracketed paste; turn
    // both on so clicks/scroll reach us as Event::Mouse and pastes arrive as
    // Event::Paste instead of a burst of key events.  Also ask the terminal to
    // disambiguate modifier keys so Shift+Enter arrives as Enter+SHIFT (the
    // composer's documented newline shortcut) instead of a plain Enter —
    // terminals without support just ignore the request.
    let _ = ratatui::crossterm::execute!(
        io::stdout(),
        PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES),
        EnableMouseCapture,
        EnableBracketedPaste
    );
    let result = App::new().run(&mut terminal);
    let _ = ratatui::crossterm::execute!(
        io::stdout(),
        PopKeyboardEnhancementFlags,
        DisableMouseCapture,
        DisableBracketedPaste
    );
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

/// Mouse-wheel scroll step for the job-log panes (lines per wheel tick).
const LOG_WHEEL_LINES: usize = 3;

/// Scrollback state for one job-log pane (Downloads / Serve).
///
/// Pinned (`None`) follows the newest line — today's autoscroll behavior.
/// Scrolled (`Some(first)`) anchors the pane to an absolute log index, so
/// lines appended while the user is reading do not move the view; scrolling
/// back down to the bottom re-pins.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LogScroll(Option<usize>);

impl LogScroll {
    /// Index of the first visible line for a `total`-line log in a
    /// `height`-row pane. Pinned shows the newest `height` lines; a scrolled
    /// anchor is clamped into range when the log shrinks or the pane grows.
    pub fn first_visible(self, total: usize, height: usize) -> usize {
        let bottom = total.saturating_sub(height);
        self.0.map_or(bottom, |first| first.min(bottom))
    }

    /// True while following the newest output (autoscroll).
    pub fn is_pinned(self) -> bool {
        self.0.is_none()
    }

    /// Re-pin to the bottom; new lines follow again.
    pub fn pin_to_bottom(&mut self) {
        self.0 = None;
    }

    /// True while the pane shows anything older than the newest page.
    pub fn is_scrolled(self, total: usize, height: usize) -> bool {
        self.0
            .is_some_and(|first| first < total.saturating_sub(height))
    }

    /// Scroll `n` lines toward older output, clamped at the oldest line.
    /// A log that fits the pane entirely cannot scroll.
    pub fn scroll_up(&mut self, n: usize, total: usize, height: usize) {
        let bottom = total.saturating_sub(height);
        if bottom == 0 {
            return;
        }
        let first = self.first_visible(total, height).saturating_sub(n);
        self.0 = Some(first);
    }

    /// Scroll `n` lines toward newer output; reaching the bottom re-pins.
    pub fn scroll_down(&mut self, n: usize, total: usize, height: usize) {
        let Some(first) = self.0 else {
            return;
        };
        let bottom = total.saturating_sub(height);
        let next = first.saturating_add(n);
        self.0 = (next < bottom).then_some(next);
    }

    /// One entry point for keys and wheel: `page` scrolls a full pane height,
    /// otherwise a short wheel step.
    pub fn scroll(&mut self, up: bool, page: bool, total: usize, height: usize) {
        if height == 0 {
            return;
        }
        let n = if page { height } else { LOG_WHEEL_LINES };
        if up {
            self.scroll_up(n, total, height);
        } else {
            self.scroll_down(n, total, height);
        }
    }
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
    /// Confirm before wiping the chat transcript (Ctrl+L / `/clear`).
    ClearChat,
    /// Custom destination for the wizard confirm step.
    DestPicker(DirectoryPicker),
}

/// Server lifecycle status shown in the tab-bar summary, derived from the
/// server job + ready flag so label and color are decided in exactly one place.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServerStatus {
    Ready,
    Starting,
    Failed,
    Stopped,
}

impl ServerStatus {
    /// Tab-bar chip text: status icon + word.
    fn label(self) -> String {
        match self {
            Self::Ready => format!("{} running", theme::icon::running()),
            Self::Starting => format!("{} starting", theme::icon::queued()),
            Self::Failed => format!("{} failed", theme::icon::error()),
            Self::Stopped => format!("{} stopped", theme::icon::idle()),
        }
    }

    /// Chip color matching the status severity.
    fn color(self) -> Color {
        match self {
            Self::Ready => theme::colors().ok,
            Self::Starting => theme::colors().warn,
            Self::Failed => theme::colors().danger,
            Self::Stopped => theme::colors().muted,
        }
    }
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
    /// Live host load (memory / CPU / models footprint); sampled on tick.
    pub live_metrics: LiveMetrics,

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
    /// Log-pane scrollback for the selected download's job log.
    pub downloads_log_scroll: LogScroll,

    // Serve
    pub serve_focus: ServeFocus,
    pub serve_idx: usize,
    pub host: String,
    pub port: String,
    /// Char-index carets for Host/Port text entry (0 = before the first char).
    pub host_cursor: usize,
    pub port_cursor: usize,
    pub server: Option<Job>,
    pub server_url: Option<String>,
    /// Log-pane scrollback for the server job log.
    pub serve_log_scroll: LogScroll,
    /// Set once the server job's log confirms it actually bound (not just spawned).
    pub server_ready: bool,
    /// Label of the model the running server was started with (chat request body).
    pub server_model: Option<String>,

    // Chat
    pub chat: ChatState,

    /// When true, keyboard focus is on the top tab bar (1–5 screens).
    /// Up from the first content row enters this mode; Down/Enter leaves it.
    pub focus_tabs: bool,

    /// Screen history for Esc back-one-level, oldest first (a real stack).
    pub back_stack: Vec<Screen>,
    /// After a guided download finishes, start the server automatically.
    pub auto_serve_after_download: bool,
    /// After the server binds, jump to Chat automatically.
    pub auto_chat_after_serve: bool,

    // Click-target rects recorded during the last draw (immediate-mode hit-testing).
    tab_hits: Cell<Vec<(Rect, usize)>>,
    pub content_list_rect: Cell<Rect>,
    /// Rect of the job-log pane on Downloads / Serve (wheel-scroll routing).
    pub log_rect: Cell<Rect>,
    /// Rect of the wizard step header row (for breadcrumb clicks).
    pub step_header_rect: Cell<Rect>,
    /// Journey banner hit target (Home / Downloads / Serve).
    pub banner_rect: Cell<Rect>,
    /// Home first-run hero hit target.
    pub hero_rect: Cell<Rect>,
    /// Active modal's popup + chip hit targets (mouse confirm/cancel).
    modal_hits: Cell<widgets::ModalHits>,
}

impl App {
    pub fn new() -> App {
        App::with_hardware(HardwareInfo::probe())
    }

    pub fn with_hardware(hardware: HardwareInfo) -> App {
        let total_ram = hardware.total_ram_bytes;
        App {
            quit: false,
            screen: Screen::Home,
            hardware,
            families: build_families(),
            modal: None,
            toasts: Vec::new(),
            show_help: false,
            // Always 0: home_actions() puts the safe default first (Browse when
            // models are installed, Quick start only on first-run empty home).
            home_idx: 0,
            live_metrics: LiveMetrics::new(total_ram),
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
            downloads_log_scroll: LogScroll::default(),
            serve_focus: ServeFocus::List,
            serve_idx: 0,
            host: "127.0.0.1".into(),
            port: "8080".into(),
            host_cursor: "127.0.0.1".chars().count(),
            port_cursor: "8080".chars().count(),
            server: None,
            server_url: None,
            serve_log_scroll: LogScroll::default(),
            server_ready: false,
            server_model: None,
            chat: ChatState::new(),
            focus_tabs: false,
            back_stack: Vec::new(),
            auto_serve_after_download: false,
            auto_chat_after_serve: false,
            tab_hits: Cell::new(Vec::new()),
            content_list_rect: Cell::new(Rect::default()),
            log_rect: Cell::new(Rect::default()),
            step_header_rect: Cell::new(Rect::default()),
            banner_rect: Cell::new(Rect::default()),
            hero_rect: Cell::new(Rect::default()),
            modal_hits: Cell::new(widgets::ModalHits::default()),
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
        // Repaint on input or when tick() reports visible activity; a fully
        // idle app sleeps in poll() instead of redrawing at 10 Hz.
        let mut dirty = true;
        while !self.quit {
            if dirty {
                terminal.draw(|frame| self.draw(frame))?;
                dirty = false;
            }
            // Block up to the 100ms tick for the first event, then drain
            // anything queued behind it so held-down keys stay responsive.
            if event::poll(Duration::from_millis(100))? {
                self.on_event(event::read()?);
                dirty = true;
            }
            while event::poll(Duration::ZERO)? {
                self.on_event(event::read()?);
                dirty = true;
            }
            if self.tick() {
                dirty = true;
            }
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

    /// Dispatch one terminal event: key presses, mouse, and bracketed paste.
    fn on_event(&mut self, event: Event) {
        match event {
            Event::Key(key) if key.kind == KeyEventKind::Press => self.on_key(key),
            Event::Mouse(mouse) => self.on_mouse(mouse),
            Event::Paste(text) => self.on_paste(&text),
            _ => {}
        }
    }

    /// True while a modal or text entry owns input, so tick-driven
    /// auto-navigation must not yank the user away mid-flow.
    fn input_busy(&self) -> bool {
        self.modal.is_some() || self.typing()
    }

    /// Advance all background jobs and time-based UI state by one poll cycle.
    /// Returns true when anything on screen could have changed.
    fn tick(&mut self) -> bool {
        let mut finished: Vec<(usize, String)> = Vec::new();
        let mut failed: Vec<String> = Vec::new();
        for (idx, task) in self.downloads.iter_mut().enumerate() {
            match task.tick() {
                DownloadOutcome::Finished => finished.push((idx, task.label.clone())),
                DownloadOutcome::Failed => failed.push(task.label.clone()),
                DownloadOutcome::Pending => {}
            }
        }
        for label in failed {
            // A failed download installed nothing — no reload, no navigation,
            // just say how to recover.
            self.toast_error(format!("{label} failed — press r to retry"));
        }
        for (idx, label) in finished {
            self.reload_families();
            self.select_download(idx);
            // Never yank the user out of a modal or text entry; toasts and the
            // auto-serve chain itself still fire.
            let may_navigate = self.screen != Screen::Chat && !self.input_busy();
            if self.auto_serve_after_download {
                self.auto_serve_after_download = false;
                self.toast_success(format!("{label} ready — starting server…"));
                if may_navigate {
                    self.navigate_to(Screen::Serve);
                }
                self.start_server_for_download(idx);
            } else {
                self.toast_success(format!("{label} ready — Enter to serve"));
                // Guided handoff: jump to Downloads unless the user is mid-flow.
                if may_navigate {
                    self.navigate_to(Screen::Downloads);
                }
            }
        }
        self.start_next_queued_download();
        let server_was_running = self.server_running();
        if let Some(job) = &mut self.server {
            job.tick();
        }
        let was_ready = self.server_ready;
        self.update_server_ready();
        if self.server_ready && !was_ready {
            if self.auto_chat_after_serve {
                self.auto_chat_after_serve = false;
                // Same anti-yank guard as the download handoff: a binding
                // server must not interrupt a modal or text field.
                if !self.input_busy() {
                    self.navigate_to(Screen::Chat);
                }
                self.toast_success("Server ready — type a message");
            } else {
                self.toast_success("Server ready — press t Chat");
            }
        } else if server_was_running && !self.server_running() && !was_ready {
            // Died before ever binding. update_server_ready only warns on the
            // ready→stopped crash, so this never double-toasts.
            let detail = self
                .server_error_line()
                .unwrap_or_else(|| "see Serve log".to_string());
            self.toast_error(format!("server failed to start — {detail}"));
        }
        // Toast on the error *edge* (None → Some) so stream failures surface
        // even while the user is scrolled up; the error itself also renders
        // inline in the transcript.
        let was_streaming = self.chat.streaming();
        let chat_had_error = self.chat.error.is_some();
        self.chat.tick();
        if !chat_had_error && let Some(message) = self.chat.error.clone() {
            self.toast_error(message);
        }
        // Capture before expiry: the tick that drops the last toast must still
        // repaint, or the stale toast lingers until the next dirty event.
        let had_toasts = !self.toasts.is_empty();
        widgets::expire_toasts(&mut self.toasts);
        self.clamp_list_indices();
        // Host load for Home gauges/chart (~2 s throttle inside sampler).
        let models_bytes: u64 = installed_variants(&self.families)
            .into_iter()
            .map(|(fi, vi)| self.families[fi].variants[vi].size)
            .sum();
        let cache_root = crate::default_hf_cache_root();
        let metrics_sampled = self.live_metrics.tick(models_bytes, &cache_root);
        // Repaint only when something visible can move on its own: download
        // spinners/progress, a binding server, a streaming reply, fresh toast,
        // or a new metrics sample. A fully idle app stays dark.
        was_streaming
            || had_toasts
            || metrics_sampled
            || !self.toasts.is_empty()
            || self.downloads.iter().any(|t| t.is_running())
            || (self.server_running() && !self.server_ready)
            || self.chat.streaming()
    }

    /// Keep selection indices in range after installs/deletes/queue changes.
    pub(crate) fn clamp_list_indices(&mut self) {
        let prev_download_idx = self.download_idx;
        let n_dl = self.downloads.len();
        if n_dl == 0 {
            self.download_idx = 0;
        } else if self.download_idx >= n_dl {
            self.download_idx = n_dl - 1;
        }
        if self.download_idx != prev_download_idx {
            // The selected row shifted under the user; re-pin its log pane.
            self.downloads_log_scroll.pin_to_bottom();
        }
        let n_serve = installed_variants(&self.families).len();
        if n_serve == 0 {
            self.serve_idx = 0;
        } else if self.serve_idx >= n_serve {
            self.serve_idx = n_serve - 1;
        }
        if self.families.is_empty() {
            self.family_idx = 0;
            self.precision_idx = 0;
        } else {
            if self.family_idx >= self.families.len() {
                self.family_idx = self.families.len() - 1;
            }
            let n_var = self.families[self.family_idx].variants.len();
            if n_var == 0 {
                self.precision_idx = 0;
            } else if self.precision_idx >= n_var {
                self.precision_idx = n_var - 1;
            }
        }
        let n_home = self.home_actions().len();
        if n_home == 0 {
            self.home_idx = 0;
        } else if self.home_idx >= n_home {
            self.home_idx = n_home - 1;
        }
    }

    /// Select a download row; moving to a different row re-pins its log pane
    /// (each row shows a different job log, so scrollback would be nonsense).
    pub(crate) fn select_download(&mut self, idx: usize) {
        if self.download_idx != idx {
            self.download_idx = idx;
            self.downloads_log_scroll.pin_to_bottom();
        }
    }

    // -- log pane scrollback ----------------------------------------------------

    /// Content height of the log pane as last drawn (0 before the first
    /// draw, which makes scroll input a safe no-op).
    fn log_pane_height(&self) -> usize {
        (self.log_rect.get().height as usize).saturating_sub(1)
    }

    /// PgUp/PgDn or wheel on the Downloads log pane.
    pub(crate) fn scroll_downloads_log(&mut self, up: bool, page: bool) {
        // Scrolling down while already pinned is a no-op; skip the log lookup.
        if !up && self.downloads_log_scroll.is_pinned() {
            return;
        }
        let total = self
            .downloads
            .get(self.download_idx)
            .and_then(|task| task.job.as_ref())
            .map_or(0, |job| job.log.len());
        let height = self.log_pane_height();
        self.downloads_log_scroll.scroll(up, page, total, height);
    }

    /// PgUp/PgDn or wheel on the Serve log pane.
    pub(crate) fn scroll_serve_log(&mut self, up: bool, page: bool) {
        if !up && self.serve_log_scroll.is_pinned() {
            return;
        }
        let total = self.server.as_ref().map_or(0, |job| job.log.len());
        let height = self.log_pane_height();
        self.serve_log_scroll.scroll(up, page, total, height);
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
        // Ctrl+1–5 always switches screens, even while typing in Chat / fields.
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && let KeyCode::Char(c @ '1'..='5') = key.code
        {
            self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
            return;
        }
        // While the tab bar owns focus, navigation shortcuts always apply —
        // do not let Chat/host/filter typing swallow 1–5 / q / arrows.
        if self.focus_tabs {
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
                    self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
                    return;
                }
                _ => {}
            }
            if self.on_key_tabs(key.code) {
                return;
            }
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
                    self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
                    return;
                }
                // Keyboard path for the guided next-step banner (b = banner).
                // Only on screens that render it; Models wizard and Chat keep
                // `b` as plain text/navigation input.
                KeyCode::Char('b')
                    if matches!(
                        self.screen,
                        Screen::Home | Screen::Downloads | Screen::Serve
                    ) =>
                {
                    self.activate_journey_banner();
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

    /// Switch screens (tab / digit jump), remembering the prior screen for Esc.
    fn goto_screen(&mut self, screen: Screen) {
        self.navigate_to(screen);
    }

    /// Navigate to a screen, pushing the current one onto the Esc back stack.
    pub(crate) fn navigate_to(&mut self, screen: Screen) {
        if self.screen != screen {
            self.back_stack.push(self.screen);
            // Long sessions must not grow the stack without bound.
            if self.back_stack.len() > 16 {
                self.back_stack.remove(0);
            }
        }
        self.screen = screen;
        self.focus_tabs = false;
    }

    /// Pop the previous screen if any. Returns true when a pop happened.
    pub(crate) fn go_back_screen(&mut self) -> bool {
        if let Some(prev) = self.back_stack.pop() {
            self.screen = prev;
            self.focus_tabs = false;
            true
        } else {
            false
        }
    }

    /// Esc/←/h "back" from a top-level screen: step back one level when there
    /// is history, otherwise land on Home (the one screen with no back).
    pub(crate) fn back_or_home(&mut self) {
        if !self.go_back_screen() {
            self.navigate_to(Screen::Home);
        }
    }

    /// Enable download → serve → chat auto-handoff for guided flows.
    pub(crate) fn enable_auto_chain(&mut self) {
        self.auto_serve_after_download = true;
        self.auto_chat_after_serve = true;
    }

    /// Move keyboard focus onto the top tab bar (1 Home · 2 Models · …).
    /// Leaves any text-entry sub-mode so the bar actually owns the keyboard.
    pub(crate) fn focus_tab_bar(&mut self) {
        self.focus_tabs = true;
        self.filtering = false;
        if matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port) {
            self.serve_focus = ServeFocus::List;
        }
    }

    /// Activate whatever the journey banner is advertising (click / Enter).
    pub(crate) fn activate_journey_banner(&mut self) {
        if self.server_ready {
            self.navigate_to(Screen::Chat);
            return;
        }
        if self.server_running() {
            self.navigate_to(Screen::Serve);
            return;
        }
        if let Some(idx) = self.downloads.iter().position(|t| t.is_ready()) {
            self.select_download(idx);
            self.navigate_to(Screen::Downloads);
            self.modal = Some(Modal::ServeReady { download_idx: idx });
            return;
        }
        if self
            .downloads
            .iter()
            .any(|t| t.is_running() || t.is_queued())
        {
            self.navigate_to(Screen::Downloads);
            return;
        }
        if installed_variants(&self.families).is_empty() {
            self.home_idx = 0;
            self.navigate_to(Screen::Home);
            // Re-enter Home quick start without recursion through banners.
            self.quick_start_from_home();
            return;
        }
        self.navigate_to(Screen::Serve);
    }

    /// Quick start entry used by Home and the journey banner.
    pub(crate) fn quick_start_from_home(&mut self) {
        let Some((fi, vi)) = self.quick_start_target() else {
            self.stage = WizardStage::Families;
            self.navigate_to(Screen::Models);
            return;
        };
        self.family_idx = fi;
        self.precision_idx = vi;
        let installed = self.families[fi].variants[vi].installed;
        let has_mtp = self.families[fi].variants[vi].mtp_alias.is_some();
        if installed && !has_mtp {
            self.auto_chat_after_serve = true;
            self.modal = Some(Modal::ServeInstalled {
                family_idx: fi,
                variant_idx: vi,
            });
            return;
        }
        // Guided chain: after confirm, download → serve → chat.
        self.enable_auto_chain();
        self.navigate_to(Screen::Models);
        if has_mtp {
            self.mtp_idx = 0;
            self.stage = WizardStage::Options;
        } else {
            self.begin_confirm(false);
        }
    }

    /// Handle keys while the tab bar owns focus. Returns true if consumed.
    fn on_key_tabs(&mut self, code: KeyCode) -> bool {
        let idx = screen_index(self.screen);
        match code {
            KeyCode::Left | KeyCode::Char('h') => {
                if idx > 0 {
                    // Navigate like digit jumps (Esc history stays correct),
                    // but keep bar focus so repeated arrows keep moving.
                    self.goto_screen(SCREENS[idx - 1]);
                    self.focus_tabs = true;
                }
                true
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if idx + 1 < SCREENS.len() {
                    self.goto_screen(SCREENS[idx + 1]);
                    self.focus_tabs = true;
                }
                true
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Enter => {
                self.focus_tabs = false;
                true
            }
            KeyCode::Up | KeyCode::Char('k') => true, // already on the bar
            KeyCode::Esc => {
                self.focus_tabs = false;
                true
            }
            // Any other key leaves the bar but stays consumed — falling
            // through to the screen handler would fire two actions for one
            // keypress (e.g. `x` opening the destructive delete modal).
            _ => {
                self.focus_tabs = false;
                true
            }
        }
    }

    /// True while a screen is consuming plain characters (filter, host/port,
    /// chat input), so global single-letter shortcuts must stay inert.
    /// Screen switches remain available via Ctrl+1–5.
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
                    self.auto_chat_after_serve = true;
                    self.start_server_for_download(download_idx);
                    self.navigate_to(Screen::Serve);
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    self.auto_chat_after_serve = true;
                    self.serve_installed(family_idx, variant_idx);
                    self.navigate_to(Screen::Serve);
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
                // Nothing typed yet: n/h dismiss like the other confirm modals
                // instead of silently starting the confirm string.
                KeyCode::Char('n') | KeyCode::Char('h') if typed.is_empty() => {}
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
            Modal::ClearChat => match code {
                KeyCode::Enter | KeyCode::Char('y') => self.clear_chat(),
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
            // The dialog owns the mouse: the first chip acts as confirm and
            // the last as cancel; a click outside dismisses like Esc.
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                let hits = self.modal_hits.get();
                let inside = |rect: Rect| {
                    mouse.column >= rect.x
                        && mouse.column < rect.x + rect.width
                        && mouse.row >= rect.y
                        && mouse.row < rect.y + rect.height
                };
                if hits.confirm.is_some_and(inside) {
                    self.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                } else if hits.cancel.is_some_and(inside) || !inside(hits.popup) {
                    self.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
                }
            }
            return;
        }
        if self.show_help {
            // Help owns the screen: swallow every mouse event so scrolls and
            // clicks cannot leak through to the screen below. A left-click
            // dismisses help, mirroring "any key closes help".
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                self.show_help = false;
            }
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollDown => self.scroll(KeyCode::Down, mouse.column, mouse.row),
            MouseEventKind::ScrollUp => self.scroll(KeyCode::Up, mouse.column, mouse.row),
            MouseEventKind::Down(MouseButton::Left) => self.on_click(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Wheel scroll: over the job-log pane on Downloads / Serve it scrolls the
    /// log; anywhere else it drives the active screen's existing up/down handler.
    fn scroll(&mut self, code: KeyCode, col: u16, row: u16) {
        if self.focus_tabs {
            let _ = self.on_key_tabs(code);
            return;
        }
        if matches!(self.screen, Screen::Downloads | Screen::Serve) {
            let rect = self.log_rect.get();
            let over_log = col >= rect.x
                && col < rect.x + rect.width
                && row >= rect.y
                && row < rect.y + rect.height;
            if over_log {
                let up = code == KeyCode::Up;
                match self.screen {
                    Screen::Downloads => self.scroll_downloads_log(up, false),
                    Screen::Serve => self.scroll_serve_log(up, false),
                    _ => {}
                }
                return;
            }
        }
        match self.screen {
            Screen::Home => self.on_key_home(code),
            Screen::Models => self.on_key_models(code),
            Screen::Downloads => self.on_key_downloads(code),
            Screen::Serve => self.on_key_serve(code),
            Screen::Chat => self.scroll_chat(code),
        }
    }

    /// Bracketed-paste payload: route to the focused text entry (chat composer,
    /// Serve host/port fields, Models filter); ignored elsewhere and while a
    /// modal owns input.
    fn on_paste(&mut self, text: &str) {
        if self.modal.is_some() || self.show_help {
            return;
        }
        match self.screen {
            Screen::Chat if self.server_ready => self.paste_chat_text(text),
            Screen::Models if self.filtering => {
                self.filter.push_str(text);
                self.clamp_family_idx_to_filter();
            }
            Screen::Serve => {
                // Single-line fields: drop newlines/control chars from pastes.
                let clean: String = text.chars().filter(|c| !c.is_control()).collect();
                match self.serve_focus {
                    ServeFocus::Host => {
                        let at = widgets::char_boundary(&self.host, self.host_cursor);
                        self.host.insert_str(at, &clean);
                        self.host_cursor += clean.chars().count();
                    }
                    ServeFocus::Port => {
                        let at = widgets::char_boundary(&self.port, self.port_cursor);
                        self.port.insert_str(at, &clean);
                        self.port_cursor += clean.chars().count();
                    }
                    ServeFocus::List => {}
                }
            }
            _ => {}
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
            self.goto_screen(SCREENS[idx]);
            return;
        }
        // Clicking content leaves tab-bar focus.
        self.focus_tabs = false;
        // Journey banner click → activate next step.
        let banner = self.banner_rect.get();
        if banner.height > 0
            && col >= banner.x
            && col < banner.x + banner.width
            && row >= banner.y
            && row < banner.y + banner.height
        {
            self.activate_journey_banner();
            return;
        }
        // Home hero click → Quick start.
        let hero = self.hero_rect.get();
        if hero.height > 0
            && col >= hero.x
            && col < hero.x + hero.width
            && row >= hero.y
            && row < hero.y + hero.height
        {
            self.home_idx = 0;
            self.quick_start_from_home();
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
                        self.select_download(idx);
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

    /// Validation message for the host field, if it holds non-empty text that
    /// is neither a hostname nor an IP literal (IPv4 or bracketed/plain IPv6).
    pub fn host_error(&self) -> Option<&'static str> {
        let trimmed = self.host.trim();
        if trimmed.is_empty() {
            return None;
        }
        let valid = trimmed
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '[' | ']'));
        if valid {
            None
        } else {
            Some("host must be a hostname or IP address")
        }
    }

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
        if let Some(err) = self.host_error() {
            self.toast_error(err);
            return;
        }
        if let Some(err) = self.port_error() {
            self.toast_error(err);
            return;
        }
        if self.server_running() {
            self.toast_warn("stop the running server first (x on Serve)");
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
        if let Some(err) = self.host_error() {
            self.toast_error(err);
            return;
        }
        if let Some(err) = self.port_error() {
            self.toast_error(err);
            return;
        }
        if self.server_running() {
            self.toast_warn("stop the running server first (x on Serve)");
            return;
        }
        let Some(task) = self.downloads.get(download_idx) else {
            return;
        };
        if !task.is_ready() {
            return;
        }
        // Prefer the path the download process reported. For direct HF-cache
        // downloads, fall back to the usable snapshot dir. Never silently fall
        // back to base-model HF resolve for MTP packages — that would serve the
        // base weights without the MTP package.
        let artifacts_dir = task.output_path().or_else(|| {
            if task.mode == DownloadMode::Direct {
                catalog::repo_snapshot_dir(task.repo_id)
            } else {
                None
            }
        });
        if task.mode == DownloadMode::Mtp && artifacts_dir.is_none() {
            self.server = Some(Job::failed(
                "MTP package path could not be resolved from the download log; re-run download-mtp or serve the package directory directly".into(),
            ));
            self.server_url = None;
            self.serve_log_scroll.pin_to_bottom();
            self.toast_error("could not find MTP package path to serve");
            return;
        }
        let label = task.label.clone();
        // MTP packages are self-contained dirs; still pass the base preset when
        // present so chat template / model_id hints stay available.
        self.spawn_server(task.preset, artifacts_dir, &label);
    }

    fn spawn_server(
        &mut self,
        preset: Option<&str>,
        artifacts_dir: Option<PathBuf>,
        model_label: &str,
    ) {
        self.server_ready = false;
        // Every path through here replaces the server job — re-pin its log.
        self.serve_log_scroll.pin_to_bottom();
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
            // Direct alias installs: resolve the HF snapshot by preset when we
            // do not yet have an explicit snapshot path (rare after a clean
            // download; common when re-serving a known alias).
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
        // Record the resolved binary in the log so Serve failures are diagnosable
        // when PATH / sibling resolution picks the wrong install.
        let bin_display = cmd.get_program().to_string_lossy().into_owned();
        match Job::spawn(cmd, None) {
            Ok(mut job) => {
                job.log
                    .push(format!("spawning {bin_display} for {model_label}"));
                self.server = Some(job);
                self.server_url = Some(format!("http://{host}:{port}"));
                self.server_model = Some(model_label.to_string());
            }
            Err(err) => {
                self.server = Some(Job::failed(format!(
                    "failed to launch server ({bin_display}): {err}"
                )));
                self.server_url = None;
                self.toast_error(format!("failed to launch server: {err}"));
            }
        }
    }

    /// Track the server job in both directions: flip `server_ready` on once the
    /// log confirms the bind, and back off if the process has since exited
    /// (crash, port conflict after start) so Chat stops accepting input.
    pub fn update_server_ready(&mut self) {
        let Some(job) = &self.server else {
            return;
        };
        if job.done.is_some() {
            if self.server_ready {
                self.server_ready = false;
                self.toast_warn("server stopped — restart it on Serve");
            }
            return;
        }
        if !self.server_ready && job.log.iter().any(|line| server_log_indicates_ready(line)) {
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
        self.serve_log_scroll.pin_to_bottom();
    }

    /// Most recent non-empty server log line, surfaced when startup fails.
    /// Prefer real startup failures over trailing MLX kernel noise (`mlx error:`).
    pub fn server_error_line(&self) -> Option<String> {
        let job = self.server.as_ref()?;
        job.done?;
        let is_hard_error = |line: &str| {
            let t = line.trim();
            t.starts_with("Error:")
                || t.contains("ERROR")
                || t.contains("panic")
                || t.contains("could not")
                || t.contains("InvalidInput")
                || t.contains("failed to launch")
                || t.contains("Address already in use")
        };
        let is_soft_error = |line: &str| {
            let t = line.trim();
            !t.is_empty()
                && !t.to_ascii_lowercase().starts_with("mlx error:")
                && (t.contains("error") || t.contains("failed"))
        };
        job.log
            .iter()
            .rev()
            .find(|line| is_hard_error(line))
            .or_else(|| job.log.iter().rev().find(|line| is_soft_error(line)))
            .or_else(|| job.log.iter().rev().find(|line| !line.trim().is_empty()))
            .cloned()
    }

    // -- download queue ---------------------------------------------------------

    pub fn start_next_queued_download(&mut self) {
        if self.downloads.iter().any(DownloadTask::is_running) {
            return;
        }
        // A launch-time failure (e.g. executable not found) sets `done` via
        // Job::failed before the first tick, so tick() never sees a
        // None → Some edge and never toasts — surface it here instead.
        let launch_error =
            if let Some(task) = self.downloads.iter_mut().find(|task| task.is_queued()) {
                task.spawn();
                task.job
                    .as_ref()
                    .filter(|job| job.done.is_some())
                    .and_then(|job| job.log.first().cloned())
            } else {
                None
            };
        if let Some(message) = launch_error {
            self.toast_error(message);
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
            return Some((
                format!("{}{pct}{speed}", task.label),
                theme::colors().accent,
            ));
        }
        let queued = self.downloads.iter().filter(|t| t.is_queued()).count();
        if queued > 0 {
            return Some((format!("{queued} queued"), theme::colors().warn));
        }
        None
    }

    fn server_status(&self) -> ServerStatus {
        match (&self.server_url, &self.server) {
            (Some(_), Some(job)) if job.done.is_none() && self.server_ready => ServerStatus::Ready,
            (Some(_), Some(job)) if job.done.is_none() => ServerStatus::Starting,
            (_, Some(job)) if job.done.is_some() => ServerStatus::Failed,
            _ => ServerStatus::Stopped,
        }
    }

    // -- rendering ------------------------------------------------------------

    /// Smallest usable terminal; below this we show a resize hint instead of
    /// silently clamping every panel to zero height.
    const MIN_TERM_WIDTH: u16 = 60;
    const MIN_TERM_HEIGHT: u16 = 15;

    pub fn draw(&self, frame: &mut Frame) {
        let term = frame.area();
        if term.width < Self::MIN_TERM_WIDTH || term.height < Self::MIN_TERM_HEIGHT {
            let popup = widgets::centered_rect(44.min(term.width), 4.min(term.height), term);
            let lines = vec![
                Line::from(Span::styled("terminal too small", theme::warn())),
                Line::from(Span::styled(
                    format!(
                        "need at least {}×{} — resize to continue",
                        Self::MIN_TERM_WIDTH,
                        Self::MIN_TERM_HEIGHT
                    ),
                    theme::label(),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), popup);
            return;
        }
        // Cleared each frame; the active content list re-records it.
        self.content_list_rect.set(Rect::default());
        self.log_rect.set(Rect::default());
        self.banner_rect.set(Rect::default());
        self.hero_rect.set(Rect::default());
        self.modal_hits.set(widgets::ModalHits::default());
        let outer = Layout::vertical([
            Constraint::Length(2), // tab bar + separator
            Constraint::Min(0),    // content
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

        // Build compact status spans and per-tab badges for the tab bar.
        let status_spans = self.build_status_spans();
        let badges = self.build_tab_badges();
        let hits = widgets::draw_tab_bar(
            frame,
            outer[0],
            &TABS,
            screen_index(self.screen),
            status_spans,
            &badges,
            self.focus_tabs,
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
            Paragraph::new(self.footer_line()).style(Style::default().fg(theme::colors().dim)),
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
        let status = self.server_status();
        spans.push(Span::styled(
            status.label(),
            Style::default().fg(status.color()),
        ));
        if let Some((dl_text, dl_color)) = self.active_download_summary() {
            // Compact form drops the speed suffix but keeps the percent sign.
            let short_dl = match dl_text.split_once('%') {
                Some((before, _)) => format!("  ↓ {}%", before.trim()),
                None => format!("  ↓ {dl_text}"),
            };
            spans.push(Span::raw("  "));
            spans.push(Span::styled(short_dl, Style::default().fg(dl_color)));
        }
        spans
    }

    /// Compact badges drawn inside tab labels (download count, serve live).
    fn build_tab_badges(&self) -> Vec<Option<widgets::TabBadge>> {
        let active_dl = self
            .downloads
            .iter()
            .filter(|t| t.is_running() || t.is_queued())
            .count();
        let ready_dl = self.downloads.iter().filter(|t| t.is_ready()).count();
        let downloads_badge = if active_dl > 0 {
            Some(widgets::TabBadge {
                text: format!("·{active_dl}"),
                style: Style::default().fg(theme::colors().accent),
            })
        } else if ready_dl > 0 {
            Some(widgets::TabBadge {
                text: format!("·{ready_dl}"),
                style: Style::default().fg(theme::colors().ok),
            })
        } else {
            None
        };
        let serve_badge = if self.server_ready {
            Some(widgets::TabBadge {
                text: theme::icon::running().into(),
                style: Style::default().fg(theme::colors().ok),
            })
        } else if self.server_running() {
            Some(widgets::TabBadge {
                text: theme::icon::queued().into(),
                style: Style::default().fg(theme::colors().warn),
            })
        } else {
            None
        };
        vec![
            None, // Home
            None, // Models
            downloads_badge,
            serve_badge,
            None, // Chat
        ]
    }

    fn footer_line(&self) -> Line<'static> {
        use theme::{key_hint, key_label, key_sep};
        let hints: Vec<Span> = if self.modal.is_some() {
            vec![key_hint("Esc"), key_label(" close")]
        } else if self.focus_tabs {
            vec![
                key_hint("←→"),
                key_label(" screen"),
                key_sep(),
                key_hint("↓/Enter"),
                key_label(" content"),
                key_sep(),
                key_hint("1-5"),
                key_label(" jump"),
                key_sep(),
                key_hint("Esc"),
                key_label(" content"),
            ]
        } else {
            match self.screen {
                Screen::Home => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" select"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                    key_sep(),
                    key_hint("q"),
                    key_label(" quit"),
                ],
                Screen::Models => match self.stage {
                    WizardStage::Families if self.filtering => vec![
                        Span::styled(
                            "type to filter",
                            Style::default().fg(theme::colors().accent),
                        ),
                        key_sep(),
                        key_hint("Enter/Esc"),
                        key_label(" done — keeps the filter"),
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
                        key_hint("f"),
                        key_label(" folder"),
                        key_sep(),
                        key_hint("r"),
                        key_label(" default"),
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
                    key_hint("r"),
                    key_label(" retry"),
                    key_sep(),
                    key_hint("o"),
                    key_label(" open"),
                    key_sep(),
                    key_hint("⌫"),
                    key_label(" remove"),
                    key_sep(),
                    key_hint("d"),
                    key_label(" clear"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" cancel"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Serve
                    if matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port) =>
                {
                    vec![
                        Span::styled("type to edit", Style::default().fg(theme::colors().accent)),
                        key_sep(),
                        key_hint("Tab"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" done"),
                    ]
                }
                Screen::Serve if self.server_ready => vec![
                    key_hint("Enter"),
                    key_label(" chat"),
                    key_sep(),
                    key_hint("c"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" stop"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Serve => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
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
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("Tab"),
                    key_label(" fields"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Chat
                    if !self.server_ready
                        && self.server.as_ref().is_some_and(|job| job.done.is_some()) =>
                {
                    // Read-only transcript after a crash/stop.
                    vec![
                        key_hint("PgUp"),
                        key_label(" scroll"),
                        key_sep(),
                        key_hint("Ctrl+Y"),
                        key_label(" copy"),
                        key_sep(),
                        key_hint("4"),
                        key_label(" Serve"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" leave"),
                    ]
                }
                Screen::Chat if !self.server_ready => vec![
                    key_hint("4"),
                    key_label(" Serve"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" home"),
                ],
                Screen::Chat => vec![
                    key_hint("Enter"),
                    key_label(" send"),
                    key_sep(),
                    key_hint("Ctrl+J"),
                    key_label(" newline"),
                    key_sep(),
                    key_hint("↑"),
                    key_label(" history"),
                    key_sep(),
                    key_hint("PgUp"),
                    key_label(" scroll"),
                    key_sep(),
                    key_hint("Ctrl+Y"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("Ctrl+R"),
                    key_label(" retry"),
                    key_sep(),
                    key_hint("/"),
                    key_label(" cmds"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" leave"),
                ],
            }
        };
        let mut out = vec![Span::raw("  ")];
        out.extend(hints);
        Line::from(out)
    }

    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let popup = widgets::centered_rect(74, 22, area);
        let contextual = match self.screen {
            Screen::Home => "Home: ↑↓ move · Enter run the highlighted action",
            Screen::Models => "Models: wizard steps · / filter · Enter next · Esc back",
            Screen::Downloads => {
                "Downloads: Enter serve when ready · x cancel · PgUp/PgDn scroll log"
            }
            Screen::Serve => {
                "Serve: Enter start · x stop · c copy URL · t chat · Tab fields · PgUp/PgDn scroll log"
            }
            Screen::Chat => {
                "Chat: Enter send · ↑ history · Ctrl+J newline · Ctrl+U clear draft · Ctrl+Y copy · Ctrl+R retry · Ctrl+L clear · / commands · PgUp/PgDn scroll"
            }
        };
        let lines = vec![
            Line::from(Span::styled(
                "AX Engine",
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(theme::colors().accent),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                contextual,
                Style::default().fg(theme::colors().text),
            )),
            Line::raw(""),
            Line::raw("Screens: 1-5 (or click tabs). While typing, use Ctrl+1-5."),
            Line::raw("  1 Home        this Mac + quick start"),
            Line::raw("  2 Models      pick family → size → optional speed-up → confirm"),
            Line::raw("  3 Downloads   queue with live progress"),
            Line::raw("  4 Serve       start/stop the local server"),
            Line::raw("  5 Chat        talk to the running model"),
            Line::raw(""),
            Line::raw("Fit badges compare download size to this Mac's memory."),
            Line::raw("Nothing downloads until you confirm. Partials resume later."),
            Line::raw(""),
            Line::raw("b runs the highlighted next step on Home / Downloads / Serve."),
            Line::raw("Log panes (Downloads / Serve) scroll with PgUp/PgDn or the mouse wheel;"),
            Line::raw("scrolling back to the bottom re-pins to the newest output."),
            Line::raw(""),
            Line::from(Span::styled(
                "Any key closes help. Esc steps back one level; q quits.",
                Style::default().fg(theme::colors().muted),
            )),
        ];
        frame.render_widget(
            Paragraph::new("").style(Style::default().bg(theme::colors().scrim_bg)),
            area,
        );
        frame.render_widget(ratatui::widgets::Clear, popup);
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::colors().accent))
                        .title(Span::styled(" Help ", theme::title())),
                )
                .wrap(Wrap { trim: false }),
            popup,
        );
    }

    fn draw_modal(&self, frame: &mut Frame, area: Rect, modal: &Modal) {
        let hits = match modal {
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
                    theme::colors().warn,
                )
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
                    theme::colors().ok,
                )
            }
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => {
                let label = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        format!(
                            "{} {}",
                            self.families[*family_idx].display_name(),
                            v.precision()
                        )
                    })
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
                    theme::colors().ok,
                )
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
                    theme::colors().warn,
                )
            }
            Modal::DeleteModel {
                family_idx,
                variant_idx,
                typed,
            } => {
                let (label, path, size, profile_label) = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        (
                            format!(
                                "{} {}",
                                self.families[*family_idx].display_name(),
                                v.precision()
                            ),
                            catalog::repo_cache_dir(v.profile.repo_id)
                                .display()
                                .to_string(),
                            catalog::format_bytes(v.size),
                            v.profile.label,
                        )
                    })
                    .unwrap_or_default();
                // Non-blocking guard rails: warn when the target is in use.
                // `server_model` is the display label for download-served
                // models and the profile label for direct serves — check both.
                let is_served = self.server_running()
                    && self
                        .server_model
                        .as_deref()
                        .is_some_and(|m| m == label || m == profile_label);
                let has_active_download = !label.is_empty()
                    && self
                        .downloads
                        .iter()
                        .any(|t| (t.is_running() || t.is_queued()) && t.label == label);
                let armed = typed == "delete";
                let typed_style = if armed {
                    Style::default().fg(theme::colors().ok)
                } else {
                    Style::default().fg(theme::colors().warn)
                };
                let mut lines = vec![
                    Line::raw(format!("Remove {label} ({size}) from disk?")),
                    Line::from(Span::styled(
                        path,
                        Style::default().fg(theme::colors().muted),
                    )),
                ];
                if is_served {
                    lines.push(Line::from(Span::styled(
                        "⚠ this model is currently served",
                        theme::warn(),
                    )));
                }
                if has_active_download {
                    lines.push(Line::from(Span::styled(
                        "⚠ download in progress",
                        theme::warn(),
                    )));
                }
                lines.push(Line::raw(""));
                lines.push(Line::from(vec![
                    Span::raw("Type 'delete' to confirm: "),
                    Span::styled(format!("{typed}_"), typed_style),
                ]));
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Delete model",
                    lines,
                    if armed {
                        vec![
                            theme::key_chip_danger("Enter delete"),
                            theme::key_sep(),
                            theme::key_chip_dim("Esc keep"),
                        ]
                    } else {
                        vec![theme::key_chip_dim("Esc keep")]
                    },
                    theme::colors().danger,
                )
            }
            Modal::StopServer => widgets::draw_modal_with(
                frame,
                area,
                "⚠ Stop server",
                vec![Line::raw("Stop the running server?")],
                vec![
                    theme::key_chip_danger("y stop"),
                    theme::key_sep(),
                    theme::key_chip_dim("Esc keep"),
                ],
                theme::colors().warn,
            ),
            Modal::ClearChat => widgets::draw_modal_with(
                frame,
                area,
                "⚠ Clear chat",
                vec![Line::raw(
                    "Clear the whole transcript? This cannot be undone.",
                )],
                vec![
                    theme::key_chip_danger("y clear"),
                    theme::key_sep(),
                    theme::key_chip_dim("Esc keep"),
                ],
                theme::colors().warn,
            ),
            Modal::DestPicker(picker) => {
                self.draw_dest_picker(frame, area, picker);
                widgets::ModalHits {
                    popup: widgets::centered_rect(
                        72.min(area.width.saturating_sub(2)),
                        22.min(area.height.saturating_sub(2)),
                        area,
                    ),
                    ..Default::default()
                }
            }
        };
        self.modal_hits.set(hits);
    }
}
