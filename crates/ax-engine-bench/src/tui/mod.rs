//! ratatui terminal UI for `ax-engine tui`.
//!
//! A guided launcher over the existing CLI subcommands: pick a model in a
//! split-panel wizard (family -> size -> confirm), watch the
//! download queue with real progress, serve an installed model, and talk to it
//! on the Chat screen.  All work runs as background child processes
//! (`ax-engine download`, `ax-engine-server`, `curl`) streamed into the UI, so
//! browsing never blocks and quitting is an explicit, confirmed action while
//! jobs are live.
//!
//! It is a child module of the `ax-engine` binary, so it reuses that binary's
//! private catalog and helpers directly via `crate::`.

mod app_draw;
mod app_input;
mod app_server;
mod catalog;
mod hardware;
mod jobs;
mod markdown;
mod metrics;
mod screens;
mod server_probe;
mod theme;
mod widgets;

#[cfg(test)]
mod tests;

use std::cell::Cell;
use std::ffi::OsString;
use std::io::{self, IsTerminal};
use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
    PushKeyboardEnhancementFlags,
};
use ratatui::layout::Rect;
use ratatui::style::Color;

use catalog::{Family, build_families, installed_variants};
use hardware::HardwareInfo;
use jobs::{DownloadOutcome, DownloadTask, Job};
use metrics::LiveMetrics;
use screens::chat::ChatState;
use server_probe::ServerHealth;
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
             wizard walks family -> size (with RAM-fit info) -> a confirm\n\
             summary before anything is downloaded. Published MTP/assistant\n\
             artifacts are already included in the selected snapshot.\n\
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
    /// From Serve with a running server: Enter on a different installed
    /// model — stop the current one and restart with the selection.
    RestartServer {
        family_idx: usize,
        variant_idx: usize,
    },
    /// Confirm before wiping the chat transcript (Ctrl+L / `/clear`).
    ClearChat,
    /// Custom destination for the wizard confirm step.
    DestPicker(DirectoryPicker),
    /// Free-form Hugging Face link / repo id entry (`d` on the Models screen).
    DownloadByLink {
        input: String,
        error: Option<String>,
    },
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
    /// Set once the HTTP listener is up (child log bind line and/or `/health`).
    pub server_ready: bool,
    /// Next server-log index to scan for a bind line (avoids O(n) full rescan).
    server_ready_scan: usize,
    /// Label of the model the running server was started with (chat request body).
    pub server_model: Option<String>,
    /// True when readiness came from a `/health` probe against a process the
    /// TUI did not spawn (CLI `ax-engine-server`, another terminal, etc.).
    /// Stop detaches rather than killing; Serve will not start a second bind.
    pub external_server: bool,
    /// In-flight background `/health` probe (never blocks the UI thread).
    server_probe: Option<Receiver<Option<ServerHealth>>>,
    /// Last time a health probe was launched (rate-limit).
    last_server_probe: Option<Instant>,
    /// Base URL the in-flight/last probe targeted (re-probe immediately on change).
    server_probe_url: Option<String>,

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
    /// Scroll offset (`ListState::offset` after render) of the content list
    /// during the last draw — a click's within-panel row must be added to
    /// this to land on the item ratatui actually drew there, since the
    /// panel's first visible row is not always item 0 once the list scrolls.
    pub content_list_offset: Cell<usize>,
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
        Self::with_hardware_and_families(hardware, build_families())
    }

    /// Like [`Self::with_hardware`], but for tests: builds the catalog
    /// shape without `build_families()`'s real HF-cache disk scan
    /// (`repo_is_installed`/`dir_size` per downloadable profile), which is
    /// environment-dependent I/O unrelated to what TUI tests assert on and
    /// can take tens of seconds to minutes on a machine with many real
    /// cached model downloads.
    #[cfg(test)]
    pub(super) fn with_hardware_for_tests(hardware: HardwareInfo) -> App {
        Self::with_hardware_and_families(hardware, catalog::build_families_uninstalled())
    }

    fn with_hardware_and_families(hardware: HardwareInfo, families: Vec<Family>) -> App {
        let total_ram = hardware.total_ram_bytes;
        App {
            quit: false,
            screen: Screen::Home,
            hardware,
            families,
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
            port: "31418".into(),
            host_cursor: "127.0.0.1".chars().count(),
            port_cursor: "31418".chars().count(),
            server: None,
            server_url: None,
            serve_log_scroll: LogScroll::default(),
            server_ready: false,
            server_ready_scan: 0,
            server_model: None,
            external_server: false,
            server_probe: None,
            last_server_probe: None,
            server_probe_url: None,
            chat: ChatState::new(),
            focus_tabs: false,
            back_stack: Vec::new(),
            auto_serve_after_download: false,
            auto_chat_after_serve: false,
            tab_hits: Cell::new(Vec::new()),
            content_list_rect: Cell::new(Rect::default()),
            content_list_offset: Cell::new(0),
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

    /// Push a toast, coalescing consecutive duplicates: when the newest toast
    /// carries the same text and level, refresh its timestamp instead of
    /// stacking a copy (e.g. repeated Enter on a still-downloading row).
    fn push_toast(&mut self, toast: widgets::Toast) {
        if let Some(last) = self.toasts.last_mut()
            && last.text == toast.text
            && last.level == toast.level
        {
            last.at = std::time::Instant::now();
            return;
        }
        self.toasts.push(toast);
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.push_toast(widgets::Toast::info(text.into()));
    }

    pub fn toast_success(&mut self, text: impl Into<String>) {
        self.push_toast(widgets::Toast::success(text.into()));
    }

    pub fn toast_warn(&mut self, text: impl Into<String>) {
        self.push_toast(widgets::Toast::warning(text.into()));
    }

    pub fn toast_error(&mut self, text: impl Into<String>) {
        self.push_toast(widgets::Toast::error(text.into()));
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
        // Material download changes (bytes/phase/log/exit). Spinner alone does
        // not count — off-screen downloads must not force 10 Hz full redraws.
        let mut download_material = false;
        for (idx, task) in self.downloads.iter_mut().enumerate() {
            let (outcome, changed) = task.tick();
            download_material |= changed;
            match outcome {
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
        let managed_was_alive = self.managed_server_alive();
        let mut server_material = false;
        if let Some(job) = &mut self.server {
            server_material = job.tick().material;
        }
        let was_ready = self.server_ready;
        let was_external = self.external_server;
        self.update_server_ready();
        // HTTP probe: discover external servers and backstop log-line readiness
        // for managed children (structured logs, RUST_LOG, etc.).
        server_material |= self.tick_server_health_probe();
        if self.server_ready && !was_ready {
            if self.auto_chat_after_serve {
                self.auto_chat_after_serve = false;
                // Same anti-yank guard as the download handoff: a binding
                // server must not interrupt a modal or text field.
                if !self.input_busy() {
                    self.navigate_to(Screen::Chat);
                }
                self.toast_success("Server ready — type a message");
            } else if self.external_server && !was_external {
                self.toast_success("Found running server — press t Chat");
            } else {
                self.toast_success("Server ready — press t Chat");
            }
        } else if managed_was_alive && !self.managed_server_alive() && !was_ready {
            // Managed child died before ever binding. update_server_ready only
            // warns on the ready→stopped crash, so this never double-toasts.
            // External-only sessions do not use this path.
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
        let chat_changed = self.chat.tick();
        if !chat_had_error && let Some(message) = self.chat.error.clone() {
            self.toast_error(message);
        }
        // Capture before expiry: the tick that drops the last toast must still
        // repaint, or the stale toast lingers until the next dirty event.
        let had_toasts = !self.toasts.is_empty();
        widgets::expire_toasts(&mut self.toasts);
        self.clamp_list_indices();
        // Host load for Home gauges/chart. Probes run on a background thread
        // (~2 s interval); only the Home screen paints them, so only dirties
        // the frame when the user can actually see the gauges.
        let models_bytes: u64 = installed_variants(&self.families)
            .into_iter()
            .map(|(fi, vi)| self.families[fi].variants[vi].size)
            .sum();
        let cache_root = crate::default_hf_cache_root();
        let metrics_sampled = self.live_metrics.tick(models_bytes, &cache_root);
        let download_running = self.downloads.iter().any(|t| t.is_running());
        // Spinner animation only matters on the Downloads screen; tab-bar
        // percent updates come through download_material at the 1 Hz sample.
        let spinner_visible = download_running && self.screen == Screen::Downloads;
        // Repaint only when something visible can move on its own. A fully
        // idle app (or one with off-screen quiet jobs) stays dark.
        was_streaming
            || chat_changed
            || had_toasts
            || (metrics_sampled && self.screen == Screen::Home)
            || !self.toasts.is_empty()
            || download_material
            || spinner_visible
            || server_material
            || (self.server_ready != was_ready)
            || (self.external_server != was_external)
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
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '[' | ']'));
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

    /// True when a managed child is still alive (not yet exited).
    pub fn managed_server_alive(&self) -> bool {
        self.server.as_ref().is_some_and(|j| j.done.is_none())
    }

    /// True when a server is in play: managed child still running, or an
    /// external listener discovered via `/health` on the configured host/port.
    pub fn server_running(&self) -> bool {
        self.managed_server_alive() || (self.external_server && self.server_ready)
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
        if self.server_ready {
            return ServerStatus::Ready;
        }
        match &self.server {
            Some(job) if job.done.is_none() => ServerStatus::Starting,
            Some(job) if job.done.is_some() => ServerStatus::Failed,
            _ => ServerStatus::Stopped,
        }
    }
}
