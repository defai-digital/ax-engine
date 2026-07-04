//! ratatui terminal UI for `ax-engine tui`.
//!
//! A native (Rust) replacement for the former Python/Textual downloader.  It is
//! a child module of the `ax-engine` binary, so it reuses that binary's private
//! catalog and helpers directly via `crate::` (`MODEL_PROFILES`,
//! `default_hf_cache_root`, `find_executable`, ...) instead of duplicating them.
//!
//! Layout: a left sidebar (Models / Downloads / Serve) plus a content pane.  The
//! Models pane is a picker — model family -> precision -> optional MTP
//! accelerator -> destination.  Downloads are queued as background child
//! `ax-engine` processes, so users can continue browsing while long model
//! downloads run.  Server launches run as child `ax-engine-server` processes
//! whose output is streamed into a log pane, keeping stdout off the alternate
//! screen while reusing the existing CLI logic untouched.

use std::cell::Cell;
use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Sparkline, Wrap};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn cmd_tui(args: &[OsString]) -> Result<u8, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "ax-engine tui — interactive model downloader and serve launcher.\n\n\
             Sidebar: Models · Downloads · Serve.  Pick a model family,\n\
             precision (4/5/6/8-bit), optional MTP accelerator, and destination;\n\
             long downloads run in the Downloads queue while you keep browsing.\n\
             Keys: ↑↓ move · → enter · ← back · / filter · d default cache · s select dir · ? help · q quit."
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
// Catalog view (derived from crate::MODEL_PROFILES)
// ---------------------------------------------------------------------------

/// One precision variant of a model, with cached install state.
struct Variant {
    profile: &'static crate::ModelProfile,
    bits: Option<u32>,
    /// `download-mtp` alias when this variant has an MTP accelerator.
    mtp_alias: Option<&'static str>,
    installed: bool,
    size: u64,
}

impl Variant {
    fn precision(&self) -> String {
        self.bits
            .map(|b| format!("{b}-bit"))
            .unwrap_or_else(|| self.profile.label.to_string())
    }
}

/// A model and the precision variants it is published in.
struct Family {
    key: String,
    variants: Vec<Variant>,
}

impl Family {
    fn has_mtp(&self) -> bool {
        self.variants.iter().any(|v| v.mtp_alias.is_some())
    }

    fn quant_summary(&self) -> String {
        let bits: Vec<String> = self
            .variants
            .iter()
            .filter_map(|v| v.bits.map(|b| format!("{b}-bit")))
            .collect();
        if bits.is_empty() {
            "--".into()
        } else {
            bits.join(", ")
        }
    }
}

/// Quantization bit-width parsed from a repo id (e.g. `...-4bit` -> 4).
fn quant_bits(repo_id: &str) -> Option<u32> {
    let lower = repo_id.to_ascii_lowercase();
    let idx = lower.find("bit")?;
    let digits: String = lower[..idx]
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    digits.parse().ok()
}

/// Family key: the label with any trailing `-Nbit` precision suffix removed.
fn family_key(label: &str) -> String {
    let lower = label.to_ascii_lowercase();
    if let Some(idx) = lower.rfind("-")
        && lower[idx + 1..].ends_with("bit")
        && lower[idx + 1..idx + 2].chars().all(|c| c.is_ascii_digit())
    {
        return label[..idx].to_string();
    }
    label.to_string()
}

/// `download-mtp` trigger alias for a model label.
///
/// Mirrors the Python catalog's per-variant `mtp_target` exactly.  The Rust
/// `MTP_DOWNLOAD_TARGETS` alias data is not consistent enough to derive this
/// (e.g. GLM-4bit has no bare alias), so the mapping is explicit and small.
fn mtp_trigger_alias(label: &str) -> Option<&'static str> {
    match label {
        "gemma4-12b" => Some("gemma-4-12b-4bit"),
        "gemma4-12b-6bit" => Some("gemma-4-12b"),
        "gemma4-26b" => Some("gemma-4-26b"),
        "gemma4-31b" => Some("gemma-4-31b"),
        "glm4.7-flash-4bit" => Some("glm-4.7-flash"),
        "qwen3.6-27b" => Some("qwen3.6-27b-6bit"),
        "qwen3.6-27b-6bit" => Some("qwen3.6-27b-6bit"),
        "qwen3.6-35b" => Some("qwen3.6-35b-a3b"),
        _ => None,
    }
}

/// HF hub cache directory for a repo id (`.../models--org--name`).
fn repo_cache_dir(repo_id: &str) -> PathBuf {
    crate::default_hf_cache_root().join(format!("models--{}", repo_id.replace('/', "--")))
}

/// The actual on-disk snapshot directory for a downloaded repo (containing
/// `config.json`/`*.safetensors`), not just the top-level HF cache wrapper.
/// Picks the most recently modified snapshot when a repo has more than one
/// cached revision. This is what the server needs for `--mlx-model-artifacts-dir`
/// — passing the wrapper dir directly would miss the actual model files, which
/// live one level down under `snapshots/<hash>/`.
fn repo_snapshot_dir(repo_id: &str) -> Option<PathBuf> {
    most_recent_subdir(&repo_cache_dir(repo_id).join("snapshots"))
}

/// The most recently modified immediate subdirectory of `dir`, if any.
fn most_recent_subdir(dir: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<(PathBuf, std::time::SystemTime)> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect();
    dirs.sort_by_key(|(_, modified)| *modified);
    dirs.pop().map(|(path, _)| path)
}

fn dir_has_content(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut it| it.next().is_some())
        .unwrap_or(false)
}

/// Recursive on-disk size, following directories but not chasing symlinks twice.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

fn build_families() -> Vec<Family> {
    let mut families: Vec<Family> = Vec::new();
    for profile in crate::MODEL_PROFILES.iter().filter(|p| p.downloadable) {
        let key = family_key(profile.label);
        let cache = repo_cache_dir(profile.repo_id);
        let installed = cache.is_dir() && dir_has_content(&cache);
        let variant = Variant {
            profile,
            bits: quant_bits(profile.repo_id),
            mtp_alias: mtp_trigger_alias(profile.label),
            installed,
            size: if installed { dir_size(&cache) } else { 0 },
        };
        match families.iter_mut().find(|f| f.key == key) {
            Some(f) => f.variants.push(variant),
            None => families.push(Family {
                key,
                variants: vec![variant],
            }),
        }
    }
    for family in &mut families {
        family.variants.sort_by_key(|v| v.bits.unwrap_or(99));
    }
    families
}

fn format_bytes(num: u64) -> String {
    let mut value = num as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if value < 1024.0 {
            return format!("{value:.1} {unit}");
        }
        value /= 1024.0;
    }
    format!("{value:.1} PB")
}

// ---------------------------------------------------------------------------
// Background job (download or server) streamed from a child process
// ---------------------------------------------------------------------------

enum JobMsg {
    Line(String),
}

const LOG_CAP: usize = 1000;
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

struct Job {
    rx: Receiver<JobMsg>,
    child: Option<Child>,
    log: Vec<String>,
    done: Option<i32>,
    /// When set, polled each tick for the live byte counter (downloads only).
    watch_dir: Option<PathBuf>,
    bytes: u64,
    speed: f64,
    /// Recent `speed` samples (downloads only), newest last, capped at `SPEED_HISTORY_CAP`.
    speed_history: Vec<u64>,
    last_poll: Option<(Instant, u64)>,
    spinner: usize,
}

const SPEED_HISTORY_CAP: usize = 120;

impl Job {
    fn spawn(mut cmd: Command, watch_dir: Option<PathBuf>) -> io::Result<Job> {
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd.spawn()?;
        let (tx, rx) = mpsc::channel();
        for pipe in [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn io::Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let tx = tx.clone();
            thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    if tx.send(JobMsg::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }
        Ok(Job {
            rx,
            child: Some(child),
            log: Vec::new(),
            done: None,
            watch_dir,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        })
    }

    /// A finished, processless job that just surfaces a launch error.
    fn failed(message: String) -> Job {
        let (_tx, rx) = mpsc::channel();
        Job {
            rx,
            child: None,
            log: vec![message],
            done: Some(-1),
            watch_dir: None,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        }
    }

    /// A still-running, processless job carrying a fixed log (test-only).
    #[cfg(test)]
    fn running_with_log(log: Vec<String>) -> Job {
        let (_tx, rx) = mpsc::channel();
        Job {
            rx,
            child: None,
            log,
            done: None,
            watch_dir: None,
            bytes: 0,
            speed: 0.0,
            speed_history: Vec::new(),
            last_poll: None,
            spinner: 0,
        }
    }

    fn tick(&mut self) {
        while let Ok(JobMsg::Line(line)) = self.rx.try_recv() {
            self.log.push(line);
            if self.log.len() > LOG_CAP {
                let overflow = self.log.len() - LOG_CAP;
                self.log.drain(0..overflow);
            }
        }
        if self.done.is_none()
            && let Some(child) = &mut self.child
            && let Ok(Some(status)) = child.try_wait()
        {
            self.done = Some(status.code().unwrap_or(-1));
        }
        if let Some(dir) = &self.watch_dir {
            let now = Instant::now();
            let bytes = dir_size(dir);
            if let Some((last_t, last_b)) = self.last_poll {
                let dt = now.duration_since(last_t).as_secs_f64();
                if dt > 0.0 {
                    let inst = bytes.saturating_sub(last_b) as f64 / dt;
                    self.speed = if self.speed == 0.0 {
                        inst
                    } else {
                        0.6 * self.speed + 0.4 * inst
                    };
                }
            }
            self.last_poll = Some((now, bytes));
            self.bytes = bytes;
            self.speed_history.push(self.speed as u64);
            if self.speed_history.len() > SPEED_HISTORY_CAP {
                let overflow = self.speed_history.len() - SPEED_HISTORY_CAP;
                self.speed_history.drain(0..overflow);
            }
        }
        self.spinner = (self.spinner + 1) % SPINNER.len();
    }

    fn cancel(&mut self) {
        if self.done.is_none()
            && let Some(child) = &mut self.child
        {
            let _ = child.kill();
            let _ = child.wait();
            self.done = Some(-130);
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
    Downloads,
    Serve,
}

#[derive(Clone, Copy, PartialEq)]
enum Focus {
    Sidebar,
    Content,
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Families,
    Precision,
    Mtp,
    Destination,
}

#[derive(Clone, Copy, PartialEq)]
enum ServeFocus {
    List,
    Host,
    Port,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DownloadMode {
    Direct,
    Mtp,
}

impl DownloadMode {
    fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Mtp => "mtp",
        }
    }
}

#[derive(Clone, Copy)]
struct PendingDownload {
    family_idx: usize,
    precision_idx: usize,
    with_mtp: bool,
}

struct DownloadTask {
    id: u64,
    label: String,
    repo_id: &'static str,
    preset: Option<&'static str>,
    mode: DownloadMode,
    subcmd: &'static str,
    target: String,
    dest: Option<PathBuf>,
    watch_dir: PathBuf,
    job: Option<Job>,
    cancelled: bool,
}

impl DownloadTask {
    fn status_label(&self) -> String {
        if self.cancelled {
            return "cancelled".into();
        }
        match self.job.as_ref().and_then(|job| job.done) {
            None if self.job.is_none() => "queued".into(),
            Some(0) => "ready".into(),
            Some(code) => format!("failed ({code})"),
            None => "running".into(),
        }
    }

    fn status_style(&self) -> Style {
        if self.cancelled {
            return Style::default().fg(Color::DarkGray);
        }
        match self.job.as_ref().and_then(|job| job.done) {
            None if self.job.is_none() => Style::default().fg(Color::Yellow),
            Some(0) => Style::default().fg(Color::Green),
            Some(_) => Style::default().fg(Color::Red),
            None => Style::default().fg(Color::Cyan),
        }
    }

    fn output_path(&self) -> Option<PathBuf> {
        if let Some(dest) = &self.dest {
            return Some(dest.clone());
        }
        self.job
            .as_ref()
            .and_then(|job| parse_output_path_from_log(&job.log))
    }

    fn is_queued(&self) -> bool {
        !self.cancelled && self.job.is_none()
    }

    fn is_running(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.done.is_none())
    }

    fn is_ready(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.done == Some(0))
    }

    fn spawn(&mut self) {
        if !self.is_queued() {
            return;
        }
        let mut cmd_result = std::env::current_exe().map(Command::new);
        if let Ok(cmd) = &mut cmd_result {
            cmd.arg(self.subcmd).arg(&self.target);
            if let Some(dest) = &self.dest {
                let flag = if self.mode == DownloadMode::Mtp {
                    "--output"
                } else {
                    "--dest"
                };
                cmd.arg(flag).arg(dest);
            }
        }
        self.job = Some(match cmd_result {
            Ok(cmd) => Job::spawn(cmd, Some(self.watch_dir.clone()))
                .unwrap_or_else(|err| Job::failed(format!("failed to launch download: {err}"))),
            Err(err) => Job::failed(format!("failed to resolve ax-engine executable: {err}")),
        });
    }

    fn tick(&mut self) -> bool {
        let Some(job) = &mut self.job else {
            return false;
        };
        let before = job.done;
        job.tick();
        before.is_none() && job.done == Some(0)
    }

    fn cancel(&mut self) {
        if let Some(job) = &mut self.job {
            job.cancel();
        } else {
            self.cancelled = true;
        }
    }
}

#[derive(Clone)]
struct DirEntry {
    label: String,
    path: PathBuf,
}

struct DirectoryPicker {
    current: PathBuf,
    entries: Vec<DirEntry>,
    selected: usize,
    error: Option<String>,
}

impl DirectoryPicker {
    fn new() -> Self {
        let mut picker = Self {
            current: nearest_existing_dir(&crate::default_hf_cache_root()),
            entries: Vec::new(),
            selected: 0,
            error: None,
        };
        picker.refresh();
        picker
    }

    fn set_current(&mut self, path: PathBuf) {
        self.current = nearest_existing_dir(&path);
        self.selected = 0;
        self.refresh();
    }

    fn refresh(&mut self) {
        self.entries.clear();
        self.error = None;
        if let Some(parent) = self.current.parent() {
            self.entries.push(DirEntry {
                label: "../".into(),
                path: parent.to_path_buf(),
            });
        }
        match std::fs::read_dir(&self.current) {
            Ok(entries) => {
                let mut dirs: Vec<DirEntry> = entries
                    .flatten()
                    .filter_map(|entry| {
                        let meta = entry.metadata().ok()?;
                        if !meta.is_dir() {
                            return None;
                        }
                        let name = entry.file_name().to_string_lossy().into_owned();
                        Some(DirEntry {
                            label: format!("{name}/"),
                            path: entry.path(),
                        })
                    })
                    .collect();
                dirs.sort_by_key(|entry| entry.label.to_ascii_lowercase());
                self.entries.extend(dirs);
            }
            Err(err) => {
                self.error = Some(format!("cannot read {}: {err}", self.current.display()));
            }
        }
        if self.selected >= self.entries.len() {
            self.selected = self.entries.len().saturating_sub(1);
        }
    }

    fn selected_path(&self) -> Option<PathBuf> {
        self.entries
            .get(self.selected)
            .map(|entry| entry.path.clone())
    }

    fn enter_selected(&mut self) {
        if let Some(path) = self.selected_path() {
            self.set_current(path);
        }
    }
}

struct App {
    quit: bool,
    tab: Tab,
    focus: Focus,
    families: Vec<Family>,

    // Models wizard
    stage: Stage,
    family_idx: usize,
    precision_idx: usize,
    mtp_idx: usize, // 0 = yes, 1 = no
    pending_download: Option<PendingDownload>,
    directory_picker: DirectoryPicker,
    /// Case-insensitive substring filter over the family list (Families stage, `/` to edit).
    filter: String,
    filtering: bool,

    // Downloads
    downloads: Vec<DownloadTask>,
    download_idx: usize,
    next_download_id: u64,

    // Serve
    serve_focus: ServeFocus,
    serve_idx: usize,
    host: String,
    port: String,
    server: Option<Job>,
    server_url: Option<String>,
    /// Set once the server job's log confirms it actually bound (not just spawned).
    server_ready: bool,

    // Click-target rects recorded during the last draw (immediate-mode hit-testing).
    sidebar_rect: Cell<Rect>,
    content_list_rect: Cell<Rect>,
    show_help: bool,
}

/// Flattened installed (family, variant) pairs for the Serve list.
fn installed_variants(families: &[Family]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (fi, family) in families.iter().enumerate() {
        for (vi, variant) in family.variants.iter().enumerate() {
            if variant.installed {
                out.push((fi, vi));
            }
        }
    }
    out
}

impl App {
    fn new() -> App {
        let families = build_families();
        App {
            quit: false,
            tab: Tab::Models,
            focus: Focus::Content,
            families,
            stage: Stage::Families,
            family_idx: 0,
            precision_idx: 0,
            mtp_idx: 0,
            pending_download: None,
            directory_picker: DirectoryPicker::new(),
            filter: String::new(),
            filtering: false,
            downloads: Vec::new(),
            download_idx: 0,
            next_download_id: 1,
            serve_focus: ServeFocus::List,
            serve_idx: 0,
            host: "127.0.0.1".into(),
            port: "8080".into(),
            server: None,
            server_url: None,
            server_ready: false,
            sidebar_rect: Cell::new(Rect::default()),
            content_list_rect: Cell::new(Rect::default()),
            show_help: false,
        }
    }

    fn reload_families(&mut self) {
        self.families = build_families();
    }

    /// Indices into `self.families` matching the active filter (all of them if empty).
    fn filtered_family_indices(&self) -> Vec<usize> {
        if self.filter.is_empty() {
            return (0..self.families.len()).collect();
        }
        let needle = self.filter.to_ascii_lowercase();
        self.families
            .iter()
            .enumerate()
            .filter(|(_, f)| f.key.to_ascii_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect()
    }

    /// Move `family_idx` to the previous/next entry within the filtered list.
    fn move_family_selection(&mut self, delta: i32) {
        let indices = self.filtered_family_indices();
        if indices.is_empty() {
            return;
        }
        let pos = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        let new_pos = if delta < 0 {
            pos.saturating_sub(1)
        } else {
            (pos + 1).min(indices.len() - 1)
        };
        self.family_idx = indices[new_pos];
    }

    /// After the filter text changes, snap `family_idx` back into the filtered set if needed.
    fn clamp_family_idx_to_filter(&mut self) {
        let indices = self.filtered_family_indices();
        if let Some(&first) = indices.first()
            && !indices.contains(&self.family_idx)
        {
            self.family_idx = first;
        }
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
            let mut reload = false;
            for task in &mut self.downloads {
                if task.tick() {
                    reload = true;
                }
            }
            self.start_next_queued_download();
            if reload {
                self.reload_families();
            }
            if let Some(job) = &mut self.server {
                job.tick();
            }
            self.update_server_ready();
        }
        for task in &mut self.downloads {
            task.cancel();
        }
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        Ok(())
    }

    // -- input ----------------------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        if self.show_help {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('?')) {
                self.show_help = false;
            }
            return;
        }
        // Ignored while typing into a serve text field or the family filter.
        let typing = (self.tab == Tab::Serve
            && self.focus == Focus::Content
            && matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port))
            || (self.tab == Tab::Models
                && self.focus == Focus::Content
                && self.stage == Stage::Families
                && self.filtering);
        if !typing && matches!(key.code, KeyCode::Char('?')) {
            self.show_help = true;
            return;
        }
        if !typing && matches!(key.code, KeyCode::Char('q')) {
            self.quit = true;
            return;
        }
        match self.focus {
            Focus::Sidebar => self.on_key_sidebar(key.code),
            Focus::Content => match self.tab {
                Tab::Models => self.on_key_models(key.code),
                Tab::Downloads => self.on_key_downloads(key.code),
                Tab::Serve => self.on_key_serve(key.code),
            },
        }
    }

    fn on_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollDown => self.scroll(KeyCode::Down),
            MouseEventKind::ScrollUp => self.scroll(KeyCode::Up),
            MouseEventKind::Down(MouseButton::Left) => self.on_click(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Wheel scroll routes to the focused pane's existing up/down handler.
    fn scroll(&mut self, code: KeyCode) {
        match self.focus {
            Focus::Sidebar => self.on_key_sidebar(code),
            Focus::Content => match self.tab {
                Tab::Models => self.on_key_models(code),
                Tab::Downloads => self.on_key_downloads(code),
                Tab::Serve => self.on_key_serve(code),
            },
        }
    }

    fn on_click(&mut self, col: u16, row: u16) {
        // Sidebar click selects a tab.
        if let Some(idx) = row_in_rect(self.sidebar_rect.get(), col, row) {
            if idx < 3 {
                self.focus = Focus::Sidebar;
                self.tab = tab_from_index(idx);
            }
            return;
        }
        // Content-list click selects the row (and drills in for the Models wizard).
        if let Some(idx) = row_in_rect(self.content_list_rect.get(), col, row) {
            self.focus = Focus::Content;
            match self.tab {
                Tab::Models => match self.stage {
                    Stage::Families => {
                        if let Some(&real) = self.filtered_family_indices().get(idx) {
                            self.family_idx = real;
                            self.on_key_models(KeyCode::Enter);
                        }
                    }
                    Stage::Precision if idx < self.families[self.family_idx].variants.len() => {
                        self.precision_idx = idx;
                        self.on_key_models(KeyCode::Enter);
                    }
                    Stage::Destination if idx < self.directory_picker.entries.len() => {
                        self.directory_picker.selected = idx;
                        self.directory_picker.enter_selected();
                    }
                    _ => {}
                },
                Tab::Downloads => {
                    if idx < self.downloads.len() {
                        self.download_idx = idx;
                    }
                }
                Tab::Serve => {
                    if idx < installed_variants(&self.families).len() {
                        self.serve_focus = ServeFocus::List;
                        self.serve_idx = idx;
                    }
                }
            }
        }
    }

    fn on_key_sidebar(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.tab = tab_from_index(tab_index(self.tab).saturating_sub(1));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let next = (tab_index(self.tab) + 1).min(2);
                self.tab = tab_from_index(next);
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => {
                self.focus = Focus::Content;
            }
            _ => {}
        }
    }

    fn on_key_models(&mut self, code: KeyCode) {
        match self.stage {
            Stage::Families if self.filtering => match code {
                KeyCode::Char(c) => {
                    self.filter.push(c);
                    self.clamp_family_idx_to_filter();
                }
                KeyCode::Backspace => {
                    self.filter.pop();
                    self.clamp_family_idx_to_filter();
                }
                KeyCode::Enter | KeyCode::Esc => self.filtering = false,
                _ => {}
            },
            Stage::Families => match code {
                KeyCode::Up | KeyCode::Char('k') => self.move_family_selection(-1),
                KeyCode::Down | KeyCode::Char('j') => self.move_family_selection(1),
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    self.precision_idx = 0;
                    self.stage = Stage::Precision;
                }
                KeyCode::Char('/') => self.filtering = true,
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    if self.filter.is_empty() {
                        self.focus = Focus::Sidebar;
                    } else {
                        self.filter.clear();
                    }
                }
                _ => {}
            },
            Stage::Precision => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.precision_idx = self.precision_idx.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let n = self.families[self.family_idx].variants.len();
                    if self.precision_idx + 1 < n {
                        self.precision_idx += 1;
                    }
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    let variant = &self.families[self.family_idx].variants[self.precision_idx];
                    if variant.mtp_alias.is_some() {
                        self.mtp_idx = 0;
                        self.stage = Stage::Mtp;
                    } else {
                        self.open_destination_picker(false);
                    }
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.stage = Stage::Families,
                _ => {}
            },
            Stage::Mtp => match code {
                KeyCode::Up | KeyCode::Down | KeyCode::Char('k') | KeyCode::Char('j') => {
                    self.mtp_idx ^= 1;
                }
                KeyCode::Char('y') => self.open_destination_picker(true),
                KeyCode::Char('n') => self.open_destination_picker(false),
                KeyCode::Enter => self.open_destination_picker(self.mtp_idx == 0),
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.stage = Stage::Precision,
                _ => {}
            },
            Stage::Destination => self.on_key_destination(code),
        }
    }

    fn on_key_destination(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.directory_picker.selected = self.directory_picker.selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.directory_picker.selected + 1 < self.directory_picker.entries.len() {
                    self.directory_picker.selected += 1;
                }
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.directory_picker.enter_selected();
            }
            KeyCode::Char('d') => self.enqueue_pending_download(None),
            KeyCode::Char('s') => {
                let Some(pending) = self.pending_download else {
                    return;
                };
                let family = &self.families[pending.family_idx];
                let variant = &family.variants[pending.precision_idx];
                let dest = explicit_destination_path(
                    &self.directory_picker.current,
                    variant,
                    pending.with_mtp,
                );
                match validate_writable_parent(&self.directory_picker.current) {
                    Ok(()) => self.enqueue_pending_download(Some(dest)),
                    Err(err) => self.directory_picker.error = Some(err),
                }
            }
            KeyCode::Char('~') => {
                if let Some(home) = home_dir() {
                    self.directory_picker.set_current(home);
                }
            }
            KeyCode::Char('r') => self.directory_picker.set_current(PathBuf::from("/")),
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                let back_to_mtp = self.pending_download.is_some_and(|pending| {
                    self.families[pending.family_idx].variants[pending.precision_idx]
                        .mtp_alias
                        .is_some()
                });
                self.stage = if back_to_mtp {
                    Stage::Mtp
                } else {
                    Stage::Precision
                };
            }
            _ => {}
        }
    }

    fn on_key_downloads(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.download_idx = self.download_idx.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.download_idx + 1 < self.downloads.len() {
                    self.download_idx += 1;
                }
            }
            KeyCode::Enter => self.start_server_for_download(),
            KeyCode::Char('x') | KeyCode::Char('c') => {
                if let Some(task) = self.downloads.get_mut(self.download_idx) {
                    task.cancel();
                }
                self.start_next_queued_download();
            }
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.focus = Focus::Sidebar,
            _ => {}
        }
    }

    fn on_key_serve(&mut self, code: KeyCode) {
        match self.serve_focus {
            ServeFocus::List => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.serve_idx = self.serve_idx.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.serve_idx + 1 < installed_variants(&self.families).len() {
                        self.serve_idx += 1;
                    }
                }
                KeyCode::Tab => self.serve_focus = ServeFocus::Host,
                KeyCode::Enter => self.start_server(),
                KeyCode::Char('x') => self.stop_server(),
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.focus = Focus::Sidebar,
                _ => {}
            },
            ServeFocus::Host => self.edit_field(code, true),
            ServeFocus::Port => self.edit_field(code, false),
        }
    }

    fn edit_field(&mut self, code: KeyCode, is_host: bool) {
        let field = if is_host {
            &mut self.host
        } else {
            &mut self.port
        };
        match code {
            KeyCode::Char(c) => field.push(c),
            KeyCode::Backspace => {
                field.pop();
            }
            KeyCode::Tab => {
                self.serve_focus = if is_host {
                    ServeFocus::Port
                } else {
                    ServeFocus::List
                };
            }
            KeyCode::Enter | KeyCode::Esc => self.serve_focus = ServeFocus::List,
            _ => {}
        }
    }

    // -- actions --------------------------------------------------------------

    fn open_destination_picker(&mut self, with_mtp: bool) {
        self.pending_download = Some(PendingDownload {
            family_idx: self.family_idx,
            precision_idx: self.precision_idx,
            with_mtp,
        });
        self.directory_picker.refresh();
        self.stage = Stage::Destination;
    }

    fn enqueue_pending_download(&mut self, dest: Option<PathBuf>) {
        let Some(pending) = self.pending_download else {
            return;
        };
        self.enqueue_download(pending, dest);
        self.pending_download = None;
        self.tab = Tab::Downloads;
        self.focus = Focus::Content;
        self.stage = Stage::Precision;
    }

    fn enqueue_download(&mut self, pending: PendingDownload, dest: Option<PathBuf>) {
        let family = &self.families[pending.family_idx];
        let variant = &family.variants[pending.precision_idx];
        let mode = if pending.with_mtp {
            DownloadMode::Mtp
        } else {
            DownloadMode::Direct
        };
        let (subcmd, target) = if pending.with_mtp {
            (
                "download-mtp",
                variant.mtp_alias.unwrap_or(variant.profile.label),
            )
        } else {
            ("download", variant.profile.label)
        };
        let watch_dir = dest
            .clone()
            .unwrap_or_else(|| repo_cache_dir(variant.profile.repo_id));
        let label = format!("{} {}", family.key, variant.precision());
        let task = DownloadTask {
            id: self.next_download_id,
            label,
            repo_id: variant.profile.repo_id,
            preset: variant.profile.preset,
            mode,
            subcmd,
            target: target.to_string(),
            dest,
            watch_dir,
            job: None,
            cancelled: false,
        };
        self.next_download_id += 1;
        self.downloads.push(task);
        self.download_idx = self.downloads.len().saturating_sub(1);
        self.start_next_queued_download();
    }

    fn start_next_queued_download(&mut self) {
        if self.downloads.iter().any(DownloadTask::is_running) {
            return;
        }
        if let Some(task) = self.downloads.iter_mut().find(|task| task.is_queued()) {
            task.spawn();
        }
    }

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

    fn start_server(&mut self) {
        if self.port_error().is_some() {
            return;
        }
        if self.server.as_ref().is_some_and(|j| j.done.is_none()) {
            return;
        }
        let pairs = installed_variants(&self.families);
        let Some(&(fi, vi)) = pairs.get(self.serve_idx) else {
            return;
        };
        let profile = self.families[fi].variants[vi].profile;
        // Prefer the exact snapshot directory over the preset+hf-cache scan: with
        // more than one precision of a model family installed, the scan can't
        // tell them apart (they share the preset's alias substring) and errors
        // out with "multiple Hugging Face cache candidates". Knowing the repo id
        // here means there's no ambiguity to resolve.
        let artifacts_dir = repo_snapshot_dir(profile.repo_id);
        self.spawn_server(profile.preset, artifacts_dir);
    }

    fn start_server_for_download(&mut self) {
        if self.port_error().is_some() {
            return;
        }
        if self.server.as_ref().is_some_and(|j| j.done.is_none()) {
            return;
        }
        let Some(task) = self.downloads.get(self.download_idx) else {
            return;
        };
        if !task.is_ready() {
            return;
        }
        let artifacts_dir = task.output_path().or_else(|| {
            if task.mode == DownloadMode::Direct && task.preset.is_none() {
                repo_snapshot_dir(task.repo_id)
            } else {
                None
            }
        });
        self.spawn_server(task.preset, artifacts_dir);
    }

    fn spawn_server(&mut self, preset: Option<&str>, artifacts_dir: Option<PathBuf>) {
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
    }

    // -- rendering ------------------------------------------------------------

    fn draw(&self, frame: &mut Frame) {
        // Cleared each frame; the active content list re-records it in render_list,
        // so stages without a list (Mtp/Progress) leave no stale click target.
        self.content_list_rect.set(Rect::default());
        let outer =
            Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(frame.area());
        let body = Layout::horizontal([Constraint::Length(18), Constraint::Min(0)]).split(outer[0]);
        self.draw_sidebar(frame, body[0]);
        match self.tab {
            Tab::Models => self.draw_models(frame, body[1]),
            Tab::Downloads => self.draw_downloads(frame, body[1]),
            Tab::Serve => self.draw_serve(frame, body[1]),
        }
        frame.render_widget(
            Paragraph::new(self.footer()).style(Style::default().fg(Color::DarkGray)),
            outer[1],
        );
        if self.show_help {
            self.draw_help(frame, frame.area());
        }
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        self.sidebar_rect.set(area);
        let active = self.focus == Focus::Sidebar;
        let items = [
            ("Models", Tab::Models),
            ("Downloads", Tab::Downloads),
            ("Serve", Tab::Serve),
        ]
        .map(|(name, tab)| {
            let selected = self.tab == tab;
            let marker = if selected { "▸ " } else { "  " };
            let mut style = Style::default();
            if selected {
                style = style.fg(Color::Cyan).add_modifier(Modifier::BOLD);
            }
            ListItem::new(Line::from(format!("{marker}{name}"))).style(style)
        });
        let border = if active { Color::Cyan } else { Color::DarkGray };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .title(" AX Engine ");
        frame.render_widget(List::new(items).block(block), area);
    }

    fn draw_models(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
        self.draw_breadcrumb(frame, chunks[0]);
        match self.stage {
            Stage::Families => self.draw_families(frame, chunks[1]),
            Stage::Precision => self.draw_precision(frame, chunks[1]),
            Stage::Mtp => self.draw_mtp(frame, chunks[1]),
            Stage::Destination => self.draw_destination(frame, chunks[1]),
        }
    }

    fn draw_breadcrumb(&self, frame: &mut Frame, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let past = Style::default().fg(Color::Gray);
        let current = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let stage_style = |target: Stage| {
            if self.stage == target {
                current
            } else if stage_index(self.stage) > stage_index(target) {
                past
            } else {
                dim
            }
        };

        let mut spans = vec![Span::styled("Family", stage_style(Stage::Families))];
        let Some(family) = self.families.get(self.family_idx) else {
            frame.render_widget(Paragraph::new(Line::from(spans)), area);
            return;
        };
        spans.push(Span::styled(
            format!(": {}", family.key),
            stage_style(Stage::Families),
        ));
        spans.push(Span::styled("  ›  ", dim));

        let variant = family.variants.get(self.precision_idx);
        let precision_text = match (self.stage, variant) {
            (Stage::Families, _) => "Precision".to_string(),
            (_, Some(v)) => format!("Precision: {}", v.precision()),
            (_, None) => "Precision".to_string(),
        };
        spans.push(Span::styled(precision_text, stage_style(Stage::Precision)));

        if variant.is_some_and(|v| v.mtp_alias.is_some()) {
            spans.push(Span::styled("  ›  ", dim));
            let mtp_text = if matches!(self.stage, Stage::Mtp | Stage::Destination) {
                format!("MTP: {}", if self.mtp_idx == 0 { "yes" } else { "no" })
            } else {
                "MTP".to_string()
            };
            spans.push(Span::styled(mtp_text, stage_style(Stage::Mtp)));
        }

        spans.push(Span::styled("  ›  ", dim));
        spans.push(Span::styled("Destination", stage_style(Stage::Destination)));

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_families(&self, frame: &mut Frame, area: Rect) {
        let indices = self.filtered_family_indices();
        let rows: Vec<ListItem> = indices
            .iter()
            .map(|&i| {
                let family = &self.families[i];
                let installed = family.variants.iter().filter(|v| v.installed).count();
                let status = if installed == family.variants.len() {
                    Span::styled("installed", Style::default().fg(Color::Green))
                } else if installed > 0 {
                    Span::styled(
                        format!("{installed}/{} installed", family.variants.len()),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    Span::styled("--", Style::default().fg(Color::DarkGray))
                };
                let mtp = if family.has_mtp() {
                    Span::styled("  ⚡MTP", Style::default().fg(Color::Magenta))
                } else {
                    Span::raw("")
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<16}", family.key),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("{:<30}", family.quant_summary()),
                        Style::default().fg(Color::Gray),
                    ),
                    status,
                    mtp,
                ]))
            })
            .collect();
        let rows = if rows.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No families match the filter.",
                Style::default().fg(Color::Yellow),
            )))]
        } else {
            rows
        };
        let selected = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        let title = if self.filtering {
            format!(" Models — filter: {}_ ", self.filter)
        } else if !self.filter.is_empty() {
            format!(" Models — filter: {} (Esc clears) ", self.filter)
        } else {
            " Models — / to filter ".to_string()
        };
        self.render_list(
            frame,
            area,
            &title,
            rows,
            selected,
            self.focus == Focus::Content,
        );
    }

    fn draw_precision(&self, frame: &mut Frame, area: Rect) {
        let family = &self.families[self.family_idx];
        let rows: Vec<ListItem> = family
            .variants
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut precision = v.precision();
                if i == 0 {
                    precision.push_str("  (recommended)");
                }
                let size = if v.installed {
                    Span::styled(
                        format!("{:<12}", format_bytes(v.size)),
                        Style::default().fg(Color::Gray),
                    )
                } else {
                    Span::styled(
                        format!("{:<12}", "--"),
                        Style::default().fg(Color::DarkGray),
                    )
                };
                let mtp = if v.mtp_alias.is_some() {
                    Span::styled("⚡MTP  ", Style::default().fg(Color::Magenta))
                } else {
                    Span::raw("      ")
                };
                let status = if v.installed {
                    Span::styled("installed", Style::default().fg(Color::Green))
                } else {
                    Span::styled("--", Style::default().fg(Color::DarkGray))
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{precision:<24}"),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    size,
                    mtp,
                    status,
                ]))
            })
            .collect();
        let title = format!(" {} — select precision ", family.key);
        self.render_list(
            frame,
            area,
            &title,
            rows,
            self.precision_idx,
            self.focus == Focus::Content,
        );
    }

    fn draw_mtp(&self, frame: &mut Frame, area: Rect) {
        let family = &self.families[self.family_idx];
        let variant = &family.variants[self.precision_idx];
        let yes = self.mtp_idx == 0;
        let opt = |label: &str, sel: bool| {
            let style = if sel {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(Span::styled(format!("  {label}  "), style))
        };
        let text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{} {}", family.key, variant.precision()),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(" supports an "),
                Span::styled(
                    "MTP speculative-decoding accelerator",
                    Style::default().fg(Color::Yellow),
                ),
                Span::raw("."),
            ]),
            Line::raw(""),
            Line::raw("It downloads alongside the base weights and speeds up decoding."),
            Line::raw(""),
            Line::from(Span::styled(
                "Include it?",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            opt("Yes — base weights + MTP", yes),
            opt("No — base weights only", !yes),
            Line::raw(""),
            Line::from(Span::styled(
                "[y]/[n] or ↑↓ + Enter   ·   Esc cancel",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" MTP accelerator ");
        frame.render_widget(
            Paragraph::new(text).block(block).wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_destination(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(7), Constraint::Min(0)]).split(area);
        let Some(pending) = self.pending_download else {
            frame.render_widget(
                Paragraph::new("No pending download.").block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title(" Download destination "),
                ),
                area,
            );
            return;
        };
        let family = &self.families[pending.family_idx];
        let variant = &family.variants[pending.precision_idx];
        let mode = if pending.with_mtp {
            "MTP package"
        } else {
            "direct model"
        };
        let explicit =
            explicit_destination_path(&self.directory_picker.current, variant, pending.with_mtp);
        let error = self
            .directory_picker
            .error
            .as_ref()
            .map(|err| Line::from(Span::styled(err.clone(), Style::default().fg(Color::Red))))
            .unwrap_or_else(|| {
                Line::from(Span::styled(
                    "d default cache · s select current directory · Enter open · ~ home · r root",
                    Style::default().fg(Color::DarkGray),
                ))
            });
        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled(
                    format!("{} {}", family.key, variant.precision()),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(" · {mode}")),
            ]),
            Line::from(format!(
                "Default: {}",
                crate::default_hf_cache_root().display()
            )),
            Line::from(format!("Custom: {}", explicit.display())),
            error,
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Download destination "),
        );
        frame.render_widget(header, chunks[0]);

        let rows: Vec<ListItem> = self
            .directory_picker
            .entries
            .iter()
            .map(|entry| ListItem::new(Line::from(entry.label.clone())))
            .collect();
        self.render_list(
            frame,
            chunks[1],
            &format!(" {} ", self.directory_picker.current.display()),
            rows,
            self.directory_picker.selected,
            self.focus == Focus::Content,
        );
    }

    fn draw_downloads(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Min(6),
            Constraint::Length(6),
            Constraint::Min(5),
        ])
        .split(area);
        let rows: Vec<ListItem> = if self.downloads.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No downloads queued. Add one from the Models tab.",
                Style::default().fg(Color::Yellow),
            )))]
        } else {
            self.downloads
                .iter()
                .map(|task| {
                    let spin = task
                        .job
                        .as_ref()
                        .filter(|job| job.done.is_none())
                        .map(|job| SPINNER[job.spinner])
                        .unwrap_or(' ');
                    let dest = task
                        .dest
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "HF cache".into());
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("#{:<3}", task.id), Style::default().fg(Color::Gray)),
                        Span::styled(format!("{:<13}", task.status_label()), task.status_style()),
                        Span::raw(format!("{spin} ")),
                        Span::styled(
                            format!("{:<24}", task.label),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{:<7}", task.mode.label()),
                            Style::default().fg(Color::Gray),
                        ),
                        Span::raw(dest),
                    ]))
                })
                .collect()
        };
        self.render_list(
            frame,
            chunks[0],
            " Downloads ",
            rows,
            self.download_idx,
            self.focus == Focus::Content,
        );

        let selected = self.downloads.get(self.download_idx);
        let details = selected
            .map(|task| {
                let path = task
                    .output_path()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "pending".into());
                let (bytes, speed) = task
                    .job
                    .as_ref()
                    .map(|job| (job.bytes, job.speed))
                    .unwrap_or((0, 0.0));
                vec![
                    Line::from(format!("target: {}", task.target)),
                    Line::from(format!("output: {path}")),
                    Line::from(format!(
                        "progress: {} · {}/s",
                        format_bytes(bytes),
                        format_bytes(speed as u64)
                    )),
                ]
            })
            .unwrap_or_else(|| vec![Line::raw("")]);
        let details_block = Block::default()
            .borders(Borders::ALL)
            .title(" Selected download (Enter serve ready · x cancel) ");
        let details_inner = details_block.inner(chunks[1]);
        frame.render_widget(details_block, chunks[1]);
        let inner_chunks =
            Layout::vertical([Constraint::Length(3), Constraint::Length(1)]).split(details_inner);
        frame.render_widget(Paragraph::new(details), inner_chunks[0]);
        let history: &[u64] = selected
            .and_then(|task| task.job.as_ref())
            .map(|job| job.speed_history.as_slice())
            .unwrap_or(&[]);
        frame.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(Color::Cyan)),
            inner_chunks[1],
        );
        self.draw_log(
            frame,
            chunks[2],
            selected.and_then(|task| task.job.as_ref()),
            " Download log ",
        );
    }

    fn draw_serve(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(6),
            Constraint::Length(1),
            Constraint::Min(3),
        ])
        .split(area);

        let pairs = installed_variants(&self.families);
        let rows: Vec<ListItem> = if pairs.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No installed models. Download one in the Models tab first.",
                Style::default().fg(Color::Yellow),
            )))]
        } else {
            pairs
                .iter()
                .map(|&(fi, vi)| {
                    let v = &self.families[fi].variants[vi];
                    let preset = v.profile.preset.unwrap_or("--");
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{:<22}", v.profile.label),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{preset:<16}"), Style::default().fg(Color::Gray)),
                        Span::raw(v.profile.repo_id),
                    ]))
                })
                .collect()
        };
        let list_active = self.focus == Focus::Content && self.serve_focus == ServeFocus::List;
        self.render_list(
            frame,
            chunks[0],
            " Installed models ",
            rows,
            self.serve_idx,
            list_active,
        );

        let host_line = field_line(
            "Host",
            &self.host,
            self.focus == Focus::Content && self.serve_focus == ServeFocus::Host,
        );
        let port_line = field_line(
            "Port",
            &self.port,
            self.focus == Focus::Content && self.serve_focus == ServeFocus::Port,
        );
        let status = match (&self.server_url, &self.server) {
            (Some(url), Some(job)) if job.done.is_none() && self.server_ready => {
                Line::from(Span::styled(
                    format!("running at {url}"),
                    Style::default().fg(Color::Green),
                ))
            }
            (Some(_), Some(job)) if job.done.is_none() => Line::from(Span::styled(
                "starting…",
                Style::default().fg(Color::Yellow),
            )),
            (_, Some(job)) if job.done.is_some() => Line::from(Span::styled(
                "failed to start (see log)",
                Style::default().fg(Color::Red),
            )),
            _ => Line::from(Span::styled(
                "stopped",
                Style::default().fg(Color::DarkGray),
            )),
        };
        let error_line = match self.port_error() {
            Some(err) => Line::from(Span::styled(err, Style::default().fg(Color::Red))),
            None => Line::raw(""),
        };
        frame.render_widget(
            Paragraph::new(vec![host_line, port_line, error_line, status]).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Server (Enter start · x stop · Tab fields) "),
            ),
            chunks[1],
        );
        frame.render_widget(Paragraph::new(""), chunks[2]);
        self.draw_log(frame, chunks[3], self.server.as_ref(), " Server log ");
    }

    fn draw_log(&self, frame: &mut Frame, area: Rect, job: Option<&Job>, title: &str) {
        let height = area.height.saturating_sub(2) as usize;
        let lines: Vec<Line> = match job {
            Some(job) => {
                let start = job.log.len().saturating_sub(height);
                job.log[start..]
                    .iter()
                    .map(|l| Line::raw(l.clone()))
                    .collect()
            }
            None => vec![Line::raw("")],
        };
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(title.to_string()),
            ),
            area,
        );
    }

    fn render_list(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        rows: Vec<ListItem>,
        selected: usize,
        active: bool,
    ) {
        // Record this list as the frame's click target (only one is drawn per frame).
        self.content_list_rect.set(area);
        let border = if active { Color::Cyan } else { Color::DarkGray };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border))
            .title(title.to_string());
        let highlight = if active {
            Style::default()
                .bg(Color::Cyan)
                .fg(Color::Black)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::REVERSED)
        };
        let list = List::new(rows)
            .block(block)
            .highlight_style(highlight)
            .highlight_symbol("▸ ");
        let mut state = ListState::default();
        state.select(Some(selected));
        frame.render_stateful_widget(list, area, &mut state);
    }

    fn footer(&self) -> Line<'static> {
        let help = match self.focus {
            Focus::Sidebar => "↑↓ pick · → enter · ? help · q quit",
            Focus::Content => match (self.tab, self.stage) {
                (Tab::Models, Stage::Families) if self.filtering => {
                    "type to filter · Enter/Esc apply"
                }
                (Tab::Models, Stage::Families) => {
                    "↑↓ move · → choose precision · / filter · ← sidebar · ? help · q quit"
                }
                (Tab::Models, Stage::Precision) => "↑↓ move · → destination · ← back · ? help",
                (Tab::Models, Stage::Mtp) => "y/n or ↑↓+Enter · Esc back",
                (Tab::Models, Stage::Destination) => {
                    "d default · s select dir · Enter open · Esc back"
                }
                (Tab::Downloads, _) => {
                    "↑↓ move · Enter serve ready · x cancel · ← sidebar · ? help"
                }
                (Tab::Serve, _) => {
                    "↑↓ move · Enter start · x stop · Tab fields · ← sidebar · ? help"
                }
            },
        };
        Line::from(format!("  {help}"))
    }

    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let popup = centered_rect(72, 20, area);
        let lines = vec![
            Line::from(Span::styled(
                "AX Engine TUI",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            Line::raw("Models"),
            Line::raw("  Enter/Right selects model, precision, and accelerator."),
            Line::raw("  / filters the family list by name; Enter/Esc closes the filter,"),
            Line::raw("  Esc again (or ← with an empty filter) clears it."),
            Line::raw("  Destination: d uses the shared HF cache; s uses the current directory;"),
            Line::raw("  ~ jumps to your home directory; r jumps to /."),
            Line::raw(""),
            Line::raw("Downloads"),
            Line::raw("  Downloads keep running while you browse other models."),
            Line::raw("  Enter serves a ready download; x cancels a running download."),
            Line::raw(""),
            Line::raw("Serve"),
            Line::raw("  Enter starts the selected installed model; x stops the server."),
            Line::raw("  Tab moves between the list and the host/port fields."),
            Line::raw(""),
            Line::from(Span::styled(
                "Esc or ? closes help. q exits and stops child jobs.",
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
}

fn stage_index(stage: Stage) -> usize {
    match stage {
        Stage::Families => 0,
        Stage::Precision => 1,
        Stage::Mtp => 2,
        Stage::Destination => 3,
    }
}

fn tab_index(tab: Tab) -> usize {
    match tab {
        Tab::Models => 0,
        Tab::Downloads => 1,
        Tab::Serve => 2,
    }
}

fn tab_from_index(index: usize) -> Tab {
    match index {
        0 => Tab::Models,
        1 => Tab::Downloads,
        _ => Tab::Serve,
    }
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn nearest_existing_dir(path: &Path) -> PathBuf {
    let mut candidate = path;
    loop {
        if candidate.is_dir() {
            return candidate.to_path_buf();
        }
        let Some(parent) = candidate.parent() else {
            break;
        };
        if parent == candidate {
            break;
        }
        candidate = parent;
    }
    home_dir()
        .filter(|path| path.is_dir())
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn explicit_destination_path(parent: &Path, variant: &Variant, with_mtp: bool) -> PathBuf {
    let leaf = if with_mtp {
        format!(
            "{}-mtp",
            sanitize_path_segment(variant.mtp_alias.unwrap_or(variant.profile.label))
        )
    } else {
        sanitize_path_segment(variant.profile.label)
    };
    parent.join(leaf)
}

fn sanitize_path_segment(value: &str) -> String {
    let mut out = String::new();
    let mut last_dash = false;
    for ch in value.chars() {
        let keep = ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-');
        if keep {
            out.push(ch);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

fn validate_writable_parent(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("not a directory: {}", path.display()));
    }
    let probe = path.join(format!(".ax-engine-tui-write-test-{}", process::id()));
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(err) => Err(format!("destination is not writable: {err}")),
    }
}

fn parse_output_path_from_log(lines: &[String]) -> Option<PathBuf> {
    for line in lines.iter().rev() {
        if let Some(rest) = line.strip_prefix("Path:") {
            return Some(PathBuf::from(rest.trim()));
        }
        if let Some(rest) = line.strip_prefix("Output dir:") {
            return Some(PathBuf::from(rest.trim()));
        }
    }
    let mut next_is_path = false;
    for line in lines {
        if next_is_path {
            let value = line.trim();
            if !value.is_empty() {
                return Some(PathBuf::from(value));
            }
        }
        next_is_path = line.trim() == "Sidecar ready at:";
    }
    None
}

/// Inner row index of a click inside a bordered widget, if it landed on a content row.
///
/// Assumes a 1-cell border and no vertical scroll offset (the lists are short
/// enough to always fit), so inner row `n` maps to item `n`.  Callers bounds-check
/// the returned index against the actual item count.
fn row_in_rect(rect: Rect, col: u16, row: u16) -> Option<usize> {
    if rect.width < 2 || rect.height < 2 {
        return None;
    }
    let inside_x = col > rect.x && col < rect.x + rect.width - 1;
    let inside_y = row > rect.y && row < rect.y + rect.height - 1;
    if inside_x && inside_y {
        Some((row - rect.y - 1) as usize)
    } else {
        None
    }
}

fn field_line(label: &str, value: &str, active: bool) -> Line<'static> {
    let value_style = if active {
        Style::default().fg(Color::Black).bg(Color::Cyan)
    } else {
        Style::default().fg(Color::Gray)
    };
    Line::from(vec![
        Span::raw(format!("{label}: ")),
        Span::styled(format!(" {value} "), value_style),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyModifiers;

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
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
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

    #[test]
    fn grouping_collapses_variants_into_families() {
        let families = build_families();
        let keys: Vec<&str> = families.iter().map(|f| f.key.as_str()).collect();
        // Every family key is unique (grouping really collapsed the flat list).
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            keys.len(),
            "family keys must be unique: {keys:?}"
        );
        // Flat profiles outnumber families (collapsing happened).
        let flat = crate::MODEL_PROFILES
            .iter()
            .filter(|p| p.downloadable)
            .count();
        assert!(
            families.len() < flat,
            "{} families from {flat} profiles",
            families.len()
        );

        let e2b = families.iter().find(|f| f.key == "gemma4-e2b").unwrap();
        assert_eq!(e2b.variants.len(), 4); // 4/5/6/8-bit
        assert!(!e2b.has_mtp());
        let g12 = families.iter().find(|f| f.key == "gemma4-12b").unwrap();
        assert!(g12.has_mtp());
        // Recommended (first) variant is the lowest bit-width.
        assert_eq!(g12.variants[0].bits, Some(4));
    }

    #[test]
    fn quant_and_family_parsing() {
        assert_eq!(quant_bits("mlx-community/gemma-4-12B-it-4bit"), Some(4));
        assert_eq!(quant_bits("mlx-community/Qwen3.6-27B-8bit"), Some(8));
        assert_eq!(family_key("gemma4-e2b-8bit"), "gemma4-e2b");
        assert_eq!(family_key("glm4.7-flash-4bit"), "glm4.7-flash");
        assert_eq!(family_key("qwen3.6-35b"), "qwen3.6-35b");
    }

    #[test]
    fn sidebar_and_family_list_render() {
        let app = App::new();
        let text = render(&app);
        assert!(text.contains("AX Engine"), "sidebar title");
        assert!(text.contains("Models"));
        assert!(text.contains("Downloads"));
        assert!(text.contains("Serve"));
        assert!(text.contains("gemma4-e2b"));
        assert!(text.contains("qwen3.6-35b"));
        assert!(text.contains('⚡'), "MTP badge should render");
    }

    #[test]
    fn precision_screen_lists_quants() {
        let mut app = App::new();
        app.family_idx = family_index(&app, "gemma4-12b");
        app.on_key_models(KeyCode::Enter);
        assert!(app.stage == Stage::Precision);
        let text = render(&app);
        assert!(text.contains("select precision"));
        assert!(text.contains("4-bit"));
        assert!(text.contains("recommended"));
        assert!(text.contains("6-bit"));
    }

    #[test]
    fn mtp_prompt_for_mtp_variant_but_not_plain() {
        let mut app = App::new();
        // gemma4-12b 4-bit has an MTP accelerator -> prompt appears.
        app.family_idx = family_index(&app, "gemma4-12b");
        app.on_key_models(KeyCode::Enter); // -> Precision
        app.precision_idx = 0; // 4-bit
        app.on_key_models(KeyCode::Enter); // -> Mtp
        assert!(app.stage == Stage::Mtp);
        assert!(render(&app).contains("Include it"));

        // gemma4-e2b has no MTP -> Enter opens destination, never Mtp.
        let mut app = App::new();
        app.family_idx = family_index(&app, "gemma4-e2b");
        app.on_key_models(KeyCode::Enter);
        let variant = &app.families[app.family_idx].variants[0];
        assert!(variant.mtp_alias.is_none());
    }

    #[test]
    fn serve_tab_renders_fields() {
        let mut app = App::new();
        app.tab = Tab::Serve;
        let text = render(&app);
        assert!(text.contains("Installed models"));
        assert!(text.contains("Host"));
        assert!(text.contains("Port"));
    }

    #[test]
    fn direct_variant_opens_destination_picker() {
        let mut app = App::new();
        app.family_idx = family_index(&app, "gemma4-e2b");
        app.on_key_models(KeyCode::Enter); // -> Precision
        app.on_key_models(KeyCode::Enter); // -> Destination
        assert!(app.stage == Stage::Destination);
        let text = render(&app);
        assert!(text.contains("Download destination"));
        assert!(text.contains("Default:"));
        assert!(text.contains("Custom:"));
        assert!(
            app.pending_download
                .is_some_and(|pending| !pending.with_mtp)
        );
    }

    #[test]
    fn downloads_tab_renders_background_queue() {
        let mut app = App::new();
        app.tab = Tab::Downloads;
        app.downloads.push(DownloadTask {
            id: 1,
            label: "gemma4-e2b 4-bit".into(),
            repo_id: "mlx-community/gemma-4-e2b-it-4bit",
            preset: Some("gemma4-e2b"),
            mode: DownloadMode::Direct,
            subcmd: "download",
            target: "gemma4-e2b".into(),
            dest: Some(PathBuf::from("/tmp/gemma4-e2b")),
            watch_dir: PathBuf::from("/tmp/gemma4-e2b"),
            job: Some(Job::failed("queued test".into())),
            cancelled: false,
        });
        let text = render(&app);
        assert!(text.contains("Downloads"));
        assert!(text.contains("gemma4-e2b"));
        assert!(text.contains("/tmp/gemma4-e2b"));
        assert!(text.contains("queued test"));
    }

    #[test]
    fn queued_download_can_be_cancelled_before_spawn() {
        let mut task = DownloadTask {
            id: 1,
            label: "gemma4-e2b 4-bit".into(),
            repo_id: "mlx-community/gemma-4-e2b-it-4bit",
            preset: Some("gemma4-e2b"),
            mode: DownloadMode::Direct,
            subcmd: "download",
            target: "gemma4-e2b".into(),
            dest: None,
            watch_dir: PathBuf::from("/tmp/gemma4-e2b"),
            job: None,
            cancelled: false,
        };
        assert_eq!(task.status_label(), "queued");
        assert!(task.is_queued());
        task.cancel();
        assert_eq!(task.status_label(), "cancelled");
        assert!(!task.is_queued());
    }

    #[test]
    fn log_parser_finds_download_output_paths() {
        assert_eq!(
            parse_output_path_from_log(&["Path: /tmp/direct".to_string()]).as_deref(),
            Some(Path::new("/tmp/direct"))
        );
        assert_eq!(
            parse_output_path_from_log(&["Output dir: /tmp/mtp".to_string()]).as_deref(),
            Some(Path::new("/tmp/mtp"))
        );
        assert_eq!(
            parse_output_path_from_log(&[
                "Sidecar ready at:".to_string(),
                "  /tmp/sidecar".to_string(),
            ])
            .as_deref(),
            Some(Path::new("/tmp/sidecar"))
        );
    }

    #[test]
    fn click_on_family_row_drills_into_precision() {
        let mut app = App::new();
        let _ = render(&app); // records content_list_rect for the families list
        let rect = app.content_list_rect.get();
        assert!(rect.height >= 2, "list rect should be recorded");
        // Click the second family's row (inner row index 1, below the top border).
        app.on_click(rect.x + 2, rect.y + 2);
        assert!(app.stage == Stage::Precision);
        assert_eq!(app.family_idx, 1);
    }

    #[test]
    fn click_on_sidebar_switches_tab() {
        let mut app = App::new();
        let _ = render(&app); // records sidebar_rect
        let rect = app.sidebar_rect.get();
        // Inner row index 1 == "Downloads".
        app.on_click(rect.x + 2, rect.y + 2);
        assert!(app.tab == Tab::Downloads);
        assert!(app.focus == Focus::Sidebar);

        let _ = render(&app);
        let rect = app.sidebar_rect.get();
        app.on_click(rect.x + 2, rect.y + 3);
        assert!(app.tab == Tab::Serve);
        assert!(app.focus == Focus::Sidebar);
    }

    #[test]
    fn scroll_moves_selection() {
        let mut app = App::new();
        assert_eq!(app.family_idx, 0);
        app.on_mouse(mouse(MouseEventKind::ScrollDown, 0, 0));
        assert_eq!(app.family_idx, 1);
        app.on_mouse(mouse(MouseEventKind::ScrollUp, 0, 0));
        assert_eq!(app.family_idx, 0);
    }

    #[test]
    fn breadcrumb_reflects_wizard_depth() {
        let mut app = App::new();
        app.family_idx = family_index(&app, "gemma4-12b");
        assert!(render(&app).contains("Family: gemma4-12b"));

        app.on_key_models(KeyCode::Enter); // -> Precision (4-bit is variant 0)
        assert!(render(&app).contains("Precision: 4-bit"));

        app.on_key_models(KeyCode::Enter); // -> Mtp (gemma4-12b 4-bit has MTP)
        assert!(app.stage == Stage::Mtp);
        assert!(render(&app).contains("MTP: yes"));
    }

    #[test]
    fn mtp_badge_is_magenta_not_yellow() {
        // Leave the default selection (family_idx 0) alone: the selected row's
        // own highlight style overrides span colors, so check the *other* MTP
        // families' badges instead — several exist in the catalog.
        let app = App::new();
        let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let colors: Vec<Color> = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .filter(|cell| cell.symbol() == "⚡")
            .map(|cell| cell.fg)
            .collect();
        assert!(!colors.is_empty(), "MTP badge glyph should render");
        assert!(
            colors.contains(&Color::Magenta),
            "at least one non-selected MTP badge should be magenta: {colors:?}"
        );
        assert!(
            !colors.contains(&Color::Yellow),
            "MTP badge must not reuse the queued-status yellow: {colors:?}"
        );
    }

    #[test]
    fn port_validation_rejects_bad_values_only() {
        let mut app = App::new();
        assert!(app.port_error().is_none(), "default port is valid");
        app.port = "".into();
        assert!(app.port_error().is_none(), "empty falls back to default");
        app.port = "abc".into();
        assert_eq!(app.port_error(), Some("port must be 1-65535"));
        app.port = "99999".into();
        assert_eq!(app.port_error(), Some("port must be 1-65535"));
        app.port = "0".into();
        assert_eq!(app.port_error(), Some("port must be 1-65535"));
        app.port = "8080".into();
        assert!(app.port_error().is_none());
    }

    #[test]
    fn server_status_waits_for_listening_line_before_going_green() {
        let mut app = App::new();
        app.server = Some(Job::running_with_log(vec!["booting model...".to_string()]));
        app.server_url = Some("http://127.0.0.1:8080".to_string());
        app.tab = Tab::Serve;
        assert!(!app.server_ready);
        assert!(render(&app).contains("starting"));

        app.server
            .as_mut()
            .unwrap()
            .log
            .push("ax-engine-server preview listening on http://127.0.0.1:8080".to_string());
        app.update_server_ready();
        assert!(app.server_ready);
        assert!(render(&app).contains("running at http://127.0.0.1:8080"));
    }

    #[test]
    fn filter_narrows_family_list_and_drill_in_maps_back() {
        let mut app = App::new();
        app.filter = "gemma4-12b".to_string();
        app.clamp_family_idx_to_filter();
        let indices = app.filtered_family_indices();
        assert_eq!(indices.len(), 1);
        assert_eq!(app.families[indices[0]].key, "gemma4-12b");
        assert_eq!(app.family_idx, indices[0]);

        let text = render(&app);
        assert!(text.contains("filter: gemma4-12b"));
        assert!(
            !text.contains("gemma4-e2b"),
            "non-matching family should be hidden"
        );

        app.on_key_models(KeyCode::Enter);
        assert!(app.stage == Stage::Precision);
        assert_eq!(app.families[app.family_idx].key, "gemma4-12b");
    }

    #[test]
    fn most_recent_subdir_is_none_for_missing_dir() {
        assert_eq!(
            most_recent_subdir(Path::new("/definitely/does/not/exist")),
            None
        );
    }

    #[test]
    fn most_recent_subdir_picks_the_only_entry() {
        let base = std::env::temp_dir().join(format!("ax-engine-tui-test-{}", process::id()));
        let snapshots = base.join("snapshots");
        let snapshot = snapshots.join("abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        assert_eq!(most_recent_subdir(&snapshots), Some(snapshot));
        std::fs::remove_dir_all(&base).unwrap();
    }
}
