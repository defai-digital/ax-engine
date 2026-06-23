//! ratatui terminal UI for `ax-engine tui`.
//!
//! A native (Rust) replacement for the former Python/Textual downloader.  It is
//! a child module of the `ax-engine` binary, so it reuses that binary's private
//! catalog and helpers directly via `crate::` (`MODEL_PROFILES`,
//! `default_hf_cache_root`, `find_executable`, ...) instead of duplicating them.
//!
//! Layout: a left sidebar (Models / Serve) plus a content pane.  The Models pane
//! is a wizard — model family -> precision -> optional MTP accelerator -> a
//! streamed download.  Downloads and the server run as child `ax-engine` /
//! `ax-engine-server` processes whose output is streamed into a log pane, which
//! keeps their stdout off the alternate screen and reuses the existing CLI
//! logic untouched.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, IsTerminal};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, Instant};

use ratatui::DefaultTerminal;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub(crate) fn cmd_tui(args: &[OsString]) -> Result<u8, String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "ax-engine tui — interactive model downloader and serve launcher.\n\n\
             Sidebar: Models · Serve.  Models is a wizard: pick a model, then a\n\
             precision (4/5/6/8-bit), then optionally an MTP accelerator, then\n\
             download.  Keys: ↑↓ move · → enter · ← back · Enter confirm · q quit."
        );
        return Ok(0);
    }
    if !io::stdout().is_terminal() {
        return Err("ax-engine tui needs an interactive terminal".into());
    }
    let mut terminal = ratatui::init();
    let result = App::new().run(&mut terminal);
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
    last_poll: Option<(Instant, u64)>,
    spinner: usize,
}

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
        }
        self.spinner = (self.spinner + 1) % SPINNER.len();
    }

    fn cancel(&mut self) {
        if self.done.is_none()
            && let Some(child) = &mut self.child
        {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Models,
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
    Progress,
}

#[derive(Clone, Copy, PartialEq)]
enum ServeFocus {
    List,
    Host,
    Port,
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
    download: Option<Job>,

    // Serve
    installed: Vec<usize>, // indices into a flattened (family, variant) view
    serve_focus: ServeFocus,
    serve_idx: usize,
    host: String,
    port: String,
    server: Option<Job>,
    server_url: Option<String>,
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
            download: None,
            installed: Vec::new(),
            serve_focus: ServeFocus::List,
            serve_idx: 0,
            host: "127.0.0.1".into(),
            port: "8080".into(),
            server: None,
            server_url: None,
        }
    }

    fn reload_families(&mut self) {
        self.families = build_families();
    }

    fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.quit {
            terminal.draw(|frame| self.draw(frame))?;
            if event::poll(Duration::from_millis(100))?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.on_key(key);
            }
            if let Some(job) = &mut self.download {
                job.tick();
            }
            if let Some(job) = &mut self.server {
                job.tick();
            }
        }
        if let Some(job) = &mut self.download {
            job.cancel();
        }
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        Ok(())
    }

    // -- input ----------------------------------------------------------------

    fn on_key(&mut self, key: KeyEvent) {
        // Global quit (ignored while typing into a serve text field).
        let typing = self.tab == Tab::Serve
            && self.focus == Focus::Content
            && matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port);
        if !typing && matches!(key.code, KeyCode::Char('q')) {
            self.quit = true;
            return;
        }
        match self.focus {
            Focus::Sidebar => self.on_key_sidebar(key.code),
            Focus::Content => match self.tab {
                Tab::Models => self.on_key_models(key.code),
                Tab::Serve => self.on_key_serve(key.code),
            },
        }
    }

    fn on_key_sidebar(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.tab = Tab::Models,
            KeyCode::Down | KeyCode::Char('j') => self.tab = Tab::Serve,
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => {
                self.focus = Focus::Content;
                if self.tab == Tab::Serve {
                    self.installed = installed_variants(&self.families)
                        .into_iter()
                        .enumerate()
                        .map(|(i, _)| i)
                        .collect();
                }
            }
            _ => {}
        }
    }

    fn on_key_models(&mut self, code: KeyCode) {
        match self.stage {
            Stage::Families => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.family_idx = self.family_idx.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.family_idx + 1 < self.families.len() {
                        self.family_idx += 1;
                    }
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    self.precision_idx = 0;
                    self.stage = Stage::Precision;
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.focus = Focus::Sidebar,
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
                        self.start_download(false);
                    }
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.stage = Stage::Families,
                _ => {}
            },
            Stage::Mtp => match code {
                KeyCode::Up | KeyCode::Down | KeyCode::Char('k') | KeyCode::Char('j') => {
                    self.mtp_idx ^= 1;
                }
                KeyCode::Char('y') => self.start_download(true),
                KeyCode::Char('n') => self.start_download(false),
                KeyCode::Enter => self.start_download(self.mtp_idx == 0),
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.stage = Stage::Precision,
                _ => {}
            },
            Stage::Progress => match code {
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('b') | KeyCode::Esc => {
                    if let Some(job) = &mut self.download {
                        job.cancel();
                    }
                    self.download = None;
                    self.reload_families();
                    self.stage = Stage::Precision;
                }
                _ => {}
            },
        }
    }

    fn on_key_serve(&mut self, code: KeyCode) {
        match self.serve_focus {
            ServeFocus::List => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.serve_idx = self.serve_idx.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.serve_idx + 1 < self.installed.len() {
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

    fn start_download(&mut self, with_mtp: bool) {
        let variant = &self.families[self.family_idx].variants[self.precision_idx];
        let (subcmd, target) = if with_mtp {
            (
                "download-mtp",
                variant.mtp_alias.unwrap_or(variant.profile.label),
            )
        } else {
            ("download", variant.profile.label)
        };
        let watch_dir = Some(repo_cache_dir(variant.profile.repo_id));
        let Ok(exe) = std::env::current_exe() else {
            return;
        };
        let mut cmd = Command::new(exe);
        cmd.arg(subcmd).arg(target);
        match Job::spawn(cmd, watch_dir) {
            Ok(job) => {
                self.download = Some(job);
                self.stage = Stage::Progress;
            }
            Err(err) => {
                self.download = Some(Job::failed(format!("failed to launch download: {err}")));
                self.stage = Stage::Progress;
            }
        }
    }

    fn start_server(&mut self) {
        if self.server.as_ref().is_some_and(|j| j.done.is_none()) {
            return;
        }
        let Some(&list_index) = self.installed.get(self.serve_idx) else {
            return;
        };
        let pairs = installed_variants(&self.families);
        let Some(&(fi, vi)) = pairs.get(list_index) else {
            return;
        };
        let profile = self.families[fi].variants[vi].profile;
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
        if let Some(preset) = profile.preset {
            cmd.arg("--preset")
                .arg(preset)
                .arg("--resolve-model-artifacts")
                .arg("hf-cache");
        } else {
            cmd.arg("--mlx-model-artifacts-dir")
                .arg(repo_cache_dir(profile.repo_id));
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

    fn stop_server(&mut self) {
        if let Some(job) = &mut self.server {
            job.cancel();
        }
        self.server = None;
        self.server_url = None;
    }

    // -- rendering ------------------------------------------------------------

    fn draw(&self, frame: &mut Frame) {
        let outer =
            Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(frame.area());
        let body = Layout::horizontal([Constraint::Length(18), Constraint::Min(0)]).split(outer[0]);
        self.draw_sidebar(frame, body[0]);
        match self.tab {
            Tab::Models => self.draw_models(frame, body[1]),
            Tab::Serve => self.draw_serve(frame, body[1]),
        }
        frame.render_widget(
            Paragraph::new(self.footer()).style(Style::default().fg(Color::DarkGray)),
            outer[1],
        );
    }

    fn draw_sidebar(&self, frame: &mut Frame, area: Rect) {
        let active = self.focus == Focus::Sidebar;
        let items = [("Models", Tab::Models), ("Serve", Tab::Serve)].map(|(name, tab)| {
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
        match self.stage {
            Stage::Families => self.draw_families(frame, area),
            Stage::Precision => self.draw_precision(frame, area),
            Stage::Mtp => self.draw_mtp(frame, area),
            Stage::Progress => self.draw_progress(frame, area),
        }
    }

    fn draw_families(&self, frame: &mut Frame, area: Rect) {
        let rows: Vec<ListItem> = self
            .families
            .iter()
            .map(|family| {
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
                    Span::styled("  ⚡MTP", Style::default().fg(Color::Yellow))
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
        self.render_list(
            frame,
            area,
            " Models ",
            rows,
            self.family_idx,
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
                    Span::styled("⚡MTP  ", Style::default().fg(Color::Yellow))
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

    fn draw_progress(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(4), Constraint::Min(0)]).split(area);
        let variant = &self.families[self.family_idx].variants[self.precision_idx];
        let job = self.download.as_ref();
        let status = match job.and_then(|j| j.done) {
            Some(0) => Line::from(Span::styled(
                "Done. Press b/Esc to go back.",
                Style::default().fg(Color::Green),
            )),
            Some(code) => Line::from(Span::styled(
                format!("Failed (exit {code}). Press b/Esc to go back."),
                Style::default().fg(Color::Red),
            )),
            None => {
                let spin = job.map(|j| SPINNER[j.spinner]).unwrap_or(' ');
                let bytes = job.map(|j| j.bytes).unwrap_or(0);
                let speed = job.map(|j| j.speed).unwrap_or(0.0);
                Line::from(Span::styled(
                    format!(
                        "{spin}  {}  ({}/s)",
                        format_bytes(bytes),
                        format_bytes(speed as u64)
                    ),
                    Style::default().fg(Color::Cyan),
                ))
            }
        };
        let head = Paragraph::new(vec![
            Line::from(Span::styled(
                format!("Downloading {}", variant.profile.repo_id),
                Style::default().add_modifier(Modifier::BOLD),
            )),
            status,
        ])
        .block(Block::default().borders(Borders::ALL).title(" Download "));
        frame.render_widget(head, chunks[0]);
        self.draw_log(frame, chunks[1], job, " Output ");
    }

    fn draw_serve(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(4),
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
        let url = match &self.server_url {
            Some(url) if self.server.as_ref().is_some_and(|j| j.done.is_none()) => {
                Line::from(Span::styled(
                    format!("running at {url}"),
                    Style::default().fg(Color::Green),
                ))
            }
            _ => Line::from(Span::styled(
                "stopped",
                Style::default().fg(Color::DarkGray),
            )),
        };
        frame.render_widget(
            Paragraph::new(vec![host_line, port_line, url]).block(
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
            Focus::Sidebar => "↑↓ pick · → enter · q quit",
            Focus::Content => match (self.tab, self.stage) {
                (Tab::Models, Stage::Families) => {
                    "↑↓ move · → choose precision · ← sidebar · q quit"
                }
                (Tab::Models, Stage::Precision) => "↑↓ move · → download · ← back · q quit",
                (Tab::Models, Stage::Mtp) => "y/n or ↑↓+Enter · Esc back",
                (Tab::Models, Stage::Progress) => "b/Esc back · q quit",
                (Tab::Serve, _) => {
                    "↑↓ move · Enter start · x stop · Tab fields · ← sidebar · q quit"
                }
            },
        };
        Line::from(format!("  {help}"))
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

        // gemma4-e2b has no MTP -> Enter would go straight to a download, never Mtp.
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
}
