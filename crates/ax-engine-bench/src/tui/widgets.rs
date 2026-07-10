//! Shared rendering primitives: bordered lists with click bookkeeping, the
//! global status strip, toast overlays, confirm modals, and the directory
//! picker reused by the wizard's custom-destination modal.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

use super::theme;

pub(super) const TOAST_TTL: Duration = Duration::from_secs(4);
const TOAST_MAX_VISIBLE: usize = 3;

/// Toast severity for color differentiation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ToastLevel {
    Info,
    Success,
    Warning,
    Error,
}

pub(super) struct Toast {
    pub text: String,
    pub at: Instant,
    pub level: ToastLevel,
}

impl Toast {
    pub fn info(text: String) -> Self {
        Self {
            text,
            at: Instant::now(),
            level: ToastLevel::Info,
        }
    }
    pub fn success(text: String) -> Self {
        Self {
            text,
            at: Instant::now(),
            level: ToastLevel::Success,
        }
    }
    pub fn warning(text: String) -> Self {
        Self {
            text,
            at: Instant::now(),
            level: ToastLevel::Warning,
        }
    }
    pub fn error(text: String) -> Self {
        Self {
            text,
            at: Instant::now(),
            level: ToastLevel::Error,
        }
    }
    fn style(&self) -> Style {
        let color = match self.level {
            ToastLevel::Info => theme::ACCENT,
            ToastLevel::Success => theme::OK,
            ToastLevel::Warning => theme::WARN,
            ToastLevel::Error => theme::DANGER,
        };
        Style::default().fg(Color::Black).bg(color)
    }
    fn icon(&self) -> &'static str {
        match self.level {
            ToastLevel::Info => "ℹ",
            ToastLevel::Success => "✓",
            ToastLevel::Warning => "⚠",
            ToastLevel::Error => "✗",
        }
    }
}

/// Drop expired toasts (called once per tick).
pub(super) fn expire_toasts(toasts: &mut Vec<Toast>) {
    toasts.retain(|toast| toast.at.elapsed() < TOAST_TTL);
}

/// Newest-first stack of small notices in the top-right corner.
pub(super) fn draw_toasts(frame: &mut Frame, area: Rect, toasts: &[Toast]) {
    for (i, toast) in toasts.iter().rev().take(TOAST_MAX_VISIBLE).enumerate() {
        let text = format!(" {} {} ", toast.icon(), toast.text);
        let width = (text.chars().count() as u16).min(area.width.saturating_sub(2));
        let rect = Rect {
            x: area.x + area.width.saturating_sub(width + 1),
            y: area.y + 1 + i as u16,
            width,
            height: 1,
        };
        frame.render_widget(Clear, rect);
        frame.render_widget(Paragraph::new(text).style(toast.style()), rect);
    }
}

/// One-line engine summary shown on every screen, above the footer.
pub(super) fn draw_status_strip(
    frame: &mut Frame,
    area: Rect,
    server: (String, Color),
    download: Option<(String, Color)>,
) {
    let mut spans = vec![
        Span::styled("  ● server ", Style::default().fg(theme::MUTED)),
        Span::styled(
            server.0,
            Style::default().fg(server.1).add_modifier(Modifier::BOLD),
        ),
    ];
    if let Some((text, color)) = download {
        spans.push(Span::styled(
            "   │   ↓ download ",
            Style::default().fg(theme::MUTED),
        ));
        spans.push(Span::styled(
            text,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::Rgb(15, 15, 15))),
        area,
    );
}

/// Modal variant with severity-colored border and styled key hints.
pub(super) fn draw_modal_with(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    mut lines: Vec<Line>,
    hints: Vec<Span<'static>>,
    border_color: Color,
) {
    let width = 64.min(area.width.saturating_sub(4)).max(20);
    let height = (lines.len() as u16 + 4).min(area.height.saturating_sub(2));
    let popup = centered_rect(width, height, area);
    lines.push(Line::raw(""));
    lines.push(Line::from(hints));
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(format!(" {title} ")),
            )
            .wrap(Wrap { trim: false }),
        popup,
    );
}

pub(super) fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

/// Bordered stateful list; records `area` into `click_target` for hit-testing.
#[allow(clippy::too_many_arguments)]
pub(super) fn render_list(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    rows: Vec<ListItem>,
    selected: usize,
    active: bool,
    click_target: &std::cell::Cell<Rect>,
) {
    click_target.set(area);
    let border = if active { theme::ACCENT } else { theme::MUTED };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border))
        .title(title.to_string());
    let highlight = if active {
        theme::highlight_active()
    } else {
        theme::highlight_inactive()
    };
    let list = List::new(rows)
        .block(block)
        .highlight_style(highlight)
        .highlight_symbol("▸ ");
    let mut state = ListState::default();
    state.select(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

/// Inner row index of a click inside a bordered widget, if it landed on a content row.
///
/// Assumes a 1-cell border and no vertical scroll offset (the lists are short
/// enough to always fit), so inner row `n` maps to item `n`.  Callers bounds-check
/// the returned index against the actual item count.
pub(super) fn row_in_rect(rect: Rect, col: u16, row: u16) -> Option<usize> {
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

pub(super) fn field_line(label: &str, value: &str, active: bool) -> Line<'static> {
    let value_style = if active {
        Style::default().fg(Color::Black).bg(theme::ACCENT)
    } else {
        Style::default().fg(theme::DIM)
    };
    let prefix = if active { "● " } else { "  " };
    Line::from(vec![
        Span::styled(
            prefix,
            Style::default().fg(if active { theme::ACCENT } else { theme::MUTED }),
        ),
        Span::raw(format!("{label}: ")),
        Span::styled(format!(" {value} "), value_style),
    ])
}

/// Scrolling log pane fed from a job's captured output.
/// Applies basic coloring to ERROR/WARN/INFO lines for scannability.
pub(super) fn draw_log(frame: &mut Frame, area: Rect, log: Option<&[String]>, title: &str) {
    let height = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = match log {
        Some(log) => {
            let start = log.len().saturating_sub(height);
            log[start..]
                .iter()
                .map(|l| {
                    let style = if l.contains("ERROR") || l.contains("error") || l.contains("panic")
                    {
                        Style::default().fg(theme::DANGER)
                    } else if l.contains("WARN") || l.contains("warn") {
                        Style::default().fg(theme::WARN)
                    } else if l.contains("INFO") || l.contains("info") {
                        Style::default().fg(theme::OK)
                    } else if l.contains("listening on") {
                        Style::default().fg(theme::OK).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme::DIM)
                    };
                    Line::from(Span::styled(l.clone(), style))
                })
                .collect()
        }
        None => vec![Line::raw("")],
    };
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(title.to_string()),
        ),
        area,
    );
}

// ---------------------------------------------------------------------------
// Directory picker (wizard custom-destination modal)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(super) struct DirEntry {
    pub label: String,
    pub path: PathBuf,
}

pub(super) struct DirectoryPicker {
    pub current: PathBuf,
    pub entries: Vec<DirEntry>,
    pub selected: usize,
    pub error: Option<String>,
}

impl DirectoryPicker {
    pub fn new() -> Self {
        let mut picker = Self {
            current: nearest_existing_dir(&crate::default_hf_cache_root()),
            entries: Vec::new(),
            selected: 0,
            error: None,
        };
        picker.refresh();
        picker
    }

    pub fn set_current(&mut self, path: PathBuf) {
        self.current = nearest_existing_dir(&path);
        self.selected = 0;
        self.refresh();
    }

    pub fn refresh(&mut self) {
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

    pub fn selected_path(&self) -> Option<PathBuf> {
        self.entries
            .get(self.selected)
            .map(|entry| entry.path.clone())
    }

    pub fn enter_selected(&mut self) {
        if let Some(path) = self.selected_path() {
            self.set_current(path);
        }
    }
}

pub(super) fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub(super) fn nearest_existing_dir(path: &Path) -> PathBuf {
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

pub(super) fn sanitize_path_segment(value: &str) -> String {
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

pub(super) fn validate_writable_parent(path: &Path) -> Result<(), String> {
    if !path.is_dir() {
        return Err(format!("not a directory: {}", path.display()));
    }
    let probe = path.join(format!(".ax-engine-tui-write-test-{}", std::process::id()));
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
