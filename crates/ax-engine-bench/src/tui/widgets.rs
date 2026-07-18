//! Shared rendering primitives: tab bar, bordered lists with click
//! bookkeeping, toast overlays, confirm modals, and the directory picker
//! reused by the wizard's custom-destination modal.

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
        Style::default().fg(theme::ON_ACCENT).bg(color)
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
            y: area.y + 2 + i as u16,
            width,
            height: 1,
        };
        frame.render_widget(Clear, rect);
        frame.render_widget(Paragraph::new(text).style(toast.style()), rect);
    }
}

/// Click targets recorded while drawing a modal: the dialog popup plus the
/// first (confirm) and last (cancel) key chip, so mouse clicks map back to
/// Enter/Esc through the normal key path.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ModalHits {
    pub popup: Rect,
    pub confirm: Option<Rect>,
    pub cancel: Option<Rect>,
}

/// Modal variant with dimmed scrim, severity-colored border, and key chips.
/// Returns the click targets for the rendered dialog.
pub(super) fn draw_modal_with(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    mut lines: Vec<Line>,
    hints: Vec<Span<'static>>,
    border_color: Color,
) -> ModalHits {
    // Dim the whole screen so the dialog clearly owns focus.
    frame.render_widget(
        Paragraph::new("").style(Style::default().bg(theme::SCRIM_BG).fg(theme::MUTED)),
        area,
    );
    let width = 64.min(area.width.saturating_sub(4)).max(20);
    let height = (lines.len() as u16 + 4).min(area.height.saturating_sub(2));
    let popup = centered_rect(width, height, area);
    // Chip hit-rects: the hint spans render on the last content row, laid out
    // consecutively from the left inner edge.
    let hints_y = popup.y + popup.height.saturating_sub(2);
    let mut chip_x = popup.x + 1;
    let mut chip_rects: Vec<Rect> = Vec::new();
    for span in &hints {
        let w = span.width() as u16;
        chip_rects.push(Rect {
            x: chip_x,
            y: hints_y,
            width: w,
            height: 1,
        });
        chip_x = chip_x.saturating_add(w);
    }
    let (confirm, cancel) = match chip_rects.len() {
        0 => (None, None),
        // A lone chip is always the dismiss affordance (e.g. "Esc keep").
        1 => (None, chip_rects.first().copied()),
        _ => (chip_rects.first().copied(), chip_rects.last().copied()),
    };
    lines.push(Line::raw(""));
    lines.push(Line::from(hints));
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(border_color))
                    .title(Span::styled(
                        format!(" {title} "),
                        Style::default()
                            .fg(border_color)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .wrap(Wrap { trim: false }),
        popup,
    );
    ModalHits {
        popup,
        confirm,
        cancel,
    }
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

/// Panel chrome: active gets a full accent frame; inactive keeps a light left bar
/// so the UI is less “box of boxes”.
pub(super) fn panel_block(title: &str, active: bool) -> Block<'static> {
    let (borders, border) = if active {
        (Borders::ALL, theme::BORDER_ACTIVE)
    } else {
        (Borders::LEFT, theme::BORDER_INACTIVE)
    };
    Block::default()
        .borders(borders)
        .border_style(Style::default().fg(border))
        .title(Span::styled(
            title.to_string(),
            if active {
                theme::title()
            } else {
                theme::label()
            },
        ))
}

/// Soft section block (left accent only) for secondary content.
pub(super) fn soft_block(title: &str) -> Block<'static> {
    panel_block(title, false)
}

/// Full active block for the focused panel.
pub(super) fn active_block(title: &str) -> Block<'static> {
    panel_block(title, true)
}

/// One-line guided next-step banner (fills `area` height 1).
pub(super) fn draw_banner(frame: &mut Frame, area: Rect, kind: ToastLevel, text: &str) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let (icon, style) = match kind {
        ToastLevel::Info => (
            theme::icon::SELECT,
            Style::default().fg(theme::ON_ACCENT).bg(theme::ACCENT),
        ),
        ToastLevel::Success => (
            theme::icon::OK,
            Style::default().fg(theme::ON_ACCENT).bg(theme::OK),
        ),
        ToastLevel::Warning => (
            theme::icon::WARN,
            Style::default().fg(theme::ON_ACCENT).bg(theme::WARN),
        ),
        ToastLevel::Error => (
            theme::icon::ERROR,
            Style::default().fg(theme::ON_ACCENT).bg(theme::DANGER),
        ),
    };
    let line = format!(" {icon}  {text} ");
    frame.render_widget(Paragraph::new(line).style(style), area);
}

/// Bordered stateful list; records `area` into `click_target` for hit-testing.
///
/// Active lists use a full frame; inactive lists use a left accent only.
/// Hit-testing still uses the full `area` rect.
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
    // Only the focused list is clickable; inactive side panels must not steal
    // (or mis-map) hit-testing — they use thinner left-only chrome.
    if active {
        click_target.set(area);
    }
    let block = panel_block(title, active);
    let highlight = if active {
        theme::highlight_active()
    } else {
        theme::highlight_inactive()
    };
    let list = List::new(rows)
        .block(block)
        .highlight_style(highlight)
        .highlight_symbol(format!("{} ", theme::icon::SELECT));
    let mut state = ListState::default();
    state.select(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

/// Inner row index of a click inside a list panel, if it landed on a content row.
///
/// Active lists use a 1-cell full border; inactive lists use a left bar only
/// (title sits on the first row). Callers pass `active` to match `render_list`.
/// No vertical scroll offset — lists are short enough to always fit.
pub(super) fn row_in_rect(rect: Rect, col: u16, row: u16) -> Option<usize> {
    row_in_rect_ex(rect, col, row, true)
}

/// Like [`row_in_rect`], with an explicit active/inactive chrome mode.
pub(super) fn row_in_rect_ex(rect: Rect, col: u16, row: u16, active: bool) -> Option<usize> {
    if rect.width < 2 || rect.height < 1 {
        return None;
    }
    if active {
        if rect.height < 2 {
            return None;
        }
        let inside_x = col > rect.x && col < rect.x + rect.width - 1;
        let inside_y = row > rect.y && row < rect.y + rect.height - 1;
        if inside_x && inside_y {
            Some((row - rect.y - 1) as usize)
        } else {
            None
        }
    } else {
        // Left border only: content starts at x+1, title/content from y.
        let inside_x = col > rect.x && col < rect.x + rect.width;
        let inside_y = row >= rect.y && row < rect.y + rect.height;
        if inside_x && inside_y {
            // Title occupies the first row of the block; skip it when present.
            let inner = row.saturating_sub(rect.y).saturating_sub(1);
            Some(inner as usize)
        } else {
            None
        }
    }
}

/// Copy text to the macOS clipboard via `pbcopy`.  Returns false when pbcopy
/// is unavailable or fails (caller surfaces a toast).
pub(super) fn copy_to_clipboard(text: &str) -> bool {
    use std::io::Write;
    use std::process::{Command, Stdio};
    Command::new("pbcopy")
        .stdin(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(stdin) = child.stdin.as_mut() {
                stdin.write_all(text.as_bytes())?;
            }
            child.wait()
        })
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Byte offset of the `idx`-th char boundary in `text` (end of string when
/// `idx` is past the last char). Char-based editing never splits a codepoint.
pub(super) fn char_boundary(text: &str, idx: usize) -> usize {
    text.char_indices()
        .nth(idx)
        .map(|(byte, _)| byte)
        .unwrap_or(text.len())
}

/// Host/Port row. When `active`, the value renders as an amber bar with an
/// inverted caret cell at char index `cursor` (trailing blank when at end).
pub(super) fn field_line(label: &str, value: &str, active: bool, cursor: usize) -> Line<'static> {
    let prefix = if active {
        format!("{} ", theme::icon::RUNNING)
    } else {
        "  ".into()
    };
    let mut spans = vec![
        Span::styled(
            prefix,
            Style::default().fg(if active { theme::SELECT } else { theme::MUTED }),
        ),
        Span::styled(format!("{label}: "), theme::label()),
    ];
    if !active {
        spans.push(Span::styled(
            format!(" {value} "),
            Style::default().fg(theme::DIM),
        ));
        return Line::from(spans);
    }
    let bar = Style::default().fg(theme::ON_SELECT).bg(theme::SELECT);
    let caret = Style::default().fg(theme::SELECT).bg(theme::ON_SELECT);
    let len = value.chars().count();
    let cursor = cursor.min(len);
    spans.push(Span::styled(" ", bar));
    for (i, c) in value.chars().enumerate() {
        spans.push(Span::styled(
            c.to_string(),
            if i == cursor { caret } else { bar },
        ));
    }
    if cursor == len {
        spans.push(Span::styled(" ", caret));
    }
    spans.push(Span::styled(" ", bar));
    Line::from(spans)
}

/// Fit `text` into exactly `width` columns for aligned list rows: space-padded
/// when shorter, truncated with a trailing `…` when longer.  Char-based, so
/// multi-byte names never get cut mid-codepoint.
pub(super) fn ellipsis(text: &str, width: usize) -> String {
    if text.chars().count() > width {
        text.chars()
            .take(width.saturating_sub(1))
            .collect::<String>()
            + "…"
    } else {
        format!("{text:<width$}")
    }
}

/// Scrolling log pane fed from a job's captured output.
/// Applies basic coloring to ERROR/WARN/INFO lines for scannability.
pub(super) fn draw_log(frame: &mut Frame, area: Rect, log: Option<&[String]>, title: &str) {
    let height = area.height.saturating_sub(1) as usize; // left bar chrome
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
                    } else if l.contains("listening on") || l.contains("preview listening") {
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
    frame.render_widget(Paragraph::new(lines).block(soft_block(title)), area);
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

/// One tab definition: (number, label, screen enum variant).
pub(super) struct TabDef {
    pub num: char,
    pub label: &'static str,
}

/// Optional badge text shown after a tab label (e.g. download count).
pub(super) struct TabBadge {
    pub text: String,
    pub style: Style,
}

/// Render the horizontal tab bar.  Returns the tab hit-test regions as a
/// `Vec<(Rect, usize)>` where `usize` is the screen index.
///
/// Active tab is a filled accent chip. Optional `badges[i]` appends a compact
/// status mark after tab `i`'s label. When `tabs_focused`, the bar is visually
/// emphasized so keyboard users can see ↑ reached the chrome.
pub(super) fn draw_tab_bar(
    frame: &mut Frame,
    area: Rect,
    tabs: &[TabDef],
    active: usize,
    status_spans: Vec<Span<'static>>,
    badges: &[Option<TabBadge>],
    tabs_focused: bool,
) -> Vec<(Rect, usize)> {
    let mut hits = Vec::new();
    // Row 0: title + tabs + status.
    // Row 1: thin separator line.
    let row0 = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: 1,
    };
    let row1 = Rect {
        x: area.x,
        y: area.y + 1,
        width: area.width,
        height: 1,
    };

    let brand = " AX Engine ";
    let mut spans: Vec<Span> = vec![
        Span::raw(" "),
        Span::styled(brand, theme::title()),
        Span::raw(" "),
    ];
    // leading space + brand + trailing space
    let mut col_cursor: u16 = 1 + brand.chars().count() as u16 + 1;

    for (i, tab) in tabs.iter().enumerate() {
        let is_active = i == active;
        let badge = badges.get(i).and_then(|b| b.as_ref());
        let label_text = match badge {
            Some(b) if !b.text.is_empty() => {
                format!(" {} {} {} ", tab.num, tab.label, b.text)
            }
            _ => format!(" {} {} ", tab.num, tab.label),
        };
        let label_width = label_text.chars().count() as u16;
        let style = if is_active && tabs_focused {
            // Keyboard focus on the bar: same amber language as list selection.
            theme::tab_keyboard_focus()
        } else if is_active {
            // Current screen: accent + underline only (not a blue fill bar).
            theme::tab_active()
        } else if tabs_focused {
            // Other tabs stay readable while the bar owns focus.
            Style::default().fg(theme::TEXT)
        } else if let Some(b) = badge.filter(|b| !b.text.is_empty()) {
            // Inactive tab with a status badge: keep dim label, tint badge via
            // the whole chip so the count remains visible.
            b.style.add_modifier(Modifier::DIM)
        } else {
            theme::tab_inactive()
        };
        spans.push(Span::styled(label_text, style));
        let tab_rect = Rect {
            x: area.x + col_cursor,
            y: row0.y,
            width: label_width,
            height: 1,
        };
        hits.push((tab_rect, i));
        col_cursor += label_width + 1; // 1 space between tabs
        spans.push(Span::raw(" "));
    }

    // Right-align status spans by padding with spaces.
    let status_width: u16 = status_spans
        .iter()
        .map(|s| s.content.chars().count() as u16)
        .sum();
    let used = col_cursor;
    let pad = area.width.saturating_sub(used + status_width + 2);
    if pad > 0 {
        spans.push(Span::raw(" ".repeat(pad as usize)));
    }
    spans.extend(status_spans);
    spans.push(Span::raw(" "));

    frame.render_widget(Paragraph::new(Line::from(spans)), row0);

    // Separator: selection amber while the tab bar owns keyboard focus.
    let sep = "─".repeat(area.width as usize);
    frame.render_widget(
        Paragraph::new(sep).style(Style::default().fg(if tabs_focused {
            theme::SELECT
        } else {
            theme::BORDER_INACTIVE
        })),
        row1,
    );

    hits
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
