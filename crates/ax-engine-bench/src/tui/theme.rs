//! Semantic color palette and style helpers for the TUI.
//!
//! Every screen pulls from this single source of truth so that accent colors
//! are consistent and meaning is conveyed through hue, not just position.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

// ---------------------------------------------------------------------------
// Semantic palette
// ---------------------------------------------------------------------------

/// Primary accent — navigation, active selections, focused borders.
pub const ACCENT: Color = Color::Cyan;

/// Success / positive — running server, installed badges, ready toasts.
pub const OK: Color = Color::Green;

/// Warning — tight fit, queued items, caution toasts.
pub const WARN: Color = Color::Yellow;

/// Danger / destructive — errors, delete modals, failed toasts.
pub const DANGER: Color = Color::Red;

/// Special / featured — MTP badges, action highlights.
#[allow(dead_code)]
pub const FEATURE: Color = Color::Magenta;

/// Muted labels, secondary text, dimmed borders.
pub const MUTED: Color = Color::DarkGray;

/// Neutral secondary — values, descriptions.
pub const DIM: Color = Color::Gray;

// ---------------------------------------------------------------------------
// Reusable style constructors
// ---------------------------------------------------------------------------

/// Active list highlight: bold white-on-accent.
pub fn highlight_active() -> Style {
    Style::default()
        .bg(ACCENT)
        .fg(Color::Black)
        .add_modifier(Modifier::BOLD)
}

/// Inactive list highlight: reversed.
pub fn highlight_inactive() -> Style {
    Style::default().add_modifier(Modifier::REVERSED)
}

/// A styled key chip like `[Enter]` for footer / modal hints.
pub fn key_chip(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(Color::Black).bg(ACCENT),
    )
}

/// A muted key chip for secondary actions.
pub fn key_chip_dim(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(Color::Black).bg(DIM),
    )
}

/// A danger key chip for destructive confirmations.
pub fn key_chip_danger(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(Color::White).bg(DANGER),
    )
}

/// Separator between key chips in footers.
pub fn key_sep() -> Span<'static> {
    Span::styled("  ", Style::default())
}

// ---------------------------------------------------------------------------
// Status indicators
// ---------------------------------------------------------------------------

/// Green dot for "running" / "ready" status in sidebar badges.
pub const OK_DOT: char = '●';

/// A dot character indicating the count of running downloads.
pub fn spinner_dot(count: usize) -> String {
    if count == 0 {
        String::new()
    } else {
        format!("({count})")
    }
}
