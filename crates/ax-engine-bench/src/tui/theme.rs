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
pub const FEATURE: Color = Color::Magenta;

/// Muted labels, secondary text, dimmed borders.
pub const MUTED: Color = Color::DarkGray;

/// Neutral secondary — values, descriptions.
pub const DIM: Color = Color::Gray;

/// Active border color for focused widgets.
pub const BORDER_ACTIVE: Color = ACCENT;

/// Inactive border color for unfocused widgets.
pub const BORDER_INACTIVE: Color = MUTED;

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

/// Separator between key hints in footers.
pub fn key_sep() -> Span<'static> {
    Span::styled(" · ", Style::default().fg(MUTED))
}

/// A key hint span: bold key in white, no background chip.
pub fn key_hint(key: &str) -> Span<'static> {
    Span::styled(
        key.to_string(),
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )
}

/// A key hint label (the description after the key).
pub fn key_label(label: &str) -> Span<'static> {
    Span::styled(label.to_string(), Style::default().fg(DIM))
}

/// A muted key chip for modal secondary actions.
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

/// A primary key chip for modal primary actions.
pub fn key_chip(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(Color::Black).bg(ACCENT),
    )
}
