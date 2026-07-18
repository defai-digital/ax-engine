//! Semantic color palette and style helpers for the TUI.
//!
//! Every screen pulls from this single source of truth so that accent colors
//! are consistent and meaning is conveyed through hue, not just position.
//! Prefer these helpers over raw `Color::…` in screens.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

// ---------------------------------------------------------------------------
// Semantic palette (truecolor with named-color fallbacks in docs/comments)
// ---------------------------------------------------------------------------

/// Primary accent — brand, tab labels, links, non-cursor chrome.
/// Intentionally *not* used for the list/cursor selection bar so users can
/// tell "where am I" apart from "which screen am I on".
pub const ACCENT: Color = Color::Rgb(56, 189, 248); // sky-400

/// Cursor / selected row fill — high-contrast warm bar, distinct from ACCENT.
/// White-on-black terminals reverse poorly when everything is sky-blue; amber
/// reads as "the focus is here" without colliding with green fits or red errors.
pub const SELECT: Color = Color::Rgb(251, 191, 36); // amber-400

/// Success / positive — running server, installed badges, ready toasts.
pub const OK: Color = Color::Rgb(74, 222, 128); // green-400

/// Warning — tight fit, queued items, caution toasts.
pub const WARN: Color = Color::Rgb(250, 204, 21); // yellow-400

/// Danger / destructive — errors, delete modals, failed toasts.
pub const DANGER: Color = Color::Rgb(248, 113, 113); // red-400

/// Special / featured — MTP badges, action highlights.
/// Kept as named Magenta so terminals without truecolor still match tests and
/// remain distinct from WARN yellow.
pub const FEATURE: Color = Color::Magenta;

/// Muted labels, secondary text, dimmed borders.
pub const MUTED: Color = Color::DarkGray;

/// Neutral secondary — values, descriptions.
pub const DIM: Color = Color::Gray;

/// Primary readable text on default background.
pub const TEXT: Color = Color::White;

/// Text on filled accent/ok chips.
pub const ON_ACCENT: Color = Color::Black;

/// Text on the selection bar (same ink as ON_ACCENT; named for call sites).
pub const ON_SELECT: Color = Color::Black;

/// Active border for the focused panel — slate, not sky-blue, so the frame
/// does not look like a second selection bar.
pub const BORDER_ACTIVE: Color = Color::Rgb(203, 213, 225); // slate-300

/// Inactive border color for unfocused widgets.
pub const BORDER_INACTIVE: Color = MUTED;

/// Filled background behind fenced code blocks in chat (dark-terminal neutral).
pub const CODE_BG: Color = Color::Rgb(32, 33, 36);

/// Scrim fill behind modals and the help overlay — solid black dim so the
/// dialog clearly owns focus.
pub const SCRIM_BG: Color = Color::Black;

// ---------------------------------------------------------------------------
// Reusable style constructors
// ---------------------------------------------------------------------------

/// Active list/cursor highlight: bold dark-on-amber (never sky-blue).
pub fn highlight_active() -> Style {
    Style::default()
        .bg(SELECT)
        .fg(ON_SELECT)
        .add_modifier(Modifier::BOLD)
}

/// Selection ghost in an unfocused panel: dim underline only (no blue wash).
pub fn highlight_inactive() -> Style {
    Style::default()
        .fg(TEXT)
        .add_modifier(Modifier::UNDERLINED | Modifier::DIM)
}

/// Bold accent title (panel headers, brand) — text only, never a filled chip.
pub fn title() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Muted secondary label.
pub fn label() -> Style {
    Style::default().fg(MUTED)
}

/// Primary body text.
pub fn body() -> Style {
    Style::default().fg(TEXT)
}

/// Dim body / path text.
pub fn body_dim() -> Style {
    Style::default().fg(DIM)
}

/// Success emphasis.
pub fn ok() -> Style {
    Style::default().fg(OK)
}

/// Warning emphasis.
pub fn warn() -> Style {
    Style::default().fg(WARN)
}

/// Danger emphasis.
pub fn danger() -> Style {
    Style::default().fg(DANGER)
}

/// Feature / MTP emphasis.
pub fn feature() -> Style {
    Style::default().fg(FEATURE)
}

/// Primary CTA row (filled accent). Actions, not cursor position.
pub fn cta() -> Style {
    Style::default()
        .fg(ON_ACCENT)
        .bg(ACCENT)
        .add_modifier(Modifier::BOLD)
}

/// Active tab for the current screen — underline + accent text only.
/// Avoids a filled blue bar that competes with the amber selection cursor.
pub fn tab_active() -> Style {
    Style::default()
        .fg(ACCENT)
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

/// Tab bar owns keyboard focus (↑ from content): filled amber so it matches
/// the selection language ("focus is here").
pub fn tab_keyboard_focus() -> Style {
    Style::default()
        .fg(ON_SELECT)
        .bg(SELECT)
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

/// Inactive tab label.
pub fn tab_inactive() -> Style {
    Style::default().fg(DIM)
}

/// Separator between key hints in footers.
pub fn key_sep() -> Span<'static> {
    Span::styled(" · ", Style::default().fg(MUTED))
}

/// A key hint span: bold key in white, no background chip.
pub fn key_hint(key: &str) -> Span<'static> {
    Span::styled(
        key.to_string(),
        Style::default().fg(TEXT).add_modifier(Modifier::BOLD),
    )
}

/// A key hint label (the description after the key).
pub fn key_label(label: &str) -> Span<'static> {
    Span::styled(label.to_string(), Style::default().fg(DIM))
}

/// A muted key chip for modal secondary actions.
pub fn key_chip_dim(label: &str) -> Span<'static> {
    Span::styled(format!(" {label} "), Style::default().fg(ON_ACCENT).bg(DIM))
}

/// A danger key chip for destructive confirmations.
pub fn key_chip_danger(label: &str) -> Span<'static> {
    Span::styled(format!(" {label} "), Style::default().fg(TEXT).bg(DANGER))
}

/// A primary key chip for modal primary actions.
pub fn key_chip(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(ON_ACCENT).bg(ACCENT),
    )
}

/// Status icon glyph contract (fixed lexicon).
pub mod icon {
    pub const OK: &str = "✓";
    pub const RUNNING: &str = "●";
    pub const QUEUED: &str = "…";
    pub const IDLE: &str = "○";
    pub const ERROR: &str = "✗";
    pub const WARN: &str = "!";
    pub const SPEED: &str = "⚡";
    pub const SELECT: &str = "›";
    pub const STAR: &str = "★";
}
