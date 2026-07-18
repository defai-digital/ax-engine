//! Semantic color palette and style helpers for the TUI.
//!
//! Every screen pulls from this single source of truth so that accent colors
//! are consistent and meaning is conveyed through hue, not just position.
//! Prefer these helpers over raw `Color::…` in screens.
//!
//! Colors are selected at startup ([`init`]): the default dark-terminal
//! palette, a light-terminal palette (`AX_TUI_THEME=light`), and a mono mode
//! honoring `NO_COLOR` (or `AX_TUI_THEME=mono`). Status glyphs likewise fall
//! back to ASCII when the locale is not UTF-8. Tests never call [`init`], so
//! they always see the dark palette.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Runtime palette
// ---------------------------------------------------------------------------

/// Semantic colors, resolved once at startup. Field docs live with the dark
/// defaults below; light/mono variants carry the same semantics.
pub struct Palette {
    /// Primary accent — brand, tab labels, links, non-cursor chrome.
    /// Intentionally *not* used for the list/cursor selection bar so users
    /// can tell "where am I" apart from "which screen am I on".
    pub accent: Color,
    /// Cursor / selected row fill — high-contrast warm bar, distinct hues
    /// from accent so focus never collides with green fits or red errors.
    pub select: Color,
    /// Success / positive — running server, installed badges, ready toasts.
    pub ok: Color,
    /// Warning — tight fit, queued items, caution toasts.
    pub warn: Color,
    /// Danger / destructive — errors, delete modals, failed toasts.
    pub danger: Color,
    /// Special / featured — MTP badges, action highlights.
    pub feature: Color,
    /// Muted labels, secondary text, dimmed borders.
    pub muted: Color,
    /// Neutral secondary — values, descriptions.
    pub dim: Color,
    /// Primary readable text on default background.
    pub text: Color,
    /// Text on filled accent/ok chips.
    pub on_accent: Color,
    /// Text on the selection bar (same ink as on_accent; named for call sites).
    pub on_select: Color,
    /// Active border for the focused panel — slate, not accent, so the frame
    /// does not look like a second selection bar.
    pub border_active: Color,
    /// Inactive border color for unfocused widgets.
    pub border_inactive: Color,
    /// Filled background behind fenced code blocks in chat.
    pub code_bg: Color,
    /// Scrim fill behind modals and the help overlay.
    pub scrim_bg: Color,
}

/// Default palette for dark terminals (truecolor).
const DARK: Palette = Palette {
    accent: Color::Rgb(56, 189, 248),  // sky-400
    select: Color::Rgb(251, 191, 36),  // amber-400
    ok: Color::Rgb(74, 222, 128),      // green-400
    warn: Color::Rgb(250, 204, 21),    // yellow-400
    danger: Color::Rgb(248, 113, 113), // red-400
    // Named Magenta so non-truecolor terminals still differentiate from WARN.
    feature: Color::Magenta,
    // slate-400: DarkGray sat below contrast guidance for secondary text.
    muted: Color::Rgb(148, 163, 184),
    dim: Color::Gray,
    text: Color::White,
    on_accent: Color::Black,
    on_select: Color::Black,
    border_active: Color::Rgb(203, 213, 225), // slate-300
    border_inactive: Color::Rgb(148, 163, 184),
    code_bg: Color::Rgb(32, 33, 36),
    scrim_bg: Color::Black,
};

/// Light-terminal palette: darker, saturated variants that keep contrast on
/// white/pale backgrounds. Selected with `AX_TUI_THEME=light`.
const LIGHT: Palette = Palette {
    accent: Color::Rgb(3, 105, 161), // sky-700
    select: Color::Rgb(217, 119, 6), // amber-600
    ok: Color::Rgb(21, 128, 61),     // green-700
    warn: Color::Rgb(161, 98, 7),    // yellow-700
    danger: Color::Rgb(185, 28, 28), // red-700
    feature: Color::Magenta,
    muted: Color::Rgb(71, 85, 105), // slate-600
    dim: Color::Rgb(100, 116, 139), // slate-500
    text: Color::Black,
    on_accent: Color::White,
    on_select: Color::Black,
    border_active: Color::Rgb(71, 85, 105),     // slate-600
    border_inactive: Color::Rgb(148, 163, 184), // slate-400
    code_bg: Color::Rgb(229, 231, 235),         // gray-200
    scrim_bg: Color::Rgb(203, 213, 225),
};

/// NO_COLOR / mono mode: no hue at all. The selection bar keeps White-on-Black
/// (reverse video on every terminal) so focus stays visible; everything else
/// defers to terminal defaults and text/bold encoding.
const MONO: Palette = Palette {
    accent: Color::Reset,
    select: Color::White,
    ok: Color::Reset,
    warn: Color::Reset,
    danger: Color::Reset,
    feature: Color::Reset,
    muted: Color::DarkGray,
    dim: Color::Gray,
    text: Color::Reset,
    on_accent: Color::Reset,
    on_select: Color::Black,
    border_active: Color::White,
    border_inactive: Color::DarkGray,
    code_bg: Color::Reset,
    scrim_bg: Color::Reset,
};

static PALETTE: OnceLock<&'static Palette> = OnceLock::new();

/// The active palette (dark unless [`init`] selected another).
pub fn colors() -> &'static Palette {
    PALETTE.get().copied().unwrap_or(&DARK)
}

/// Resolve the palette + glyph set from the environment. Called once by the
/// TUI entry point; repeats are harmless (the first call wins).
pub fn init() {
    let palette = if std::env::var_os("NO_COLOR").is_some() {
        &MONO
    } else {
        match std::env::var("AX_TUI_THEME").ok().as_deref() {
            Some("light") => &LIGHT,
            Some("mono") => &MONO,
            _ => &DARK,
        }
    };
    let _ = PALETTE.set(palette);

    let locale = std::env::var("LC_ALL")
        .or_else(|_| std::env::var("LC_CTYPE"))
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default()
        .to_uppercase();
    let utf8 = locale.contains("UTF-8") || locale.contains("UTF8");
    let _ = icon::GLYPHS.set(if utf8 { &icon::UNICODE } else { &icon::ASCII });
}

// ---------------------------------------------------------------------------
// Reusable style constructors
// ---------------------------------------------------------------------------

/// Active list/cursor highlight: bold dark-on-amber (never sky-blue).
pub fn highlight_active() -> Style {
    Style::default()
        .bg(colors().select)
        .fg(colors().on_select)
        .add_modifier(Modifier::BOLD)
}

/// Selection ghost in an unfocused panel: dim underline only (no blue wash).
pub fn highlight_inactive() -> Style {
    Style::default()
        .fg(colors().text)
        .add_modifier(Modifier::UNDERLINED | Modifier::DIM)
}

/// Bold accent title (panel headers, brand) — text only, never a filled chip.
pub fn title() -> Style {
    Style::default()
        .fg(colors().accent)
        .add_modifier(Modifier::BOLD)
}

/// Muted secondary label.
pub fn label() -> Style {
    Style::default().fg(colors().muted)
}

/// Primary body text.
pub fn body() -> Style {
    Style::default().fg(colors().text)
}

/// Dim body / path text.
pub fn body_dim() -> Style {
    Style::default().fg(colors().dim)
}

/// Success emphasis.
pub fn ok() -> Style {
    Style::default().fg(colors().ok)
}

/// Warning emphasis.
pub fn warn() -> Style {
    Style::default().fg(colors().warn)
}

/// Danger emphasis.
pub fn danger() -> Style {
    Style::default().fg(colors().danger)
}

/// Feature / MTP emphasis.
pub fn feature() -> Style {
    Style::default().fg(colors().feature)
}

/// Primary CTA row (filled accent). Actions, not cursor position.
pub fn cta() -> Style {
    Style::default()
        .fg(colors().on_accent)
        .bg(colors().accent)
        .add_modifier(Modifier::BOLD)
}

/// Active tab for the current screen — underline + accent text only.
/// Avoids a filled blue bar that competes with the amber selection cursor.
pub fn tab_active() -> Style {
    Style::default()
        .fg(colors().accent)
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

/// Tab bar owns keyboard focus (↑ from content): filled amber so it matches
/// the selection language ("focus is here").
pub fn tab_keyboard_focus() -> Style {
    Style::default()
        .fg(colors().on_select)
        .bg(colors().select)
        .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
}

/// Inactive tab label.
pub fn tab_inactive() -> Style {
    Style::default().fg(colors().dim)
}

/// Separator between key hints in footers.
pub fn key_sep() -> Span<'static> {
    Span::styled(" · ", Style::default().fg(colors().muted))
}

/// A key hint span: bold key in white, no background chip.
pub fn key_hint(key: &str) -> Span<'static> {
    Span::styled(
        key.to_string(),
        Style::default()
            .fg(colors().text)
            .add_modifier(Modifier::BOLD),
    )
}

/// A key hint label (the description after the key).
pub fn key_label(label: &str) -> Span<'static> {
    Span::styled(label.to_string(), Style::default().fg(colors().dim))
}

/// A muted key chip for modal secondary actions.
pub fn key_chip_dim(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(colors().on_accent).bg(colors().dim),
    )
}

/// A danger key chip for destructive confirmations.
pub fn key_chip_danger(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(colors().text).bg(colors().danger),
    )
}

/// A primary key chip for modal primary actions.
pub fn key_chip(label: &str) -> Span<'static> {
    Span::styled(
        format!(" {label} "),
        Style::default().fg(colors().on_accent).bg(colors().accent),
    )
}

/// Status icon glyph contract (fixed lexicon). Unicode by default, ASCII when
/// the locale is not UTF-8 (see [`init`]).
pub mod icon {
    use std::sync::OnceLock;

    pub(super) struct Glyphs {
        pub ok: &'static str,
        pub running: &'static str,
        pub queued: &'static str,
        pub idle: &'static str,
        pub error: &'static str,
        pub warn: &'static str,
        pub speed: &'static str,
        pub select: &'static str,
        pub star: &'static str,
    }

    pub(super) const UNICODE: Glyphs = Glyphs {
        ok: "✓",
        running: "●",
        queued: "…",
        idle: "○",
        error: "✗",
        warn: "!",
        speed: "⚡",
        select: "›",
        star: "★",
    };

    pub(super) const ASCII: Glyphs = Glyphs {
        ok: "+",
        running: "*",
        queued: "~",
        idle: "-",
        error: "x",
        warn: "!",
        speed: "^",
        select: ">",
        star: "*",
    };

    pub(super) static GLYPHS: OnceLock<&'static Glyphs> = OnceLock::new();

    fn g() -> &'static Glyphs {
        GLYPHS.get().copied().unwrap_or(&UNICODE)
    }

    pub fn ok() -> &'static str {
        g().ok
    }
    pub fn running() -> &'static str {
        g().running
    }
    pub fn queued() -> &'static str {
        g().queued
    }
    pub fn idle() -> &'static str {
        g().idle
    }
    pub fn error() -> &'static str {
        g().error
    }
    pub fn warn() -> &'static str {
        g().warn
    }
    pub fn speed() -> &'static str {
        g().speed
    }
    pub fn select() -> &'static str {
        g().select
    }
    pub fn star() -> &'static str {
        g().star
    }
}
