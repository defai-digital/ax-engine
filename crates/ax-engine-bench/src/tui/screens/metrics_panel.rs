//! Mac-native host card for the TUI Home screen.
//!
//! ## Design principles (not an nvtop clone)
//!
//! nvtop is built for **discrete GPUs** (util %, VRAM, clocks, encode, multi-device
//! plots). Apple Silicon does not expose that model cleanly. Copying nvtop’s
//! layout with fake “GPU %” series is misleading.
//!
//! What macOS gives us cheaply and honestly:
//! - Unified memory used / free / total (`vm_stat` + `hw.memsize`)
//! - CPU average across cores (`ps` %cpu / ncpu) + load average
//! - Free disk on the model cache volume
//! - Top host RSS processes (who is using memory)
//! - AX-only: installed model weight on disk vs unified RAM (serving headroom)
//!
//! What we deliberately **omit**: discrete GPU util, VRAM, fan, board power,
//! per-GPU process tables — none are first-class without privileged probes.
//!
//! ## Layout (Activity Monitor–ish card, launcher-friendly)
//! ```text
//! ┌ This Mac ─────────────────────────────────────────────┐
//! │ 18 cores · load 2.4 · 1.2T free disk                  │
//! │ Memory   36G used · 91G free · 128G total             │
//! │ ████████░░░░ 28%   ▁▂▃▄▅▆   (level + short trend)     │
//! │ CPU      12% avg                                      │
//! │ ███░░░░░░░░░ 12%   ▁▂▁▃▂                              │
//! │ Headroom models 12G on disk (9% of unified RAM)       │
//! │ ██░░░░░░░░░░                                          │
//! │ Using RAM  Code 3.6G · server 1.5G · …                │
//! └───────────────────────────────────────────────────────┘
//! ```
//!
//! Widgets: **Gauge** (now) + **Sparkline** (trend). No multi-axis Chart —
//! that shape implies nvtop-style multi-GPU telemetry we do not have.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};

use crate::tui::catalog;
use crate::tui::metrics::{LiveMetrics, PressureBand};
use crate::tui::theme;

/// Preferred height for the Mac host card on Home.
pub(crate) const PREFERRED_HEIGHT: u16 = 11;

/// Draw the Mac host card. Works down to ~6 rows; trims sections if shorter.
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 4 {
        return;
    }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
        .title(Span::styled(" This Mac ", theme::title()));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Adaptive sections: always identity + memory; add CPU / headroom / procs
    // only when height allows.
    match inner.height {
        0..=3 => {
            draw_identity(frame, inner, metrics);
        }
        4..=6 => {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(2),
                Constraint::Min(1),
            ])
            .split(inner);
            draw_identity(frame, rows[0], metrics);
            draw_resource_row(
                frame,
                rows[1],
                "Memory",
                mem_summary(metrics),
                metrics.mem_ratio(),
                &hist(metrics, HistKind::Mem),
                theme::ACCENT,
            );
            draw_resource_row(
                frame,
                rows[2],
                "CPU",
                cpu_summary(metrics),
                metrics.cpu_ratio(),
                &hist(metrics, HistKind::Cpu),
                theme::OK,
            );
        }
        7..=9 => {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Length(1),
                Constraint::Min(1),
            ])
            .split(inner);
            draw_identity(frame, rows[0], metrics);
            draw_resource_row(
                frame,
                rows[1],
                "Memory",
                mem_summary(metrics),
                metrics.mem_ratio(),
                &hist(metrics, HistKind::Mem),
                theme::ACCENT,
            );
            draw_resource_row(
                frame,
                rows[2],
                "CPU",
                cpu_summary(metrics),
                metrics.cpu_ratio(),
                &hist(metrics, HistKind::Cpu),
                theme::OK,
            );
            draw_headroom_line(frame, rows[3], metrics);
            draw_using_ram(frame, rows[4], metrics);
        }
        _ => {
            let rows = Layout::vertical([
                Constraint::Length(1),
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Length(2),
                Constraint::Min(1),
            ])
            .split(inner);
            draw_identity(frame, rows[0], metrics);
            draw_resource_row(
                frame,
                rows[1],
                "Memory",
                mem_summary(metrics),
                metrics.mem_ratio(),
                &hist(metrics, HistKind::Mem),
                theme::ACCENT,
            );
            draw_resource_row(
                frame,
                rows[2],
                "CPU",
                cpu_summary(metrics),
                metrics.cpu_ratio(),
                &hist(metrics, HistKind::Cpu),
                theme::OK,
            );
            draw_headroom_block(frame, rows[3], metrics);
            draw_using_ram(frame, rows[4], metrics);
        }
    }
}

enum HistKind {
    Mem,
    Cpu,
}

fn hist(m: &LiveMetrics, kind: HistKind) -> Vec<u64> {
    match kind {
        HistKind::Mem => m.mem_history.iter().copied().collect(),
        HistKind::Cpu => m.cpu_history.iter().copied().collect(),
    }
}

fn draw_identity(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let cores = m
        .logical_cpus
        .map(|n| format!("{n} cores"))
        .unwrap_or_else(|| "? cores".into());
    let load = m
        .load_1m
        .map(|l| format!("load {l:.2}"))
        .unwrap_or_else(|| "load —".into());
    let ram = m
        .total_ram_bytes
        .map(|b| format!("{} unified", catalog::format_bytes(b)))
        .unwrap_or_else(|| "RAM —".into());
    let disk = m
        .free_disk_bytes
        .map(|b| format!("{} free disk", catalog::format_bytes(b)))
        .unwrap_or_else(|| "disk —".into());
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(
                format!("{cores} · {load} · {ram} · {disk}"),
                theme::body_dim(),
            ),
        ])),
        area,
    );
}

/// One resource: title+numbers on line 1; gauge ‖ sparkline on line 2.
fn draw_resource_row(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    summary: String,
    ratio: Option<f64>,
    history: &[u64],
    series: ratatui::style::Color,
) {
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).split(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {title:<7}"),
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(summary, theme::body_dim()),
        ])),
        rows[0],
    );
    if rows[1].height == 0 {
        return;
    }
    let cols =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(rows[1]);
    let ratio = ratio.unwrap_or(0.0);
    let color = match LiveMetrics::pressure_band(ratio) {
        PressureBand::High => theme::DANGER,
        PressureBand::Medium => theme::WARN,
        PressureBand::Low => series,
    };
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(color).bg(theme::MUTED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.0}%", ratio * 100.0)),
        cols[0],
    );
    if !history.is_empty() {
        frame.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(series)),
            cols[1],
        );
    }
}

fn draw_headroom_line(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " Models ",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(headroom_summary(m), theme::body_dim()),
        ])),
        area,
    );
}

fn draw_headroom_block(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).split(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " Models ",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(headroom_summary(m), theme::body_dim()),
        ])),
        rows[0],
    );
    if rows[1].height == 0 {
        return;
    }
    let ratio = m.models_ratio().unwrap_or(0.0);
    let color = match LiveMetrics::pressure_band(ratio) {
        PressureBand::High => theme::DANGER,
        PressureBand::Medium => theme::WARN,
        PressureBand::Low => theme::FEATURE,
    };
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(color).bg(theme::MUTED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.0}% of unified RAM", ratio * 100.0)),
        rows[1],
    );
}

fn draw_using_ram(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let mut spans = vec![Span::styled(
        " Using  ",
        Style::default()
            .fg(theme::TEXT)
            .add_modifier(Modifier::BOLD),
    )];
    if m.top_procs.is_empty() {
        spans.push(Span::styled("sampling processes…", theme::label()));
    } else {
        for (i, p) in m.top_procs.iter().take(4).enumerate() {
            if i > 0 {
                spans.push(Span::styled(" · ", theme::label()));
            }
            spans.push(Span::styled(
                format!(
                    "{} {}",
                    short_name(&p.name),
                    catalog::format_bytes(p.rss_bytes)
                ),
                theme::body(),
            ));
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn short_name(name: &str) -> &str {
    const MAX: usize = 14;
    if name.chars().count() <= MAX {
        name
    } else {
        // Keep a stable prefix; full name is truncated for density.
        let end = name
            .char_indices()
            .nth(MAX.saturating_sub(1))
            .map(|(i, _)| i)
            .unwrap_or(name.len());
        &name[..end]
    }
}

fn mem_summary(m: &LiveMetrics) -> String {
    match (m.used_ram_bytes, m.free_ram_bytes, m.total_ram_bytes) {
        (Some(used), Some(free), Some(total)) => format!(
            "{} used · {} free · {} total",
            catalog::format_bytes(used),
            catalog::format_bytes(free),
            catalog::format_bytes(total)
        ),
        (Some(used), None, Some(total)) => format!(
            "{} / {} used",
            catalog::format_bytes(used),
            catalog::format_bytes(total)
        ),
        (None, None, Some(total)) => format!("{} total", catalog::format_bytes(total)),
        _ => "unavailable".into(),
    }
}

fn cpu_summary(m: &LiveMetrics) -> String {
    match (m.cpu_percent, m.logical_cpus, m.load_1m) {
        (Some(p), Some(n), Some(load)) => format!("{p:.0}% avg · load {load:.2} · {n} cores"),
        (Some(p), Some(n), None) => format!("{p:.0}% avg · {n} cores"),
        (Some(p), None, _) => format!("{p:.0}% avg"),
        _ => "unavailable".into(),
    }
}

fn headroom_summary(m: &LiveMetrics) -> String {
    match m.total_ram_bytes {
        Some(total) if total > 0 => format!(
            "{} installed on disk · {:.0}% of unified RAM (GPU shares this pool)",
            catalog::format_bytes(m.models_bytes),
            m.models_ratio().unwrap_or(0.0) * 100.0
        ),
        _ => format!(
            "{} installed on disk",
            catalog::format_bytes(m.models_bytes)
        ),
    }
}
