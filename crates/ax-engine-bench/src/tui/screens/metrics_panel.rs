//! Host load panel inspired by htop / nvtop / nvidia-htop, adapted for Mac.
//!
//! Layout (nvtop-style meters + htop-style process strip):
//! ```text
//!  Live host · Apple Silicon (unified memory — not discrete VRAM)
//!  Mem   36G used / 91G free of 128G   ████████░░ 28%
//!        ▁▂▃▄▅▆▇▅▃                     ← sparkline trend
//!  CPU   12% · load 2.44 · 18 cores    ███░░░░░░░ 12%
//!        ▁▂▁▃▂▄▃
//!  Disk  1.2T free · Models 12G (9% RAM, GPU shares pool)
//!  Top   Code 3.6G · server 1.5G · …   ← htop-style consumers
//! ```

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};

use crate::tui::catalog;
use crate::tui::metrics::{LiveMetrics, PressureBand};
use crate::tui::theme;

/// Draw the host load block. Needs ~12 rows for full layout; degrades if short.
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 4 {
        return;
    }
    let block = Block::default()
        .borders(Borders::LEFT)
        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
        .title(Span::styled(
            " Live host · unified memory (Mac) ",
            theme::title(),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Prefer full nvtop-like layout when tall enough; else compress.
    if inner.height >= 11 {
        let bands = Layout::vertical([
            Constraint::Length(3), // mem
            Constraint::Length(3), // cpu
            Constraint::Length(2), // disk + models
            Constraint::Min(2),    // top procs
        ])
        .split(inner);
        draw_mem_band(frame, bands[0], metrics);
        draw_cpu_band(frame, bands[1], metrics);
        draw_disk_models(frame, bands[2], metrics);
        draw_top_procs(frame, bands[3], metrics);
    } else {
        let bands = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(2),
            Constraint::Min(1),
        ])
        .split(inner);
        draw_compact_meter(
            frame,
            bands[0],
            "Mem",
            mem_summary(metrics),
            metrics.mem_ratio(),
            &metrics.mem_history.iter().copied().collect::<Vec<_>>(),
        );
        draw_compact_meter(
            frame,
            bands[1],
            "CPU",
            cpu_summary(metrics),
            metrics.cpu_ratio(),
            &metrics.cpu_history.iter().copied().collect::<Vec<_>>(),
        );
        draw_top_procs(frame, bands[2], metrics);
    }
}

fn draw_mem_band(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let hist: Vec<u64> = m.mem_history.iter().copied().collect();
    draw_meter_with_spark(
        frame,
        area,
        "Mem",
        mem_summary(m),
        m.mem_ratio(),
        &hist,
        theme::ACCENT,
    );
}

fn draw_cpu_band(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let hist: Vec<u64> = m.cpu_history.iter().copied().collect();
    draw_meter_with_spark(
        frame,
        area,
        "CPU",
        cpu_summary(m),
        m.cpu_ratio(),
        &hist,
        theme::OK,
    );
}

fn draw_disk_models(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let disk = m
        .free_disk_bytes
        .map(|b| format!("{} free disk", catalog::format_bytes(b)))
        .unwrap_or_else(|| "disk ?".into());
    let models = match m.total_ram_bytes {
        Some(total) if total > 0 => format!(
            "Models {} ({:.0}% RAM · GPU shares pool)",
            catalog::format_bytes(m.models_bytes),
            m.models_ratio().unwrap_or(0.0) * 100.0
        ),
        _ => format!("Models {}", catalog::format_bytes(m.models_bytes)),
    };
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).split(area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " Disk  ",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(disk, theme::body_dim()),
        ])),
        rows[0],
    );
    let models_ratio = m.models_ratio().unwrap_or(0.0);
    frame.render_widget(
        Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(band_color(LiveMetrics::pressure_band(models_ratio)))
                    .bg(theme::MUTED),
            )
            .ratio(models_ratio.clamp(0.0, 1.0))
            .label(models),
        rows[1],
    );
}

fn draw_top_procs(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    if area.height == 0 {
        return;
    }
    let mut spans = vec![Span::styled(
        " Top   ",
        Style::default()
            .fg(theme::TEXT)
            .add_modifier(Modifier::BOLD),
    )];
    if m.top_procs.is_empty() {
        spans.push(Span::styled("no process sample", theme::label()));
    } else {
        for (i, p) in m.top_procs.iter().take(4).enumerate() {
            if i > 0 {
                spans.push(Span::styled(" · ", theme::label()));
            }
            spans.push(Span::styled(
                format!("{} {}", p.name, catalog::format_bytes(p.rss_bytes)),
                theme::body(),
            ));
        }
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// Full meter: label, gauge, sparkline (nvtop-style).
fn draw_meter_with_spark(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    value_label: String,
    ratio: Option<f64>,
    history: &[u64],
    color: ratatui::style::Color,
) {
    let rows = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .split(area);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(" {title:<5}"),
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(value_label, theme::body_dim()),
        ])),
        rows[0],
    );

    let ratio = ratio.unwrap_or(0.0);
    // nvidia-htop -c style: free=green, moderate=yellow, full=red; else series color.
    let gauge_color = match LiveMetrics::pressure_band(ratio) {
        PressureBand::High => theme::DANGER,
        PressureBand::Medium => theme::WARN,
        PressureBand::Low => color,
    };
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(gauge_color).bg(theme::MUTED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{:.0}%", ratio * 100.0)),
        rows[1],
    );

    if !history.is_empty() && rows[2].height > 0 {
        frame.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(color)),
            rows[2],
        );
    }
}

/// One-line compact meter when vertical space is tight.
fn draw_compact_meter(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    value_label: String,
    ratio: Option<f64>,
    history: &[u64],
) {
    let cols =
        Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)]).split(area);
    let ratio = ratio.unwrap_or(0.0);
    let color = band_color(LiveMetrics::pressure_band(ratio));
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(color).bg(theme::MUTED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{title} {value_label}")),
        cols[0],
    );
    if !history.is_empty() {
        frame.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(color)),
            cols[1],
        );
    }
}

fn band_color(band: PressureBand) -> ratatui::style::Color {
    match band {
        PressureBand::Low => theme::OK,
        PressureBand::Medium => theme::WARN,
        PressureBand::High => theme::DANGER,
    }
}

fn mem_summary(m: &LiveMetrics) -> String {
    match (m.used_ram_bytes, m.free_ram_bytes, m.total_ram_bytes) {
        (Some(used), Some(free), Some(total)) => format!(
            "{} used / {} free of {}",
            catalog::format_bytes(used),
            catalog::format_bytes(free),
            catalog::format_bytes(total)
        ),
        (Some(used), None, Some(total)) => {
            format!(
                "{} / {} used",
                catalog::format_bytes(used),
                catalog::format_bytes(total)
            )
        }
        (None, None, Some(total)) => format!("— / {} total", catalog::format_bytes(total)),
        _ => "unavailable".into(),
    }
}

fn cpu_summary(m: &LiveMetrics) -> String {
    match (m.cpu_percent, m.logical_cpus, m.load_1m) {
        (Some(p), Some(n), Some(load)) => {
            format!("{p:.0}% avg · load {load:.2} · {n} cores")
        }
        (Some(p), Some(n), None) => format!("{p:.0}% avg · {n} cores"),
        (Some(p), None, _) => format!("{p:.0}% avg"),
        _ => "unavailable".into(),
    }
}
