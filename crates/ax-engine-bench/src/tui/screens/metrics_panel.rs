//! Host monitor for Apple Silicon — compact meters on top, **large charts at bottom**.
//!
//! ## Layout
//!
//! ```text
//! ┌ This Mac · Apple M5 Max · 18 CPU · 40 GPU · 128G unified ──────────┐
//! │ MEM ████ 28%  │ CPU ██ 12%  │ GPU █ 5% · 1.0G mem                  │
//! │ PID  RSS  %MEM  COMMAND                                            │
//! ├────────────────────────────────────────────────────────────────────┤
//! │ Mem  38%  ──────────────── continuous line (own row) ────────────  │
//! │ CPU  12%  ──────────────── continuous line (own row) ────────────  │
//! │ GPU   5%  ──────────────── continuous line (own row) ────────────  │
//! └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each metric gets its **own** chart row. Overlaying three Braille series in
//! one Chart makes later series erase earlier cells — the CPU line looked
//! broken even when history was fine. Stacked single-series charts fix that.
//!
//! Right edge of each series is forced to the live gauge value.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table,
};

use crate::tui::catalog;
use crate::tui::metrics::{LiveMetrics, PressureBand};
use crate::tui::theme;

/// Preferred total panel height (meters + procs + stacked charts).
pub(crate) const PREFERRED_HEIGHT: u16 = 22;

/// Fixed header (identity + meters).
const STRIP_H: u16 = 2;
/// Compact process strip.
const PROCS_H: u16 = 4;

/// Draw the host monitor. Charts sit at the **bottom** and take leftover height.
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 5 {
        draw_device_strip(frame, area, metrics);
        return;
    }

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
        .title(Span::styled(" This Mac ", theme::title()));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    // Charts last (bottom) with Constraint::Min so they eat remaining rows.
    let chunks = if inner.height >= 12 {
        Layout::vertical([
            Constraint::Length(STRIP_H),
            Constraint::Length(PROCS_H.min(inner.height.saturating_sub(STRIP_H + 6))),
            Constraint::Min(6),
        ])
        .split(inner)
    } else if inner.height >= 8 {
        Layout::vertical([
            Constraint::Length(STRIP_H),
            Constraint::Length(2),
            Constraint::Min(4),
        ])
        .split(inner)
    } else {
        Layout::vertical([Constraint::Length(STRIP_H), Constraint::Min(2)]).split(inner)
    };

    draw_device_strip(frame, chunks[0], metrics);
    if chunks.len() == 2 {
        draw_util_charts(frame, chunks[1], metrics);
        return;
    }
    draw_process_table(frame, chunks[1], metrics);
    draw_util_charts(frame, chunks[2], metrics);
}

/// Identity line + MEM / CPU / GPU meters (same colors as chart series).
fn draw_device_strip(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    if area.height == 0 {
        return;
    }
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).split(area);

    let chip = m.chip_name.as_deref().unwrap_or("Apple Silicon");
    let cpu_part = match (m.logical_cpus, m.perf_cpus, m.eff_cpus) {
        (Some(n), Some(p), Some(e)) if p + e == n || p + e > 0 => {
            format!("{n} CPU ({p}P+{e}E)")
        }
        (Some(n), _, _) => format!("{n} CPU"),
        _ => "? CPU".into(),
    };
    let gpu_part = m
        .gpu_cores
        .map(|c| format!("{c} GPU"))
        .unwrap_or_else(|| "GPU".into());
    let ram = m
        .total_ram_bytes
        .map(|b| format!("{} unified", catalog::format_bytes(b)))
        .unwrap_or_else(|| "RAM —".into());
    let load = m
        .load_1m
        .map(|l| format!("load {l:.2}"))
        .unwrap_or_else(|| "load —".into());
    let disk = m
        .free_disk_bytes
        .map(|b| format!("{} free disk", catalog::format_bytes(b)))
        .unwrap_or_else(|| "disk —".into());

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(
                format!("{chip} · {cpu_part} · {gpu_part} · {ram} · {load} · {disk}"),
                theme::body_dim(),
            ),
        ])),
        rows[0],
    );

    if rows[1].height == 0 {
        return;
    }
    let meters = Layout::horizontal([
        Constraint::Percentage(40),
        Constraint::Percentage(30),
        Constraint::Percentage(30),
    ])
    .split(rows[1]);
    draw_inline_meter(
        frame,
        meters[0],
        "MEM",
        mem_bar_label(m),
        m.mem_ratio(),
        theme::ACCENT,
    );
    draw_inline_meter(
        frame,
        meters[1],
        "CPU",
        cpu_bar_label(m),
        m.cpu_ratio(),
        theme::OK,
    );
    draw_inline_meter(
        frame,
        meters[2],
        "GPU",
        gpu_bar_label(m),
        m.gpu_ratio(),
        theme::FEATURE,
    );
}

fn draw_inline_meter(
    frame: &mut Frame,
    area: Rect,
    tag: &str,
    label: String,
    ratio: Option<f64>,
    series: Color,
) {
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
            .label(format!("{tag} {label}")),
        area,
    );
}

/// Three stacked single-series charts (Mem / CPU / GPU) — no Braille overdraw.
fn draw_util_charts(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    if area.height == 0 {
        return;
    }

    let mem_now = m.mem_ratio().map(|r| (r * 100.0).clamp(0.0, 100.0));
    let cpu_now = m.cpu_percent.map(|p| p.clamp(0.0, 100.0));
    let gpu_now = m.gpu_percent.map(|p| p.clamp(0.0, 100.0));

    let mem_pts = series_points(&m.mem_history, mem_now);
    let cpu_pts = series_points(&m.cpu_history, cpu_now);
    let gpu_pts = series_points(&m.gpu_history, gpu_now);

    // Give each series its own band so lines never erase each other.
    let bands = if area.height >= 9 {
        Layout::vertical([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(area)
    } else if area.height >= 6 {
        Layout::vertical([
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
            Constraint::Ratio(1, 3),
        ])
        .split(area)
    } else {
        // Tiny: fall back to CPU-only so at least one continuous line shows.
        draw_one_series(frame, area, "CPU", cpu_now, &cpu_pts, theme::OK, true);
        return;
    };

    draw_one_series(
        frame,
        bands[0],
        "Mem",
        mem_now,
        &mem_pts,
        theme::ACCENT,
        true,
    );
    draw_one_series(frame, bands[1], "CPU", cpu_now, &cpu_pts, theme::OK, true);
    draw_one_series(
        frame,
        bands[2],
        "GPU",
        gpu_now,
        &gpu_pts,
        theme::FEATURE,
        true,
    );
}

fn draw_one_series(
    frame: &mut Frame,
    area: Rect,
    name: &str,
    now: Option<f64>,
    pts: &[(f64, f64)],
    color: Color,
    show_y_labels: bool,
) {
    if area.height == 0 {
        return;
    }
    let title = format!(" {name} {} ", fmt_pct(now));
    if pts.len() < 2 {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!("  {name}: collecting samples…"),
                theme::label(),
            )))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::MUTED))
                    .title(Span::styled(title, Style::default().fg(color))),
            ),
            area,
        );
        return;
    }

    let x_max = (pts.len().saturating_sub(1)).max(1) as f64;
    let dataset = Dataset::default()
        .name(name)
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(pts);

    let y_labels = if show_y_labels && area.height >= 3 {
        vec![
            Span::styled("0", theme::label()),
            Span::styled("100", theme::label()),
        ]
    } else {
        vec![]
    };

    let mut y_axis = Axis::default()
        .style(Style::default().fg(theme::MUTED))
        .bounds([0.0, 100.0]);
    if !y_labels.is_empty() {
        y_axis = y_axis.labels(y_labels);
    }

    let chart = Chart::new(vec![dataset])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(
                    title,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )),
        )
        .x_axis(
            Axis::default()
                .style(Style::default().fg(theme::MUTED))
                .bounds([0.0, x_max]),
        )
        .y_axis(y_axis);

    frame.render_widget(chart, area);
}

/// Build chart points from history, ending at the live gauge value.
///
/// - History samples become x=0..n-1
/// - Live `now` overwrites the last y (or seeds two points if empty) so the
///   right edge always matches the meters above.
/// - Never inserts a gap: if live is missing, keep history as-is.
fn series_points(history: &std::collections::VecDeque<u64>, now: Option<f64>) -> Vec<(f64, f64)> {
    let mut pts: Vec<(f64, f64)> = history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, (*v as f64).clamp(0.0, 100.0)))
        .collect();

    if let Some(y) = now {
        let y = y.clamp(0.0, 100.0);
        if let Some(last) = pts.last_mut() {
            last.1 = y;
        } else {
            // No history yet: flat line at live value so the series is visible.
            pts.push((0.0, y));
            pts.push((1.0, y));
        }
    }

    // Line graph needs ≥2 points to draw a segment.
    if pts.len() == 1 {
        let y = pts[0].1;
        pts.push((1.0, y));
    }
    pts
}

fn fmt_pct(v: Option<f64>) -> String {
    match v {
        Some(p) => format!("{p:.0}%"),
        None => "—".into(),
    }
}

/// Process table: PID · RSS · %MEM · COMMAND.
fn draw_process_table(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let header = Row::new(vec!["PID", "RSS", "%MEM", "COMMAND"]).style(
        Style::default()
            .fg(theme::ACCENT)
            .add_modifier(Modifier::BOLD),
    );
    let total = m.total_ram_bytes.unwrap_or(0).max(1);
    let rows: Vec<Row> = if m.top_procs.is_empty() {
        vec![Row::new(vec![
            Cell::from("—"),
            Cell::from("—"),
            Cell::from("—"),
            Cell::from(Span::styled("no process sample yet", theme::label())),
        ])]
    } else {
        m.top_procs
            .iter()
            .take(area.height.saturating_sub(2).max(1) as usize)
            .map(|p| {
                let pct = (p.rss_bytes as f64 / total as f64 * 100.0).clamp(0.0, 100.0);
                Row::new(vec![
                    Cell::from(p.pid.to_string()),
                    Cell::from(catalog::format_bytes(p.rss_bytes)),
                    Cell::from(format!("{pct:.0}%")),
                    Cell::from(p.name.clone()),
                ])
                .style(theme::body())
            })
            .collect()
    };

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(6),
            Constraint::Min(12),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(theme::MUTED))
            .title(Span::styled(
                " Top memory (host RSS) · GPU mem is unified, not VRAM ",
                theme::label(),
            )),
    )
    .column_spacing(1);

    frame.render_widget(table, area);
}

fn mem_bar_label(m: &LiveMetrics) -> String {
    match (m.used_ram_bytes, m.total_ram_bytes) {
        (Some(u), Some(t)) => format!(
            "{:.0}%  {}/{}",
            m.mem_ratio().unwrap_or(0.0) * 100.0,
            catalog::format_bytes(u),
            catalog::format_bytes(t)
        ),
        _ => "—".into(),
    }
}

fn cpu_bar_label(m: &LiveMetrics) -> String {
    match m.cpu_percent {
        Some(p) => format!("{p:.0}%"),
        None => "—".into(),
    }
}

fn gpu_bar_label(m: &LiveMetrics) -> String {
    match (m.gpu_percent, m.gpu_mem_bytes) {
        (Some(p), Some(mem)) => format!("{p:.0}% · {} mem", catalog::format_bytes(mem)),
        (Some(p), None) => format!("{p:.0}%"),
        (None, Some(mem)) => format!("— · {} mem", catalog::format_bytes(mem)),
        (None, None) => "—".into(),
    }
}

#[cfg(test)]
mod series_tests {
    use super::series_points;
    use std::collections::VecDeque;

    #[test]
    fn series_ends_at_live_gauge() {
        let mut hist = VecDeque::new();
        hist.push_back(10);
        hist.push_back(20);
        hist.push_back(30);
        let pts = series_points(&hist, Some(42.0));
        assert_eq!(pts.len(), 3);
        assert!((pts.last().unwrap().1 - 42.0).abs() < 1e-9);
        assert!((pts[0].1 - 10.0).abs() < 1e-9);
        assert!((pts[1].1 - 20.0).abs() < 1e-9);
    }

    #[test]
    fn empty_history_seeds_from_live() {
        let hist = VecDeque::new();
        let pts = series_points(&hist, Some(55.0));
        assert!(pts.len() >= 2);
        assert!(pts.iter().all(|(_, y)| (*y - 55.0).abs() < 1e-9));
    }

    #[test]
    fn missing_live_keeps_history_continuous() {
        let mut hist = VecDeque::new();
        for v in [5u64, 10, 15, 20] {
            hist.push_back(v);
        }
        let pts = series_points(&hist, None);
        assert_eq!(pts.len(), 4);
        assert!((pts[3].1 - 20.0).abs() < 1e-9);
    }
}
