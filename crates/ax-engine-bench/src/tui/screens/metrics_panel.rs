//! nvtop-inspired host monitor for Mac (unified memory).
//!
//! Layout mirrors https://github.com/Syllo/nvtop (screenshot NVTOP_ex1):
//! ```text
//! ┌ Device strip: chip · cores · MEM meter · CPU meter · models ──────┐
//! │                                                                   │
//! │   100% ┤         ╭─ multi-series Chart (CPU / Mem / Models)       │
//! │        │    ╭────╯                                                │
//! │     0% ┴────────────────────────────────────────────────          │
//! │                                                                   │
//! ├ PID      RSS     %MEM  COMMAND  (htop-style process strip) ───────┤
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! No discrete GPU util/temp/power on Apple Silicon without privileged probes.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, LegendPosition, Paragraph, Row,
    Table,
};

use crate::tui::catalog;
use crate::tui::metrics::{LiveMetrics, PressureBand};
use crate::tui::theme;

/// Draw an nvtop-like host panel. Best at height ≥ 14; degrades below that.
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 6 {
        draw_device_strip(frame, area, metrics);
        return;
    }

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
        .title(Span::styled(
            " Host monitor · unified memory (Mac) ",
            theme::title(),
        ));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    // nvtop proportions: thin device header, large plot, process table.
    let chunks = if inner.height >= 14 {
        Layout::vertical([
            Constraint::Length(2), // device strip + meters
            Constraint::Min(6),    // chart
            Constraint::Length(6), // process table
        ])
        .split(inner)
    } else if inner.height >= 10 {
        Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(inner)
    } else {
        Layout::vertical([Constraint::Length(2), Constraint::Min(3)]).split(inner)
    };

    draw_device_strip(frame, chunks[0], metrics);
    if chunks.len() >= 2 {
        draw_util_chart(frame, chunks[1], metrics);
    }
    if chunks.len() >= 3 {
        draw_process_table(frame, chunks[2], metrics);
    }
}

/// Top strip like nvtop device lines: identity + inline MEM/CPU gauges.
fn draw_device_strip(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    if area.height == 0 {
        return;
    }
    let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).split(area);

    // Line 1: device identity (chip is not in LiveMetrics — show cores/load).
    let cores = m
        .logical_cpus
        .map(|n| format!("{n} cores"))
        .unwrap_or_else(|| "? cores".into());
    let load = m
        .load_1m
        .map(|l| format!("load {l:.2}"))
        .unwrap_or_else(|| "load —".into());
    let disk = m
        .free_disk_bytes
        .map(|b| format!("{} free disk", catalog::format_bytes(b)))
        .unwrap_or_else(|| "disk —".into());
    let models = format!(
        "models {} ({:.0}% RAM)",
        catalog::format_bytes(m.models_bytes),
        m.models_ratio().unwrap_or(0.0) * 100.0
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                " Device ",
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("Apple Silicon · {cores} · {load} · {disk} · {models}"),
                theme::body_dim(),
            ),
        ])),
        rows[0],
    );

    if rows[1].height == 0 {
        return;
    }
    // Line 2: side-by-side MEM + CPU meters (nvtop header bars).
    let meters =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[1]);
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
    // Prefix tag in label like nvtop "GPU [||||]  MEM [||||]"
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(color).bg(theme::MUTED))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{tag} {label}")),
        area,
    );
}

/// Multi-series utilization chart (0–100%), nvtop plot pane.
fn draw_util_chart(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    // Own the series data for this frame (Chart holds references).
    let cpu_pts: Vec<(f64, f64)> = m
        .cpu_history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v as f64))
        .collect();
    let mem_pts: Vec<(f64, f64)> = m
        .mem_history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v as f64))
        .collect();
    let models_pts: Vec<(f64, f64)> = m
        .models_history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v as f64))
        .collect();

    let x_max = cpu_pts
        .len()
        .max(mem_pts.len())
        .max(models_pts.len())
        .saturating_sub(1)
        .max(1) as f64;

    let mut datasets = Vec::new();
    if !cpu_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("CPU %")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::OK))
                .data(&cpu_pts),
        );
    }
    if !mem_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Mem %")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::ACCENT))
                .data(&mem_pts),
        );
    }
    if !models_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("Models %RAM")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::FEATURE))
                .data(&models_pts),
        );
    }

    if datasets.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  collecting samples…",
                theme::label(),
            )))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::MUTED))
                    .title(" Utilization % "),
            ),
            area,
        );
        return;
    }

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::MUTED))
                .title(Span::styled(
                    " Utilization % · CPU · Mem · Models/RAM ",
                    theme::label(),
                )),
        )
        .x_axis(
            Axis::default()
                .style(Style::default().fg(theme::MUTED))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::styled("past", theme::label()),
                    Span::styled("now", theme::label()),
                ]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(theme::MUTED))
                .bounds([0.0, 100.0])
                .labels(vec![
                    Span::styled("0", theme::label()),
                    Span::styled("50", theme::label()),
                    Span::styled("100", theme::label()),
                ]),
        )
        .legend_position(Some(LegendPosition::TopRight));

    frame.render_widget(chart, area);
}

/// Process table like nvtop footer (PID · RSS · %MEM · COMMAND).
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
                " Top memory (host RSS) · not GPU VRAM ",
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
