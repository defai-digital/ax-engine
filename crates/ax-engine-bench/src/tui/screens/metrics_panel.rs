//! Host monitor for Apple Silicon — big utilization chart + CPU/GPU meters.
//!
//! ## Layout (chart-first, Mac-honest data)
//!
//! ```text
//! ┌ This Mac · Apple M5 Max · 18 CPU (8P+10E) · 40 GPU · 128G unified ─┐
//! │ MEM ████░░ 28% 36G/128G │ CPU ██ 12% │ GPU █ 5%  · 1.0G GPU mem   │
//! │ ┌ Utilization % · CPU · GPU · Mem ──────────────────────────────┐ │
//! │ │ 100% ┤         ╭─ multi-series Chart (Braille lines)          │ │
//! │ │      │    ╭────╯                                              │ │
//! │ │   0% ┴────────────────────────────────────────────────        │ │
//! │ └───────────────────────────────────────────────────────────────┘ │
//! │ PID      RSS     %MEM  COMMAND                                    │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Sources (no sudo)
//! - CPU % · load · P/E cores — `ps` / `sysctl`
//! - GPU util % · GPU mem in use · chip · cores — `ioreg` IOAccelerator
//! - Unified memory — `vm_stat` (not discrete VRAM)
//!
//! Chart series are real: CPU, GPU device util, unified mem pressure.
//! We do **not** invent discrete VRAM/power/fan plots.

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

/// Preferred height so the chart has room (device strip + plot + processes).
pub(crate) const PREFERRED_HEIGHT: u16 = 16;

/// Draw the host monitor. Best at height ≥ 12; degrades cleanly below that.
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

    // Proportions: identity meters, large chart, optional process table.
    let chunks = if inner.height >= 14 {
        Layout::vertical([
            Constraint::Length(2), // identity + meters
            Constraint::Min(6),    // chart
            Constraint::Length(5), // process table
        ])
        .split(inner)
    } else if inner.height >= 10 {
        Layout::vertical([
            Constraint::Length(2),
            Constraint::Min(4),
            Constraint::Length(3),
        ])
        .split(inner)
    } else if inner.height >= 7 {
        Layout::vertical([Constraint::Length(2), Constraint::Min(3)]).split(inner)
    } else {
        Layout::vertical([Constraint::Length(2), Constraint::Min(1)]).split(inner)
    };

    draw_device_strip(frame, chunks[0], metrics);
    if chunks.len() >= 2 {
        draw_util_chart(frame, chunks[1], metrics);
    }
    if chunks.len() >= 3 {
        draw_process_table(frame, chunks[2], metrics);
    }
}

/// Identity line + MEM / CPU / GPU meters.
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

/// Multi-series utilization chart (0–100%): CPU · GPU · Mem.
fn draw_util_chart(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let cpu_pts: Vec<(f64, f64)> = m
        .cpu_history
        .iter()
        .enumerate()
        .map(|(i, v)| (i as f64, *v as f64))
        .collect();
    let gpu_pts: Vec<(f64, f64)> = m
        .gpu_history
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

    let x_max = cpu_pts
        .len()
        .max(gpu_pts.len())
        .max(mem_pts.len())
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
    if !gpu_pts.is_empty() {
        datasets.push(
            Dataset::default()
                .name("GPU %")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(theme::FEATURE))
                .data(&gpu_pts),
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

    if datasets.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  collecting CPU / GPU / mem samples…",
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
                    " Utilization % · CPU · GPU · Mem (unified) ",
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
