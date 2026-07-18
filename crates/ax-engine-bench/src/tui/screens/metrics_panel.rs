//! Host monitor for Apple Silicon — compact meters + **one multi-series chart**.
//!
//! ## Balancing continuous lines vs 3 colours + slow sampling
//!
//! | Goal | Approach |
//! |---|---|
//! | One chart, 3 colours | Single plot area, Mem / CPU / GPU |
//! | No “missing” gaps | **Line-only** Canvas (no scatter points that erase cells) |
//! | Slow 2 s samples | **Right-align** samples in a fixed time window so early
//! | | history does not stretch 3 points across the full width |
//! | Probe failure | Hold last value (lockstep histories in `metrics.rs`) |
//! | Right edge = gauges | Last point forced to live MEM/CPU/GPU % |
//!
//! ```text
//! ┌ This Mac · chip · CPU · GPU · unified RAM ─────────────────────────┐
//! │ MEM ████  │ CPU ██  │ GPU █                                       │
//! │ PID  RSS  %MEM  COMMAND                                           │
//! │ Utilization  Mem 38% · CPU 12% · GPU 5%                           │
//! │ 100 ┤                                    ╭─ growing from "now"    │
//! │   0 ┴────────────────────────────────────┴─────────────────────── │
//! │    ~4m                                                    now     │
//! └───────────────────────────────────────────────────────────────────┘
//! ```

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::canvas::{Canvas, Line as CanvasLine};
use ratatui::widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table};

use crate::tui::catalog;
use crate::tui::metrics::{HISTORY_LEN, LiveMetrics, PressureBand};
use crate::tui::theme;

/// Preferred minimum host panel height when callers size explicitly.
pub(crate) const PREFERRED_HEIGHT: u16 = 26;

const STRIP_H: u16 = 2;
/// Header + up to 10 process rows (title sits on the border). Fixed — extra
/// vertical space always goes to the utilization chart.
const PROCS_H: u16 = 12;
/// Never shrink the utilization chart below this.
const CHART_MIN_H: u16 = 10;

/// Draw the host monitor. Chart sits at the **bottom** and takes all leftover height.
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 5 {
        draw_device_strip(frame, area, metrics);
        return;
    }

    let outer = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::colors().border_inactive))
        .title(Span::styled(" This Mac ", theme::title()));
    let inner = outer.inner(area);
    frame.render_widget(outer, area);

    // strip (fixed) → top-10 procs (fixed) → chart (Min = all remaining rows).
    let chunks = if inner.height >= STRIP_H + PROCS_H + CHART_MIN_H {
        Layout::vertical([
            Constraint::Length(STRIP_H),
            Constraint::Length(PROCS_H),
            Constraint::Min(CHART_MIN_H),
        ])
        .split(inner)
    } else if inner.height >= STRIP_H + 6 + CHART_MIN_H {
        // Mid height: shrink process table first so the chart still grows.
        let procs = (inner.height - STRIP_H - CHART_MIN_H).clamp(4, PROCS_H);
        Layout::vertical([
            Constraint::Length(STRIP_H),
            Constraint::Length(procs),
            Constraint::Min(CHART_MIN_H),
        ])
        .split(inner)
    } else if inner.height >= 8 {
        Layout::vertical([
            Constraint::Length(STRIP_H),
            Constraint::Length(3),
            Constraint::Min(4),
        ])
        .split(inner)
    } else {
        Layout::vertical([Constraint::Length(STRIP_H), Constraint::Min(2)]).split(inner)
    };

    draw_device_strip(frame, chunks[0], metrics);
    if chunks.len() == 2 {
        draw_util_chart(frame, chunks[1], metrics);
        return;
    }
    draw_process_table(frame, chunks[1], metrics);
    draw_util_chart(frame, chunks[2], metrics);
}

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
        theme::colors().accent,
    );
    draw_inline_meter(
        frame,
        meters[1],
        "CPU",
        cpu_bar_label(m),
        m.cpu_ratio(),
        theme::colors().ok,
    );
    draw_inline_meter(
        frame,
        meters[2],
        "GPU",
        gpu_bar_label(m),
        m.gpu_ratio(),
        theme::colors().feature,
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
        PressureBand::High => theme::colors().danger,
        PressureBand::Medium => theme::colors().warn,
        PressureBand::Low => series,
    };
    frame.render_widget(
        Gauge::default()
            .gauge_style(Style::default().fg(color).bg(theme::colors().muted))
            .ratio(ratio.clamp(0.0, 1.0))
            .label(format!("{tag} {label}")),
        area,
    );
}

/// Single multi-series chart: line-only Canvas, fixed time window, right-aligned.
fn draw_util_chart(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    if area.height == 0 {
        return;
    }

    let mem_now = m.mem_ratio().map(|r| (r * 100.0).clamp(0.0, 100.0));
    let cpu_now = m.cpu_percent.map(|p| p.clamp(0.0, 100.0));
    let gpu_now = m.gpu_percent.map(|p| p.clamp(0.0, 100.0));

    // Fixed window = history capacity so sparse early samples sit at "now"
    // instead of being stretched across the full width (which looks broken).
    let window = HISTORY_LEN.max(2);
    let mem_pts = series_points(&m.mem_history, mem_now, window);
    let cpu_pts = series_points(&m.cpu_history, cpu_now, window);
    let gpu_pts = series_points(&m.gpu_history, gpu_now, window);

    let title = Line::from(vec![
        Span::styled(" Utilization  ", theme::label()),
        Span::styled(
            format!("Mem {} ", fmt_pct(mem_now)),
            Style::default().fg(theme::colors().accent),
        ),
        Span::styled(
            format!("CPU {} ", fmt_pct(cpu_now)),
            Style::default().fg(theme::colors().ok),
        ),
        Span::styled(
            format!("GPU {} ", fmt_pct(gpu_now)),
            Style::default().fg(theme::colors().feature),
        ),
    ]);

    let has_any = mem_pts.len() >= 2 || cpu_pts.len() >= 2 || gpu_pts.len() >= 2;
    if !has_any {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "  collecting Mem / CPU / GPU samples…",
                theme::label(),
            )))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(theme::colors().muted))
                    .title(title),
            ),
            area,
        );
        return;
    }

    let x_max = (window.saturating_sub(1)).max(1) as f64;
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::colors().muted))
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Y labels | canvas | bottom labels
    let body = if inner.height >= 4 {
        Layout::vertical([Constraint::Min(2), Constraint::Length(1)]).split(inner)
    } else {
        Layout::vertical([Constraint::Min(1)]).split(inner)
    };
    let plot_row = body[0];
    let cols = if plot_row.width >= 8 {
        Layout::horizontal([Constraint::Length(4), Constraint::Min(4)]).split(plot_row)
    } else {
        Layout::horizontal([Constraint::Length(0), Constraint::Min(1)]).split(plot_row)
    };

    if cols[0].width > 0 && cols[0].height > 0 {
        let y_labels = if cols[0].height >= 3 {
            Layout::vertical([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(cols[0])
        } else {
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(cols[0])
        };
        frame.render_widget(
            Paragraph::new(Span::styled("100", theme::label())),
            y_labels[0],
        );
        if y_labels.len() >= 3 {
            frame.render_widget(
                Paragraph::new(Span::styled("  0", theme::label())),
                y_labels[2],
            );
        }
    }

    // Line-only: no scatter Points layer (that was erasing series mid-line).
    // Draw order Mem → GPU → CPU so CPU wins residual cell collisions.
    // Single-letter end labels: the title legend is color-only, so label each
    // series at "now" for readers who cannot tell the hues apart.
    let mem_pts_c = mem_pts.clone();
    let cpu_pts_c = cpu_pts.clone();
    let gpu_pts_c = gpu_pts.clone();
    let canvas = Canvas::default()
        .marker(symbols::Marker::Braille)
        .x_bounds([0.0, x_max])
        .y_bounds([0.0, 100.0])
        .paint(move |ctx| {
            draw_polyline(ctx, &mem_pts_c, theme::colors().accent);
            draw_polyline(ctx, &gpu_pts_c, theme::colors().feature);
            draw_polyline(ctx, &cpu_pts_c, theme::colors().ok);
            for (pts, tag, color) in [
                (&mem_pts_c, "M", theme::colors().accent),
                (&gpu_pts_c, "G", theme::colors().feature),
                (&cpu_pts_c, "C", theme::colors().ok),
            ] {
                if let Some(&(_, y)) = pts.last() {
                    ctx.print(x_max, y, Span::styled(tag, Style::default().fg(color)));
                }
            }
        });
    frame.render_widget(canvas, cols[1]);

    if body.len() >= 2 {
        let x_row = Layout::horizontal([Constraint::Length(4), Constraint::Min(1)]).split(body[1]);
        if x_row[1].width >= 8 {
            let ends = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(x_row[1]);
            frame.render_widget(Paragraph::new(Span::styled("~4m", theme::label())), ends[0]);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled("now", theme::label())).right_aligned()),
                ends[1],
            );
        }
    }
}

fn draw_polyline(
    ctx: &mut ratatui::widgets::canvas::Context<'_>,
    pts: &[(f64, f64)],
    color: Color,
) {
    if pts.len() < 2 {
        return;
    }
    for pair in pts.windows(2) {
        ctx.draw(&CanvasLine {
            x1: pair[0].0,
            y1: pair[0].1,
            x2: pair[1].0,
            y2: pair[1].1,
            color,
        });
    }
}

/// Build right-aligned points in a fixed `[0, window)` time axis.
///
/// Slow sampling + left-growing x made 3–5 points stretch across the whole
/// chart (looked like missing data). Right-align so the trail grows from `now`
/// backward as history fills.
fn series_points(
    history: &std::collections::VecDeque<u64>,
    now: Option<f64>,
    window: usize,
) -> Vec<(f64, f64)> {
    let window = window.max(2);
    let mut vals: Vec<f64> = history
        .iter()
        .map(|v| (*v as f64).clamp(0.0, 100.0))
        .collect();

    if let Some(y) = now {
        let y = y.clamp(0.0, 100.0);
        if let Some(last) = vals.last_mut() {
            *last = y;
        } else {
            vals.push(y);
        }
    }

    if vals.is_empty() {
        return Vec::new();
    }
    if vals.len() == 1 {
        // Need a segment; hold flat for one step left of "now".
        vals.insert(0, vals[0]);
    }

    // Cap to window; drop oldest if we somehow exceed.
    if vals.len() > window {
        let skip = vals.len() - window;
        vals = vals[skip..].to_vec();
    }

    let start = window.saturating_sub(vals.len());
    vals.into_iter()
        .enumerate()
        .map(|(i, y)| ((start + i) as f64, y))
        .collect()
}

fn fmt_pct(v: Option<f64>) -> String {
    match v {
        Some(p) => format!("{p:.0}%"),
        None => "—".into(),
    }
}

fn draw_process_table(frame: &mut Frame, area: Rect, m: &LiveMetrics) {
    let header = Row::new(vec!["PID", "RSS", "%MEM", "COMMAND"]).style(
        Style::default()
            .fg(theme::colors().accent)
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
            .border_style(Style::default().fg(theme::colors().muted))
            .title(Span::styled(
                " Top 10 memory (host RSS) · unified, not VRAM ",
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
        let pts = series_points(&hist, Some(42.0), 10);
        assert_eq!(pts.len(), 3);
        assert!((pts.last().unwrap().1 - 42.0).abs() < 1e-9);
        // Right-aligned into window of 10 → x starts at 7.
        assert!((pts[0].0 - 7.0).abs() < 1e-9);
        assert!((pts.last().unwrap().0 - 9.0).abs() < 1e-9);
    }

    #[test]
    fn empty_history_seeds_from_live_right_aligned() {
        let hist = VecDeque::new();
        let pts = series_points(&hist, Some(55.0), 8);
        assert!(pts.len() >= 2);
        assert!(pts.iter().all(|(_, y)| (*y - 55.0).abs() < 1e-9));
        assert!((pts.last().unwrap().0 - 7.0).abs() < 1e-9);
    }

    #[test]
    fn missing_live_keeps_history_continuous() {
        let mut hist = VecDeque::new();
        for v in [5u64, 10, 15, 20] {
            hist.push_back(v);
        }
        let pts = series_points(&hist, None, 20);
        assert_eq!(pts.len(), 4);
        assert!((pts[3].1 - 20.0).abs() < 1e-9);
        // Right-aligned: last x = 19.
        assert!((pts[3].0 - 19.0).abs() < 1e-9);
    }

    #[test]
    fn full_window_starts_at_zero() {
        let mut hist = VecDeque::new();
        for v in 0..10u64 {
            hist.push_back(v);
        }
        let pts = series_points(&hist, Some(9.0), 10);
        assert_eq!(pts.len(), 10);
        assert!((pts[0].0 - 0.0).abs() < 1e-9);
        assert!((pts[9].0 - 9.0).abs() < 1e-9);
    }
}
