//! Host load panel: Gauge (now) + Sparkline (trend) — ratatui demo pattern.
//!
//! Not using `Chart` (axes waste height on Home) or `BarChart` (categorical).

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph, Sparkline};

use crate::tui::catalog;
use crate::tui::metrics::LiveMetrics;
use crate::tui::theme;

/// Draw a compact three-row metrics block (memory / CPU / models-on-unified).
pub(crate) fn draw_live_metrics(frame: &mut Frame, area: Rect, metrics: &LiveMetrics) {
    if area.height < 5 {
        return;
    }
    let block = Block::default()
        .borders(Borders::LEFT)
        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
        .title(Span::styled(" Live load ", theme::title()));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Three metric bands: each = label line + gauge + sparkline.
    let bands = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Min(3),
    ])
    .split(inner);

    let mem_hist: Vec<u64> = metrics.mem_history.iter().copied().collect();
    let cpu_hist: Vec<u64> = metrics.cpu_history.iter().copied().collect();
    let models_hist: Vec<u64> = metrics.models_history.iter().copied().collect();
    draw_metric_band(
        frame,
        bands[0],
        "Memory",
        mem_label(metrics),
        metrics.mem_ratio(),
        &mem_hist,
        theme::ACCENT,
    );
    draw_metric_band(
        frame,
        bands[1],
        "CPU",
        cpu_label(metrics),
        metrics.cpu_ratio(),
        &cpu_hist,
        theme::OK,
    );
    draw_metric_band(
        frame,
        bands[2],
        "Models",
        models_label(metrics),
        metrics.models_ratio(),
        &models_hist,
        theme::FEATURE,
    );
}

fn draw_metric_band(
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
                format!(" {title:<7}"),
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(value_label, theme::body_dim()),
        ])),
        rows[0],
    );

    let ratio = ratio.unwrap_or(0.0);
    let gauge_color = if ratio >= 0.85 {
        theme::DANGER
    } else if ratio >= 0.70 {
        theme::WARN
    } else {
        color
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

fn mem_label(m: &LiveMetrics) -> String {
    match (m.used_ram_bytes, m.total_ram_bytes) {
        (Some(used), Some(total)) => {
            format!(
                "{} / {} used",
                catalog::format_bytes(used),
                catalog::format_bytes(total)
            )
        }
        (None, Some(total)) => format!("— / {} total", catalog::format_bytes(total)),
        _ => "unavailable".into(),
    }
}

fn cpu_label(m: &LiveMetrics) -> String {
    match (m.cpu_percent, m.logical_cpus, m.load_1m) {
        (Some(p), Some(n), Some(load)) => {
            format!("{p:.0}% avg · load {load:.2} · {n} cores")
        }
        (Some(p), Some(n), None) => format!("{p:.0}% avg · {n} cores"),
        (Some(p), None, _) => format!("{p:.0}% avg"),
        _ => "unavailable".into(),
    }
}

fn models_label(m: &LiveMetrics) -> String {
    match m.total_ram_bytes {
        Some(total) if total > 0 => format!(
            "{} on disk · {:.0}% of RAM (unified w/ GPU)",
            catalog::format_bytes(m.models_bytes),
            m.models_ratio().unwrap_or(0.0) * 100.0
        ),
        _ => format!("{} installed", catalog::format_bytes(m.models_bytes)),
    }
}
