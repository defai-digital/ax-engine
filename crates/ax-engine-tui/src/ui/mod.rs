use crate::app::{AppState, AppTab, LoadState};
use crate::contracts::{ArtifactEntry, DoctorReport, ModelCard};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Wrap};

pub fn render(frame: &mut Frame<'_>, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(frame.area());

    let titles = AppTab::ALL
        .iter()
        .map(|tab| Line::from(tab.title()))
        .collect::<Vec<_>>();
    let selected = AppTab::ALL
        .iter()
        .position(|tab| *tab == state.selected_tab)
        .unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .block(
            Block::default()
                .title("AX Engine Manager")
                .borders(Borders::ALL),
        )
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, chunks[0]);

    match state.selected_tab {
        AppTab::Readiness => render_readiness(frame, state, chunks[1]),
        AppTab::Models => render_models(frame, state, chunks[1]),
        AppTab::Server => render_server(frame, state, chunks[1]),
        AppTab::Benchmarks => render_benchmarks(frame, state, chunks[1]),
        AppTab::Artifacts => render_artifacts(frame, state, chunks[1]),
    }

    frame.render_widget(
        Paragraph::new(state.status_message.as_str())
            .style(Style::default().fg(Color::DarkGray))
            .wrap(Wrap { trim: true }),
        chunks[2],
    );
}

fn render_readiness(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    let lines = match &state.doctor {
        LoadState::Ready(report) => readiness_lines(report),
        LoadState::Unavailable(message) => unavailable_lines("Doctor unavailable", message),
        LoadState::NotLoaded(message) => unavailable_lines("Doctor not loaded", message),
    };
    frame.render_widget(panel("Readiness", lines), area);
}

fn render_models(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    let mut lines = Vec::new();
    match &state.doctor {
        LoadState::Ready(report) => {
            lines.push(Line::from(vec![
                Span::styled("model_artifacts: ", key_style()),
                Span::raw(report.model_artifacts.status.clone()),
            ]));
            lines.push(Line::from(format!(
                "path: {}",
                report
                    .model_artifacts
                    .path
                    .as_deref()
                    .unwrap_or("not selected")
            )));
            lines.push(Line::from(format!(
                "config={} manifest={} safetensors={}",
                report.model_artifacts.config_present,
                report.model_artifacts.manifest_present,
                report.model_artifacts.safetensors_present
            )));
            lines.push(Line::from(format!(
                "model_type: {}",
                report
                    .model_artifacts
                    .model_type
                    .as_deref()
                    .unwrap_or("unknown")
            )));
            if let Some(quantization) = report.model_artifacts.quantization.as_ref() {
                lines.push(Line::from(format!(
                    "quantization: {} {}-bit group_size={}",
                    quantization.mode, quantization.bits, quantization.group_size
                )));
            }
            if !report.model_artifacts.issues.is_empty() {
                lines.push(Line::from("model blockers:"));
                lines.extend(
                    report
                        .model_artifacts
                        .issues
                        .iter()
                        .map(|issue| Line::from(format!("  - {issue}"))),
                );
            }
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            lines.extend(unavailable_lines("Model readiness unavailable", message));
        }
    }

    if let LoadState::Ready(models) = &state.server.models {
        lines.push(Line::from(""));
        lines.push(Line::from("server models:"));
        lines.extend(models.data.iter().map(model_card_line));
    }

    frame.render_widget(panel("Models", lines), area);
}

fn render_server(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    let mut lines = Vec::new();
    lines.push(Line::from(format!(
        "base_url: {}",
        state.server.base_url.as_deref().unwrap_or("not configured")
    )));
    match &state.server.health {
        LoadState::Ready(health) => {
            lines.push(Line::from(format!("health: {}", health.status)));
            lines.push(Line::from(format!("service: {}", health.service)));
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            lines.push(Line::from(format!("health: unavailable ({message})")));
        }
    }
    match &state.server.runtime {
        LoadState::Ready(runtime) => {
            let backend = runtime
                .runtime
                .get("selected_backend")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            lines.push(Line::from(format!("model_id: {}", runtime.model_id)));
            lines.push(Line::from(format!("selected_backend: {backend}")));
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            lines.push(Line::from(format!("runtime: unavailable ({message})")));
        }
    }
    frame.render_widget(panel("Server", lines), area);
}

fn render_benchmarks(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    let lines = match &state.benchmark_summary {
        LoadState::Ready(summary) => vec![
            Line::from(format!("command: {}", summary.command)),
            Line::from(format!("status: {}", summary.status)),
            Line::from(format!("result_dir: {}", summary.result_dir)),
            Line::from(format!(
                "manifest: {}",
                summary.manifest.as_deref().unwrap_or("none")
            )),
        ],
        LoadState::Unavailable(message) => {
            unavailable_lines("Benchmark summary unavailable", message)
        }
        LoadState::NotLoaded(message) => unavailable_lines("Benchmark summary not loaded", message),
    };
    frame.render_widget(panel("Benchmarks", lines), area);
}

fn render_artifacts(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    match &state.artifacts {
        LoadState::Ready(entries) => {
            let items = if entries.is_empty() {
                vec![ListItem::new("No benchmark artifacts found.")]
            } else {
                entries.iter().map(artifact_item).collect()
            };
            frame.render_widget(
                List::new(items).block(
                    Block::default()
                        .title(format!(
                            "Artifacts {}",
                            state.artifacts_root.as_deref().unwrap_or("")
                        ))
                        .borders(Borders::ALL),
                ),
                area,
            );
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            frame.render_widget(
                panel(
                    "Artifacts",
                    unavailable_lines("Artifacts unavailable", message),
                ),
                area,
            );
        }
    }
}

fn readiness_lines(report: &DoctorReport) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("status: ", key_style()),
            Span::styled(report.status.clone(), status_style(&report.status)),
        ]),
        Line::from(format!("mlx_runtime_ready: {}", report.mlx_runtime_ready)),
        Line::from(format!("bringup_allowed: {}", report.bringup_allowed)),
        Line::from(format!("workflow: {}", report.workflow.mode)),
        Line::from(format!(
            "source_root: {}",
            report.workflow.source_root.as_deref().unwrap_or("none")
        )),
        Line::from(format!(
            "model_artifacts: {}",
            report.model_artifacts.status
        )),
    ];
    if report.issues.is_empty() {
        lines.push(Line::from("issues: none"));
    } else {
        lines.push(Line::from("issues:"));
        lines.extend(
            report
                .issues
                .iter()
                .map(|issue| Line::from(format!("  - {issue}"))),
        );
    }
    lines
}

fn unavailable_lines(title: &str, message: &str) -> Vec<Line<'static>> {
    vec![
        Line::from(Span::styled(
            title.to_string(),
            Style::default().fg(Color::Yellow),
        )),
        Line::from(message.to_string()),
    ]
}

fn panel(title: &str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(
            Block::default()
                .title(title.to_string())
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: false })
}

fn model_card_line(card: &ModelCard) -> Line<'static> {
    let backend = card
        .runtime
        .get("selected_backend")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    Line::from(format!("  - {} ({backend})", card.id))
}

fn artifact_item(entry: &ArtifactEntry) -> ListItem<'static> {
    let marker = if entry.contract_failure_present {
        "contract_failure"
    } else {
        entry.status.as_str()
    };
    ListItem::new(format!(
        "{}  {}  {}",
        entry.name,
        marker,
        entry.path.display()
    ))
}

fn key_style() -> Style {
    Style::default().fg(Color::Cyan)
}

fn status_style(status: &str) -> Style {
    match status {
        "ready" | "ok" | "pass" => Style::default().fg(Color::Green),
        "not_ready" | "contract_failure" => Style::default().fg(Color::Red),
        _ => Style::default().fg(Color::Yellow),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{AppState, LoadState};
    use crate::contracts::{ModelArtifactsReport, WorkflowCommand, WorkflowReport};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    fn sample_state(tab: AppTab) -> AppState {
        let mut state = AppState::empty();
        state.selected_tab = tab;
        state.doctor = LoadState::Ready(DoctorReport {
            schema_version: "ax.engine_bench.doctor.v1".to_string(),
            status: "ready".to_string(),
            mlx_runtime_ready: true,
            bringup_allowed: true,
            workflow: WorkflowReport {
                mode: "source_checkout".to_string(),
                cwd: "/repo".to_string(),
                source_root: Some("/repo".to_string()),
                doctor: command("doctor"),
                server: command("server"),
                generate_manifest: command("generate-manifest"),
                benchmark: command("scenario"),
                download_model: Some(command("download")),
            },
            model_artifacts: ModelArtifactsReport {
                selected: true,
                status: "ready".to_string(),
                path: Some("/models/qwen".to_string()),
                exists: true,
                is_dir: true,
                config_present: true,
                manifest_present: true,
                safetensors_present: true,
                model_type: Some("qwen3".to_string()),
                quantization: Some(crate::contracts::QuantizationReport {
                    mode: "affine".to_string(),
                    group_size: 64,
                    bits: 4,
                }),
                issues: Vec::new(),
            },
            issues: Vec::new(),
            notes: Vec::new(),
            performance_advice: Vec::new(),
        });
        state
    }

    fn command(name: &str) -> WorkflowCommand {
        WorkflowCommand {
            argv: vec![name.to_string()],
            cwd: Some("/repo".to_string()),
        }
    }

    fn render_snapshot(state: &AppState) -> String {
        let backend = TestBackend::new(100, 28);
        let mut terminal = Terminal::new(backend).expect("terminal should create");
        terminal
            .draw(|frame| render(frame, state))
            .expect("draw should succeed");
        format!("{:?}", terminal.backend().buffer())
    }

    #[test]
    fn readiness_snapshot_contains_core_status() {
        let snapshot = render_snapshot(&sample_state(AppTab::Readiness));

        assert!(snapshot.contains("Readiness"));
        assert!(snapshot.contains("status:"));
        assert!(snapshot.contains("source_checkout"));
        assert!(snapshot.contains("model_artifacts"));
    }

    #[test]
    fn models_snapshot_contains_model_artifacts() {
        let snapshot = render_snapshot(&sample_state(AppTab::Models));

        assert!(snapshot.contains("Models"));
        assert!(snapshot.contains("/models/qwen"));
        assert!(snapshot.contains("qwen3"));
        assert!(snapshot.contains("4-bit"));
    }

    #[test]
    fn server_snapshot_shows_unavailable_state() {
        let snapshot = render_snapshot(&sample_state(AppTab::Server));

        assert!(snapshot.contains("Server"));
        assert!(snapshot.contains("server URL not configured"));
    }

    #[test]
    fn benchmark_and_artifact_snapshots_cover_read_only_tabs() {
        let benchmarks = render_snapshot(&sample_state(AppTab::Benchmarks));
        let artifacts = render_snapshot(&sample_state(AppTab::Artifacts));

        assert!(benchmarks.contains("Benchmark summary not loaded"));
        assert!(artifacts.contains("artifact root not configured"));
    }
}
