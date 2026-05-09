use crate::app::{
    AppState, AppTab, LoadState, ModelDownloadStatus, ModelFamily, ModelSelectorTarget,
    ServerControlButton, ServerControlSelection, ServerUrlKind,
};
use crate::contracts::{ArtifactEntry, DoctorReport, ModelCard, WorkflowCommand};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Wrap};

pub fn render(frame: &mut Frame<'_>, state: &AppState) {
    let chunks = app_chunks(frame.area());

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
        AppTab::Jobs => render_jobs(frame, state, chunks[1]),
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

pub fn tab_at_position(area: Rect, column: u16, row: u16) -> Option<AppTab> {
    let header = app_chunks(area)[0];
    if row != header.y.saturating_add(1) {
        return None;
    }
    if column <= header.x || column >= header.x.saturating_add(header.width).saturating_sub(1) {
        return None;
    }

    let mut start = header.x.saturating_add(1);
    for tab in AppTab::ALL {
        let width = tab.title().chars().count() as u16;
        let end = start.saturating_add(width);
        if column >= start && column < end {
            return Some(tab);
        }
        start = end.saturating_add(1);
    }
    None
}

pub fn server_control_at_position(
    area: Rect,
    column: u16,
    row: u16,
) -> Option<ServerControlSelection> {
    let body = app_chunks(area)[1];
    if column <= body.x || column >= body.x.saturating_add(body.width).saturating_sub(1) {
        return None;
    }

    let content_x = body.x.saturating_add(1);
    let content_y = body.y.saturating_add(1);
    if row == content_y.saturating_add(SERVER_BUTTON_LINE) {
        return server_button_at_column(content_x, column).map(ServerControlSelection::Button);
    }

    let first_url_row = content_y.saturating_add(SERVER_FIRST_URL_LINE);
    let url_count = ServerUrlKind::ALL.len() as u16;
    if row >= first_url_row && row < first_url_row.saturating_add(url_count) {
        let index = usize::from(row.saturating_sub(first_url_row));
        return Some(ServerControlSelection::Url(ServerUrlKind::ALL[index]));
    }

    None
}

pub fn model_selector_at_position(
    area: Rect,
    state: &AppState,
    column: u16,
    row: u16,
) -> Option<ModelSelectorTarget> {
    let body = app_chunks(area)[1];
    if column <= body.x || column >= body.x.saturating_add(body.width).saturating_sub(1) {
        return None;
    }

    let content_y = body.y.saturating_add(1);
    if row == content_y.saturating_add(MODEL_FAMILY_LINE) {
        return model_family_at_column(body.x.saturating_add(1), column)
            .map(ModelSelectorTarget::Family);
    }

    let first_size_row = content_y.saturating_add(MODEL_FIRST_SIZE_LINE);
    let size_count = state.model_download.entries().len() as u16;
    if row >= first_size_row && row < first_size_row.saturating_add(size_count) {
        let index = usize::from(row.saturating_sub(first_size_row));
        return Some(ModelSelectorTarget::Size(index));
    }

    let download_row = first_size_row.saturating_add(size_count).saturating_add(3);
    if row == download_row
        && column >= body.x.saturating_add(3)
        && column < body.x.saturating_add(13)
    {
        return Some(ModelSelectorTarget::Download);
    }

    None
}

const MODEL_FAMILY_LINE: u16 = 1;
const MODEL_FIRST_SIZE_LINE: u16 = 4;
const MODEL_FAMILY_PREFIX: &str = "family: ";

fn model_family_at_column(content_x: u16, column: u16) -> Option<ModelFamily> {
    let mut start = content_x.saturating_add(MODEL_FAMILY_PREFIX.chars().count() as u16);
    for family in ModelFamily::ALL {
        let end = start.saturating_add(family.label().chars().count() as u16 + 2);
        if column >= start && column < end {
            return Some(family);
        }
        start = end.saturating_add(1);
    }
    None
}

const SERVER_BUTTON_LINE: u16 = 3;
const SERVER_FIRST_URL_LINE: u16 = 13;
const SERVER_BUTTON_PREFIX: &str = "buttons: ";

fn server_button_at_column(content_x: u16, column: u16) -> Option<ServerControlButton> {
    let mut start = content_x.saturating_add(SERVER_BUTTON_PREFIX.chars().count() as u16);
    for button in ServerControlButton::ALL {
        let end = start.saturating_add(button.label().chars().count() as u16 + 2);
        if column >= start && column < end {
            return Some(button);
        }
        start = end.saturating_add(1);
    }
    None
}

fn app_chunks(area: Rect) -> std::rc::Rc<[Rect]> {
    Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(2),
        ])
        .split(area)
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
    let mut lines = model_selector_lines(state);
    lines.push(Line::from(""));
    lines.push(Line::from("current_model:"));
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

fn model_selector_lines(state: &AppState) -> Vec<Line<'static>> {
    let entry = state.model_download.selected_entry();
    let entries = state.model_download.entries();
    let mut lines = vec![
        Line::from(vec![
            Span::styled("download_selector: ", key_style()),
            Span::raw("choose family and size, then press d or click Download"),
        ]),
        model_family_line(state),
        Line::from(format!("selected_size: {}", entry.label)),
        Line::from("size_options:"),
    ];

    lines.extend(entries.iter().enumerate().map(|(index, entry)| {
        let selected = index == state.model_download.size_index;
        let marker = if selected { "*" } else { " " };
        let line = format!("  {marker} {} - {}", entry.label, entry.note);
        if selected {
            Line::styled(
                line,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Line::from(line)
        }
    }));
    lines.push(Line::from(""));
    lines.push(Line::from(format!("repo_id: {}", entry.repo_id)));
    lines.push(Line::from(format!(
        "command: python scripts/download_model.py {} --json",
        entry.repo_id
    )));
    lines.push(model_download_button_line(state));
    lines.push(Line::from(format!(
        "download_status: {}",
        model_download_status_label(&state.model_download.status)
    )));
    lines
}

fn model_family_line(state: &AppState) -> Line<'static> {
    let mut spans = vec![Span::raw(MODEL_FAMILY_PREFIX)];
    for (index, family) in ModelFamily::ALL.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(" "));
        }
        let label = format!("[{}]", family.label());
        if state.model_download.family == family {
            spans.push(Span::styled(
                label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::raw(label));
        }
    }
    Line::from(spans)
}

fn model_download_button_line(state: &AppState) -> Line<'static> {
    let selected = state.model_download.selected_target == Some(ModelSelectorTarget::Download);
    if selected {
        Line::styled(
            "[Download]",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Line::from("[Download]")
    }
}

fn model_download_status_label(status: &ModelDownloadStatus) -> String {
    match status {
        ModelDownloadStatus::Idle => "idle".to_string(),
        ModelDownloadStatus::Running(repo_id) => format!("running {repo_id}"),
        ModelDownloadStatus::Succeeded { model_dir, .. } => format!("ready {model_dir}"),
        ModelDownloadStatus::Failed { message, .. } => format!("failed {message}"),
    }
}

fn render_server(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    frame.render_widget(panel("Server", server_lines(state)), area);
}

fn server_lines(state: &AppState) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("control_status: ", key_style()),
            Span::raw(server_control_status(state)),
        ]),
        Line::from(format!("host: {}", state.server_control.host)),
        Line::from(format!("port: {}", state.server_control.port)),
        server_button_line(state),
        Line::from(format!("selection: {}", server_selection_label(state))),
        Line::from(""),
        Line::from(format!("base_url: {}", state.server_base_url())),
    ];

    match &state.server.health {
        LoadState::Ready(health) => {
            lines.push(Line::from(format!("health: {}", health.status)));
            lines.push(Line::from(format!("service: {}", health.service)));
        }
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            lines.push(Line::from(format!("health: unavailable ({message})")));
            lines.push(Line::from("service: unknown"));
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
            lines.push(Line::from(format!("model_id: unavailable ({message})")));
            lines.push(Line::from("selected_backend: unknown"));
        }
    }

    lines.push(Line::from(""));
    lines.push(Line::from("urls:"));
    lines.extend(state.server_endpoints().into_iter().map(|endpoint| {
        let selected =
            state.server_control.selected == Some(ServerControlSelection::Url(endpoint.kind));
        let marker = if selected { "*" } else { " " };
        let line = format!("  {marker} {}: {}", endpoint.kind.label(), endpoint.url);
        if selected {
            Line::styled(
                line,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Line::from(line)
        }
    }));
    lines
}

fn server_control_status(state: &AppState) -> &'static str {
    match &state.server.health {
        LoadState::Ready(_) => "connected",
        LoadState::Unavailable(_) if state.server.base_url.is_some() => "configured, unavailable",
        LoadState::NotLoaded(_) if state.server.base_url.is_some() => "configured",
        _ => "local preview",
    }
}

fn server_button_line(state: &AppState) -> Line<'static> {
    let mut spans = vec![Span::raw(SERVER_BUTTON_PREFIX)];
    for (index, button) in ServerControlButton::ALL.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::raw(" "));
        }
        let selected =
            state.server_control.selected == Some(ServerControlSelection::Button(button));
        let label = format!("[{}]", button.label());
        if selected {
            spans.push(Span::styled(
                label,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::raw(label));
        }
    }
    Line::from(spans)
}

fn server_selection_label(state: &AppState) -> String {
    match state.server_control.selected {
        Some(ServerControlSelection::Button(button)) => {
            format!("{} (preview)", button.command_name())
        }
        Some(ServerControlSelection::Url(kind)) => format!("{} URL", kind.label()),
        None => "none".to_string(),
    }
}

fn render_jobs(frame: &mut Frame<'_>, state: &AppState, area: ratatui::layout::Rect) {
    let lines = match &state.doctor {
        LoadState::Ready(report) => job_plan_lines(report),
        LoadState::Unavailable(message) | LoadState::NotLoaded(message) => {
            unavailable_lines("Job plan not loaded", message)
        }
    };
    frame.render_widget(panel("Jobs", lines), area);
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

fn job_plan_lines(report: &DoctorReport) -> Vec<Line<'static>> {
    let planned_jobs = 3
        + usize::from(report.workflow.download_model.is_some())
        + usize::from(report.workflow.source_root.is_some());
    let mut lines = vec![Line::from(format!("planned_jobs: {planned_jobs}"))];
    if let Some(command) = report.workflow.download_model.as_ref() {
        push_job_lines(
            &mut lines,
            JobProjection {
                id: "download-model",
                label: "Download model",
                kind: "download_model",
                evidence: "readiness",
                command,
                artifact_required: false,
            },
        );
    }
    push_job_lines(
        &mut lines,
        JobProjection {
            id: "generate-manifest",
            label: "Generate model manifest",
            kind: "generate_manifest",
            evidence: "readiness",
            command: &report.workflow.generate_manifest,
            artifact_required: false,
        },
    );
    push_job_lines(
        &mut lines,
        JobProjection {
            id: "server-launch",
            label: "Start local server",
            kind: "server_launch",
            evidence: "route_contract",
            command: &report.workflow.server,
            artifact_required: false,
        },
    );
    if report.workflow.source_root.is_some() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("server-smoke: ", key_style()),
            Span::raw("Run server smoke check"),
        ]));
        lines.push(Line::from("kind: server_smoke"));
        lines.push(Line::from("evidence: route_contract"));
        lines.push(Line::from("owns_process: true"));
        lines.push(Line::from(format!(
            "cwd: {}",
            report.workflow.source_root.as_deref().unwrap_or("current")
        )));
        lines.push(Line::from("command: bash scripts/check-server-preview.sh"));
    }
    push_job_lines(
        &mut lines,
        JobProjection {
            id: "benchmark-scenario",
            label: "Run benchmark scenario",
            kind: "benchmark_scenario",
            evidence: "workload_contract",
            command: &report.workflow.benchmark,
            artifact_required: true,
        },
    );
    lines
}

struct JobProjection<'a> {
    id: &'static str,
    label: &'static str,
    kind: &'static str,
    evidence: &'static str,
    command: &'a WorkflowCommand,
    artifact_required: bool,
}

fn push_job_lines(lines: &mut Vec<Line<'static>>, job: JobProjection<'_>) {
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(format!("{}: ", job.id), key_style()),
        Span::raw(job.label),
    ]));
    lines.push(Line::from(format!("kind: {}", job.kind)));
    lines.push(Line::from(format!("evidence: {}", job.evidence)));
    lines.push(Line::from("owns_process: true"));
    lines.push(Line::from(format!(
        "cwd: {}",
        job.command.cwd.as_deref().unwrap_or("current")
    )));
    lines.push(Line::from(format!(
        "command: {}",
        command_text(job.command)
    )));
    if job.artifact_required {
        lines.push(Line::from("official_result_guard: artifact path required"));
    }
}

fn command_text(command: &WorkflowCommand) -> String {
    let Some((program, args)) = command.argv.split_first() else {
        return "missing command".to_string();
    };
    format!("{program}{}", command_args_suffix(args))
}

fn command_args_suffix(args: &[String]) -> String {
    if args.is_empty() {
        String::new()
    } else {
        format!(" {}", args.join(" "))
    }
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
        let backend = TestBackend::new(100, 64);
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
        assert!(snapshot.contains("host: 127.0.0.1"));
        assert!(snapshot.contains("port: 8080"));
        assert!(snapshot.contains("[Start]"));
        assert!(snapshot.contains("[Stop]"));
        assert!(snapshot.contains("http://127.0.0.1:8080/health"));
        assert!(snapshot.contains("http://127.0.0.1:8080/v1/chat/completions"));
        assert!(snapshot.contains("server URL not configured"));
    }

    #[test]
    fn jobs_snapshot_projects_phase2_plan_with_evidence_labels() {
        let snapshot = render_snapshot(&sample_state(AppTab::Jobs));

        assert!(snapshot.contains("Jobs"));
        assert!(snapshot.contains("planned_jobs: 5"));
        assert!(snapshot.contains("server_launch"));
        assert!(snapshot.contains("route_contract"));
        assert!(snapshot.contains("benchmark_scenario"));
        assert!(snapshot.contains("workload_contract"));
        assert!(snapshot.contains("artifact path required"));
    }

    #[test]
    fn benchmark_and_artifact_snapshots_cover_read_only_tabs() {
        let benchmarks = render_snapshot(&sample_state(AppTab::Benchmarks));
        let artifacts = render_snapshot(&sample_state(AppTab::Artifacts));

        assert!(benchmarks.contains("Benchmark summary not loaded"));
        assert!(artifacts.contains("artifact root not configured"));
    }

    #[test]
    fn tab_hit_testing_maps_clicks_to_header_tabs() {
        let area = ratatui::layout::Rect::new(0, 0, 80, 24);

        assert_eq!(tab_at_position(area, 1, 1), Some(AppTab::Readiness));
        assert_eq!(tab_at_position(area, 11, 1), Some(AppTab::Models));
        assert_eq!(tab_at_position(area, 18, 1), Some(AppTab::Server));
        assert_eq!(tab_at_position(area, 25, 1), Some(AppTab::Jobs));
        assert_eq!(tab_at_position(area, 30, 1), Some(AppTab::Benchmarks));
        assert_eq!(tab_at_position(area, 41, 1), Some(AppTab::Artifacts));
        assert_eq!(tab_at_position(area, 0, 1), None);
        assert_eq!(tab_at_position(area, 1, 2), None);
    }

    #[test]
    fn server_hit_testing_maps_clicks_to_buttons_and_urls() {
        let area = ratatui::layout::Rect::new(0, 0, 100, 32);

        assert_eq!(
            server_control_at_position(area, 11, 7),
            Some(ServerControlSelection::Button(ServerControlButton::Start))
        );
        assert_eq!(
            server_control_at_position(area, 19, 7),
            Some(ServerControlSelection::Button(ServerControlButton::Stop))
        );
        assert_eq!(
            server_control_at_position(area, 26, 7),
            Some(ServerControlSelection::Button(ServerControlButton::Restart))
        );
        assert_eq!(
            server_control_at_position(area, 4, 17),
            Some(ServerControlSelection::Url(ServerUrlKind::Health))
        );
        assert_eq!(
            server_control_at_position(area, 4, 23),
            Some(ServerControlSelection::Url(ServerUrlKind::Completions))
        );
        assert_eq!(server_control_at_position(area, 4, 16), None);
    }
}
