//! Downloads screen: sequential background queue with a real progress gauge
//! (watched bytes vs. the static catalog total), phase labels from
//! `--progress-json`, speed sparkline, and the child process log.

use std::process::Command;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Gauge, ListItem, Paragraph, Sparkline};

use crate::tui::catalog;
use crate::tui::jobs::{self, SPINNER};
use crate::tui::theme;
use crate::tui::widgets::{self, ToastLevel};
use crate::tui::{App, Modal, Screen};

impl App {
    pub(crate) fn on_key_downloads(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.download_idx == 0 {
                    self.focus_tab_bar();
                } else {
                    self.download_idx = self.download_idx.saturating_sub(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.download_idx + 1 < self.downloads.len() {
                    self.download_idx += 1;
                }
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self
                    .downloads
                    .get(self.download_idx)
                    .is_some_and(|task| task.is_ready())
                {
                    self.modal = Some(Modal::ServeReady {
                        download_idx: self.download_idx,
                    });
                }
            }
            KeyCode::Char('r') => self.retry_selected_download(),
            KeyCode::Char('o') => self.reveal_selected_download(),
            KeyCode::Backspace | KeyCode::Delete => self.remove_selected_if_done(),
            KeyCode::Char('d') => self.clear_finished_downloads(),
            KeyCode::Char('x') | KeyCode::Char('c') => {
                let Some(task) = self.downloads.get_mut(self.download_idx) else {
                    return;
                };
                if task.is_running() {
                    self.modal = Some(Modal::CancelDownload {
                        download_idx: self.download_idx,
                    });
                } else if task.is_queued() {
                    task.cancel();
                    let label = task.label.clone();
                    self.toast_warn(format!("{label} removed from queue"));
                }
            }
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                if !self.go_back_screen() {
                    self.screen = Screen::Home;
                    self.focus_tabs = false;
                    self.previous_screen = None;
                }
            }
            _ => {}
        }
    }

    fn retry_selected_download(&mut self) {
        let Some(task) = self.downloads.get_mut(self.download_idx) else {
            return;
        };
        if !(task.is_failed() || task.cancelled) {
            self.toast_warn("only failed or cancelled downloads can be retried");
            return;
        }
        let label = task.label.clone();
        task.requeue();
        self.start_next_queued_download();
        self.toast(format!("{label} re-queued"));
    }

    fn remove_selected_if_done(&mut self) {
        let Some(task) = self.downloads.get(self.download_idx) else {
            return;
        };
        if task.is_running() || task.is_queued() {
            self.toast_warn("cancel the download first (x)");
            return;
        }
        if !task.is_done() {
            return;
        }
        let label = task.label.clone();
        self.downloads.remove(self.download_idx);
        self.clamp_list_indices();
        self.toast(format!("{label} removed from list"));
    }

    fn clear_finished_downloads(&mut self) {
        let before = self.downloads.len();
        // Keep only active work; drop ready / failed / cancelled rows.
        self.downloads.retain(|t| t.is_running() || t.is_queued());
        let removed = before.saturating_sub(self.downloads.len());
        self.clamp_list_indices();
        if removed == 0 {
            self.toast("no finished items to clear");
        } else {
            self.toast_success(format!(
                "cleared {removed} finished item{}",
                if removed == 1 { "" } else { "s" }
            ));
        }
    }

    fn reveal_selected_download(&mut self) {
        let Some(task) = self.downloads.get(self.download_idx) else {
            return;
        };
        let path = task
            .output_path()
            .or_else(|| Some(task.watch_dir.clone()))
            .filter(|p| p.exists());
        let Some(path) = path else {
            self.toast_warn("no local path to open yet");
            return;
        };
        let opened = Command::new("open")
            .arg(&path)
            .spawn()
            .map(|mut child| child.wait().map(|s| s.success()).unwrap_or(false))
            .unwrap_or(false);
        if opened {
            self.toast_success(format!("opened {}", path.display()));
        } else {
            self.toast_error("could not open path (open unavailable)");
        }
    }

    pub(crate) fn draw_downloads(&self, frame: &mut Frame, area: Rect) {
        let selected_ready = self
            .downloads
            .get(self.download_idx)
            .is_some_and(|t| t.is_ready());
        let banner = if selected_ready {
            Some((
                ToastLevel::Success,
                "Ready — click or Enter to serve this model".to_string(),
            ))
        } else if let Some(task) = self.downloads.iter().find(|t| t.is_running()) {
            let pct = task
                .progress_ratio()
                .map(|r| format!("{:.0}%", r * 100.0))
                .unwrap_or_else(|| "…".into());
            Some((
                ToastLevel::Info,
                format!("Downloading {} · {pct}", task.label),
            ))
        } else if self
            .downloads
            .get(self.download_idx)
            .is_some_and(|t| t.is_failed())
        {
            Some((
                ToastLevel::Error,
                "Failed — press r to retry · ⌫ remove · d clear finished".into(),
            ))
        } else {
            None
        };

        let body_area = if let Some((kind, text)) = &banner {
            let split = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
            self.banner_rect.set(split[0]);
            widgets::draw_banner(frame, split[0], *kind, text);
            split[1]
        } else {
            area
        };

        let narrow = body_area.width < 100;
        if narrow {
            let panels = Layout::vertical([Constraint::Percentage(45), Constraint::Percentage(55)])
                .split(body_area);
            self.draw_download_queue(frame, panels[0]);
            let right =
                Layout::vertical([Constraint::Length(7), Constraint::Min(3)]).split(panels[1]);
            self.draw_download_details(frame, right[0]);
            widgets::draw_log(
                frame,
                right[1],
                self.downloads
                    .get(self.download_idx)
                    .and_then(|task| task.job.as_ref())
                    .map(|job| job.log.as_slice()),
                " Log ",
            );
        } else {
            let panels =
                Layout::horizontal([Constraint::Percentage(48), Constraint::Percentage(52)])
                    .split(body_area);
            self.draw_download_queue(frame, panels[0]);
            let right =
                Layout::vertical([Constraint::Length(7), Constraint::Min(3)]).split(panels[1]);
            self.draw_download_details(frame, right[0]);
            widgets::draw_log(
                frame,
                right[1],
                self.downloads
                    .get(self.download_idx)
                    .and_then(|task| task.job.as_ref())
                    .map(|job| job.log.as_slice()),
                " Log ",
            );
        }
    }

    fn draw_download_queue(&self, frame: &mut Frame, area: Rect) {
        let rows: Vec<ListItem> = if self.downloads.is_empty() {
            vec![
                ListItem::new(Line::raw("")),
                ListItem::new(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("No downloads yet", theme::warn()),
                ])),
                ListItem::new(Line::raw("")),
                ListItem::new(Line::from(vec![
                    Span::styled("  Press ", theme::label()),
                    Span::styled(
                        "2",
                        Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        " Models to pick a model and queue a download.",
                        theme::label(),
                    ),
                ])),
            ]
        } else {
            self.downloads
                .iter()
                .map(|task| {
                    let spin = task
                        .job
                        .as_ref()
                        .filter(|job| job.done.is_none())
                        .map(|job| SPINNER[job.spinner])
                        .unwrap_or(' ');
                    let pct = if task.is_running() {
                        task.progress_ratio()
                            .map(|r| format!("{:>4.0}%", r * 100.0))
                            .unwrap_or_else(|| "    ".into())
                    } else {
                        "    ".into()
                    };
                    let status_icon = match task.status_label().as_str() {
                        "cancelled" => theme::icon::IDLE,
                        "queued" => theme::icon::QUEUED,
                        "ready" => theme::icon::OK,
                        "running" => theme::icon::RUNNING,
                        _ => theme::icon::ERROR,
                    };
                    let status_style = match task.status_label().as_str() {
                        "cancelled" => theme::label(),
                        "queued" => theme::warn(),
                        "ready" => theme::ok(),
                        "running" => Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                        _ => theme::danger(),
                    };
                    let dest_short = task
                        .dest
                        .as_ref()
                        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                        .unwrap_or_else(|| "cache".into());
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("{status_icon} "), status_style),
                        Span::styled(format!("{:<7}", task.status_label()), status_style),
                        Span::raw(format!("{spin}{pct} ")),
                        Span::styled(
                            format!("{:<18}", task.label),
                            Style::default()
                                .fg(theme::TEXT)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{:<4}", task.mode.label()), theme::body_dim()),
                        Span::styled(dest_short, theme::label()),
                    ]))
                })
                .collect()
        };
        widgets::render_list(
            frame,
            area,
            " Queue · r retry · o open · ⌫ remove · d clear ",
            rows,
            self.download_idx,
            true,
            &self.content_list_rect,
        );
    }

    fn draw_download_details(&self, frame: &mut Frame, area: Rect) {
        let block = widgets::active_block(" Progress ");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let Some(task) = self.downloads.get(self.download_idx) else {
            return;
        };
        let chunks = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(inner);

        let (bytes, speed) = task
            .job
            .as_ref()
            .map(|job| (job.bytes, job.speed))
            .unwrap_or((0, 0.0));
        let gauge_label = if task.is_ready() {
            "100% · complete".to_string()
        } else {
            match (task.progress_ratio(), task.total_bytes) {
                (Some(ratio), Some(total)) => {
                    let eta = task
                        .eta_seconds()
                        .map(|s| format!(" · ETA {}", jobs::format_eta(s)))
                        .unwrap_or_default();
                    format!(
                        "{:.0}% · {} of ~{} · {}/s{eta}",
                        ratio * 100.0,
                        catalog::format_bytes(bytes.min(total)),
                        catalog::format_bytes(total),
                        catalog::format_bytes(speed as u64),
                    )
                }
                _ => format!(
                    "{} · {}/s",
                    catalog::format_bytes(bytes),
                    catalog::format_bytes(speed as u64)
                ),
            }
        };
        let ratio = if task.is_ready() {
            1.0
        } else {
            task.progress_ratio().unwrap_or(0.0)
        };
        frame.render_widget(
            Gauge::default()
                .gauge_style(Style::default().fg(theme::ACCENT).bg(theme::MUTED))
                .ratio(ratio)
                .label(gauge_label),
            chunks[0],
        );
        let phase = if task.is_ready() {
            "done".to_string()
        } else {
            task.phase.clone().unwrap_or_else(|| {
                if task.is_running() {
                    "downloading".into()
                } else {
                    task.status_label()
                }
            })
        };
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("phase:  ", theme::label()),
                Span::styled(phase, theme::body()),
            ])),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("target: ", theme::label()),
                Span::styled(task.target.clone(), theme::body()),
            ])),
            chunks[2],
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("output: ", theme::label()),
                Span::styled(
                    task.output_path()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "pending".into()),
                    theme::body_dim(),
                ),
            ])),
            chunks[3],
        );
        let history: &[u64] = task
            .job
            .as_ref()
            .map(|job| job.speed_history.as_slice())
            .unwrap_or(&[]);
        frame.render_widget(
            Sparkline::default()
                .data(history)
                .style(Style::default().fg(theme::ACCENT)),
            chunks[4],
        );
    }
}
