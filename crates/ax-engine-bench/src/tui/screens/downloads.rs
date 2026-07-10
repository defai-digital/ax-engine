//! Downloads screen: sequential background queue with a real progress gauge
//! (watched bytes vs. the static catalog total), phase labels from
//! `--progress-json`, speed sparkline, and the child process log.

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, ListItem, Paragraph, Sparkline};

use crate::tui::catalog;
use crate::tui::jobs::{self, SPINNER};
use crate::tui::widgets;
use crate::tui::{App, Modal, Screen};

impl App {
    pub(crate) fn on_key_downloads(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.download_idx = self.download_idx.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.download_idx + 1 < self.downloads.len() {
                    self.download_idx += 1;
                }
            }
            KeyCode::Enter => {
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
            KeyCode::Char('x') | KeyCode::Char('c') => {
                let Some(task) = self.downloads.get_mut(self.download_idx) else {
                    return;
                };
                if task.is_running() {
                    // Confirmed: multi-GB progress is at stake.
                    self.modal = Some(Modal::CancelDownload {
                        download_idx: self.download_idx,
                    });
                } else if task.is_queued() {
                    task.cancel();
                    let label = task.label.clone();
                    self.toast(format!("{label} removed from queue"));
                }
            }
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.screen = Screen::Home,
            _ => {}
        }
    }

    pub(crate) fn draw_downloads(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Min(6),
            Constraint::Length(7),
            Constraint::Min(5),
        ])
        .split(area);
        let rows: Vec<ListItem> = if self.downloads.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No downloads yet. Pick a model on the Models screen (2).",
                Style::default().fg(Color::Yellow),
            )))]
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
                    let status_style = match task.status_label().as_str() {
                        "cancelled" => Style::default().fg(Color::DarkGray),
                        "queued" => Style::default().fg(Color::Yellow),
                        "ready" => Style::default().fg(Color::Green),
                        "running" => Style::default().fg(Color::Cyan),
                        _ => Style::default().fg(Color::Red),
                    };
                    ListItem::new(Line::from(vec![
                        Span::styled(format!("#{:<3}", task.id), Style::default().fg(Color::Gray)),
                        Span::styled(format!("{:<13}", task.status_label()), status_style),
                        Span::raw(format!("{spin} {pct} ")),
                        Span::styled(
                            format!("{:<24}", task.label),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{:<7}", task.mode.label()),
                            Style::default().fg(Color::Gray),
                        ),
                        Span::raw(
                            task.dest
                                .as_ref()
                                .map(|path| path.display().to_string())
                                .unwrap_or_else(|| "HF cache".into()),
                        ),
                    ]))
                })
                .collect()
        };
        widgets::render_list(
            frame,
            chunks[0],
            " Downloads ",
            rows,
            self.download_idx,
            true,
            &self.content_list_rect,
        );

        self.draw_download_details(frame, chunks[1]);
        widgets::draw_log(
            frame,
            chunks[2],
            self.downloads
                .get(self.download_idx)
                .and_then(|task| task.job.as_ref())
                .map(|job| job.log.as_slice()),
            " Download log ",
        );
    }

    fn draw_download_details(&self, frame: &mut Frame, area: Rect) {
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" Selected download (Enter serve when ready · x cancel) ");
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
                        // A watch dir can pre-exist with old revisions (cache
                        // hit / resume), so cap the shown bytes at the total.
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
                .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
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
                Span::styled("phase:  ", Style::default().fg(Color::DarkGray)),
                Span::raw(phase),
            ])),
            chunks[1],
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("target: ", Style::default().fg(Color::DarkGray)),
                Span::raw(task.target.clone()),
            ])),
            chunks[2],
        );
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("output: ", Style::default().fg(Color::DarkGray)),
                Span::raw(
                    task.output_path()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "pending".into()),
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
                .style(Style::default().fg(Color::Cyan)),
            chunks[4],
        );
    }
}
