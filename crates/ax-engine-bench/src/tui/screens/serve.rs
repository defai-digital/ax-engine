//! Serve screen: pick an installed model, start/stop `ax-engine-server`,
//! copy the URL, see a ready-to-paste curl example, and jump to Chat.

use std::io::Write;
use std::process::{Command, Stdio};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, ListItem, Paragraph};

use crate::tui::catalog::{self, installed_variants};
use crate::tui::widgets::{self, field_line};
use crate::tui::{App, Modal, Screen, ServeFocus};

impl App {
    pub(crate) fn on_key_serve(&mut self, code: KeyCode) {
        match self.serve_focus {
            ServeFocus::List => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.serve_idx = self.serve_idx.saturating_sub(1)
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.serve_idx + 1 < installed_variants(&self.families).len() {
                        self.serve_idx += 1;
                    }
                }
                KeyCode::Tab => self.serve_focus = ServeFocus::Host,
                KeyCode::Enter => {
                    let pairs = installed_variants(&self.families);
                    if let Some(&(fi, vi)) = pairs.get(self.serve_idx) {
                        self.serve_installed(fi, vi);
                    }
                }
                KeyCode::Char('x') => {
                    if self.server_running() {
                        self.modal = Some(Modal::StopServer);
                    }
                }
                KeyCode::Char('c') => self.copy_server_url(),
                KeyCode::Char('t') => self.screen = Screen::Chat,
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => self.screen = Screen::Home,
                _ => {}
            },
            ServeFocus::Host => self.edit_field(code, true),
            ServeFocus::Port => self.edit_field(code, false),
        }
    }

    fn edit_field(&mut self, code: KeyCode, is_host: bool) {
        let field = if is_host {
            &mut self.host
        } else {
            &mut self.port
        };
        match code {
            KeyCode::Char(c) => field.push(c),
            KeyCode::Backspace => {
                field.pop();
            }
            KeyCode::Tab => {
                self.serve_focus = if is_host {
                    ServeFocus::Port
                } else {
                    ServeFocus::List
                };
            }
            KeyCode::Enter | KeyCode::Esc => self.serve_focus = ServeFocus::List,
            _ => {}
        }
    }

    /// Put the server URL on the clipboard via `pbcopy` (macOS host).
    fn copy_server_url(&mut self) {
        let Some(url) = self.server_url.clone() else {
            return;
        };
        let copied = Command::new("pbcopy")
            .stdin(Stdio::piped())
            .spawn()
            .and_then(|mut child| {
                if let Some(stdin) = child.stdin.as_mut() {
                    stdin.write_all(url.as_bytes())?;
                }
                child.wait()
            })
            .map(|status| status.success())
            .unwrap_or(false);
        if copied {
            self.toast_success("URL copied");
        } else {
            self.toast_error("copy failed (pbcopy unavailable)");
        }
    }

    pub(crate) fn draw_serve(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([
            Constraint::Min(3),
            Constraint::Length(8),
            Constraint::Min(3),
        ])
        .split(area);

        let pairs = installed_variants(&self.families);
        let rows: Vec<ListItem> = if pairs.is_empty() {
            vec![
                ListItem::new(Line::raw("")),
                ListItem::new(Line::from(vec![
                    Span::raw("  "),
                    Span::styled(" ▶ ", Style::default().fg(Color::Black).bg(Color::Cyan)),
                    Span::raw(" No installed models"),
                ])),
                ListItem::new(Line::raw("")),
                ListItem::new(Line::from(Span::styled(
                    "  Download one on the Models screen (2) first.",
                    Style::default().fg(Color::DarkGray),
                ))),
            ]
        } else {
            pairs
                .iter()
                .map(|&(fi, vi)| {
                    let family = &self.families[fi];
                    let v = &family.variants[vi];
                    let fit = super::home::fit_span(catalog::ram_fit(
                        v.size_estimate(),
                        self.hardware.total_ram_bytes,
                    ));
                    ListItem::new(Line::from(vec![
                        Span::styled(
                            format!("{:<22}", v.profile.label),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            format!("{:<12}", catalog::format_bytes(v.size)),
                            Style::default().fg(Color::Gray),
                        ),
                        fit,
                        Span::raw(v.profile.repo_id),
                    ]))
                })
                .collect()
        };
        let list_active = self.serve_focus == ServeFocus::List;
        widgets::render_list(
            frame,
            chunks[0],
            " Installed models — Enter to serve ",
            rows,
            self.serve_idx,
            list_active,
            &self.content_list_rect,
        );

        self.draw_server_panel(frame, chunks[1]);
        widgets::draw_log(
            frame,
            chunks[2],
            self.server.as_ref().map(|job| job.log.as_slice()),
            " Server log ",
        );
    }

    fn draw_server_panel(&self, frame: &mut Frame, area: Rect) {
        let host_line = field_line("Host", &self.host, self.serve_focus == ServeFocus::Host);
        let port_line = field_line("Port", &self.port, self.serve_focus == ServeFocus::Port);
        let status = match (&self.server_url, &self.server) {
            (Some(url), Some(job)) if job.done.is_none() && self.server_ready => Line::from(vec![
                Span::styled(" ● ", Style::default().fg(Color::Green)),
                Span::styled("running at ", Style::default().fg(Color::Green)),
                Span::styled(
                    url.clone(),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("   c copy · t chat", Style::default().fg(Color::DarkGray)),
            ]),
            (Some(_), Some(job)) if job.done.is_none() => Line::from(vec![
                Span::styled(" ◌ ", Style::default().fg(Color::Yellow)),
                Span::styled(
                    "starting… (large models can take a minute to load)",
                    Style::default().fg(Color::Yellow),
                ),
            ]),
            (_, Some(job)) if job.done.is_some() => Line::from(vec![
                Span::styled(" ✗ ", Style::default().fg(Color::Red)),
                Span::styled(
                    format!(
                        "failed: {}",
                        self.server_error_line()
                            .unwrap_or_else(|| "see log below".into())
                    ),
                    Style::default().fg(Color::Red),
                ),
            ]),
            _ => Line::from(vec![
                Span::styled(" ○ ", Style::default().fg(Color::DarkGray)),
                Span::styled("stopped", Style::default().fg(Color::DarkGray)),
            ]),
        };
        let error_line = match self.port_error() {
            Some(err) => Line::from(Span::styled(err, Style::default().fg(Color::Red))),
            None => Line::raw(""),
        };
        let curl_line = match (&self.server_url, self.server_ready) {
            (Some(url), true) => Line::from(Span::styled(
                format!(
                    "curl {url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}'"
                ),
                Style::default().fg(Color::DarkGray),
            )),
            _ => Line::raw(""),
        };
        frame.render_widget(
            Paragraph::new(vec![host_line, port_line, error_line, status, curl_line]).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Server (Enter start · x stop · Tab fields) "),
            ),
            area,
        );
    }
}
