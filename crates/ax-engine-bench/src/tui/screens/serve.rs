//! Serve screen: pick an installed model, start/stop `ax-engine-server`,
//! copy the URL, see a ready-to-paste curl example, and jump to Chat.
//! When the server is live, the layout prioritizes status + chat handoff.

use std::io::Write;
use std::process::{Command, Stdio};

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{ListItem, Paragraph};

use crate::tui::catalog::{self, installed_variants};
use crate::tui::theme;
use crate::tui::widgets::{self, ToastLevel, field_line};
use crate::tui::{App, Modal, Screen, ServeFocus};

impl App {
    pub(crate) fn on_key_serve(&mut self, code: KeyCode) {
        match self.serve_focus {
            ServeFocus::List => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.serve_idx == 0 {
                        self.focus_tab_bar();
                    } else {
                        self.serve_idx = self.serve_idx.saturating_sub(1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.serve_idx + 1 < installed_variants(&self.families).len() {
                        self.serve_idx += 1;
                    }
                }
                KeyCode::Tab => self.serve_focus = ServeFocus::Host,
                KeyCode::Enter => {
                    if self.server_ready {
                        self.navigate_to(Screen::Chat);
                    } else {
                        let pairs = installed_variants(&self.families);
                        if let Some(&(fi, vi)) = pairs.get(self.serve_idx) {
                            self.auto_chat_after_serve = true;
                            self.serve_installed(fi, vi);
                        }
                    }
                }
                KeyCode::Char('x') => {
                    if self.server_running() {
                        self.modal = Some(Modal::StopServer);
                    }
                }
                KeyCode::Char('c') => self.copy_server_url(),
                KeyCode::Char('t') => self.navigate_to(Screen::Chat),
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc if !self.go_back_screen() => {
                    self.screen = Screen::Home;
                    self.focus_tabs = false;
                    self.previous_screen = None;
                }
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
        let banner = if self.server_ready {
            Some((
                ToastLevel::Success,
                "Running — click or Enter to open Chat · c copy · x stop".to_string(),
            ))
        } else if self.server_running() {
            Some((
                ToastLevel::Warning,
                "Starting… large models can take a minute to load".to_string(),
            ))
        } else {
            None
        };

        let body = if let Some((kind, text)) = &banner {
            let split = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
            self.banner_rect.set(split[0]);
            widgets::draw_banner(frame, split[0], *kind, text);
            split[1]
        } else {
            area
        };

        // Live server: status console first; model list is secondary.
        // Idle: model list first, then config + log.
        if self.server_ready {
            let panels = Layout::vertical([
                Constraint::Length(8),
                Constraint::Min(4),
                Constraint::Length(4),
            ])
            .split(body);
            self.draw_server_panel(frame, panels[0]);
            widgets::draw_log(
                frame,
                panels[1],
                self.server.as_ref().map(|job| job.log.as_slice()),
                " Log ",
            );
            self.draw_serve_model_list(frame, panels[2]);
        } else if self.server_running() {
            let panels = Layout::vertical([
                Constraint::Length(7),
                Constraint::Min(4),
                Constraint::Length(5),
            ])
            .split(body);
            self.draw_server_panel(frame, panels[0]);
            widgets::draw_log(
                frame,
                panels[1],
                self.server.as_ref().map(|job| job.log.as_slice()),
                " Log ",
            );
            self.draw_serve_model_list(frame, panels[2]);
        } else {
            let panels = Layout::vertical([
                Constraint::Min(3),
                Constraint::Length(7),
                Constraint::Min(3),
            ])
            .split(body);
            self.draw_serve_model_list(frame, panels[0]);
            self.draw_server_panel(frame, panels[1]);
            widgets::draw_log(
                frame,
                panels[2],
                self.server.as_ref().map(|job| job.log.as_slice()),
                " Log ",
            );
        }
    }

    fn draw_serve_model_list(&self, frame: &mut Frame, area: Rect) {
        let pairs = installed_variants(&self.families);
        let rows: Vec<ListItem> = if pairs.is_empty() {
            vec![
                ListItem::new(Line::raw("")),
                ListItem::new(Line::from(vec![
                    Span::raw("  "),
                    Span::styled("No installed models", theme::warn()),
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
                    Span::styled(" Models to download one first.", theme::label()),
                ])),
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
                            format!("{:<16}", family.display_name()),
                            Style::default()
                                .fg(theme::TEXT)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{:<7}", v.precision()), theme::body_dim()),
                        Span::styled(
                            format!("{:>9}", catalog::format_bytes(v.size)),
                            theme::body_dim(),
                        ),
                        Span::raw(" "),
                        fit,
                    ]))
                })
                .collect()
        };
        let list_active = self.serve_focus == ServeFocus::List;
        widgets::render_list(
            frame,
            area,
            if self.server_ready {
                " Switch model "
            } else {
                " Models — Enter to serve "
            },
            rows,
            self.serve_idx,
            list_active && !self.server_ready,
            &self.content_list_rect,
        );
    }

    fn draw_server_panel(&self, frame: &mut Frame, area: Rect) {
        let host_line = field_line("Host", &self.host, self.serve_focus == ServeFocus::Host);
        let port_line = field_line("Port", &self.port, self.serve_focus == ServeFocus::Port);
        let status = match (&self.server_url, &self.server) {
            (Some(url), Some(job)) if job.done.is_none() && self.server_ready => Line::from(vec![
                Span::styled(format!(" {} ", theme::icon::RUNNING), theme::ok()),
                Span::styled("running at ", theme::ok()),
                Span::styled(
                    url.clone(),
                    Style::default().fg(theme::OK).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(" Enter chat ", theme::cta()),
                Span::styled("  c copy · x stop", theme::label()),
            ]),
            (Some(_), Some(job)) if job.done.is_none() => Line::from(vec![
                Span::styled(format!(" {} ", theme::icon::QUEUED), theme::warn()),
                Span::styled(
                    "starting… (large models can take a minute to load)",
                    theme::warn(),
                ),
            ]),
            (_, Some(job)) if job.done.is_some() => Line::from(vec![
                Span::styled(format!(" {} ", theme::icon::ERROR), theme::danger()),
                Span::styled(
                    format!(
                        "failed: {}",
                        self.server_error_line()
                            .unwrap_or_else(|| "see log below".into())
                    ),
                    theme::danger(),
                ),
            ]),
            _ => Line::from(vec![
                Span::styled(format!(" {} ", theme::icon::IDLE), theme::label()),
                Span::styled("stopped — pick a model and press Enter", theme::label()),
            ]),
        };
        let error_line = match self.port_error() {
            Some(err) => Line::from(Span::styled(err, theme::danger())),
            None => Line::raw(""),
        };
        let curl_line = match (&self.server_url, self.server_ready) {
            (Some(url), true) => Line::from(Span::styled(
                format!(
                    "curl {url}/v1/chat/completions -H 'Content-Type: application/json' -d '{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}'"
                ),
                theme::label(),
            )),
            _ => Line::raw(""),
        };
        let active = matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port)
            || self.server_ready
            || self.server_running();
        frame.render_widget(
            Paragraph::new(vec![host_line, port_line, error_line, status, curl_line]).block(
                if active {
                    widgets::active_block(" Server ")
                } else {
                    widgets::soft_block(" Server ")
                },
            ),
            area,
        );
    }
}
