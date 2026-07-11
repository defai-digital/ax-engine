//! Chat screen: talk to the running server over its OpenAI-compatible
//! `/v1/chat/completions` endpoint.  Requests stream through a `curl -sN`
//! child process (the crate has no HTTP client dependency, and the server is
//! plain-HTTP localhost), reusing the same `Job` line-reader as downloads;
//! SSE `data:` lines append deltas to the transcript as they arrive.

use std::process::Command;

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect, Size};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tui_scrollview::ScrollView;

use crate::tui::jobs::Job;
use crate::tui::{App, Screen, widgets};

pub(crate) struct ChatMessage {
    pub from_user: bool,
    pub content: String,
}

pub(in crate::tui) struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    /// Cursor position in `input`, in characters.
    pub cursor: usize,
    /// Transcript scroll position.  RefCell because rendering (which happens
    /// with `&App`) has to clamp/apply the offset, mirroring the `Cell`
    /// click-target pattern in `App`.
    pub scroll: std::cell::RefCell<tui_scrollview::ScrollViewState>,
    /// While true the transcript follows new tokens (jumps to the bottom on
    /// every frame).  Scrolling up detaches; scrolling back to the bottom or
    /// sending a message re-attaches.
    pub autoscroll: bool,
    pub job: Option<Job>,
    pub error: Option<String>,
}

impl ChatState {
    pub fn new() -> Self {
        ChatState {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: std::cell::RefCell::new(tui_scrollview::ScrollViewState::new()),
            autoscroll: true,
            job: None,
            error: None,
        }
    }

    pub fn streaming(&self) -> bool {
        self.job.as_ref().is_some_and(|job| job.done.is_none())
    }

    pub fn cancel(&mut self) {
        if let Some(job) = &mut self.job {
            job.cancel();
        }
        self.job = None;
    }

    /// Drain streamed lines into the transcript; called every poll tick.
    pub fn tick(&mut self) {
        let Some(job) = &mut self.job else {
            return;
        };
        let fresh = job.tick();
        let mut finished = false;
        for line in &fresh {
            match parse_sse_line(line) {
                SseEvent::Delta(text) => {
                    if let Some(last) = self.messages.last_mut()
                        && !last.from_user
                    {
                        last.content.push_str(&text);
                    }
                }
                SseEvent::Done => finished = true,
                SseEvent::Error(message) => {
                    self.error = Some(message);
                    finished = true;
                }
                SseEvent::Ignored => {}
            }
        }
        if finished || job.done.is_some() {
            // Curl exiting non-zero without any delta = connection-level failure.
            if self.error.is_none()
                && job.done.is_some_and(|code| code != 0)
                && self
                    .messages
                    .last()
                    .is_some_and(|m| !m.from_user && m.content.is_empty())
            {
                self.error = Some(format!(
                    "request failed: {}",
                    job.log
                        .iter()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .cloned()
                        .unwrap_or_else(|| "no response from server".into())
                ));
            }
            self.job = None;
        }
    }
}

/// One parsed server-sent-events line from the completions stream.
#[derive(Debug, PartialEq)]
pub(crate) enum SseEvent {
    Delta(String),
    Done,
    Error(String),
    Ignored,
}

pub(crate) fn parse_sse_line(line: &str) -> SseEvent {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return SseEvent::Ignored;
    }
    let payload = match trimmed.strip_prefix("data:") {
        Some(rest) => rest.trim(),
        // A bare JSON object outside SSE framing is an error envelope
        // (e.g. {"error": ...} from a non-streaming failure response).
        None => {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
                && let Some(error) = value.get("error")
            {
                let message = error
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("server returned an error");
                return SseEvent::Error(message.to_string());
            }
            return SseEvent::Ignored;
        }
    };
    if payload == "[DONE]" {
        return SseEvent::Done;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return SseEvent::Ignored;
    };
    if let Some(error) = value.get("error") {
        let message = error
            .get("message")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("server returned an error");
        return SseEvent::Error(message.to_string());
    }
    let delta = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(serde_json::Value::as_str);
    match delta {
        Some(text) if !text.is_empty() => SseEvent::Delta(text.to_string()),
        _ => SseEvent::Ignored,
    }
}

impl App {
    pub(crate) fn on_key_chat(&mut self, key: KeyEvent) {
        // Hint screen (no ready server): a plain info page, not an input.
        // Global keys (1-5, q, ?) were already handled; back keys go Home.
        if !self.server_ready {
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') | KeyCode::Backspace
            ) {
                self.screen = Screen::Home;
            }
            return;
        }
        // Esc first: stop a live stream, otherwise leave the screen.
        if key.code == KeyCode::Esc {
            if self.chat.streaming() {
                self.chat.cancel();
                self.toast_warn("reply cancelled");
            } else {
                self.screen = Screen::Home;
            }
            return;
        }
        if key.code == KeyCode::Char('u') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.chat.input.clear();
            self.chat.cursor = 0;
            return;
        }
        match key.code {
            KeyCode::Enter => self.send_chat_message(),
            KeyCode::Char(c) => {
                let byte_idx = char_to_byte_idx(&self.chat.input, self.chat.cursor);
                self.chat.input.insert(byte_idx, c);
                self.chat.cursor += 1;
            }
            KeyCode::Backspace => {
                if self.chat.cursor > 0 {
                    self.chat.cursor -= 1;
                    let byte_idx = char_to_byte_idx(&self.chat.input, self.chat.cursor);
                    self.chat.input.remove(byte_idx);
                }
            }
            KeyCode::Left => self.chat.cursor = self.chat.cursor.saturating_sub(1),
            KeyCode::Right => {
                self.chat.cursor = (self.chat.cursor + 1).min(self.chat.input.chars().count());
            }
            KeyCode::Home => self.chat.cursor = 0,
            KeyCode::End => self.chat.cursor = self.chat.input.chars().count(),
            KeyCode::PageUp => self.scroll_transcript(true, true),
            KeyCode::PageDown => self.scroll_transcript(false, true),
            KeyCode::Up => self.scroll_transcript(true, false),
            KeyCode::Down => self.scroll_transcript(false, false),
            _ => {}
        }
    }

    pub(crate) fn scroll_chat(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up => self.scroll_transcript(true, false),
            KeyCode::Down => self.scroll_transcript(false, false),
            _ => {}
        }
    }

    /// Scroll the transcript and maintain the follow-new-tokens attachment:
    /// any upward scroll detaches, and scrolling down past the end (offset
    /// stops moving) re-attaches.
    fn scroll_transcript(&mut self, up: bool, page: bool) {
        let mut state = self.chat.scroll.borrow_mut();
        if up {
            self.chat.autoscroll = false;
            if page {
                state.scroll_page_up();
            } else {
                state.scroll_up();
            }
        } else {
            if page {
                state.scroll_page_down();
            } else {
                state.scroll_down();
            }
            if state.is_at_bottom() {
                self.chat.autoscroll = true;
            }
        }
    }

    fn send_chat_message(&mut self) {
        if self.chat.streaming() || self.chat.input.trim().is_empty() {
            return;
        }
        let Some(url) = self.server_url.clone().filter(|_| self.server_ready) else {
            self.chat.error = Some("start a server first (Serve screen, 4)".into());
            return;
        };
        let prompt = std::mem::take(&mut self.chat.input);
        self.chat.cursor = 0;
        self.chat.error = None;
        self.chat.autoscroll = true;
        self.chat.messages.push(ChatMessage {
            from_user: true,
            content: prompt,
        });
        // No "model" field: the server 400s on any name that differs from its
        // configured model_id and accepts requests that omit it entirely.
        let body = serde_json::json!({
            "messages": self
                .chat
                .messages
                .iter()
                .map(|m| {
                    serde_json::json!({
                        "role": if m.from_user { "user" } else { "assistant" },
                        "content": m.content,
                    })
                })
                .collect::<Vec<_>>(),
            "stream": true,
        });
        self.chat.messages.push(ChatMessage {
            from_user: false,
            content: String::new(),
        });
        let mut cmd = Command::new("curl");
        cmd.args([
            "-sN",
            "--max-time",
            "600",
            "-X",
            "POST",
            &format!("{url}/v1/chat/completions"),
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ]);
        self.chat.job = Some(
            Job::spawn_with_stdin(cmd, Some(body.to_string()), None)
                .unwrap_or_else(|err| Job::failed(format!("failed to launch curl: {err}"))),
        );
    }

    pub(crate) fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        if !self.server_ready {
            // Centered "no server" card.
            let card_w = 52u16.min(area.width.saturating_sub(4));
            let card_h = 7u16.min(area.height.saturating_sub(2));
            let card = widgets::centered_rect(card_w, card_h, area);
            let lines = vec![
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        "No server running",
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::raw(""),
                Line::from(vec![Span::raw(
                    "  Start one on the Serve screen, then come back.",
                )]),
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        "4",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" go to Serve", Style::default().fg(Color::DarkGray)),
                ]),
            ];
            frame.render_widget(
                Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::DarkGray))
                        .title(" Chat "),
                ),
                card,
            );
            return;
        }
        let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).split(area);
        self.draw_chat_transcript(frame, chunks[0]);
        self.draw_chat_input(frame, chunks[1]);
    }

    fn draw_chat_transcript(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for message in self.chat.messages.iter() {
            // Inline badge on the same line as the content.
            let (badge, badge_style, content_style) = if message.from_user {
                (
                    "[You]",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(Color::White),
                )
            } else {
                let model_name = self.server_model.as_deref().unwrap_or("Model");
                // Truncate long model names for the badge.
                let short = if model_name.len() > 16 {
                    &model_name[..14]
                } else {
                    model_name
                };
                let badge_text = format!("[{short}]");
                // We need a &'static str for Span::styled, but we own the string.
                // Use Span::raw for dynamic content.
                let _ = badge_text;
                (
                    "",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                    Style::default().fg(Color::Rgb(200, 220, 200)),
                )
            };
            let content = if !message.from_user && message.content.is_empty() {
                if self.chat.streaming() { "…" } else { "" }
            } else {
                &message.content
            };
            let first_line = if message.from_user {
                Line::from(vec![
                    Span::styled(format!("{badge} "), badge_style),
                    Span::styled(
                        content.lines().next().unwrap_or("").to_string(),
                        content_style,
                    ),
                ])
            } else {
                let model_name = self.server_model.as_deref().unwrap_or("Model");
                let short = if model_name.len() > 16 {
                    &model_name[..14]
                } else {
                    model_name
                };
                Line::from(vec![
                    Span::styled(
                        format!("[{short}] "),
                        Style::default()
                            .fg(Color::Green)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        content.lines().next().unwrap_or("").to_string(),
                        content_style,
                    ),
                ])
            };
            lines.push(first_line);
            // Additional lines (indented to align after badge).
            for part in content.lines().skip(1) {
                lines.push(Line::from(Span::styled(format!("  {part}"), content_style)));
            }
            lines.push(Line::raw(""));
        }
        if let Some(error) = &self.chat.error {
            lines.push(Line::from(vec![
                Span::styled("✗ ", Style::default().fg(Color::Red)),
                Span::styled(error.clone(), Style::default().fg(Color::Red)),
            ]));
        }
        let title = if self.chat.streaming() {
            " Chat — replying… (Esc cancels) "
        } else {
            " Chat "
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if self.chat.streaming() {
                Color::Yellow
            } else {
                Color::DarkGray
            }))
            .title(title);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let text_width = inner.width.saturating_sub(1).max(1);
        let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
        let content_height = (paragraph.line_count(text_width) as u16).max(1);
        let mut scroll_view = ScrollView::new(Size::new(text_width, content_height));
        scroll_view.render_widget(paragraph, Rect::new(0, 0, text_width, content_height));
        let mut state = self.chat.scroll.borrow_mut();
        if self.chat.autoscroll {
            state.scroll_to_bottom();
        }
        frame.render_stateful_widget(scroll_view, inner, &mut state);
    }

    fn draw_chat_input(&self, frame: &mut Frame, area: Rect) {
        let before: String = self.chat.input.chars().take(self.chat.cursor).collect();
        let at: String = self
            .chat
            .input
            .chars()
            .nth(self.chat.cursor)
            .map(|c| c.to_string())
            .unwrap_or_else(|| " ".into());
        let after: String = self.chat.input.chars().skip(self.chat.cursor + 1).collect();
        let line = Line::from(vec![
            Span::styled(
                " > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(before),
            Span::styled(at, Style::default().bg(Color::Cyan).fg(Color::Black)),
            Span::raw(after),
        ]);
        // Single-line inline input — no bordered block.
        frame.render_widget(Paragraph::new(line), area);
    }
}

fn char_to_byte_idx(text: &str, char_idx: usize) -> usize {
    text.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(text.len())
}
