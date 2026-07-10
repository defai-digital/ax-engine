//! Chat screen: talk to the running server over its OpenAI-compatible
//! `/v1/chat/completions` endpoint.  Requests stream through a `curl -sN`
//! child process (the crate has no HTTP client dependency, and the server is
//! plain-HTTP localhost), reusing the same `Job` line-reader as downloads;
//! SSE `data:` lines append deltas to the transcript as they arrive.

use std::process::Command;

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use crate::tui::jobs::Job;
use crate::tui::{App, Screen};

pub(crate) struct ChatMessage {
    pub from_user: bool,
    pub content: String,
}

pub(crate) struct ChatState {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    /// Cursor position in `input`, in characters.
    pub cursor: usize,
    /// Lines scrolled up from the bottom of the transcript.
    pub scroll_up: u16,
    pub job: Option<Job>,
    pub error: Option<String>,
}

impl ChatState {
    pub fn new() -> Self {
        ChatState {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll_up: 0,
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
        // Esc first: stop a live stream, otherwise leave the screen.
        if key.code == KeyCode::Esc {
            if self.chat.streaming() {
                self.chat.cancel();
                self.toast("reply cancelled");
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
            KeyCode::PageUp => self.chat.scroll_up = self.chat.scroll_up.saturating_add(5),
            KeyCode::PageDown => self.chat.scroll_up = self.chat.scroll_up.saturating_sub(5),
            KeyCode::Up => self.chat.scroll_up = self.chat.scroll_up.saturating_add(1),
            KeyCode::Down => self.chat.scroll_up = self.chat.scroll_up.saturating_sub(1),
            _ => {}
        }
    }

    pub(crate) fn scroll_chat(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up => self.chat.scroll_up = self.chat.scroll_up.saturating_add(1),
            KeyCode::Down => self.chat.scroll_up = self.chat.scroll_up.saturating_sub(1),
            _ => {}
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
        self.chat.scroll_up = 0;
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
            frame.render_widget(
                Paragraph::new(vec![
                    Line::raw(""),
                    Line::from(Span::styled(
                        "  No server running.",
                        Style::default().add_modifier(Modifier::BOLD),
                    )),
                    Line::raw(""),
                    Line::raw("  Start one on the Serve screen (press 4), then come back here"),
                    Line::raw("  to chat with the model."),
                ])
                .block(Block::default().borders(Borders::ALL).title(" Chat ")),
                area,
            );
            return;
        }
        let chunks = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).split(area);
        self.draw_chat_transcript(frame, chunks[0]);
        self.draw_chat_input(frame, chunks[1]);
    }

    fn draw_chat_transcript(&self, frame: &mut Frame, area: Rect) {
        let width = area.width.saturating_sub(2).max(1) as usize;
        let mut lines: Vec<Line> = Vec::new();
        for message in &self.chat.messages {
            let (who, style) = if message.from_user {
                (
                    "You",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    self.server_model.as_deref().unwrap_or("Model"),
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                )
            };
            lines.push(Line::from(Span::styled(format!("{who}:"), style)));
            let content = if !message.from_user && message.content.is_empty() {
                if self.chat.streaming() { "…" } else { "" }
            } else {
                &message.content
            };
            for part in content.split('\n') {
                lines.push(Line::raw(format!("  {part}")));
            }
            lines.push(Line::raw(""));
        }
        if let Some(error) = &self.chat.error {
            lines.push(Line::from(Span::styled(
                format!("⚠ {error}"),
                Style::default().fg(Color::Red),
            )));
        }
        // Bottom-anchor: estimate wrapped rows so new tokens stay in view
        // unless the user scrolled up.
        let height = area.height.saturating_sub(2) as usize;
        let wrapped: usize = lines
            .iter()
            .map(|line| {
                let chars: usize = line.iter().map(|span| span.content.chars().count()).sum();
                (chars.max(1)).div_ceil(width)
            })
            .sum();
        let bottom_offset = wrapped.saturating_sub(height) as u16;
        let offset = bottom_offset.saturating_sub(self.chat.scroll_up);
        let title = if self.chat.streaming() {
            " Chat — replying… (Esc cancels) "
        } else {
            " Chat "
        };
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(title))
                .wrap(Wrap { trim: false })
                .scroll((offset, 0)),
            area,
        );
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
            Span::raw(before),
            Span::styled(at, Style::default().bg(Color::Cyan).fg(Color::Black)),
            Span::raw(after),
        ]);
        frame.render_widget(
            Paragraph::new(line).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Cyan))
                    .title(" Message — Enter sends "),
            ),
            area,
        );
    }
}

fn char_to_byte_idx(text: &str, char_idx: usize) -> usize {
    text.char_indices()
        .nth(char_idx)
        .map(|(i, _)| i)
        .unwrap_or(text.len())
}
