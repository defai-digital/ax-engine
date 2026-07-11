//! Chat screen: talk to the running server over its OpenAI-compatible
//! `/v1/chat/completions` endpoint.  Requests stream through a `curl -sN`
//! child process (the crate has no HTTP client dependency, and the server is
//! plain-HTTP localhost), reusing the same `Job` line-reader as downloads;
//! SSE `data:` lines append deltas to the transcript as they arrive.

use std::process::Command;

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect, Size};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tui_scrollview::ScrollView;

use crate::tui::jobs::Job;
use crate::tui::theme;
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
            if matches!(key.code, KeyCode::Up | KeyCode::Char('k')) {
                self.focus_tab_bar();
                return;
            }
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') | KeyCode::Backspace
            ) {
                if !self.go_back_screen() {
                    self.screen = Screen::Home;
                    self.focus_tabs = false;
                    self.previous_screen = None;
                }
            }
            return;
        }
        // Esc first: stop a live stream, otherwise leave the screen.
        if key.code == KeyCode::Esc {
            if self.chat.streaming() {
                self.chat.cancel();
                self.toast_warn("reply cancelled");
            } else if !self.go_back_screen() {
                self.screen = Screen::Home;
                self.focus_tabs = false;
                self.previous_screen = None;
            }
            return;
        }
        if key.code == KeyCode::Char('u') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.chat.input.clear();
            self.chat.cursor = 0;
            return;
        }
        // Multi-line: Ctrl+J / Shift+Enter / Alt+Enter insert a newline; bare Enter sends.
        if key.code == KeyCode::Char('j') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.insert_chat_char('\n');
            return;
        }
        if key.code == KeyCode::Enter
            && (key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT))
        {
            self.insert_chat_char('\n');
            return;
        }
        match key.code {
            KeyCode::Enter => self.send_chat_message(),
            KeyCode::Char(c) => self.insert_chat_char(c),
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
            KeyCode::Home => {
                // Start of current line (after last newline), in char indices.
                let before: Vec<char> = self.chat.input.chars().take(self.chat.cursor).collect();
                self.chat.cursor = before
                    .iter()
                    .rposition(|c| *c == '\n')
                    .map(|i| i + 1)
                    .unwrap_or(0);
            }
            KeyCode::End => {
                // End of current line (before next newline), in char indices.
                let after: Vec<char> = self.chat.input.chars().skip(self.chat.cursor).collect();
                if let Some(rel) = after.iter().position(|c| *c == '\n') {
                    self.chat.cursor += rel;
                } else {
                    self.chat.cursor = self.chat.input.chars().count();
                }
            }
            KeyCode::PageUp => self.scroll_transcript(true, true),
            KeyCode::PageDown => self.scroll_transcript(false, true),
            // Prefer vertical cursor motion inside multi-line drafts; otherwise scroll.
            // Empty draft + at top of transcript: further Up focuses the tab bar.
            KeyCode::Up => {
                if self.chat.input.contains('\n') {
                    // Top of multi-line draft → tab bar.
                    if !self.move_chat_cursor_vert(-1) {
                        self.focus_tab_bar();
                    }
                } else if self.chat.input.is_empty() {
                    // Empty composer: scroll transcript, then promote to tabs.
                    let at_top = self.chat.scroll.borrow().offset().y == 0;
                    if at_top {
                        self.focus_tab_bar();
                    } else {
                        self.scroll_transcript(true, false);
                    }
                } else {
                    self.scroll_transcript(true, false);
                }
            }
            KeyCode::Down => {
                if self.chat.input.contains('\n') {
                    let _ = self.move_chat_cursor_vert(1);
                } else {
                    self.scroll_transcript(false, false);
                }
            }
            _ => {}
        }
    }

    fn insert_chat_char(&mut self, c: char) {
        let byte_idx = char_to_byte_idx(&self.chat.input, self.chat.cursor);
        self.chat.input.insert(byte_idx, c);
        self.chat.cursor += 1;
    }

    /// Move the cursor up/down by one line within a multi-line draft.
    /// Returns false when the move is already blocked (top/bottom of draft).
    fn move_chat_cursor_vert(&mut self, dir: i32) -> bool {
        let text = &self.chat.input;
        let chars: Vec<char> = text.chars().collect();
        let cursor = self.chat.cursor.min(chars.len());
        // Find start of current line and column.
        let line_start = chars[..cursor]
            .iter()
            .rposition(|c| *c == '\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        let col = cursor - line_start;
        if dir < 0 {
            if line_start == 0 {
                return false;
            }
            let prev_end = line_start - 1; // the newline
            let prev_start = chars[..prev_end]
                .iter()
                .rposition(|c| *c == '\n')
                .map(|i| i + 1)
                .unwrap_or(0);
            let prev_len = prev_end - prev_start;
            self.chat.cursor = prev_start + col.min(prev_len);
            true
        } else {
            let next_nl = chars[cursor..]
                .iter()
                .position(|c| *c == '\n')
                .map(|i| cursor + i);
            let Some(nl) = next_nl else {
                return false;
            };
            let next_start = nl + 1;
            let next_end = chars[next_start..]
                .iter()
                .position(|c| *c == '\n')
                .map(|i| next_start + i)
                .unwrap_or(chars.len());
            let next_len = next_end - next_start;
            self.chat.cursor = next_start + col.min(next_len);
            true
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
            self.chat.error = Some("start a server first (press 4 Serve)".into());
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
            let card_w = 54u16.min(area.width.saturating_sub(4));
            let card_h = 8u16.min(area.height.saturating_sub(2));
            let card = widgets::centered_rect(card_w, card_h, area);
            let lines = vec![
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        "No server running",
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::raw(""),
                Line::from(Span::styled(
                    "  Start a model, then come back to chat.",
                    theme::label(),
                )),
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  Press "),
                    Span::styled(
                        "4",
                        Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" for Serve", theme::label()),
                ]),
            ];
            frame.render_widget(
                Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::BORDER_INACTIVE))
                        .title(Span::styled(" Chat ", theme::title())),
                ),
                card,
            );
            return;
        }
        // Grow the composer with hard newlines + soft wrap (cap height).
        let input_inner_cols = area.width.saturating_sub(4).max(8) as usize;
        let visual_lines = count_visual_lines(&self.chat.input, input_inner_cols).clamp(1, 6);
        let input_height = ((visual_lines as u16) + 2)
            .min(area.height.saturating_sub(4))
            .max(3);
        let chunks =
            Layout::vertical([Constraint::Min(3), Constraint::Length(input_height)]).split(area);
        self.draw_chat_transcript(frame, chunks[0]);
        self.draw_chat_input(frame, chunks[1]);
    }

    fn draw_chat_transcript(&self, frame: &mut Frame, area: Rect) {
        let mut lines: Vec<Line> = Vec::new();
        for message in self.chat.messages.iter() {
            let content = if !message.from_user && message.content.is_empty() {
                if self.chat.streaming() { "…" } else { "" }
            } else {
                &message.content
            };
            let first_line = if message.from_user {
                Line::from(vec![
                    Span::styled(
                        "You  ",
                        Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        content.lines().next().unwrap_or("").to_string(),
                        theme::body(),
                    ),
                ])
            } else {
                let model_name = self.server_model.as_deref().unwrap_or("Model");
                let short: String = model_name.chars().take(14).collect();
                Line::from(vec![
                    Span::styled(
                        format!("{short}  "),
                        Style::default().fg(theme::OK).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        content.lines().next().unwrap_or("").to_string(),
                        theme::body_dim(),
                    ),
                ])
            };
            lines.push(first_line);
            // Additional lines (indented to align after badge).
            let indent = if message.from_user { "     " } else { "      " };
            for part in content.lines().skip(1) {
                lines.push(Line::from(Span::styled(
                    format!("{indent}{part}"),
                    if message.from_user {
                        theme::body()
                    } else {
                        theme::body_dim()
                    },
                )));
            }
            lines.push(Line::raw(""));
        }
        if let Some(error) = &self.chat.error {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::icon::ERROR), theme::danger()),
                Span::styled(error.clone(), theme::danger()),
            ]));
        }
        let title = if self.chat.streaming() {
            " Chat — replying… (Esc cancels) "
        } else {
            " Chat "
        };
        let block = if self.chat.streaming() {
            widgets::active_block(title).border_style(Style::default().fg(theme::WARN))
        } else {
            widgets::soft_block(title)
        };
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
        let prompt = format!("{} ", theme::icon::SELECT);
        let mut cells: Vec<(char, bool)> = prompt.chars().map(|c| (c, false)).collect();
        for (i, c) in self.chat.input.chars().enumerate() {
            if i == self.chat.cursor {
                // Cursor sits on this char (or a trailing space if at end).
            }
            cells.push((c, i == self.chat.cursor));
        }
        if self.chat.cursor >= self.chat.input.chars().count() {
            cells.push((' ', true));
        }
        let cols = area.width.saturating_sub(2).max(8) as usize;
        let visual = layout_cells(&cells, cols);
        let mut lines: Vec<Line> = visual
            .into_iter()
            .map(|row| {
                let mut spans: Vec<Span<'static>> = Vec::new();
                let mut run = String::new();
                let mut run_is_cursor = false;
                for (ch, is_cursor) in row {
                    if is_cursor != run_is_cursor && !run.is_empty() {
                        let style = if run_is_cursor {
                            Style::default().bg(theme::ACCENT).fg(theme::ON_ACCENT)
                        } else {
                            theme::body()
                        };
                        spans.push(Span::styled(std::mem::take(&mut run), style));
                    }
                    run_is_cursor = is_cursor;
                    // Newlines are structural only — show a soft space at EOL.
                    run.push(if ch == '\n' { ' ' } else { ch });
                }
                if !run.is_empty() || run_is_cursor {
                    let style = if run_is_cursor {
                        Style::default().bg(theme::ACCENT).fg(theme::ON_ACCENT)
                    } else {
                        theme::body()
                    };
                    spans.push(Span::styled(run, style));
                }
                if spans.is_empty() {
                    spans.push(Span::styled(" ", theme::body()));
                }
                Line::from(spans)
            })
            .collect();
        if lines.is_empty() {
            lines.push(Line::from(Span::styled(" ", theme::body())));
        }
        if (lines.len() as u16) + 1 < area.height {
            lines.push(Line::from(Span::styled(
                "Enter send · Ctrl+J / Shift+Enter newline · Ctrl+1-5 screens · Esc leave",
                theme::label(),
            )));
        }
        frame.render_widget(
            Paragraph::new(lines).block(widgets::active_block(" Message ")),
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

/// Soft-wrap + hard-newline visual line count for the composer.
fn count_visual_lines(text: &str, cols: usize) -> usize {
    let cols = cols.max(1);
    let mut lines = 1usize;
    // Leading prompt is ~2 cols.
    let mut col = 2usize;
    for c in text.chars() {
        if c == '\n' {
            lines += 1;
            col = 0;
        } else {
            if col >= cols {
                lines += 1;
                col = 0;
            }
            col += 1;
        }
    }
    lines
}

/// Lay out cells into visual rows, wrapping at `cols` and breaking on `\n`.
fn layout_cells(cells: &[(char, bool)], cols: usize) -> Vec<Vec<(char, bool)>> {
    let cols = cols.max(1);
    let mut rows: Vec<Vec<(char, bool)>> = vec![Vec::new()];
    let mut col = 0usize;
    for &(ch, is_cursor) in cells {
        if ch == '\n' {
            rows.last_mut().unwrap().push(('\n', is_cursor));
            rows.push(Vec::new());
            col = 0;
            continue;
        }
        if col >= cols {
            rows.push(Vec::new());
            col = 0;
        }
        rows.last_mut().unwrap().push((ch, is_cursor));
        col += 1;
    }
    rows
}
