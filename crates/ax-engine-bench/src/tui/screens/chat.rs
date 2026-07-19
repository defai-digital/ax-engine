//! Chat screen: talk to the running server over its OpenAI-compatible
//! `/v1/chat/completions` endpoint.  Requests stream through a `curl -sN`
//! child process (the crate has no HTTP client dependency, and the server is
//! plain-HTTP localhost), reusing the same `Job` line-reader as downloads;
//! SSE `data:` lines append deltas to the transcript as they arrive.

use std::process::Command;
use std::rc::Rc;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect, Size};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use tui_scrollview::ScrollView;
use unicode_width::UnicodeWidthChar;

use crate::tui::jobs::Job;
use crate::tui::markdown::render_markdown;
use crate::tui::theme;
use crate::tui::{App, Modal, widgets};

pub(crate) struct ChatMessage {
    pub from_user: bool,
    pub content: String,
    /// Per-reply timing/throughput, filled when an assistant stream finishes.
    pub stats: Option<ReplyStats>,
}

/// Client-side reply measurements.  The server's SSE stream carries no usage
/// chunk, so token counts are estimates (`chars/4`) and rendered with `~`.
/// `ttft` is exact (send → first delta).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReplyStats {
    pub ttft: Duration,
    pub elapsed: Duration,
    pub est_tokens: usize,
}

impl ReplyStats {
    /// `0.8s TTFT · 12.3s · ~412 tok · ~33 tok/s`
    pub(crate) fn summary(&self) -> String {
        let secs = self.elapsed.as_secs_f64();
        let rate = if secs > 0.05 {
            self.est_tokens as f64 / secs
        } else {
            0.0
        };
        format!(
            "{:.1}s TTFT · {:.1}s · ~{} tok · ~{rate:.0} tok/s",
            self.ttft.as_secs_f64(),
            secs,
            self.est_tokens,
        )
    }
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
    /// Composer visual-row window start.  Cell because rendering (which
    /// happens with `&App`) clamps it so the cursor row stays inside the
    /// height-capped box, mirroring the transcript `scroll` RefCell pattern.
    pub input_scroll: std::cell::Cell<usize>,
    /// Whole-transcript render cache keyed by content hash + wrap width.
    /// `Rc` so a cache hit only clones a pointer, not every styled span.
    pub transcript_cache: std::cell::RefCell<Option<TranscriptCache>>,
    pub job: Option<Job>,
    pub error: Option<String>,
    /// Sent prompts, oldest first (readline-style recall with ↑/↓).
    pub history: Vec<String>,
    /// Active history navigation: (index into `history`, stashed draft).
    pub hist_nav: Option<(usize, String)>,
    /// When the current request was sent (stats: TTFT/elapsed).
    pub send_at: Option<Instant>,
    /// First streamed delta arrival (stats: TTFT).
    pub first_delta_at: Option<Instant>,
    /// Characters streamed so far in the current reply (stats: est. tokens).
    pub stream_chars: usize,
    /// Ctrl+T toggle: render settled thinking blocks in full instead of the
    /// collapsed preview. In-progress (streaming) thinking always renders
    /// fully so the user can watch the reasoning arrive.
    pub thinking_expanded: bool,
}

/// Cap on stored prompt history entries.
const HISTORY_CAP: usize = 100;

/// Settled thinking blocks collapse to this many leading lines; the rest
/// sits behind a one-line "N more" summary until Ctrl+T expands them.
const THINKING_PREVIEW_LINES: usize = 2;

/// Cached transcript lines + wrapped height for a (content key, width) pair.
pub(crate) struct TranscriptCache {
    key: u64,
    width: u16,
    lines: Rc<Vec<Line<'static>>>,
    content_height: u16,
}

impl ChatState {
    pub fn new() -> Self {
        ChatState {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: std::cell::RefCell::new(tui_scrollview::ScrollViewState::new()),
            autoscroll: true,
            input_scroll: std::cell::Cell::new(0),
            transcript_cache: std::cell::RefCell::new(None),
            job: None,
            error: None,
            history: Vec::new(),
            hist_nav: None,
            send_at: None,
            first_delta_at: None,
            stream_chars: 0,
            thinking_expanded: false,
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
    /// Returns true when the transcript or stream status may have changed.
    pub fn tick(&mut self) -> bool {
        let Some(job) = &mut self.job else {
            return false;
        };
        let job_tick = job.tick();
        let mut finished = false;
        for line in &job_tick.fresh {
            match parse_sse_line(line) {
                SseEvent::Delta(text) => {
                    if self.first_delta_at.is_none() {
                        self.first_delta_at = Some(Instant::now());
                    }
                    self.stream_chars += text.chars().count();
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
            self.finalize_stats();
            self.job = None;
            return true;
        }
        // Live stream: always dirty so the title's elapsed/tok-s counter moves
        // between SSE chunks (even when this tick carried no new content).
        let _ = job_tick.material;
        true
    }

    /// Stamp the just-finished assistant reply with its measured stats.
    pub(crate) fn finalize_stats(&mut self) {
        let Some(send_at) = self.send_at.take() else {
            return;
        };
        if self.stream_chars == 0 {
            return;
        }
        let stats = ReplyStats {
            ttft: self
                .first_delta_at
                .take()
                .map(|at| at.saturating_duration_since(send_at))
                .unwrap_or_default(),
            elapsed: send_at.elapsed(),
            est_tokens: self.stream_chars.div_ceil(4),
        };
        if let Some(last) = self.messages.last_mut()
            && !last.from_user
        {
            last.stats = Some(stats);
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
        // Server job present but exited (crashed or stopped mid-session):
        // the transcript stays on screen in read-only mode below.
        let server_stopped =
            !self.server_ready && self.server.as_ref().is_some_and(|job| job.done.is_some());
        // Info cards (no server job yet, or server still starting): a plain
        // info page, not an input.  Global keys (1-5, q, ?) were already
        // handled; back keys go Home.
        if !self.server_ready && !server_stopped {
            if matches!(key.code, KeyCode::Up | KeyCode::Char('k')) {
                self.focus_tab_bar();
                return;
            }
            if matches!(
                key.code,
                KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') | KeyCode::Backspace
            ) {
                self.back_or_home();
            }
            return;
        }
        // Esc first: stop a live stream, otherwise leave the screen.
        if key.code == KeyCode::Esc {
            if self.chat.streaming() {
                self.chat.cancel();
                self.toast_warn("reply cancelled");
            } else {
                self.back_or_home();
            }
            return;
        }
        if server_stopped {
            // Read-only: transcript scroll/copy/clear keep working; composer
            // edits and retry (which would drop the last answer) stay off.
            let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
            match key.code {
                KeyCode::PageUp => self.scroll_transcript(true, true),
                KeyCode::PageDown => self.scroll_transcript(false, true),
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.chat.scroll.borrow().offset().y == 0 {
                        self.focus_tab_bar();
                    } else {
                        self.scroll_transcript(true, false);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => self.scroll_transcript(false, false),
                KeyCode::Char('y') if ctrl => self.copy_last_reply(),
                KeyCode::Char('l') if ctrl => self.modal = Some(Modal::ClearChat),
                KeyCode::Char('t') if ctrl => self.toggle_thinking(),
                KeyCode::Enter => self.toast_error("server not running — start one on Serve (4)"),
                _ => {}
            }
            return;
        }
        if key.code == KeyCode::Char('u') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.chat.input.clear();
            self.chat.cursor = 0;
            return;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('l') => {
                    self.modal = Some(Modal::ClearChat);
                    return;
                }
                KeyCode::Char('y') => {
                    self.copy_last_reply();
                    return;
                }
                KeyCode::Char('r') => {
                    self.retry_last();
                    return;
                }
                KeyCode::Char('t') => {
                    self.toggle_thinking();
                    return;
                }
                _ => {}
            }
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
                self.chat.hist_nav = None;
                if self.chat.cursor > 0 {
                    self.chat.cursor -= 1;
                    let byte_idx = char_to_byte_idx(&self.chat.input, self.chat.cursor);
                    self.chat.input.remove(byte_idx);
                }
            }
            KeyCode::Delete => {
                self.chat.hist_nav = None;
                let len = self.chat.input.chars().count();
                if self.chat.cursor < len {
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
            // Single-line draft: ↑/↓ walk prompt history (draft stashed and
            // restored), matching readline muscle memory. Multi-line drafts
            // keep vertical cursor motion. Transcript scroll = PgUp/PgDn/wheel.
            KeyCode::Up => {
                if self.chat.input.contains('\n') && self.chat.hist_nav.is_none() {
                    // Top of multi-line draft → tab bar.
                    if !self.move_chat_cursor_vert(-1) {
                        self.focus_tab_bar();
                    }
                } else {
                    self.history_recall(-1);
                }
            }
            KeyCode::Down => {
                if self.chat.input.contains('\n') && self.chat.hist_nav.is_none() {
                    if !self.move_chat_cursor_vert(1) {
                        self.scroll_transcript(false, false);
                    }
                } else {
                    self.history_recall(1);
                }
            }
            _ => {}
        }
    }

    fn insert_chat_char(&mut self, c: char) {
        // Editing detaches from history navigation (readline-style).
        self.chat.hist_nav = None;
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

    /// Bracketed paste into the composer: normalize line endings, insert at
    /// the cursor as one edit (no per-character key storm).
    pub(crate) fn paste_chat_text(&mut self, text: &str) {
        let clean = text.replace("\r\n", "\n").replace('\r', "\n");
        if clean.is_empty() {
            return;
        }
        self.chat.hist_nav = None;
        let byte_idx = char_to_byte_idx(&self.chat.input, self.chat.cursor);
        self.chat.input.insert_str(byte_idx, &clean);
        self.chat.cursor += clean.chars().count();
    }

    /// Readline-style prompt history on ↑/↓ (single-line drafts only; the
    /// in-progress draft is stashed and restored on the way back down).
    /// With no history yet, ↑/↓ keep the scroll / tab-bar-promote behavior.
    fn history_recall(&mut self, dir: i32) {
        if self.chat.history.is_empty() {
            if dir < 0 {
                let at_top = self.chat.scroll.borrow().offset().y == 0;
                if at_top {
                    self.focus_tab_bar();
                } else {
                    self.scroll_transcript(true, false);
                }
            } else {
                self.scroll_transcript(false, false);
            }
            return;
        }
        match self.chat.hist_nav {
            None => {
                if dir >= 0 {
                    self.scroll_transcript(false, false);
                    return;
                }
                let idx = self.chat.history.len() - 1;
                let draft = std::mem::take(&mut self.chat.input);
                self.chat.hist_nav = Some((idx, draft));
                self.chat.input = self.chat.history[idx].clone();
            }
            Some((idx, _)) if dir < 0 => {
                if idx == 0 {
                    // ↑ past the oldest entry: restore the draft, promote to tabs.
                    let draft = self
                        .chat
                        .hist_nav
                        .take()
                        .map(|(_, draft)| draft)
                        .unwrap_or_default();
                    self.chat.input = draft;
                    self.chat.cursor = self.chat.input.chars().count();
                    self.focus_tab_bar();
                    return;
                }
                let idx = idx - 1;
                if let Some(nav) = self.chat.hist_nav.as_mut() {
                    nav.0 = idx;
                }
                self.chat.input = self.chat.history[idx].clone();
            }
            Some((idx, _)) => {
                if idx + 1 < self.chat.history.len() {
                    let idx = idx + 1;
                    if let Some(nav) = self.chat.hist_nav.as_mut() {
                        nav.0 = idx;
                    }
                    self.chat.input = self.chat.history[idx].clone();
                } else {
                    // ↓ past the newest entry: back to the stashed draft.
                    let draft = self
                        .chat
                        .hist_nav
                        .take()
                        .map(|(_, draft)| draft)
                        .unwrap_or_default();
                    self.chat.input = draft;
                }
            }
        }
        self.chat.cursor = self.chat.input.chars().count();
    }

    /// Ctrl+L / `/clear` (confirmed via `Modal::ClearChat`): wipe the
    /// transcript and any stuck error.
    pub(crate) fn clear_chat(&mut self) {
        self.chat.messages.clear();
        self.chat.error = None;
        self.toast("chat cleared");
    }

    /// Ctrl+T: expand/collapse every thinking block in the transcript.
    fn toggle_thinking(&mut self) {
        self.chat.thinking_expanded = !self.chat.thinking_expanded;
    }

    /// Ctrl+Y / `/copy`: copy the last assistant reply's answer (thinking
    /// block excluded) to the clipboard.
    fn copy_last_reply(&mut self) {
        let reply = self
            .chat
            .messages
            .iter()
            .rev()
            .find(|m| !m.from_user && !m.content.is_empty())
            .map(|m| split_thinking(&m.content).1.to_string());
        match reply {
            None => self.toast_warn("no reply to copy yet"),
            Some(text) if widgets::copy_to_clipboard(&text) => {
                self.toast_success("reply copied");
            }
            Some(_) => self.toast_error("copy failed (pbcopy unavailable)"),
        }
    }

    /// Ctrl+R / `/retry`: drop the last assistant answer and regenerate it
    /// from the same transcript prefix.
    fn retry_last(&mut self) {
        if self.chat.streaming() {
            return;
        }
        let Some(last_user) = self.chat.messages.iter().rposition(|m| m.from_user) else {
            self.toast_warn("nothing to retry yet");
            return;
        };
        self.chat.messages.truncate(last_user + 1);
        self.chat.error = None;
        self.stream_reply();
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
        if self.chat.streaming() {
            self.toast("reply in progress — Esc cancels");
            return;
        }
        if self.chat.input.trim().is_empty() {
            return;
        }
        if !self.server_ready || self.server_url.is_none() {
            self.chat.error = Some("start a server first (press 4 Serve)".into());
            return;
        }
        let prompt = std::mem::take(&mut self.chat.input);
        self.chat.cursor = 0;
        if prompt.trim_start().starts_with('/') {
            self.run_slash_command(prompt.trim());
            return;
        }
        if self.chat.history.last() != Some(&prompt) {
            self.chat.history.push(prompt.clone());
            if self.chat.history.len() > HISTORY_CAP {
                self.chat.history.remove(0);
            }
        }
        self.chat.hist_nav = None;
        self.chat.error = None;
        self.chat.autoscroll = true;
        self.chat.messages.push(ChatMessage {
            from_user: true,
            content: prompt,
            stats: None,
        });
        self.stream_reply();
    }

    /// `/clear` `/copy` `/retry` `/help` — local commands, never sent to the model.
    fn run_slash_command(&mut self, cmd: &str) {
        match cmd {
            "/clear" => self.modal = Some(Modal::ClearChat),
            "/copy" => self.copy_last_reply(),
            "/retry" => self.retry_last(),
            "/help" | "/?" => self.show_help = true,
            other => {
                self.chat.error = Some(format!(
                    "unknown command {other} — try /clear /copy /retry /help"
                ));
            }
        }
    }

    /// Spawn the streaming request for the current transcript and push the
    /// empty assistant placeholder.  Shared by send and retry.
    fn stream_reply(&mut self) {
        let Some(url) = self.server_url.clone().filter(|_| self.server_ready) else {
            self.chat.error = Some("start a server first (press 4 Serve)".into());
            return;
        };
        // No "model" field: the server 400s on any name that differs from its
        // configured model_id and accepts requests that omit it entirely.
        // Assistant history sends the answer only — thinking blocks are local
        // display artifacts, not prompt material.
        let messages: Vec<serde_json::Value> = self
            .chat
            .messages
            .iter()
            .map(|m| {
                let content = if m.from_user {
                    m.content.clone()
                } else {
                    split_thinking(&m.content).1.to_string()
                };
                serde_json::json!({
                    "role": if m.from_user { "user" } else { "assistant" },
                    "content": content,
                })
            })
            .collect();
        let body = serde_json::json!({
            "messages": messages,
            "stream": true,
        });
        self.chat.messages.push(ChatMessage {
            from_user: false,
            content: String::new(),
            stats: None,
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
        self.chat.send_at = Some(Instant::now());
        self.chat.first_delta_at = None;
        self.chat.stream_chars = 0;
    }

    pub(crate) fn draw_chat(&self, frame: &mut Frame, area: Rect) {
        // Server job present but exited: keep the normal layout below so the
        // transcript stays readable, with an inline banner explaining sends
        // are off.  No job / still starting get a centered info card.
        let server_stopped =
            !self.server_ready && self.server.as_ref().is_some_and(|job| job.done.is_some());
        if !self.server_ready && !server_stopped {
            let starting = self.managed_server_alive();
            // Centered info card (warn accent while a managed child binds).
            // When idle we still probe the Serve host/port for an external
            // `ax-engine-server` started in another terminal — show that target.
            let probe_url = self
                .configured_server_url()
                .unwrap_or_else(|| "http://127.0.0.1:31418".into());
            let card_w = 58u16.min(area.width.saturating_sub(4));
            let card_h = 9u16.min(area.height.saturating_sub(2));
            let card = widgets::centered_rect(card_w, card_h, area);
            let (heading, detail, heading_style) = if starting {
                (
                    "Server starting…".to_string(),
                    "  The model is loading; this can take a minute.".to_string(),
                    Style::default()
                        .fg(theme::colors().warn)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                (
                    "Looking for server…".to_string(),
                    format!("  Probing {probe_url}/health (external servers OK)"),
                    Style::default()
                        .fg(theme::colors().text)
                        .add_modifier(Modifier::BOLD),
                )
            };
            let lines = vec![
                Line::raw(""),
                Line::from(vec![Span::raw("  "), Span::styled(heading, heading_style)]),
                Line::raw(""),
                Line::from(Span::styled(detail, theme::label())),
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  Press "),
                    Span::styled(
                        "4",
                        Style::default()
                            .fg(theme::colors().accent)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" to change host/port or start one here", theme::label()),
                ]),
            ];
            frame.render_widget(
                Paragraph::new(lines).block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(if starting {
                            theme::colors().warn
                        } else {
                            theme::colors().border_inactive
                        }))
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
        if server_stopped {
            let chunks = Layout::vertical([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(input_height),
            ])
            .split(area);
            widgets::draw_banner(
                frame,
                chunks[0],
                widgets::ToastLevel::Warning,
                "server stopped — press 4 for Serve",
            );
            self.draw_chat_transcript(frame, chunks[1]);
            self.draw_chat_input(frame, chunks[2]);
            return;
        }
        let chunks =
            Layout::vertical([Constraint::Min(3), Constraint::Length(input_height)]).split(area);
        self.draw_chat_transcript(frame, chunks[0]);
        self.draw_chat_input(frame, chunks[1]);
    }

    /// Hash of every input the transcript lines depend on. The streaming tail
    /// is the only message whose content mutates in place, so older messages
    /// hit the cache while a reply streams in.
    fn transcript_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let messages = &self.chat.messages;
        messages.len().hash(&mut h);
        for message in messages {
            message.from_user.hash(&mut h);
            message.content.len().hash(&mut h);
            message.stats.is_some().hash(&mut h);
        }
        if let Some(last) = messages.last() {
            last.content.hash(&mut h);
        }
        self.chat.streaming().hash(&mut h);
        self.chat.error.hash(&mut h);
        self.chat.thinking_expanded.hash(&mut h);
        self.server_model.hash(&mut h);
        h.finish()
    }

    fn build_transcript_lines(&self) -> Vec<Line<'static>> {
        let mut lines: Vec<Line> = Vec::new();
        let last_idx = self.chat.messages.len().saturating_sub(1);
        for (idx, message) in self.chat.messages.iter().enumerate() {
            if message.from_user {
                let mut spans = vec![Span::styled(
                    "You  ",
                    Style::default()
                        .fg(theme::colors().accent)
                        .add_modifier(Modifier::BOLD),
                )];
                let mut parts = message.content.lines();
                spans.push(Span::styled(
                    parts.next().unwrap_or("").to_string(),
                    theme::body(),
                ));
                lines.push(Line::from(spans));
                for part in parts {
                    lines.push(Line::from(Span::styled(
                        format!("     {part}"),
                        theme::body(),
                    )));
                }
            } else {
                let model_name = self.server_model.as_deref().unwrap_or("Model");
                let short = widgets::ellipsis(model_name, 14);
                lines.push(Line::from(Span::styled(
                    short,
                    Style::default()
                        .fg(theme::colors().ok)
                        .add_modifier(Modifier::BOLD),
                )));
                let (thinking, answer) = split_thinking(&message.content);
                if let Some(think) = thinking {
                    let dim = Style::default()
                        .fg(theme::colors().muted)
                        .add_modifier(Modifier::ITALIC);
                    // Still-streaming thinking renders in full so the
                    // reasoning is visible as it arrives; settled blocks
                    // collapse to a short preview unless Ctrl+T expanded them.
                    let streaming_this = self.chat.streaming() && idx == last_idx;
                    let expanded = self.chat.thinking_expanded || streaming_this;
                    let think_lines: Vec<&str> = think.lines().collect();
                    let marker = if expanded {
                        "▾ Thinking"
                    } else {
                        "▸ Thinking"
                    };
                    lines.push(Line::from(Span::styled(marker, dim)));
                    if expanded || think_lines.len() <= THINKING_PREVIEW_LINES {
                        for part in think_lines {
                            lines.push(Line::from(Span::styled(part.to_string(), dim)));
                        }
                    } else {
                        for part in think_lines.iter().take(THINKING_PREVIEW_LINES) {
                            lines.push(Line::from(Span::styled(part.to_string(), dim)));
                        }
                        let more = think_lines.len() - THINKING_PREVIEW_LINES;
                        lines.push(Line::from(Span::styled(
                            format!("… {more} more thinking lines (Ctrl+T to expand)"),
                            dim,
                        )));
                    }
                }
                if answer.is_empty() {
                    // Placeholder while the (empty) reply is streaming in.
                    if self.chat.streaming() && idx == last_idx {
                        lines.push(Line::from(Span::styled("…", theme::body_dim())));
                    }
                } else if self.chat.streaming() && idx == last_idx {
                    // Plain text while tokens stream: re-parsing growing
                    // markdown every frame was the chat hot path. Full
                    // markdown runs once when the reply finishes.
                    for part in answer.lines() {
                        lines.push(Line::from(Span::styled(part.to_string(), theme::body())));
                    }
                } else {
                    lines.extend(render_markdown(answer));
                }
                if let Some(stats) = message.stats {
                    lines.push(Line::from(Span::styled(stats.summary(), theme::label())));
                }
            }
            lines.push(Line::raw(""));
        }
        if let Some(error) = &self.chat.error {
            lines.push(Line::from(vec![
                Span::styled(format!("{} ", theme::icon::error()), theme::danger()),
                Span::styled(error.clone(), theme::danger()),
            ]));
        }
        if lines.is_empty() {
            // Fresh session: one dim centered cue instead of a blank pane.
            lines.push(
                Line::from(Span::styled(
                    "Ask your model anything — / for commands",
                    theme::label(),
                ))
                .centered(),
            );
        }
        lines
    }

    fn draw_chat_transcript(&self, frame: &mut Frame, area: Rect) {
        let title = if self.chat.streaming() {
            let est = self.chat.stream_chars.div_ceil(4);
            let secs = self
                .chat
                .send_at
                .map(|at| at.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            let rate = if secs > 0.05 { est as f64 / secs } else { 0.0 };
            format!(" Chat — replying… ~{est} tok · ~{rate:.0} tok/s (Esc cancels) ")
        } else {
            " Chat ".to_string()
        };
        let block = if self.chat.streaming() {
            widgets::active_block(&title).border_style(Style::default().fg(theme::colors().warn))
        } else {
            widgets::soft_block(&title)
        };
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let text_width = inner.width.saturating_sub(1).max(1);
        let key = self.transcript_key();
        let (lines, content_height) = {
            let mut cache = self.chat.transcript_cache.borrow_mut();
            match cache.as_ref() {
                Some(c) if c.key == key && c.width == text_width => {
                    (Rc::clone(&c.lines), c.content_height)
                }
                Some(c) if c.key == key => {
                    // Same content, new wrap width — remeasure without rebuild.
                    let paragraph =
                        Paragraph::new(Text::from(c.lines.as_slice())).wrap(Wrap { trim: false });
                    let height = (paragraph.line_count(text_width) as u16).max(1);
                    let lines = Rc::clone(&c.lines);
                    *cache = Some(TranscriptCache {
                        key,
                        width: text_width,
                        lines: Rc::clone(&lines),
                        content_height: height,
                    });
                    (lines, height)
                }
                _ => {
                    let built = Rc::new(self.build_transcript_lines());
                    let paragraph =
                        Paragraph::new(Text::from(built.as_slice())).wrap(Wrap { trim: false });
                    let height = (paragraph.line_count(text_width) as u16).max(1);
                    *cache = Some(TranscriptCache {
                        key,
                        width: text_width,
                        lines: Rc::clone(&built),
                        content_height: height,
                    });
                    (built, height)
                }
            }
        };
        let paragraph = Paragraph::new(Text::from(lines.as_slice())).wrap(Wrap { trim: false });
        let mut scroll_view = ScrollView::new(Size::new(text_width, content_height));
        scroll_view.render_widget(paragraph, Rect::new(0, 0, text_width, content_height));
        let mut state = self.chat.scroll.borrow_mut();
        if self.chat.autoscroll {
            state.scroll_to_bottom();
        }
        frame.render_stateful_widget(scroll_view, inner, &mut state);
    }

    fn draw_chat_input(&self, frame: &mut Frame, area: Rect) {
        let prompt = format!("{} ", theme::icon::select());
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
        // The box caps at ~6 visual rows while drafts can grow past that:
        // scroll a window over the rows so the cursor's row is always inside.
        let cursor_row = visual
            .iter()
            .position(|row| row.iter().any(|&(_, is_cursor)| is_cursor))
            .unwrap_or(0);
        let visible = area.height.saturating_sub(2).max(1) as usize;
        let offset = clamp_composer_offset(
            self.chat.input_scroll.get(),
            cursor_row,
            visible,
            visual.len(),
        );
        self.chat.input_scroll.set(offset);
        let mut lines: Vec<Line> = visual
            .into_iter()
            .skip(offset)
            .take(visible)
            .map(|row| {
                let mut spans: Vec<Span<'static>> = Vec::new();
                let mut run = String::new();
                let mut run_is_cursor = false;
                for (ch, is_cursor) in row {
                    if is_cursor != run_is_cursor && !run.is_empty() {
                        let style = if run_is_cursor {
                            Style::default()
                                .bg(theme::colors().select)
                                .fg(theme::colors().on_select)
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
                        Style::default()
                            .bg(theme::colors().select)
                            .fg(theme::colors().on_select)
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
/// Columns are display widths, so CJK/emoji don't drift the height estimate.
pub(crate) fn count_visual_lines(text: &str, cols: usize) -> usize {
    let cols = cols.max(1);
    let mut lines = 1usize;
    // Leading prompt is ~2 cols.
    let mut col = 2usize;
    for c in text.chars() {
        if c == '\n' {
            lines += 1;
            col = 0;
        } else {
            let w = UnicodeWidthChar::width(c).unwrap_or(0);
            if col + w > cols {
                lines += 1;
                col = 0;
            }
            col += w;
        }
    }
    lines
}

/// Clamp the composer window start so `cursor_row` stays visible inside a
/// window of `visible` rows over `total` visual rows.  Shrinking drafts pull
/// the offset back to 0 on their own.
pub(crate) fn clamp_composer_offset(
    offset: usize,
    cursor_row: usize,
    visible: usize,
    total: usize,
) -> usize {
    let visible = visible.max(1);
    let mut offset = offset.min(total.saturating_sub(visible));
    if cursor_row < offset {
        offset = cursor_row;
    } else if cursor_row >= offset + visible {
        offset = cursor_row + 1 - visible;
    }
    offset
}

/// Lay out cells into visual rows, wrapping at `cols` (display width) and
/// breaking on `\n`.
fn layout_cells(cells: &[(char, bool)], cols: usize) -> Vec<Vec<(char, bool)>> {
    let cols = cols.max(1);
    let mut rows: Vec<Vec<(char, bool)>> = vec![Vec::new()];
    let mut col = 0usize;
    for &(ch, is_cursor) in cells {
        if ch == '\n' {
            if let Some(row) = rows.last_mut() {
                row.push(('\n', is_cursor));
            }
            rows.push(Vec::new());
            col = 0;
            continue;
        }
        let w = UnicodeWidthChar::width(ch).unwrap_or(0);
        if col > 0 && col + w > cols {
            rows.push(Vec::new());
            col = 0;
        }
        if let Some(row) = rows.last_mut() {
            row.push((ch, is_cursor));
        }
        col += w;
    }
    rows
}

/// Split assistant content into `(thinking, answer)`.
///
/// The server pre-fills `<think>` into Qwen prompts, so the stream often
/// carries only the closing tag; a leading `<think>` is stripped when the
/// model emits it.  With an opening tag but no close yet (mid-stream),
/// everything counts as thinking.  Render-time split keeps the streaming
/// append path trivial and works for retries/history equally.
pub(crate) fn split_thinking(content: &str) -> (Option<&str>, &str) {
    let body = content.strip_prefix("<think>").unwrap_or(content);
    match body.split_once("</think>") {
        Some((think, answer)) => {
            let think = think.trim();
            let answer = answer.trim_start();
            (if think.is_empty() { None } else { Some(think) }, answer)
        }
        None if content.starts_with("<think>") => {
            let think = body.trim();
            (if think.is_empty() { None } else { Some(think) }, "")
        }
        None => (None, content),
    }
}
