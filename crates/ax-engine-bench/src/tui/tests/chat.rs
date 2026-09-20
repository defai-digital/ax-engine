//! Chat: transcript rendering, input and history, slash commands, paste,
//! streaming stats, thinking blocks, and composer width.
use super::super::jobs::Job;
use super::super::screens::chat::{
    ChatMessage, ReplyStats, SseEvent, count_visual_lines, parse_sse_line, split_thinking,
};
use super::super::{Modal, Screen, WizardStage};
use super::{chat_ready_app, ctrl_key, key, new_app, render, type_text};
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

#[test]
fn sse_lines_parse() {
    assert_eq!(
        parse_sse_line(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#),
        SseEvent::Delta("Hi".into())
    );
    assert_eq!(parse_sse_line("data: [DONE]"), SseEvent::Done);
    assert_eq!(parse_sse_line(""), SseEvent::Ignored);
    assert_eq!(parse_sse_line(": keepalive"), SseEvent::Ignored);
    assert_eq!(
        parse_sse_line(r#"{"error":{"message":"boom"}}"#),
        SseEvent::Error("boom".into())
    );
    assert_eq!(
        parse_sse_line(r#"data: {"error":{"message":"bad request"}}"#),
        SseEvent::Error("bad request".into())
    );
    assert_eq!(
        parse_sse_line(r#"data: {"choices":[{"delta":{}}]}"#),
        SseEvent::Ignored
    );
}

#[test]
fn chat_without_server_shows_hint() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    let text = render(&app);
    assert!(
        text.contains("Looking for server") || text.contains("/health"),
        "chat should show external probe target, got: {text}"
    );
    // The hint screen is not an input: typing and Enter do nothing.
    for c in "hello".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    app.on_key(key(KeyCode::Enter));
    assert!(app.chat.messages.is_empty());
    assert!(app.chat.input.is_empty());
}

#[test]
fn chat_input_edits_at_cursor_and_q_is_text() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.server_ready = true; // input capture only exists with a live server
    for c in "aq".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    assert!(!app.quit, "q in chat is text, not quit");
    assert_eq!(app.chat.input, "aq");
    app.on_key(key(KeyCode::Left));
    app.on_key(key(KeyCode::Char('x')));
    assert_eq!(app.chat.input, "axq");
    app.on_key(key(KeyCode::Backspace));
    assert_eq!(app.chat.input, "aq");
    let mut clear = key(KeyCode::Char('u'));
    clear.modifiers = KeyModifiers::CONTROL;
    app.on_key(clear);
    assert!(app.chat.input.is_empty());
}

#[test]
fn chat_ctrl_j_inserts_newline_and_enter_sends_structure() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:8080".into());
    for c in "hi".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    let mut nl = key(KeyCode::Char('j'));
    nl.modifiers = KeyModifiers::CONTROL;
    app.on_key(nl);
    for c in "there".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    assert_eq!(app.chat.input, "hi\nthere");
    // Vertical motion stays in the draft.
    app.on_key(key(KeyCode::Up));
    assert!(app.chat.cursor <= 2, "up moves onto first line");
    app.on_key(key(KeyCode::Home));
    assert_eq!(app.chat.cursor, 0);
    // Shift+Enter also inserts a newline.
    let mut shift_enter = key(KeyCode::Enter);
    shift_enter.modifiers = KeyModifiers::SHIFT;
    app.on_key(shift_enter);
    assert!(
        app.chat.input.starts_with('\n')
            || app.chat.input.contains("\n\n")
            || app.chat.input.matches('\n').count() >= 2
    );
}

#[test]
fn chat_ctrl_digit_switches_screens_while_typing() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.server_ready = true;
    app.on_key(key(KeyCode::Char('h')));
    assert_eq!(app.chat.input, "h");
    // Bare digit is text while chat is ready.
    app.on_key(key(KeyCode::Char('2')));
    assert_eq!(app.chat.input, "h2");
    assert_eq!(app.screen, Screen::Chat);
    // Ctrl+2 always jumps to Models.
    let mut jump = key(KeyCode::Char('2'));
    jump.modifiers = KeyModifiers::CONTROL;
    app.on_key(jump);
    assert_eq!(app.screen, Screen::Models);
}

#[test]
fn recommended_star_matches_quick_start_target() {
    let mut app = new_app();
    let (fi, vi) = app.quick_start_target().expect("catalog is not empty");
    assert!(app.is_recommended_variant(fi, vi));
    app.screen = Screen::Models;
    app.family_idx = fi;
    app.stage = WizardStage::Precision;
    let text = render(&app);
    assert!(
        text.contains('★') || text.contains("rec") || text.contains("recommended"),
        "Quick start variant should show the recommended star"
    );
}

#[test]
fn chat_esc_leaves_screen_when_not_streaming() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn chat_scroll_detaches_and_reattaches_follow_mode() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.screen = Screen::Chat;
    for i in 0..80 {
        app.chat
            .messages
            .push(super::super::screens::chat::ChatMessage {
                from_user: i % 2 == 0,
                content: format!("message {i}"),
                stats: None,
            });
    }
    assert!(app.chat.autoscroll, "follows new tokens by default");
    let _ = render(&app); // records page size + jumps to bottom
    app.on_key(key(KeyCode::PageUp));
    assert!(!app.chat.autoscroll, "scrolling up detaches");
    // Scrolling down until the offset stops moving re-attaches.
    for _ in 0..200 {
        app.on_key(key(KeyCode::Down));
    }
    assert!(app.chat.autoscroll, "hitting the bottom re-attaches");
}

#[test]
fn chat_transcript_renders_when_server_ready() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.server_model = Some("gemma4-e2b".into());
    app.screen = Screen::Chat;
    app.chat
        .messages
        .push(super::super::screens::chat::ChatMessage {
            from_user: true,
            content: "What is AX?".into(),
            stats: None,
        });
    app.chat
        .messages
        .push(super::super::screens::chat::ChatMessage {
            from_user: false,
            content: "A local inference engine.".into(),
            stats: None,
        });
    let text = render(&app);
    assert!(text.contains("You"));
    assert!(text.contains("What is AX?"));
    assert!(text.contains("gemma4-e2b"));
    assert!(text.contains("A local inference engine."));
    assert!(text.contains("Message"), "composer block should render");
    assert!(
        text.contains('›') || text.contains(">"),
        "input prompt should render"
    );
}

// ---------------------------------------------------------------------------
// Chat P0: history, slash commands, paste, stats, thinking, composer width
// ---------------------------------------------------------------------------

#[test]
fn chat_send_pushes_history_and_dedupes_last() {
    let mut app = chat_ready_app();
    type_text(&mut app, "hello");
    app.on_key(key(KeyCode::Enter));
    assert_eq!(app.chat.history, vec!["hello".to_string()]);
    assert!(app.chat.input.is_empty());
    assert!(app.chat.messages[0].from_user);
    type_text(&mut app, "hello");
    app.on_key(key(KeyCode::Enter));
    assert_eq!(
        app.chat.history.len(),
        1,
        "consecutive dupes are not stored"
    );
}

#[test]
fn chat_history_recall_restores_draft_on_the_way_down() {
    let mut app = chat_ready_app();
    app.chat.history = vec!["first".to_string(), "second".to_string()];
    type_text(&mut app, "draft");
    app.on_key(key(KeyCode::Up));
    assert_eq!(app.chat.input, "second");
    app.on_key(key(KeyCode::Up));
    assert_eq!(app.chat.input, "first");
    app.on_key(key(KeyCode::Down));
    assert_eq!(app.chat.input, "second");
    app.on_key(key(KeyCode::Down));
    assert_eq!(
        app.chat.input, "draft",
        "down past newest restores the draft"
    );
    assert!(app.chat.hist_nav.is_none());
}

#[test]
fn chat_history_up_past_oldest_restores_draft_and_promotes_tabs() {
    let mut app = chat_ready_app();
    app.chat.history = vec!["only".to_string()];
    type_text(&mut app, "draft");
    app.on_key(key(KeyCode::Up));
    assert_eq!(app.chat.input, "only");
    app.on_key(key(KeyCode::Up));
    assert!(
        app.focus_tabs,
        "up past the oldest entry promotes to the tab bar"
    );
    assert_eq!(app.chat.input, "draft");
    assert!(app.chat.hist_nav.is_none());
}

#[test]
fn chat_up_without_history_keeps_scroll_and_tab_promotion() {
    let mut app = chat_ready_app();
    app.on_key(key(KeyCode::Up));
    assert!(
        app.focus_tabs,
        "empty history + empty input: up reaches tab bar"
    );
}

#[test]
fn chat_history_is_capped() {
    let mut app = chat_ready_app();
    app.chat.history = (0..100).map(|i| format!("prompt {i}")).collect();
    type_text(&mut app, "newest");
    app.on_key(key(KeyCode::Enter));
    assert_eq!(app.chat.history.len(), 100);
    assert_eq!(app.chat.history.last().map(String::as_str), Some("newest"));
    assert_eq!(
        app.chat.history.first().map(String::as_str),
        Some("prompt 1")
    );
}

#[test]
fn chat_slash_clear_empties_transcript() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: true,
        content: "hi".into(),
        stats: None,
    });
    type_text(&mut app, "/clear");
    app.on_key(key(KeyCode::Enter));
    // Clearing asks for confirmation first; Enter confirms.
    assert!(matches!(app.modal, Some(Modal::ClearChat)));
    assert!(!app.chat.messages.is_empty());
    app.on_key(key(KeyCode::Enter));
    assert!(app.chat.messages.is_empty());
    assert!(app.chat.input.is_empty());
}

#[test]
fn chat_unknown_slash_command_reports_error() {
    let mut app = chat_ready_app();
    type_text(&mut app, "/bogus");
    app.on_key(key(KeyCode::Enter));
    let err = app.chat.error.unwrap_or_default();
    assert!(err.contains("unknown command"), "got: {err}");
}

#[test]
fn chat_ctrl_l_clears_transcript() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: true,
        content: "hi".into(),
        stats: None,
    });
    app.on_key(ctrl_key('l'));
    // Clearing asks for confirmation first; Enter confirms.
    assert!(matches!(app.modal, Some(Modal::ClearChat)));
    assert!(!app.chat.messages.is_empty());
    app.on_key(key(KeyCode::Enter));
    assert!(app.chat.messages.is_empty());
}

#[test]
fn chat_clear_modal_confirm_or_dismiss() {
    // y confirms like Enter and toasts.
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: true,
        content: "hi".into(),
        stats: None,
    });
    app.on_key(ctrl_key('l'));
    app.on_key(key(KeyCode::Char('y')));
    assert!(app.chat.messages.is_empty(), "y confirms the clear");
    assert!(
        app.toasts.iter().any(|t| t.text.contains("chat cleared")),
        "clearing raises a toast"
    );

    // Esc / n / h / Left all dismiss and keep the transcript.
    for dismiss in [
        key(KeyCode::Esc),
        key(KeyCode::Char('n')),
        key(KeyCode::Char('h')),
        key(KeyCode::Left),
    ] {
        let mut app = chat_ready_app();
        app.chat.messages.push(ChatMessage {
            from_user: true,
            content: "hi".into(),
            stats: None,
        });
        app.on_key(ctrl_key('l'));
        assert!(matches!(app.modal, Some(Modal::ClearChat)));
        app.on_key(dismiss);
        assert!(app.modal.is_none(), "dismiss closes the modal");
        assert_eq!(app.chat.messages.len(), 1, "transcript survives dismiss");
    }
}

#[test]
fn chat_ctrl_r_retry_truncates_answer_and_respawns() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: true,
        content: "hi".into(),
        stats: None,
    });
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "hello".into(),
        stats: None,
    });
    app.on_key(ctrl_key('r'));
    assert_eq!(app.chat.messages.len(), 2, "user turn + fresh placeholder");
    assert!(app.chat.messages[0].from_user);
    assert!(!app.chat.messages[1].from_user);
    assert!(app.chat.messages[1].content.is_empty());
    assert!(app.chat.job.is_some(), "retry should spawn a new stream");
    app.chat.cancel();
}

#[test]
fn chat_paste_inserts_at_cursor_and_normalizes_crlf() {
    let mut app = chat_ready_app();
    type_text(&mut app, "ab");
    app.on_key(key(KeyCode::Left));
    app.on_paste("X\r\nY");
    assert_eq!(app.chat.input, "aX\nYb");
}

#[test]
fn chat_paste_ignored_without_server() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.on_paste("hello");
    assert!(app.chat.input.is_empty());
}

#[test]
fn split_thinking_handles_all_stream_shapes() {
    // Prompt-prefilled open tag: only the close tag arrives in the stream.
    let (think, answer) = split_thinking("let me reason</think>\n\nThe answer.");
    assert_eq!(think, Some("let me reason"));
    assert_eq!(answer, "The answer.");
    // Full tags emitted in content.
    let (think, answer) = split_thinking("<think>deep</think>result");
    assert_eq!(think, Some("deep"));
    assert_eq!(answer, "result");
    // Mid-stream thinking (open tag, no close yet).
    let (think, answer) = split_thinking("<think>still going");
    assert_eq!(think, Some("still going"));
    assert_eq!(answer, "");
    // Plain reply: untouched.
    let (think, answer) = split_thinking("just text");
    assert_eq!(think, None);
    assert_eq!(answer, "just text");
    // Empty thinking block (reasoning skipped) is suppressed.
    let (think, answer) = split_thinking("<think>\n\n</think>\n\nplain");
    assert_eq!(think, None);
    assert_eq!(answer, "plain");
}

#[test]
fn chat_transcript_renders_thinking_dimmed_and_answer() {
    let mut app = chat_ready_app();
    app.server_model = Some("qwen3-4b".into());
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "pondering</think>\n\n**Answer** here.".into(),
        stats: None,
    });
    let text = render(&app);
    assert!(text.contains("Thinking"));
    assert!(text.contains("pondering"));
    assert!(text.contains("Answer"));
    assert!(text.contains("here."));
}

#[test]
fn finalize_stats_estimates_tokens_and_ttft() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "x".repeat(400),
        stats: None,
    });
    app.chat.send_at = Some(std::time::Instant::now());
    app.chat.first_delta_at = Some(std::time::Instant::now());
    app.chat.stream_chars = 400;
    app.chat.finalize_stats();
    let stats = app.chat.messages[0].stats;
    assert!(stats.is_some(), "stats recorded");
    let Some(stats) = stats else {
        return;
    };
    assert_eq!(stats.est_tokens, 100, "chars/4 estimate");
    assert!(stats.elapsed < Duration::from_secs(5));
}

#[test]
fn reply_stats_summary_format() {
    let stats = ReplyStats {
        ttft: Duration::from_millis(800),
        elapsed: Duration::from_millis(12_300),
        est_tokens: 412,
    };
    assert_eq!(stats.summary(), "0.8s TTFT · 12.3s · ~412 tok · ~33 tok/s");
}

#[test]
fn chat_streaming_title_shows_live_stats() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "partial".into(),
        stats: None,
    });
    app.chat.job = Some(Job::running_with_log(vec![]));
    app.chat.send_at = Some(std::time::Instant::now());
    app.chat.stream_chars = 40;
    let text = render(&app);
    assert!(text.contains("~10 tok"), "title carries live est. tokens");
    assert!(text.contains("tok/s"));
}

#[test]
fn composer_counts_display_width_for_cjk() {
    // Prompt occupies 2 cols; each CJK char takes 2 more.
    assert_eq!(count_visual_lines("ab", 4), 1);
    assert_eq!(count_visual_lines("你好", 4), 2);
    assert_eq!(count_visual_lines("你好你好", 4), 3);
    assert_eq!(count_visual_lines("abcd", 4), 2);
}

#[test]
fn server_ready_flips_off_when_process_exits() {
    let mut app = new_app();
    app.server = Some(Job::failed("boom".into()));
    app.server_ready = true;
    app.update_server_ready();
    assert!(!app.server_ready, "exited server must drop the ready flag");
}

#[test]
fn parse_health_body_accepts_ok_payload() {
    let health = super::super::server_probe::parse_health_body(
        r#"{"status":"ok","service":"ax-engine-server","model_id":"gemma4-e2b"}"#,
    )
    .expect("ok health");
    assert_eq!(health.model_id.as_deref(), Some("gemma4-e2b"));
    assert!(super::super::server_probe::parse_health_body(r#"{"status":"degraded"}"#).is_none());
    assert!(
        super::super::server_probe::parse_health_body(r#"{"status":"ok","service":"other"}"#)
            .is_none(),
        "foreign service must not attach"
    );
    assert!(super::super::server_probe::parse_health_body("not-json").is_none());
}

#[test]
fn format_http_base_url_brackets_bare_ipv6_hosts() {
    assert_eq!(
        super::super::server_probe::format_http_base_url("127.0.0.1", "31418"),
        "http://127.0.0.1:31418"
    );
    // Without brackets the parser would read the port as part of the host.
    assert_eq!(
        super::super::server_probe::format_http_base_url("::1", "31418"),
        "http://[::1]:31418"
    );
    // An already-bracketed host must not get a second pair.
    assert_eq!(
        super::super::server_probe::format_http_base_url("[::1]", "31418"),
        "http://[::1]:31418"
    );
}

#[test]
fn external_health_probe_marks_server_ready_without_child_job() {
    // User started `ax-engine-server` outside the TUI — Chat must still light up.
    let mut app = new_app();
    assert!(!app.server_ready);
    assert!(!app.server_running());
    let changed = app.apply_server_health(Some(super::super::server_probe::ServerHealth {
        model_id: Some("gemma4-e2b".into()),
    }));
    assert!(changed);
    assert!(app.server_ready);
    assert!(app.external_server);
    assert!(app.server_running());
    assert_eq!(app.server_url.as_deref(), Some("http://127.0.0.1:31418"));
    assert_eq!(app.server_model.as_deref(), Some("gemma4-e2b"));

    app.screen = Screen::Chat;
    let text = render(&app);
    assert!(
        !text.contains("Looking for server"),
        "chat empty-state must hide once external server is attached: {text}"
    );
    assert!(
        text.contains("running") || app.server_ready,
        "status should report ready"
    );
}

#[test]
fn external_health_loss_detaches_ready_state() {
    let mut app = new_app();
    app.apply_server_health(Some(super::super::server_probe::ServerHealth {
        model_id: Some("gemma4-e2b".into()),
    }));
    assert!(app.server_ready && app.external_server);
    let changed = app.apply_server_health(None);
    assert!(changed);
    assert!(!app.server_ready);
    assert!(!app.external_server);
    assert!(app.server_url.is_none());
    assert!(app.server_model.is_none());
}

#[test]
fn managed_health_probe_does_not_mark_external() {
    // Fallback readiness via /health while our child is still starting.
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec!["booting…".into()]));
    app.server_url = Some("http://127.0.0.1:31418".into());
    app.server_model = Some("local-label".into());
    app.apply_server_health(Some(super::super::server_probe::ServerHealth {
        model_id: Some("from-health".into()),
    }));
    assert!(app.server_ready);
    assert!(
        !app.external_server,
        "managed child must own Stop, not detach-only"
    );
    assert_eq!(
        app.server_model.as_deref(),
        Some("local-label"),
        "managed label wins over health model_id"
    );
}

#[test]
fn stop_external_server_detaches_without_child() {
    let mut app = new_app();
    app.apply_server_health(Some(super::super::server_probe::ServerHealth {
        model_id: Some("gemma4-e2b".into()),
    }));
    app.screen = Screen::Serve;
    app.on_key(key(KeyCode::Char('x')));
    assert!(matches!(app.modal, Some(Modal::StopServer)));
    app.on_key(key(KeyCode::Char('y')));
    assert!(!app.server_ready);
    assert!(!app.external_server);
    assert!(
        app.toasts
            .iter()
            .any(|t| t.text.contains("detached") || t.text.contains("external")),
        "expected detach toast, got {:?}",
        app.toasts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn serve_panel_shows_external_badge_when_attached() {
    let mut app = new_app();
    app.apply_server_health(Some(super::super::server_probe::ServerHealth {
        model_id: Some("gemma4-e2b".into()),
    }));
    app.screen = Screen::Serve;
    let text = render(&app);
    assert!(text.contains("external"), "serve panel: {text}");
    assert!(
        text.contains("http://127.0.0.1:31418"),
        "serve panel: {text}"
    );
}
