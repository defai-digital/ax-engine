//! Assorted regression tests: serve switch-restart, thinking collapse, toast
//! coalescing, chart legend, download-status mapping, host validation.
use super::super::jobs::{DownloadStatus, Job};
use super::super::screens::chat::ChatMessage;
use super::super::{App, Modal, Screen};
use super::{chat_ready_app, ctrl_key, family_index, key, new_app, render, test_task, type_text};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Wave 5: serve switch-restart, thinking collapse, toast coalescing,
// chart legend/x-label, download status enum
// ---------------------------------------------------------------------------

/// Serve screen with a ready server on variant 0 of a family that also has
/// variant 1 "installed" (flag-only; nothing touches the disk or a process).
fn serve_app_with_two_installed() -> (App, usize) {
    let mut app = new_app();
    let fi = family_index(&app, "ax-qwen3.5-9b");
    app.families[fi].variants[0].installed = true;
    app.families[fi].variants[1].installed = true;
    app.server = Some(Job::running_with_log(vec![]));
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.server_model = Some(app.families[fi].variants[0].profile.label.to_string());
    app.screen = Screen::Serve;
    (app, fi)
}

#[test]
fn serve_enter_on_other_model_opens_restart_modal() {
    let (mut app, fi) = serve_app_with_two_installed();
    // The machine may have real installs; find our pair's row by identity.
    let pairs = super::super::catalog::installed_variants(&app.families);
    app.serve_idx = pairs.iter().position(|&p| p == (fi, 1)).unwrap();
    app.on_key(key(KeyCode::Enter));
    assert!(
        matches!(
            app.modal,
            Some(Modal::RestartServer {
                family_idx,
                variant_idx
            }) if family_idx == fi && variant_idx == 1
        ),
        "Enter on a different installed model asks to restart"
    );
    assert!(app.server_running(), "nothing stops before confirmation");
    let text = render(&app);
    assert!(
        text.contains("Restart the server with"),
        "modal body renders: {text:.200}"
    );
}

#[test]
fn serve_enter_on_served_model_still_goes_to_chat() {
    let (mut app, fi) = serve_app_with_two_installed();
    let pairs = super::super::catalog::installed_variants(&app.families);
    app.serve_idx = pairs.iter().position(|&p| p == (fi, 0)).unwrap();
    app.on_key(key(KeyCode::Enter));
    assert!(app.modal.is_none(), "the served row keeps the chat handoff");
    assert_eq!(app.screen, Screen::Chat);
}

#[test]
fn serve_restart_confirm_stops_then_starts_selected_model() {
    let (mut app, fi) = serve_app_with_two_installed();
    let pairs = super::super::catalog::installed_variants(&app.families);
    app.serve_idx = pairs.iter().position(|&p| p == (fi, 1)).unwrap();
    app.on_key(key(KeyCode::Enter));
    assert!(matches!(app.modal, Some(Modal::RestartServer { .. })));
    // Keep the restart process-free: an invalid port makes the serve path
    // bail with a validation toast before any child process is launched.
    app.port = "not-a-port".into();
    app.on_key(key(KeyCode::Char('y')));
    assert!(app.modal.is_none(), "y confirms and closes the modal");
    assert!(app.server.is_none(), "the old server job was stopped");
    assert!(!app.server_ready);
    assert!(app.server_model.is_none());
    assert!(
        app.toasts.iter().any(|t| t.text.contains("port must be")),
        "the restart reached the serve path (blocked at validation)"
    );
    assert!(
        !app.auto_chat_after_serve,
        "a restart refused at validation must not arm the chat handoff, \
         or an unrelated later serve would auto-jump to Chat"
    );
}

#[test]
fn serve_restart_modal_dismiss_keys_keep_the_server() {
    for dismiss in [
        key(KeyCode::Esc),
        key(KeyCode::Char('n')),
        key(KeyCode::Char('h')),
        key(KeyCode::Left),
    ] {
        let (mut app, fi) = serve_app_with_two_installed();
        let pairs = super::super::catalog::installed_variants(&app.families);
        app.serve_idx = pairs.iter().position(|&p| p == (fi, 1)).unwrap();
        app.on_key(key(KeyCode::Enter));
        assert!(matches!(app.modal, Some(Modal::RestartServer { .. })));
        app.on_key(dismiss);
        assert!(app.modal.is_none(), "dismiss closes the modal");
        assert!(app.server_running(), "server keeps running");
        assert!(app.server_ready);
        assert_eq!(app.screen, Screen::Serve);
    }
}

#[test]
fn chat_thinking_collapses_past_two_lines_and_ctrl_t_expands() {
    let mut app = chat_ready_app();
    app.server_model = Some("qwen3-4b".into());
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "alpha\nbravo\ncharlie\ndelta</think>\n\nThe answer.".into(),
        stats: None,
    });
    let text = render(&app);
    assert!(
        text.contains("alpha") && text.contains("bravo"),
        "collapsed preview keeps the first 2 lines"
    );
    assert!(
        !text.contains("charlie") && !text.contains("delta"),
        "later thinking lines are collapsed away"
    );
    assert!(
        text.contains("2 more thinking lines (Ctrl+T to expand)"),
        "summary line renders: {text}"
    );
    assert!(text.contains("The answer."), "the answer never collapses");

    app.on_key(ctrl_key('t'));
    let text = render(&app);
    assert!(
        text.contains("charlie") && text.contains("delta"),
        "Ctrl+T expands every thinking block"
    );
    assert!(!text.contains("more thinking lines"));

    app.on_key(ctrl_key('t'));
    let text = render(&app);
    assert!(!text.contains("charlie"), "Ctrl+T again re-collapses");
}

#[test]
fn chat_thinking_within_budget_renders_fully() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "short one\nshort two</think>\n\nAnswer.".into(),
        stats: None,
    });
    let text = render(&app);
    assert!(text.contains("short one") && text.contains("short two"));
    assert!(
        !text.contains("more thinking lines"),
        "a 2-line block needs no summary line"
    );
}

#[test]
fn chat_thinking_streams_fully_while_in_progress() {
    let mut app = chat_ready_app();
    app.chat.messages.push(ChatMessage {
        from_user: false,
        content: "<think>r1\nr2\nr3\nr4".into(), // close tag has not arrived
        stats: None,
    });
    app.chat.job = Some(Job::running_with_log(vec![]));
    app.chat.send_at = Some(std::time::Instant::now());
    let text = render(&app);
    assert!(
        text.contains("r3") && text.contains("r4"),
        "in-progress thinking renders as it arrives, uncollapsed"
    );
}

#[test]
fn identical_consecutive_toasts_coalesce_and_refresh_timestamp() {
    let mut app = new_app();
    app.toast("still downloading");
    let first_at = app.toasts[0].at;
    std::thread::sleep(Duration::from_millis(5));
    app.toast("still downloading");
    assert_eq!(app.toasts.len(), 1, "same text + level coalesces");
    assert!(app.toasts[0].at > first_at, "the timestamp is refreshed");
    // Same text at a different level is a different toast.
    app.toast_warn("still downloading");
    assert_eq!(app.toasts.len(), 2);
    // Different text stacks as usual.
    app.toast("queued — starts after current download");
    assert_eq!(app.toasts.len(), 3);
}

#[test]
fn download_status_maps_every_state() {
    let queued = test_task(None);
    assert_eq!(queued.status(), DownloadStatus::Queued);
    assert_eq!(queued.status_label(), "queued");

    let running = test_task(Some(Job::running_with_log(vec![])));
    assert_eq!(running.status(), DownloadStatus::Running);
    assert_eq!(running.status_label(), "running");

    let ready = test_task(Some(Job::exited(0)));
    assert_eq!(ready.status(), DownloadStatus::Ready);
    assert_eq!(ready.status_label(), "ready");

    let failed = test_task(Some(Job::exited(3)));
    assert_eq!(failed.status(), DownloadStatus::Failed);
    assert_eq!(
        failed.status_label(),
        "failed (3)",
        "label keeps the exit code"
    );

    let mut cancelled = test_task(None);
    cancelled.cancel();
    assert_eq!(cancelled.status(), DownloadStatus::Cancelled);
    assert_eq!(cancelled.status_label(), "cancelled");

    // The cancel flag wins over a finished job (user intent beats exit code).
    let mut cancelled_ready = test_task(Some(Job::exited(0)));
    cancelled_ready.cancelled = true;
    assert_eq!(cancelled_ready.status(), DownloadStatus::Cancelled);
}

#[test]
fn util_chart_legend_entries_and_derived_x_label_render() {
    let metrics = super::super::metrics::LiveMetrics::for_tests();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|frame| {
            super::super::screens::metrics_panel::draw_live_metrics(frame, frame.area(), &metrics);
        })
        .unwrap();
    let text: String = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect();
    // Legend entries carry their series letter, not just a color.
    for entry in ["Mem", "CPU", "GPU"] {
        assert!(text.contains(entry), "legend entry {entry}: {text:.200}");
    }
    // The x window label comes from HISTORY_LEN × SAMPLE_INTERVAL.
    let secs = super::super::metrics::HISTORY_LEN as u64
        * super::super::metrics::SAMPLE_INTERVAL.as_secs();
    let expected = format!("~{}m", secs / 60);
    assert_eq!(
        super::super::screens::metrics_panel::window_label(),
        expected
    );
    assert_eq!(expected, "~4m", "120 samples x 2 s = 4 minutes");
    assert!(
        text.contains(&expected),
        "x-axis label {expected}: {text:.200}"
    );
}

#[test]
fn chat_multiline_down_at_bottom_scrolls_transcript() {
    let mut app = chat_ready_app();
    for i in 0..40 {
        app.chat.messages.push(ChatMessage {
            from_user: false,
            content: format!("line {i}"),
            stats: None,
        });
    }
    type_text(&mut app, "hi");
    let mut nl = key(KeyCode::Char('j'));
    nl.modifiers = KeyModifiers::CONTROL;
    app.on_key(nl);
    type_text(&mut app, "there");
    assert!(app.chat.input.contains('\n'));
    let before = app.chat.scroll.borrow().offset().y;
    app.on_key(key(KeyCode::Down));
    let after = app.chat.scroll.borrow().offset().y;
    assert!(
        after > before,
        "Down at bottom of multi-line draft scrolls transcript (before={before} after={after})"
    );
}

#[test]
fn chat_delete_key_forward_deletes_at_cursor() {
    let mut app = chat_ready_app();
    type_text(&mut app, "abcd");
    app.on_key(key(KeyCode::Home));
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.chat.input, "bcd");
    assert_eq!(app.chat.cursor, 0);
    app.on_key(key(KeyCode::Right));
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.chat.input, "bd");
    assert_eq!(app.chat.cursor, 1);
    app.on_key(key(KeyCode::End));
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.chat.input, "bd", "Delete at end is a no-op");
    app.chat.input = "aé日".into();
    app.chat.cursor = 1;
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.chat.input, "a日");
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.chat.input, "a");
}

#[test]
fn chat_typing_gate_excludes_crashed_server() {
    let mut app = chat_ready_app();
    assert!(app.typing(), "live server => chat is typing");
    if let Some(job) = app.server.as_mut() {
        job.done = Some(1);
    }
    assert!(
        !app.typing(),
        "crashed server => chat is not typing (read-only)"
    );
    app.on_key(key(KeyCode::Char('q')));
    assert!(
        app.chat.input.is_empty(),
        "read-only chat does not capture letter keys"
    );
}

#[test]
fn chat_paste_blocked_when_server_crashed() {
    // A running server accepts pastes.
    let mut app = chat_ready_app();
    app.on_paste("hello");
    assert_eq!(app.chat.input, "hello");
    app.chat.input.clear();
    app.chat.cursor = 0;
    // Simulate a crash — job.done = Some(_) but server_ready hasn't flipped
    // yet (one-tick gap between child exit and update_server_ready).
    if let Some(job) = app.server.as_mut() {
        job.done = Some(1);
    }
    app.on_paste("world");
    assert!(
        app.chat.input.is_empty(),
        "read-only chat must not accept paste"
    );
    // And the user should see a toast (not silent drop).
    assert!(
        !app.toasts.is_empty(),
        "crashed-server paste must surface a toast"
    );
}

#[test]
fn host_validation_accepts_underscores() {
    let mut app = new_app();
    app.host = "my_server.local".to_string();
    assert!(
        app.host_error().is_none(),
        "underscores are valid in Docker/K8s hostnames"
    );
    app.host = "valid-host.example.com".to_string();
    assert!(app.host_error().is_none());
    app.host = "127.0.0.1".to_string();
    assert!(app.host_error().is_none());
    app.host = "host with spaces".to_string();
    assert!(app.host_error().is_some(), "spaces must still be rejected");
    app.host = "".to_string();
    assert!(
        app.host_error().is_none(),
        "empty host is allowed (defaults)"
    );
}
