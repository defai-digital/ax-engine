//! Downloads and Serve screens plus their shared log-pane scrollback: queue
//! rendering, cancel/serve prompts, host/port validation, server readiness,
//! and log scrolling.
use super::super::jobs::Job;
use super::super::{App, Modal, Screen, ServeFocus};
use super::{key, mouse, new_app, render, test_task};
use ratatui::crossterm::event::KeyCode;
use ratatui::crossterm::event::MouseEventKind;
use std::process;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

#[test]
fn downloads_screen_renders_queue_and_gauge() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads
        .push(test_task(Some(Job::failed("queued test".into()))));
    let text = render(&app);
    assert!(text.contains("Queue") || text.contains("Downloads"));
    assert!(text.contains("gemma4-e2b"));
    assert!(text.contains("/tmp/gemma4-e2b"));
    assert!(text.contains("queued test"));
    assert!(text.contains("phase:"));
}

#[test]
fn enter_on_ready_download_asks_before_serving() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    let mut task = test_task(Some(Job::failed("x".into())));
    if let Some(job) = &mut task.job {
        job.done = Some(0); // mark ready
    }
    app.downloads.push(task);
    app.on_key(key(KeyCode::Enter));
    assert!(matches!(app.modal, Some(Modal::ServeReady { .. })));
    let text = render(&app);
    assert!(text.contains("Start the server with"));
}

#[test]
fn cancel_running_download_asks_first() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads
        .push(test_task(Some(Job::running_with_log(vec![]))));
    app.on_key(key(KeyCode::Char('x')));
    assert!(matches!(app.modal, Some(Modal::CancelDownload { .. })));
    // Backing out leaves it running.
    app.on_key(key(KeyCode::Char('n')));
    assert!(app.modal.is_none());
    assert!(app.downloads[0].is_running());
}

#[test]
fn cancelled_running_download_shows_cancelled_not_failed_code() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads
        .push(test_task(Some(Job::running_with_log(vec![]))));
    app.on_key(key(KeyCode::Char('x')));
    app.on_key(key(KeyCode::Char('y')));
    let task = &app.downloads[0];
    assert!(task.cancelled);
    assert_eq!(task.status_label(), "cancelled");
    assert!(!task.is_failed(), "user-cancelled is not a failure");
    let text = render(&app);
    assert!(text.contains("cancelled"), "queue row shows cancelled");
    assert!(
        !text.contains("failed (-130)"),
        "killed-by-user must not surface as failed (-130)"
    );
}

#[test]
fn download_failure_edge_toasts_retry_hint() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    // A real child that exits non-zero at once produces the None → Some
    // failure edge inside App::tick without any mocking.
    let job = Job::spawn(process::Command::new("false"), None).expect("spawn false");
    app.downloads.push(test_task(Some(job)));
    for _ in 0..200 {
        app.tick();
        if app.downloads[0]
            .job
            .as_ref()
            .is_some_and(|j| j.done.is_some())
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        app.downloads[0].is_failed(),
        "child exit marks the task failed"
    );
    assert!(
        app.toasts
            .iter()
            .any(|t| t.level == super::super::widgets::ToastLevel::Error
                && t.text.contains("failed")
                && t.text.contains("retry")),
        "expected the retry-hint error toast: {:?}",
        app.toasts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Serve
// ---------------------------------------------------------------------------

#[test]
fn serve_screen_renders_fields() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    let text = render(&app);
    assert!(text.contains("Models") || text.contains("Installed"));
    assert!(text.contains("Host"));
    assert!(text.contains("Port"));
}

#[test]
fn port_validation_rejects_bad_values_only() {
    let mut app = new_app();
    assert_eq!(app.port, "31418");
    assert!(app.port_error().is_none(), "default port is valid");
    app.port = "".into();
    assert!(app.port_error().is_none(), "empty falls back to default");
    app.port = "abc".into();
    assert_eq!(app.port_error(), Some("port must be 1-65535"));
    app.port = "99999".into();
    assert_eq!(app.port_error(), Some("port must be 1-65535"));
    app.port = "0".into();
    assert_eq!(app.port_error(), Some("port must be 1-65535"));
    app.port = "8080".into();
    assert!(app.port_error().is_none());
}

#[test]
fn server_status_waits_for_listening_line_before_going_green() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec!["booting model...".to_string()]));
    app.server_url = Some("http://127.0.0.1:8080".to_string());
    app.server_model = Some("gemma4-e2b".to_string());
    app.screen = Screen::Serve;
    assert!(!app.server_ready);
    assert!(render(&app).contains("starting"));

    app.server
        .as_mut()
        .unwrap()
        .log
        .push("ax-engine-server preview listening on http://127.0.0.1:8080".to_string());
    app.update_server_ready();
    assert!(app.server_ready);
    let text = render(&app);
    assert!(text.contains("running at"));
    assert!(text.contains("http://127.0.0.1:8080"));
    assert!(text.contains("curl"), "ready server shows a curl example");
}

#[test]
fn server_ready_accepts_tracing_bind_address_line() {
    // When RUST_LOG is set, ax-engine-server emits structured tracing instead of
    // the "listening on http://" operator line. The TUI must still go green.
    assert!(crate::tui::server_probe::server_log_indicates_ready(
        "2026-07-18T11:21:03.181885Z  INFO ax-engine-server preview listening bind_address=127.0.0.1:8080 model_id=gemma4-e2b"
    ));
    assert!(crate::tui::server_probe::server_log_indicates_ready(
        "ax-engine-server preview listening on http://127.0.0.1:8080 model_id=gemma4-e2b"
    ));
    assert!(!crate::tui::server_probe::server_log_indicates_ready(
        "booting model..."
    ));
    assert!(!crate::tui::server_probe::server_log_indicates_ready(
        "mlx error: [Primitive::output_shapes] CustomKernel cannot infer output shapes."
    ));

    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![
        "2026-07-18T11:21:03Z  INFO ax-engine-server preview listening bind_address=127.0.0.1:8080"
            .to_string(),
    ]));
    app.server_url = Some("http://127.0.0.1:8080".to_string());
    app.update_server_ready();
    assert!(
        app.server_ready,
        "tracing-format ready line must mark the server ready"
    );
}

#[test]
fn failed_server_surfaces_last_log_line() {
    let mut app = new_app();
    app.server = Some(Job::failed("model weights not found at /nope".into()));
    app.server_url = Some("http://127.0.0.1:8080".to_string());
    app.screen = Screen::Serve;
    let text = render(&app);
    assert!(text.contains("failed:"));
    assert!(text.contains("model weights not found"));
}

#[test]
fn failed_server_prefers_error_line_over_trailing_noise() {
    let mut app = new_app();
    let mut job = Job::failed("ignored".into());
    job.log = vec![
        "spawning /tmp/ax-engine-server for gemma4-e2b".into(),
        "Error: Custom { kind: InvalidInput, error: \"could not infer --model-id\" }".into(),
        "mlx error: [Primitive::output_shapes] CustomKernel cannot infer output shapes.".into(),
        "".into(),
    ];
    job.done = Some(1);
    app.server = Some(job);
    app.server_url = Some("http://127.0.0.1:8080".to_string());
    let err = app.server_error_line().expect("error line");
    assert!(
        err.contains("could not infer") || err.contains("InvalidInput"),
        "expected real error, got {err:?}"
    );
}

#[test]
fn stopping_server_asks_first() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.screen = Screen::Serve;
    app.on_key(key(KeyCode::Char('x')));
    assert!(matches!(app.modal, Some(Modal::StopServer)));
}

#[test]
fn server_dying_before_ready_toasts_error() {
    let mut app = new_app();
    // A server job that dies before printing a bind line: tick must surface
    // an error toast (the ready->stopped crash path warns separately).
    app.server = Some(Job::spawn(process::Command::new("false"), None).expect("spawn false"));
    app.server_url = Some("http://127.0.0.1:8080".into());
    assert!(app.server_running());
    assert!(!app.server_ready);
    for _ in 0..200 {
        app.tick();
        if !app.server_running() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(!app.server_running(), "child exit ends the running state");
    assert!(!app.server_ready);
    assert!(
        app.toasts
            .iter()
            .any(|t| t.level == super::super::widgets::ToastLevel::Error
                && t.text.contains("server failed to start")),
        "expected a failed-to-start error toast: {:?}",
        app.toasts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn serve_footer_shows_field_editing_hints() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    for focus in [ServeFocus::Host, ServeFocus::Port] {
        app.serve_focus = focus;
        let text = render(&app);
        assert!(
            text.contains("type to edit"),
            "field focus advertises typing: {text:.200}"
        );
        assert!(
            !text.contains("Enter start"),
            "list-mode hints must not show while editing a field"
        );
    }
}

#[test]
fn serve_fields_support_caret_editing() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    app.serve_focus = ServeFocus::Host;
    app.host_cursor = app.host.chars().count();
    // Insert at the caret, not blindly at the end.
    app.on_key(key(KeyCode::Left));
    app.on_key(key(KeyCode::Left));
    app.on_key(key(KeyCode::Char('X')));
    assert_eq!(app.host, "127.0.0X.1");
    // Backspace removes before the caret, Delete removes at it.
    app.on_key(key(KeyCode::Backspace));
    assert_eq!(app.host, "127.0.0.1");
    app.on_key(key(KeyCode::Home));
    app.on_key(key(KeyCode::Delete));
    assert_eq!(app.host, "27.0.0.1");
    app.on_key(key(KeyCode::End));
    app.on_key(key(KeyCode::Char('9')));
    assert_eq!(app.host, "27.0.0.19");
    // Tab moves Host → Port with the caret at the end of the port field.
    app.on_key(key(KeyCode::Tab));
    assert!(matches!(app.serve_focus, ServeFocus::Port));
    assert_eq!(app.port_cursor, app.port.chars().count());
}

#[test]
fn serve_paste_inserts_at_caret_and_strips_controls() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    app.serve_focus = ServeFocus::Port;
    app.port = "800".into();
    app.port_cursor = 1;
    app.on_paste("8\n8");
    assert_eq!(app.port, "88800");
    assert_eq!(app.port_cursor, 3);
}

#[test]
fn serve_rejects_invalid_host_before_spawning() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    app.host = "bad host!".into();
    app.serve_installed(0, 0);
    assert!(app.server.is_none(), "invalid host must not spawn");
    assert!(
        app.toasts.iter().any(|t| t.text.contains("host")),
        "expected a host validation toast: {:?}",
        app.toasts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Log pane scrollback (Downloads / Serve)
// ---------------------------------------------------------------------------

/// Downloads app whose selected task carries `lines` canned log lines.
fn downloads_app_with_log(lines: usize) -> App {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    let log = (0..lines).map(|i| format!("log line {i:03}")).collect();
    app.downloads
        .push(test_task(Some(Job::running_with_log(log))));
    app
}

/// Content height of the log pane as last drawn; fails if never drawn.
fn log_view_height(app: &App) -> usize {
    let rect = app.log_rect.get();
    assert!(rect.height > 1, "log pane must be drawn before scrolling");
    (rect.height - 1) as usize
}

fn push_download_log_lines(app: &mut App, from: usize, count: usize) {
    let job = app.downloads[0].job.as_mut().unwrap();
    for i in from..from + count {
        job.log.push(format!("log line {i:03}"));
    }
}

#[test]
fn downloads_log_pgup_scrolls_and_pgdn_repins() {
    let mut app = downloads_app_with_log(120);
    let text = render(&app);
    // Pinned at the bottom: newest lines visible, oldest not, no indicator.
    assert!(text.contains("log line 119"));
    assert!(!text.contains("log line 000"));
    assert!(!text.contains("scrolled"));
    assert!(app.downloads_log_scroll.is_pinned());

    let height = log_view_height(&app);
    for _ in 0..10 {
        app.on_key(key(KeyCode::PageUp));
    }
    assert_eq!(
        app.downloads_log_scroll.first_visible(120, height),
        0,
        "repeated PgUp lands on (and clamps at) the oldest line"
    );
    let text = render(&app);
    assert!(
        text.contains("log line 000"),
        "oldest lines visible: {text}"
    );
    assert!(!text.contains("log line 119"));
    assert!(
        text.contains("↑ scrolled (PgDn to bottom)"),
        "scrolled indicator in the pane title: {text}"
    );

    for _ in 0..10 {
        app.on_key(key(KeyCode::PageDown));
    }
    assert!(
        app.downloads_log_scroll.is_pinned(),
        "reaching the bottom re-pins"
    );
    let text = render(&app);
    assert!(text.contains("log line 119"));
    assert!(
        !text.contains("↑ scrolled"),
        "indicator hides when pinned: {text}"
    );
}

#[test]
fn wheel_over_downloads_log_scrolls_log_not_queue_selection() {
    let mut app = downloads_app_with_log(120);
    app.downloads
        .push(test_task(Some(Job::running_with_log(vec![]))));
    let _ = render(&app); // records log + queue hit rects
    let log_rect = app.log_rect.get();
    let (lx, ly) = (log_rect.x + 2, log_rect.y + 2);

    app.on_mouse(mouse(MouseEventKind::ScrollUp, lx, ly));
    assert!(
        !app.downloads_log_scroll.is_pinned(),
        "wheel over the log pane scrolls the log"
    );
    assert_eq!(app.download_idx, 0, "queue selection must not move");
    app.on_mouse(mouse(MouseEventKind::ScrollDown, lx, ly));
    assert!(
        app.downloads_log_scroll.is_pinned(),
        "wheeling back down re-pins"
    );

    // Wheel outside the log pane keeps today's behavior (row selection).
    let queue = app.content_list_rect.get();
    app.on_mouse(mouse(MouseEventKind::ScrollDown, queue.x + 2, queue.y + 2));
    assert_eq!(app.download_idx, 1);
    assert!(app.downloads_log_scroll.is_pinned());
}

#[test]
fn downloads_log_scroll_clamps_at_top_and_bottom() {
    let mut app = downloads_app_with_log(30);
    let _ = render(&app);
    let height = log_view_height(&app);
    let bottom = 30usize.saturating_sub(height);
    assert!(bottom > 0, "the log must overflow the pane for this test");

    app.on_key(key(KeyCode::PageUp)); // one page overshoots a short overflow
    assert_eq!(
        app.downloads_log_scroll.first_visible(30, height),
        0,
        "cannot scroll past the top"
    );
    app.on_key(key(KeyCode::PageDown));
    assert!(
        app.downloads_log_scroll.is_pinned(),
        "bottom clamps to pinned instead of overshooting"
    );
}

#[test]
fn changing_download_row_resets_log_scroll() {
    let mut app = downloads_app_with_log(120);
    app.downloads.push(test_task(Some(Job::running_with_log(
        (0..120).map(|i| format!("other line {i:03}")).collect(),
    ))));
    let _ = render(&app);
    app.on_key(key(KeyCode::PageUp));
    assert!(!app.downloads_log_scroll.is_pinned());

    app.on_key(key(KeyCode::Down)); // move selection to the second row
    assert_eq!(app.download_idx, 1);
    assert!(
        app.downloads_log_scroll.is_pinned(),
        "switching rows re-pins the log pane"
    );
    app.on_key(key(KeyCode::Up));
    assert_eq!(app.download_idx, 0);
    assert!(
        app.downloads_log_scroll.is_pinned(),
        "coming back to the row stays pinned (no stale offset)"
    );
}

#[test]
fn serve_log_scrolls_with_keys_and_wheel() {
    let mut app = new_app();
    app.screen = Screen::Serve;
    app.server = Some(Job::running_with_log(
        (0..120).map(|i| format!("serve line {i:03}")).collect(),
    ));
    let _ = render(&app);

    app.on_key(key(KeyCode::PageUp));
    assert!(!app.serve_log_scroll.is_pinned());
    let text = render(&app);
    assert!(
        text.contains("↑ scrolled"),
        "serve log shows the indicator: {text}"
    );
    // Field-editing focus does not block log scrolling.
    app.serve_focus = ServeFocus::Host;
    app.on_key(key(KeyCode::PageDown));
    assert!(app.serve_log_scroll.is_pinned());
    app.serve_focus = ServeFocus::List;

    // Wheel over the log pane scrolls it too.
    let rect = app.log_rect.get();
    app.on_mouse(mouse(MouseEventKind::ScrollUp, rect.x + 2, rect.y + 2));
    assert!(!app.serve_log_scroll.is_pinned());

    // Replacing the server job re-pins the pane.
    app.stop_server();
    assert!(app.serve_log_scroll.is_pinned());
}

#[test]
fn new_log_lines_stay_put_when_scrolled_and_follow_when_pinned() {
    let mut app = downloads_app_with_log(60);
    let _ = render(&app);
    let height = log_view_height(&app);

    // Pinned: appended lines keep the view on the newest output.
    push_download_log_lines(&mut app, 60, 20); // lines 060..=079
    assert_eq!(
        app.downloads_log_scroll.first_visible(80, height),
        80 - height
    );
    let text = render(&app);
    assert!(
        text.contains("log line 079"),
        "pinned view follows new lines"
    );

    // Scrolled up: the same lines stay in view as more output arrives.
    app.on_key(key(KeyCode::PageUp));
    let anchored = app.downloads_log_scroll.first_visible(80, height);
    push_download_log_lines(&mut app, 80, 20); // lines 080..=099
    assert_eq!(
        app.downloads_log_scroll.first_visible(100, height),
        anchored,
        "scrolled view must not move when new lines arrive"
    );
    let text = render(&app);
    let anchored_line = format!("log line {anchored:03}");
    assert!(
        text.contains(&anchored_line),
        "anchored line still on screen: {text}"
    );
    assert!(!text.contains("log line 099"), "no yank to the bottom");
}
