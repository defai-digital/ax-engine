//! Screen navigation and global keys, plus the Home screen: tab switching,
//! Esc history, quit confirmation, toasts, and tiny-terminal fallbacks.
use super::super::jobs::Job;
use super::super::metrics::{
    parse_loadavg_1m, parse_ps_cpu_percent, parse_ps_top_rss, parse_vm_stat_free_bytes,
    parse_vm_stat_used_bytes,
};
use super::super::{Modal, Screen, WizardStage};
use super::{key, mouse, new_app, render, render_sized, test_task};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::KeyCode;
use ratatui::crossterm::event::{MouseButton, MouseEventKind};

// ---------------------------------------------------------------------------
// Navigation and global keys
// ---------------------------------------------------------------------------

#[test]
fn app_starts_on_home_with_hardware_summary() {
    let app = new_app();
    assert_eq!(app.screen, Screen::Home);
    let text = render(&app);
    assert!(
        text.contains("This Mac")
            || text.contains("Memory")
            || text.contains("Quick start")
            || text.contains("CPU")
    );
    assert!(text.contains("Quick start"));
    assert!(text.contains("Browse all models"));
    assert!(
        text.contains("Get started")
            || text.contains("Start here")
            || text.contains("ready")
            || text.contains("Downloading")
            || text.contains("Server")
            || text.contains("Actions")
            || text.contains("This Mac")
    );
}

#[test]
fn home_default_action_is_browse_when_models_installed() {
    use super::super::catalog::installed_variants;
    use super::super::screens::home::HomeAction;

    let app = new_app();
    let actions = app.home_actions();
    let selected = actions
        .get(app.home_idx)
        .map(|(_, action)| *action)
        .expect("home has at least one action");
    if installed_variants(&app.families).is_empty() {
        assert_eq!(
            selected,
            HomeAction::QuickStart,
            "first-run default should be Quick start"
        );
        assert_eq!(actions[0].1, HomeAction::QuickStart);
    } else {
        assert_eq!(
            selected,
            HomeAction::Browse,
            "with installed models, Enter must not immediately serve; default Browse"
        );
        assert_eq!(actions[0].1, HomeAction::Browse);
        assert!(
            actions.iter().any(|(_, a)| *a == HomeAction::QuickStart),
            "Quick start remains available as a non-default shortcut"
        );
    }
}

#[test]
fn live_metrics_panel_renders_gauges() {
    let mut app = new_app();
    app.live_metrics = super::super::metrics::LiveMetrics::for_tests();
    let text = render(&app);
    assert!(
        text.contains("This Mac")
            || text.contains("CPU")
            || text.contains("GPU")
            || text.contains("MEM")
            || text.contains("Utilization")
            || text.contains("unified"),
        "home should show Mac host monitor with CPU/GPU: {text:.240}"
    );
}

#[test]
fn metrics_parsers_unit() {
    let vm = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages active: 10.
Pages speculative: 20.
Pages wired down: 5.
Pages purgeable: 5.
Pages occupied by compressor: 1.
";
    assert_eq!(parse_vm_stat_used_bytes(vm), Some(16 * 16_384));
    assert_eq!(parse_vm_stat_free_bytes(vm), Some(125 * 16_384));
    assert!((parse_ps_cpu_percent("50.0\n50.0\n", 2.0).unwrap() - 50.0).abs() < 1e-6);
    assert!((parse_loadavg_1m("{ 0.5 0.6 0.7 }").unwrap() - 0.5).abs() < 1e-6);
    let tops = parse_ps_top_rss(" 200  9 /usr/bin/foo\n 100  8 bar\n", 2);
    assert_eq!(tops[0].name, "foo");
    assert_eq!(tops[0].rss_bytes, 200 * 1024);
}

#[test]
fn live_metrics_shows_htop_style_top_and_free() {
    let mut app = new_app();
    app.live_metrics = super::super::metrics::LiveMetrics::for_tests();
    let text = render(&app);
    assert!(
        text.contains("CPU")
            || text.contains("GPU")
            || text.contains("This Mac")
            || text.contains("MEM"),
        "Mac host monitor meters expected: {text:.200}"
    );
    assert!(
        text.contains("Code")
            || text.contains("RSS")
            || text.contains("COMMAND")
            || text.contains("free")
            || text.contains("unified"),
        "should surface process strip or identity: {text:.200}"
    );
}

#[test]
fn ioreg_gpu_parser_unit() {
    use super::super::metrics::parse_ioreg_gpu;
    let raw = r#"
"PerformanceStatistics" = {"Device Utilization %"=42,"In use system memory"=2048}
"model" = "Apple M4 Pro"
"gpu-core-count" = 20
"#;
    let s = parse_ioreg_gpu(raw);
    assert!((s.gpu_percent.unwrap() - 42.0).abs() < 1e-6);
    assert_eq!(s.gpu_cores, Some(20));
    assert_eq!(s.chip_name.as_deref(), Some("Apple M4 Pro"));
    assert_eq!(s.gpu_mem_bytes, Some(2048));
}

#[test]
fn number_keys_switch_screens() {
    let mut app = new_app();
    app.on_key(key(KeyCode::Char('2')));
    assert_eq!(app.screen, Screen::Models);
    app.on_key(key(KeyCode::Char('3')));
    assert_eq!(app.screen, Screen::Downloads);
    app.on_key(key(KeyCode::Char('4')));
    assert_eq!(app.screen, Screen::Serve);
    app.on_key(key(KeyCode::Char('5')));
    assert_eq!(app.screen, Screen::Chat);
    // Without a ready server, Chat is a hint screen — digits still navigate.
    app.on_key(key(KeyCode::Char('1')));
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn up_from_first_row_focuses_tab_bar_then_left_right_switch() {
    let mut app = new_app();
    assert_eq!(app.screen, Screen::Home);
    assert!(!app.focus_tabs);
    // Home starts on the first action — further Up reaches the tab bar.
    app.on_key(key(KeyCode::Up));
    assert!(app.focus_tabs, "Up at top of content focuses the tab bar");
    let text = render(&app);
    assert!(text.contains("Home") && text.contains("Models"));
    // Left/Right while focused switch screens and stay on the bar.
    app.on_key(key(KeyCode::Right));
    assert_eq!(app.screen, Screen::Models);
    assert!(app.focus_tabs);
    app.on_key(key(KeyCode::Right));
    assert_eq!(app.screen, Screen::Downloads);
    // Down returns focus to content.
    app.on_key(key(KeyCode::Down));
    assert!(!app.focus_tabs);
    assert_eq!(app.screen, Screen::Downloads);
}

#[test]
fn tab_bar_focus_works_even_when_chat_is_typing() {
    // Regression: focus_tabs used to be ignored while Chat was in typing mode,
    // so the bar looked focused but keys still went into the composer.
    let mut app = new_app();
    app.screen = Screen::Chat;
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.focus_tab_bar();
    assert!(app.focus_tabs);
    // Chat is the last tab — Left goes to Serve and stays on the bar.
    app.on_key(key(KeyCode::Left));
    assert_eq!(app.screen, Screen::Serve);
    assert!(app.focus_tabs);
    // Digits jump screens while the bar is focused (not typed into chat).
    app.on_key(key(KeyCode::Char('5')));
    assert_eq!(app.screen, Screen::Chat);
    assert!(!app.focus_tabs);
    assert!(
        app.chat.input.is_empty(),
        "tab focus must not type into chat"
    );
    // Re-focus bar from chat and ensure 'q' quits instead of typing.
    app.focus_tab_bar();
    app.on_key(key(KeyCode::Char('q')));
    assert!(app.quit);
    assert!(app.chat.input.is_empty());
}

#[test]
fn esc_walks_back_through_screen_history() {
    let mut app = new_app();
    // Home → Models → Downloads pushes a real history stack.
    app.on_key(key(KeyCode::Char('2')));
    assert_eq!(app.screen, Screen::Models);
    app.on_key(key(KeyCode::Char('3')));
    assert_eq!(app.screen, Screen::Downloads);
    // First Esc pops to Models, second all the way to Home — a single
    // previous-screen slot used to lose the middle stop.
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Models);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
    // Nothing left on the stack: Esc stays on Home.
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn serve_while_running_surfaces_toast_instead_of_silent_no_op() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.serve_installed(0, 0);
    assert!(
        app.toasts
            .iter()
            .any(|t| t.text.contains("stop the running")),
        "expected a warning toast when a server is already running"
    );
}

#[test]
fn quick_start_enables_auto_chain_flags() {
    let mut app = new_app();
    app.quick_start_from_home();
    if app.modal.is_some() {
        // Recommended model already installed on this machine.
        assert!(
            app.auto_chat_after_serve,
            "installed quick start should auto-open Chat after serve"
        );
    } else {
        assert!(
            app.auto_serve_after_download && app.auto_chat_after_serve,
            "guided quick start should arm download→serve→chat"
        );
        assert_eq!(app.screen, Screen::Models);
    }
}

#[test]
fn failed_download_can_be_retried() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads
        .push(test_task(Some(Job::failed("network error".into()))));
    assert!(app.downloads[0].is_failed());
    app.on_key(key(KeyCode::Char('r')));
    // requeue + start_next may immediately spawn a job.
    assert!(!app.downloads[0].is_failed(), "retry clears failed status");
    assert!(
        app.downloads[0].is_queued()
            || app.downloads[0].is_running()
            || app.downloads[0].job.is_some(),
        "retry re-arms the download"
    );
}

#[test]
fn esc_backs_one_screen_level() {
    let mut app = new_app();
    app.navigate_to(Screen::Models);
    app.navigate_to(Screen::Downloads);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Models);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn chat_hint_screen_never_traps() {
    let mut app = new_app();
    app.screen = Screen::Chat;
    // No ready server: plain characters are ignored, not captured as input.
    app.on_key(key(KeyCode::Char('z')));
    assert!(app.chat.input.is_empty());
    // Number keys switch screens; Esc/Left step back one level.
    app.on_key(key(KeyCode::Char('4')));
    assert_eq!(app.screen, Screen::Serve);
    app.on_key(key(KeyCode::Char('5')));
    assert_eq!(app.screen, Screen::Chat);
    app.on_key(key(KeyCode::Left));
    assert_eq!(
        app.screen,
        Screen::Serve,
        "back one level restores prior screen"
    );
    // The stack still remembers the original Chat visit; one more Esc pops
    // there, and only with the stack empty does Esc fall back to Home.
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Chat);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn help_closes_on_any_key() {
    let mut app = new_app();
    app.on_key(key(KeyCode::Char('?')));
    assert!(app.show_help);
    app.on_key(key(KeyCode::Down));
    assert!(!app.show_help, "any key closes help");
}

#[test]
fn theme_defaults_to_dark_palette_and_unicode_glyphs() {
    // Tests never call theme::init(), so the dark palette + Unicode glyphs
    // must be the fallback regardless of the host locale.
    assert_eq!(
        super::super::theme::colors().accent,
        ratatui::style::Color::Rgb(56, 189, 248)
    );
    assert_eq!(super::super::theme::icon::ok(), "✓");
}

#[test]
fn tiny_terminal_gets_resize_hint_instead_of_broken_layout() {
    let app = new_app();
    let text = render_sized(&app, 50, 10);
    assert!(text.contains("terminal too small"), "got: {text}");
    // Back at a normal size the regular chrome renders again.
    let text = render(&app);
    assert!(text.contains("Home"));
}

#[test]
fn esc_steps_back_one_level_and_never_quits() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.stage = WizardStage::Precision;
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.stage, WizardStage::Families);
    app.on_key(key(KeyCode::Esc));
    assert_eq!(app.screen, Screen::Home);
    // On Home, Esc moves up to the tab bar — never quits.
    app.on_key(key(KeyCode::Esc));
    assert!(app.focus_tabs);
    assert!(!app.quit, "Esc never quits; use q");
    app.on_key(key(KeyCode::Esc));
    assert!(!app.focus_tabs);
    assert!(!app.quit);
}

#[test]
fn quit_is_immediate_when_idle_and_confirmed_when_busy() {
    let mut app = new_app();
    app.downloads.push(test_task(None)); // queued counts as busy
    app.on_key(key(KeyCode::Char('q')));
    assert!(!app.quit);
    assert!(matches!(app.modal, Some(Modal::Quit { .. })));
    app.on_key(key(KeyCode::Char('y')));
    assert!(app.quit);

    let mut idle = new_app();
    idle.on_key(key(KeyCode::Char('q')));
    assert!(idle.quit, "no jobs -> quit without a modal");
}

#[test]
fn quit_modal_can_be_dismissed() {
    let mut app = new_app();
    app.downloads.push(test_task(None));
    app.on_key(key(KeyCode::Char('q')));
    assert!(matches!(app.modal, Some(Modal::Quit { .. })));
    app.on_key(key(KeyCode::Esc));
    assert!(app.modal.is_none());
    assert!(!app.quit);
    // Left backs out too — arrow keys must never feel stuck in a dialog.
    app.on_key(key(KeyCode::Char('q')));
    assert!(matches!(app.modal, Some(Modal::Quit { .. })));
    app.on_key(key(KeyCode::Left));
    assert!(app.modal.is_none());
    assert!(!app.quit);
}

#[test]
fn click_on_tab_bar_switches_screen() {
    let mut app = new_app();
    let _ = render(&app); // records tab_hits
    let hits = app.tab_hits.take();
    assert!(
        hits.len() >= 3,
        "expected at least 3 tabs, got {}",
        hits.len()
    );
    let models_rect = hits[1].0;
    let downloads_rect = hits[2].0;
    app.tab_hits.set(hits);
    app.on_click(models_rect.x + 1, models_rect.y);
    assert_eq!(app.screen, Screen::Models);
    app.on_click(downloads_rect.x + 1, downloads_rect.y);
    assert_eq!(app.screen, Screen::Downloads);
}

#[test]
fn tab_bar_reports_server_state() {
    let app = new_app();
    // Tab bar now shows compact status: "○ stopped".
    assert!(render(&app).contains("stopped"));
}

#[test]
fn toasts_render_and_expire() {
    let mut app = new_app();
    app.toast("hello toast");
    assert!(render(&app).contains("hello toast"));
    app.toasts[0].at = std::time::Instant::now() - std::time::Duration::from_secs(10);
    super::super::widgets::expire_toasts(&mut app.toasts);
    assert!(app.toasts.is_empty());
}

#[test]
fn b_key_activates_banner_only_where_rendered() {
    // Home with a ready server: b runs the banner action -> Chat.
    let mut app = new_app();
    app.server_ready = true;
    app.on_key(key(KeyCode::Char('b')));
    assert_eq!(app.screen, Screen::Chat);

    // Models renders no journey banner: b falls through to the wizard and
    // must not navigate.
    let mut app = new_app();
    app.screen = Screen::Models;
    app.server_ready = true;
    app.on_key(key(KeyCode::Char('b')));
    assert_eq!(app.screen, Screen::Models);
    assert_eq!(app.stage, WizardStage::Families);
}

#[test]
fn tab_bar_fallthrough_unfocuses_without_double_action() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.stage = WizardStage::Precision;
    // Make the row deletable so a leaked `x` would open the delete modal.
    app.families[app.family_idx].variants[app.precision_idx].installed = true;
    app.focus_tab_bar();
    assert!(app.focus_tabs);
    app.on_key(key(KeyCode::Char('x')));
    assert!(!app.focus_tabs, "any other key leaves the bar");
    assert!(
        app.modal.is_none(),
        "x is consumed by the bar, not also opening the delete modal"
    );
}

#[test]
fn mouse_events_are_swallowed_while_help_is_open() {
    let mut app = new_app();
    let _ = render(&app); // records tab hit rects
    let hits = app.tab_hits.take();
    let models_rect = hits[1].0;
    app.tab_hits.set(hits);
    app.show_help = true;
    // Left-click closes help but must not leak through to the tab bar.
    app.on_mouse(mouse(
        MouseEventKind::Down(MouseButton::Left),
        models_rect.x + 1,
        models_rect.y,
    ));
    assert!(!app.show_help, "left-click closes help");
    assert_eq!(app.screen, Screen::Home, "click must not switch screens");
    // Scrolls are swallowed entirely: help stays open, selection unmoved.
    app.show_help = true;
    app.on_mouse(mouse(MouseEventKind::ScrollDown, 0, 0));
    assert!(app.show_help, "scroll does not close help");
    assert_eq!(app.home_idx, 0, "scroll must not reach the screen below");
}

#[test]
fn all_screens_render_on_tiny_geometry() {
    for (w, h) in [(40u16, 10u16), (24, 8), (0, 0)] {
        for screen in [
            Screen::Home,
            Screen::Models,
            Screen::Downloads,
            Screen::Serve,
            Screen::Chat,
        ] {
            let mut app = new_app();
            app.screen = screen;
            let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
            terminal
                .draw(|frame| app.draw(frame))
                .unwrap_or_else(|err| panic!("{screen:?} at {w}x{h}: {err}"));
        }
    }
}

// ---------------------------------------------------------------------------
// Home
// ---------------------------------------------------------------------------

#[test]
fn quick_start_targets_smallest_fitting_model() {
    let mut app = new_app();
    // Installed variants report real on-disk bytes from this machine's HF
    // cache, which would make the pick depend on what the developer has
    // downloaded; clear install state so only static catalog estimates count.
    for family in &mut app.families {
        for variant in &mut family.variants {
            variant.installed = false;
            variant.size = 0;
        }
    }
    let (fi, vi) = app.quick_start_target().expect("catalog is not empty");
    let family = &app.families[fi];
    // Smallest fitting chat model: the plain 4-bit Qwen 3.5 9B pack —
    // embedding and diffusion families never win the chat quick start.
    assert_eq!(family.key, "ax-qwen3.5-9b");
    assert_eq!(family.variants[vi].bits, Some(4));
    assert_eq!(family.variants[vi].size_estimate(), Some(6_463_848_363));
}
