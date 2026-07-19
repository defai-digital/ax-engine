//! TUI state-machine and rendering tests. Test-only module: unwrap/expect/panic
//! are fine per the workspace lint convention (see [workspace.lints.clippy]).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::catalog::{self, RamFit, build_families, family_key, most_recent_subdir, quant_bits};
use super::hardware::{HardwareInfo, parse_df_available_kib};
use super::jobs::{
    DownloadMode, DownloadStatus, DownloadTask, Job, format_eta, parse_output_path_from_log,
    parse_progress_event,
};
use super::metrics::{
    parse_loadavg_1m, parse_ps_cpu_percent, parse_ps_top_rss, parse_vm_stat_free_bytes,
    parse_vm_stat_used_bytes,
};
use super::screens::chat::{
    ChatMessage, ReplyStats, SseEvent, count_visual_lines, parse_sse_line, split_thinking,
};
use super::{App, Modal, Screen, ServeFocus, WizardStage};

use std::path::{Path, PathBuf};
use std::process;
use std::time::Duration;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

fn new_app() -> App {
    App::with_hardware(HardwareInfo::for_tests())
}

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        modifiers: KeyModifiers::empty(),
        kind: KeyEventKind::Press,
        state: ratatui::crossterm::event::KeyEventState::empty(),
    }
}

fn ctrl_key(c: char) -> KeyEvent {
    KeyEvent {
        code: KeyCode::Char(c),
        modifiers: KeyModifiers::CONTROL,
        kind: KeyEventKind::Press,
        state: ratatui::crossterm::event::KeyEventState::empty(),
    }
}

fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::empty(),
    }
}

/// Render the app to an off-screen buffer and flatten it to text.
fn render(app: &App) -> String {
    render_sized(app, 120, 40)
}

fn render_sized(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    terminal
        .backend()
        .buffer()
        .content
        .iter()
        .map(|cell| cell.symbol())
        .collect()
}

fn family_index(app: &App, key: &str) -> usize {
    app.families.iter().position(|f| f.key == key).unwrap()
}

fn test_task(job: Option<Job>) -> DownloadTask {
    DownloadTask {
        label: "gemma4-e2b 4-bit".into(),
        repo_id: "mlx-community/gemma-4-e2b-it-4bit",
        preset: Some("gemma4-e2b"),
        mode: DownloadMode::Direct,
        subcmd: "download",
        target: "gemma4-e2b".into(),
        dest: Some(PathBuf::from("/tmp/gemma4-e2b")),
        watch_dir: PathBuf::from("/tmp/gemma4-e2b"),
        resolved_path: None,
        total_bytes: Some(3_583_088_661),
        phase: None,
        job,
        cancelled: false,
    }
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

#[test]
fn grouping_collapses_variants_into_families() {
    let families = build_families();
    let keys: Vec<&str> = families.iter().map(|f| f.key.as_str()).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        keys.len(),
        "family keys must be unique: {keys:?}"
    );
    let flat = crate::MODEL_PROFILES
        .iter()
        .filter(|p| p.downloadable)
        .count();
    assert!(
        families.len() < flat,
        "{} families from {flat} profiles",
        families.len()
    );

    let e2b = families.iter().find(|f| f.key == "gemma4-e2b").unwrap();
    assert_eq!(e2b.variants.len(), 4); // 4/5/6/8-bit
    assert!(!e2b.has_mtp());
    let g12 = families.iter().find(|f| f.key == "gemma4-12b").unwrap();
    assert!(g12.has_mtp());
    // Recommended (first) variant is the lowest bit-width.
    assert_eq!(g12.variants[0].bits, Some(4));
}

#[test]
fn quant_and_family_parsing() {
    assert_eq!(quant_bits("mlx-community/gemma-4-12B-it-4bit"), Some(4));
    assert_eq!(quant_bits("mlx-community/Qwen3.6-27B-8bit"), Some(8));
    assert_eq!(quant_bits("mlx-community/gpt-oss-20b-MXFP4-Q4"), Some(4));
    assert_eq!(quant_bits("mlx-community/gpt-oss-120b-MXFP4-Q4"), Some(4));
    assert_eq!(family_key("gemma4-e2b-8bit"), "gemma4-e2b");
    assert_eq!(family_key("glm4.7-flash-4bit"), "glm4.7-flash");
    assert_eq!(family_key("qwen3.6-35b"), "qwen3.6-35b");
    assert_eq!(family_key("gpt-oss-20b"), "gpt-oss-20b");
}

#[test]
fn secondary_and_gpt_oss_profiles_are_downloadable() {
    let families = build_families();
    for key in [
        "qwen3.5-9b",
        "glm4.7-flash",
        "llama3.1-8b",
        "llama3.3-70b",
        "llama4-scout",
        "mistral-small",
        "ministral-8b",
        "devstral-small",
        "gpt-oss-20b",
        "gpt-oss-120b",
    ] {
        let family = families
            .iter()
            .find(|f| f.key == key)
            .unwrap_or_else(|| panic!("missing TUI family {key}"));
        assert!(
            !family.variants.is_empty(),
            "{key} must expose at least one downloadable variant"
        );
        assert!(
            family.variants.iter().all(|v| v.profile.downloadable),
            "{key} variants must be downloadable"
        );
    }
    let gpt20 = families.iter().find(|f| f.key == "gpt-oss-20b").unwrap();
    assert!(!gpt20.is_primary());
    assert_eq!(gpt20.variants[0].precision(), "MXFP4-Q4");
    assert!(gpt20.variants[0].profile.approx_size_bytes.is_some());
}

#[test]
fn every_downloadable_profile_has_a_size_estimate() {
    for profile in crate::MODEL_PROFILES.iter().filter(|p| p.downloadable) {
        assert!(
            profile.approx_size_bytes.is_some(),
            "{} is downloadable but has no approx_size_bytes",
            profile.label
        );
    }
}

#[test]
fn ram_fit_thresholds() {
    let gib = 1024u64 * 1024 * 1024;
    // 3 GB model on 64 GB RAM: comfortable.
    assert_eq!(
        catalog::ram_fit(Some(3 * gib), Some(64 * gib)),
        RamFit::Fits
    );
    // 40 GB model on 64 GB RAM: footprint ~49.5/64 = 77% -> tight.
    assert_eq!(
        catalog::ram_fit(Some(40 * gib), Some(64 * gib)),
        RamFit::Tight
    );
    // 60 GB model on 64 GB RAM: too large.
    assert_eq!(
        catalog::ram_fit(Some(60 * gib), Some(64 * gib)),
        RamFit::TooLarge
    );
    assert_eq!(catalog::ram_fit(None, Some(64 * gib)), RamFit::Unknown);
    assert_eq!(catalog::ram_fit(Some(gib), None), RamFit::Unknown);
}

#[test]
fn most_recent_subdir_is_none_for_missing_dir() {
    assert_eq!(
        most_recent_subdir(Path::new("/definitely/does/not/exist")),
        None
    );
}

#[test]
fn most_recent_subdir_picks_the_only_entry() {
    let base = std::env::temp_dir().join(format!("ax-engine-tui-test-{}", process::id()));
    let snapshots = base.join("snapshots");
    let snapshot = snapshots.join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    assert_eq!(most_recent_subdir(&snapshots), Some(snapshot));
    std::fs::remove_dir_all(&base).unwrap();
}

// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

#[test]
fn df_available_column_parses() {
    let output = "\
Filesystem   1024-blocks       Used Available Capacity iused ifree %iused  Mounted on
/dev/disk3s5  971350180  530692820 419382948    56%  915272 4193829480    0%   /System/Volumes/Data
";
    assert_eq!(parse_df_available_kib(output), Some(419_382_948));
    assert_eq!(parse_df_available_kib("garbage"), None);
}

// ---------------------------------------------------------------------------
// Jobs / downloads
// ---------------------------------------------------------------------------

#[test]
fn progress_event_lines_parse() {
    assert_eq!(
        parse_progress_event(r#"{"event":"progress","done":85,"total":100,"file":"snapshot"}"#),
        Some((85, 100, "snapshot".into()))
    );
    assert_eq!(parse_progress_event(r#"{"event":"other"}"#), None);
    assert_eq!(parse_progress_event("plain text"), None);
    assert_eq!(
        parse_progress_event(r#"{"schema_version":"ax.download_model.v1"}"#),
        None
    );
}

#[test]
fn eta_formatting() {
    assert_eq!(format_eta(42), "42s");
    assert_eq!(format_eta(90), "1m30s");
    assert_eq!(format_eta(3720), "1h02m");
}

#[test]
fn queued_download_can_be_cancelled_before_spawn() {
    let mut task = test_task(None);
    task.dest = None;
    assert_eq!(task.status_label(), "queued");
    assert!(task.is_queued());
    task.cancel();
    assert_eq!(task.status_label(), "cancelled");
    assert!(!task.is_queued());
}

#[test]
fn progress_ratio_and_eta_need_totals() {
    let mut task = test_task(None);
    task.total_bytes = None;
    assert_eq!(task.progress_ratio(), None);
    assert_eq!(task.eta_seconds(), None);
    task.total_bytes = Some(100);
    // No job yet: 0 bytes downloaded.
    assert_eq!(task.progress_ratio(), Some(0.0));
}

#[test]
fn log_parser_finds_download_output_paths() {
    assert_eq!(
        parse_output_path_from_log(&["Path: /tmp/direct".to_string()]).as_deref(),
        Some(Path::new("/tmp/direct"))
    );
    assert_eq!(
        parse_output_path_from_log(&["Output dir: /tmp/mtp".to_string()]).as_deref(),
        Some(Path::new("/tmp/mtp"))
    );
    assert_eq!(
        parse_output_path_from_log(&["Output dir:      /tmp/gemma-mtp".to_string()]).as_deref(),
        Some(Path::new("/tmp/gemma-mtp"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Sidecar ready at:".to_string(),
            "  /tmp/sidecar".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/sidecar"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Gemma 4 assistant MTP package ready at:".to_string(),
            "  /tmp/gemma-package".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/gemma-package"))
    );
    assert_eq!(
        parse_output_path_from_log(&[
            "Next:".to_string(),
            "  ax-engine serve /tmp/from-next --port 31418".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/from-next"))
    );
}

#[test]
fn artifact_dir_usable_requires_real_model_files() {
    let root = std::env::temp_dir().join(format!("ax-tui-artifact-usable-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("mkdir root");

    let empty = root.join("empty");
    std::fs::create_dir_all(&empty).expect("mkdir");
    assert!(!catalog::artifact_dir_usable(&empty));

    let with_config = root.join("cfg");
    std::fs::create_dir_all(&with_config).expect("mkdir");
    std::fs::write(with_config.join("config.json"), "{}").expect("write");
    assert!(catalog::artifact_dir_usable(&with_config));

    let with_weight = root.join("wt");
    std::fs::create_dir_all(&with_weight).expect("mkdir");
    std::fs::write(with_weight.join("model.safetensors"), b"x").expect("write");
    assert!(catalog::artifact_dir_usable(&with_weight));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn mtp_serve_without_package_path_fails_closed() {
    let mut app = new_app();
    let mut task = test_task(Some(Job::failed("ok".into())));
    task.mode = DownloadMode::Mtp;
    task.subcmd = "download-mtp";
    task.dest = None;
    task.resolved_path = None;
    if let Some(job) = &mut task.job {
        job.done = Some(0);
        job.log.clear(); // no Path: / Output dir: lines
    }
    app.downloads.push(task);
    app.start_server_for_download(0);
    let job = app.server.expect("server job should be set");
    assert_eq!(
        job.done,
        Some(-1),
        "must not spawn real server without path"
    );
    assert!(
        job.log.iter().any(|line| line.contains("MTP package path")),
        "error should mention MTP package path: {:?}",
        job.log
    );
}

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
    use super::catalog::installed_variants;
    use super::screens::home::HomeAction;

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
    app.live_metrics = super::metrics::LiveMetrics::for_tests();
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
    app.live_metrics = super::metrics::LiveMetrics::for_tests();
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
    use super::metrics::parse_ioreg_gpu;
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
        super::theme::colors().accent,
        ratatui::style::Color::Rgb(56, 189, 248)
    );
    assert_eq!(super::theme::icon::ok(), "✓");
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
    super::widgets::expire_toasts(&mut app.toasts);
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
    assert_eq!(family.key, "gemma4-e2b");
    assert_eq!(family.variants[vi].bits, Some(4));
}

// ---------------------------------------------------------------------------
// Models wizard
// ---------------------------------------------------------------------------

#[test]
fn family_list_renders_with_sizes_and_mtp_badge() {
    let mut app = new_app();
    app.screen = Screen::Models;
    let text = render(&app);
    assert!(text.contains("Models"));
    assert!(text.contains("Gemma 4 E2B"));
    assert!(text.contains("Qwen 3.6 35B"));
    assert!(text.contains("bit"), "family rows show quant bits");
    assert!(text.contains('⚡'), "MTP badge should render");
    assert!(text.contains("Step 1 of"), "step header present");
}

#[test]
fn precision_screen_lists_quants_with_fit_badges() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.stage, WizardStage::Precision);
    let text = render(&app);
    assert!(text.contains("size"));
    assert!(text.contains("4-bit"));
    assert!(text.contains("6-bit"));
    assert!(
        text.contains("fits"),
        "fit badge rendered for 64GB test RAM"
    );
    // Recommended star only marks the Quick start pick (smallest fit), not
    // every family's first variant — gemma4-12b is not Quick start.
    assert!(
        text.contains("Step 2 of 4"),
        "gemma4-12b has a speed-up step"
    );
    assert!(text.contains("Speed-up") || text.contains("Size"));
}

#[test]
fn mtp_options_step_appears_only_for_mtp_variants() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.on_key_models(KeyCode::Enter); // -> Precision
    app.precision_idx = 0; // 4-bit has an MTP accelerator
    app.on_key_models(KeyCode::Enter); // -> Options
    assert_eq!(app.stage, WizardStage::Options);
    let text = render(&app);
    assert!(text.contains("Optional speed-up"));
    assert!(text.contains("Include the speed-up"));
    assert!(text.contains("Step 3 of 4"));

    let app2 = new_app();
    let e2b = family_index(&app2, "gemma4-e2b");
    assert!(app2.families[e2b].variants[0].mtp_alias.is_none());
}

#[test]
fn confirm_step_shows_summary_and_default_destination() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-e2b");
    app.precision_idx = 0;
    app.begin_confirm(false);
    assert_eq!(app.stage, WizardStage::Confirm);
    let text = render(&app);
    assert!(text.contains("Confirm download"));
    assert!(text.contains("Gemma 4 E2B"));
    assert!(text.contains("default cache"));
    assert!(text.contains("Free disk"));
    assert!(
        text.contains("Step 3 of 3"),
        "no speed-up step for gemma4-e2b"
    );
}

#[test]
fn confirm_enqueues_and_jumps_to_downloads() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-e2b");
    app.precision_idx = 0;
    app.begin_confirm(false);
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.screen, Screen::Downloads);
    assert_eq!(app.downloads.len(), 1);
    let task = &app.downloads[0];
    assert_eq!(task.subcmd, "download");
    assert!(task.dest.is_none(), "default destination is the HF cache");
    assert!(!app.toasts.is_empty(), "queueing raises a toast");
}

#[test]
fn mtp_confirm_uses_mtp_target_repo_and_combined_size() {
    let mut app = new_app();
    app.screen = Screen::Models;
    // qwen3.6-27b 4-bit maps to the 6-bit MTP target — the plan must follow
    // the target repo, not the selected variant's repo.
    app.family_idx = family_index(&app, "qwen3.6-27b");
    app.precision_idx = 0;
    app.begin_confirm(true);
    app.on_key_models(KeyCode::Enter);
    let task = &app.downloads[0];
    assert_eq!(task.subcmd, "download-mtp");
    assert_eq!(task.repo_id, "mlx-community/Qwen3.6-27B-6bit");
    assert_eq!(task.total_bytes, Some(22_804_828_230 + 4_503_752_416));
}

#[test]
fn click_on_family_row_drills_into_precision() {
    let mut app = new_app();
    app.screen = Screen::Models;
    let _ = render(&app); // records content_list_rect for the families list
    let rect = app.content_list_rect.get();
    assert!(rect.height >= 2, "list rect should be recorded");
    app.on_click(rect.x + 2, rect.y + 2);
    assert_eq!(app.stage, WizardStage::Precision);
    assert_eq!(app.family_idx, 1);
}

#[test]
fn click_on_completed_step_header_navigates_back() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.on_key_models(KeyCode::Enter);
    app.precision_idx = 0;
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.stage, WizardStage::Options);

    let _ = render(&app);
    let rect = app.step_header_rect.get();
    let model_offset = "Step 3 of 4 — ".chars().count();
    app.on_click(rect.x + model_offset as u16, rect.y);

    assert_eq!(app.stage, WizardStage::Families);
}

#[test]
fn scroll_moves_family_selection() {
    let mut app = new_app();
    app.screen = Screen::Models;
    assert_eq!(app.family_idx, 0);
    app.on_mouse(mouse(MouseEventKind::ScrollDown, 0, 0));
    assert_eq!(app.family_idx, 1);
    app.on_mouse(mouse(MouseEventKind::ScrollUp, 0, 0));
    assert_eq!(app.family_idx, 0);
}

#[test]
fn filter_narrows_family_list_and_drill_in_maps_back() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.filter = "gemma4-12b".to_string();
    app.clamp_family_idx_to_filter();
    let indices = app.filtered_family_indices();
    assert_eq!(indices.len(), 1);
    assert_eq!(app.families[indices[0]].key, "gemma4-12b");
    assert_eq!(app.family_idx, indices[0]);

    let text = render(&app);
    assert!(text.contains("filter: gemma4-12b"));
    assert!(
        !text.contains("Gemma 4 E2B") && !text.contains("gemma4-e2b"),
        "non-matching family should be hidden"
    );

    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.stage, WizardStage::Precision);
    assert_eq!(app.families[app.family_idx].key, "gemma4-12b");
}

#[test]
fn filter_mode_arrows_move_selection_without_leaving_filter() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.filtering = true;
    assert_eq!(app.family_idx, 0);
    app.on_key(key(KeyCode::Down));
    assert!(app.filtering, "Down stays in filter mode");
    assert_eq!(app.family_idx, 1);
    app.on_key(key(KeyCode::Up));
    assert!(app.filtering, "Up stays in filter mode");
    assert_eq!(app.family_idx, 0);
}

#[test]
fn paste_in_filter_mode_appends_and_clamps_selection() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.filtering = true;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.on_paste("gemma4-e2b");
    assert_eq!(app.filter, "gemma4-e2b");
    assert_eq!(
        app.families[app.family_idx].key, "gemma4-e2b",
        "selection snaps back into the filtered set"
    );
}

#[test]
fn mtp_badge_is_magenta_not_yellow() {
    // Leave the default selection (family_idx 0) alone: the selected row's
    // own highlight style overrides span colors, so check the *other* MTP
    // families' badges instead — several exist in the catalog.
    let mut app = new_app();
    app.screen = Screen::Models;
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    let colors: Vec<ratatui::style::Color> = terminal
        .backend()
        .buffer()
        .content
        .iter()
        .filter(|cell| cell.symbol() == "⚡")
        .map(|cell| cell.fg)
        .collect();
    assert!(!colors.is_empty(), "MTP badge glyph should render");
    assert!(
        colors.contains(&ratatui::style::Color::Magenta),
        "at least one non-selected MTP badge should be magenta: {colors:?}"
    );
    assert!(
        !colors.contains(&ratatui::style::Color::Yellow),
        "MTP badge must not reuse the queued-status yellow: {colors:?}"
    );
}

#[test]
fn right_key_advances_through_wizard_steps() {
    let mut app = new_app();
    app.screen = Screen::Models;
    assert_eq!(app.stage, WizardStage::Families);
    // Right on Families advances to Precision.
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.stage, WizardStage::Precision);
    // Right on Precision advances to Options (gemma4-12b has MTP).
    app.family_idx = family_index(&app, "gemma4-12b");
    app.precision_idx = 0;
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.stage, WizardStage::Options);
    // Right on Options advances to Confirm.
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.stage, WizardStage::Confirm);
}

#[test]
fn left_key_steps_back_through_wizard() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.precision_idx = 0;
    app.stage = WizardStage::Confirm;
    app.pending = Some(super::PendingDownload {
        family_idx: app.family_idx,
        precision_idx: 0,
        with_mtp: true,
    });
    // Left on Confirm with MTP variant goes back to Options.
    app.on_key_models(KeyCode::Left);
    assert_eq!(app.stage, WizardStage::Options);
    // Left on Options goes back to Precision.
    app.on_key_models(KeyCode::Left);
    assert_eq!(app.stage, WizardStage::Precision);
    // Left on Precision goes back to Families.
    app.on_key_models(KeyCode::Left);
    assert_eq!(app.stage, WizardStage::Families);
    // Left on Families (with empty filter) goes Home.
    app.on_key_models(KeyCode::Left);
    assert_eq!(app.screen, Screen::Home);
}

#[test]
fn right_key_on_confirm_triggers_download() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-e2b");
    app.precision_idx = 0;
    app.begin_confirm(false);
    assert_eq!(app.stage, WizardStage::Confirm);
    // Right on Confirm confirms the download (same as Enter).
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.screen, Screen::Downloads);
    assert_eq!(app.downloads.len(), 1);
}

#[test]
fn click_on_options_selects_and_advances() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-12b");
    app.precision_idx = 0;
    app.stage = WizardStage::Options;
    // Click row 0 (Yes with MTP).
    app.on_click_models(0);
    assert_eq!(app.mtp_idx, 0);
    assert_eq!(app.stage, WizardStage::Confirm);
    assert!(app.pending.is_some_and(|p| p.with_mtp));

    // Reset and click row 1 (No MTP).
    app.stage = WizardStage::Options;
    app.pending = None;
    app.on_click_models(1);
    assert_eq!(app.mtp_idx, 1);
    assert_eq!(app.stage, WizardStage::Confirm);
    assert!(app.pending.is_some_and(|p| !p.with_mtp));
}

#[test]
fn click_on_precision_row_selects_and_advances() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "gemma4-e2b");
    app.stage = WizardStage::Precision;
    let _ = render(&app);
    let rect = app.content_list_rect.get();
    assert!(rect.height >= 2);
    // Click the first precision row (4-bit).
    app.on_click(rect.x + 2, rect.y + 1);
    assert_eq!(app.precision_idx, 0);
    // Should leave Precision: either Confirm (not installed) or a modal (installed).
    assert!(
        app.stage != WizardStage::Precision || app.modal.is_some(),
        "clicking a precision row must advance or open a modal"
    );
}

#[test]
fn modal_click_outside_dismisses_and_chips_act() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_url = Some("http://127.0.0.1:8080".into());
    app.modal = Some(Modal::StopServer);
    let _ = render(&app); // draw once so chip hit-rects get recorded
    // Click far outside the centered popup → dismisses like Esc.
    app.on_mouse(mouse(MouseEventKind::Down(MouseButton::Left), 0, 0));
    assert!(app.modal.is_none(), "click outside dismisses the modal");
    assert!(app.server_running(), "dismissal must not stop the server");
    // Reopen and click the confirm chip → same as pressing Enter.
    app.modal = Some(Modal::StopServer);
    let _ = render(&app);
    let confirm = app
        .modal_hits
        .get()
        .confirm
        .expect("stop modal has a confirm chip");
    app.on_mouse(mouse(
        MouseEventKind::Down(MouseButton::Left),
        confirm.x + 1,
        confirm.y,
    ));
    assert!(app.modal.is_none());
    assert!(
        app.toasts.iter().any(|t| t.text.contains("server stopped")),
        "confirm chip triggers the modal action: {:?}",
        app.toasts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn delete_modal_requires_typed_word() {
    let mut app = new_app();
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    // Enter with the wrong word keeps the modal open (and deletes nothing).
    app.on_key(key(KeyCode::Char('x')));
    app.on_key(key(KeyCode::Enter));
    assert!(matches!(app.modal, Some(Modal::DeleteModel { .. })));
    let text = render(&app);
    assert!(text.contains("Type 'delete' to confirm"));
    // Esc closes without deleting.
    app.on_key(key(KeyCode::Esc));
    assert!(app.modal.is_none());
}

#[test]
fn delete_modal_n_dismisses_only_when_nothing_typed() {
    let mut app = new_app();
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    // "no" dismisses like every other confirm modal while the confirm string
    // is still empty.
    app.on_key(key(KeyCode::Char('n')));
    assert!(app.modal.is_none());
    // Once typing has started, n is just another character of the word.
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    app.on_key(key(KeyCode::Char('d')));
    app.on_key(key(KeyCode::Char('n')));
    assert!(matches!(&app.modal, Some(Modal::DeleteModel { typed, .. }) if typed == "dn"));
}

#[test]
fn delete_modal_h_dismisses_only_when_nothing_typed() {
    let mut app = new_app();
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    // h (vim-left) dismisses like Esc/n while the confirm string is empty.
    app.on_key(key(KeyCode::Char('h')));
    assert!(app.modal.is_none());
    // Once typing has started, h joins the confirm word instead.
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    app.on_key(key(KeyCode::Char('x')));
    app.on_key(key(KeyCode::Char('h')));
    assert!(matches!(&app.modal, Some(Modal::DeleteModel { typed, .. }) if typed == "xh"));
}

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
            .any(|t| t.level == super::widgets::ToastLevel::Error
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
    assert!(crate::tui::server_log_indicates_ready(
        "2026-07-18T11:21:03.181885Z  INFO ax-engine-server preview listening bind_address=127.0.0.1:8080 model_id=gemma4-e2b"
    ));
    assert!(crate::tui::server_log_indicates_ready(
        "ax-engine-server preview listening on http://127.0.0.1:8080 model_id=gemma4-e2b"
    ));
    assert!(!crate::tui::server_log_indicates_ready("booting model..."));
    assert!(!crate::tui::server_log_indicates_ready(
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
            .any(|t| t.level == super::widgets::ToastLevel::Error
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
    assert!(render(&app).contains("No server running"));
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
        app.chat.messages.push(super::screens::chat::ChatMessage {
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
    app.chat.messages.push(super::screens::chat::ChatMessage {
        from_user: true,
        content: "What is AX?".into(),
        stats: None,
    });
    app.chat.messages.push(super::screens::chat::ChatMessage {
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

/// App with a "ready" server (points at the discard port; spawned curls fail
/// fast with connection-refused rather than talking to anything real).
fn chat_ready_app() -> App {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.server_ready = true;
    app.server_url = Some("http://127.0.0.1:9".into());
    app.screen = Screen::Chat;
    app
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
}

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

// ---------------------------------------------------------------------------
// Wave 5: serve switch-restart, thinking collapse, toast coalescing,
// chart legend/x-label, download status enum
// ---------------------------------------------------------------------------

/// Serve screen with a ready server on variant 0 of a family that also has
/// variant 1 "installed" (flag-only; nothing touches the disk or a process).
fn serve_app_with_two_installed() -> (App, usize) {
    let mut app = new_app();
    let fi = family_index(&app, "gemma4-e2b");
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
    let pairs = super::catalog::installed_variants(&app.families);
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
    let pairs = super::catalog::installed_variants(&app.families);
    app.serve_idx = pairs.iter().position(|&p| p == (fi, 0)).unwrap();
    app.on_key(key(KeyCode::Enter));
    assert!(app.modal.is_none(), "the served row keeps the chat handoff");
    assert_eq!(app.screen, Screen::Chat);
}

#[test]
fn serve_restart_confirm_stops_then_starts_selected_model() {
    let (mut app, fi) = serve_app_with_two_installed();
    let pairs = super::catalog::installed_variants(&app.families);
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
        app.auto_chat_after_serve,
        "restart re-arms the chat handoff like a fresh serve"
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
        let pairs = super::catalog::installed_variants(&app.families);
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
    let metrics = super::metrics::LiveMetrics::for_tests();
    let mut terminal = Terminal::new(TestBackend::new(100, 30)).unwrap();
    terminal
        .draw(|frame| {
            super::screens::metrics_panel::draw_live_metrics(frame, frame.area(), &metrics);
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
    let secs = super::metrics::HISTORY_LEN as u64 * super::metrics::SAMPLE_INTERVAL.as_secs();
    let expected = format!("~{}m", secs / 60);
    assert_eq!(super::screens::metrics_panel::window_label(), expected);
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
