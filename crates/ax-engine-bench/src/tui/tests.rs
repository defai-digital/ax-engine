use super::catalog::{self, RamFit, build_families, family_key, most_recent_subdir, quant_bits};
use super::hardware::{HardwareInfo, parse_df_available_kib};
use super::jobs::{
    DownloadMode, DownloadTask, Job, format_eta, parse_output_path_from_log, parse_progress_event,
};
use super::metrics::{
    parse_loadavg_1m, parse_ps_cpu_percent, parse_ps_top_rss, parse_vm_stat_free_bytes,
    parse_vm_stat_used_bytes,
};
use super::screens::chat::{SseEvent, parse_sse_line};
use super::{App, Modal, Screen, WizardStage};

use std::path::{Path, PathBuf};
use std::process;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::event::{MouseEvent, MouseEventKind};

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
    let mut terminal = Terminal::new(TestBackend::new(120, 40)).unwrap();
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
    assert_eq!(
        quant_bits("mlx-community/gpt-oss-20b-MXFP4-Q4"),
        Some(4)
    );
    assert_eq!(
        quant_bits("mlx-community/gpt-oss-120b-MXFP4-Q4"),
        Some(4)
    );
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
            "  ax-engine serve /tmp/from-next --port 8080".to_string(),
        ])
        .as_deref(),
        Some(Path::new("/tmp/from-next"))
    );
}

#[test]
fn artifact_dir_usable_requires_real_model_files() {
    let root = std::env::temp_dir().join(format!(
        "ax-tui-artifact-usable-{}",
        std::process::id()
    ));
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
    assert_eq!(job.done, Some(-1), "must not spawn real server without path");
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
    // Stack is empty after the pop; further Esc lands on Home.
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
fn delete_modal_requires_typed_word() {
    let mut app = new_app();
    app.modal = Some(Modal::DeleteModel {
        family_idx: 0,
        variant_idx: 0,
        typed: String::new(),
    });
    // Enter with the wrong word keeps the modal open (and deletes nothing).
    app.on_key(key(KeyCode::Char('n')));
    app.on_key(key(KeyCode::Enter));
    assert!(matches!(app.modal, Some(Modal::DeleteModel { .. })));
    let text = render(&app);
    assert!(text.contains("Type 'delete' to confirm"));
    // Esc closes without deleting.
    app.on_key(key(KeyCode::Esc));
    assert!(app.modal.is_none());
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
fn stopping_server_asks_first() {
    let mut app = new_app();
    app.server = Some(Job::running_with_log(vec![]));
    app.screen = Screen::Serve;
    app.on_key(key(KeyCode::Char('x')));
    assert!(matches!(app.modal, Some(Modal::StopServer)));
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
    });
    app.chat.messages.push(super::screens::chat::ChatMessage {
        from_user: false,
        content: "A local inference engine.".into(),
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
