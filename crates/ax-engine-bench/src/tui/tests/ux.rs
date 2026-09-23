//! Catalog refresh and user-facing rendering contracts.
use super::super::catalog::{CatalogRefresh, build_families_from_repo_ids_uninstalled};
use super::super::{Screen, WizardStage};
use super::{Terminal, TestBackend, key, new_app, render, render_sized, test_task};
use ratatui::crossterm::event::KeyCode;

fn catalog(ids: &[&str]) -> Vec<super::super::catalog::Family> {
    build_families_from_repo_ids_uninstalled(&ids.iter().map(|id| (*id).into()).collect::<Vec<_>>())
}

#[test]
fn refresh_preserves_repository_selection_and_filter_after_reordering() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.families = catalog(&["AutomatosX/Zeta-MLX-4bit"]);
    app.filter = "zeta".into();
    let (tx, rx) = std::sync::mpsc::channel();
    app.families_reload = Some(rx);
    app.catalog_loading = true;
    tx.send(CatalogRefresh::Hub {
        ids: vec![
            "AutomatosX/Alpha-MLX-4bit".into(),
            "AutomatosX/Zeta-MLX-4bit".into(),
        ],
        families: catalog(&["AutomatosX/Alpha-MLX-4bit", "AutomatosX/Zeta-MLX-4bit"]),
    })
    .unwrap();
    assert!(app.tick_families_reload());
    assert_eq!(
        app.families[app.family_idx].variants[0].model.repo_id,
        "AutomatosX/Zeta-MLX-4bit"
    );
    assert_eq!(app.filter, "zeta");
    assert!(!app.catalog_loading);
}

#[test]
fn refresh_failure_keeps_previous_catalog_and_persistent_recovery_hint() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.families = catalog(&["AutomatosX/Zeta-MLX-4bit"]);
    app.hub_repo_ids = Some(vec!["AutomatosX/Zeta-MLX-4bit".into()]);
    let (tx, rx) = std::sync::mpsc::channel();
    app.families_reload = Some(rx);
    tx.send(CatalogRefresh::Fallback {
        error: "HTTP 503".into(),
        families: vec![],
    })
    .unwrap();
    assert!(app.tick_families_reload());
    app.toasts.clear();
    assert_eq!(app.families.len(), 1);
    let text = render(&app);
    assert!(text.contains("previous list"));
    assert!(text.contains("R retry"));
}

#[test]
fn first_fetch_failure_does_not_insert_hardcoded_models() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.families.clear();
    let (tx, rx) = std::sync::mpsc::channel();
    app.families_reload = Some(rx);
    tx.send(CatalogRefresh::Fallback {
        error: "offline".into(),
        families: vec![],
    })
    .unwrap();
    assert!(app.tick_families_reload());
    app.toasts.clear();
    assert!(app.families.is_empty());
    let text = render(&app);
    assert!(text.contains("local models in Downloads"));
    assert!(text.contains("R retry"));
}

#[test]
fn refresh_waits_until_confirmation_has_finished() {
    let mut app = new_app();
    app.begin_confirm();
    let original_count = app.families.len();
    let (tx, rx) = std::sync::mpsc::channel();
    app.families_reload = Some(rx);
    tx.send(CatalogRefresh::Hub {
        ids: vec![],
        families: vec![],
    })
    .unwrap();
    assert!(!app.tick_families_reload());
    assert_eq!(app.families.len(), original_count);
    assert!(render(&app).contains("Download summary"));
    app.stage = WizardStage::Families;
    assert!(app.tick_families_reload());
    assert!(app.families.is_empty());
}

#[test]
fn refresh_key_does_not_replace_an_inflight_request_or_filter_text() {
    let mut app = new_app();
    app.screen = Screen::Models;
    let (tx, rx) = std::sync::mpsc::channel();
    app.families_reload = Some(rx);
    app.on_key(key(KeyCode::Char('R')));
    assert!(!app.families_reload_again);
    tx.send(CatalogRefresh::Hub {
        ids: vec![],
        families: vec![],
    })
    .unwrap();
    assert!(app.tick_families_reload());
    app.families = catalog(&["AutomatosX/Zeta-MLX-4bit"]);
    app.filtering = true;
    app.on_key(key(KeyCode::Char('R')));
    assert_eq!(app.filter, "R");
    assert!(app.families_reload.is_none());
}

#[test]
fn fixed_model_header_is_not_clickable_and_first_row_still_is() {
    let mut app = new_app();
    app.screen = Screen::Models;
    let text = render(&app);
    assert!(text.contains("Bits"));
    assert!(text.contains("Local"));
    let rect = app.content_list_rect.get();
    app.on_click(rect.x + 3, rect.y);
    assert_eq!(app.stage, WizardStage::Families);
    app.on_click(rect.x + 3, rect.y + 1);
    assert_eq!(app.stage, WizardStage::Families);
}

#[test]
fn narrow_wizard_remains_navigable_and_version_stays_at_bottom_right() {
    let mut app = new_app();
    app.screen = Screen::Models;
    for (width, height) in [(60, 15), (80, 24), (160, 40)] {
        app.stage = WizardStage::Families;
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|frame| app.draw(frame)).unwrap();
        let footer: String = (0..width)
            .map(|x| terminal.backend().buffer()[(x, height - 1)].symbol())
            .collect();
        assert!(footer.ends_with(concat!("v", env!("CARGO_PKG_VERSION"))));
        assert!(render_sized(&app, width, height).contains("R refresh"));
        app.on_key_models(KeyCode::Enter);
        assert!(render_sized(&app, width, height).contains("Step 2 of 3"));
        app.on_key_models(KeyCode::Enter);
        assert!(render_sized(&app, width, height).contains("Repository"));
    }
}

#[test]
fn unknown_hub_repo_has_no_size_fit_or_certification_claim() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.families = catalog(&["AutomatosX/Qwen3.5-New-MLX-4bit-MTP"]);
    let text = render(&app);
    assert!(text.contains("unverified"));
    assert!(!text.contains("certified"));
    assert!(!text.contains("speed-up"));
    assert!(!text.contains(" fits"));
    assert_eq!(app.families[0].variants[0].model.approx_size_bytes, None);
}

#[test]
fn failed_download_defaults_to_short_error_with_full_log_available() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    let mut job = super::super::jobs::Job::exited(2);
    job.log = vec![
        serde_json::json!({
            "schema_version": "ax.download_model.v1",
            "errors": ["Unknown model target\nFull usage and catalog follow"],
        })
        .to_string(),
    ];
    app.downloads.push(test_task(Some(job)));
    let text = render(&app);
    assert!(text.contains("Unknown model target"));
    assert!(!text.contains("Full usage and catalog"));
    assert!(text.contains("v full log"));
    app.on_key(key(KeyCode::Char('v')));
    assert!(render(&app).contains("Full usage and catalog"));
}

fn click_action(app: &mut super::super::App, action: super::super::ToolbarAction) {
    let _ = render(app);
    let hits = app.toolbar_hits.take();
    let rect = hits
        .iter()
        .find(|(_, candidate)| *candidate == action)
        .unwrap()
        .0;
    app.toolbar_hits.set(hits);
    app.on_mouse(super::mouse(
        ratatui::crossterm::event::MouseEventKind::Down(
            ratatui::crossterm::event::MouseButton::Left,
        ),
        rect.x + 1,
        rect.y,
    ));
}

#[test]
fn mouse_can_enter_download_confirmation_and_back_out_without_escape() {
    use super::super::ToolbarAction;
    let mut app = new_app();
    app.screen = Screen::Models;
    click_action(&mut app, ToolbarAction::Download);
    assert_eq!(app.stage, WizardStage::Precision);
    click_action(&mut app, ToolbarAction::Download);
    assert_eq!(app.stage, WizardStage::Confirm);
    click_action(&mut app, ToolbarAction::Back);
    assert_eq!(app.stage, WizardStage::Precision);
    click_action(&mut app, ToolbarAction::Back);
    assert_eq!(app.stage, WizardStage::Families);
    assert!(app.downloads.is_empty());
}

#[test]
fn local_library_scans_all_publishers_and_delete_is_repository_scoped() {
    use super::super::catalog::{delete_local_cache, scan_local_models};
    let root = std::env::temp_dir().join(format!("ax-tui-library-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (repository, revisions) in [
        ("models--community--Independent", 2),
        ("models--AutomatosX--Unlisted", 1),
    ] {
        for revision in 0..revisions {
            let path = root
                .join(repository)
                .join("snapshots")
                .join(format!("rev-{revision}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("config.json"), "{}").unwrap();
            std::fs::write(path.join("model.safetensors"), "test weights").unwrap();
        }
    }
    let partial = root.join("models--another--Partial/snapshots/rev");
    std::fs::create_dir_all(&partial).unwrap();
    std::fs::write(partial.join("config.json"), "{}").unwrap();
    let models = scan_local_models(&root).unwrap();
    assert_eq!(models.len(), 3);
    let community = models
        .iter()
        .find(|model| model.repo_id == "community/Independent")
        .unwrap();
    assert!(community.ready);
    assert_eq!(community.revisions, 2);
    assert!(
        !models
            .iter()
            .find(|model| model.repo_id == "another/Partial")
            .unwrap()
            .ready
    );
    let mut forged = community.clone();
    forged.cache_dir = root.clone();
    assert!(delete_local_cache(&forged, &root).is_err());
    delete_local_cache(community, &root).unwrap();
    assert!(!community.cache_dir.exists());
    assert!(root.join("models--AutomatosX--Unlisted").is_dir());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn mouse_can_manage_non_catalog_local_models_and_cancel_deletion() {
    use super::super::{Modal, ToolbarAction, catalog::LocalModel};
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads_show_library = true;
    app.families.clear();
    let path =
        std::path::PathBuf::from("/tmp/test-hub/models--community--Independent/snapshots/rev");
    app.local_models.push(LocalModel {
        repo_id: "community/Independent".into(),
        snapshot: path.clone(),
        cache_dir: path.parent().unwrap().parent().unwrap().into(),
        size: 1024,
        revisions: 1,
        ready: true,
    });
    assert!(render(&app).contains("community/Independent"));
    click_action(&mut app, ToolbarAction::Serve);
    assert!(matches!(&app.modal, Some(Modal::ServeLocal(model)) if model.snapshot == path));
    let _ = render(&app);
    let cancel = app.modal_hits.get().cancel.unwrap();
    app.on_mouse(super::mouse(
        ratatui::crossterm::event::MouseEventKind::Down(
            ratatui::crossterm::event::MouseButton::Left,
        ),
        cancel.x + 1,
        cancel.y,
    ));
    assert!(app.modal.is_none());
    assert!(app.server.is_none());
    click_action(&mut app, ToolbarAction::Delete);
    assert!(matches!(app.modal, Some(Modal::DeleteLocal(_))));
    let text = render_sized(&app, 60, 15);
    assert!(text.contains("Delete files"));
    assert!(text.contains("Cancel"));
    let cancel = app.modal_hits.get().cancel.unwrap();
    app.on_mouse(super::mouse(
        ratatui::crossterm::event::MouseEventKind::Down(
            ratatui::crossterm::event::MouseButton::Left,
        ),
        cancel.x + 1,
        cancel.y,
    ));
    assert!(app.modal.is_none());
    assert_eq!(app.local_models.len(), 1);
    click_action(&mut app, ToolbarAction::Transfers);
    assert!(!app.downloads_show_library);
    click_action(&mut app, ToolbarAction::Library);
    assert!(app.downloads_show_library);
}

#[cfg(unix)]
#[test]
fn local_scan_and_delete_reject_repository_symlinks() {
    use super::super::catalog::{LocalModel, delete_local_cache, scan_local_models};
    let root = std::env::temp_dir().join(format!("ax-tui-symlink-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let outside = root.join("outside");
    std::fs::create_dir_all(outside.join("snapshots/rev")).unwrap();
    std::fs::write(outside.join("keep.txt"), "preserve").unwrap();
    let cache_dir = root.join("models--community--Link");
    std::os::unix::fs::symlink(&outside, &cache_dir).unwrap();
    assert!(scan_local_models(&root).unwrap().is_empty());
    let model = LocalModel {
        repo_id: "community/Link".into(),
        snapshot: cache_dir.join("snapshots/rev"),
        cache_dir,
        size: 0,
        revisions: 1,
        ready: false,
    };
    assert!(delete_local_cache(&model, &root).is_err());
    assert!(outside.join("keep.txt").is_file());
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn local_scan_errors_preserve_previous_list_and_are_visible() {
    let mut app = new_app();
    app.screen = Screen::Downloads;
    app.downloads_show_library = true;
    let (tx, rx) = std::sync::mpsc::channel();
    app.local_models_reload = Some(rx);
    app.local_models_loading = true;
    tx.send(Err("Cannot read snapshot cache: permission denied".into()))
        .unwrap();
    assert!(app.tick_local_models());
    assert!(!app.local_models_loading);
    assert!(render(&app).contains("permission denied"));
}

#[test]
fn delete_guard_distinguishes_managed_paths_and_unknown_external_servers() {
    use super::super::{catalog::LocalModel, jobs::Job};
    let mut app = new_app();
    let model = LocalModel {
        repo_id: "community/Independent".into(),
        cache_dir: "/tmp/cache/models--community--Independent".into(),
        snapshot: "/tmp/cache/models--community--Independent/snapshots/rev".into(),
        size: 0,
        revisions: 1,
        ready: true,
    };
    app.server = Some(Job::running_with_log(vec![]));
    app.server_artifacts_dir = Some("/tmp/cache/models--other--Repo/snapshots/rev".into());
    assert!(!app.local_model_in_use(&model));
    app.server_artifacts_dir = Some(model.snapshot.clone());
    assert!(app.local_model_in_use(&model));
    app.server_artifacts_dir = Some("/tmp/cache/models--other--Repo/snapshots/rev".into());
    app.external_server = true;
    assert!(app.local_model_in_use(&model));
}
