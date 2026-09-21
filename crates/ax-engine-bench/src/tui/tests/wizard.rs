//! The Models wizard: family and precision lists, the confirm step, filtering,
//! row/step clicks, download-by-link, and the delete modal.
use super::super::catalog::{self};
use super::super::jobs::Job;
use super::super::{Modal, Screen, WizardStage};
use super::{family_index, key, mouse, new_app, render, render_sized};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::KeyCode;
use ratatui::crossterm::event::{MouseButton, MouseEventKind};
use ratatui::widgets::{Paragraph, Wrap};

// ---------------------------------------------------------------------------
// Models wizard
// ---------------------------------------------------------------------------

#[test]
fn family_list_renders_with_sizes_and_mtp_badge() {
    let mut app = new_app();
    app.screen = Screen::Models;
    let text = render(&app);
    assert!(text.contains("Models"));
    assert!(text.contains("AX Qwen 3.5 9B"));
    assert!(text.contains("AX Qwen 3.6 35B"));
    assert!(text.contains("bit"), "family rows show quant bits");
    assert!(text.contains('⚡'), "MTP badge should render");
    assert!(text.contains("Step 1 of"), "step header present");
}

#[test]
fn wide_family_panel_renders_long_names_in_full() {
    let mut app = new_app();
    app.screen = Screen::Models;
    // The longest catalog display name is the row a fixed 16-column cap used
    // to cut off mid-word.
    let longest = app
        .families
        .iter()
        .map(|family| family.display_name())
        .max_by_key(|name| name.chars().count())
        .expect("catalog is non-empty");
    assert!(
        longest.chars().count() > 16,
        "fixture must exceed the old fixed cap: {longest}"
    );
    // Keep the Details panel on a different family so the only source of this
    // exact string in the buffer is the left-hand list row.
    app.family_idx = app
        .families
        .iter()
        .position(|family| family.display_name() != longest)
        .expect("more than one family");
    let text = render_sized(&app, 200, 50);
    assert!(
        text.contains(&longest),
        "a wide panel must render the full family name, not an ellipsis: {longest}"
    );
}

#[test]
fn precision_screen_lists_quants_with_fit_badges() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-gemma4-12b");
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
    // The wizard is three steps for every family now — AutomatosX packs
    // bundle their MTP artifacts, so there is no separate speed-up step.
    assert!(text.contains("Step 2 of 3"));
    assert!(text.contains("Size"));
}

#[test]
fn uninstalled_variant_enter_goes_straight_to_confirm() {
    // The Options (MTP yes/no) step is gone: AutomatosX packs bundle their
    // MTP artifacts, so the wizard is Model › Size › Confirm.
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.on_key_models(KeyCode::Enter); // -> Precision
    app.precision_idx = 0;
    app.on_key_models(KeyCode::Enter); // -> Confirm (no Options stage)
    assert_eq!(app.stage, WizardStage::Confirm);
    let text = render(&app);
    assert!(text.contains("Step 3 of 3"));
    assert!(
        text.contains("included in snapshot"),
        "bundled MTP is reported in the summary"
    );
}

#[test]
fn confirm_step_shows_summary_and_default_destination() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-qwen3.5-9b");
    app.precision_idx = 0;
    app.begin_confirm();
    assert_eq!(app.stage, WizardStage::Confirm);
    let text = render(&app);
    assert!(text.contains("Confirm download"));
    assert!(text.contains("AX Qwen 3.5 9B"));
    assert!(text.contains("default cache"));
    assert!(text.contains("Free disk"));
    assert!(text.contains("Step 3 of 3"));
}

#[test]
fn confirm_enqueues_and_jumps_to_downloads() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-qwen3.5-9b");
    app.precision_idx = 0;
    app.begin_confirm();
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.screen, Screen::Downloads);
    assert_eq!(app.downloads.len(), 1);
    let task = &app.downloads[0];
    assert!(
        task.repo_id.starts_with("AutomatosX/"),
        "managed downloads target the AutomatosX org: {}",
        task.repo_id
    );
    assert!(task.dest.is_none(), "default destination is the HF cache");
    assert!(!app.toasts.is_empty(), "queueing raises a toast");
}

#[test]
fn bundled_mtp_confirm_uses_pack_repo_and_size() {
    let mut app = new_app();
    app.screen = Screen::Models;
    // The AX pack ships MTP in the same repo, so the plan is simply the
    // selected variant's repo and size — no separate MTP target lookup.
    app.family_idx = family_index(&app, "ax-qwen3.6-27b");
    app.precision_idx = 0; // OptiQ 4-bit flagship sorts first
    app.begin_confirm();
    app.on_key_models(KeyCode::Enter);
    let task = &app.downloads[0];
    assert_eq!(task.repo_id, "AutomatosX/AX-Qwen3.6-27B-MLX-OptiQ-4bit-MTP");
    assert_eq!(task.total_bytes, Some(20_239_552_902));
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
fn click_on_scrolled_family_row_selects_the_visible_family() {
    // Regression test: ratatui's List widget auto-scrolls to keep the
    // selected item on screen, so once a list scrolls, the panel's first
    // rendered row is no longer item 0. A click handler that ignores this
    // (as `row_in_rect` alone does — it only reports a within-panel row)
    // would select whatever item happens to sit at that raw index instead
    // of the item actually drawn there.
    let mut app = new_app();
    app.screen = Screen::Models;
    let last = app.families.len() - 1;
    app.family_idx = last;
    // A short terminal keeps the families panel well under the full list
    // height, forcing ratatui to scroll to keep `family_idx` visible.
    let _ = render_sized(&app, 100, 15);
    let rect = app.content_list_rect.get();
    let offset = app.content_list_offset.get();
    assert!(
        offset > 0,
        "fixture must actually force a scroll to exercise this bug; got offset {offset}"
    );
    // Click the first visible row (raw index 0 within the panel).
    app.on_click(rect.x + 2, rect.y + 1);
    assert_eq!(
        app.family_idx, offset,
        "click on the first visible row must select the family actually drawn there \
         (index `offset`), not family 0 as a naive unscrolled row index would"
    );
}

#[test]
fn click_on_completed_step_header_navigates_back() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.on_key_models(KeyCode::Enter);
    app.precision_idx = 0;
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.stage, WizardStage::Confirm);

    let _ = render(&app);
    let rect = app.step_header_rect.get();
    let model_offset = "Step 3 of 3 — ".chars().count();
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
    let keys: Vec<&str> = indices
        .iter()
        .map(|&i| app.families[i].key.as_str())
        .collect();
    assert_eq!(indices.len(), 2);
    assert!(keys.contains(&"ax-gemma4-12b"));
    assert!(keys.contains(&"ax-gemma4-12b-axq"));
    assert!(indices.contains(&app.family_idx));

    let text = render(&app);
    assert!(text.contains("filter: gemma4-12b"));
    assert!(
        !text.contains("AX Qwen 3.5 9B") && !text.contains("ax-qwen3.5-9b"),
        "non-matching family should be hidden"
    );

    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.on_key_models(KeyCode::Enter);
    assert_eq!(app.stage, WizardStage::Precision);
    assert_eq!(app.families[app.family_idx].key, "ax-gemma4-12b");
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
    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.on_paste("ax-qwen3.5-9b");
    assert_eq!(app.filter, "ax-qwen3.5-9b");
    assert_eq!(
        app.families[app.family_idx].key, "ax-qwen3.5-9b",
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
    // Right on Precision advances straight to Confirm (no Options stage:
    // AutomatosX packs bundle their MTP artifacts).
    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.precision_idx = 0;
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.stage, WizardStage::Confirm);
}

#[test]
fn left_key_steps_back_through_wizard() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-gemma4-12b");
    app.precision_idx = 0;
    app.stage = WizardStage::Confirm;
    app.pending = Some(super::super::PendingDownload {
        family_idx: app.family_idx,
        precision_idx: 0,
    });
    // Left on Confirm goes back to Precision.
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
    app.family_idx = family_index(&app, "ax-qwen3.5-9b");
    app.precision_idx = 0;
    app.begin_confirm();
    assert_eq!(app.stage, WizardStage::Confirm);
    // Right on Confirm confirms the download (same as Enter).
    app.on_key_models(KeyCode::Right);
    assert_eq!(app.screen, Screen::Downloads);
    assert_eq!(app.downloads.len(), 1);
}

#[test]
fn download_by_link_modal_queues_parsed_repo() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.on_key(key(KeyCode::Char('d')));
    assert!(matches!(app.modal, Some(Modal::DownloadByLink { .. })));
    for c in "https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    app.on_key(key(KeyCode::Enter));
    assert!(app.modal.is_none(), "a valid link closes the modal");
    assert_eq!(app.screen, Screen::Downloads);
    assert_eq!(app.downloads.len(), 1);
    let task = &app.downloads[0];
    assert_eq!(task.repo_id, "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP");
    assert_eq!(task.target, "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP");
    assert_eq!(task.label, task.repo_id);
    assert!(task.preset.is_none(), "free-form repos have no preset");
    assert_eq!(task.total_bytes, None, "no catalog size estimate");
    assert_eq!(
        task.watch_dir,
        catalog::repo_cache_dir("AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP")
    );
}

#[test]
fn download_by_link_revision_uses_at_form_target() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.on_key(key(KeyCode::Char('d')));
    for c in "owner/repo@v1".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    app.on_key(key(KeyCode::Enter));
    assert_eq!(app.downloads.len(), 1);
    let task = &app.downloads[0];
    assert_eq!(task.repo_id, "owner/repo");
    // The CLI download arg parses `@rev` and forwards --revision itself.
    assert_eq!(task.target, "owner/repo@v1");
}

#[test]
fn download_by_link_rejects_invalid_input() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.on_key(key(KeyCode::Char('d')));
    for c in "not-a-repo".chars() {
        app.on_key(key(KeyCode::Char(c)));
    }
    app.on_key(key(KeyCode::Enter));
    assert!(app.downloads.is_empty(), "invalid input queues nothing");
    let Some(Modal::DownloadByLink { input, error }) = &app.modal else {
        panic!("modal should stay open with an inline error");
    };
    assert_eq!(input, "not-a-repo", "typed input survives the rejection");
    assert!(error.is_some(), "parse error is shown inline");
    // Editing the buffer clears the inline error.
    app.on_key(key(KeyCode::Char('x')));
    assert!(matches!(&app.modal, Some(Modal::DownloadByLink { error, .. }) if error.is_none()));
}

#[test]
fn download_by_link_paste_fills_input() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.on_key(key(KeyCode::Char('d')));
    app.on_paste("https://huggingface.co/owner/repo\n");
    let Some(Modal::DownloadByLink { input, .. }) = &app.modal else {
        panic!("modal should still be open");
    };
    assert_eq!(
        input, "https://huggingface.co/owner/repo",
        "paste lands in the modal buffer without control chars"
    );
}

#[test]
fn click_on_precision_row_selects_and_advances() {
    let mut app = new_app();
    app.screen = Screen::Models;
    app.family_idx = family_index(&app, "ax-qwen3.5-9b");
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
fn wrapped_row_count_never_reports_zero_rows() {
    // An empty paragraph still occupies one terminal row. Callers size scroll
    // buffers and modal heights from this value, so a zero would collapse the
    // viewport to nothing.
    let empty = Paragraph::new("").wrap(Wrap { trim: false });
    assert_eq!(super::super::widgets::wrapped_row_count(&empty, 20), 1);
}

#[test]
fn wrapped_row_count_tracks_wrap_width() {
    // Pins the semantics ratatui documents for `Paragraph::line_count`, which
    // this helper is the only caller of: the same text needs more rows once the
    // wrap width drops below its length.
    let paragraph = Paragraph::new("Hello World").wrap(Wrap { trim: false });
    assert_eq!(super::super::widgets::wrapped_row_count(&paragraph, 20), 1);
    assert_eq!(super::super::widgets::wrapped_row_count(&paragraph, 10), 2);
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
