//! The local snapshot library is independent of the remote model catalog.
use crate::tui::{App, Modal, catalog, theme, widgets};
use ratatui::crossterm::event::KeyCode;
use ratatui::{
    Frame,
    layout::{Constraint, Layout, Rect},
    text::{Line, Span},
    widgets::{ListItem, Paragraph, Wrap},
};

impl App {
    pub(crate) fn reload_local_models(&mut self) {
        if self.local_models_reload.is_some() {
            return;
        }
        let root = crate::default_hf_cache_root();
        let (tx, rx) = std::sync::mpsc::channel();
        self.local_models_loading = true;
        self.local_models_error = None;
        self.local_models_reload = Some(rx);
        std::thread::spawn(move || {
            let _ = tx.send(catalog::scan_local_models(&root));
        });
    }

    pub(crate) fn tick_local_models(&mut self) -> bool {
        let Some(rx) = &self.local_models_reload else {
            return false;
        };
        let models = match rx.try_recv() {
            Ok(Ok(models)) => models,
            Ok(Err(error)) => {
                self.local_models_error = Some(error);
                self.local_models_reload = None;
                self.local_models_loading = false;
                return true;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => return false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.local_models_reload = None;
                self.local_models_loading = false;
                self.local_models_error = Some("Local model scan stopped unexpectedly".into());
                self.toast_error("Local model scan failed — click Refresh to retry");
                return true;
            }
        };
        let selected = self
            .local_models
            .get(self.local_model_idx)
            .map(|model| model.repo_id.clone());
        self.local_model_idx = selected
            .and_then(|repo| models.iter().position(|model| model.repo_id == repo))
            .unwrap_or(0);
        self.local_models = models;
        self.local_models_reload = None;
        self.local_models_loading = false;
        true
    }

    pub(crate) fn on_key_library(&mut self, code: KeyCode) {
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.local_model_idx = self.local_model_idx.saturating_sub(1)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.local_model_idx =
                    (self.local_model_idx + 1).min(self.local_models.len().saturating_sub(1))
            }
            KeyCode::Enter | KeyCode::Char('s') => {
                let Some(model) = self.local_models.get(self.local_model_idx) else {
                    return;
                };
                if model.ready {
                    self.modal = Some(Modal::ServeLocal(model.clone()));
                } else {
                    self.toast_warn(
                        "This snapshot is incomplete — finish its download before serving",
                    );
                }
            }
            KeyCode::Char('x') => {
                if let Some(model) = self.local_models.get(self.local_model_idx) {
                    self.modal = Some(Modal::DeleteLocal(model.clone()));
                }
            }
            KeyCode::Char('R') => self.reload_local_models(),
            KeyCode::Char('t') => self.downloads_show_library = false,
            KeyCode::Char('o') => {
                if let Some(model) = self.local_models.get(self.local_model_idx) {
                    let _ = std::process::Command::new("open")
                        .arg(&model.cache_dir)
                        .spawn();
                }
            }
            KeyCode::Esc | KeyCode::Left => self.back_or_home(),
            _ => {}
        }
    }

    pub(crate) fn delete_local_model(&mut self, model: &catalog::LocalModel) {
        let expected = catalog::repo_cache_dir(&model.repo_id);
        if expected != model.cache_dir
            || !std::fs::symlink_metadata(&expected).is_ok_and(|meta| meta.file_type().is_dir())
        {
            self.toast_error("Model cache path changed — refresh the local list before deleting");
            return;
        }
        if self.local_model_in_use(model) {
            self.toast_warn(
                "Stop the server using this cache and this model's downloads before deleting",
            );
            return;
        }
        match catalog::delete_local_cache(model, &crate::default_hf_cache_root()) {
            Ok(()) => {
                // Discard a scan started before deletion; its result may still
                // contain the removed directory. A fresh scan follows below.
                self.local_models_reload = None;
                self.local_models_loading = false;
                self.local_models
                    .retain(|entry| entry.repo_id != model.repo_id);
                self.local_model_idx = self
                    .local_model_idx
                    .min(self.local_models.len().saturating_sub(1));
                self.toast_success(format!("Deleted {} from local storage", model.repo_id));
                self.reload_families();
            }
            Err(error) => self.toast_error(format!("Could not delete model: {error}")),
        }
    }

    pub(crate) fn draw_local_library(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(5),
        ])
        .split(area);
        let status = if self.local_models_loading {
            "Scanning local snapshots…".into()
        } else if let Some(error) = &self.local_models_error {
            format!("{error} · R retry")
        } else {
            format!(
                "{} local models · all publishers · {} transfer jobs",
                self.local_models.len(),
                self.downloads.len()
            )
        };
        frame.render_widget(Paragraph::new(status).style(theme::label()), rows[0]);
        let name_width = rows[1].width.saturating_sub(27) as usize;
        let items: Vec<_> = self
            .local_models
            .iter()
            .map(|model| {
                ListItem::new(Line::from(vec![
                    Span::styled(widgets::ellipsis(&model.repo_id, name_width), theme::body()),
                    Span::styled(
                        format!(" {:>9}  ", catalog::format_bytes(model.size)),
                        theme::body_dim(),
                    ),
                    Span::styled(
                        if model.ready { "Ready" } else { "Incomplete" },
                        if model.ready {
                            theme::ok()
                        } else {
                            theme::warn()
                        },
                    ),
                ]))
            })
            .collect();
        let items = if items.is_empty() {
            vec![ListItem::new(
                "No local snapshots. Open Models to download a model.",
            )]
        } else {
            items
        };
        widgets::render_list_with_header(
            frame,
            rows[1],
            " Local models ",
            Some(Line::from(format!(
                "  {} {:>9}  Status",
                widgets::ellipsis("Repository", name_width),
                "Disk"
            ))),
            items,
            self.local_model_idx,
            true,
            &self.content_list_rect,
            &self.content_list_offset,
        );
        let detail = self.local_models.get(self.local_model_idx).map(|model| format!(
            "{}\nSnapshot: {}\n{} cached revision(s). Delete removes this repository's local cache.",
            model.repo_id, model.snapshot.display(), model.revisions
        )).unwrap_or_else(|| format!("Snapshot cache: {}", crate::default_hf_cache_root().display()));
        frame.render_widget(
            Paragraph::new(detail)
                .wrap(Wrap { trim: false })
                .block(widgets::soft_block(" Selected model ")),
            rows[2],
        );
    }

    pub(crate) fn local_model_in_use(&self, model: &catalog::LocalModel) -> bool {
        (self.server_running()
            && (self.external_server
                || self
                    .server_artifacts_dir
                    .as_ref()
                    .is_none_or(|path| path.starts_with(&model.cache_dir))))
            || self.downloads.iter().any(|task| {
                task.repo_id == model.repo_id && (task.is_running() || task.is_queued())
            })
    }
}
