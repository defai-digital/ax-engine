//! Models wizard: split-panel layout with family list on the left and
//! precision/options/confirm on the right.  The left panel is always visible
//! so the user never loses context when navigating between wizard steps.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{ListItem, Paragraph, Wrap};

use crate::tui::catalog::{self, RamFit};
use crate::tui::jobs::{DownloadMode, DownloadTask};
use crate::tui::theme;
use crate::tui::widgets::{self, DirectoryPicker};
use crate::tui::{App, Modal, PendingDownload, Screen, WizardStage};

impl App {
    // -- wizard input -----------------------------------------------------------

    pub(crate) fn on_key_models(&mut self, code: KeyCode) {
        match self.stage {
            WizardStage::Families if self.filtering => match code {
                KeyCode::Char(c) => {
                    self.filter.push(c);
                    self.clamp_family_idx_to_filter();
                }
                KeyCode::Backspace => {
                    self.filter.pop();
                    self.clamp_family_idx_to_filter();
                }
                // Wheel scroll routes here too (App::scroll → on_key_models):
                // move within the filtered list without leaving filter mode.
                KeyCode::Up => {
                    let _ = self.move_family_selection(-1);
                }
                KeyCode::Down => {
                    let _ = self.move_family_selection(1);
                }
                KeyCode::Enter | KeyCode::Esc => self.filtering = false,
                _ => {}
            },
            WizardStage::Families => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if !self.move_family_selection(-1) {
                        self.focus_tab_bar();
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let _ = self.move_family_selection(1);
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    self.precision_idx = 0;
                    self.stage = WizardStage::Precision;
                }
                KeyCode::Char('/') => self.filtering = true,
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    if !self.filter.is_empty() {
                        self.filter.clear();
                    } else if !self.go_back_screen() {
                        self.screen = Screen::Home;
                        self.focus_tabs = false;
                        self.previous_screen = None;
                    }
                }
                _ => {}
            },
            WizardStage::Precision => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.precision_idx == 0 {
                        self.focus_tab_bar();
                    } else {
                        self.precision_idx = self.precision_idx.saturating_sub(1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    let n = self.families[self.family_idx].variants.len();
                    if self.precision_idx + 1 < n {
                        self.precision_idx += 1;
                    }
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    let variant = &self.families[self.family_idx].variants[self.precision_idx];
                    if variant.mtp_alias.is_some() {
                        self.mtp_idx = 0;
                        self.stage = WizardStage::Options;
                    } else if variant.installed {
                        self.modal = Some(Modal::ServeInstalled {
                            family_idx: self.family_idx,
                            variant_idx: self.precision_idx,
                        });
                    } else {
                        self.begin_confirm(false);
                    }
                }
                KeyCode::Char('x') => {
                    let variant = &self.families[self.family_idx].variants[self.precision_idx];
                    if variant.installed {
                        self.modal = Some(Modal::DeleteModel {
                            family_idx: self.family_idx,
                            variant_idx: self.precision_idx,
                            typed: String::new(),
                        });
                    }
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    self.stage = WizardStage::Families
                }
                _ => {}
            },
            WizardStage::Options => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.mtp_idx == 0 {
                        self.focus_tab_bar();
                    } else {
                        self.mtp_idx = 0;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    self.mtp_idx = 1;
                }
                KeyCode::Char('y') => self.begin_confirm(true),
                KeyCode::Char('n') => self.begin_confirm(false),
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    self.begin_confirm(self.mtp_idx == 0)
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    self.stage = WizardStage::Precision
                }
                _ => {}
            },
            WizardStage::Confirm => match code {
                KeyCode::Up | KeyCode::Char('k') => self.focus_tab_bar(),
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => self.confirm_download(),
                KeyCode::Char('c') => {
                    self.modal = Some(Modal::DestPicker(DirectoryPicker::new()));
                }
                KeyCode::Char('d') => {
                    self.confirm_dest = None;
                    self.toast("using the shared HF cache");
                }
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    let back_to_options = self.pending.is_some_and(|pending| {
                        self.families[pending.family_idx].variants[pending.precision_idx]
                            .mtp_alias
                            .is_some()
                    });
                    self.stage = if back_to_options {
                        WizardStage::Options
                    } else {
                        WizardStage::Precision
                    };
                }
                _ => {}
            },
        }
    }

    pub(crate) fn on_click_models(&mut self, idx: usize) {
        match self.stage {
            WizardStage::Families => {
                if let Some(&real) = self.filtered_family_indices().get(idx) {
                    self.family_idx = real;
                    self.on_key_models(KeyCode::Enter);
                }
            }
            WizardStage::Precision if idx < self.families[self.family_idx].variants.len() => {
                self.precision_idx = idx;
                self.on_key_models(KeyCode::Enter);
            }
            WizardStage::Options => {
                if idx == 0 {
                    self.mtp_idx = 0;
                } else if idx == 1 {
                    self.mtp_idx = 1;
                }
                self.on_key_models(KeyCode::Enter);
            }
            _ => {}
        }
    }

    pub(crate) fn on_step_header_click(&mut self, offset: usize) {
        let steps = self.wizard_steps();
        let current = steps
            .iter()
            .position(|(_, stage)| *stage == self.stage)
            .unwrap_or(0);
        let mut cursor = format!("Step {} of {} — ", current + 1, steps.len())
            .chars()
            .count();
        for (index, (label, stage)) in steps.iter().enumerate() {
            if index > 0 {
                cursor += " › ".chars().count();
            }
            let end = cursor + label.chars().count();
            if offset >= cursor && offset < end {
                if index <= current {
                    self.stage = *stage;
                }
                return;
            }
            cursor = end;
        }
    }

    /// Indices into `self.families` matching the active filter (all of them if empty).
    pub(crate) fn filtered_family_indices(&self) -> Vec<usize> {
        if self.filter.is_empty() {
            return (0..self.families.len()).collect();
        }
        let needle = self.filter.to_ascii_lowercase();
        self.families
            .iter()
            .enumerate()
            .filter(|(_, f)| {
                if f.key.to_ascii_lowercase().contains(&needle)
                    || f.display_name().to_ascii_lowercase().contains(&needle)
                {
                    return true;
                }
                // Match aliases / repo ids so "/gpt" and "/llama" find secondary stacks.
                f.variants.iter().any(|v| {
                    v.profile.repo_id.to_ascii_lowercase().contains(&needle)
                        || v.profile
                            .aliases
                            .iter()
                            .any(|alias| alias.to_ascii_lowercase().contains(&needle))
                })
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Move `family_idx` to the previous/next entry within the filtered list.
    /// Returns false when an Up press is already at the first row (caller may
    /// promote focus to the tab bar).
    fn move_family_selection(&mut self, delta: i32) -> bool {
        let indices = self.filtered_family_indices();
        if indices.is_empty() {
            return false;
        }
        let pos = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        if delta < 0 && pos == 0 {
            return false;
        }
        let new_pos = if delta < 0 {
            pos.saturating_sub(1)
        } else {
            (pos + 1).min(indices.len() - 1)
        };
        self.family_idx = indices[new_pos];
        true
    }

    /// After the filter text changes, snap `family_idx` back into the filtered set if needed.
    pub(crate) fn clamp_family_idx_to_filter(&mut self) {
        let indices = self.filtered_family_indices();
        if let Some(&first) = indices.first()
            && !indices.contains(&self.family_idx)
        {
            self.family_idx = first;
        }
    }

    // -- confirm / enqueue --------------------------------------------------------

    /// Enter the confirm step for the current family/precision selection.
    pub(crate) fn begin_confirm(&mut self, with_mtp: bool) {
        self.pending = Some(PendingDownload {
            family_idx: self.family_idx,
            precision_idx: self.precision_idx,
            with_mtp,
        });
        self.confirm_dest = None;
        self.stage = WizardStage::Confirm;
        self.screen = Screen::Models;
    }

    /// (subcmd, target alias, base repo id, total size estimate) for a pending download.
    fn pending_plan(
        &self,
        pending: PendingDownload,
    ) -> (&'static str, &'static str, &'static str, Option<u64>) {
        let variant = &self.families[pending.family_idx].variants[pending.precision_idx];
        if pending.with_mtp {
            let alias = variant.mtp_alias.unwrap_or(variant.profile.label);
            if let Some(target) = crate::mtp_download_target_for_model(alias) {
                let total = match (target.approx_base_bytes, target.approx_extra_bytes) {
                    (Some(base), Some(extra)) => Some(base + extra),
                    (Some(base), None) => Some(base),
                    _ => None,
                };
                return ("download-mtp", alias, target.repo_id, total);
            }
            ("download-mtp", alias, variant.profile.repo_id, None)
        } else {
            (
                "download",
                variant.profile.label,
                variant.profile.repo_id,
                variant.profile.approx_size_bytes,
            )
        }
    }

    fn confirm_download(&mut self) {
        let Some(pending) = self.pending else {
            return;
        };
        // Fail before enqueueing so the queue does not sit on a permanent
        // "failed" row for a missing Python dep (common after a bare mlx-only
        // venv install).
        if let Err(err) = crate::ensure_download_python_deps() {
            self.toast_error(err);
            return;
        }
        let (subcmd, target, repo_id, total_bytes) = self.pending_plan(pending);
        let variant = &self.families[pending.family_idx].variants[pending.precision_idx];
        let dest = self.confirm_dest.as_ref().map(|parent| {
            explicit_destination_path(
                parent,
                variant.profile.label,
                variant.mtp_alias,
                pending.with_mtp,
            )
        });
        let watch_dir = dest
            .clone()
            .unwrap_or_else(|| catalog::repo_cache_dir(repo_id));
        let label = format!(
            "{} {}",
            self.families[pending.family_idx].display_name(),
            variant.precision()
        );
        let mode = if pending.with_mtp {
            DownloadMode::Mtp
        } else {
            DownloadMode::Direct
        };
        let task = DownloadTask {
            label: label.clone(),
            repo_id,
            preset: variant.profile.preset,
            mode,
            subcmd,
            target: target.to_string(),
            dest,
            watch_dir,
            resolved_path: None,
            total_bytes,
            phase: None,
            job: None,
            cancelled: false,
        };
        self.downloads.push(task);
        self.download_idx = self.downloads.len().saturating_sub(1);
        self.start_next_queued_download();
        self.toast(format!("{label} queued"));
        self.pending = None;
        self.confirm_dest = None;
        self.stage = WizardStage::Precision;
        self.screen = Screen::Downloads;
    }

    /// Remove an installed variant's HF-cache directory.
    pub(crate) fn delete_installed_variant(&mut self, family_idx: usize, variant_idx: usize) {
        let Some(variant) = self
            .families
            .get(family_idx)
            .and_then(|f| f.variants.get(variant_idx))
        else {
            return;
        };
        let label = format!(
            "{} {}",
            self.families[family_idx].display_name(),
            variant.precision()
        );
        let dir = catalog::repo_cache_dir(variant.profile.repo_id);
        match std::fs::remove_dir_all(&dir) {
            Ok(()) => self.toast_success(format!("{label} deleted")),
            Err(err) => self.toast_error(format!("delete failed: {err}")),
        }
        self.reload_families();
    }

    // -- rendering ------------------------------------------------------------

    /// Wizard steps for the header: Options only exists for MTP-capable variants.
    fn wizard_steps(&self) -> Vec<(&'static str, WizardStage)> {
        let has_options = self
            .families
            .get(self.family_idx)
            .map(|family| match self.stage {
                WizardStage::Families => family.has_mtp(),
                _ => family
                    .variants
                    .get(self.precision_idx)
                    .is_some_and(|v| v.mtp_alias.is_some()),
            })
            .unwrap_or(false);
        let mut steps = vec![
            ("Model", WizardStage::Families),
            ("Size", WizardStage::Precision),
        ];
        if has_options {
            steps.push(("Speed-up", WizardStage::Options));
        }
        steps.push(("Confirm", WizardStage::Confirm));
        steps
    }

    /// Split-panel layout: left panel (40%) is the family list, right panel
    /// (60%) shows precision/options/confirm depending on the wizard stage.
    pub(crate) fn draw_models(&self, frame: &mut Frame, area: Rect) {
        let panels = Layout::horizontal([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area);

        // Left panel: family list (always visible).
        let left_active = self.stage == WizardStage::Families;
        self.draw_families_panel(frame, panels[0], left_active);

        // Right panel: step header + content.
        let right = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(panels[1]);
        self.draw_step_header(frame, right[0]);
        match self.stage {
            WizardStage::Families => {
                // Show a helpful hint when no family is selected yet.
                self.draw_right_hint(frame, right[1]);
            }
            WizardStage::Precision => self.draw_precision(frame, right[1]),
            WizardStage::Options => self.draw_options(frame, right[1]),
            WizardStage::Confirm => self.draw_confirm(frame, right[1]),
        }
    }

    fn draw_right_hint(&self, frame: &mut Frame, area: Rect) {
        let family = &self.families[self.family_idx];
        let mut lines = vec![
            Line::raw(""),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    family.display_name(),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw("  "),
                Span::styled(format!("{} · ", family.key), theme::label()),
                Span::styled(compact_quant_summary(family), theme::body_dim()),
                if family.is_primary() {
                    Span::styled(" · primary", theme::body_dim())
                } else {
                    Span::styled(" · preview direct", theme::label())
                },
                if family.has_mtp() {
                    Span::styled(
                        format!("  {} speed-up", theme::icon::SPEED),
                        theme::feature(),
                    )
                } else {
                    Span::raw("")
                },
            ]),
            Line::raw(""),
            Line::from(Span::styled("  Sizes on this machine:", theme::label())),
        ];
        for (i, v) in family.variants.iter().enumerate() {
            let fit = catalog::ram_fit(v.size_estimate(), self.hardware.total_ram_bytes);
            let rec = if self.is_recommended_variant(self.family_idx, i) {
                format!(" {}", theme::icon::STAR)
            } else {
                String::new()
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    format!("{:<7}{rec}", v.precision()),
                    Style::default().fg(theme::TEXT),
                ),
                Span::styled(
                    format!("{:>9}  ", catalog::format_approx_bytes(v.size_estimate())),
                    theme::body_dim(),
                ),
                super::home::fit_span(fit),
                if v.installed {
                    Span::styled(format!(" {}", theme::icon::OK), theme::ok())
                } else {
                    Span::raw("")
                },
            ]));
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                "Enter",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" or → pick a size", theme::label()),
        ]));
        frame.render_widget(
            Paragraph::new(lines)
                .block(widgets::soft_block(" Details "))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    /// Compact step header: `Step 2 of 4 — Model › Precision › Options › Confirm`
    fn draw_step_header(&self, frame: &mut Frame, area: Rect) {
        self.step_header_rect.set(area);
        let steps = self.wizard_steps();
        let current = steps
            .iter()
            .position(|(_, stage)| *stage == self.stage)
            .unwrap_or(0);
        let mut spans = vec![Span::styled(
            format!(" Step {} of {} — ", current + 1, steps.len()),
            theme::label(),
        )];
        for (index, (label, _)) in steps.iter().enumerate() {
            if index > 0 {
                spans.push(Span::styled(" › ", theme::label()));
            }
            let style = if index == current {
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else if index < current {
                Style::default()
                    .fg(theme::OK)
                    .add_modifier(Modifier::UNDERLINED)
            } else {
                theme::label()
            };
            spans.push(Span::styled(*label, style));
        }
        // Selection breadcrumb.
        let mut selection = Vec::new();
        if self.stage != WizardStage::Families {
            selection.push(self.families[self.family_idx].display_name());
            selection.push(self.families[self.family_idx].variants[self.precision_idx].precision());
        }
        if matches!(self.stage, WizardStage::Confirm)
            && let Some(pending) = self.pending
            && self.families[pending.family_idx].variants[pending.precision_idx]
                .mtp_alias
                .is_some()
        {
            selection.push(if pending.with_mtp {
                "with speed-up".into()
            } else {
                "no speed-up".into()
            });
        }
        if !selection.is_empty() {
            spans.push(Span::raw("   "));
            spans.push(Span::styled(selection.join(" · "), theme::body_dim()));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Family list panel — always visible on the left.
    fn draw_families_panel(&self, frame: &mut Frame, area: Rect, active: bool) {
        let indices = self.filtered_family_indices();
        let rows: Vec<ListItem> = indices
            .iter()
            .map(|&i| {
                let family = &self.families[i];
                let installed = family.installed_count();
                let status = if installed == family.variants.len() {
                    Span::styled(format!("{}all", theme::icon::OK), theme::ok())
                } else if installed > 0 {
                    Span::styled(
                        format!("{installed}/{}", family.variants.len()),
                        theme::ok(),
                    )
                } else {
                    Span::styled("--", theme::label())
                };
                let mtp = if family.has_mtp() {
                    Span::styled(format!(" {}", theme::icon::SPEED), theme::feature())
                } else {
                    Span::raw("")
                };
                let tier = if family.is_primary() {
                    Span::raw("")
                } else {
                    Span::styled(" preview", theme::label())
                };
                let name = family.display_name();
                let name_cell = widgets::ellipsis(&name, 16);
                // Compact quant: "4–8b" when multi, else single / MXFP4.
                let quant = compact_quant_summary(family);
                ListItem::new(Line::from(vec![
                    Span::styled(
                        name_cell,
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(format!(" {quant:<9}"), theme::body_dim()),
                    status,
                    mtp,
                    tier,
                ]))
            })
            .collect();
        let rows = if rows.is_empty() {
            vec![ListItem::new(Line::from(vec![
                Span::styled("No models match the filter.", theme::warn()),
                Span::styled(" — Esc clears the filter", theme::label()),
            ]))]
        } else {
            rows
        };
        let selected = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        let title = if self.filtering {
            format!(" Models — filter: {}_ ", self.filter)
        } else if !self.filter.is_empty() {
            format!(" Models — filter: {} ", self.filter)
        } else {
            " Models — / to filter ".to_string()
        };
        widgets::render_list(
            frame,
            area,
            &title,
            rows,
            selected,
            active,
            &self.content_list_rect,
        );
    }

    fn draw_precision(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).split(area);
        let family = &self.families[self.family_idx];
        let rows: Vec<ListItem> = family
            .variants
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let rec = if self.is_recommended_variant(self.family_idx, i) {
                    format!(" {}rec", theme::icon::STAR)
                } else {
                    String::new()
                };
                let size = Span::styled(
                    format!("{:>9}", catalog::format_approx_bytes(v.size_estimate())),
                    theme::body_dim(),
                );
                let fit = super::home::fit_span(catalog::ram_fit(
                    v.size_estimate(),
                    self.hardware.total_ram_bytes,
                ));
                let mtp = if v.mtp_alias.is_some() {
                    Span::styled(format!(" {}", theme::icon::SPEED), theme::feature())
                } else {
                    Span::raw("  ")
                };
                let status = if v.installed {
                    Span::styled(format!(" {}", theme::icon::OK), theme::ok())
                } else {
                    Span::raw("  ")
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<7}{rec}", v.precision()),
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    size,
                    Span::raw(" "),
                    fit,
                    mtp,
                    status,
                ]))
            })
            .collect();
        let title = format!(" {} — size ", family.display_name());
        widgets::render_list(
            frame,
            chunks[0],
            &title,
            rows,
            self.precision_idx,
            true,
            &self.content_list_rect,
        );
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                format!(
                    "  {} = Quick start pick · ↑↓ move · Enter select",
                    theme::icon::STAR
                ),
                theme::label(),
            ))),
            chunks[1],
        );
    }

    fn draw_options(&self, frame: &mut Frame, area: Rect) {
        let family = &self.families[self.family_idx];
        let variant = &family.variants[self.precision_idx];
        let yes = self.mtp_idx == 0;
        let extra = variant
            .mtp_size_estimate()
            .map(|(_, extra)| catalog::format_approx_bytes(extra))
            .unwrap_or_else(|| "size varies".into());
        let opt = |label: &str, sel: bool| {
            let style = if sel { theme::cta() } else { theme::body_dim() };
            let marker = if sel {
                format!("{} ", theme::icon::SELECT)
            } else {
                "  ".into()
            };
            Line::from(Span::styled(format!("  {marker}{label}  "), style))
        };
        let text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{} {}", family.display_name(), variant.precision()),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" can run faster with an optional package.", theme::body()),
            ]),
            Line::raw(""),
            Line::from(Span::styled(
                format!("Faster generation downloads a small extra package ({extra})"),
                theme::body_dim(),
            )),
            Line::from(Span::styled(
                "and speeds up replies. Quality stays the same.",
                theme::body_dim(),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                "Include the speed-up?",
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            opt("Yes — faster generation (recommended)", yes),
            opt("No — base model only", !yes),
        ];
        let block = widgets::active_block(&format!(" {} Optional speed-up ", theme::icon::SPEED));
        frame.render_widget(
            Paragraph::new(text)
                .block(block.border_style(Style::default().fg(theme::FEATURE)))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_confirm(&self, frame: &mut Frame, area: Rect) {
        let Some(pending) = self.pending else {
            frame.render_widget(
                Paragraph::new("Nothing selected yet — pick a model first.")
                    .block(widgets::soft_block(" Confirm ")),
                area,
            );
            return;
        };
        let family = &self.families[pending.family_idx];
        let variant = &family.variants[pending.precision_idx];
        let (_, _, _, total) = self.pending_plan(pending);
        let fit = catalog::ram_fit(
            total.or(variant.size_estimate()),
            self.hardware.total_ram_bytes,
        );
        let dest_text = match &self.confirm_dest {
            Some(parent) => explicit_destination_path(
                parent,
                variant.profile.label,
                variant.mtp_alias,
                pending.with_mtp,
            )
            .display()
            .to_string(),
            None => format!(
                "{} (default cache)",
                crate::default_hf_cache_root().display()
            ),
        };
        let free = self.hardware.free_disk_bytes;
        let free_after = match (free, total) {
            (Some(free), Some(total)) => Some(free.saturating_sub(total)),
            _ => None,
        };
        let row = |label: &str, value: String| {
            Line::from(vec![
                Span::styled(format!("  {label:<12}"), theme::label()),
                Span::styled(value, theme::body()),
            ])
        };
        let mut lines = vec![
            Line::from(Span::styled(
                "  Download summary",
                Style::default()
                    .fg(theme::ACCENT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            row(
                "Model",
                format!("{} {}", family.display_name(), variant.precision()),
            ),
            row(
                "Speed-up",
                if variant.mtp_alias.is_none() {
                    "not available".into()
                } else if pending.with_mtp {
                    format!("{} included", theme::icon::SPEED)
                } else {
                    "no".into()
                },
            ),
            row("Download", catalog::format_approx_bytes(total)),
            row("Into", dest_text),
        ];
        if let (Some(free), Some(after)) = (free, free_after) {
            lines.push(row(
                "Free disk",
                format!(
                    "{} now → {} after",
                    catalog::format_bytes(free),
                    catalog::format_bytes(after)
                ),
            ));
        }
        if !fit.plain().is_empty() {
            lines.push(row("Memory", fit.plain().into()));
        }
        lines.push(Line::raw(""));
        match fit {
            RamFit::TooLarge => lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", theme::icon::WARN),
                    Style::default().fg(theme::ON_ACCENT).bg(theme::DANGER),
                ),
                Span::raw(" "),
                Span::styled(
                    "This model likely exceeds this Mac's memory.",
                    theme::danger(),
                ),
            ])),
            RamFit::Tight => lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", theme::icon::WARN),
                    Style::default().fg(theme::ON_ACCENT).bg(theme::WARN),
                ),
                Span::raw(" "),
                Span::styled("Fits, but leaves little headroom.", theme::warn()),
            ])),
            _ => {}
        }
        if let (Some(free), Some(total)) = (free, total)
            && total > free
        {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", theme::icon::WARN),
                    Style::default().fg(theme::ON_ACCENT).bg(theme::DANGER),
                ),
                Span::raw(" "),
                Span::styled("Not enough free disk space.", theme::danger()),
            ]));
        }
        if variant.installed && !pending.with_mtp {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {} ", theme::icon::OK),
                    Style::default().fg(theme::ON_ACCENT).bg(theme::OK),
                ),
                Span::raw(" "),
                Span::styled(
                    "Already installed — re-checks the cached copy.",
                    theme::ok(),
                ),
            ]));
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(" Enter download ", theme::cta()),
            Span::styled(
                "  c change folder  ·  d default cache  ·  Esc back",
                theme::label(),
            ),
        ]));
        frame.render_widget(
            Paragraph::new(lines)
                .block(widgets::active_block(&format!(
                    " {} Confirm download ",
                    theme::icon::OK
                )))
                .wrap(Wrap { trim: false }),
            area,
        );
    }

    /// Directory-picker modal body (drawn over the confirm step).
    pub(crate) fn draw_dest_picker(&self, frame: &mut Frame, area: Rect, picker: &DirectoryPicker) {
        let popup = widgets::centered_rect(
            72.min(area.width.saturating_sub(2)),
            22.min(area.height.saturating_sub(2)),
            area,
        );
        frame.render_widget(ratatui::widgets::Clear, popup);
        let chunks = Layout::vertical([Constraint::Length(2), Constraint::Min(0)]).split(popup);
        let error = picker
            .error
            .as_ref()
            .map(|err| Line::from(Span::styled(err.clone(), theme::danger())))
            .unwrap_or_else(|| {
                Line::from(Span::styled(
                    "Enter open · s use this folder · d default cache · ~ home · Esc cancel",
                    theme::label(),
                ))
            });
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    " Download into a custom folder ",
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                )),
                error,
            ]),
            chunks[0],
        );
        let rows: Vec<ListItem> = picker
            .entries
            .iter()
            .map(|entry| ListItem::new(Line::from(entry.label.clone())))
            .collect();
        widgets::render_list(
            frame,
            chunks[1],
            &format!(" {} ", picker.current.display()),
            rows,
            picker.selected,
            true,
            &self.content_list_rect,
        );
    }
}

/// Compact quant range for dense family rows (`4–8b` / `6b`).
fn compact_quant_summary(family: &crate::tui::catalog::Family) -> String {
    let all_mxfp4 = family
        .variants
        .iter()
        .all(|v| v.profile.repo_id.to_ascii_lowercase().contains("mxfp4"));
    if all_mxfp4 && !family.variants.is_empty() {
        return "MXFP4".into();
    }
    let bits: Vec<u32> = family.variants.iter().filter_map(|v| v.bits).collect();
    match bits.as_slice() {
        [] => "--".into(),
        [one] => format!("{one}b"),
        many => {
            let lo = many.iter().copied().min().unwrap_or(0);
            let hi = many.iter().copied().max().unwrap_or(0);
            if lo == hi {
                format!("{lo}b")
            } else {
                format!("{lo}–{hi}b")
            }
        }
    }
}

/// Leaf directory a custom-destination download lands in: `<parent>/<label>`
/// (or `<parent>/<mtp-alias>-mtp` for MTP packages).
pub(crate) fn explicit_destination_path(
    parent: &std::path::Path,
    label: &str,
    mtp_alias: Option<&str>,
    with_mtp: bool,
) -> PathBuf {
    let leaf = if with_mtp {
        format!(
            "{}-mtp",
            widgets::sanitize_path_segment(mtp_alias.unwrap_or(label))
        )
    } else {
        widgets::sanitize_path_segment(label)
    };
    parent.join(leaf)
}
