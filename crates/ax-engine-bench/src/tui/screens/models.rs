//! Models wizard: Step 1 family -> Step 2 precision -> Step 3 options (MTP,
//! only when available) -> Step 4 confirm.  Nothing is downloaded until the
//! confirm step is accepted; the destination defaults to the shared HF cache
//! and is only surfaced there (as a changeable field) instead of being a
//! mandatory directory-browsing stage.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, ListItem, Paragraph, Wrap};

use crate::tui::catalog::{self, RamFit};
use crate::tui::jobs::{DownloadMode, DownloadTask};
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
                KeyCode::Enter | KeyCode::Esc => self.filtering = false,
                _ => {}
            },
            WizardStage::Families => match code {
                KeyCode::Up | KeyCode::Char('k') => self.move_family_selection(-1),
                KeyCode::Down | KeyCode::Char('j') => self.move_family_selection(1),
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    self.precision_idx = 0;
                    self.stage = WizardStage::Precision;
                }
                KeyCode::Char('/') => self.filtering = true,
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    if self.filter.is_empty() {
                        self.screen = Screen::Home;
                    } else {
                        self.filter.clear();
                    }
                }
                _ => {}
            },
            WizardStage::Precision => match code {
                KeyCode::Up | KeyCode::Char('k') => {
                    self.precision_idx = self.precision_idx.saturating_sub(1)
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
                        // Even an installed base can still gain the MTP package.
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
                KeyCode::Up | KeyCode::Down | KeyCode::Char('k') | KeyCode::Char('j') => {
                    self.mtp_idx ^= 1;
                }
                KeyCode::Char('y') => self.begin_confirm(true),
                KeyCode::Char('n') => self.begin_confirm(false),
                KeyCode::Enter => self.begin_confirm(self.mtp_idx == 0),
                KeyCode::Left | KeyCode::Char('h') | KeyCode::Esc => {
                    self.stage = WizardStage::Precision
                }
                _ => {}
            },
            WizardStage::Confirm => match code {
                KeyCode::Enter => self.confirm_download(),
                KeyCode::Char('c') => {
                    self.modal = Some(Modal::DestPicker(DirectoryPicker::new()));
                }
                KeyCode::Char('d') => {
                    self.confirm_dest = None;
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
            _ => {}
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
            .filter(|(_, f)| f.key.to_ascii_lowercase().contains(&needle))
            .map(|(i, _)| i)
            .collect()
    }

    /// Move `family_idx` to the previous/next entry within the filtered list.
    fn move_family_selection(&mut self, delta: i32) {
        let indices = self.filtered_family_indices();
        if indices.is_empty() {
            return;
        }
        let pos = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        let new_pos = if delta < 0 {
            pos.saturating_sub(1)
        } else {
            (pos + 1).min(indices.len() - 1)
        };
        self.family_idx = indices[new_pos];
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
            self.families[pending.family_idx].key,
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

    /// Remove an installed variant's HF-cache directory.  `models rm` refuses
    /// cache paths by design, so the TUI owns this deletion — it only ever
    /// targets the `models--org--name` directory it derived itself, after the
    /// typed confirmation in the modal.
    pub(crate) fn delete_installed_variant(&mut self, family_idx: usize, variant_idx: usize) {
        let Some(variant) = self
            .families
            .get(family_idx)
            .and_then(|f| f.variants.get(variant_idx))
        else {
            return;
        };
        let label = format!("{} {}", self.families[family_idx].key, variant.precision());
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
        let has_options = match self.stage {
            WizardStage::Families => self.families[self.family_idx].has_mtp(),
            _ => self.families[self.family_idx].variants[self.precision_idx]
                .mtp_alias
                .is_some(),
        };
        let mut steps = vec![
            ("Model", WizardStage::Families),
            ("Precision", WizardStage::Precision),
        ];
        if has_options {
            steps.push(("Options", WizardStage::Options));
        }
        steps.push(("Confirm", WizardStage::Confirm));
        steps
    }

    pub(crate) fn draw_models(&self, frame: &mut Frame, area: Rect) {
        let chunks = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
        self.draw_step_header(frame, chunks[0]);
        match self.stage {
            WizardStage::Families => self.draw_families(frame, chunks[1]),
            WizardStage::Precision => self.draw_precision(frame, chunks[1]),
            WizardStage::Options => self.draw_options(frame, chunks[1]),
            WizardStage::Confirm => self.draw_confirm(frame, chunks[1]),
        }
    }

    /// `Step 2 of 4 — Precision   gemma4-12b · 4-bit` header line.
    fn draw_step_header(&self, frame: &mut Frame, area: Rect) {
        let steps = self.wizard_steps();
        let current = steps
            .iter()
            .position(|(_, stage)| *stage == self.stage)
            .unwrap_or(0);
        let mut spans = vec![Span::styled(
            format!(
                "Step {} of {} — {}   ",
                current + 1,
                steps.len(),
                steps[current].0
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )];
        let mut selection = Vec::new();
        if self.stage != WizardStage::Families {
            selection.push(self.families[self.family_idx].key.clone());
            selection.push(self.families[self.family_idx].variants[self.precision_idx].precision());
        }
        if matches!(self.stage, WizardStage::Confirm)
            && let Some(pending) = self.pending
            && self.families[pending.family_idx].variants[pending.precision_idx]
                .mtp_alias
                .is_some()
        {
            selection.push(if pending.with_mtp {
                "with MTP".into()
            } else {
                "no MTP".into()
            });
        }
        if !selection.is_empty() {
            spans.push(Span::styled(
                selection.join(" · "),
                Style::default().fg(Color::Gray),
            ));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn draw_families(&self, frame: &mut Frame, area: Rect) {
        let indices = self.filtered_family_indices();
        let rows: Vec<ListItem> = indices
            .iter()
            .map(|&i| {
                let family = &self.families[i];
                let installed = family.installed_count();
                let status = if installed == family.variants.len() {
                    Span::styled("installed", Style::default().fg(Color::Green))
                } else if installed > 0 {
                    Span::styled(
                        format!("{installed}/{} installed", family.variants.len()),
                        Style::default().fg(Color::Green),
                    )
                } else {
                    Span::styled("--", Style::default().fg(Color::DarkGray))
                };
                let mtp = if family.has_mtp() {
                    Span::styled("  ⚡MTP", Style::default().fg(Color::Magenta))
                } else {
                    Span::raw("")
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{:<16}", family.key),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!("{:<29}", family.quant_summary()),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(
                        format!(
                            "from {:<12}",
                            catalog::format_approx_bytes(family.min_size_estimate())
                        ),
                        Style::default().fg(Color::Gray),
                    ),
                    status,
                    mtp,
                ]))
            })
            .collect();
        let rows = if rows.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No families match the filter.",
                Style::default().fg(Color::Yellow),
            )))]
        } else {
            rows
        };
        let selected = indices
            .iter()
            .position(|&i| i == self.family_idx)
            .unwrap_or(0);
        let title = if self.filtering {
            format!(" Choose a model — filter: {}_ ", self.filter)
        } else if !self.filter.is_empty() {
            format!(" Choose a model — filter: {} (Esc clears) ", self.filter)
        } else {
            " Choose a model — / to filter ".to_string()
        };
        widgets::render_list(
            frame,
            area,
            &title,
            rows,
            selected,
            true,
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
                let mut precision = v.precision();
                if i == 0 {
                    precision.push_str("  (recommended)");
                }
                let size = Span::styled(
                    format!("{:<14}", catalog::format_approx_bytes(v.size_estimate())),
                    Style::default().fg(Color::Gray),
                );
                let fit = super::home::fit_span(catalog::ram_fit(
                    v.size_estimate(),
                    self.hardware.total_ram_bytes,
                ));
                let mtp = if v.mtp_alias.is_some() {
                    Span::styled("⚡MTP  ", Style::default().fg(Color::Magenta))
                } else {
                    Span::raw("      ")
                };
                let status = if v.installed {
                    Span::styled("installed", Style::default().fg(Color::Green))
                } else {
                    Span::styled("--", Style::default().fg(Color::DarkGray))
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("{precision:<24}"),
                        Style::default().add_modifier(Modifier::BOLD),
                    ),
                    size,
                    fit,
                    mtp,
                    status,
                ]))
            })
            .collect();
        let title = format!(" {} — choose a precision ", family.key);
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
                "  Lower bits → smaller download, faster generation · higher bits → better quality",
                Style::default().fg(Color::DarkGray),
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
            let style = if sel {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            let marker = if sel { "▸ " } else { "  " };
            Line::from(Span::styled(format!("  {marker}{label}  "), style))
        };
        let text = vec![
            Line::from(vec![
                Span::styled(
                    format!("{} {}", family.key, variant.precision()),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(" has an optional speed-up available."),
            ]),
            Line::raw(""),
            Line::raw("MTP (multi-token prediction) downloads a small extra package alongside"),
            Line::from(format!(
                "the weights ({extra}) and makes text generation noticeably faster.",
            )),
            Line::raw("There is no quality cost — the model's answers stay the same."),
            Line::raw(""),
            Line::from(Span::styled(
                "Include the speed-up?",
                Style::default().add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            opt("Yes — include MTP (recommended)", yes),
            opt("No — base weights only", !yes),
            Line::raw(""),
            Line::from(vec![
                Span::styled("  y", Style::default().fg(Color::Black).bg(Color::Cyan)),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled("n", Style::default().fg(Color::Black).bg(Color::Cyan)),
                Span::styled(" or ↑↓+Enter  ·  ", Style::default().fg(Color::DarkGray)),
                Span::styled("Esc", Style::default().fg(Color::Black).bg(Color::Gray)),
                Span::styled(" back", Style::default().fg(Color::DarkGray)),
            ]),
        ];
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Magenta))
            .title(" ⚡ Optional speed-up ");
        frame.render_widget(
            Paragraph::new(text).block(block).wrap(Wrap { trim: false }),
            area,
        );
    }

    fn draw_confirm(&self, frame: &mut Frame, area: Rect) {
        let Some(pending) = self.pending else {
            frame.render_widget(
                Paragraph::new("Nothing selected yet — pick a model first.")
                    .block(Block::default().borders(Borders::ALL).title(" Confirm ")),
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
                "{} (shared HF cache)",
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
                Span::styled(
                    format!("  {label:<12}"),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(value, Style::default().fg(Color::White)),
            ])
        };
        let mut lines = vec![
            Line::from(Span::styled(
                "  ─── Download summary ───",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            row("Model", format!("{} {}", family.key, variant.precision())),
            row(
                "Speed-up",
                if variant.mtp_alias.is_none() {
                    "not available for this model".into()
                } else if pending.with_mtp {
                    "⚡ MTP included".into()
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
        lines.push(Line::raw(""));
        match fit {
            RamFit::TooLarge => lines.push(Line::from(vec![
                Span::styled("  ⚠ ", Style::default().fg(Color::Black).bg(Color::Red)),
                Span::raw(" "),
                Span::styled(
                    "This model likely exceeds this machine's memory and may not serve reliably.",
                    Style::default().fg(Color::Red),
                ),
            ])),
            RamFit::Tight => lines.push(Line::from(vec![
                Span::styled("  ⚠ ", Style::default().fg(Color::Black).bg(Color::Yellow)),
                Span::raw(" "),
                Span::styled(
                    "Fits, but leaves little headroom — expect memory pressure under load.",
                    Style::default().fg(Color::Yellow),
                ),
            ])),
            _ => {}
        }
        if let (Some(free), Some(total)) = (free, total)
            && total > free
        {
            lines.push(Line::from(vec![
                Span::styled("  ⚠ ", Style::default().fg(Color::Black).bg(Color::Red)),
                Span::raw(" "),
                Span::styled(
                    "Not enough free disk space for this download.",
                    Style::default().fg(Color::Red),
                ),
            ]));
        }
        if variant.installed && !pending.with_mtp {
            lines.push(Line::from(vec![
                Span::styled("  ✓ ", Style::default().fg(Color::Black).bg(Color::Green)),
                Span::raw(" "),
                Span::styled(
                    "Already installed — confirming re-checks the cached copy (fast).",
                    Style::default().fg(Color::Green),
                ),
            ]));
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(" Enter ", Style::default().fg(Color::Black).bg(Color::Cyan)),
            Span::styled(" start download    ", Style::default().fg(Color::DarkGray)),
            Span::styled(" c ", Style::default().fg(Color::Black).bg(Color::Gray)),
            Span::styled(" folder    ", Style::default().fg(Color::DarkGray)),
            Span::styled(" d ", Style::default().fg(Color::Black).bg(Color::Gray)),
            Span::styled(" default    ", Style::default().fg(Color::DarkGray)),
            Span::styled(" Esc ", Style::default().fg(Color::Black).bg(Color::Gray)),
            Span::styled(" back", Style::default().fg(Color::DarkGray)),
        ]));
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(Color::Cyan))
                        .title(" ✓ Confirm download "),
                )
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
            .map(|err| Line::from(Span::styled(err.clone(), Style::default().fg(Color::Red))))
            .unwrap_or_else(|| {
                Line::from(Span::styled(
                    "Enter open · s use this folder · d default cache · ~ home · Esc cancel",
                    Style::default().fg(Color::DarkGray),
                ))
            });
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    " Download into a custom folder ",
                    Style::default().add_modifier(Modifier::BOLD),
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
