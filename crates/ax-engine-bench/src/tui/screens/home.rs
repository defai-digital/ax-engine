//! Home screen: guided dashboard with a primary path and machine summary.
//! First run emphasizes a full-width Quick start card; returning users get
//! a compact two-column layout with a journey banner when a next step is ready.

use ratatui::Frame;
use ratatui::crossterm::event::KeyCode;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, ListItem, Paragraph};

use crate::tui::catalog::{self, RamFit, installed_variants};
use crate::tui::theme;
use crate::tui::widgets::{self, ToastLevel};
use crate::tui::{App, Modal, Screen, WizardStage};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HomeAction {
    QuickStart,
    Browse,
    Serve,
    Chat,
    Help,
}

impl App {
    /// (label, action) rows for the Home action list.
    pub(crate) fn home_actions(&self) -> Vec<(String, HomeAction)> {
        let mut actions = Vec::new();
        let quick = match self.quick_start_target() {
            Some((fi, vi)) => {
                let family = &self.families[fi];
                let variant = &family.variants[vi];
                if variant.installed {
                    format!(
                        "Serve {} {} (installed)",
                        family.display_name(),
                        variant.precision()
                    )
                } else {
                    format!(
                        "Download {} {} ({})",
                        family.display_name(),
                        variant.precision(),
                        catalog::format_approx_bytes(variant.size_estimate()),
                    )
                }
            }
            None => "Browse models".to_string(),
        };
        actions.push((format!("Quick start — {quick}"), HomeAction::QuickStart));
        actions.push(("Browse all models".into(), HomeAction::Browse));
        if !installed_variants(&self.families).is_empty() {
            actions.push(("Serve an installed model".into(), HomeAction::Serve));
        }
        if self.server_ready {
            actions.push(("Open chat".into(), HomeAction::Chat));
        }
        actions.push(("Help".into(), HomeAction::Help));
        actions
    }

    /// Smallest catalog variant that fits in RAM.
    pub(crate) fn quick_start_target(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, u64, RamFit)> = None;
        for (fi, family) in self.families.iter().enumerate() {
            for (vi, variant) in family.variants.iter().enumerate() {
                let Some(size) = variant.size_estimate() else {
                    continue;
                };
                let fit = catalog::ram_fit(Some(size), self.hardware.total_ram_bytes);
                if fit == RamFit::TooLarge {
                    continue;
                }
                let better = match &best {
                    None => true,
                    Some((_, _, best_size, best_fit)) => {
                        let rank = |f: RamFit| if f == RamFit::Fits { 0 } else { 1 };
                        (rank(fit), size) < (rank(*best_fit), *best_size)
                    }
                };
                if better {
                    best = Some((fi, vi, size, fit));
                }
            }
        }
        best.map(|(fi, vi, _, _)| (fi, vi))
    }

    /// Whether `(family_idx, variant_idx)` is the same pick as Quick start.
    pub(crate) fn is_recommended_variant(&self, family_idx: usize, variant_idx: usize) -> bool {
        self.quick_start_target()
            .is_some_and(|(fi, vi)| fi == family_idx && vi == variant_idx)
    }

    /// Guided next-step copy for Home / Downloads banners.
    pub(crate) fn journey_next_step(&self) -> Option<(ToastLevel, String)> {
        if self.server_ready {
            return Some((
                ToastLevel::Success,
                "Server ready — press 5 Chat (or Open chat) to try the model".into(),
            ));
        }
        if self.server_running() {
            return Some((
                ToastLevel::Warning,
                "Server starting — large models can take a minute…".into(),
            ));
        }
        if let Some(task) = self.downloads.iter().find(|t| t.is_ready()) {
            return Some((
                ToastLevel::Success,
                format!(
                    "{} ready — press 3 Downloads, then Enter to serve",
                    task.label
                ),
            ));
        }
        if let Some(task) = self.downloads.iter().find(|t| t.is_running()) {
            let pct = task
                .progress_ratio()
                .map(|r| format!(" · {:.0}%", r * 100.0))
                .unwrap_or_default();
            return Some((
                ToastLevel::Info,
                format!("Downloading {}{pct} — press 3 for progress", task.label),
            ));
        }
        if installed_variants(&self.families).is_empty() {
            return Some((
                ToastLevel::Info,
                "Get started — Enter on Quick start to download a model that fits this Mac".into(),
            ));
        }
        None
    }

    pub(crate) fn on_key_home(&mut self, code: KeyCode) {
        let actions = self.home_actions();
        // Clamp selection if the action list shrank (e.g. after server stop).
        if self.home_idx >= actions.len() {
            self.home_idx = actions.len().saturating_sub(1);
        }
        match code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.home_idx == 0 {
                    self.focus_tab_bar();
                } else {
                    self.home_idx = self.home_idx.saturating_sub(1);
                }
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.home_idx = (self.home_idx + 1).min(actions.len().saturating_sub(1));
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                match actions.get(self.home_idx).map(|(_, action)| *action) {
                    Some(HomeAction::QuickStart) => self.quick_start(),
                    Some(HomeAction::Browse) => {
                        self.stage = WizardStage::Families;
                        self.screen = Screen::Models;
                    }
                    Some(HomeAction::Serve) => self.screen = Screen::Serve,
                    Some(HomeAction::Chat) => self.screen = Screen::Chat,
                    Some(HomeAction::Help) => self.show_help = true,
                    None => {}
                }
            }
            // One level back from Home content → tab bar (never quit).
            KeyCode::Esc => self.focus_tab_bar(),
            _ => {}
        }
    }

    /// Jump the wizard straight to the recommended model.
    fn quick_start(&mut self) {
        let Some((fi, vi)) = self.quick_start_target() else {
            self.stage = WizardStage::Families;
            self.screen = Screen::Models;
            return;
        };
        self.family_idx = fi;
        self.precision_idx = vi;
        let variant = &self.families[fi].variants[vi];
        if variant.installed && variant.mtp_alias.is_none() {
            self.modal = Some(Modal::ServeInstalled {
                family_idx: fi,
                variant_idx: vi,
            });
            return;
        }
        self.screen = Screen::Models;
        if variant.mtp_alias.is_some() {
            self.mtp_idx = 0;
            self.stage = WizardStage::Options;
        } else {
            self.begin_confirm(false);
        }
    }

    pub(crate) fn draw_home(&self, frame: &mut Frame, area: Rect) {
        let first_run = installed_variants(&self.families).is_empty()
            && self.downloads.is_empty()
            && !self.server_running();

        let has_banner = self.journey_next_step().is_some();
        let top = if has_banner {
            Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area)
        } else {
            Layout::vertical([Constraint::Length(0), Constraint::Min(0)]).split(area)
        };
        if let Some((kind, text)) = self.journey_next_step() {
            widgets::draw_banner(frame, top[0], kind, &text);
        }

        let body = top[1];
        if first_run {
            self.draw_home_first_run(frame, body);
        } else {
            self.draw_home_returning(frame, body);
        }
    }

    /// Full-width hero: primary CTA, then compact machine strip + secondary actions.
    fn draw_home_first_run(&self, frame: &mut Frame, area: Rect) {
        let rows = Layout::vertical([
            Constraint::Length(7),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(area);

        self.draw_home_hero(frame, rows[0]);
        self.draw_home_machine_strip(frame, rows[1]);
        self.draw_home_actions(frame, rows[2], true);
    }

    fn draw_home_returning(&self, frame: &mut Frame, area: Rect) {
        let columns = Layout::horizontal([Constraint::Percentage(46), Constraint::Percentage(54)])
            .split(area);
        let left = Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).split(columns[0]);
        self.draw_home_hardware(frame, left[0]);
        self.draw_home_installed(frame, left[1]);
        self.draw_home_actions(frame, columns[1], false);
    }

    fn draw_home_hero(&self, frame: &mut Frame, area: Rect) {
        let (title, detail, fit_line) = match self.quick_start_target() {
            Some((fi, vi)) => {
                let family = &self.families[fi];
                let variant = &family.variants[vi];
                let fit = catalog::ram_fit(variant.size_estimate(), self.hardware.total_ram_bytes);
                (
                    format!(
                        "Quick start  ·  {} {}",
                        family.display_name(),
                        variant.precision()
                    ),
                    format!(
                        "Download {} · {} · Enter to begin",
                        catalog::format_approx_bytes(variant.size_estimate()),
                        fit.plain()
                    ),
                    fit.plain().to_string(),
                )
            }
            None => (
                "Quick start".into(),
                "Browse the model catalog and pick a size that fits.".into(),
                String::new(),
            ),
        };
        let selected = self.home_idx == 0;
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if selected {
                theme::ACCENT
            } else {
                theme::BORDER_INACTIVE
            }))
            .title(Span::styled(" Start here ", theme::title()));
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let lines = vec![
            Line::from(Span::styled(
                format!("  {}", title),
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::raw(""),
            Line::from(Span::styled(format!("  {detail}"), theme::body_dim())),
            if fit_line.is_empty() {
                Line::raw("")
            } else {
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(
                        format!(" {} recommended for this Mac ", theme::icon::STAR),
                        Style::default().fg(theme::ON_ACCENT).bg(theme::ACCENT),
                    ),
                ])
            },
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    if selected {
                        " Enter  run  "
                    } else {
                        " ↑↓  then Enter  "
                    },
                    theme::cta(),
                ),
            ]),
        ];
        frame.render_widget(Paragraph::new(lines), inner);
        // Click target = hero + action list share; set list rect to action area only later.
    }

    fn draw_home_machine_strip(&self, frame: &mut Frame, area: Rect) {
        let ram = self
            .hardware
            .total_ram_bytes
            .map(catalog::format_bytes)
            .unwrap_or_else(|| "?".into());
        let disk = self
            .hardware
            .free_disk_bytes
            .map(catalog::format_bytes)
            .unwrap_or_else(|| "?".into());
        let chip = self.hardware.chip.clone().unwrap_or_else(|| "Mac".into());
        let line = Line::from(vec![
            Span::styled("  ", theme::label()),
            Span::styled(
                chip,
                Style::default()
                    .fg(theme::TEXT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  ·  ", theme::label()),
            Span::styled(format!("{ram} memory"), theme::body()),
            Span::styled("  ·  ", theme::label()),
            Span::styled(format!("{disk} free"), theme::body()),
        ]);
        frame.render_widget(
            Paragraph::new(vec![line]).block(widgets::soft_block(" This machine ")),
            area,
        );
    }

    fn draw_home_hardware(&self, frame: &mut Frame, area: Rect) {
        let ram = self
            .hardware
            .total_ram_bytes
            .map(catalog::format_bytes)
            .unwrap_or_else(|| "unknown".into());
        let disk = self
            .hardware
            .free_disk_bytes
            .map(catalog::format_bytes)
            .unwrap_or_else(|| "unknown".into());
        let installed = installed_variants(&self.families).len();
        let lines = vec![
            Line::from(vec![
                Span::styled("  Chip      ", theme::label()),
                Span::styled(
                    self.hardware
                        .chip
                        .clone()
                        .unwrap_or_else(|| "unknown".into()),
                    Style::default()
                        .fg(theme::TEXT)
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Memory    ", theme::label()),
                Span::styled(format!("{ram} unified"), theme::body()),
            ]),
            Line::from(vec![
                Span::styled("  Free disk ", theme::label()),
                Span::styled(disk, theme::body()),
            ]),
            Line::from(vec![
                Span::styled("  Installed ", theme::label()),
                if installed > 0 {
                    Span::styled(
                        format!("{installed} model{}", if installed == 1 { "" } else { "s" }),
                        theme::ok(),
                    )
                } else {
                    Span::styled("no models yet", theme::label())
                },
            ]),
        ];
        frame.render_widget(
            Paragraph::new(lines).block(widgets::soft_block(" This machine ")),
            area,
        );
    }

    fn draw_home_actions(&self, frame: &mut Frame, area: Rect, first_run: bool) {
        let rows: Vec<ListItem> = self
            .home_actions()
            .into_iter()
            .enumerate()
            .map(|(i, (label, action))| {
                let selected = i == self.home_idx;
                let is_primary = action == HomeAction::QuickStart;
                // On first-run the hero owns Quick start; list starts at secondary actions.
                // Still list all actions so keyboard indices stay consistent.
                let style = if is_primary && selected {
                    theme::cta()
                } else if is_primary {
                    Style::default()
                        .fg(theme::ACCENT)
                        .add_modifier(Modifier::BOLD)
                } else if selected {
                    Style::default().fg(theme::ON_ACCENT).bg(theme::ACCENT)
                } else {
                    Style::default().fg(theme::TEXT)
                };
                let prefix = if selected {
                    format!("{} ", theme::icon::SELECT)
                } else {
                    "  ".into()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(label, style),
                ]))
            })
            .collect();
        let title = if first_run { " Also " } else { " Actions " };
        self.content_list_rect.set(area);
        let list = ratatui::widgets::List::new(rows)
            .block(widgets::active_block(title))
            .highlight_style(theme::highlight_active())
            .highlight_symbol("");
        let mut state = ratatui::widgets::ListState::default();
        state.select(Some(self.home_idx));
        frame.render_stateful_widget(list, area, &mut state);
    }

    fn draw_home_installed(&self, frame: &mut Frame, area: Rect) {
        let pairs = installed_variants(&self.families);
        let lines: Vec<Line> = if pairs.is_empty() {
            vec![
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled("No models installed yet", theme::warn()),
                ]),
                Line::raw(""),
                Line::from(vec![
                    Span::raw("  Press "),
                    Span::styled(
                        "Enter",
                        Style::default()
                            .fg(theme::TEXT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" on Quick start, or ", theme::label()),
                    Span::styled(
                        "2",
                        Style::default()
                            .fg(theme::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(" for Models.", theme::label()),
                ]),
            ]
        } else {
            pairs
                .iter()
                .map(|&(fi, vi)| {
                    let family = &self.families[fi];
                    let variant = &family.variants[vi];
                    let fit =
                        catalog::ram_fit(variant.size_estimate(), self.hardware.total_ram_bytes);
                    Line::from(vec![
                        Span::styled(
                            format!("  {} ", theme::icon::RUNNING),
                            Style::default().fg(match fit {
                                RamFit::Fits => theme::OK,
                                RamFit::Tight => theme::WARN,
                                RamFit::TooLarge => theme::DANGER,
                                RamFit::Unknown => theme::MUTED,
                            }),
                        ),
                        Span::styled(
                            format!("{:<18}", family.display_name()),
                            Style::default()
                                .fg(theme::TEXT)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!("{:<10}", variant.precision()), theme::body_dim()),
                        Span::styled(
                            format!("{:<12}", catalog::format_bytes(variant.size)),
                            theme::label(),
                        ),
                        fit_span(fit),
                    ])
                })
                .collect()
        };
        frame.render_widget(
            Paragraph::new(lines).block(widgets::soft_block(" Installed models ")),
            area,
        );
    }
}

/// Colored fit badge used across Home / wizard / Serve rows.
pub(crate) fn fit_span(fit: RamFit) -> Span<'static> {
    let color = match fit {
        RamFit::Fits => theme::OK,
        RamFit::Tight => theme::WARN,
        RamFit::TooLarge => theme::DANGER,
        RamFit::Unknown => theme::MUTED,
    };
    Span::styled(format!("{:<10}", fit.label()), Style::default().fg(color))
}
