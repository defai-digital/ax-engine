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
use crate::tui::{App, Screen, WizardStage};

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
    ///
    /// Returning users (installed models) get **Browse** first so Enter does
    /// not immediately serve/infer. First-run keeps **Quick start** first as
    /// the guided download path.
    pub(crate) fn home_actions(&self) -> Vec<(String, HomeAction)> {
        let mut actions = Vec::new();
        let has_installed = !installed_variants(&self.families).is_empty();
        let quick = match self.quick_start_target() {
            Some((fi, vi)) => {
                let family = &self.families[fi];
                let variant = &family.variants[vi];
                if variant.installed {
                    format!(
                        "Serve {} {} (shortcut)",
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
        let quick_row = (format!("Quick start — {quick}"), HomeAction::QuickStart);

        if has_installed {
            // Safe default order: explore first, explicit serve, optional shortcut.
            actions.push(("Browse all models".into(), HomeAction::Browse));
            actions.push(("Serve an installed model".into(), HomeAction::Serve));
            if self.server_ready {
                actions.push(("Open chat".into(), HomeAction::Chat));
            }
            actions.push(quick_row);
            actions.push(("Help".into(), HomeAction::Help));
        } else {
            actions.push(quick_row);
            actions.push(("Browse all models".into(), HomeAction::Browse));
            actions.push(("Help".into(), HomeAction::Help));
        }
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
                "Server ready — press b or click to open Chat".into(),
            ));
        }
        if self.server_running() {
            return Some((
                ToastLevel::Warning,
                "Server starting — press b or click to open Serve log…".into(),
            ));
        }
        if let Some(task) = self.downloads.iter().find(|t| t.is_ready()) {
            return Some((
                ToastLevel::Success,
                format!("{} ready — press b or click to serve", task.label),
            ));
        }
        if let Some(task) = self.downloads.iter().find(|t| t.is_running()) {
            let pct = task
                .progress_ratio()
                .map(|r| format!(" · {:.0}%", r * 100.0))
                .unwrap_or_default();
            return Some((
                ToastLevel::Info,
                format!(
                    "Downloading {}{pct} — press b or click to watch progress",
                    task.label
                ),
            ));
        }
        if installed_variants(&self.families).is_empty() {
            return Some((
                ToastLevel::Info,
                "Get started — press b or click for Quick start".into(),
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
                // If a journey banner is showing and Quick start is selected,
                // prefer the banner action when it's the natural next step.
                match actions.get(self.home_idx).map(|(_, action)| *action) {
                    Some(HomeAction::QuickStart) => self.quick_start_from_home(),
                    Some(HomeAction::Browse) => {
                        self.stage = WizardStage::Families;
                        self.navigate_to(Screen::Models);
                    }
                    Some(HomeAction::Serve) => self.navigate_to(Screen::Serve),
                    Some(HomeAction::Chat) => self.navigate_to(Screen::Chat),
                    Some(HomeAction::Help) => self.show_help = true,
                    None => {}
                }
            }
            // One level back from Home content → tab bar (never quit).
            KeyCode::Esc => self.focus_tab_bar(),
            _ => {}
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
            self.banner_rect.set(top[0]);
            widgets::draw_banner(frame, top[0], kind, &text);
        }

        let body = top[1];
        if first_run {
            self.draw_home_first_run(frame, body);
        } else {
            self.draw_home_returning(frame, body);
        }
    }

    /// First-run: hero + actions on top; host monitor fills all remaining height.
    fn draw_home_first_run(&self, frame: &mut Frame, area: Rect) {
        let launcher_h = launcher_band_height(area.height);
        // Split launcher ~40/60 hero/actions.
        let hero_h = (launcher_h * 5 / 12)
            .max(4)
            .min(launcher_h.saturating_sub(4));
        let actions_h = launcher_h.saturating_sub(hero_h);
        // Min(0) host = every leftover row goes to chart/procs (no dead space).
        let rows = Layout::vertical([
            Constraint::Length(hero_h),
            Constraint::Length(actions_h),
            Constraint::Min(14),
        ])
        .split(area);

        self.draw_home_hero(frame, rows[0]);
        self.draw_home_actions(frame, rows[1], true);
        super::metrics_panel::draw_live_metrics(frame, rows[2], &self.live_metrics);
    }

    /// Returning: installed + actions on top; host monitor fills all remaining height.
    fn draw_home_returning(&self, frame: &mut Frame, area: Rect) {
        let launcher_h = launcher_band_height(area.height);
        let rows =
            Layout::vertical([Constraint::Length(launcher_h), Constraint::Min(14)]).split(area);

        let top = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(rows[0]);
        self.draw_home_installed(frame, top[0]);
        self.draw_home_actions(frame, top[1], false);
        super::metrics_panel::draw_live_metrics(frame, rows[1], &self.live_metrics);
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
        self.hero_rect.set(area);
        let quick_selected = self
            .home_actions()
            .get(self.home_idx)
            .is_some_and(|(_, a)| *a == HomeAction::QuickStart);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if quick_selected {
                theme::SELECT
            } else {
                theme::BORDER_INACTIVE
            }))
            .title(Span::styled(
                " Start here · press b or click ",
                theme::title(),
            ));
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
                    if quick_selected {
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

    fn draw_home_actions(&self, frame: &mut Frame, area: Rect, first_run: bool) {
        let rows: Vec<ListItem> = self
            .home_actions()
            .into_iter()
            .enumerate()
            .map(|(i, (label, action))| {
                let selected = i == self.home_idx;
                let is_quick = action == HomeAction::QuickStart;
                // Selection always uses the amber cursor bar so focus is obvious.
                // Quick start is no longer styled as a permanent primary CTA when
                // unselected — that looked like the default action on every row.
                let style = if selected {
                    theme::highlight_active()
                } else if is_quick {
                    Style::default().fg(theme::DIM)
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
            // Compact band: show a few rows; overflow summarized (space goes to chart).
            let max_rows = area.height.saturating_sub(2).max(1) as usize;
            let mut lines: Vec<Line> = pairs
                .iter()
                .take(max_rows.saturating_sub(usize::from(pairs.len() > max_rows)))
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
                            widgets::ellipsis(&family.display_name(), 18),
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
                .collect();
            if pairs.len() > lines.len() {
                let more = pairs.len() - lines.len();
                lines.push(Line::from(Span::styled(
                    format!("  … +{more} more · Models tab"),
                    theme::label(),
                )));
            }
            lines
        };
        frame.render_widget(
            Paragraph::new(lines).block(widgets::soft_block(" Installed models ")),
            area,
        );
    }
}

/// Fixed height for installed + actions (readable, not growing with the terminal).
/// Remaining Home rows all go to the host panel so the chart can expand.
fn launcher_band_height(content_h: u16) -> u16 {
    use super::metrics_panel::PREFERRED_HEIGHT;
    // Prefer 8–12 rows for launcher; leave at least PREFERRED_HEIGHT for host when possible.
    let reserve_host = PREFERRED_HEIGHT.min(content_h.saturating_sub(8));
    let max = content_h.saturating_sub(reserve_host).min(12);
    max.max(8).min(content_h.saturating_sub(10).max(4))
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
