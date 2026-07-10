//! Home screen: hardware summary, engine status, installed models, and the
//! guided entry points (Quick start first for new users).

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, ListItem, Paragraph};

use crate::tui::catalog::{self, RamFit, installed_variants};
use crate::tui::{App, Modal, Screen, WizardStage, widgets};
use ratatui::crossterm::event::KeyCode;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HomeAction {
    QuickStart,
    Browse,
    Serve,
    Help,
}

impl App {
    /// (label, action) rows for the Home action list.  Quick start resolves to
    /// the smallest model that fits this machine's RAM.
    pub(crate) fn home_actions(&self) -> Vec<(String, HomeAction)> {
        let mut actions = Vec::new();
        let quick = match self.quick_start_target() {
            Some((fi, vi)) => {
                let family = &self.families[fi];
                let variant = &family.variants[vi];
                if variant.installed {
                    format!(
                        "Quick start — serve {} {} (installed)",
                        family.key,
                        variant.precision()
                    )
                } else {
                    format!(
                        "Quick start — download {} {} ({})",
                        family.key,
                        variant.precision(),
                        catalog::format_approx_bytes(variant.size_estimate()),
                    )
                }
            }
            None => "Quick start".to_string(),
        };
        actions.push((quick, HomeAction::QuickStart));
        actions.push(("Browse all models".into(), HomeAction::Browse));
        actions.push(("Serve an installed model".into(), HomeAction::Serve));
        actions.push(("Help".into(), HomeAction::Help));
        actions
    }

    /// Smallest catalog variant that fits in RAM (preferring a comfortable
    /// fit over a tight one), as (family_idx, variant_idx).
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

    pub(crate) fn on_key_home(&mut self, code: KeyCode) {
        let actions = self.home_actions();
        match code {
            KeyCode::Up | KeyCode::Char('k') => self.home_idx = self.home_idx.saturating_sub(1),
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
                    Some(HomeAction::Help) => self.show_help = true,
                    None => {}
                }
            }
            _ => {}
        }
    }

    /// Jump the wizard straight to the recommended model: installed -> offer
    /// to serve it; MTP-capable -> options step; otherwise -> confirm step.
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
        let chunks = Layout::vertical([
            Constraint::Length(6),
            Constraint::Length(5),
            Constraint::Min(0),
        ])
        .split(area);
        self.draw_home_hardware(frame, chunks[0]);
        self.draw_home_actions(frame, chunks[1]);
        self.draw_home_installed(frame, chunks[2]);
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
                Span::styled("Machine   ", Style::default().fg(Color::DarkGray)),
                Span::raw(
                    self.hardware
                        .chip
                        .clone()
                        .unwrap_or_else(|| "unknown".into()),
                ),
            ]),
            Line::from(vec![
                Span::styled("Memory    ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!("{ram} unified")),
            ]),
            Line::from(vec![
                Span::styled("Free disk ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!("{disk} on the download volume")),
            ]),
            Line::from(vec![
                Span::styled("Installed ", Style::default().fg(Color::DarkGray)),
                Span::raw(format!(
                    "{installed} model{}",
                    if installed == 1 { "" } else { "s" }
                )),
            ]),
        ];
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" This machine "),
            ),
            area,
        );
    }

    fn draw_home_actions(&self, frame: &mut Frame, area: Rect) {
        let first_run = installed_variants(&self.families).is_empty();
        let rows: Vec<ListItem> = self
            .home_actions()
            .into_iter()
            .enumerate()
            .map(|(i, (label, action))| {
                let mut style = Style::default();
                if first_run && action == HomeAction::QuickStart {
                    style = style.fg(Color::Cyan).add_modifier(Modifier::BOLD);
                }
                let _ = i;
                ListItem::new(Line::from(label)).style(style)
            })
            .collect();
        let title = if first_run {
            " Get started — download a model, serve it, then chat with it "
        } else {
            " Actions "
        };
        widgets::render_list(
            frame,
            area,
            title,
            rows,
            self.home_idx,
            true,
            &self.content_list_rect,
        );
    }

    fn draw_home_installed(&self, frame: &mut Frame, area: Rect) {
        let pairs = installed_variants(&self.families);
        let lines: Vec<Line> = if pairs.is_empty() {
            vec![Line::from(Span::styled(
                "No models installed yet — Quick start will guide you.",
                Style::default().fg(Color::Yellow),
            ))]
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
                            format!("{:<18}", family.key),
                            Style::default().add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(format!("{:<10}", variant.precision())),
                        Span::styled(
                            format!("{:<12}", catalog::format_bytes(variant.size)),
                            Style::default().fg(Color::Gray),
                        ),
                        fit_span(fit),
                    ])
                })
                .collect()
        };
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Installed models "),
            ),
            area,
        );
    }
}

/// Colored fit badge used across Home / wizard / Serve rows.
pub(crate) fn fit_span(fit: RamFit) -> Span<'static> {
    let color = match fit {
        RamFit::Fits => Color::Green,
        RamFit::Tight => Color::Yellow,
        RamFit::TooLarge => Color::Red,
        RamFit::Unknown => Color::DarkGray,
    };
    Span::styled(format!("{:<10}", fit.label()), Style::default().fg(color))
}
