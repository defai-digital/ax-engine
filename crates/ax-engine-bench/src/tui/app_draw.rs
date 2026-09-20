//! Rendering for the TUI `App`: the top-level frame layout, status/tab/footer
//! chrome, the inline help overlay, and modal dialogs.
//!
//! Split out of the `tui` root, which had grown into a single ~2.2k-line
//! `impl App`. Everything here is reached only through `draw`, so the helpers
//! stay private to this module.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::catalog;
use super::theme;
use super::widgets;
use super::{App, Modal, Screen, ServeFocus, TABS, WizardStage, screen_index};

impl App {
    /// Smallest usable terminal; below this we show a resize hint instead of
    /// silently clamping every panel to zero height.
    const MIN_TERM_WIDTH: u16 = 60;
    const MIN_TERM_HEIGHT: u16 = 15;

    pub fn draw(&self, frame: &mut Frame) {
        let term = frame.area();
        if term.width < Self::MIN_TERM_WIDTH || term.height < Self::MIN_TERM_HEIGHT {
            let popup = widgets::centered_rect(44.min(term.width), 4.min(term.height), term);
            let lines = vec![
                Line::from(Span::styled("terminal too small", theme::warn())),
                Line::from(Span::styled(
                    format!(
                        "need at least {}×{} — resize to continue",
                        Self::MIN_TERM_WIDTH,
                        Self::MIN_TERM_HEIGHT
                    ),
                    theme::label(),
                )),
            ];
            frame.render_widget(Paragraph::new(lines), popup);
            return;
        }
        // Cleared each frame; the active content list re-records it.
        self.content_list_rect.set(Rect::default());
        self.content_list_offset.set(0);
        self.log_rect.set(Rect::default());
        self.banner_rect.set(Rect::default());
        self.hero_rect.set(Rect::default());
        self.modal_hits.set(widgets::ModalHits::default());
        let outer = Layout::vertical([
            Constraint::Length(2), // tab bar + separator
            Constraint::Min(0),    // content
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

        // Build compact status spans and per-tab badges for the tab bar.
        let status_spans = self.build_status_spans();
        let badges = self.build_tab_badges();
        let hits = widgets::draw_tab_bar(
            frame,
            outer[0],
            &TABS,
            screen_index(self.screen),
            status_spans,
            &badges,
            self.focus_tabs,
        );
        self.tab_hits.set(hits);

        match self.screen {
            Screen::Home => self.draw_home(frame, outer[1]),
            Screen::Models => self.draw_models(frame, outer[1]),
            Screen::Downloads => self.draw_downloads(frame, outer[1]),
            Screen::Serve => self.draw_serve(frame, outer[1]),
            Screen::Chat => self.draw_chat(frame, outer[1]),
        }

        frame.render_widget(
            Paragraph::new(self.footer_line()).style(Style::default().fg(theme::colors().dim)),
            outer[2],
        );
        widgets::draw_toasts(frame, frame.area(), &self.toasts);
        if self.show_help {
            self.draw_help(frame, frame.area());
        }
        if let Some(modal) = &self.modal {
            self.draw_modal(frame, frame.area(), modal);
        }
    }

    /// Build compact status spans for the right side of the tab bar.
    fn build_status_spans(&self) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let status = self.server_status();
        spans.push(Span::styled(
            status.label(),
            Style::default().fg(status.color()),
        ));
        if let Some((dl_text, dl_color)) = self.active_download_summary() {
            // Compact form drops the speed suffix but keeps the percent sign.
            let short_dl = match dl_text.split_once('%') {
                Some((before, _)) => format!("  ↓ {}%", before.trim()),
                None => format!("  ↓ {dl_text}"),
            };
            spans.push(Span::raw("  "));
            spans.push(Span::styled(short_dl, Style::default().fg(dl_color)));
        }
        spans
    }

    /// Compact badges drawn inside tab labels (download count, serve live).
    fn build_tab_badges(&self) -> Vec<Option<widgets::TabBadge>> {
        let active_dl = self
            .downloads
            .iter()
            .filter(|t| t.is_running() || t.is_queued())
            .count();
        let ready_dl = self.downloads.iter().filter(|t| t.is_ready()).count();
        let downloads_badge = if active_dl > 0 {
            Some(widgets::TabBadge {
                text: format!("·{active_dl}"),
                style: Style::default().fg(theme::colors().accent),
            })
        } else if ready_dl > 0 {
            Some(widgets::TabBadge {
                text: format!("·{ready_dl}"),
                style: Style::default().fg(theme::colors().ok),
            })
        } else {
            None
        };
        let serve_badge = if self.server_ready {
            Some(widgets::TabBadge {
                text: theme::icon::running().into(),
                style: Style::default().fg(theme::colors().ok),
            })
        } else if self.server_running() {
            Some(widgets::TabBadge {
                text: theme::icon::queued().into(),
                style: Style::default().fg(theme::colors().warn),
            })
        } else {
            None
        };
        vec![
            None, // Home
            None, // Models
            downloads_badge,
            serve_badge,
            None, // Chat
        ]
    }

    fn footer_line(&self) -> Line<'static> {
        use theme::{key_hint, key_label, key_sep};
        let hints: Vec<Span> = if self.modal.is_some() {
            vec![key_hint("Esc"), key_label(" close")]
        } else if self.focus_tabs {
            vec![
                key_hint("←→"),
                key_label(" screen"),
                key_sep(),
                key_hint("↓/Enter"),
                key_label(" content"),
                key_sep(),
                key_hint("1-5"),
                key_label(" jump"),
                key_sep(),
                key_hint("Esc"),
                key_label(" content"),
            ]
        } else {
            match self.screen {
                Screen::Home => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" select"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                    key_sep(),
                    key_hint("q"),
                    key_label(" quit"),
                ],
                Screen::Models => match self.stage {
                    WizardStage::Families if self.filtering => vec![
                        Span::styled(
                            "type to filter",
                            Style::default().fg(theme::colors().accent),
                        ),
                        key_sep(),
                        key_hint("Enter/Esc"),
                        key_label(" done — keeps the filter"),
                    ],
                    WizardStage::Families => vec![
                        key_hint("↑↓"),
                        key_label(" move"),
                        key_sep(),
                        key_hint("Enter"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("/"),
                        key_label(" filter"),
                        key_sep(),
                        key_hint("d"),
                        key_label(" by link"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" home"),
                    ],
                    WizardStage::Precision => vec![
                        key_hint("↑↓"),
                        key_label(" move"),
                        key_sep(),
                        key_hint("Enter"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("x"),
                        key_label(" delete"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" back"),
                    ],
                    WizardStage::Confirm => vec![
                        key_hint("Enter"),
                        key_label(" download"),
                        key_sep(),
                        key_hint("f"),
                        key_label(" folder"),
                        key_sep(),
                        key_hint("r"),
                        key_label(" default"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" back"),
                    ],
                },
                Screen::Downloads => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" serve"),
                    key_sep(),
                    key_hint("r"),
                    key_label(" retry"),
                    key_sep(),
                    key_hint("o"),
                    key_label(" open"),
                    key_sep(),
                    key_hint("⌫"),
                    key_label(" remove"),
                    key_sep(),
                    key_hint("d"),
                    key_label(" clear"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" cancel"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Serve
                    if matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port) =>
                {
                    vec![
                        Span::styled("type to edit", Style::default().fg(theme::colors().accent)),
                        key_sep(),
                        key_hint("Tab"),
                        key_label(" next"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" done"),
                    ]
                }
                Screen::Serve if self.server_ready => vec![
                    key_hint("Enter"),
                    key_label(" chat/switch"),
                    key_sep(),
                    key_hint("c"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" stop"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Serve => vec![
                    key_hint("↑↓"),
                    key_label(" move"),
                    key_sep(),
                    key_hint("Enter"),
                    key_label(" start"),
                    key_sep(),
                    key_hint("x"),
                    key_label(" stop"),
                    key_sep(),
                    key_hint("c"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("t"),
                    key_label(" chat"),
                    key_sep(),
                    key_hint("PgUp/Dn"),
                    key_label(" log"),
                    key_sep(),
                    key_hint("Tab"),
                    key_label(" fields"),
                    key_sep(),
                    key_hint("b"),
                    key_label(" next step"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" back"),
                ],
                Screen::Chat
                    if !self.server_ready
                        && self.server.as_ref().is_some_and(|job| job.done.is_some()) =>
                {
                    // Read-only transcript after a crash/stop.
                    vec![
                        key_hint("PgUp"),
                        key_label(" scroll"),
                        key_sep(),
                        key_hint("Ctrl+Y"),
                        key_label(" copy"),
                        key_sep(),
                        key_hint("Ctrl+T"),
                        key_label(" thinking"),
                        key_sep(),
                        key_hint("4"),
                        key_label(" Serve"),
                        key_sep(),
                        key_hint("Esc"),
                        key_label(" leave"),
                    ]
                }
                Screen::Chat if !self.server_ready => vec![
                    key_hint("4"),
                    key_label(" Serve"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" home"),
                ],
                Screen::Chat => vec![
                    key_hint("Enter"),
                    key_label(" send"),
                    key_sep(),
                    key_hint("Ctrl+J"),
                    key_label(" newline"),
                    key_sep(),
                    key_hint("↑"),
                    key_label(" history"),
                    key_sep(),
                    key_hint("PgUp"),
                    key_label(" scroll"),
                    key_sep(),
                    key_hint("Ctrl+Y"),
                    key_label(" copy"),
                    key_sep(),
                    key_hint("Ctrl+R"),
                    key_label(" retry"),
                    key_sep(),
                    key_hint("Ctrl+T"),
                    key_label(" thinking"),
                    key_sep(),
                    key_hint("/"),
                    key_label(" cmds"),
                    key_sep(),
                    key_hint("Esc"),
                    key_label(" leave"),
                ],
            }
        };
        let mut out = vec![Span::raw("  ")];
        out.extend(hints);
        Line::from(out)
    }

    fn draw_help(&self, frame: &mut Frame, area: Rect) {
        let popup = widgets::centered_rect(74, 22, area);
        let contextual = match self.screen {
            Screen::Home => "Home: ↑↓ move · Enter run the highlighted action",
            Screen::Models => {
                "Models: wizard steps · / filter · d download by link · Enter next · Esc back"
            }
            Screen::Downloads => {
                "Downloads: Enter serve when ready · x cancel · PgUp/PgDn scroll log"
            }
            Screen::Serve => {
                "Serve: Enter start/switch · x stop · c copy URL · t chat · Tab fields · PgUp/PgDn scroll log"
            }
            Screen::Chat => {
                "Chat: Enter send · ↑ history · Ctrl+J newline · Ctrl+U clear draft · Ctrl+Y copy · Ctrl+R retry · Ctrl+L clear · Ctrl+T thinking · / commands · PgUp/PgDn scroll"
            }
        };
        let lines = vec![
            Line::from(Span::styled(
                "AX Engine",
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(theme::colors().accent),
            )),
            Line::raw(""),
            Line::from(Span::styled(
                contextual,
                Style::default().fg(theme::colors().text),
            )),
            Line::raw(""),
            Line::raw("Screens: 1-5 (or click tabs). While typing, use Ctrl+1-5."),
            Line::raw("  1 Home        this Mac + quick start"),
            Line::raw("  2 Models      pick family → size → confirm"),
            Line::raw("  3 Downloads   queue with live progress"),
            Line::raw("  4 Serve       start/stop the local server"),
            Line::raw("  5 Chat        talk to the running model"),
            Line::raw(""),
            Line::raw("Fit badges compare download size to this Mac's memory."),
            Line::raw("Nothing downloads until you confirm. Partials resume later."),
            Line::raw(""),
            Line::raw("b runs the highlighted next step on Home / Downloads / Serve."),
            Line::raw("Log panes (Downloads / Serve) scroll with PgUp/PgDn or the mouse wheel;"),
            Line::raw("scrolling back to the bottom re-pins to the newest output."),
            Line::raw(""),
            Line::from(Span::styled(
                "Any key closes help. Esc steps back one level; q quits.",
                Style::default().fg(theme::colors().muted),
            )),
        ];
        frame.render_widget(
            Paragraph::new("").style(Style::default().bg(theme::colors().scrim_bg)),
            area,
        );
        frame.render_widget(ratatui::widgets::Clear, popup);
        frame.render_widget(
            Paragraph::new(lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(theme::colors().accent))
                        .title(Span::styled(" Help ", theme::title())),
                )
                .wrap(Wrap { trim: false }),
            popup,
        );
    }

    fn draw_modal(&self, frame: &mut Frame, area: Rect, modal: &Modal) {
        let hits = match modal {
            Modal::Quit { downloads, server } => {
                let mut lines = Vec::new();
                if *downloads > 0 {
                    lines.push(Line::raw(format!(
                        "{downloads} download{} still in progress — partial data resumes on the next run.",
                        if *downloads == 1 { "" } else { "s" }
                    )));
                }
                if *server {
                    lines.push(Line::raw("The server is running and will be stopped."));
                }
                lines.push(Line::raw(""));
                lines.push(Line::from(Span::styled(
                    "Quit AX Engine?",
                    Style::default().add_modifier(Modifier::BOLD),
                )));
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Quit",
                    lines,
                    vec![
                        theme::key_chip_danger("y quit"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc stay"),
                    ],
                    theme::colors().warn,
                )
            }
            Modal::ServeReady { download_idx } => {
                let label = self
                    .downloads
                    .get(*download_idx)
                    .map(|t| t.label.clone())
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "✓ Serve model",
                    vec![Line::raw(format!("Start the server with {label}?"))],
                    vec![
                        theme::key_chip("y serve"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc not now"),
                    ],
                    theme::colors().ok,
                )
            }
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => {
                let label = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        format!(
                            "{} {}",
                            self.families[*family_idx].display_name(),
                            v.precision()
                        )
                    })
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "✓ Already installed",
                    vec![
                        Line::raw(format!("{label} is already downloaded.")),
                        Line::raw("Serve it now instead?"),
                    ],
                    vec![
                        theme::key_chip("y serve"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc back"),
                    ],
                    theme::colors().ok,
                )
            }
            Modal::CancelDownload { download_idx } => {
                let label = self
                    .downloads
                    .get(*download_idx)
                    .map(|t| t.label.clone())
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Cancel download",
                    vec![Line::raw(format!(
                        "Stop downloading {label}? Partial data resumes if you retry later."
                    ))],
                    vec![
                        theme::key_chip_danger("y cancel"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc keep"),
                    ],
                    theme::colors().warn,
                )
            }
            Modal::DeleteModel {
                family_idx,
                variant_idx,
                typed,
            } => {
                let (label, path, size, profile_label) = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        (
                            format!(
                                "{} {}",
                                self.families[*family_idx].display_name(),
                                v.precision()
                            ),
                            catalog::repo_cache_dir(v.profile.repo_id)
                                .display()
                                .to_string(),
                            catalog::format_bytes(v.size),
                            v.profile.label,
                        )
                    })
                    .unwrap_or_default();
                // Non-blocking guard rails: warn when the target is in use.
                // `server_model` is the display label for download-served
                // models and the profile label for direct serves — check both.
                let is_served = self.server_running()
                    && self
                        .server_model
                        .as_deref()
                        .is_some_and(|m| m == label || m == profile_label);
                let has_active_download = !label.is_empty()
                    && self
                        .downloads
                        .iter()
                        .any(|t| (t.is_running() || t.is_queued()) && t.label == label);
                let armed = typed == "delete";
                let typed_style = if armed {
                    Style::default().fg(theme::colors().ok)
                } else {
                    Style::default().fg(theme::colors().warn)
                };
                let mut lines = vec![
                    Line::raw(format!("Remove {label} ({size}) from disk?")),
                    Line::from(Span::styled(
                        path,
                        Style::default().fg(theme::colors().muted),
                    )),
                ];
                if is_served {
                    lines.push(Line::from(Span::styled(
                        "⚠ this model is currently served",
                        theme::warn(),
                    )));
                }
                if has_active_download {
                    lines.push(Line::from(Span::styled(
                        "⚠ download in progress",
                        theme::warn(),
                    )));
                }
                lines.push(Line::raw(""));
                lines.push(Line::from(vec![
                    Span::raw("Type 'delete' to confirm: "),
                    Span::styled(format!("{typed}_"), typed_style),
                ]));
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Delete model",
                    lines,
                    if armed {
                        vec![
                            theme::key_chip_danger("Enter delete"),
                            theme::key_sep(),
                            theme::key_chip_dim("Esc keep"),
                        ]
                    } else {
                        vec![theme::key_chip_dim("Esc keep")]
                    },
                    theme::colors().danger,
                )
            }
            Modal::StopServer => {
                let (title, body, confirm) = if self.external_server && !self.managed_server_alive()
                {
                    (
                        "⚠ Detach external server",
                        "This server was started outside the TUI. Detach from it? (process keeps running)",
                        "y detach",
                    )
                } else {
                    ("⚠ Stop server", "Stop the running server?", "y stop")
                };
                widgets::draw_modal_with(
                    frame,
                    area,
                    title,
                    vec![Line::raw(body)],
                    vec![
                        theme::key_chip_danger(confirm),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc keep"),
                    ],
                    theme::colors().warn,
                )
            }
            Modal::RestartServer {
                family_idx,
                variant_idx,
            } => {
                let label = self
                    .families
                    .get(*family_idx)
                    .and_then(|f| f.variants.get(*variant_idx))
                    .map(|v| {
                        format!(
                            "{} {}",
                            self.families[*family_idx].display_name(),
                            v.precision()
                        )
                    })
                    .unwrap_or_default();
                widgets::draw_modal_with(
                    frame,
                    area,
                    "⚠ Restart server",
                    vec![
                        Line::raw(format!("Restart the server with {label}?")),
                        Line::raw("The current model stops."),
                    ],
                    vec![
                        theme::key_chip_danger("y restart"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc keep"),
                    ],
                    theme::colors().warn,
                )
            }
            Modal::ClearChat => widgets::draw_modal_with(
                frame,
                area,
                "⚠ Clear chat",
                vec![Line::raw(
                    "Clear the whole transcript? This cannot be undone.",
                )],
                vec![
                    theme::key_chip_danger("y clear"),
                    theme::key_sep(),
                    theme::key_chip_dim("Esc keep"),
                ],
                theme::colors().warn,
            ),
            Modal::DestPicker(picker) => {
                self.draw_dest_picker(frame, area, picker);
                widgets::ModalHits {
                    popup: widgets::centered_rect(
                        72.min(area.width.saturating_sub(2)),
                        22.min(area.height.saturating_sub(2)),
                        area,
                    ),
                    ..Default::default()
                }
            }
            Modal::DownloadByLink { input, error } => {
                let mut lines = vec![
                    Line::raw("Hugging Face link or owner/repo id (optionally @revision):"),
                    Line::from(Span::styled(
                        format!("{input}_"),
                        Style::default().fg(theme::colors().text),
                    )),
                ];
                if let Some(message) = error {
                    lines.push(Line::raw(""));
                    lines.push(Line::from(Span::styled(message.clone(), theme::danger())));
                }
                widgets::draw_modal_with(
                    frame,
                    area,
                    "Download by link",
                    lines,
                    vec![
                        theme::key_chip("Enter download"),
                        theme::key_sep(),
                        theme::key_chip_dim("Esc cancel"),
                    ],
                    theme::colors().accent,
                )
            }
        };
        self.modal_hits.set(hits);
    }
}
