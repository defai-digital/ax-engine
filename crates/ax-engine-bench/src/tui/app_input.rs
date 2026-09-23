//! Keyboard, mouse, and paste handling for the TUI `App`: routing input to the
//! active screen, the tab/wizard/modal key maps, and click hit-testing.
//!
//! Split out of the `tui` root, which had grown into a single ~2.2k-line
//! `impl App`. Methods the root or the sibling screens call are `pub(super)`:
//! associated functions are private to the module holding the impl block,
//! unlike `App`'s fields, which stay private to `tui` and are reachable from
//! every module inside it.

use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::layout::Rect;

use super::catalog::installed_variants;
use super::jobs::Job;
use super::widgets;
use super::{App, Modal, SCREENS, Screen, ServeFocus, ToolbarAction, WizardStage, screen_index};

impl App {
    pub(crate) fn on_toolbar_action(&mut self, action: ToolbarAction) {
        use ToolbarAction::*;
        self.focus_tabs = false;
        self.filtering = false;
        match action {
            Back if self.screen == Screen::Models => self.on_key_models(KeyCode::Esc),
            Back => self.back_or_home(),
            Models => self.navigate_to(Screen::Models),
            Library => {
                self.downloads_show_library = true;
                self.reload_local_models();
                self.navigate_to(Screen::Downloads);
            }
            Transfers => self.downloads_show_library = false,
            ChooseSize | Download => self.on_key_models(KeyCode::Enter),
            Refresh if self.screen == Screen::Models => self.start_hub_catalog(),
            Refresh => self.reload_local_models(),
            Serve if self.screen == Screen::Downloads => self.on_key_downloads(KeyCode::Enter),
            Serve => self.on_key_serve(KeyCode::Enter),
            Delete => self.on_key_library(KeyCode::Char('x')),
            Reveal if self.downloads_show_library => self.on_key_library(KeyCode::Char('o')),
            Reveal => self.on_key_downloads(KeyCode::Char('o')),
            Retry => self.on_key_downloads(KeyCode::Char('r')),
            CancelDownload => self.on_key_downloads(KeyCode::Char('x')),
            ToggleLog => self.on_key_downloads(KeyCode::Char('v')),
            StopServer => self.on_key_serve(KeyCode::Char('x')),
        }
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_quit();
            return;
        }
        if self.modal.is_some() {
            self.on_key_modal(key.code);
            return;
        }
        if self.show_help {
            // Any key dismisses help.
            self.show_help = false;
            return;
        }
        // Ctrl+1–5 always switches screens, even while typing in Chat / fields.
        if key.modifiers.contains(KeyModifiers::CONTROL)
            && let KeyCode::Char(c @ '1'..='5') = key.code
        {
            self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
            return;
        }
        // While the tab bar owns focus, navigation shortcuts always apply —
        // do not let Chat/host/filter typing swallow 1–5 / q / arrows.
        if self.focus_tabs {
            match key.code {
                KeyCode::Char('?') => {
                    self.show_help = true;
                    return;
                }
                KeyCode::Char('q') => {
                    self.request_quit();
                    return;
                }
                KeyCode::Char(c @ '1'..='5') => {
                    self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
                    return;
                }
                _ => {}
            }
            if self.on_key_tabs(key.code) {
                return;
            }
        }
        if !self.typing() {
            match key.code {
                KeyCode::Char('?') => {
                    self.show_help = true;
                    return;
                }
                KeyCode::Char('q') => {
                    self.request_quit();
                    return;
                }
                KeyCode::Char(c @ '1'..='5') => {
                    self.goto_screen(SCREENS[(c as usize) - ('1' as usize)]);
                    return;
                }
                // Keyboard path for the guided next-step banner (b = banner).
                // Only on screens that render it; Models wizard and Chat keep
                // `b` as plain text/navigation input.
                KeyCode::Char('b')
                    if matches!(
                        self.screen,
                        Screen::Home | Screen::Downloads | Screen::Serve
                    ) =>
                {
                    self.activate_journey_banner();
                    return;
                }
                _ => {}
            }
        }
        match self.screen {
            Screen::Home => self.on_key_home(key.code),
            Screen::Models => self.on_key_models(key.code),
            Screen::Downloads => self.on_key_downloads(key.code),
            Screen::Serve => self.on_key_serve(key.code),
            Screen::Chat => self.on_key_chat(key),
        }
    }

    /// Switch screens (tab / digit jump), remembering the prior screen for Esc.
    fn goto_screen(&mut self, screen: Screen) {
        if screen == Screen::Downloads {
            self.downloads_show_library = true;
            self.reload_local_models();
        }
        self.navigate_to(screen);
    }

    /// Navigate to a screen, pushing the current one onto the Esc back stack.
    pub(crate) fn navigate_to(&mut self, screen: Screen) {
        if self.screen != screen {
            self.back_stack.push(self.screen);
            // Long sessions must not grow the stack without bound.
            if self.back_stack.len() > 16 {
                self.back_stack.remove(0);
            }
        }
        self.screen = screen;
        self.focus_tabs = false;
    }

    /// Pop the previous screen if any. Returns true when a pop happened.
    pub(crate) fn go_back_screen(&mut self) -> bool {
        if let Some(prev) = self.back_stack.pop() {
            self.screen = prev;
            self.focus_tabs = false;
            true
        } else {
            false
        }
    }

    /// Esc/←/h "back" from a top-level screen: step back one level when there
    /// is history, otherwise land on Home (the one screen with no back).
    pub(crate) fn back_or_home(&mut self) {
        if !self.go_back_screen() {
            self.navigate_to(Screen::Home);
        }
    }

    /// Enable download → serve → chat auto-handoff for guided flows.
    pub(crate) fn enable_auto_chain(&mut self) {
        self.auto_serve_after_download = true;
        self.auto_chat_after_serve = true;
    }

    /// Move keyboard focus onto the top tab bar (1 Home · 2 Models · …).
    /// Leaves any text-entry sub-mode so the bar actually owns the keyboard.
    pub(crate) fn focus_tab_bar(&mut self) {
        self.focus_tabs = true;
        self.filtering = false;
        if matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port) {
            self.serve_focus = ServeFocus::List;
        }
    }

    /// Activate whatever the journey banner is advertising (click / Enter).
    pub(crate) fn activate_journey_banner(&mut self) {
        if self.server_ready {
            self.navigate_to(Screen::Chat);
            return;
        }
        if self.server_running() {
            self.navigate_to(Screen::Serve);
            return;
        }
        if let Some(idx) = self.downloads.iter().position(|t| t.is_ready()) {
            self.select_download(idx);
            self.navigate_to(Screen::Downloads);
            self.modal = Some(Modal::ServeReady { download_idx: idx });
            return;
        }
        if self
            .downloads
            .iter()
            .any(|t| t.is_running() || t.is_queued())
        {
            self.navigate_to(Screen::Downloads);
            return;
        }
        if installed_variants(&self.families).is_empty() {
            self.home_idx = 0;
            self.navigate_to(Screen::Home);
            // Re-enter Home quick start without recursion through banners.
            self.quick_start_from_home();
            return;
        }
        self.navigate_to(Screen::Serve);
    }

    /// Quick start entry used by Home and the journey banner.
    pub(crate) fn quick_start_from_home(&mut self) {
        let Some((fi, vi)) = self.quick_start_target() else {
            self.stage = WizardStage::Families;
            self.navigate_to(Screen::Models);
            return;
        };
        self.family_idx = fi;
        self.precision_idx = vi;
        let installed = self.families[fi].variants[vi].installed;
        if installed {
            // The modal confirm arms auto-chat only once a child launches.
            self.modal = Some(Modal::ServeInstalled {
                family_idx: fi,
                variant_idx: vi,
            });
            return;
        }
        // Guided chain: after confirm, download → serve → chat.
        self.enable_auto_chain();
        self.navigate_to(Screen::Models);
        self.begin_confirm();
    }

    /// Handle keys while the tab bar owns focus. Returns true if consumed.
    fn on_key_tabs(&mut self, code: KeyCode) -> bool {
        let idx = screen_index(self.screen);
        match code {
            KeyCode::Left | KeyCode::Char('h') => {
                if idx > 0 {
                    // Navigate like digit jumps (Esc history stays correct),
                    // but keep bar focus so repeated arrows keep moving.
                    self.goto_screen(SCREENS[idx - 1]);
                    self.focus_tabs = true;
                }
                true
            }
            KeyCode::Right | KeyCode::Char('l') => {
                if idx + 1 < SCREENS.len() {
                    self.goto_screen(SCREENS[idx + 1]);
                    self.focus_tabs = true;
                }
                true
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Enter => {
                self.focus_tabs = false;
                true
            }
            KeyCode::Up | KeyCode::Char('k') => true, // already on the bar
            KeyCode::Esc => {
                self.focus_tabs = false;
                true
            }
            // Any other key leaves the bar but stays consumed — falling
            // through to the screen handler would fire two actions for one
            // keypress (e.g. `x` opening the destructive delete modal).
            _ => {
                self.focus_tabs = false;
                true
            }
        }
    }

    /// The family filter box owns keyboard input only on the Families stage;
    /// the `filtering` flag can go stale across stage/screen switches.
    pub(crate) fn family_filter_active(&self) -> bool {
        self.stage == WizardStage::Families && self.filtering
    }

    /// True while a screen is consuming plain characters (filter, host/port,
    /// chat input), so global single-letter shortcuts must stay inert.
    /// Screen switches remain available via Ctrl+1–5.
    pub(super) fn typing(&self) -> bool {
        match self.screen {
            Screen::Models => self.family_filter_active(),
            Screen::Serve => matches!(self.serve_focus, ServeFocus::Host | ServeFocus::Port),
            Screen::Chat => {
                self.server_ready && !self.server.as_ref().is_some_and(|job| job.done.is_some())
            }
            _ => false,
        }
    }

    /// `q`/Ctrl-C: quit immediately when idle, confirm when jobs are live.
    fn request_quit(&mut self) {
        let downloads = self
            .downloads
            .iter()
            .filter(|t| t.is_running() || t.is_queued())
            .count();
        let server = self.server.as_ref().is_some_and(Job::is_running);
        if downloads == 0 && !server {
            self.quit = true;
        } else {
            self.modal = Some(Modal::Quit { downloads, server });
        }
    }

    fn on_key_modal(&mut self, code: KeyCode) {
        let Some(modal) = self.modal.take() else {
            return;
        };
        match modal {
            Modal::ServeLocal(model) => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    self.auto_chat_after_serve = self.serve_local_model(&model);
                    self.navigate_to(Screen::Serve);
                }
                KeyCode::Esc | KeyCode::Char('n') => {}
                _ => self.modal = Some(Modal::ServeLocal(model)),
            },
            Modal::DeleteLocal(model) => match code {
                KeyCode::Char('y') => self.delete_local_model(&model),
                KeyCode::Esc | KeyCode::Char('n') => {}
                _ => self.modal = Some(Modal::DeleteLocal(model)),
            },
            Modal::Quit { .. } => match code {
                KeyCode::Enter | KeyCode::Char('y') => self.quit = true,
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ServeReady { download_idx } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    // Serve shows the child log (or the failure line and
                    // binary path), so navigate either way; only a launched
                    // child arms the chat handoff.
                    self.auto_chat_after_serve = self.start_server_for_download(download_idx);
                    self.navigate_to(Screen::Serve);
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ServeInstalled {
                family_idx,
                variant_idx,
            } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    let launched = self.serve_installed(family_idx, variant_idx);
                    self.auto_chat_after_serve = launched;
                    self.navigate_to(Screen::Serve);
                    if launched {
                        self.stage = WizardStage::Families;
                        self.pending = None;
                    }
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::CancelDownload { download_idx } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    if let Some(task) = self.downloads.get_mut(download_idx) {
                        task.cancel();
                        let label = task.label.clone();
                        self.toast_warn(format!("{label} cancelled"));
                    }
                    self.start_next_queued_download();
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::DeleteModel {
                family_idx,
                variant_idx,
                mut typed,
            } => match code {
                KeyCode::Esc | KeyCode::Left => {}
                // Nothing typed yet: n/h dismiss like the other confirm modals
                // instead of silently starting the confirm string.
                KeyCode::Char('n') | KeyCode::Char('h') if typed.is_empty() => {}
                KeyCode::Enter if typed == "delete" => {
                    self.delete_installed_variant(family_idx, variant_idx);
                }
                KeyCode::Char(c) => {
                    typed.push(c);
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
                KeyCode::Backspace => {
                    typed.pop();
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
                _ => {
                    self.modal = Some(Modal::DeleteModel {
                        family_idx,
                        variant_idx,
                        typed,
                    });
                }
            },
            Modal::StopServer => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    let external_only = self.external_server && !self.managed_server_alive();
                    self.stop_server();
                    // stop_server toasts for external detach; managed gets the stop toast here.
                    if !external_only {
                        self.toast_warn("server stopped");
                    }
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::RestartServer {
                family_idx,
                variant_idx,
            } => match code {
                KeyCode::Enter | KeyCode::Char('y') => {
                    // Same start path as Serve Enter / ServeInstalled, just
                    // after stopping the currently served model.
                    self.stop_server();
                    self.auto_chat_after_serve = self.serve_installed(family_idx, variant_idx);
                }
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::ClearChat => match code {
                KeyCode::Enter | KeyCode::Char('y') => self.clear_chat(),
                KeyCode::Esc | KeyCode::Char('n') | KeyCode::Left | KeyCode::Char('h') => {}
                _ => self.modal = Some(modal),
            },
            Modal::DestPicker(mut picker) => match code {
                KeyCode::Esc => {}
                KeyCode::Left | KeyCode::Char('h') => {
                    if let Some(parent) = picker.current.parent().map(std::path::Path::to_path_buf)
                    {
                        picker.set_current(parent);
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    picker.selected = picker.selected.saturating_sub(1);
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if picker.selected + 1 < picker.entries.len() {
                        picker.selected += 1;
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                    picker.enter_selected();
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Char('~') => {
                    if let Some(home) = widgets::home_dir() {
                        picker.set_current(home);
                    }
                    self.modal = Some(Modal::DestPicker(picker));
                }
                KeyCode::Char('d') => {
                    self.confirm_dest = None;
                    self.toast("using the shared HF cache");
                }
                KeyCode::Char('s') => match widgets::validate_writable_parent(&picker.current) {
                    Ok(()) => {
                        self.confirm_dest = Some(picker.current.clone());
                    }
                    Err(err) => {
                        picker.error = Some(err);
                        self.modal = Some(Modal::DestPicker(picker));
                    }
                },
                _ => self.modal = Some(Modal::DestPicker(picker)),
            },
            Modal::DownloadByLink {
                mut input,
                mut error,
            } => match code {
                KeyCode::Esc => {}
                KeyCode::Enter => match self.queue_download_by_link(&input) {
                    // Queued: the modal stays closed (taken above).
                    Ok(()) => {}
                    Err(message) => {
                        self.modal = Some(Modal::DownloadByLink {
                            input,
                            error: Some(message),
                        });
                    }
                },
                _ => {
                    match code {
                        KeyCode::Char(c) => {
                            input.push(c);
                            error = None;
                        }
                        KeyCode::Backspace => {
                            input.pop();
                            error = None;
                        }
                        _ => {}
                    }
                    self.modal = Some(Modal::DownloadByLink { input, error });
                }
            },
        }
    }

    pub(super) fn on_mouse(&mut self, mouse: MouseEvent) {
        if self.modal.is_some() {
            // The dialog owns the mouse: the first chip acts as confirm and
            // the last as cancel; a click outside dismisses like Esc.
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                let hits = self.modal_hits.get();
                let inside = |rect: Rect| {
                    mouse.column >= rect.x
                        && mouse.column < rect.x + rect.width
                        && mouse.row >= rect.y
                        && mouse.row < rect.y + rect.height
                };
                if hits.confirm.is_some_and(inside) {
                    match self.modal.take() {
                        Some(Modal::DeleteLocal(model)) => self.delete_local_model(&model),
                        Some(Modal::DeleteModel {
                            family_idx,
                            variant_idx,
                            ..
                        }) => {
                            self.delete_installed_variant(family_idx, variant_idx);
                        }
                        modal => {
                            self.modal = modal;
                            self.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
                        }
                    }
                } else if hits.cancel.is_some_and(inside) || !inside(hits.popup) {
                    self.on_key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE));
                }
            }
            return;
        }
        if self.show_help {
            // Help owns the screen: swallow every mouse event so scrolls and
            // clicks cannot leak through to the screen below. A left-click
            // dismisses help, mirroring "any key closes help".
            if matches!(mouse.kind, MouseEventKind::Down(MouseButton::Left)) {
                self.show_help = false;
            }
            return;
        }
        match mouse.kind {
            MouseEventKind::ScrollDown => self.scroll(KeyCode::Down, mouse.column, mouse.row),
            MouseEventKind::ScrollUp => self.scroll(KeyCode::Up, mouse.column, mouse.row),
            MouseEventKind::Down(MouseButton::Left) => self.on_click(mouse.column, mouse.row),
            _ => {}
        }
    }

    /// Wheel scroll: over the job-log pane on Downloads / Serve it scrolls the
    /// log; anywhere else it drives the active screen's existing up/down handler.
    fn scroll(&mut self, code: KeyCode, col: u16, row: u16) {
        if self.focus_tabs {
            let _ = self.on_key_tabs(code);
            return;
        }
        if matches!(self.screen, Screen::Downloads | Screen::Serve) {
            let rect = self.log_rect.get();
            let over_log = col >= rect.x
                && col < rect.x + rect.width
                && row >= rect.y
                && row < rect.y + rect.height;
            if over_log {
                let up = code == KeyCode::Up;
                match self.screen {
                    Screen::Downloads => self.scroll_downloads_log(up, false),
                    Screen::Serve => self.scroll_serve_log(up, false),
                    _ => {}
                }
                return;
            }
        }
        match self.screen {
            Screen::Home => self.on_key_home(code),
            Screen::Models => self.on_key_models(code),
            Screen::Downloads => self.on_key_downloads(code),
            Screen::Serve => self.on_key_serve(code),
            Screen::Chat => self.scroll_chat(code),
        }
    }

    /// Bracketed-paste payload: route to the focused text entry (chat composer,
    /// Serve host/port fields, Models filter, download-by-link modal); ignored
    /// elsewhere and while another modal owns input.
    pub(super) fn on_paste(&mut self, text: &str) {
        // Single-line field: drop newlines/control chars from pastes.
        if let Some(Modal::DownloadByLink { input, error }) = &mut self.modal {
            let clean: String = text.chars().filter(|c| !c.is_control()).collect();
            input.push_str(&clean);
            *error = None;
            return;
        }
        if self.modal.is_some() || self.show_help {
            return;
        }
        match self.screen {
            Screen::Chat
                if self.server_ready
                    && !self.server.as_ref().is_some_and(|job| job.done.is_some()) =>
            {
                self.paste_chat_text(text);
            }
            Screen::Chat if self.server.as_ref().is_some_and(|job| job.done.is_some()) => {
                self.toast_warn("server stopped — start one on Serve (4)");
            }
            Screen::Models if self.family_filter_active() => {
                self.filter.push_str(text);
                self.clamp_family_idx_to_filter();
            }
            Screen::Serve => {
                // Single-line fields: drop newlines/control chars from pastes.
                let clean: String = text.chars().filter(|c| !c.is_control()).collect();
                match self.serve_focus {
                    ServeFocus::Host => {
                        let at = widgets::char_boundary(&self.host, self.host_cursor);
                        self.host.insert_str(at, &clean);
                        self.host_cursor += clean.chars().count();
                    }
                    ServeFocus::Port => {
                        let at = widgets::char_boundary(&self.port, self.port_cursor);
                        self.port.insert_str(at, &clean);
                        self.port_cursor += clean.chars().count();
                    }
                    ServeFocus::List => {}
                }
            }
            _ => {}
        }
    }

    pub fn on_click(&mut self, col: u16, row: u16) {
        let hits = self.toolbar_hits.take();
        let action = hits
            .iter()
            .find_map(|(rect, action)| rect.contains((col, row).into()).then_some(*action));
        self.toolbar_hits.set(hits);
        if let Some(action) = action {
            self.on_toolbar_action(action);
            return;
        }
        // Tab bar click switches screens.
        let tab_hits = self.tab_hits.take();
        let clicked_tab = tab_hits.iter().find_map(|(rect, idx)| {
            (col >= rect.x
                && col < rect.x + rect.width
                && row >= rect.y
                && row < rect.y + rect.height)
                .then_some(*idx)
        });
        self.tab_hits.set(tab_hits);
        if let Some(idx) = clicked_tab {
            self.goto_screen(SCREENS[idx]);
            return;
        }
        // Clicking content leaves tab-bar focus.
        self.focus_tabs = false;
        // Journey banner click → activate next step.
        let banner = self.banner_rect.get();
        if banner.height > 0
            && col >= banner.x
            && col < banner.x + banner.width
            && row >= banner.y
            && row < banner.y + banner.height
        {
            self.activate_journey_banner();
            return;
        }
        // Home hero click → Quick start.
        let hero = self.hero_rect.get();
        if hero.height > 0
            && col >= hero.x
            && col < hero.x + hero.width
            && row >= hero.y
            && row < hero.y + hero.height
        {
            self.home_idx = 0;
            self.quick_start_from_home();
            return;
        }
        // Step header click (breadcrumb navigation) on the Models screen.
        if self.screen == Screen::Models {
            let hdr = self.step_header_rect.get();
            if hdr.height > 0 && row == hdr.y && col >= hdr.x && col < hdr.x + hdr.width {
                let offset = (col - hdr.x) as usize;
                self.on_step_header_click(offset);
                return;
            }
        }
        // Content-list click selects the row (and drills in for the wizard).
        // `row_in_rect` returns a position within the visible window only;
        // add the list's last-drawn scroll offset to get the real item index.
        if let Some(idx) = widgets::row_in_rect(self.content_list_rect.get(), col, row)
            .map(|raw| raw + self.content_list_offset.get())
        {
            match self.screen {
                Screen::Home => {
                    if idx < self.home_actions().len() {
                        self.home_idx = idx;
                        self.on_key_home(KeyCode::Enter);
                    }
                }
                Screen::Models => self.on_click_models(idx),
                Screen::Downloads => {
                    if self.downloads_show_library {
                        self.local_model_idx = idx.min(self.local_models.len().saturating_sub(1));
                        return;
                    }
                    if idx < self.downloads.len() {
                        self.select_download(idx);
                    }
                }
                Screen::Serve => {
                    if idx < installed_variants(&self.families).len() {
                        self.serve_focus = ServeFocus::List;
                        self.serve_idx = idx;
                    }
                }
                Screen::Chat => {}
            }
        }
    }
}
