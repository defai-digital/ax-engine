use crate::app::{AppState, AppTab, ServerControlSelection};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    NextTab,
    PreviousTab,
    SelectTab(AppTab),
    SelectServerControl(ServerControlSelection),
    Quit,
}

pub fn reduce(state: &mut AppState, action: Action) {
    match action {
        Action::NextTab => state.next_tab(),
        Action::PreviousTab => state.previous_tab(),
        Action::SelectTab(tab) => state.select_tab(tab),
        Action::SelectServerControl(selection) => state.select_server_control(selection),
        Action::Quit => state.should_quit = true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actions_move_between_tabs_without_terminal() {
        let mut state = AppState::empty();

        reduce(&mut state, Action::NextTab);
        assert_eq!(state.selected_tab, AppTab::Models);

        reduce(&mut state, Action::PreviousTab);
        assert_eq!(state.selected_tab, AppTab::Readiness);

        reduce(&mut state, Action::SelectTab(AppTab::Benchmarks));
        assert_eq!(state.selected_tab, AppTab::Benchmarks);

        reduce(
            &mut state,
            Action::SelectServerControl(ServerControlSelection::Button(
                crate::app::ServerControlButton::Start,
            )),
        );
        assert_eq!(
            state.server_control.selected,
            Some(ServerControlSelection::Button(
                crate::app::ServerControlButton::Start
            ))
        );
        assert!(state.status_message.contains("server start"));

        reduce(&mut state, Action::Quit);
        assert!(state.should_quit);
    }
}
