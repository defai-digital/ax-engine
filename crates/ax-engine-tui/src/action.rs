use crate::app::AppState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    NextTab,
    PreviousTab,
    Quit,
}

pub fn reduce(state: &mut AppState, action: Action) {
    match action {
        Action::NextTab => state.next_tab(),
        Action::PreviousTab => state.previous_tab(),
        Action::Quit => state.should_quit = true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::AppTab;

    #[test]
    fn actions_move_between_tabs_without_terminal() {
        let mut state = AppState::empty();

        reduce(&mut state, Action::NextTab);
        assert_eq!(state.selected_tab, AppTab::Models);

        reduce(&mut state, Action::PreviousTab);
        assert_eq!(state.selected_tab, AppTab::Readiness);

        reduce(&mut state, Action::Quit);
        assert!(state.should_quit);
    }
}
