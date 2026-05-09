use crate::app::{AppState, AppTab};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    NextTab,
    PreviousTab,
    SelectTab(AppTab),
    Quit,
}

pub fn reduce(state: &mut AppState, action: Action) {
    match action {
        Action::NextTab => state.next_tab(),
        Action::PreviousTab => state.previous_tab(),
        Action::SelectTab(tab) => state.select_tab(tab),
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

        reduce(&mut state, Action::Quit);
        assert!(state.should_quit);
    }
}
