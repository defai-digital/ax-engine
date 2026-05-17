use ax_engine_sdk::SelectedBackend;

use crate::app_state::AppState;

pub(crate) fn is_selected(state: &AppState) -> bool {
    state.runtime_report.selected_backend == SelectedBackend::MlxLmDelegated
}
