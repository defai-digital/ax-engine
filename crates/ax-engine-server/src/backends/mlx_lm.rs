use ax_engine_sdk::{EngineSessionError, MlxLmConfig, SelectedBackend};

use crate::app_state::AppState;

pub(crate) fn is_selected(state: &AppState) -> bool {
    state.runtime_report.selected_backend == SelectedBackend::MlxLmDelegated
}

pub(crate) fn config(state: &AppState) -> Result<MlxLmConfig, EngineSessionError> {
    state
        .session_config
        .mlx_lm_backend
        .clone()
        .ok_or(EngineSessionError::MissingMlxLmConfig)
}
