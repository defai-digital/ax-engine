use ax_engine_sdk::{EngineSessionError, LlamaCppConfig, SelectedBackend};

use crate::app_state::AppState;

pub(crate) fn supports_server_chat(state: &AppState) -> bool {
    state.runtime_report.selected_backend == SelectedBackend::LlamaCpp
        && matches!(
            state.session_config.llama_backend.as_ref(),
            Some(LlamaCppConfig::ServerCompletion(_))
        )
}

pub(crate) fn config(state: &AppState) -> Result<LlamaCppConfig, EngineSessionError> {
    state
        .session_config
        .llama_backend
        .clone()
        .ok_or(EngineSessionError::MissingLlamaCppConfig {
            selected_backend: state.runtime_report.selected_backend,
        })
}
