use ax_engine_sdk::{LlamaCppConfig, SelectedBackend};

use crate::app_state::AppState;

pub(crate) fn supports_server_chat(state: &AppState) -> bool {
    state.runtime_report.selected_backend == SelectedBackend::LlamaCpp
        && matches!(
            state.session_config.llama_backend.as_ref(),
            Some(LlamaCppConfig::ServerCompletion(_))
        )
}
