use ax_engine_sdk::{
    EngineSessionError, GenerateResponse, MlxLmChatGenerateRequest, MlxLmConfig, MlxLmStreamHandle,
    RuntimeReport, SelectedBackend, run_blocking_chat_generate, start_streaming_chat_generate,
};

use crate::app_state::LiveState;

pub(crate) fn is_selected(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::MlxLmDelegated
}

pub(crate) fn config(live: &LiveState) -> Result<MlxLmConfig, EngineSessionError> {
    live.session_config
        .mlx_lm_backend
        .clone()
        .ok_or(EngineSessionError::MissingMlxLmConfig)
}

pub(crate) fn run_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &MlxLmConfig,
    request: &MlxLmChatGenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    run_blocking_chat_generate(request_id, runtime, config, request)
        .map_err(EngineSessionError::from)
}

pub(crate) fn start_chat_stream(
    runtime: &RuntimeReport,
    config: &MlxLmConfig,
    request: &MlxLmChatGenerateRequest,
) -> Result<MlxLmStreamHandle, EngineSessionError> {
    start_streaming_chat_generate(runtime, config, request).map_err(EngineSessionError::from)
}
