use ax_engine_sdk::{
    EngineSessionError, GenerateResponse, RuntimeReport, SelectedBackend, VllmChatGenerateRequest,
    VllmConfig, VllmStreamHandle, run_blocking_vllm_chat_generate,
    start_streaming_vllm_chat_generate,
};

use crate::app_state::LiveState;

pub(crate) fn is_selected(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::Vllm
}

pub(crate) fn config(live: &LiveState) -> Result<VllmConfig, EngineSessionError> {
    live.session_config
        .vllm_backend
        .clone()
        .ok_or(EngineSessionError::MissingVllmConfig)
}

pub(crate) fn run_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &VllmChatGenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    run_blocking_vllm_chat_generate(request_id, runtime, config, request)
        .map_err(EngineSessionError::from)
}

pub(crate) fn start_streaming_chat_generate(
    runtime: &RuntimeReport,
    config: &VllmConfig,
    request: &VllmChatGenerateRequest,
) -> Result<VllmStreamHandle, EngineSessionError> {
    start_streaming_vllm_chat_generate(runtime, config, request).map_err(EngineSessionError::from)
}
