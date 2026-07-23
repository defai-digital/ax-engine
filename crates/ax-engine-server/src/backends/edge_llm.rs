use ax_engine_sdk::{
    EdgeLlmChatGenerateRequest, EdgeLlmConfig, EdgeLlmStreamHandle, EngineSessionError,
    GenerateResponse, RuntimeReport, SelectedBackend, run_blocking_edge_llm_chat_generate,
    start_streaming_edge_llm_chat_generate,
};

use crate::app_state::LiveState;

pub(crate) fn is_selected(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::TensorRtEdgeLlm
}

pub(crate) fn config(live: &LiveState) -> Result<EdgeLlmConfig, EngineSessionError> {
    live.session_config
        .edge_llm_backend
        .clone()
        .ok_or(EngineSessionError::MissingEdgeLlmConfig)
}

pub(crate) fn run_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &EdgeLlmChatGenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    run_blocking_edge_llm_chat_generate(request_id, runtime, config, request)
        .map_err(EngineSessionError::from)
}

pub(crate) fn start_streaming_chat_generate(
    runtime: &RuntimeReport,
    config: &EdgeLlmConfig,
    request: &EdgeLlmChatGenerateRequest,
) -> Result<EdgeLlmStreamHandle, EngineSessionError> {
    start_streaming_edge_llm_chat_generate(runtime, config, request).map_err(EngineSessionError::from)
}
