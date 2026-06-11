use ax_engine_sdk::{
    EngineSessionError, GenerateResponse, LlamaCppChatGenerateRequest, LlamaCppConfig,
    LlamaCppStreamHandle, RuntimeReport, SelectedBackend, run_blocking_llama_cpp_chat_generate,
    start_streaming_llama_cpp_chat_generate,
};

use crate::app_state::LiveState;

pub(crate) fn supports_server_chat(live: &LiveState) -> bool {
    live.runtime_report.selected_backend == SelectedBackend::LlamaCpp
        && matches!(
            live.session_config.llama_backend.as_ref(),
            Some(LlamaCppConfig::ServerCompletion(_))
        )
}

pub(crate) fn config(live: &LiveState) -> Result<LlamaCppConfig, EngineSessionError> {
    live.session_config
        .llama_backend
        .clone()
        .ok_or(EngineSessionError::MissingLlamaCppConfig {
            selected_backend: live.runtime_report.selected_backend,
        })
}

pub(crate) fn run_chat_generate(
    request_id: u64,
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &LlamaCppChatGenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    run_blocking_llama_cpp_chat_generate(request_id, runtime, config, request)
        .map_err(EngineSessionError::from)
}

pub(crate) fn start_streaming_chat_generate(
    runtime: &RuntimeReport,
    config: &LlamaCppConfig,
    request: &LlamaCppChatGenerateRequest,
) -> Result<LlamaCppStreamHandle, EngineSessionError> {
    start_streaming_llama_cpp_chat_generate(runtime, config, request)
        .map_err(EngineSessionError::from)
}
