use crate::backend::{RuntimeReport, SelectedBackend};
use crate::generate::{GenerateRequest, GenerateResponse};
use crate::llama_cpp::{
    LlamaCppConfig, LlamaCppStreamHandle, run_blocking_generate, start_streaming_generate,
};
use crate::mlx_lm::run_blocking_generate as run_mlx_lm_generate;

use super::EngineSession;
use super::config::EngineSessionConfig;
use super::errors::EngineSessionError;

pub(super) fn run_delegated_generate_with_config(
    config: &EngineSessionConfig,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    let runtime = config.runtime_report();
    run_delegated_generate_prevalidated(config, &runtime, request_id, request)
}

pub(super) fn run_delegated_generate_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    match config.resolved_backend.selected_backend {
        SelectedBackend::LlamaCpp => {
            run_llama_cpp_generate_prevalidated(config, runtime, request_id, request)
        }
        SelectedBackend::MlxLmDelegated => {
            let mlx_lm_backend = config
                .mlx_lm_backend
                .as_ref()
                .ok_or(EngineSessionError::MissingMlxLmConfig)?;
            run_mlx_lm_generate(request_id, runtime, mlx_lm_backend, request)
                .map_err(EngineSessionError::from)
        }
        SelectedBackend::Mlx => {
            let mut session = EngineSession::new(config.clone())?;
            session.generate_with_request_id(request_id, request.clone())
        }
    }
}

fn run_llama_cpp_generate_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    request_id: u64,
    request: &GenerateRequest,
) -> Result<GenerateResponse, EngineSessionError> {
    let llama_backend = resolved_llama_cpp_backend(config)?;

    run_blocking_generate(request_id, runtime, llama_backend, request)
        .map_err(EngineSessionError::from)
}

pub(super) fn start_llama_cpp_stream_prevalidated(
    config: &EngineSessionConfig,
    runtime: &RuntimeReport,
    _request_id: u64,
    request: &GenerateRequest,
) -> Result<(RuntimeReport, LlamaCppStreamHandle, SelectedBackend), EngineSessionError> {
    let llama_backend = resolved_llama_cpp_backend(config)?;

    start_streaming_generate(runtime, llama_backend, request)
        .map(|stream| {
            (
                runtime.clone(),
                stream,
                config.resolved_backend.selected_backend,
            )
        })
        .map_err(EngineSessionError::from)
}

fn resolved_llama_cpp_backend(
    config: &EngineSessionConfig,
) -> Result<&LlamaCppConfig, EngineSessionError> {
    config
        .llama_backend
        .as_ref()
        .ok_or(EngineSessionError::MissingLlamaCppConfig {
            selected_backend: config.resolved_backend.selected_backend,
        })
}
