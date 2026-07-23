use ax_engine_core::{EngineCoreError, MetalRuntimeError, RequestMultimodalInputError};
use thiserror::Error;

use crate::backend::{BackendContractError, SelectedBackend};
use crate::edge_llm::EdgeLlmBackendError;
use crate::llama_cpp::LlamaCppBackendError;
use crate::mlx_lm::MlxLmBackendError;

#[derive(Debug, Error)]
pub enum EngineSessionError {
    #[error(transparent)]
    BackendContract(#[from] BackendContractError),
    #[error(transparent)]
    LlamaCpp(#[from] LlamaCppBackendError),
    #[error(transparent)]
    MlxLm(#[from] MlxLmBackendError),
    #[error(transparent)]
    EdgeLlm(#[from] EdgeLlmBackendError),
    #[error("max_batch_tokens must be greater than zero")]
    InvalidMaxBatchTokens,
    #[error("generate request requires input_tokens or input_text")]
    EmptyInputTokens,
    #[error("generate request max_output_tokens must be greater than zero")]
    InvalidMaxOutputTokens,
    #[error(
        "no_repeat_ngram_size={no_repeat_ngram_size} requires ngram_window to be positive and at least as large"
    )]
    InvalidNoRepeatNgram {
        no_repeat_ngram_size: u32,
        ngram_window: u32,
    },
    #[error(
        "MLX preview session only accepts pre-tokenized input_tokens; input_text requires a llama.cpp backend"
    )]
    MlxBackendRequiresTokenizedInput,
    #[error(
        "multimodal_inputs require native MLX backend; delegated backend {selected_backend:?} is text-only"
    )]
    MultimodalInputsRequireNativeMlx { selected_backend: SelectedBackend },
    #[error(
        "multimodal prompt is {prompt_len} tokens but max_batch_tokens is {max_batch_tokens}; multimodal prefill must complete in one scheduler step — raise max_batch_tokens or reduce media size"
    )]
    MultimodalPromptExceedsMaxBatchTokens {
        prompt_len: u32,
        max_batch_tokens: u32,
    },
    #[error(transparent)]
    InvalidMultimodalInputs(#[from] RequestMultimodalInputError),
    #[error(
        "MLX mode requires validated Metal runtime artifacts; deterministic fallback is internal-only and must be explicitly enabled"
    )]
    MlxRuntimeArtifactsRequired,
    #[error(
        "MLX runtime is not available; enable MLX mode support or configure a llama.cpp backend"
    )]
    MlxRuntimeUnavailable,
    #[error(
        "MLX MTP was required, but the loaded model package has no validated MTP head or assistant drafter"
    )]
    MlxMtpRequiredButUnavailable,
    #[error("request_id must be greater than zero")]
    InvalidRequestId,
    #[error("unsupported support tier cannot start an engine session")]
    UnsupportedSupportTier,
    #[error(
        "AX Engine v6 requires Apple M2 Max-or-newer CPU/GPU (32 GB RAM minimum); detected unsupported host {detected_host}. Set AX_ALLOW_UNSUPPORTED_HOST=1 only for internal development or CI bring-up."
    )]
    UnsupportedHostHardware { detected_host: String },
    #[error("llama.cpp backend {selected_backend:?} requires llama_backend config")]
    MissingLlamaCppConfig { selected_backend: SelectedBackend },
    #[error("mlx-lm delegated backend requires mlx_lm_backend config")]
    MissingMlxLmConfig,
    #[error("tensor_rt_edge_llm backend requires edge_llm_backend config")]
    MissingEdgeLlmConfig,
    #[error("delegated backend {selected_backend:?} is missing runtime report")]
    MissingDelegatedRuntime { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp backend {selected_backend:?} does not support {operation} in the preview SDK session yet"
    )]
    LlamaCppDoesNotSupportLifecycle {
        selected_backend: SelectedBackend,
        operation: &'static str,
    },
    #[error("mlx-lm delegated backend does not support streaming in this preview contract")]
    MlxLmDoesNotSupportStreaming,
    #[error("mlx-lm delegated backend does not support {operation} in the preview SDK lifecycle")]
    MlxLmDoesNotSupportLifecycle { operation: &'static str },
    #[error(
        "tensor_rt_edge_llm backend does not support {operation} via EngineSession lifecycle yet; use OpenAI chat routes"
    )]
    EdgeLlmDoesNotSupportLifecycle { operation: &'static str },
    #[error(
        "stateless streaming is not supported for the native MLX backend ({selected_backend:?}); \
         use EngineSession streaming methods instead"
    )]
    NativeBackendStatelessStreamNotSupported { selected_backend: SelectedBackend },
    #[error(
        "llama.cpp stream for request {request_id} on backend {selected_backend:?} ended before a terminal stop marker"
    )]
    LlamaCppStreamEndedBeforeStop {
        request_id: u64,
        selected_backend: SelectedBackend,
    },
    #[error(
        "mlx-lm delegated stream for request {request_id} ended before a terminal finish_reason \
         or [DONE] marker; the connection may have closed prematurely"
    )]
    MlxLmStreamEndedBeforeStop { request_id: u64 },
    #[error("request {request_id} is missing from session state")]
    MissingRequestSnapshot { request_id: u64 },
    #[error(
        "stream for request {request_id} ended after {observed_event_count} events without a final response"
    )]
    StreamEndedWithoutResponse {
        request_id: u64,
        observed_event_count: u64,
    },
    #[error("request {request_id} did not terminate within {max_steps} steps")]
    RequestDidNotTerminate { request_id: u64, max_steps: u64 },
    #[error("request {request_id} violated session report invariants: {message}")]
    RequestReportInvariantViolation {
        request_id: u64,
        message: &'static str,
    },
    #[error(transparent)]
    Core(#[from] EngineCoreError),
    #[error(transparent)]
    MetalRuntime(#[from] MetalRuntimeError),
    #[error("embedding is not supported by the active backend")]
    EmbeddingNotSupported,
    #[error("embedding failed: {message}")]
    EmbeddingFailed { message: &'static str },
}
