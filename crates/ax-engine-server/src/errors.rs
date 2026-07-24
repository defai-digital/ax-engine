use ax_engine_sdk::{
    EdgeLlmBackendError, EngineSessionError, LlamaCppBackendError, MlxLmBackendError,
    VllmBackendError,
};
use axum::Json;
use axum::http::StatusCode;
use serde::Serialize;

use crate::admission::AdmissionError;
use crate::generation::service::GenerationServiceError;

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub(crate) error: ErrorBody,
}

impl ErrorResponse {
    pub(crate) fn server_error(message: String) -> Self {
        Self {
            error: ErrorBody {
                error_type: "server_error",
                code: Some("engine_error".to_string()),
                param: None,
                message,
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorBody {
    #[serde(rename = "type")]
    pub(crate) error_type: &'static str,
    pub(crate) code: Option<String>,
    pub(crate) param: Option<String>,
    pub(crate) message: String,
}

pub(crate) fn error_response(
    status: StatusCode,
    code: &'static str,
    message: String,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorBody {
                error_type: openai_error_type(status),
                code: Some(code.to_string()),
                param: None,
                message,
            },
        }),
    )
}

pub(crate) fn admission_error_response(error: AdmissionError) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        AdmissionError::Draining => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_loading",
            "the current model is draining for replacement; retry after loading completes"
                .to_string(),
        ),
        AdmissionError::Saturated => error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "concurrency_limit",
            "server is at its maximum concurrent engine-job limit; retry shortly".to_string(),
        ),
        AdmissionError::StaleGeneration => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_changed",
            "the model changed while this request was being prepared; retry against the current model"
                .to_string(),
        ),
    }
}

pub(crate) fn map_session_error(error: EngineSessionError) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        EngineSessionError::EmptyInputTokens
        | EngineSessionError::InvalidMaxOutputTokens
        | EngineSessionError::InvalidNoRepeatNgram { .. }
        | EngineSessionError::MlxBackendRequiresTokenizedInput
        | EngineSessionError::MultimodalInputsRequireNativeMlx { .. }
        | EngineSessionError::MultimodalPromptExceedsMaxBatchTokens { .. }
        | EngineSessionError::InvalidMultimodalInputs(_)
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::MlxMtpRequiredButUnavailable
        | EngineSessionError::LlamaCppDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportLifecycle { .. }
        | EngineSessionError::EdgeLlmDoesNotSupportLifecycle { .. }
        | EngineSessionError::TensorRtLlmDoesNotSupportLifecycle { .. }
        | EngineSessionError::VllmDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportStreaming
        | EngineSessionError::NativeBackendStatelessStreamNotSupported { .. }
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported { .. })
        | EngineSessionError::RequestDidNotTerminate { .. }
        | EngineSessionError::MissingRequestSnapshot { .. } => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingInputText { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::UnsupportedTokenPrompt { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::AmbiguousPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingInputText)
        | EngineSessionError::MlxLm(MlxLmBackendError::UnsupportedTokenPrompt)
        | EngineSessionError::MlxLm(MlxLmBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::MissingInputText)
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::UnsupportedTokenPrompt)
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::Vllm(VllmBackendError::InvalidRequest(_))
        | EngineSessionError::Vllm(VllmBackendError::MissingInputText)
        | EngineSessionError::Vllm(VllmBackendError::UnsupportedTokenPrompt) => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingStreamChoice { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::MissingStreamChoice { .. })
        | EngineSessionError::Vllm(VllmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::Vllm(VllmBackendError::InvalidAssistantContent { .. })
        | EngineSessionError::Vllm(VllmBackendError::InvalidResponseJson { .. })
        | EngineSessionError::Vllm(VllmBackendError::Sse(_)) => {
            error_response(StatusCode::BAD_GATEWAY, "backend_error", error.to_string())
        }
        EngineSessionError::Vllm(VllmBackendError::HttpStatus { status, .. }) => {
            let status = match status {
                401 | 403 => StatusCode::SERVICE_UNAVAILABLE,
                429 => StatusCode::TOO_MANY_REQUESTS,
                400..=499 => StatusCode::BAD_REQUEST,
                _ => StatusCode::BAD_GATEWAY,
            };
            error_response(status, "vllm_upstream_error", error.to_string())
        }
        EngineSessionError::Vllm(VllmBackendError::HttpRequest { ref source, .. }) => {
            let message = source.to_string().to_ascii_lowercase();
            let status = if message.contains("timed out") || message.contains("timeout") {
                StatusCode::GATEWAY_TIMEOUT
            } else {
                StatusCode::BAD_GATEWAY
            };
            error_response(status, "vllm_transport_error", error.to_string())
        }
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingLlamaCppConfig { .. }
        | EngineSessionError::MissingMlxLmConfig
        | EngineSessionError::MissingEdgeLlmConfig
        | EngineSessionError::MissingTensorRtLlmConfig
        | EngineSessionError::MissingVllmConfig
        | EngineSessionError::MissingDelegatedRuntime { .. }
        | EngineSessionError::LlamaCppStreamEndedBeforeStop { .. }
        | EngineSessionError::MlxLmStreamEndedBeforeStop { .. }
        | EngineSessionError::MlxRuntimeArtifactsRequired
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandLaunch { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandFailed { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::CommandTimedOut { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::NonUtf8Output { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::SerializeRequestJson { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpRequest { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpStatus { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::HttpResponseRead { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::InvalidResponseJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::SerializeRequestJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::HttpRequest { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::HttpStatus { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::InvalidResponseJson { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::SseRead { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::InvalidStreamChunk { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::SerializeRequestJson { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::HttpRequest { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::HttpStatus { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::InvalidResponseJson { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::SseRead { .. })
        | EngineSessionError::EdgeLlm(EdgeLlmBackendError::InvalidStreamChunk { .. })
        | EngineSessionError::Vllm(VllmBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::Vllm(VllmBackendError::HttpConfig { .. })
        | EngineSessionError::Vllm(VllmBackendError::ModelNotAdvertised { .. })
        | EngineSessionError::UnsupportedHostHardware { .. } => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "unsupported_host",
            error.to_string(),
        ),
        EngineSessionError::EmbeddingNotSupported => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            error.to_string(),
        ),
        EngineSessionError::RequestReportInvariantViolation { .. }
        | EngineSessionError::StreamEndedWithoutResponse { .. }
        | EngineSessionError::EmbeddingFailed { .. }
        | EngineSessionError::Core(_)
        | EngineSessionError::MetalRuntime(_)
        | EngineSessionError::MlxRuntimeUnavailable
        | EngineSessionError::Vllm(VllmBackendError::SerializeRequestJson { .. }) => {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "engine_error",
                error.to_string(),
            )
        }
    }
}

pub(crate) fn map_generation_service_error(
    error: GenerationServiceError,
) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        GenerationServiceError::Engine(error) => map_session_error(error),
        GenerationServiceError::Saturated => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "service_saturated",
            "native generation command queue is saturated; retry shortly".to_string(),
        ),
        GenerationServiceError::Unavailable => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "service_unavailable",
            "native generation worker is unavailable".to_string(),
        ),
    }
}

pub(crate) fn map_blocking_task_error(
    error: tokio::task::JoinError,
) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "engine_error",
        format!("blocking server task failed: {error}"),
    )
}

pub(crate) fn request_not_found_response(request_id: u64) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::NOT_FOUND,
        "request_not_found",
        format!("request {request_id} is missing from preview session state"),
    )
}

fn openai_error_type(status: StatusCode) -> &'static str {
    if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_sdk::{Gemma4UnifiedRuntimeInputError, RequestMultimodalInputError};

    #[test]
    fn invalid_multimodal_inputs_map_to_invalid_request() {
        let (status, body) = map_session_error(EngineSessionError::InvalidMultimodalInputs(
            RequestMultimodalInputError::Gemma4Unified(
                Gemma4UnifiedRuntimeInputError::InvalidField {
                    field: "images[0].span".to_string(),
                    message: "replacement span exceeds prompt".to_string(),
                },
            ),
        ));

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body = body.0;
        assert_eq!(body.error.code.as_deref(), Some("invalid_request"));
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(
            body.error
                .message
                .contains("invalid Gemma4 unified runtime input")
        );
    }

    #[test]
    fn invalid_no_repeat_ngram_maps_to_invalid_request() {
        let (status, body) = map_session_error(EngineSessionError::InvalidNoRepeatNgram {
            no_repeat_ngram_size: 4,
            ngram_window: 3,
        });

        assert_eq!(status, StatusCode::BAD_REQUEST);
        let body = body.0;
        assert_eq!(body.error.code.as_deref(), Some("invalid_request"));
        assert_eq!(body.error.error_type, "invalid_request_error");
        assert!(body.error.message.contains("no_repeat_ngram_size=4"));
    }
}
