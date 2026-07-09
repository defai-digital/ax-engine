use ax_engine_sdk::{EngineSessionError, LlamaCppBackendError, MlxLmBackendError};
use axum::Json;
use axum::http::StatusCode;
use serde::Serialize;

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

pub(crate) fn map_session_error(error: EngineSessionError) -> (StatusCode, Json<ErrorResponse>) {
    match error {
        EngineSessionError::EmptyInputTokens
        | EngineSessionError::InvalidMaxOutputTokens
        | EngineSessionError::MlxBackendRequiresTokenizedInput
        | EngineSessionError::MultimodalInputsRequireNativeMlx { .. }
        | EngineSessionError::MultimodalPromptExceedsMaxBatchTokens { .. }
        | EngineSessionError::InvalidMultimodalInputs(_)
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::LlamaCppDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportLifecycle { .. }
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
        | EngineSessionError::MlxLm(MlxLmBackendError::BackendConfigMismatch { .. }) => {
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request",
                error.to_string(),
            )
        }
        EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingStreamChoice { .. }) => {
            error_response(StatusCode::BAD_GATEWAY, "backend_error", error.to_string())
        }
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingLlamaCppConfig { .. }
        | EngineSessionError::MissingMlxLmConfig
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
        | EngineSessionError::MlxRuntimeUnavailable => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "engine_error",
            error.to_string(),
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
    use ax_engine_sdk::Gemma4UnifiedRuntimeInputError;

    #[test]
    fn invalid_multimodal_inputs_map_to_invalid_request() {
        let (status, body) = map_session_error(EngineSessionError::InvalidMultimodalInputs(
            Gemma4UnifiedRuntimeInputError::InvalidField {
                field: "images[0].span".to_string(),
                message: "replacement span exceeds prompt".to_string(),
            },
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
}
