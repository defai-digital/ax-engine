use ax_engine_sdk::{EngineSessionError, LlamaCppBackendError, MlxLmBackendError};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    ax_engine._ax_engine,
    EngineError,
    PyRuntimeError,
    "Base class for AX Engine runtime errors."
);
pyo3::create_exception!(
    ax_engine._ax_engine,
    EngineBackendError,
    EngineError,
    "Raised when a delegated backend process, transport, or response fails."
);
pyo3::create_exception!(
    ax_engine._ax_engine,
    EngineInferenceError,
    EngineError,
    "Raised when AX Engine cannot complete inference after input validation succeeds."
);
pyo3::create_exception!(
    ax_engine._ax_engine,
    EngineStateError,
    EngineError,
    "Raised when a Python session is closed, poisoned, or already streaming."
);

pub(crate) fn py_engine_state_error(message: &'static str) -> PyErr {
    EngineStateError::new_err(message)
}

pub(crate) fn to_py_runtime_error(error: EngineSessionError) -> PyErr {
    match error {
        EngineSessionError::EmptyInputTokens
        | EngineSessionError::InvalidMaxOutputTokens
        | EngineSessionError::MlxBackendRequiresTokenizedInput
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::LlamaCppDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportLifecycle { .. }
        | EngineSessionError::MlxLmDoesNotSupportStreaming
        | EngineSessionError::NativeBackendStatelessStreamNotSupported { .. }
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::StreamingNotSupported { .. })
        | EngineSessionError::RequestDidNotTerminate { .. }
        | EngineSessionError::MissingRequestSnapshot { .. } => {
            PyValueError::new_err(error.to_string())
        }
        EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingInputText { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::UnsupportedTokenPrompt { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::AmbiguousPromptInput { .. })
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::BackendConfigMismatch { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingInputText)
        | EngineSessionError::MlxLm(MlxLmBackendError::UnsupportedTokenPrompt)
        | EngineSessionError::MlxLm(MlxLmBackendError::BackendConfigMismatch { .. }) => {
            PyValueError::new_err(error.to_string())
        }
        EngineSessionError::BackendContract(_)
        | EngineSessionError::MissingLlamaCppConfig { .. }
        | EngineSessionError::MissingMlxLmConfig
        | EngineSessionError::MissingDelegatedRuntime { .. }
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
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingCompletionChoice { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::SseRead { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::InvalidStreamChunk { .. })
        | EngineSessionError::MlxLm(MlxLmBackendError::MissingStreamChoice { .. }) => {
            EngineBackendError::new_err(error.to_string())
        }
        EngineSessionError::LlamaCppStreamEndedBeforeStop { .. }
        | EngineSessionError::MlxRuntimeArtifactsRequired
        | EngineSessionError::MlxRuntimeUnavailable
        | EngineSessionError::UnsupportedHostHardware { .. }
        | EngineSessionError::RequestReportInvariantViolation { .. }
        | EngineSessionError::StreamEndedWithoutResponse { .. }
        | EngineSessionError::EmbeddingNotSupported
        | EngineSessionError::EmbeddingFailed { .. }
        | EngineSessionError::Core(_)
        | EngineSessionError::MetalRuntime(_) => EngineInferenceError::new_err(error.to_string()),
    }
}
