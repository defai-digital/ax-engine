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
        | EngineSessionError::MultimodalInputsRequireNativeMlx { .. }
        | EngineSessionError::MultimodalPromptExceedsMaxBatchTokens { .. }
        | EngineSessionError::InvalidMultimodalInputs(_)
        | EngineSessionError::InvalidMaxBatchTokens
        | EngineSessionError::InvalidRequestId
        | EngineSessionError::UnsupportedSupportTier
        | EngineSessionError::MlxMtpRequiredButUnavailable
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
        | EngineSessionError::LlamaCpp(LlamaCppBackendError::MissingCompletionChoice { .. })
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
        | EngineSessionError::MlxLmStreamEndedBeforeStop { .. }
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

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use std::sync::Once;

    fn init_python() {
        static PYTHON_INIT: Once = Once::new();
        PYTHON_INIT.call_once(pyo3::Python::initialize);
    }

    #[test]
    fn python_engine_errors_use_custom_exception_hierarchy() {
        init_python();

        Python::attach(|py| {
            let backend_error = to_py_runtime_error(EngineSessionError::LlamaCpp(
                LlamaCppBackendError::HttpStatus {
                    endpoint: "http://127.0.0.1:8081/completion".to_string(),
                    status: 502,
                    body: "bad gateway".to_string(),
                },
            ));
            assert!(backend_error.is_instance_of::<EngineBackendError>(py));
            assert!(backend_error.is_instance_of::<EngineError>(py));
            assert!(backend_error.is_instance_of::<PyRuntimeError>(py));

            let inference_error = to_py_runtime_error(EngineSessionError::EmbeddingNotSupported);
            assert!(inference_error.is_instance_of::<EngineInferenceError>(py));
            assert!(inference_error.is_instance_of::<EngineError>(py));
            assert!(inference_error.is_instance_of::<PyRuntimeError>(py));

            let state_error = py_engine_state_error("session is closed");
            assert!(state_error.is_instance_of::<EngineStateError>(py));
            assert!(state_error.is_instance_of::<EngineError>(py));
            assert!(state_error.is_instance_of::<PyRuntimeError>(py));

            let validation_error = to_py_runtime_error(EngineSessionError::EmptyInputTokens);
            assert!(validation_error.is_instance_of::<PyValueError>(py));
            assert!(!validation_error.is_instance_of::<EngineError>(py));
        });
    }
}
