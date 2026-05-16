use thiserror::Error;

#[allow(dead_code)]
#[derive(Debug, Error)]
pub(crate) enum CliError {
    #[error("{0}")]
    Usage(String),
    #[error("{0}")]
    Runtime(String),
    #[error("{0}")]
    Contract(String),
    #[error("{0}")]
    Correctness(String),
    #[error("{0}")]
    Performance(String),
}

impl From<ax_engine_core::EngineCoreError> for CliError {
    fn from(value: ax_engine_core::EngineCoreError) -> Self {
        Self::Runtime(value.to_string())
    }
}
