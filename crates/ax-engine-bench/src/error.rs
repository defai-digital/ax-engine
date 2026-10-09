use thiserror::Error;

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
    /// Reserved for the documented exit-code 4 (see docs/BENCH-DESIGN.md).
    #[allow(dead_code)]
    #[error("{0}")]
    Performance(String),
}

impl From<ax_engine_core::EngineCoreError> for CliError {
    fn from(value: ax_engine_core::EngineCoreError) -> Self {
        Self::Runtime(value.to_string())
    }
}
