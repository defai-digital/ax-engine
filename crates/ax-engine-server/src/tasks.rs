use ax_engine_sdk::EngineSessionError;
use axum::Json;
use axum::http::StatusCode;

use crate::errors::{ErrorResponse, map_blocking_task_error, map_session_error};

pub(crate) async fn run_blocking_session_task<T, F>(
    operation: F,
) -> Result<T, (StatusCode, Json<ErrorResponse>)>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, EngineSessionError> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(map_blocking_task_error)?
        .map_err(map_session_error)
}

/// Like [`run_blocking_session_task`] but for operations that already produce
/// an HTTP error response. Used to keep CPU- or subprocess-heavy request
/// preprocessing (e.g. inline media decoding, which can wait on an ffmpeg
/// child process) off the async executor threads.
pub(crate) async fn run_blocking_http_task<T, F>(
    operation: F,
) -> Result<T, (StatusCode, Json<ErrorResponse>)>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, (StatusCode, Json<ErrorResponse>)> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(map_blocking_task_error)?
}
