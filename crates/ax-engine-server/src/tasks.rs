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
