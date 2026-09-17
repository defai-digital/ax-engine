use ax_engine_sdk::EngineSessionError;
use axum::Json;
use axum::http::StatusCode;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::errors::{ErrorResponse, error_response, map_blocking_task_error, map_session_error};

const MEDIA_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Clone, Debug)]
pub(crate) struct BlockingTaskControl {
    cancelled: Arc<AtomicBool>,
    deadline: Instant,
}

impl BlockingTaskControl {
    pub(crate) fn new(timeout: Duration) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: Instant::now() + timeout,
        }
    }

    pub(crate) fn check(&self) -> Result<(), &'static str> {
        if self.cancelled.load(Ordering::Acquire) {
            Err("media preprocessing was cancelled")
        } else if Instant::now() >= self.deadline {
            Err("media preprocessing exceeded its time limit")
        } else {
            Ok(())
        }
    }

    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl Default for BlockingTaskControl {
    fn default() -> Self {
        Self::new(MEDIA_TIMEOUT)
    }
}

struct CancelOnDrop(BlockingTaskControl);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// Shared by all model generations and API surfaces in one server.
#[derive(Clone)]
pub(crate) struct MediaPreprocessor {
    slots: Arc<tokio::sync::Semaphore>,
    timeout: Duration,
}

impl Default for MediaPreprocessor {
    fn default() -> Self {
        Self {
            slots: Arc::new(tokio::sync::Semaphore::new(2)),
            timeout: MEDIA_TIMEOUT,
        }
    }
}

fn media_timeout(message: &str) -> (StatusCode, Json<ErrorResponse>) {
    error_response(
        StatusCode::REQUEST_TIMEOUT,
        "media_timeout",
        message.to_string(),
    )
}

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

impl MediaPreprocessor {
    pub(crate) async fn run<T, F>(
        &self,
        operation: F,
    ) -> Result<T, (StatusCode, Json<ErrorResponse>)>
    where
        T: Send + 'static,
        F: FnOnce(BlockingTaskControl) -> Result<T, (StatusCode, Json<ErrorResponse>)>
            + Send
            + 'static,
    {
        let permit = Arc::clone(&self.slots).try_acquire_owned().map_err(|_| {
            error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "media_capacity_exceeded",
                "inline media preprocessing is at capacity; retry after an active request finishes"
                    .to_string(),
            )
        })?;
        let control = BlockingTaskControl::new(self.timeout);
        let _cancel = CancelOnDrop(control.clone());
        let task = tokio::task::spawn_blocking(move || {
            // The actual worker owns capacity even if its HTTP future disappears.
            let _permit = permit;
            control.check().map_err(media_timeout)?;
            let result = operation(control.clone());
            control.check().map_err(media_timeout)?;
            result
        });
        tokio::time::timeout(self.timeout, task)
            .await
            .map_err(|_| media_timeout("media preprocessing exceeded its time limit"))?
            .map_err(map_blocking_task_error)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn wait_for_capacity(pool: &MediaPreprocessor) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while pool.slots.available_permits() != 1 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn cancelled_http_future_keeps_capacity_until_worker_exits() {
        let pool = MediaPreprocessor {
            slots: Arc::new(tokio::sync::Semaphore::new(1)),
            timeout: Duration::from_secs(5),
        };
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_pool = pool.clone();
        let task = tokio::spawn(async move {
            worker_pool
                .run(move |control| {
                    started_tx.send(control).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                    Ok(())
                })
                .await
        });
        let control = started_rx.await.unwrap();
        let saturated = pool.run(|_| Ok(())).await.unwrap_err();
        assert_eq!(saturated.0, StatusCode::TOO_MANY_REQUESTS);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(control.check().is_err());
        assert_eq!(pool.slots.available_permits(), 0);
        release_tx.send(()).unwrap();
        wait_for_capacity(&pool).await;
        pool.run(|_| Ok(())).await.unwrap();
    }

    #[tokio::test]
    async fn timeout_signals_cancellation_without_releasing_a_running_worker() {
        let pool = MediaPreprocessor {
            slots: Arc::new(tokio::sync::Semaphore::new(1)),
            timeout: Duration::from_millis(50),
        };
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_pool = pool.clone();
        let task = tokio::spawn(async move {
            worker_pool
                .run(move |control| {
                    started_tx.send(control).unwrap();
                    release_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                    Ok(())
                })
                .await
        });
        let control = started_rx.await.unwrap();
        assert_eq!(
            task.await.unwrap().unwrap_err().0,
            StatusCode::REQUEST_TIMEOUT
        );
        assert!(control.check().is_err());
        assert_eq!(pool.slots.available_permits(), 0);
        release_tx.send(()).unwrap();
        wait_for_capacity(&pool).await;
    }

    #[tokio::test]
    async fn worker_panic_returns_capacity() {
        let pool = MediaPreprocessor {
            slots: Arc::new(tokio::sync::Semaphore::new(1)),
            timeout: Duration::from_secs(2),
        };
        assert!(
            pool.run::<(), _>(|_| panic!("decoder failure"))
                .await
                .is_err()
        );
        assert_eq!(pool.slots.available_permits(), 1);
    }
}
