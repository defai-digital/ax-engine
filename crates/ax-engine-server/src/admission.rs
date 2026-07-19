use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

pub(crate) struct AdmissionController {
    active_jobs: AtomicUsize,
    draining: parking_lot::Mutex<bool>,
    idle_notify: Notify,
    semaphore: Option<Arc<Semaphore>>,
}

impl AdmissionController {
    pub(crate) fn new(limit: Option<usize>) -> Self {
        Self {
            active_jobs: AtomicUsize::new(0),
            draining: parking_lot::Mutex::new(false),
            idle_notify: Notify::new(),
            semaphore: limit.map(|limit| Arc::new(Semaphore::new(limit))),
        }
    }

    pub(crate) fn try_admit(self: &Arc<Self>) -> Result<AdmissionPermit, AdmissionError> {
        let draining = self.draining.lock();
        if *draining {
            return Err(AdmissionError::Draining);
        }
        let semaphore_permit = self
            .semaphore
            .as_ref()
            .map(|semaphore| Arc::clone(semaphore).try_acquire_owned())
            .transpose()
            .map_err(|_| AdmissionError::Saturated)?;
        self.active_jobs.fetch_add(1, Ordering::AcqRel);
        drop(draining);
        Ok(AdmissionPermit {
            controller: Arc::clone(self),
            semaphore_permit,
        })
    }

    pub(crate) fn active_jobs(&self) -> usize {
        self.active_jobs.load(Ordering::Acquire)
    }

    pub(crate) fn begin_drain(self: &Arc<Self>) -> AdmissionDrainGuard {
        *self.draining.lock() = true;
        AdmissionDrainGuard {
            controller: Arc::clone(self),
        }
    }

    pub(crate) async fn wait_for_idle(&self) {
        loop {
            let notified = self.idle_notify.notified();
            if self.active_jobs() == 0 {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionError {
    Draining,
    Saturated,
    StaleGeneration,
}

pub(crate) struct AdmissionPermit {
    controller: Arc<AdmissionController>,
    semaphore_permit: Option<OwnedSemaphorePermit>,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        // Return bounded capacity before publishing the job as inactive. This
        // keeps active_jobs() == 0 from racing with a still-held semaphore
        // permit and transiently reporting Saturated to the next admission.
        self.semaphore_permit.take();
        let previous = self.controller.active_jobs.fetch_sub(1, Ordering::AcqRel);
        if previous == 1 {
            self.controller.idle_notify.notify_waiters();
        }
    }
}

pub(crate) struct AdmissionDrainGuard {
    controller: Arc<AdmissionController>,
}

impl Drop for AdmissionDrainGuard {
    fn drop(&mut self) {
        *self.controller.draining.lock() = false;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;

    #[test]
    fn limited_controller_holds_capacity_until_job_permit_drops() {
        let controller = Arc::new(AdmissionController::new(Some(1)));
        let permit = controller
            .try_admit()
            .expect("first job should be admitted");
        assert_eq!(controller.active_jobs(), 1);
        assert_eq!(
            controller.try_admit().err(),
            Some(AdmissionError::Saturated)
        );

        drop(permit);
        assert_eq!(controller.active_jobs(), 0);
        assert!(controller.try_admit().is_ok());
    }

    #[test]
    fn unlimited_controller_still_tracks_engine_job_lifetime() {
        let controller = Arc::new(AdmissionController::new(None));
        let first = controller.try_admit().expect("job should be admitted");
        let second = controller.try_admit().expect("job should be admitted");
        assert_eq!(controller.active_jobs(), 2);

        drop((first, second));
        assert_eq!(controller.active_jobs(), 0);
    }

    #[tokio::test]
    async fn timed_out_frontend_does_not_release_running_blocking_job() {
        let controller = Arc::new(AdmissionController::new(Some(1)));
        let permit = controller.try_admit().expect("job should be admitted");
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let handle = tokio::task::spawn_blocking(move || {
            let _permit = permit;
            entered_tx.send(()).expect("test should receive entry");
            release_rx.recv().expect("test should release job");
        });
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("blocking job should start");

        assert!(
            tokio::time::timeout(Duration::from_millis(10), handle)
                .await
                .is_err(),
            "frontend timeout should detach the still-running blocking job"
        );
        assert_eq!(
            controller.try_admit().err(),
            Some(AdmissionError::Saturated)
        );
        assert_eq!(controller.active_jobs(), 1);

        release_tx.send(()).expect("blocking job should release");
        tokio::time::timeout(Duration::from_secs(1), async {
            while controller.active_jobs() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking job should release its permit");
        assert!(controller.try_admit().is_ok());
    }

    #[tokio::test]
    async fn drain_rejects_new_jobs_and_waits_for_existing_job() {
        let controller = Arc::new(AdmissionController::new(None));
        let permit = controller.try_admit().expect("job should be admitted");
        let drain = controller.begin_drain();
        assert_eq!(controller.try_admit().err(), Some(AdmissionError::Draining));

        drop(permit);
        controller.wait_for_idle().await;
        drop(drain);
        assert!(controller.try_admit().is_ok());
    }
}
