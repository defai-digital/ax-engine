//! Experimental typed unconditional NCM API. Separate from generic chat/generate.
use ax_engine_core::{
    EngineCore, RequestId,
    ncm::{
        ConceptProgress, NcmCancellation, NcmError, NcmFinishReason, NcmRequestConfig, NcmSampling,
    },
};
use serde::{Deserialize, Serialize};
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum NcmSamplingMode {
    Greedy,
    SeededV1 { seed: u64 },
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NcmOutputMode {
    #[default]
    FullSequence128,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NcmGenerateRequest {
    pub sampling: NcmSamplingMode,
    #[serde(default)]
    pub output_mode: NcmOutputMode,
}
#[derive(Clone, Debug, Serialize)]
pub struct NcmGenerateResponse {
    pub request_id: u64,
    pub artifact_identity: String,
    pub artifact_revision: &'static str,
    pub raw_bpe_ids: Vec<u32>,
    pub visible_bpe_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: &'static str,
    pub prompt_tokens: u32,
    pub output_tokens: usize,
    pub work_units: u8,
    pub time_to_final_commit_ms: u128,
}
struct Owner {
    core: Mutex<EngineCore>,
    cancellation: Mutex<Option<NcmCancellation>>,
    closed: AtomicBool,
    next_id: AtomicU64,
}
impl Owner {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        if let Some(c) = self
            .cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
        {
            c.cancel();
        }
        if let Ok(mut core) = self.core.try_lock() {
            core.cancel_ncm();
        }
    }
}

/// Loaded once; one active request, no waiting queue. Dropping closes the session.
pub struct NcmSession {
    owner: Arc<Owner>,
    identity: String,
    timeout: Duration,
}
impl NcmSession {
    pub fn load_experimental(
        path: &Path,
        memory_budget_bytes: u64,
        timeout: Duration,
    ) -> Result<Self, NcmError> {
        if timeout.is_zero() || timeout > Duration::from_secs(3600) {
            return Err(NcmError::Contract(
                "request deadline must be in (0,1h]".into(),
            ));
        }
        crate::host::validate_local_host()
            .map_err(|host| NcmError::Runtime(format!("unsupported MLX host: {host}")))?;
        let runner = ax_engine_mlx::ncm::NcmRunner::load_experimental(path, memory_budget_bytes)?;
        let identity = runner.identity().to_string();
        Ok(Self {
            owner: Arc::new(Owner {
                core: Mutex::new(runner.into_core()),
                cancellation: Mutex::new(None),
                closed: AtomicBool::new(false),
                next_id: AtomicU64::new(1),
            }),
            identity,
            timeout,
        })
    }
    pub fn generate_unconditional(
        &self,
        request: NcmGenerateRequest,
    ) -> Result<NcmGeneration, NcmError> {
        let sampling = match (request.sampling, request.output_mode) {
            (NcmSamplingMode::Greedy, NcmOutputMode::FullSequence128) => NcmSampling::Greedy,
            (NcmSamplingMode::SeededV1 { seed }, NcmOutputMode::FullSequence128) => {
                NcmSampling::SeededV1 { seed }
            }
        };
        if self.owner.closed.load(Ordering::Acquire) {
            return Err(NcmError::Contract("session closed".into()));
        }
        let mut core = self.owner.core.try_lock().map_err(|error| match error {
            std::sync::TryLockError::WouldBlock => NcmError::Busy,
            std::sync::TryLockError::Poisoned(_) => {
                NcmError::Runtime("session lock poisoned".into())
            }
        })?;
        let id = self
            .owner
            .next_id
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| NcmError::Contract("request ID space exhausted".into()))?;
        let cancellation =
            core.submit_ncm(RequestId(id), NcmRequestConfig { sampling }, self.timeout)?;
        *self
            .owner
            .cancellation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cancellation.clone());
        if self.owner.closed.load(Ordering::Acquire) {
            core.cancel_ncm();
            return Err(NcmError::Cancelled);
        }
        Ok(NcmGeneration {
            owner: Arc::clone(&self.owner),
            identity: self.identity.clone(),
            cancellation,
            done: false,
            started: Instant::now(),
        })
    }
    pub fn close(&self) {
        self.owner.close();
    }
}
impl Drop for NcmSession {
    fn drop(&mut self) {
        self.owner.close();
    }
}

/// Polling emits progress only; text becomes visible in the final response.
pub struct NcmGeneration {
    owner: Arc<Owner>,
    identity: String,
    cancellation: NcmCancellation,
    done: bool,
    started: Instant,
}
impl NcmGeneration {
    pub fn cancellation(&self) -> NcmCancellation {
        self.cancellation.clone()
    }
    pub fn step(&mut self) -> Result<(ConceptProgress, Option<NcmGenerateResponse>), NcmError> {
        if self.done {
            return Err(NcmError::NoRequest);
        }
        if self.cancellation.is_cancelled() {
            self.owner
                .core
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cancel_ncm();
            self.done = true;
            return Err(NcmError::Cancelled);
        }
        let mut core = self
            .owner
            .core
            .lock()
            .map_err(|_| NcmError::Runtime("session lock poisoned".into()))?;
        match core.step_ncm() {
            Ok(update) => {
                let response = update.output.map(|out| {
                    self.done = true;
                    NcmGenerateResponse {
                        request_id: update.request_id.0,
                        artifact_identity: self.identity.clone(),
                        artifact_revision: ax_engine_mlx::ncm::artifact::REVISION,
                        output_tokens: out.visible_bpe_ids.len(),
                        raw_bpe_ids: out.raw_bpe_ids,
                        visible_bpe_ids: out.visible_bpe_ids,
                        text: out.text,
                        finish_reason: match out.finish_reason {
                            NcmFinishReason::Stop => "stop",
                            NcmFinishReason::Length => "length",
                        },
                        prompt_tokens: 0,
                        work_units: update.progress.epoch,
                        time_to_final_commit_ms: self.started.elapsed().as_millis(),
                    }
                });
                Ok((update.progress, response))
            }
            Err(e) => {
                self.done = true;
                Err(e)
            }
        }
    }
    pub fn finish(mut self) -> Result<NcmGenerateResponse, NcmError> {
        loop {
            if let (_, Some(response)) = self.step()? {
                return Ok(response);
            }
        }
    }
}
impl Drop for NcmGeneration {
    fn drop(&mut self) {
        if !self.done {
            self.cancellation.cancel();
            self.owner
                .core
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .cancel_ncm();
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn request_rejects_unknown_and_explicit_default_controls() {
        assert!(serde_json::from_str::<NcmGenerateRequest>(r#"{"sampling":"greedy"}"#).is_ok());
        for key in [
            "seed",
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "prompt",
            "tools",
            "stop",
            "stream",
            "logprobs",
        ] {
            let mut value = serde_json::json!({"sampling":"greedy"});
            value[key] = serde_json::json!(0);
            assert!(
                serde_json::from_value::<NcmGenerateRequest>(value).is_err(),
                "{key}"
            );
        }
        assert!(serde_json::from_str::<NcmGenerateRequest>("{}").is_err());
        assert!(serde_json::from_str::<NcmGenerateRequest>(r#"{"sampling":"sample"}"#).is_err());
    }

    #[test]
    fn seeded_mode_requires_a_nested_unsigned_seed_and_rejects_extra_controls() {
        let request = serde_json::from_str::<NcmGenerateRequest>(
            r#"{"sampling":{"seeded_v1":{"seed":18446744073709551615}}}"#,
        )
        .unwrap();
        assert_eq!(
            request.sampling,
            NcmSamplingMode::SeededV1 { seed: u64::MAX }
        );
        assert_eq!(
            serde_json::to_value(request).unwrap()["sampling"],
            serde_json::json!({"seeded_v1": {"seed": u64::MAX}})
        );
        for invalid in [
            r#"{"sampling":"seeded_v1"}"#,
            r#"{"sampling":{"seeded_v1":{}}}"#,
            r#"{"sampling":{"seeded_v1":{"seed":-1}}}"#,
            r#"{"sampling":{"seeded_v1":{"seed":1.5}}}"#,
            r#"{"sampling":{"seeded_v1":{"seed":1,"seed":2}}}"#,
            r#"{"sampling":{"seeded_v1":{"seed":1,"temperature":1}}}"#,
            r#"{"sampling":{"seeded_v1":{"seed":1}},"top_p":1}"#,
        ] {
            assert!(
                serde_json::from_str::<NcmGenerateRequest>(invalid).is_err(),
                "{invalid}"
            );
        }
    }
    #[derive(Debug)]
    struct FakeRunner {
        barrier: Option<Arc<std::sync::Barrier>>,
    }
    impl ax_engine_core::ExecutionRunner for FakeRunner {
        fn run(&self, input: ax_engine_core::RunnerInput) -> ax_engine_core::RunnerOutput {
            ax_engine_core::ExecutionRunner::run(&ax_engine_core::DeterministicRunner, input)
        }
        fn begin_ncm(&self, _id: RequestId, _config: NcmRequestConfig) -> Result<(), NcmError> {
            Ok(())
        }
        fn run_ncm(
            &self,
            work: ax_engine_core::ncm::ConceptWork,
        ) -> Result<ax_engine_core::ncm::ConceptUpdate, NcmError> {
            if let Some(b) = &self.barrier {
                b.wait();
                b.wait();
            }
            let progress = work.before.advance()?;
            Ok(ax_engine_core::ncm::ConceptUpdate {
                request_id: work.request_id,
                progress,
                output: (progress.phase == ax_engine_core::ncm::ConceptPhase::Finished).then(
                    || ax_engine_core::ncm::NcmOutput {
                        raw_bpe_ids: vec![1; 128],
                        visible_bpe_ids: vec![],
                        text: String::new(),
                        finish_reason: NcmFinishReason::Length,
                    },
                ),
            })
        }
    }
    fn test_session(barrier: Option<Arc<std::sync::Barrier>>) -> NcmSession {
        let mut core = EngineCore::with_runtime_components(
            ax_engine_core::KvManagerConfig::validated(ax_engine_core::CacheGroupId(0), 16, 1),
            FakeRunner { barrier },
            ax_engine_core::DeterministicSampler,
        );
        core.set_generation_kind(ax_engine_core::GenerationKind::NextConcept);
        NcmSession {
            owner: Arc::new(Owner {
                core: Mutex::new(core),
                cancellation: Mutex::new(None),
                closed: AtomicBool::new(false),
                next_id: AtomicU64::new(1),
            }),
            identity: "synthetic-contract-fixture".into(),
            timeout: Duration::from_secs(60),
        }
    }
    fn request() -> NcmGenerateRequest {
        NcmGenerateRequest {
            sampling: NcmSamplingMode::Greedy,
            output_mode: NcmOutputMode::FullSequence128,
        }
    }
    #[test]
    fn request_ids_advance_and_abandoned_handles_release() {
        let session = test_session(None);
        let first = session
            .generate_unconditional(request())
            .unwrap()
            .finish()
            .unwrap();
        drop(session.generate_unconditional(request()).unwrap());
        let next = session
            .generate_unconditional(request())
            .unwrap()
            .finish()
            .unwrap();
        assert_eq!(first.request_id, 1);
        assert_eq!(next.request_id, 3);
    }
    #[test]
    #[allow(clippy::panic, clippy::unwrap_used)]
    fn poisoned_cancellation_slot_does_not_orphan_a_request() {
        let session = test_session(None);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = session.owner.cancellation.lock().unwrap();
            panic!("intentional poison");
        }));
        assert!(
            session
                .generate_unconditional(request())
                .unwrap()
                .finish()
                .is_ok()
        );
    }
    #[test]
    #[allow(clippy::unwrap_used)]
    fn session_close_during_work_cancels_before_commit() {
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let session = test_session(Some(barrier.clone()));
        let mut handle = session.generate_unconditional(request()).unwrap();
        let worker = std::thread::spawn(move || handle.step());
        barrier.wait();
        session.close();
        barrier.wait();
        assert!(matches!(worker.join().unwrap(), Err(NcmError::Cancelled)));
    }
}
