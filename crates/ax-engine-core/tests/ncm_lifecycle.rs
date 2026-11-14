#![allow(clippy::unwrap_used)]
use ax_engine_core::{
    CacheGroupId, DeterministicRunner, DeterministicSampler, EngineCore, ExecutionRunner,
    GenerationKind, KvManagerConfig, RequestId, RunnerInput, RunnerOutput,
    ncm::{
        ConceptPhase, ConceptUpdate, ConceptWork, NcmError, NcmFinishReason, NcmOutput,
        NcmRequestConfig, NcmSampling,
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;
#[derive(Debug)]
struct Fake {
    leased: Arc<AtomicBool>,
    released: Arc<AtomicUsize>,
    invalid: bool,
}
impl ExecutionRunner for Fake {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        DeterministicRunner.run(input)
    }
    fn begin_ncm(&self, _id: RequestId, _c: NcmRequestConfig) -> Result<(), NcmError> {
        self.leased
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| NcmError::Busy)
    }
    fn run_ncm(&self, work: ConceptWork) -> Result<ConceptUpdate, NcmError> {
        let after = if self.invalid {
            work.before
        } else {
            work.before.advance()?
        };
        Ok(ConceptUpdate {
            request_id: work.request_id,
            progress: after,
            output: (after.phase == ConceptPhase::Finished).then(|| NcmOutput {
                raw_bpe_ids: vec![1; 128],
                visible_bpe_ids: vec![],
                text: String::new(),
                finish_reason: NcmFinishReason::Length,
            }),
        })
    }
    fn release_request_state(&self, _id: RequestId) {
        self.leased.store(false, Ordering::Release);
        self.released.fetch_add(1, Ordering::Relaxed);
    }
}
fn setup(invalid: bool) -> (EngineCore, Arc<AtomicBool>, Arc<AtomicUsize>) {
    let leased = Arc::new(AtomicBool::new(false));
    let released = Arc::new(AtomicUsize::new(0));
    let mut core = EngineCore::with_runtime_components(
        KvManagerConfig::validated(CacheGroupId(0), 16, 1),
        Fake {
            leased: leased.clone(),
            released: released.clone(),
            invalid,
        },
        DeterministicSampler,
    );
    core.set_generation_kind(GenerationKind::NextConcept);
    (core, leased, released)
}
fn config() -> NcmRequestConfig {
    NcmRequestConfig {
        sampling: NcmSampling::Greedy,
    }
}
#[test]
fn fourteen_typed_steps_commit_empty_output_without_kv() {
    let (mut c, leased, released) = setup(false);
    for _ in 0..3 {
        c.submit_ncm(RequestId(1), config(), Duration::from_secs(1))
            .unwrap();
        assert!(leased.load(Ordering::Acquire));
        assert!(matches!(
            c.submit_ncm(RequestId(2), config(), Duration::from_secs(1)),
            Err(NcmError::Busy)
        ));
        for epoch in 1..=14 {
            let u = c.step_ncm().unwrap();
            assert_eq!(u.progress.epoch, epoch);
            assert_eq!(u.output.is_some(), epoch == 14);
        }
        assert!(!leased.load(Ordering::Acquire));
        assert_eq!(c.kv_manager().used_block_count(), 0);
        assert_eq!(c.kv_manager().available_block_count(), 1);
        assert_eq!(c.step_ncm(), Err(NcmError::NoRequest));
    }
    assert_eq!(released.load(Ordering::Relaxed), 3);
}
#[test]
fn cancel_at_every_boundary_releases_and_never_commits() {
    let (mut c, leased, released) = setup(false);
    for count in 0..14 {
        let token = c
            .submit_ncm(RequestId(1), config(), Duration::from_secs(1))
            .unwrap();
        for _ in 0..count {
            assert!(c.step_ncm().unwrap().output.is_none());
        }
        assert!(token.cancel());
        assert_eq!(c.step_ncm(), Err(NcmError::Cancelled));
        assert!(!leased.load(Ordering::Acquire));
    }
    assert_eq!(released.load(Ordering::Relaxed), 14);
}
#[test]
fn scheduled_no_op_fails_immediately_and_releases() {
    let (mut c, leased, released) = setup(true);
    c.submit_ncm(RequestId(1), config(), Duration::from_secs(1))
        .unwrap();
    assert!(matches!(c.step_ncm(), Err(NcmError::Contract(_))));
    assert!(!leased.load(Ordering::Acquire));
    assert_eq!(released.load(Ordering::Relaxed), 1);
}
#[test]
fn generic_steps_and_other_generation_kinds_cannot_enter_typed_runtime() {
    let (mut c, _, _) = setup(false);
    assert!(c.step(128, true).is_err());
    c.set_generation_kind(GenerationKind::Autoregressive);
    assert!(matches!(
        c.submit_ncm(RequestId(1), config(), Duration::from_secs(1)),
        Err(NcmError::Unsupported)
    ));
}
