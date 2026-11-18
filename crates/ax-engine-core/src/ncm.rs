//! Typed, KV-free NextConcept execution contracts. No transport or MLX state.

use crate::{RequestId, WorkUnitKind};
use std::sync::{
    Arc,
    atomic::{AtomicU8, Ordering},
};
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NcmError {
    #[error("NextConcept requires a typed unconditional submission")]
    NcmSubmissionRequired,
    #[error("NextConcept is not supported by this runner")]
    Unsupported,
    #[error("NextConcept runtime is busy (one active request, no queue)")]
    Busy,
    #[error("no active NextConcept request")]
    NoRequest,
    #[error("NextConcept request was cancelled")]
    Cancelled,
    #[error("NextConcept request deadline exceeded")]
    Deadline,
    #[error("invalid NextConcept contract: {0}")]
    Contract(String),
    #[error("NextConcept runner failed: {0}")]
    Runtime(String),
    #[error("NextConcept memory budget is insufficient")]
    MemoryBudget,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcmSampling {
    Greedy,
    SeededV1 { seed: u64 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NcmRequestConfig {
    pub sampling: NcmSampling,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConceptPhase {
    Init,
    Predict(u8),
    Decode,
    Commit,
    Finished,
}

impl ConceptPhase {
    pub fn next(self) -> Result<Self, NcmError> {
        match self {
            Self::Init => Ok(Self::Predict(1)),
            Self::Predict(n @ 1..=10) => Ok(Self::Predict(n + 1)),
            Self::Predict(11) => Ok(Self::Decode),
            Self::Decode => Ok(Self::Commit),
            Self::Commit => Ok(Self::Finished),
            _ => Err(NcmError::Contract("invalid or terminal phase".into())),
        }
    }
    pub fn work_unit(self) -> Result<WorkUnitKind, NcmError> {
        match self {
            Self::Init => Ok(WorkUnitKind::ConceptInit),
            Self::Predict(1..=11) => Ok(WorkUnitKind::ConceptLevel),
            Self::Decode => Ok(WorkUnitKind::ConceptDecode),
            Self::Commit => Ok(WorkUnitKind::ConceptCommit),
            _ => Err(NcmError::Contract("no work for this phase".into())),
        }
    }
    pub fn completed_levels(self) -> Result<u8, NcmError> {
        match self {
            Self::Init => Ok(0),
            Self::Predict(n @ 1..=11) => Ok(n),
            Self::Predict(_) => Err(NcmError::Contract("invalid level".into())),
            _ => Ok(12),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConceptProgress {
    pub epoch: u8,
    pub phase: ConceptPhase,
}
impl Default for ConceptProgress {
    fn default() -> Self {
        Self {
            epoch: 0,
            phase: ConceptPhase::Init,
        }
    }
}
impl ConceptProgress {
    pub fn advance(self) -> Result<Self, NcmError> {
        if self.epoch >= 14 {
            return Err(NcmError::Contract("work budget exhausted".into()));
        }
        Ok(Self {
            epoch: self.epoch + 1,
            phase: self.phase.next()?,
        })
    }
}

/// Cancel and visible commit share one atomic linearization point.
#[derive(Clone, Debug, Default)]
pub struct NcmCancellation(Arc<AtomicU8>);
impl NcmCancellation {
    pub fn cancel(&self) -> bool {
        self.0
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire) == 1
    }
    pub(crate) fn commit(&self) -> Result<(), NcmError> {
        self.0
            .compare_exchange(0, 2, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| NcmError::Cancelled)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NcmFinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NcmOutput {
    pub raw_bpe_ids: Vec<u32>,
    pub visible_bpe_ids: Vec<u32>,
    pub text: String,
    pub finish_reason: NcmFinishReason,
}
impl NcmOutput {
    pub fn validate(&self) -> Result<(), NcmError> {
        if self.raw_bpe_ids.len() != 128 || self.raw_bpe_ids.iter().any(|&v| v >= 8192) {
            return Err(NcmError::Contract("expected 128 raw BPE IDs".into()));
        }
        let expected = self
            .raw_bpe_ids
            .iter()
            .copied()
            .take_while(|&v| v != 0)
            .filter(|&v| v != 1)
            .collect::<Vec<_>>();
        let reason = if self.raw_bpe_ids.contains(&0) {
            NcmFinishReason::Stop
        } else {
            NcmFinishReason::Length
        };
        if self.visible_bpe_ids != expected
            || self.finish_reason != reason
            || (self.visible_bpe_ids.is_empty() != self.text.is_empty())
        {
            return Err(NcmError::Contract(
                "inconsistent visible BPE/termination".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ConceptWork {
    pub request_id: RequestId,
    pub config: NcmRequestConfig,
    pub before: ConceptProgress,
    pub cancellation: NcmCancellation,
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConceptUpdate {
    pub request_id: RequestId,
    pub progress: ConceptProgress,
    pub output: Option<NcmOutput>,
}

#[derive(Debug)]
pub(crate) struct ConceptRequest {
    pub id: RequestId,
    pub config: NcmRequestConfig,
    pub progress: ConceptProgress,
    pub cancellation: NcmCancellation,
    started: Instant,
    timeout: Duration,
}
impl ConceptRequest {
    pub fn new(
        id: RequestId,
        config: NcmRequestConfig,
        timeout: Duration,
    ) -> Result<Self, NcmError> {
        if id.0 == 0 || timeout.is_zero() || timeout > Duration::from_secs(3600) {
            return Err(NcmError::Contract(
                "nonzero request ID and deadline <= 1h required".into(),
            ));
        }
        Ok(Self {
            id,
            config,
            progress: ConceptProgress::default(),
            cancellation: NcmCancellation::default(),
            started: Instant::now(),
            timeout,
        })
    }
    pub fn check(&self) -> Result<(), NcmError> {
        if self.cancellation.is_cancelled() {
            return Err(NcmError::Cancelled);
        }
        if self.started.elapsed() >= self.timeout {
            return Err(NcmError::Deadline);
        }
        Ok(())
    }
    pub fn validate(&self, update: &ConceptUpdate) -> Result<(), NcmError> {
        if update.request_id != self.id || update.progress != self.progress.advance()? {
            return Err(NcmError::Contract(
                "missing, skipped or incorrect phase progress".into(),
            ));
        }
        if (update.progress.phase == ConceptPhase::Finished) != update.output.is_some() {
            return Err(NcmError::Contract(
                "output is permitted only at final commit".into(),
            ));
        }
        if let Some(output) = &update.output {
            output.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    #[test]
    fn finite_sequence_has_fourteen_units_and_no_skips() {
        let mut p = ConceptProgress::default();
        for epoch in 1..=14 {
            p.phase.work_unit().unwrap();
            p = p.advance().unwrap();
            assert_eq!(p.epoch, epoch);
        }
        assert_eq!(p.phase, ConceptPhase::Finished);
        assert!(p.advance().is_err());
        assert!(ConceptPhase::Predict(0).next().is_err());
        assert!(ConceptPhase::Predict(12).next().is_err());
    }
    #[test]
    fn no_op_wrong_id_and_early_output_are_rejected() {
        let r = ConceptRequest::new(
            RequestId(1),
            NcmRequestConfig {
                sampling: NcmSampling::Greedy,
            },
            Duration::from_secs(1),
        )
        .unwrap();
        let mut u = ConceptUpdate {
            request_id: r.id,
            progress: r.progress,
            output: None,
        };
        assert!(r.validate(&u).is_err());
        u.progress = r.progress.advance().unwrap();
        assert!(r.validate(&u).is_ok());
        u.request_id = RequestId(2);
        assert!(r.validate(&u).is_err());
    }
    #[test]
    fn cancellation_and_commit_are_ordered() {
        let c = NcmCancellation::default();
        assert!(c.cancel());
        assert!(c.commit().is_err());
        let c = NcmCancellation::default();
        c.commit().unwrap();
        assert!(!c.cancel());
        assert!(!c.is_cancelled());
    }
    #[test]
    fn successful_empty_output_is_valid() {
        let o = NcmOutput {
            raw_bpe_ids: vec![0; 128],
            visible_bpe_ids: vec![],
            text: String::new(),
            finish_reason: NcmFinishReason::Stop,
        };
        assert!(o.validate().is_ok());
    }
    #[test]
    fn output_cannot_forge_text_for_empty_visible_ids() {
        let output = NcmOutput {
            raw_bpe_ids: vec![0; 128],
            visible_bpe_ids: vec![],
            text: "unexpected".into(),
            finish_reason: NcmFinishReason::Stop,
        };
        assert!(output.validate().is_err());
    }
    #[test]
    fn deadline_and_invalid_level_are_rejected() {
        let mut r = ConceptRequest::new(
            RequestId(1),
            NcmRequestConfig {
                sampling: NcmSampling::Greedy,
            },
            Duration::from_secs(1),
        )
        .unwrap();
        r.started = Instant::now() - Duration::from_secs(2);
        assert_eq!(r.check(), Err(NcmError::Deadline));
        assert!(ConceptPhase::Predict(255).completed_levels().is_err());
    }
}
