//! Test-only host argmax over the rows used by the scheduler-facing cursor.

use super::*;

#[derive(serde::Serialize)]
pub(crate) struct Decision {
    pub position: usize,
    pub primary: u32,
    pub draft: Option<u32>,
    pub remaining: usize,
    pub target: u32,
    pub next: u32,
    pub accepted: usize,
    pub emitted: Vec<u32>,
}

thread_local! {
    static ACTIVE: RefCell<Option<Vec<Decision>>> = const { RefCell::new(None) };
}

pub(crate) fn capture<T>(call: impl FnOnce() -> T) -> (T, Vec<Decision>) {
    assert!(ACTIVE.replace(Some(Vec::new())).is_none());
    let result = call();
    (result, ACTIVE.take().unwrap_or_default())
}

fn host_argmax(logits: &MlxArray) -> u32 {
    let row = mlx_sys::contiguous(&astype(logits, MlxDtype::Float32, None), None);
    assert!(try_eval(&[&row]).is_ok());
    let shape = row.shape();
    assert!(
        matches!(shape.split_last(), Some((&width, leading)) if width > 0 && leading.iter().all(|&n| n == 1))
    );
    let values = row.data_f32();
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mut best = 0;
    for (index, &logit) in values.iter().enumerate().skip(1) {
        if logit > values[best] {
            best = index;
        }
    }
    best as u32
}

pub(super) fn record(
    position: usize,
    primary: u32,
    remaining: usize,
    terminal_ids: &[u32],
    observation: &CandidateStepObservation,
    accepted: usize,
    emitted: &[u32],
) {
    ACTIVE.with_borrow_mut(|slot| {
        let Some(decisions) = slot.as_mut() else {
            return;
        };
        assert_eq!(observation.target_schedule, "canonical_singleton");
        let Some(correction) = observation.canonical_correction_logits.as_ref() else {
            return;
        };
        let target = host_argmax(correction);
        let next = host_argmax(&observation.next_logits);
        let draft = observation.draft_token;
        let eligible =
            remaining > 1 && !terminal_ids.contains(&primary) && !terminal_ids.contains(&target);
        assert_eq!(accepted, usize::from(eligible && draft == Some(target)));
        if accepted > 0 {
            assert_eq!(emitted, &[target, next]);
        } else {
            assert_eq!(next, target);
            assert_eq!(emitted, &[target]);
        }
        decisions.push(Decision {
            position,
            primary,
            draft,
            remaining,
            target,
            next,
            accepted,
            emitted: emitted.to_vec(),
        });
    });
}

#[test]
fn flash_next_mtp_host_argmax_uses_first_exact_tie() {
    let logits = reshape(
        &MlxArray::from_f32_slice(&[-1.0, 3.0, 3.0, 1.0]),
        &[1, 4],
        None,
    );
    assert_eq!(host_argmax(&logits), 1);
}

#[test]
fn flash_next_mtp_host_audit_checks_terminal_and_budget() {
    let logits = MlxArray::from_f32_slice(&[-1.0, 3.0]);
    let observation = CandidateStepObservation {
        target_schedule: "canonical_singleton",
        draft_token: Some(1),
        verification_logits: None,
        canonical_correction_logits: Some(logits.clone()),
        canonical_primary: None,
        next_logits: logits,
    };
    let (_, decisions) = capture(|| {
        record(4, 0, 2, &[1], &observation, 0, &[1]);
        record(4, 0, 1, &[], &observation, 0, &[1]);
        record(4, 0, 2, &[], &observation, 1, &[1, 1]);
    });
    assert_eq!(decisions.len(), 3);
    assert_eq!(decisions.iter().map(|row| row.accepted).sum::<usize>(), 1);
}
