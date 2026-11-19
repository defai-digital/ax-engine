//! Test-only logit capture for paired greedy disclosures.
//!
//! Hold lazy rows during replay and evaluate them after the arm finishes. The
//! independently measured performance run uses the ordinary release binary.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use mlx_sys::{MlxArray, MlxDtype, astype, contiguous, reshape, slice, try_eval};
use serde::Serialize;
use std::collections::BTreeMap;

/// The verifier observation is tied to committed outputs. Ordinary target
/// projections can include work later discarded by a pipeline or rollback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Source {
    Target,
    Verifier,
}

struct Observation {
    logits: MlxArray,
    row: usize,
    source: Source,
}

pub(crate) struct Capture {
    first_position: usize,
    rows: BTreeMap<usize, Vec<Observation>>,
}

thread_local! {
    static ACTIVE: std::cell::RefCell<Option<Capture>> = const { std::cell::RefCell::new(None) };
}

pub(crate) fn start(prompt_len: usize, indices: &[usize]) {
    assert!(prompt_len > 0);
    ACTIVE.with_borrow_mut(|active| {
        assert!(active.is_none(), "a logit replay is already active");
        *active = Some(Capture {
            first_position: prompt_len - 1,
            rows: indices.iter().map(|&index| (index, Vec::new())).collect(),
        });
    });
}

pub(crate) fn finish() -> Capture {
    ACTIVE.take().expect("active logit replay")
}

/// `position` is the absolute input position that predicted the first row.
pub(crate) fn observe(position: usize, logits: &MlxArray, count: usize, source: Source) {
    ACTIVE.with_borrow_mut(|active| {
        let Some(capture) = active else { return };
        for row in 0..count {
            let Some(index) = (position + row).checked_sub(capture.first_position) else {
                continue;
            };
            let Some(observations) = capture.rows.get_mut(&index) else {
                continue;
            };
            observations.push(Observation {
                logits: logits.clone(),
                row,
                source,
            });
        }
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub(crate) struct Scores {
    pub(crate) source: Source,
    pub(crate) emitted_logit: f32,
    pub(crate) rival_logit: f32,
    pub(crate) paired_gap: f64,
    pub(crate) runner_up_gap: f64,
}

fn score_row(row: &[f32], emitted: u32, rival: u32, source: Source) -> Option<Scores> {
    let emitted_id = emitted as usize;
    let emitted_logit = *row.get(emitted_id)?;
    let rival_logit = *row.get(rival as usize)?;
    let mut runner_up = f32::NEG_INFINITY;
    for (id, &logit) in row.iter().enumerate() {
        if !logit.is_finite() || logit > emitted_logit {
            return None;
        }
        if id != emitted_id {
            runner_up = runner_up.max(logit);
        }
    }
    if !runner_up.is_finite() {
        return None;
    }
    Some(Scores {
        source,
        emitted_logit,
        rival_logit,
        paired_gap: f64::from(emitted_logit) - f64::from(rival_logit),
        runner_up_gap: f64::from(emitted_logit) - f64::from(runner_up),
    })
}

impl Observation {
    fn scores(&self, emitted: u32, rival: u32) -> Result<Option<Scores>, String> {
        let shape = self.logits.shape();
        let Some(&vocab) = shape.last() else {
            return Err("logits have no vocabulary axis".into());
        };
        if vocab < 2 {
            return Err("logits have fewer than two vocabulary entries".into());
        }
        let logits = if self.source == Source::Verifier {
            if shape.len() != 2 || self.row >= shape[0] as usize {
                return Err("verifier logit row is out of bounds".into());
            }
            let row = self.row as i32;
            slice(&self.logits, &[row, 0], &[row + 1, vocab], &[1, 1], None)
        } else {
            self.logits.clone()
        };
        let logits = contiguous(&astype(&logits, MlxDtype::Float32, None), None);
        try_eval(&[&logits])?;
        let values = logits.data_f32();
        if values.len() != vocab as usize {
            return Err("expected a singleton target logit row".into());
        }
        Ok(score_row(values, emitted, rival, self.source))
    }
}

impl Capture {
    pub(crate) fn scores(&self, index: usize, emitted: u32, rival: u32) -> Result<Scores, String> {
        let observations = self
            .rows
            .get(&index)
            .ok_or("output index was not requested")?;
        let source = if observations
            .iter()
            .any(|row| row.source == Source::Verifier)
        {
            Source::Verifier
        } else {
            Source::Target
        };
        let mut result = None;
        for row in observations.iter().filter(|row| row.source == source) {
            if let Some(scores) = row.scores(emitted, rival)? {
                if result.is_some_and(|prior| prior != scores) {
                    return Err("target rows that select the output disagree on its scores".into());
                }
                result = Some(scores);
            }
        }
        result.ok_or_else(|| "no captured row selects the emitted token".into())
    }
}

#[test]
fn paired_gaps_use_emitted_and_rival_scores_and_preserve_ties() {
    let rows = [1.0, 4.0, 4.0, 3.0];
    let tied = score_row(&rows, 2, 1, Source::Target).unwrap();
    assert_eq!(tied.paired_gap, 0.0);
    assert_eq!(tied.runner_up_gap, 0.0);
    let other = score_row(&rows, 1, 0, Source::Verifier).unwrap();
    assert_eq!(other.paired_gap, 3.0);
    assert_eq!(other.runner_up_gap, 0.0);
    assert!(score_row(&rows, 0, 1, Source::Target).is_none());
    assert!(score_row(&rows, 1, 9, Source::Target).is_none());
    assert!(score_row(&[f32::NAN, 4.0], 1, 0, Source::Target).is_none());
}

#[test]
fn captures_committed_verifier_rows_and_rejects_ambiguous_targets() {
    let target_a = MlxArray::from_f32_slice(&[4.0, 1.0, 2.0]);
    let target_b = MlxArray::from_f32_slice(&[4.0, 2.0, 1.0]);
    let verify = reshape(
        &MlxArray::from_f32_slice(&[3.0, 1.0, 2.0, 1.0, 4.0, 2.0]),
        &[2, 3],
        None,
    );
    start(7, &[0, 1, 2, 3]);
    observe(5, &target_b, 1, Source::Target);
    observe(6, &target_a, 1, Source::Target);
    observe(6, &verify, 2, Source::Verifier);
    observe(8, &target_a, 1, Source::Target);
    observe(8, &target_b, 1, Source::Target);
    let capture = finish();
    let first = capture.scores(0, 0, 1).unwrap();
    assert_eq!(first.source, Source::Verifier);
    assert_eq!(first.paired_gap, 2.0);
    assert_eq!(capture.scores(1, 1, 2).unwrap().paired_gap, 2.0);
    assert!(capture.scores(1, 0, 1).is_err());
    assert!(capture.scores(2, 0, 1).is_err());
    assert!(capture.scores(3, 0, 1).is_err());
}
