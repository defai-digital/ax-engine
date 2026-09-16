//! Experimental Flash Next draft graph with authoritative primary verification.
//!
//! Flash Next MTP guarantees greedy-token identity with direct decode for the
//! same request, and bounds the logit and state divergence introduced by
//! batched verification. Bit-exact logits and serialized state versus direct
//! decode are no longer part of the contract. This matches the 27B route's
//! ADR-003 D5 greedy-parity rule.
//!
//! The sidecar has no published official forward oracle. Its candidate input
//! combiner uses a shared hidden projection per residual stream and adds a
//! separately projected token embedding. This hypothesis is never authority
//! for committed tokens; the primary graph verifies every proposal.

use std::time::Instant;

#[cfg(test)]
use std::cell::Cell;

use super::qwen4_exp::{self, Qwen4ExpOutput, Qwen4ExpState};
use super::shared::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::weights::qwen4_exp::{Qwen4ExpAttentionBranch, Qwen4ExpWeights};
use crate::weights::qwen4_exp_mtp::Qwen4ExpMtpWeights;
use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, astype, concatenate, multiply, reshape, rms_norm, slice,
    try_eval,
};

fn norm(input: &MlxArray, gain: &MlxArray, eps: f32) -> MlxArray {
    let unit = rms_norm(&astype(input, MlxDtype::Float32, None), None, eps, None);
    astype(&multiply(&unit, gain, None), input.dtype(), None)
}

fn prepare_head_input(
    head: &Qwen4ExpMtpWeights,
    stream_hidden: &MlxArray,
    next_tokens: &[u32],
) -> Result<MlxArray, String> {
    let layout = head.graph.layout;
    let (hidden, streams, width) = (
        layout.hidden_size() as i32,
        layout.stream_count() as i32,
        layout.packed_width() as i32,
    );
    let sequence =
        i32::try_from(next_tokens.len()).map_err(|_| "MTP sequence exceeds tensor dimensions")?;
    if next_tokens.is_empty()
        || stream_hidden.shape() != [1, sequence, width]
        || !matches!(
            stream_hidden.dtype(),
            MlxDtype::Float16 | MlxDtype::Bfloat16 | MlxDtype::Float32
        )
        || head.graph.layers.len() != 1
        || head.graph.layers[0].ple.is_some()
        || !matches!(
            head.graph.layers[0].attention,
            Qwen4ExpAttentionBranch::Qsa(_)
        )
        || next_tokens
            .iter()
            .any(|&token| token as usize >= head.graph.token_embedding.weight.shape()[0] as usize)
    {
        return Err("invalid Flash Next MTP input or one-layer QSA graph".into());
    }
    let stream_rows = sequence
        .checked_mul(streams)
        .ok_or("MTP stream rows exceed tensor dimensions")?;
    let embeddings = super::embed_tokens(next_tokens, &head.graph.token_embedding, hidden as usize);
    let normalized_hidden = norm(stream_hidden, &head.pre_fc_norm_hidden, head.rms_eps);
    let normalized_hidden = reshape(&normalized_hidden, &[1, stream_rows, hidden], None);
    let projected_hidden = qw_with_policy(
        &normalized_hidden,
        &head.fc_hidden,
        ProjectionBatchPolicy::RowExact,
    );
    let projected_hidden = reshape(&projected_hidden, &[1, sequence, streams, hidden], None);
    let normalized_embedding = norm(&embeddings, &head.pre_fc_norm_embedding, head.rms_eps);
    let projected_embedding = qw_with_policy(
        &normalized_embedding,
        &head.fc_embedding,
        ProjectionBatchPolicy::RowExact,
    );
    let projected_embedding = reshape(&projected_embedding, &[1, sequence, 1, hidden], None);
    Ok(reshape(
        &add(&projected_hidden, &projected_embedding, None),
        &[1, sequence, width],
        None,
    ))
}

/// Draft state belongs to the separate one-layer graph, never the trunk.
/// The returned state is unpublished until its caller commits the transaction.
pub(crate) fn head_forward(
    head: &Qwen4ExpMtpWeights,
    stream_hidden: &MlxArray,
    next_tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
) -> Result<Qwen4ExpOutput, String> {
    let prepared = prepare_head_input(head, stream_hidden, next_tokens)?;
    qwen4_exp::forward_prepared(
        &head.graph,
        next_tokens,
        prepared,
        state,
        owner,
        ProjectionBatchPolicy::RowExact,
    )
}

fn head_advance_cache(
    head: &Qwen4ExpMtpWeights,
    stream_hidden: &MlxArray,
    next_tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
) -> Result<Qwen4ExpState, String> {
    let prepared = prepare_head_input(head, stream_hidden, next_tokens)?;
    qwen4_exp::advance_prepared_qsa_cache(
        &head.graph,
        next_tokens,
        prepared,
        state,
        owner,
        ProjectionBatchPolicy::RowExact,
    )
}

#[cfg(test)]
thread_local! {
    static TRUNK_FORWARD_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn trunk_forward(
    trunk: &Qwen4ExpWeights,
    tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
) -> Result<Qwen4ExpOutput, String> {
    #[cfg(test)]
    TRUNK_FORWARD_COUNT.with(|count| count.set(count.get().saturating_add(1)));
    qwen4_exp::forward(trunk, tokens, state, owner, ProjectionBatchPolicy::Shared)
}

fn token_at_row(logits: &MlxArray, row: i32) -> Result<u32, String> {
    let shape = logits.shape();
    if shape.len() != 2 || row < 0 || row >= shape[0] {
        return Err("Flash Next MTP logits row is missing".into());
    }
    let row_logits = slice(logits, &[row, 0], &[row + 1, shape[1]], &[1, 1], None);
    let token = argmax(&row_logits, None);
    try_eval(&[&token])?;
    Ok(token.data_u32()[0])
}

fn next_token(output: &Qwen4ExpOutput) -> Result<u32, String> {
    let shape = output.logits.shape();
    if shape.len() != 2 || shape[0] <= 0 {
        return Err("Flash Next MTP logits row is missing".into());
    }
    token_at_row(&output.logits, shape[0] - 1)
}

fn take_sequence_row(array: &MlxArray, row: i32) -> Result<MlxArray, String> {
    let shape = array.shape();
    if shape.len() != 3 || shape[0] != 1 || row < 0 || row >= shape[1] {
        return Err("Flash Next MTP batched sequence row is missing".into());
    }
    Ok(slice(
        array,
        &[0, row, 0],
        &[1, row + 1, shape[2]],
        &[1, 1, 1],
        None,
    ))
}

fn output_row(output: &Qwen4ExpOutput, row: i32) -> Result<Qwen4ExpOutput, String> {
    let logits = {
        let shape = output.logits.shape();
        if shape.len() != 2 || row < 0 || row >= shape[0] {
            return Err("Flash Next MTP batched logits row is missing".into());
        }
        slice(
            &output.logits,
            &[row, 0],
            &[row + 1, shape[1]],
            &[1, 1],
            None,
        )
    };
    let stream_hidden = take_sequence_row(&output.stream_hidden, row)?;
    let hidden = take_sequence_row(&output.hidden, row)?;
    try_eval(&[&logits, &stream_hidden, &hidden])?;
    Ok(Qwen4ExpOutput {
        stream_hidden,
        hidden,
        logits,
        state: output.state.clone(),
    })
}

fn sum_verify_wall_us(correction_wall_us: u32, bonus_wall_us: u32, rejection_wall_us: u32) -> u32 {
    correction_wall_us
        .saturating_add(bonus_wall_us)
        .saturating_add(rejection_wall_us)
}

pub(crate) struct VerifiedStep {
    pub committed: Vec<u32>,
    pub accepted: bool,
    pub after_primary: Qwen4ExpOutput,
    pub after_draft: Option<Qwen4ExpOutput>,
    pub next_primary: u32,
    pub correction_wall_us: u32,
    pub bonus_wall_us: u32,
    pub rejection_wall_us: u32,
}

impl VerifiedStep {
    fn verify_wall_us(&self) -> u32 {
        sum_verify_wall_us(
            self.correction_wall_us,
            self.bonus_wall_us,
            self.rejection_wall_us,
        )
    }
}

/// `primary` must be the authoritative token from this checkpoint's logits.
/// A mismatched or truncated draft is never evaluated as a committed token.
///
/// When more than one output slot remains, verification is one length-2 Shared
/// trunk forward of `[primary, draft]`. The correction token is logits row 0.
/// Acceptance commits the two-token state and reads the bonus from row 1
/// (`bonus_wall_us` is then zero). Rejection runs one extra singleton forward
/// of `[primary]` from the original state so the committed state matches
/// direct decode (`rejection_wall_us`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_one(
    trunk: &Qwen4ExpWeights,
    state: &Qwen4ExpState,
    owner: u64,
    primary: u32,
    draft: u32,
    remaining: usize,
    terminal_ids: &[u32],
) -> Result<VerifiedStep, String> {
    if remaining == 0 {
        return Err("Flash Next MTP requires remaining output budget".into());
    }
    if remaining == 1 {
        let started = Instant::now();
        let after_primary = trunk_forward(trunk, &[primary], state, owner)?;
        let correction_wall_us = elapsed_us(started);
        let correction = next_token(&after_primary)?;
        return Ok(VerifiedStep {
            committed: vec![primary],
            accepted: false,
            after_primary,
            after_draft: None,
            next_primary: correction,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us: 0,
        });
    }
    let started = Instant::now();
    let batched = trunk_forward(trunk, &[primary, draft], state, owner)?;
    let correction_wall_us = elapsed_us(started);
    if batched.logits.shape().first().copied() != Some(2) {
        return Err("Flash Next MTP batched verify requires two logit rows".into());
    }
    let correction = token_at_row(&batched.logits, 0)?;
    let accepted = !terminal_ids.contains(&primary)
        && !terminal_ids.contains(&correction)
        && draft == correction;
    if accepted {
        let bonus = token_at_row(&batched.logits, 1)?;
        Ok(VerifiedStep {
            committed: vec![primary, draft],
            accepted: true,
            after_primary: output_row(&batched, 0)?,
            after_draft: Some(output_row(&batched, 1)?),
            next_primary: bonus,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us: 0,
        })
    } else {
        let started = Instant::now();
        let after_primary = trunk_forward(trunk, &[primary], state, owner)?;
        let rejection_wall_us = elapsed_us(started);
        let next_primary = next_token(&after_primary)?;
        Ok(VerifiedStep {
            committed: vec![primary],
            accepted: false,
            after_primary,
            after_draft: None,
            next_primary,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us,
        })
    }
}

/// Request-owned draft history, kept separate from the authoritative KV cache.
#[derive(Clone)]
pub(crate) struct Qwen4ExpDraftCursor {
    draft_state: Qwen4ExpState,
    stream_hidden: Option<MlxArray>,
    owner: u64,
    pub proposed: usize,
    pub accepted: usize,
}

pub(crate) struct CursorStep {
    pub trunk_state: Qwen4ExpState,
    pub committed_len: usize,
    pub emitted: Vec<u32>,
    pub accepted: bool,
    pub draft_wall_us: u32,
    pub correction_wall_us: u32,
    pub bonus_wall_us: u32,
    pub rejection_wall_us: u32,
    pub verify_wall_us: u32,
}

struct AdvancedStep {
    trunk_state: Qwen4ExpState,
    consumed: Vec<u32>,
    next_primary: u32,
    accepted: bool,
    draft_wall_us: u32,
    correction_wall_us: u32,
    bonus_wall_us: u32,
    rejection_wall_us: u32,
    verify_wall_us: u32,
}

fn elapsed_us(started: Instant) -> u32 {
    u32::try_from(started.elapsed().as_micros()).unwrap_or(u32::MAX)
}

impl Qwen4ExpDraftCursor {
    pub(crate) fn new(head: &Qwen4ExpMtpWeights, trunk_owner: u64) -> Self {
        let owner = trunk_owner ^ 0x5146_4e4d_5450_0001;
        Self {
            draft_state: Qwen4ExpState::new(&head.graph, owner),
            stream_hidden: None,
            owner,
            proposed: 0,
            accepted: 0,
        }
    }

    pub(crate) fn aligned(&self, trunk: &Qwen4ExpState) -> bool {
        self.stream_hidden.is_some()
            && self.draft_state.position().checked_add(1) == Some(trunk.position())
    }

    /// Pair each committed prompt token with the preceding trunk row. Only
    /// one row survives a chunk boundary; no second trunk prefill is needed.
    pub(crate) fn absorb(
        &mut self,
        head: &Qwen4ExpMtpWeights,
        tokens: &[u32],
        rows: &MlxArray,
    ) -> Result<(), String> {
        let count = i32::try_from(tokens.len()).map_err(|_| "MTP prompt chunk too large")?;
        let width = head.graph.layout.packed_width() as i32;
        if count == 0 || rows.shape() != [1, count, width] {
            return Err("invalid MTP prompt stream rows".into());
        }
        let preceding = if count > 1 {
            Some(slice(
                rows,
                &[0, 0, 0],
                &[1, count - 1, width],
                &[1, 1, 1],
                None,
            ))
        } else {
            None
        };
        let (pairs, shifted) = match (&self.stream_hidden, preceding) {
            (Some(previous), Some(prefix)) => {
                (Some(concatenate(&[previous, &prefix], 1, None)), tokens)
            }
            (Some(previous), None) => (Some(previous.clone()), tokens),
            (None, prefix) => (prefix, &tokens[1..]),
        };
        let staged = if let Some(pairs) = pairs {
            head_advance_cache(head, &pairs, shifted, &self.draft_state, self.owner)?
        } else {
            self.draft_state.clone()
        };
        let last = slice(
            rows,
            &[0, count - 1, 0],
            &[1, count, width],
            &[1, 1, 1],
            None,
        );
        try_eval(&[&last])?;
        self.draft_state = staged;
        self.stream_hidden = Some(last);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        trunk: &Qwen4ExpWeights,
        head: &Qwen4ExpMtpWeights,
        state: &Qwen4ExpState,
        trunk_owner: u64,
        primary: u32,
        remaining: usize,
        terminal_ids: &[u32],
    ) -> Result<AdvancedStep, String> {
        if remaining == 0 || !self.aligned(state) || self.owner == trunk_owner {
            return Err("invalid Flash Next MTP budget or draft/trunk alignment".into());
        }
        let hidden = self
            .stream_hidden
            .as_ref()
            .ok_or("missing MTP stream row")?;
        if remaining == 1 {
            // No draft can be accepted, but its QSA history must stay aligned.
            let started = Instant::now();
            let draft_state =
                head_advance_cache(head, hidden, &[primary], &self.draft_state, self.owner)?;
            let draft_wall_us = elapsed_us(started);
            let started = Instant::now();
            let output = trunk_forward(trunk, &[primary], state, trunk_owner)?;
            let next_primary = next_token(&output)?;
            let correction_wall_us = elapsed_us(started);
            self.draft_state = draft_state;
            self.stream_hidden = Some(output.stream_hidden);
            return Ok(AdvancedStep {
                trunk_state: output.state,
                consumed: vec![primary],
                next_primary,
                accepted: false,
                draft_wall_us,
                correction_wall_us,
                bonus_wall_us: 0,
                rejection_wall_us: 0,
                verify_wall_us: correction_wall_us,
            });
        }
        let draft_started = Instant::now();
        let proposed = head_forward(head, hidden, &[primary], &self.draft_state, self.owner)?;
        let draft = next_token(&proposed)?;
        let mut draft_wall_us = elapsed_us(draft_started);
        let verified = verify_one(
            trunk,
            state,
            trunk_owner,
            primary,
            draft,
            remaining,
            terminal_ids,
        )?;
        let accepted = verified.accepted;
        let next_primary = verified.next_primary;
        let correction_wall_us = verified.correction_wall_us;
        let bonus_wall_us = verified.bonus_wall_us;
        let rejection_wall_us = verified.rejection_wall_us;
        let verify_wall_us = verified.verify_wall_us();
        let consumed = verified.committed;
        let (draft_state, final_output) = if let Some(after_draft) = verified.after_draft {
            let alignment_started = Instant::now();
            let aligned = head_advance_cache(
                head,
                &verified.after_primary.stream_hidden,
                &[draft],
                &proposed.state,
                self.owner,
            )?;
            draft_wall_us = draft_wall_us.saturating_add(elapsed_us(alignment_started));
            (aligned, after_draft)
        } else {
            (proposed.state, verified.after_primary)
        };
        self.draft_state = draft_state;
        self.stream_hidden = Some(final_output.stream_hidden);
        self.proposed += 1;
        self.accepted += usize::from(accepted);
        Ok(AdvancedStep {
            trunk_state: final_output.state,
            consumed,
            next_primary,
            accepted,
            draft_wall_us,
            correction_wall_us,
            bonus_wall_us,
            rejection_wall_us,
            verify_wall_us,
        })
    }

    /// The runner has already emitted `primary`. Return only newly predicted
    /// tokens, and never feed a terminal correction merely to obtain a bonus.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        trunk: &Qwen4ExpWeights,
        head: &Qwen4ExpMtpWeights,
        state: &Qwen4ExpState,
        trunk_owner: u64,
        primary: u32,
        remaining: usize,
        terminal_ids: &[u32],
    ) -> Result<CursorStep, String> {
        let advanced = self.advance(
            trunk,
            head,
            state,
            trunk_owner,
            primary,
            remaining,
            terminal_ids,
        )?;
        let committed_len = advanced.consumed.len();
        let mut emitted = advanced.consumed.into_iter().skip(1).collect::<Vec<_>>();
        emitted.push(advanced.next_primary);
        Ok(CursorStep {
            trunk_state: advanced.trunk_state,
            committed_len,
            emitted,
            accepted: advanced.accepted,
            draft_wall_us: advanced.draft_wall_us,
            correction_wall_us: advanced.correction_wall_us,
            bonus_wall_us: advanced.bonus_wall_us,
            rejection_wall_us: advanced.rejection_wall_us,
            verify_wall_us: advanced.verify_wall_us,
        })
    }
}

/// A development candidate session. No implicit attachment or readiness grant.
#[cfg(test)]
#[derive(Clone)]
pub(crate) struct CandidateSession {
    pub trunk_state: Qwen4ExpState,
    pub draft_state: Qwen4ExpState,
    pub stream_hidden: MlxArray,
    pub primary: u32,
    trunk_owner: u64,
    draft_owner: u64,
    pub proposed: usize,
    pub accepted: usize,
}

#[cfg(test)]
impl CandidateSession {
    /// Match production's n-1 plus singleton prefill schedule, then warm the
    /// draft cache with shifted prompt pairs. No prompt token is speculative.
    pub(crate) fn prefill(
        trunk: &Qwen4ExpWeights,
        head: &Qwen4ExpMtpWeights,
        tokens: &[u32],
        trunk_owner: u64,
        draft_owner: u64,
    ) -> Result<Self, String> {
        if tokens.is_empty() || trunk_owner == draft_owner {
            return Err("MTP needs a prompt and distinct trunk/draft owners".into());
        }
        let mut trunk_state = Qwen4ExpState::new(trunk, trunk_owner);
        let mut draft_state = Qwen4ExpState::new(&head.graph, draft_owner);
        if tokens.len() > 1 {
            let prefix = qwen4_exp::forward(
                trunk,
                &tokens[..tokens.len() - 1],
                &trunk_state,
                trunk_owner,
                ProjectionBatchPolicy::Shared,
            )?;
            let warmed = head_advance_cache(
                head,
                &prefix.stream_hidden,
                &tokens[1..],
                &draft_state,
                draft_owner,
            )?;
            trunk_state = prefix.state;
            draft_state = warmed;
        }
        let output = qwen4_exp::forward(
            trunk,
            &tokens[tokens.len() - 1..],
            &trunk_state,
            trunk_owner,
            ProjectionBatchPolicy::Shared,
        )?;
        Ok(Self {
            primary: next_token(&output)?,
            trunk_state: output.state,
            draft_state,
            stream_hidden: output.stream_hidden,
            trunk_owner,
            draft_owner,
            proposed: 0,
            accepted: 0,
        })
    }

    pub(crate) fn step(
        &mut self,
        trunk: &Qwen4ExpWeights,
        head: &Qwen4ExpMtpWeights,
        remaining: usize,
        terminal_ids: &[u32],
    ) -> Result<Vec<u32>, String> {
        let mut cursor = Qwen4ExpDraftCursor {
            draft_state: self.draft_state.clone(),
            stream_hidden: Some(self.stream_hidden.clone()),
            owner: self.draft_owner,
            proposed: self.proposed,
            accepted: self.accepted,
        };
        let advanced = cursor.advance(
            trunk,
            head,
            &self.trunk_state,
            self.trunk_owner,
            self.primary,
            remaining,
            terminal_ids,
        )?;
        self.trunk_state = advanced.trunk_state;
        self.draft_state = cursor.draft_state;
        self.stream_hidden = cursor.stream_hidden.ok_or("missing committed MTP row")?;
        self.primary = advanced.next_primary;
        self.proposed = cursor.proposed;
        self.accepted = cursor.accepted;
        Ok(advanced.consumed)
    }
}

#[cfg(test)]
pub(crate) mod mtp_parity {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const DEFAULT_REAL_PACK_TOLERANCE: f32 = 0.1;
    const REAL_PACK_TOLERANCE_ENV: &str = "AX_FLASH_NEXT_MTP_REAL_TOLERANCE";
    const TOLERANCE_SOURCE_SYNTHETIC: &str = "synthetic";
    const TOLERANCE_SOURCE_REAL_PACK: &str = "real-pack-env";

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct MtpTolerance {
        pub limit: f32,
        pub source: &'static str,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(crate) struct MtpDivergence {
        pub scale: f32,
        pub max_abs: f32,
        pub relative: f32,
    }

    impl MtpDivergence {
        pub(crate) fn zero() -> Self {
            Self {
                scale: 0.0,
                max_abs: 0.0,
                relative: 0.0,
            }
        }

        pub(crate) fn max_relative(self, other: Self) -> Self {
            if other.relative > self.relative {
                other
            } else {
                self
            }
        }
    }

    pub(crate) fn mtp_numeric_tolerance(dtype: MlxDtype) -> f32 {
        match dtype {
            MlxDtype::Float32 => 1e-3,
            _ => 3e-2,
        }
    }

    fn real_pack_env_set() -> bool {
        std::env::var_os("AX_FLASH_NEXT_REAL_PACK").is_some()
            || std::env::var_os("AX_FLASH_NEXT_CANDIDATE_PACK_DIR").is_some()
    }

    pub(crate) fn parse_mtp_real_pack_tolerance(raw: Option<&str>) -> f32 {
        let Some(value) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
            return DEFAULT_REAL_PACK_TOLERANCE;
        };
        let parsed = value.parse::<f32>();
        assert!(
            matches!(parsed, Ok(limit) if limit.is_finite() && limit >= 0.0),
            "{REAL_PACK_TOLERANCE_ENV} must be a finite non-negative f32, got {value:?}"
        );
        parsed.unwrap()
    }

    /// Selects the relative MTP numeric tolerance for this run.
    ///
    /// Fixture runs keep the synthetic floors (1e-3 F32, 3e-2 otherwise). On a
    /// real quantized 125B graph the batched two-token verify schedule can
    /// diverge from two singletons by the same order as the official
    /// chunked-versus-recurrent prefill schedule difference (max abs 2.2
    /// logits). When `AX_FLASH_NEXT_REAL_PACK` or
    /// `AX_FLASH_NEXT_CANDIDATE_PACK_DIR` is set, the relative limit is
    /// `AX_FLASH_NEXT_MTP_REAL_TOLERANCE` (default 0.1).
    pub(crate) fn mtp_run_tolerance(dtype: MlxDtype) -> MtpTolerance {
        mtp_run_tolerance_from_parts(
            dtype,
            real_pack_env_set(),
            std::env::var(REAL_PACK_TOLERANCE_ENV).ok().as_deref(),
        )
    }

    pub(crate) fn mtp_run_tolerance_from_parts(
        dtype: MlxDtype,
        real_pack: bool,
        real_tolerance_env: Option<&str>,
    ) -> MtpTolerance {
        if real_pack {
            MtpTolerance {
                limit: parse_mtp_real_pack_tolerance(real_tolerance_env),
                source: TOLERANCE_SOURCE_REAL_PACK,
            }
        } else {
            MtpTolerance {
                limit: mtp_numeric_tolerance(dtype),
                source: TOLERANCE_SOURCE_SYNTHETIC,
            }
        }
    }

    fn f32_values(array: &MlxArray) -> Vec<f32> {
        let array = mlx_sys::contiguous(&astype(array, MlxDtype::Float32, None), None);
        try_eval(&[&array]).unwrap();
        array.data_f32().to_vec()
    }

    fn max_abs_relative_to_scale(actual: &[f32], expected: &[f32]) -> MtpDivergence {
        assert_eq!(actual.len(), expected.len(), "MTP tensor length mismatch");
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| {
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0.0f32, f32::max);
        let relative = if scale > 0.0 {
            max_abs / scale
        } else {
            max_abs
        };
        MtpDivergence {
            scale,
            max_abs,
            relative,
        }
    }

    pub(crate) fn mtp_logits_divergence(actual: &MlxArray, expected: &MlxArray) -> MtpDivergence {
        max_abs_relative_to_scale(&f32_values(actual), &f32_values(expected))
    }

    pub(crate) fn mtp_state_divergence(
        actual: &Qwen4ExpState,
        expected: &Qwen4ExpState,
    ) -> MtpDivergence {
        assert_eq!(actual.position(), expected.position());
        let actual_arrays = actual.arrays();
        let expected_arrays = expected.arrays();
        assert_eq!(actual_arrays.len(), expected_arrays.len());
        actual_arrays
            .iter()
            .zip(&expected_arrays)
            .map(|(actual, expected)| mtp_logits_divergence(actual, expected))
            .fold(MtpDivergence::zero(), MtpDivergence::max_relative)
    }

    pub(crate) fn assert_mtp_logits_close(actual: &MlxArray, expected: &MlxArray, dtype: MlxDtype) {
        let divergence = mtp_logits_divergence(actual, expected);
        let tolerance = mtp_run_tolerance(dtype);
        assert!(
            divergence.relative <= tolerance.limit,
            "MTP logit relative max abs {} exceeds {} (source={}, logit_scale={}, max_abs={})",
            divergence.relative,
            tolerance.limit,
            tolerance.source,
            divergence.scale,
            divergence.max_abs
        );
    }

    pub(crate) fn assert_mtp_state_close(
        actual: &Qwen4ExpState,
        expected: &Qwen4ExpState,
        dtype: MlxDtype,
    ) {
        assert_eq!(actual.position(), expected.position());
        let actual_arrays = actual.arrays();
        let expected_arrays = expected.arrays();
        assert_eq!(actual_arrays.len(), expected_arrays.len());
        let tolerance = mtp_run_tolerance(dtype);
        for (index, (actual, expected)) in actual_arrays.iter().zip(&expected_arrays).enumerate() {
            let divergence = mtp_logits_divergence(actual, expected);
            assert!(
                divergence.relative <= tolerance.limit,
                "MTP state array {index} relative max abs {} exceeds {} (source={}, scale={}, max_abs={})",
                divergence.relative,
                tolerance.limit,
                tolerance.source,
                divergence.scale,
                divergence.max_abs
            );
        }
    }

    pub(crate) fn take_trunk_forward_count() -> usize {
        TRUNK_FORWARD_COUNT.with(|count| count.replace(0))
    }

    #[test]
    fn mtp_numeric_tolerance_matches_f32_and_bf16_fixtures() {
        assert_eq!(mtp_numeric_tolerance(MlxDtype::Float32), 1e-3);
        assert_eq!(mtp_numeric_tolerance(MlxDtype::Bfloat16), 3e-2);
    }

    #[test]
    fn mtp_run_tolerance_parses_real_pack_env_and_keeps_synthetic_default() {
        let unset = mtp_run_tolerance_from_parts(MlxDtype::Bfloat16, true, None);
        assert_eq!(unset.limit, 0.1);
        assert_eq!(unset.source, "real-pack-env");
        let empty = mtp_run_tolerance_from_parts(MlxDtype::Bfloat16, true, Some(""));
        assert_eq!(empty.limit, 0.1);
        assert_eq!(empty.source, "real-pack-env");
        let override_limit = mtp_run_tolerance_from_parts(MlxDtype::Float32, true, Some(" 0.08 "));
        assert_eq!(override_limit.limit, 0.08);
        assert_eq!(override_limit.source, "real-pack-env");
        assert_eq!(parse_mtp_real_pack_tolerance(None), 0.1);
        assert_eq!(parse_mtp_real_pack_tolerance(Some("0.2")), 0.2);
        let synthetic = mtp_run_tolerance_from_parts(MlxDtype::Float32, false, Some("0.9"));
        assert_eq!(synthetic.limit, 1e-3);
        assert_eq!(synthetic.source, "synthetic");
        let synthetic_bf16 = mtp_run_tolerance_from_parts(MlxDtype::Bfloat16, false, Some("0.9"));
        assert_eq!(synthetic_bf16.limit, 3e-2);
        assert_eq!(synthetic_bf16.source, "synthetic");
    }

    #[test]
    fn mtp_relative_divergence_is_zero_for_identical_values() {
        let identical = max_abs_relative_to_scale(&[1.0, -2.0, 0.5], &[1.0, -2.0, 0.5]);
        assert_eq!(identical.relative, 0.0);
        assert_eq!(identical.max_abs, 0.0);
        assert_eq!(identical.scale, 2.0);
        let shifted = max_abs_relative_to_scale(&[1.02, 0.0], &[1.0, 0.0]);
        assert_eq!(shifted.scale, 1.0);
        assert!((shifted.max_abs - 0.02).abs() <= 1e-6);
        assert!(shifted.relative > 0.0 && shifted.relative <= 0.021);
    }
}

#[cfg(test)]
mod cursor_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::kv_cache::MlxKVCache;
    use std::path::PathBuf;

    fn bytes(state: &Qwen4ExpState) -> Vec<u8> {
        state_bytes(state, 1)
    }

    fn state_bytes(state: &Qwen4ExpState, layers: usize) -> Vec<u8> {
        let mut cache = MlxKVCache::new_contiguous(layers);
        cache.qwen4_exp = Some(state.clone());
        cache.advance(state.position());
        cache.serialize_to_bytes()
    }

    fn values(array: &MlxArray) -> Vec<f32> {
        let array = mlx_sys::contiguous(&astype(array, MlxDtype::Float32, None), None);
        try_eval(&[&array]).unwrap();
        array.data_f32().to_vec()
    }

    fn broken_lazy_array(original: &MlxArray) -> MlxArray {
        let kernel = mlx_sys::MlxMetalKernel::new(
            "ax_flash_next_mtp_cache_failure",
            &["input"],
            &["output"],
            "output[thread_position_in_grid.x] = ax_intentionally_undefined_symbol;",
            "",
            true,
        );
        kernel
            .try_apply_with_template(
                &[original],
                &[mlx_sys::KernelOutputSpec {
                    shape: original.shape(),
                    dtype: original.dtype(),
                }],
                &[],
                (original.shape().iter().product(), 1, 1),
                (32, 1, 1),
                None,
            )
            .unwrap()
            .remove(0)
    }

    #[test]
    #[ignore = "requires synthetic or real Flash Next MTP artifacts and Metal"]
    fn flash_next_final_budget_skips_proposal_and_preserves_transactional_state() {
        let (root, manifest, paging) =
            if let Some(root) = std::env::var_os("AX_FLASH_NEXT_REAL_PACK") {
                let root = PathBuf::from(root);
                let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&root).unwrap();
                (
                    root,
                    artifacts.manifest().clone(),
                    crate::expert_stream::StreamExpertsMode::On,
                )
            } else {
                let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
                let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
                manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
                (root, manifest, crate::expert_stream::StreamExpertsMode::Off)
            };
        let mut trunk =
            crate::weights::qwen4_exp::load_with_paging_policy(&root, &manifest, paging, 1)
                .unwrap();
        let mut head = crate::weights::qwen4_exp_mtp::load(&root, &manifest, &trunk).unwrap();
        let owner = 1801;
        let tokens = [1, 2, 3, 4];
        let prefill = qwen4_exp::forward(
            &trunk,
            &tokens,
            &Qwen4ExpState::new(&trunk, owner),
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let primary = next_token(&prefill).unwrap();
        let mut cursor = Qwen4ExpDraftCursor::new(&head, owner);
        cursor
            .absorb(&head, &tokens, &prefill.stream_hidden)
            .unwrap();
        let before = bytes(&cursor.draft_state);
        let trunk_before = state_bytes(&prefill.state, trunk.layers.len());
        let hidden_before = cursor.stream_hidden.as_ref().unwrap().clone();
        let expected_head = head_forward(
            &head,
            &hidden_before,
            &[primary],
            &cursor.draft_state,
            cursor.owner,
        )
        .unwrap();
        let expected = qwen4_exp::forward(
            &trunk,
            &[primary],
            &prefill.state,
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let expected_token = next_token(&expected).unwrap();

        let head_projection = head.graph.lm_head.weight.clone();
        head.graph.lm_head.weight = broken_lazy_array(&head_projection);
        let mut actual = cursor.clone();
        let result = actual
            .step(&trunk, &head, &prefill.state, owner, primary, 1, &[])
            .expect("final budget must not evaluate discarded draft logits");
        assert_eq!(result.emitted, vec![expected_token]);
        assert_eq!(result.committed_len, 1);
        assert!(!result.accepted);
        assert_eq!((actual.proposed, actual.accepted), (0, 0));
        assert_eq!(bytes(&actual.draft_state), bytes(&expected_head.state));
        assert_eq!(
            state_bytes(&result.trunk_state, trunk.layers.len()),
            state_bytes(&expected.state, trunk.layers.len())
        );
        assert_eq!(
            values(actual.stream_hidden.as_ref().unwrap()),
            values(&expected.stream_hidden)
        );
        assert!(actual.aligned(&result.trunk_state));
        assert!(
            cursor
                .step(&trunk, &head, &prefill.state, owner, primary, 2, &[])
                .is_err()
        );
        head.graph.lm_head.weight = head_projection;

        // Required cache work and a later primary failure must publish neither state.
        for fail_primary in [false, true] {
            let saved = if fail_primary {
                trunk.lm_head.weight.clone()
            } else {
                head.fc_hidden.weight.clone()
            };
            if fail_primary {
                trunk.lm_head.weight = broken_lazy_array(&saved);
            } else {
                head.fc_hidden.weight = broken_lazy_array(&saved);
            }
            assert!(
                cursor
                    .step(&trunk, &head, &prefill.state, owner, primary, 1, &[])
                    .is_err()
            );
            if fail_primary {
                trunk.lm_head.weight = saved;
            } else {
                head.fc_hidden.weight = saved;
            }
            assert_eq!(bytes(&cursor.draft_state), before);
            assert_eq!(
                state_bytes(&prefill.state, trunk.layers.len()),
                trunk_before
            );
            assert_eq!((cursor.proposed, cursor.accepted), (0, 0));
            assert_eq!(
                values(cursor.stream_hidden.as_ref().unwrap()),
                values(&hidden_before)
            );
        }
        let retry = cursor
            .step(&trunk, &head, &prefill.state, owner, primary, 1, &[])
            .unwrap();
        assert_eq!(retry.emitted, result.emitted);
        assert_eq!(bytes(&cursor.draft_state), bytes(&actual.draft_state));
        assert_eq!(
            state_bytes(&retry.trunk_state, trunk.layers.len()),
            state_bytes(&result.trunk_state, trunk.layers.len())
        );
        let mut reference = Qwen4ExpDraftCursor {
            draft_state: expected_head.state,
            stream_hidden: Some(expected.stream_hidden),
            owner: cursor.owner,
            proposed: 0,
            accepted: 0,
        };
        let resumed = cursor
            .step(
                &trunk,
                &head,
                &retry.trunk_state,
                owner,
                expected_token,
                2,
                &[],
            )
            .unwrap();
        let control = reference
            .step(
                &trunk,
                &head,
                &expected.state,
                owner,
                expected_token,
                2,
                &[],
            )
            .unwrap();
        assert_eq!(resumed.emitted, control.emitted);
        assert_eq!(
            state_bytes(&resumed.trunk_state, trunk.layers.len()),
            state_bytes(&control.trunk_state, trunk.layers.len())
        );
        assert_eq!(bytes(&cursor.draft_state), bytes(&reference.draft_state));
        assert_eq!(
            values(cursor.stream_hidden.as_ref().unwrap()),
            values(reference.stream_hidden.as_ref().unwrap())
        );
        assert_eq!(
            (cursor.proposed, cursor.accepted),
            (reference.proposed, reference.accepted)
        );
    }

    #[test]
    #[ignore = "requires synthetic or real Flash Next MTP artifacts and Metal"]
    fn flash_next_cache_only_head_matches_full_state_and_preserves_failures() {
        let (root, manifest) = if let Some(root) = std::env::var_os("AX_FLASH_NEXT_REAL_PACK") {
            let root = PathBuf::from(root);
            let artifacts = ax_engine_core::NativeModelArtifacts::from_dir(&root).unwrap();
            (root, artifacts.manifest().clone())
        } else {
            let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
            let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
            manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
            (root, manifest)
        };
        let trunk = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let mut head = crate::weights::qwen4_exp_mtp::load(&root, &manifest, &trunk).unwrap();
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let output = qwen4_exp::forward(
            &trunk,
            &tokens,
            &Qwen4ExpState::new(&trunk, 1701),
            1701,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let owner = 1702;
        let width = head.graph.layout.packed_width() as i32;
        assert!(
            qwen4_exp::advance_prepared_qsa_cache(
                &trunk,
                &tokens,
                output.stream_hidden.clone(),
                &output.state,
                1701,
                ProjectionBatchPolicy::RowExact,
            )
            .is_err()
        );
        for chunk_size in [1, 3, 9] {
            let mut reference = Qwen4ExpState::new(&head.graph, owner);
            let mut actual = reference.clone();
            let mut offset = 0;
            for chunk in tokens.chunks(chunk_size) {
                let end = offset + chunk.len();
                let rows = slice(
                    &output.stream_hidden,
                    &[0, offset as i32, 0],
                    &[1, end as i32, width],
                    &[1, 1, 1],
                    None,
                );
                reference = head_forward(&head, &rows, chunk, &reference, owner)
                    .unwrap()
                    .state;
                actual = head_advance_cache(&head, &rows, chunk, &actual, owner).unwrap();
                assert_eq!(bytes(&actual), bytes(&reference));
                offset = end;
            }
            let row = slice(
                &output.stream_hidden,
                &[0, 8, 0],
                &[1, 9, width],
                &[1, 1, 1],
                None,
            );
            let expected = head_forward(&head, &row, &[2], &reference, owner).unwrap();
            let proposal = head_forward(&head, &row, &[2], &actual, owner).unwrap();
            assert_eq!(proposal.logits.data_f32(), expected.logits.data_f32());
            assert_eq!(bytes(&proposal.state), bytes(&expected.state));

            let before = bytes(&actual);
            assert!(head_advance_cache(&head, &row, &[2], &actual, owner + 1).is_err());
            assert!(head_advance_cache(&head, &row, &[u32::MAX], &actual, owner).is_err());
            assert!(head_advance_cache(&head, &row, &[], &actual, owner).is_err());

            // A vocabulary projection is not a dependency of the QSA cache.
            let original = head.graph.lm_head.weight.clone();
            head.graph.lm_head.weight = broken_lazy_array(&original);
            let cached = head_advance_cache(&head, &row, &[2], &actual, owner).unwrap();
            assert_eq!(bytes(&cached), bytes(&expected.state));
            assert!(head_forward(&head, &row, &[2], &actual, owner).is_err());
            head.graph.lm_head.weight = original;

            // The pre-FC hidden projection is required. Its failure must not
            // publish any cache growth, and retry must recover exactly.
            let original = head.fc_hidden.weight.clone();
            head.fc_hidden.weight = broken_lazy_array(&original);
            assert!(head_advance_cache(&head, &row, &[2], &actual, owner).is_err());
            head.fc_hidden.weight = original;
            assert_eq!(bytes(&actual), before);
            let recovered = head_advance_cache(&head, &row, &[2], &actual, owner).unwrap();
            assert_eq!(bytes(&recovered), bytes(&expected.state));
        }
    }

    #[test]
    #[ignore = "requires synthetic Flash Next MTP artifacts"]
    fn flash_next_cursor_absorb_chunking_preserves_exact_draft_keys_and_failure_state() {
        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
        let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
        let trunk = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let head = crate::weights::qwen4_exp_mtp::load(&root, &manifest, &trunk).unwrap();
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let owner = 1601;
        let output = qwen4_exp::forward(
            &trunk,
            &tokens,
            &Qwen4ExpState::new(&trunk, owner),
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let width = head.graph.layout.packed_width() as i32;
        let mut whole = Qwen4ExpDraftCursor::new(&head, owner);
        whole.absorb(&head, &tokens, &output.stream_hidden).unwrap();
        for chunk_size in [1, 2, 3, 4, 8] {
            let mut split = Qwen4ExpDraftCursor::new(&head, owner);
            let mut offset = 0;
            for chunk in tokens.chunks(chunk_size) {
                let end = offset + chunk.len();
                let rows = slice(
                    &output.stream_hidden,
                    &[0, offset as i32, 0],
                    &[1, end as i32, width],
                    &[1, 1, 1],
                    None,
                );
                split.absorb(&head, chunk, &rows).unwrap();
                offset = end;
            }
            for (index, (actual, expected)) in split
                .draft_state
                .arrays()
                .iter()
                .zip(whole.draft_state.arrays())
                .enumerate()
            {
                let actual = mlx_sys::contiguous(&astype(actual, MlxDtype::Float32, None), None);
                let expected =
                    mlx_sys::contiguous(&astype(expected, MlxDtype::Float32, None), None);
                try_eval(&[&actual, &expected]).unwrap();
                let max_error = actual
                    .data_f32()
                    .iter()
                    .zip(expected.data_f32())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "draft cache chunk_size={chunk_size} array={index} max_error={max_error}"
                );
            }
            assert!(
                bytes(&split.draft_state) == bytes(&whole.draft_state),
                "draft byte mismatch at chunk_size={chunk_size}"
            );
            let before = bytes(&split.draft_state);
            let hidden = split.stream_hidden.as_ref().unwrap().clone();
            assert!(split.absorb(&head, &[u32::MAX], &hidden).is_err());
            assert_eq!(bytes(&split.draft_state), before);
            assert_eq!((split.proposed, split.accepted), (0, 0));
            assert!(
                split
                    .step(&trunk, &head, &output.state, owner, 0, 0, &[])
                    .is_err()
            );
            assert_eq!(bytes(&split.draft_state), before);
            let expected = head_forward(
                &head,
                whole.stream_hidden.as_ref().unwrap(),
                &[1],
                &whole.draft_state,
                whole.owner,
            )
            .unwrap();
            let actual = head_forward(
                &head,
                split.stream_hidden.as_ref().unwrap(),
                &[1],
                &split.draft_state,
                split.owner,
            )
            .unwrap();
            assert_eq!(bytes(&actual.state), bytes(&expected.state));
            assert_eq!(next_token(&actual).unwrap(), next_token(&expected).unwrap());
        }
    }

    #[test]
    #[ignore = "requires synthetic Flash Next MTP artifacts"]
    fn flash_next_mtp_verify_one_counts_trunk_forwards() {
        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
        let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
        let trunk = crate::weights::qwen4_exp::load(&root, &manifest).unwrap();
        let owner = 1901;
        let tokens = [1_u32, 2, 3, 4];
        let prefill = qwen4_exp::forward(
            &trunk,
            &tokens,
            &Qwen4ExpState::new(&trunk, owner),
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let primary = next_token(&prefill).unwrap();
        let target = qwen4_exp::forward(
            &trunk,
            &[primary],
            &prefill.state,
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .unwrap();
        let correct_draft = next_token(&target).unwrap();
        let vocabulary = trunk.token_embedding.weight.shape()[0] as u32;
        assert!(vocabulary > 1);
        let wrong_draft = (correct_draft + 1) % vocabulary;

        mtp_parity::take_trunk_forward_count();
        let accepted = verify_one(
            &trunk,
            &prefill.state,
            owner,
            primary,
            correct_draft,
            2,
            &[],
        )
        .unwrap();
        assert!(accepted.accepted);
        assert_eq!(accepted.bonus_wall_us, 0);
        assert_eq!(accepted.rejection_wall_us, 0);
        assert_eq!(accepted.verify_wall_us(), accepted.correction_wall_us);
        assert_eq!(mtp_parity::take_trunk_forward_count(), 1);

        let rejected =
            verify_one(&trunk, &prefill.state, owner, primary, wrong_draft, 2, &[]).unwrap();
        assert!(!rejected.accepted);
        assert_eq!(rejected.bonus_wall_us, 0);
        assert_eq!(
            rejected.verify_wall_us(),
            sum_verify_wall_us(
                rejected.correction_wall_us,
                rejected.bonus_wall_us,
                rejected.rejection_wall_us
            )
        );
        assert_eq!(mtp_parity::take_trunk_forward_count(), 2);

        let last = verify_one(
            &trunk,
            &prefill.state,
            owner,
            primary,
            correct_draft,
            1,
            &[],
        )
        .unwrap();
        assert!(!last.accepted);
        assert_eq!(last.bonus_wall_us, 0);
        assert_eq!(last.rejection_wall_us, 0);
        assert_eq!(mtp_parity::take_trunk_forward_count(), 1);
    }
}
