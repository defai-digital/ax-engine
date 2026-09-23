//! Experimental Flash Next draft graph with authoritative primary verification.
//!
//! Admitted MXFP4 trunks verify and retain state with ordinary singleton
//! target transitions. Pure-affine trunks retain the legacy batched verifier.
//! Neither schedule grants qualification: target token and state identity,
//! including exact ties, require independent direct-trajectory evidence.
//!
//! The sidecar has no published official forward oracle. Its candidate input
//! combiner uses a shared hidden projection per residual stream and adds a
//! separately projected token embedding. This hypothesis is never authority
//! for committed tokens; the primary graph verifies every proposal.

use std::time::Instant;

#[cfg(test)]
use std::cell::{Cell, RefCell};

use super::qwen4_exp::{self, Qwen4ExpOutput, Qwen4ExpState};
use super::shared::utils::{ProjectionBatchPolicy, qw_with_policy};
use crate::weights::qwen4_exp::{Qwen4ExpAttentionBranch, Qwen4ExpTargetSchedule, Qwen4ExpWeights};
use crate::weights::qwen4_exp_mtp::Qwen4ExpMtpWeights;
use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, astype, concatenate, multiply, reshape, rms_norm, slice, topk,
    try_eval,
};

#[cfg(test)]
#[path = "qwen4_exp_mtp_canonical_tests.rs"]
mod canonical_tests;

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
    static TRUNK_FORWARD_TOKENS: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
    static FAIL_TARGET_CALL: Cell<Option<usize>> = const { Cell::new(None) };
    static FAIL_ACCEPTED_CATCHUP: Cell<bool> = const { Cell::new(false) };
}

fn trunk_forward(
    trunk: &Qwen4ExpWeights,
    tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
) -> Result<Qwen4ExpOutput, String> {
    #[cfg(test)]
    {
        TRUNK_FORWARD_COUNT.with(|count| count.set(count.get().saturating_add(1)));
        TRUNK_FORWARD_TOKENS.with(|calls| calls.borrow_mut().push(tokens.to_vec()));
    }
    let output = match &trunk.target_schedule {
        Qwen4ExpTargetSchedule::CanonicalSingleton => {
            if tokens.len() != 1 {
                return Err("Flash Next canonical verification requires a singleton".into());
            }
            qwen4_exp::forward(trunk, tokens, state, owner, ProjectionBatchPolicy::Shared)
        }
        Qwen4ExpTargetSchedule::LegacyBatched => qwen4_exp::forward_with_verifier_policy(
            trunk,
            tokens,
            state,
            owner,
            ProjectionBatchPolicy::Shared,
            ProjectionBatchPolicy::RowExact,
        ),
        Qwen4ExpTargetSchedule::Unavailable(reason) => {
            return Err(format!(
                "Flash Next MTP target schedule unavailable: {reason}"
            ));
        }
    }?;
    #[cfg(test)]
    if FAIL_TARGET_CALL.with(|fail| fail.get())
        == Some(TRUNK_FORWARD_TOKENS.with(|calls| calls.borrow().len()))
    {
        return Err("injected failure after materialized target transition".into());
    }
    Ok(output)
}

#[cfg(test)]
pub(crate) fn target_schedule_name(trunk: &Qwen4ExpWeights) -> &'static str {
    match trunk.target_schedule {
        Qwen4ExpTargetSchedule::CanonicalSingleton => "canonical_singleton",
        Qwen4ExpTargetSchedule::LegacyBatched => "legacy_batched",
        Qwen4ExpTargetSchedule::Unavailable(_) => "unavailable",
    }
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

/// Top-1 minus top-2 of one logits row, in the row's native logit units.
pub(crate) fn top_two_margin(logits: &MlxArray, row: i32) -> Result<f32, String> {
    let shape = logits.shape();
    if shape.len() != 2 || row < 0 || row >= shape[0] {
        return Err("Flash Next MTP logits row is missing".into());
    }
    if shape[1] < 2 {
        return Ok(0.0);
    }
    let row_logits = slice(logits, &[row, 0], &[row + 1, shape[1]], &[1, 1], None);
    let top = astype(&topk(&row_logits, 2, None), MlxDtype::Float32, None);
    try_eval(&[&top])?;
    let values = top.data_f32();
    if values.len() < 2 {
        return Ok(0.0);
    }
    let first = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let second = values.iter().copied().fold(f32::INFINITY, f32::min);
    if !first.is_finite() || !second.is_finite() {
        return Err("Flash Next MTP top-two logits are not finite".into());
    }
    Ok(first - second)
}

/// Integer milli-logits for route decisions. Negative and non-finite margins
/// saturate to 0; values above `u32::MAX` saturate to `u32::MAX`.
pub(crate) fn correction_margin_milli(margin: f32) -> u32 {
    if margin.is_nan() || margin <= 0.0 {
        return 0;
    }
    let milli = margin * 1000.0;
    if !milli.is_finite() || milli >= u32::MAX as f32 {
        u32::MAX
    } else {
        milli as u32
    }
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
    #[cfg(test)]
    pub target_schedule: &'static str,
    #[cfg(test)]
    pub verification_logits: Option<MlxArray>,
    pub committed: Vec<u32>,
    pub accepted: bool,
    /// Row-0 (post-`primary`) logits and hidden, always. On the canonical
    /// schedule and on both rejection paths, `.state` is genuinely the
    /// checkpoint after committing only `primary`, matching the row-0 view.
    /// On the **legacy-accepted** path (`output_row(&batched, 0)`, below)
    /// `.state` is the batched `[primary, draft]` state instead, one token
    /// ahead of what this field's logits/hidden represent -- callers must
    /// resume from `.stream_hidden`/`.hidden` (as `advance` already does),
    /// never from `.state`, when a legacy-accepted step is involved, or the
    /// cache advances one token too far. See `after_draft.state` for the
    /// correct post-draft checkpoint on that path.
    pub after_primary: Qwen4ExpOutput,
    pub after_draft: Option<Qwen4ExpOutput>,
    pub next_primary: u32,
    pub correction_wall_us: u32,
    pub bonus_wall_us: u32,
    pub rejection_wall_us: u32,
    /// Top-two margin of the correction decision, including on rejection.
    pub correction_margin: f32,
    /// Top-two margin of the bonus logits row on acceptance; 0 otherwise.
    pub bonus_margin: f32,
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

#[cfg(test)]
#[derive(Clone, Copy)]
struct VerifyDiagnosticInput {
    position: usize,
    primary: u32,
    draft: u32,
    remaining: usize,
}

#[cfg(test)]
fn verify_diagnostic(
    input: VerifyDiagnosticInput,
    step: &VerifiedStep,
    batched_row0_token: Option<u32>,
) -> Result<serde_json::Value, String> {
    let batched = batched_row0_token
        .map(|token| serde_json::json!({"token": token, "margin": step.correction_margin}));
    let canonical = step.target_schedule == "canonical_singleton";
    let singleton = if canonical {
        Some(serde_json::json!({
            "token": next_token(&step.after_primary)?,
            "margin": step.correction_margin,
            "source": "canonical_primary",
        }))
    } else if step.accepted {
        None
    } else {
        Some(serde_json::json!({
            "token": step.next_primary,
            "margin": top_two_margin(&step.after_primary.logits, 0)?,
            "source": if batched_row0_token.is_some() { "rejection_replay" } else { "one_slot_budget" },
        }))
    };
    Ok(serde_json::json!({
        "schema": "ax-engine.flash-next.verify-diagnostic.v1",
        "qualification": false,
        "performance_claim": false,
        "position": input.position,
        "primary": input.primary,
        "draft": input.draft,
        "remaining": input.remaining,
        "target_schedule": step.target_schedule,
        "accepted": step.accepted,
        "committed_len": step.committed.len(),
        "batched_row0": batched,
        "singleton": singleton,
        "batched_singleton_argmax_equal": batched_row0_token
            .filter(|_| !step.accepted).map(|token| token == step.next_primary),
    }))
}

#[cfg(test)]
fn emit_verify_diagnostic(
    input: VerifyDiagnosticInput,
    step: &VerifiedStep,
    batched_row0_token: Option<u32>,
) -> Result<(), String> {
    if std::env::var("AX_FLASH_NEXT_VERIFY_DIAGNOSTICS").as_deref() != Ok("1") {
        return Ok(());
    }
    let event = verify_diagnostic(input, step, batched_row0_token)?;
    eprintln!("FLASH_NEXT_VERIFY_DIAGNOSTIC {event}");
    Ok(())
}

/// `primary` must be the authoritative token from this checkpoint's logits.
/// A mismatched or truncated draft is never evaluated as a committed token.
///
/// Canonical verification decides and commits ordinary singletons. Legacy
/// verification retains its length-2 batch, followed by replay on rejection.
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
    #[cfg(test)]
    let diagnostic_input = VerifyDiagnosticInput {
        position: state.position(),
        primary,
        draft,
        remaining,
    };
    if matches!(
        trunk.target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    ) {
        let started = Instant::now();
        let after_primary = trunk_forward(trunk, &[primary], state, owner)?;
        let correction = next_token(&after_primary)?;
        let correction_margin = top_two_margin(&after_primary.logits, 0)?;
        let correction_wall_us = elapsed_us(started);
        let accepted = remaining > 1
            && !terminal_ids.contains(&primary)
            && !terminal_ids.contains(&correction)
            && draft == correction;
        let (after_draft, next_primary, bonus_margin, bonus_wall_us) = if accepted {
            let started = Instant::now();
            let output = trunk_forward(trunk, &[draft], &after_primary.state, owner)?;
            let bonus = next_token(&output)?;
            let margin = top_two_margin(&output.logits, 0)?;
            (Some(output), bonus, margin, elapsed_us(started))
        } else {
            (None, correction, 0.0, 0)
        };
        let verified = VerifiedStep {
            #[cfg(test)]
            target_schedule: target_schedule_name(trunk),
            #[cfg(test)]
            verification_logits: None,
            committed: if accepted {
                vec![primary, draft]
            } else {
                vec![primary]
            },
            accepted,
            after_primary,
            after_draft,
            next_primary,
            correction_wall_us,
            bonus_wall_us,
            rejection_wall_us: 0,
            correction_margin,
            bonus_margin,
        };
        #[cfg(test)]
        emit_verify_diagnostic(diagnostic_input, &verified, None)?;
        return Ok(verified);
    }
    if remaining == 1 {
        let started = Instant::now();
        let after_primary = trunk_forward(trunk, &[primary], state, owner)?;
        let correction_wall_us = elapsed_us(started);
        let correction = next_token(&after_primary)?;
        let correction_margin = top_two_margin(&after_primary.logits, 0)?;
        let verified = VerifiedStep {
            #[cfg(test)]
            target_schedule: target_schedule_name(trunk),
            #[cfg(test)]
            verification_logits: None,
            committed: vec![primary],
            accepted: false,
            after_primary,
            after_draft: None,
            next_primary: correction,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us: 0,
            correction_margin,
            bonus_margin: 0.0,
        };
        #[cfg(test)]
        emit_verify_diagnostic(diagnostic_input, &verified, None)?;
        return Ok(verified);
    }
    let started = Instant::now();
    let batched = trunk_forward(trunk, &[primary, draft], state, owner)?;
    let correction_wall_us = elapsed_us(started);
    if batched.logits.shape().first().copied() != Some(2) {
        return Err("Flash Next MTP batched verify requires two logit rows".into());
    }
    let correction = token_at_row(&batched.logits, 0)?;
    let correction_margin = top_two_margin(&batched.logits, 0)?;
    let accepted = !terminal_ids.contains(&primary)
        && !terminal_ids.contains(&correction)
        && draft == correction;
    if accepted {
        let bonus = token_at_row(&batched.logits, 1)?;
        let bonus_margin = top_two_margin(&batched.logits, 1)?;
        let verified = VerifiedStep {
            #[cfg(test)]
            target_schedule: target_schedule_name(trunk),
            #[cfg(test)]
            verification_logits: Some(batched.logits.clone()),
            committed: vec![primary, draft],
            accepted: true,
            // Row-0 view of the batched output: correct logits/hidden, but
            // `.state` carries the batched [primary, draft] state (one
            // token ahead of "post-primary") -- see the field doc on
            // VerifiedStep::after_primary. Never resume from this `.state`.
            after_primary: output_row(&batched, 0)?,
            after_draft: Some(output_row(&batched, 1)?),
            next_primary: bonus,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us: 0,
            correction_margin,
            bonus_margin,
        };
        #[cfg(test)]
        emit_verify_diagnostic(diagnostic_input, &verified, Some(correction))?;
        Ok(verified)
    } else {
        let started = Instant::now();
        let after_primary = trunk_forward(trunk, &[primary], state, owner)?;
        let rejection_wall_us = elapsed_us(started);
        let next_primary = next_token(&after_primary)?;
        let verified = VerifiedStep {
            #[cfg(test)]
            target_schedule: target_schedule_name(trunk),
            #[cfg(test)]
            verification_logits: Some(batched.logits.clone()),
            committed: vec![primary],
            accepted: false,
            after_primary,
            after_draft: None,
            next_primary,
            correction_wall_us,
            bonus_wall_us: 0,
            rejection_wall_us,
            correction_margin,
            bonus_margin: 0.0,
        };
        #[cfg(test)]
        emit_verify_diagnostic(diagnostic_input, &verified, Some(correction))?;
        Ok(verified)
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
    pub correction_margin: f32,
    #[allow(dead_code)]
    pub bonus_margin: f32,
}

struct AdvancedStep {
    #[cfg(test)]
    observation: CandidateStepObservation,
    trunk_state: Qwen4ExpState,
    consumed: Vec<u32>,
    next_primary: u32,
    accepted: bool,
    draft_wall_us: u32,
    correction_wall_us: u32,
    bonus_wall_us: u32,
    rejection_wall_us: u32,
    verify_wall_us: u32,
    correction_margin: f32,
    bonus_margin: f32,
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

    /// Parts needed to persist this cursor alongside a prefix-cache snapshot,
    /// or `None` if the cursor is not currently aligned with `trunk` (an
    /// unaligned cursor has nothing valid to persist).
    pub(crate) fn prefix_snapshot_parts(
        &self,
        trunk: &Qwen4ExpState,
    ) -> Option<(&Qwen4ExpState, &MlxArray)> {
        self.aligned(trunk).then(|| {
            (
                &self.draft_state,
                self.stream_hidden
                    .as_ref()
                    .expect("aligned implies stream_hidden is Some"),
            )
        })
    }

    /// Reconstruct a cursor from a prefix-snapshot sidecar payload, rebinding
    /// it onto `head`'s graph under the current request's trunk owner (the
    /// payload's own `owner` field is discarded and replaced: the trunk owner
    /// changes per load/session, so a raw stored owner would be stale).
    /// Returns `Err` on any structural problem (bad payload, shape/dtype
    /// mismatch against `head`, or a decoded state that ends up misaligned
    /// with `trunk`); every failure path means "no cursor", never a partial
    /// or best-effort cursor.
    pub(crate) fn from_prefix_snapshot(
        head: &Qwen4ExpMtpWeights,
        trunk_owner: u64,
        trunk: &Qwen4ExpState,
        bytes: &[u8],
    ) -> Result<Self, String> {
        let (mut draft_state, stream_hidden) =
            crate::kv_cache::MlxKVCache::try_deserialize_qwen4_exp_draft_cursor(bytes)
                .map_err(|error| format!("MTP draft-cursor payload failed to decode: {error}"))?;
        let owner = trunk_owner ^ 0x5146_4e4d_5450_0001;
        // Match the trunk restore path's rigor: validate every layer against
        // the head's graph (shapes, dtypes, branch kinds) and only then adopt
        // the freshly derived owner; a payload decoded under a stale owner is
        // never adopted as-is.
        draft_state
            .rebind_for_model(&head.graph, owner)
            .map_err(|error| {
                format!("MTP draft-cursor payload does not match the head graph: {error}")
            })?;
        // Per-cursor-lifetime telemetry does not survive a persistence
        // round-trip; only correctness state does.
        let cursor = Self {
            draft_state,
            stream_hidden: Some(stream_hidden),
            owner,
            proposed: 0,
            accepted: 0,
        };
        if cursor.aligned(trunk) {
            Ok(cursor)
        } else {
            Err(format!(
                "MTP draft-cursor payload position {} is misaligned with trunk position {}",
                cursor.draft_state.position(),
                trunk.position()
            ))
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
            let correction_margin = top_two_margin(&output.logits, 0)?;
            #[cfg(test)]
            let canonical_primary = canonical_primary_observation(trunk, &output);
            self.draft_state = draft_state;
            self.stream_hidden = Some(output.stream_hidden);
            return Ok(AdvancedStep {
                #[cfg(test)]
                observation: CandidateStepObservation {
                    target_schedule: target_schedule_name(trunk),
                    draft_token: None,
                    verification_logits: None,
                    canonical_primary,
                    canonical_correction_logits: matches!(
                        trunk.target_schedule,
                        Qwen4ExpTargetSchedule::CanonicalSingleton
                    )
                    .then(|| output.logits.clone()),
                    next_logits: output.logits.clone(),
                },
                trunk_state: output.state,
                consumed: vec![primary],
                next_primary,
                accepted: false,
                draft_wall_us,
                correction_wall_us,
                bonus_wall_us: 0,
                rejection_wall_us: 0,
                verify_wall_us: correction_wall_us,
                correction_margin,
                bonus_margin: 0.0,
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
        let correction_margin = verified.correction_margin;
        let bonus_margin = verified.bonus_margin;
        #[cfg(test)]
        let canonical_correction_logits = matches!(
            trunk.target_schedule,
            Qwen4ExpTargetSchedule::CanonicalSingleton
        )
        .then(|| verified.after_primary.logits.clone());
        #[cfg(test)]
        let canonical_primary = canonical_primary_observation(trunk, &verified.after_primary);
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
            #[cfg(test)]
            if FAIL_ACCEPTED_CATCHUP.with(Cell::get) {
                return Err("injected failure after materialized accepted head catch-up".into());
            }
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
            #[cfg(test)]
            observation: CandidateStepObservation {
                target_schedule: target_schedule_name(trunk),
                draft_token: Some(draft),
                verification_logits: verified.verification_logits,
                canonical_correction_logits,
                canonical_primary,
                next_logits: final_output.logits.clone(),
            },
            trunk_state: final_output.state,
            consumed,
            next_primary,
            accepted,
            draft_wall_us,
            correction_wall_us,
            bonus_wall_us,
            rejection_wall_us,
            verify_wall_us,
            correction_margin,
            bonus_margin,
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
            correction_margin: advanced.correction_margin,
            bonus_margin: advanced.bonus_margin,
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
    pub primary_logits: MlxArray,
    trunk_owner: u64,
    draft_owner: u64,
    pub proposed: usize,
    pub accepted: usize,
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct CanonicalPrimaryObservation {
    pub state: Qwen4ExpState,
    pub stream_hidden: MlxArray,
}

#[cfg(test)]
fn canonical_primary_observation(
    trunk: &Qwen4ExpWeights,
    output: &Qwen4ExpOutput,
) -> Option<CanonicalPrimaryObservation> {
    matches!(
        trunk.target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    )
    .then(|| CanonicalPrimaryObservation {
        state: output.state.clone(),
        stream_hidden: output.stream_hidden.clone(),
    })
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct CandidateStepObservation {
    pub target_schedule: &'static str,
    pub draft_token: Option<u32>,
    /// Only legacy length-2 verifier rows; canonical transitions are separate.
    pub verification_logits: Option<MlxArray>,
    pub canonical_correction_logits: Option<MlxArray>,
    pub canonical_primary: Option<CanonicalPrimaryObservation>,
    pub next_logits: MlxArray,
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
            primary_logits: output.logits,
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
        self.step_observed(trunk, head, remaining, terminal_ids)
            .map(|(consumed, _)| consumed)
    }

    pub(crate) fn step_observed(
        &mut self,
        trunk: &Qwen4ExpWeights,
        head: &Qwen4ExpMtpWeights,
        remaining: usize,
        terminal_ids: &[u32],
    ) -> Result<(Vec<u32>, CandidateStepObservation), String> {
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
        self.primary_logits = advanced.observation.next_logits.clone();
        self.proposed = cursor.proposed;
        self.accepted = cursor.accepted;
        Ok((advanced.consumed, advanced.observation))
    }
}

#[cfg(test)]
pub(crate) mod mtp_parity {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const DEFAULT_REAL_PACK_TOLERANCE: f32 = 0.1;
    const REAL_PACK_TOLERANCE_ENV: &str = "AX_FLASH_NEXT_MTP_REAL_TOLERANCE";
    const TIE_MARGIN_ENV: &str = "AX_FLASH_NEXT_MTP_TIE_MARGIN";
    const DEFAULT_TIE_MARGIN: f32 = 0.5;
    const TOLERANCE_SOURCE_SYNTHETIC: &str = "synthetic";
    const TOLERANCE_SOURCE_REAL_PACK: &str = "real-pack-env";

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct TieDivergence {
        pub position: usize,
        pub tokens: [u32; 2],
        pub margin: f32,
    }

    impl TieDivergence {
        pub(crate) fn to_json(&self) -> serde_json::Value {
            serde_json::json!({
                "position": self.position,
                "tokens": self.tokens,
                "margin": self.margin,
            })
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct GreedyIdentityReport {
        pub greedy_identity: bool,
        pub identity_until_first_tie: bool,
        pub tie_divergences: Vec<TieDivergence>,
        pub compared_positions: usize,
    }

    impl GreedyIdentityReport {
        pub(crate) fn exact() -> Self {
            Self {
                greedy_identity: true,
                identity_until_first_tie: true,
                tie_divergences: Vec::new(),
                compared_positions: 0,
            }
        }
    }

    pub(crate) fn parse_mtp_tie_margin(raw: Option<&str>) -> f32 {
        let Some(value) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
            return DEFAULT_TIE_MARGIN;
        };
        let parsed = value.parse::<f32>();
        assert!(
            matches!(parsed, Ok(limit) if limit.is_finite() && limit >= 0.0),
            "{TIE_MARGIN_ENV} must be a finite non-negative f32, got {value:?}"
        );
        parsed.unwrap()
    }

    pub(crate) fn mtp_tie_margin() -> f32 {
        parse_mtp_tie_margin(std::env::var(TIE_MARGIN_ENV).ok().as_deref())
    }

    pub(crate) fn greedy_mismatch_is_not_a_near_tie(
        position: usize,
        direct: u32,
        mtp: u32,
        margin: f32,
        tie_margin: f32,
    ) -> String {
        format!(
            "MTP greedy mismatch at position {position} is not a documented near-tie: direct={direct} mtp={mtp} margin={margin} tie_margin={tie_margin}"
        )
    }

    /// Classify one greedy mismatch. `Ok(Some(tie))` means stop comparing.
    pub(crate) fn classify_greedy_mismatch(
        position: usize,
        direct: u32,
        mtp: u32,
        margin: f32,
        tie_margin: f32,
    ) -> Result<Option<TieDivergence>, String> {
        if direct == mtp {
            return Ok(None);
        }
        if margin <= tie_margin {
            Ok(Some(TieDivergence {
                position,
                tokens: [direct, mtp],
                margin,
            }))
        } else {
            Err(greedy_mismatch_is_not_a_near_tie(
                position, direct, mtp, margin, tie_margin,
            ))
        }
    }

    pub(crate) fn greedy_identity_until_tie(
        direct: &[u32],
        mtp: &[u32],
        margin: f32,
        tie_margin: f32,
    ) -> Result<GreedyIdentityReport, String> {
        let shared = direct.len().min(mtp.len());
        let mut report = GreedyIdentityReport::exact();
        for position in 0..shared {
            report.compared_positions = position + 1;
            if direct[position] == mtp[position] {
                continue;
            }
            report.greedy_identity = false;
            if let Some(tie) = classify_greedy_mismatch(
                position,
                direct[position],
                mtp[position],
                margin,
                tie_margin,
            )? {
                report.tie_divergences.push(tie);
                return Ok(report);
            }
        }
        if direct.len() != mtp.len() {
            report.greedy_identity = false;
            return Err(format!(
                "MTP greedy length mismatch without a documented near-tie: direct_len={} mtp_len={} margin={margin} tie_margin={tie_margin}",
                direct.len(),
                mtp.len()
            ));
        }
        Ok(report)
    }

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
        mtp_state_array_records(actual, expected)
            .into_iter()
            .map(|record| record.divergence)
            .fold(MtpDivergence::zero(), MtpDivergence::max_relative)
    }

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct MtpStateArrayRecord {
        pub index: usize,
        pub layer: usize,
        pub kind: &'static str,
        pub dtype: String,
        pub shape: Vec<i32>,
        pub divergence: MtpDivergence,
    }

    impl MtpStateArrayRecord {
        pub(crate) fn to_json(&self) -> serde_json::Value {
            serde_json::json!({
                "index": self.index,
                "layer": self.layer,
                "kind": self.kind,
                "dtype": self.dtype,
                "shape": self.shape,
                "scale": self.divergence.scale,
                "max_abs": self.divergence.max_abs,
                "relative": self.divergence.relative,
            })
        }
    }

    pub(crate) fn mtp_state_array_records(
        actual: &Qwen4ExpState,
        expected: &Qwen4ExpState,
    ) -> Vec<MtpStateArrayRecord> {
        assert_eq!(actual.position(), expected.position());
        let actual_arrays = actual.arrays();
        let expected_arrays = expected.arrays();
        let descriptors = actual.array_descriptors();
        assert_eq!(actual_arrays.len(), expected_arrays.len());
        assert_eq!(descriptors.len(), actual_arrays.len());
        assert_eq!(expected.array_descriptors(), descriptors);
        actual_arrays
            .iter()
            .zip(&expected_arrays)
            .zip(descriptors)
            .map(|((actual_array, expected_array), descriptor)| {
                assert_eq!(actual_array.shape(), expected_array.shape());
                assert_eq!(actual_array.dtype(), expected_array.dtype());
                MtpStateArrayRecord {
                    index: descriptor.index,
                    layer: descriptor.layer,
                    kind: descriptor.kind,
                    dtype: format!("{:?}", actual_array.dtype()),
                    shape: actual_array.shape(),
                    divergence: mtp_logits_divergence(actual_array, expected_array),
                }
            })
            .collect()
    }

    pub(crate) fn mtp_state_arrays_json(records: &[MtpStateArrayRecord]) -> serde_json::Value {
        serde_json::Value::Array(records.iter().map(MtpStateArrayRecord::to_json).collect())
    }

    pub(crate) fn eprint_mtp_state_array_table(label: &str, records: &[MtpStateArrayRecord]) {
        eprintln!("MTP {label} per-array state:");
        for record in records {
            eprintln!(
                "  [{index}] layer={layer} kind={kind} dtype={dtype} shape={shape:?} scale={scale} max_abs={max_abs} relative={relative}",
                index = record.index,
                layer = record.layer,
                kind = record.kind,
                dtype = record.dtype,
                shape = record.shape,
                scale = record.divergence.scale,
                max_abs = record.divergence.max_abs,
                relative = record.divergence.relative,
            );
        }
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
        let tolerance = mtp_run_tolerance(dtype);
        for record in mtp_state_array_records(actual, expected) {
            assert!(
                record.divergence.relative <= tolerance.limit,
                "MTP state array {} layer {} kind {} relative max abs {} exceeds {} (source={}, scale={}, max_abs={})",
                record.index,
                record.layer,
                record.kind,
                record.divergence.relative,
                tolerance.limit,
                tolerance.source,
                record.divergence.scale,
                record.divergence.max_abs
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
    fn mtp_tie_margin_default_and_env_parse() {
        assert_eq!(parse_mtp_tie_margin(None), 0.5);
        assert_eq!(parse_mtp_tie_margin(Some("")), 0.5);
        assert_eq!(parse_mtp_tie_margin(Some(" 0.25 ")), 0.25);
        assert_eq!(parse_mtp_tie_margin(Some("0")), 0.0);
    }

    #[test]
    fn correction_margin_milli_saturates() {
        assert_eq!(correction_margin_milli(0.5), 500);
        assert_eq!(correction_margin_milli(0.0), 0);
        assert_eq!(correction_margin_milli(-0.1), 0);
        assert_eq!(correction_margin_milli(f32::NAN), 0);
        assert_eq!(correction_margin_milli(f32::INFINITY), u32::MAX);
        assert_eq!(correction_margin_milli(1.0e12), u32::MAX);
    }

    #[test]
    fn greedy_identity_until_tie_records_near_tie_and_rejects_wide_margin() {
        let direct = [11751, 13, 271, 760];
        let mtp = [11751, 13, 561, 6511];
        let tied = greedy_identity_until_tie(&direct, &mtp, 0.0, 0.5).unwrap();
        assert!(!tied.greedy_identity);
        assert!(tied.identity_until_first_tie);
        assert_eq!(tied.tie_divergences.len(), 1);
        assert_eq!(tied.tie_divergences[0].position, 2);
        assert_eq!(tied.tie_divergences[0].tokens, [271, 561]);
        assert_eq!(tied.compared_positions, 3);
        let exact = greedy_identity_until_tie(&direct, &direct, 0.0, 0.5).unwrap();
        assert!(exact.greedy_identity);
        assert!(exact.identity_until_first_tie);
        assert!(exact.tie_divergences.is_empty());
        assert_eq!(exact.compared_positions, direct.len());
        let wide = greedy_identity_until_tie(&direct, &mtp, 1.25, 0.5).unwrap_err();
        assert_eq!(
            wide,
            greedy_mismatch_is_not_a_near_tie(2, 271, 561, 1.25, 0.5)
        );
        assert!(wide.contains("MTP greedy mismatch at position 2 is not a documented near-tie"));
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

    #[test]
    fn mtp_state_array_json_lists_kind_layer_and_divergence() {
        let record = MtpStateArrayRecord {
            index: 69,
            layer: 31,
            kind: "qsa.k",
            dtype: "Bfloat16".into(),
            shape: vec![1, 8, 2, 4],
            divergence: MtpDivergence {
                scale: 0.34330982,
                max_abs: 0.09761965,
                relative: 0.28434855,
            },
        };
        let json = record.to_json();
        assert_eq!(json["index"], 69);
        assert_eq!(json["layer"], 31);
        assert_eq!(json["kind"], "qsa.k");
        assert_eq!(json["dtype"], "Bfloat16");
        assert_eq!(json["shape"], serde_json::json!([1, 8, 2, 4]));
        assert!(json["scale"].as_f64().unwrap() > 0.0);
        assert!(json["max_abs"].as_f64().unwrap() > 0.0);
        assert!(json["relative"].as_f64().unwrap() > 0.2);
        let table = mtp_state_arrays_json(&[record]);
        assert_eq!(table.as_array().unwrap().len(), 1);
        assert_eq!(table[0]["kind"], "qsa.k");
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
        assert!(accepted.correction_margin.is_finite() && accepted.correction_margin >= 0.0);
        assert!(accepted.bonus_margin.is_finite() && accepted.bonus_margin >= 0.0);
        assert_eq!(mtp_parity::take_trunk_forward_count(), 1);

        let rejected =
            verify_one(&trunk, &prefill.state, owner, primary, wrong_draft, 2, &[]).unwrap();
        assert!(!rejected.accepted);
        assert_eq!(rejected.bonus_wall_us, 0);
        assert_eq!(rejected.bonus_margin, 0.0);
        assert!(rejected.correction_margin.is_finite() && rejected.correction_margin >= 0.0);
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
        assert_eq!(last.bonus_margin, 0.0);
        assert!(last.correction_margin.is_finite() && last.correction_margin >= 0.0);
        assert_eq!(mtp_parity::take_trunk_forward_count(), 1);

        let input = VerifyDiagnosticInput {
            position: prefill.state.position(),
            primary,
            draft: correct_draft,
            remaining: 2,
        };
        let accepted_before = state_bytes(
            &accepted.after_draft.as_ref().unwrap().state,
            trunk.layers.len(),
        );
        let accepted_event = verify_diagnostic(input, &accepted, Some(correct_draft)).unwrap();
        assert_eq!(accepted_event["batched_row0"]["token"], correct_draft);
        assert!(accepted_event["singleton"].is_null());
        assert!(accepted_event["batched_singleton_argmax_equal"].is_null());
        assert_eq!(
            state_bytes(
                &accepted.after_draft.as_ref().unwrap().state,
                trunk.layers.len()
            ),
            accepted_before
        );

        let rejected_before = state_bytes(&rejected.after_primary.state, trunk.layers.len());
        let expected_margin = top_two_margin(&rejected.after_primary.logits, 0).unwrap();
        let mut diagnostic_step = rejected;
        // Distinguish the recorded verifier margin from the replay margin.
        diagnostic_step.correction_margin = expected_margin + 7.0;
        let event = verify_diagnostic(input, &diagnostic_step, Some(wrong_draft)).unwrap();
        assert_eq!(
            event["batched_row0"]["margin"],
            serde_json::json!(expected_margin + 7.0)
        );
        assert_eq!(
            event["singleton"]["margin"],
            serde_json::json!(expected_margin)
        );
        assert_eq!(event["singleton"]["token"], diagnostic_step.next_primary);
        assert_eq!(event["singleton"]["source"], "rejection_replay");
        assert_eq!(event["batched_singleton_argmax_equal"], false);
        assert_eq!(
            state_bytes(&diagnostic_step.after_primary.state, trunk.layers.len()),
            rejected_before
        );

        let last_event = verify_diagnostic(
            VerifyDiagnosticInput {
                remaining: 1,
                ..input
            },
            &last,
            None,
        )
        .unwrap();
        assert!(last_event["batched_row0"].is_null());
        assert_eq!(last_event["singleton"]["source"], "one_slot_budget");
        assert!(last_event["batched_singleton_argmax_equal"].is_null());
        assert_eq!(mtp_parity::take_trunk_forward_count(), 0);

        let terminal = verify_one(
            &trunk,
            &prefill.state,
            owner,
            primary,
            correct_draft,
            2,
            &[correct_draft],
        )
        .unwrap();
        assert!(!terminal.accepted);
        assert_eq!(terminal.committed, vec![primary]);
        assert_eq!(terminal.next_primary, correct_draft);
        assert_eq!(mtp_parity::take_trunk_forward_count(), 2);

        use sha2::{Digest, Sha256};
        let controls: Vec<_> = [&accepted, &diagnostic_step, &last, &terminal]
            .into_iter()
            .map(|step| {
                let state = step.after_draft.as_ref().unwrap_or(&step.after_primary);
                serde_json::json!({
                    "committed": step.committed,
                    "next_primary": step.next_primary,
                    "state_sha256": format!("{:x}", Sha256::digest(state_bytes(&state.state, trunk.layers.len()))),
                })
            })
            .collect();
        eprintln!(
            "FLASH_NEXT_VERIFY_CONTROL {}",
            serde_json::json!({
                "synthetic_fixture": true, "qualification": false, "controls": controls,
            })
        );
    }

    /// Bit values of an array widened to Float32, for bit-exact comparisons.
    fn bits(array: &MlxArray) -> Vec<u32> {
        let widened = mlx_sys::contiguous(&astype(array, MlxDtype::Float32, None), None);
        try_eval(&[&widened]).unwrap();
        assert!(widened.data_f32().iter().all(|value| value.is_finite()));
        widened.data_f32().iter().map(|v| v.to_bits()).collect()
    }

    /// Load the synthetic Flash Next MTP fixture shared by the artifact-backed
    /// tests in this module.
    fn flash_next_mtp_fixture() -> (super::super::ModelConfig, crate::weights::ModelWeights) {
        let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
        let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
        manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
        manifest.runtime_status = ax_engine_core::NativeRuntimeStatus::default();
        let artifacts =
            ax_engine_core::NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap();
        let cfg = super::super::ModelConfig::from_manifest(artifacts.manifest());
        let mut weights = crate::weights::load_weights(&artifacts).unwrap();
        weights.qwen4_exp_mtp = Some(Box::new(
            crate::weights::qwen4_exp_mtp::load(
                artifacts.root_dir(),
                artifacts.manifest(),
                weights.qwen4_exp.as_ref().unwrap(),
            )
            .unwrap(),
        ));
        (cfg, weights)
    }

    /// Cold-prefill `tokens` through the Flash Next MTP helper, returning the
    /// cache, the greedy first token, and the live aligned cursor.
    fn flash_next_mtp_cold_prefill(
        cfg: &super::super::ModelConfig,
        weights: &crate::weights::ModelWeights,
        tokens: &[u32],
    ) -> (MlxKVCache, Option<u32>, Qwen4ExpDraftCursor) {
        use crate::generate::chunked_prefill_flash_next_mtp;
        let head = weights.qwen4_exp_mtp.as_deref().unwrap();
        let mut cache = MlxKVCache::new_contiguous(cfg.layer_count);
        let mut cursor = Some(Qwen4ExpDraftCursor::new(head, cfg.compile_cache_identity));
        let first = chunked_prefill_flash_next_mtp(
            cfg,
            weights,
            tokens,
            &mut cache,
            tokens.len(),
            true,
            &mut cursor,
        );
        (cache, first, cursor.unwrap())
    }

    #[test]
    #[ignore = "requires synthetic Flash Next MTP artifacts and Metal"]
    fn flash_next_prefix_snapshot_restored_cursor_matches_live_bit_for_bit() {
        let (cfg, weights) = flash_next_mtp_fixture();
        let head = weights.qwen4_exp_mtp.as_deref().unwrap();
        let trunk = weights.qwen4_exp.as_deref().unwrap();
        let owner = cfg.compile_cache_identity;

        // 1-2. A cold prefill produces a live, aligned cursor A and the trunk
        //      at the same position.
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (cache, first, mut cursor_a) = flash_next_mtp_cold_prefill(&cfg, &weights, &prompt);
        let first = first.expect("completing prefill returns the greedy first token");
        let trunk_state = cache.qwen4_exp.clone().expect("Flash Next trunk state");
        assert!(cursor_a.aligned(&trunk_state));

        // 3. Serialize cursor A as a prefix-snapshot sidecar payload and
        //    restore it as cursor B under the SAME trunk owner (same-process
        //    reuse, the simplest case).
        let (draft_state, stream_hidden) = cursor_a
            .prefix_snapshot_parts(&trunk_state)
            .expect("an aligned cursor exposes snapshot parts");
        let payload = MlxKVCache::serialize_qwen4_exp_draft_cursor(draft_state, stream_hidden);
        let mut cursor_b =
            Qwen4ExpDraftCursor::from_prefix_snapshot(head, owner, &trunk_state, &payload)
                .expect("a payload from an aligned cursor must restore");
        assert!(cursor_b.aligned(&trunk_state));
        assert_eq!(
            (cursor_b.proposed, cursor_b.accepted),
            (0, 0),
            "per-cursor telemetry must not survive the round-trip"
        );

        // 4. Drive one identical verify cycle on both cursors and require
        //    bit-identical results: every tensor operation on this path is
        //    bit-exact, so any divergence would be a restore bug, not noise.
        let budget = 4;
        let step_a = cursor_a
            .step(trunk, head, &trunk_state, owner, first, budget, &[])
            .unwrap();
        let step_b = cursor_b
            .step(trunk, head, &trunk_state, owner, first, budget, &[])
            .unwrap();
        assert_eq!(
            step_a.emitted, step_b.emitted,
            "proposals must be identical"
        );
        assert_eq!(step_a.accepted, step_b.accepted);
        assert_eq!(step_a.committed_len, step_b.committed_len);
        assert_eq!(
            step_a.correction_margin, step_b.correction_margin,
            "correction margins must be bit-identical"
        );
        assert_eq!(
            step_a.bonus_margin, step_b.bonus_margin,
            "bonus margins must be bit-identical"
        );
        assert_eq!(
            state_bytes(&step_a.trunk_state, trunk.layers.len()),
            state_bytes(&step_b.trunk_state, trunk.layers.len()),
            "committed trunk state must be byte-identical"
        );
        assert_eq!(
            bytes(&cursor_a.draft_state),
            bytes(&cursor_b.draft_state),
            "draft state must be byte-identical after the step"
        );
        assert_eq!(
            bits(cursor_a.stream_hidden.as_ref().unwrap()),
            bits(cursor_b.stream_hidden.as_ref().unwrap()),
            "the stream row must be bit-identical after the step"
        );
        assert_eq!(
            (cursor_a.proposed, cursor_a.accepted),
            (cursor_b.proposed, cursor_b.accepted),
            "post-step telemetry must agree"
        );
    }

    #[test]
    #[ignore = "requires synthetic Flash Next MTP artifacts and Metal"]
    fn flash_next_prefix_snapshot_restore_fails_closed() {
        let (cfg, mut weights) = flash_next_mtp_fixture();
        let owner = cfg.compile_cache_identity;

        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (cache, _first, cursor_a) = flash_next_mtp_cold_prefill(&cfg, &weights, &prompt);
        let trunk_state = cache.qwen4_exp.clone().expect("Flash Next trunk state");
        let (draft_state, stream_hidden) = cursor_a
            .prefix_snapshot_parts(&trunk_state)
            .expect("an aligned cursor exposes snapshot parts");
        let payload = MlxKVCache::serialize_qwen4_exp_draft_cursor(draft_state, stream_hidden);

        // A payload whose draft position does not match the given trunk must
        // return Err: never a panic, never a silently misaligned cursor.
        let (mut short_cache, _short_first, _short_cursor) =
            flash_next_mtp_cold_prefill(&cfg, &weights, &prompt[..4]);
        let short_trunk = short_cache.qwen4_exp.take().expect("short trunk state");
        assert_ne!(short_trunk.position(), trunk_state.position());
        let trunk_payload = cache.serialize_to_bytes();

        // The head is borrowed mutably from here on (dtype flip + restore),
        // so every immutable use of the weights above had to come first.
        let head = weights.qwen4_exp_mtp.as_mut().unwrap().as_mut();

        // A payload from a DIFFERENT head graph must be rejected, not
        // adopted. Flip the dtype the rebind validation derives from the head
        // (the same mechanism a real dtype-mismatched pack would trip) and
        // put it back.
        let original_embedding = head.graph.token_embedding.weight.clone();
        let foreign_dtype = match original_embedding.dtype() {
            MlxDtype::Float32 => MlxDtype::Bfloat16,
            _ => MlxDtype::Float32,
        };
        head.graph.token_embedding.weight =
            mlx_sys::contiguous(&astype(&original_embedding, foreign_dtype, None), None);
        let foreign =
            Qwen4ExpDraftCursor::from_prefix_snapshot(head, owner, &trunk_state, &payload);
        head.graph.token_embedding.weight = original_embedding;
        let error = foreign
            .err()
            .expect("a foreign head graph must fail closed");
        assert!(!error.is_empty());

        let mismatched =
            Qwen4ExpDraftCursor::from_prefix_snapshot(head, owner, &short_trunk, &payload);
        assert!(
            mismatched.is_err(),
            "a position-mismatched payload must fail closed"
        );

        // A truncated payload must fail closed at decode, not best-effort.
        let truncated = Qwen4ExpDraftCursor::from_prefix_snapshot(
            head,
            owner,
            &trunk_state,
            &payload[..payload.len() / 2],
        );
        assert!(truncated.is_err(), "a truncated payload must fail closed");

        // A trunk blob (AXKB magic) must never decode as a cursor payload.
        let as_trunk = Qwen4ExpDraftCursor::from_prefix_snapshot(
            head,
            owner,
            &trunk_state,
            trunk_payload.as_slice(),
        );
        assert!(as_trunk.is_err(), "a trunk blob is not a cursor payload");
    }
}

/// Test-only trained-head oracle: permute draft `lm_head` rows in memory.
#[cfg(test)]
pub(crate) mod trained_head {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::weights::QuantizedWeight;
    use mlx_sys::{eval, take};

    pub(crate) const PERMUTE_HEAD_ENV: &str = "AX_FLASH_NEXT_MTP_PERMUTE_HEAD";
    pub(crate) const PERMUTE_HEAD_SEED: u64 = 20260916;

    pub(crate) fn permute_head_opt_in(raw: Option<&str>) -> bool {
        raw.map(str::trim).is_some_and(|value| {
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
        })
    }

    pub(crate) fn permute_head_requested() -> bool {
        permute_head_opt_in(std::env::var(PERMUTE_HEAD_ENV).ok().as_deref())
    }

    /// SplitMix-style Fisher–Yates. Deterministic for a fixed seed.
    pub(crate) fn row_permutation(rows: usize, seed: u64) -> Vec<u32> {
        assert!(rows > 1, "draft output projection needs at least two rows");
        let mut order: Vec<u32> = (0..u32::try_from(rows).expect("vocab fits u32")).collect();
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        for i in (1..rows).rev() {
            state = state
                .wrapping_mul(0xD1B5_4A32_D192_ED03)
                .wrapping_add(0x9E37_79B9_7F4A_7C15);
            let j = (state as usize) % (i + 1);
            order.swap(i, j);
        }
        order
    }

    pub(crate) fn agreement_rate(agreed: &[bool]) -> f64 {
        if agreed.is_empty() {
            0.0
        } else {
            agreed.iter().filter(|agreed| **agreed).count() as f64 / agreed.len() as f64
        }
    }

    fn u32_vector(values: &[u32]) -> MlxArray {
        let bytes = std::mem::size_of_val(values);
        MlxArray::from_raw_data(
            values.as_ptr().cast(),
            bytes,
            &[i32::try_from(values.len()).expect("permutation length fits i32")],
            MlxDtype::Uint32,
        )
    }

    fn take_output_rows(array: &MlxArray, indices: &MlxArray) -> MlxArray {
        assert!(
            !array.shape().is_empty() && array.shape()[0] == indices.shape()[0],
            "permuted tensor must share the output-row axis"
        );
        take(array, indices, 0, None)
    }

    pub(crate) fn permute_quantized_output_rows(
        weight: &QuantizedWeight,
        seed: u64,
    ) -> QuantizedWeight {
        let rows = usize::try_from(weight.weight.shape()[0]).expect("output rows fit usize");
        let indices = u32_vector(&row_permutation(rows, seed));
        let mut out = weight.clone();
        out.weight = take_output_rows(&weight.weight, &indices);
        if let Some(scales) = &weight.scales {
            out.scales = Some(take_output_rows(scales, &indices));
        }
        if let Some(biases) = &weight.biases {
            out.biases = Some(take_output_rows(biases, &indices));
        }
        if let Some(bias) = &weight.linear_bias {
            out.linear_bias = Some(take_output_rows(bias, &indices));
        }
        // Decode caches still index the unpermuted vocab axis.
        out.decode_weight_t = None;
        out.decode_q2_weight = None;
        out.decode_q2_scales = None;
        out.decode_q2_biases = None;
        let mut live = vec![&out.weight];
        if let Some(scales) = &out.scales {
            live.push(scales);
        }
        if let Some(biases) = &out.biases {
            live.push(biases);
        }
        if let Some(bias) = &out.linear_bias {
            live.push(bias);
        }
        eval(&live);
        out
    }

    /// Shuffle the draft graph's output projection in memory. Trunk `lm_head`
    /// stays untouched because the sidecar originally shared that handle.
    pub(crate) fn permute_draft_output_projection(head: &mut Qwen4ExpMtpWeights, seed: u64) {
        head.graph.lm_head = permute_quantized_output_rows(&head.graph.lm_head, seed);
    }

    #[test]
    fn flash_next_mtp_permute_head_opt_in_is_strictly_truthy() {
        for enabled in ["1", "true", "TRUE", "yes", " Yes "] {
            assert!(permute_head_opt_in(Some(enabled)), "{enabled:?}");
        }
        for disabled in ["", "0", "false", "no", "permute", "2"] {
            assert!(!permute_head_opt_in(Some(disabled)), "{disabled:?}");
        }
        assert!(!permute_head_opt_in(None));
    }

    #[test]
    fn flash_next_mtp_permute_head_rows_is_deterministic_non_identity_permutation() {
        let first = row_permutation(32, PERMUTE_HEAD_SEED);
        let again = row_permutation(32, PERMUTE_HEAD_SEED);
        assert_eq!(first, again);
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..32).collect::<Vec<u32>>());
        assert_ne!(first, (0..32).collect::<Vec<u32>>());
        assert_ne!(row_permutation(32, PERMUTE_HEAD_SEED ^ 1), first);
        assert_eq!(agreement_rate(&[]), 0.0);
        assert_eq!(agreement_rate(&[true, true, false, true]), 0.75);
    }
}
