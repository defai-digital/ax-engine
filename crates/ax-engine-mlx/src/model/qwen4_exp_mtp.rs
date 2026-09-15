//! Experimental Flash Next draft graph with authoritative primary verification.
//!
//! The sidecar has no published official forward oracle. Its candidate input
//! combiner uses a shared hidden projection per residual stream and adds a
//! separately projected token embedding. This hypothesis is never authority
//! for committed tokens; the primary singleton graph verifies every proposal.

use std::time::Instant;

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

fn next_token(output: &Qwen4ExpOutput) -> Result<u32, String> {
    let shape = output.logits.shape();
    let last = slice(
        &output.logits,
        &[shape[0] - 1, 0],
        &[shape[0], shape[1]],
        &[1, 1],
        None,
    );
    let token = argmax(&last, None);
    try_eval(&[&token])?;
    Ok(token.data_u32()[0])
}

pub(crate) struct VerifiedStep {
    pub committed: Vec<u32>,
    pub accepted: bool,
    pub after_primary: Qwen4ExpOutput,
    pub after_draft: Option<Qwen4ExpOutput>,
    pub next_primary: u32,
}

/// `primary` must be the authoritative token from this checkpoint's logits.
/// A mismatched or truncated draft is never evaluated as a committed token.
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
    let after_primary = qwen4_exp::forward(
        trunk,
        &[primary],
        state,
        owner,
        ProjectionBatchPolicy::Shared,
    )?;
    let correction = next_token(&after_primary)?;
    let accepted = remaining > 1
        && !terminal_ids.contains(&primary)
        && !terminal_ids.contains(&correction)
        && draft == correction;
    let mut committed = vec![primary];
    let after_draft = if accepted {
        let output = qwen4_exp::forward(
            trunk,
            &[draft],
            &after_primary.state,
            owner,
            ProjectionBatchPolicy::Shared,
        )?;
        committed.push(draft);
        Some(output)
    } else {
        None
    };
    let next_primary = if let Some(output) = &after_draft {
        next_token(output)?
    } else {
        correction
    };
    Ok(VerifiedStep {
        committed,
        accepted,
        after_primary,
        after_draft,
        next_primary,
    })
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
    pub verify_wall_us: u32,
}

struct AdvancedStep {
    trunk_state: Qwen4ExpState,
    consumed: Vec<u32>,
    next_primary: u32,
    accepted: bool,
    draft_wall_us: u32,
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
            let output = qwen4_exp::forward(
                trunk,
                &[primary],
                state,
                trunk_owner,
                ProjectionBatchPolicy::Shared,
            )?;
            let next_primary = next_token(&output)?;
            let verify_wall_us = elapsed_us(started);
            self.draft_state = draft_state;
            self.stream_hidden = Some(output.stream_hidden);
            return Ok(AdvancedStep {
                trunk_state: output.state,
                consumed: vec![primary],
                next_primary,
                accepted: false,
                draft_wall_us,
                verify_wall_us,
            });
        }
        let draft_started = Instant::now();
        let proposed = head_forward(head, hidden, &[primary], &self.draft_state, self.owner)?;
        let draft = next_token(&proposed)?;
        let mut draft_wall_us = elapsed_us(draft_started);
        let verify_started = Instant::now();
        let verified = verify_one(
            trunk,
            state,
            trunk_owner,
            primary,
            draft,
            remaining,
            terminal_ids,
        )?;
        let verify_wall_us = elapsed_us(verify_started);
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
        self.accepted += usize::from(verified.accepted);
        Ok(AdvancedStep {
            trunk_state: final_output.state,
            consumed: verified.committed,
            next_primary: verified.next_primary,
            accepted: verified.accepted,
            draft_wall_us,
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
}
