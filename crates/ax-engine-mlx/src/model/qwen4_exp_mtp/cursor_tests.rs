
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
    let (root, manifest, paging) = if let Some(root) = std::env::var_os("AX_FLASH_NEXT_REAL_PACK") {
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
        crate::weights::qwen4_exp::load_with_paging_policy(&root, &manifest, paging, 1).unwrap();
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
            let expected = mlx_sys::contiguous(&astype(expected, MlxDtype::Float32, None), None);
            try_eval(&[&actual, &expected]).unwrap();
            let max_error = actual
                .data_f32()
                .iter()
                .zip(expected.data_f32())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!("draft cache chunk_size={chunk_size} array={index} max_error={max_error}");
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

    let rejected = verify_one(&trunk, &prefill.state, owner, primary, wrong_draft, 2, &[]).unwrap();
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
    let foreign = Qwen4ExpDraftCursor::from_prefix_snapshot(head, owner, &trunk_state, &payload);
    head.graph.token_embedding.weight = original_embedding;
    let error = foreign
        .err()
        .expect("a foreign head graph must fail closed");
    assert!(!error.is_empty());

    let mismatched = Qwen4ExpDraftCursor::from_prefix_snapshot(head, owner, &short_trunk, &payload);
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
