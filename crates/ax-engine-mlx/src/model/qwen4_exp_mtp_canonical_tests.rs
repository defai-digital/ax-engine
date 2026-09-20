//! Transaction controls for the canonical schedule, using complete synthetic
//! F32/BF16 graphs. These are not real-pack numerical qualification.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::kv_cache::MlxKVCache;
use crate::weights::QuantizedWeight;
use std::path::PathBuf;

#[path = "qwen4_exp_mtp_transaction_evidence.rs"]
mod transaction_evidence;

struct TargetHooks;

impl TargetHooks {
    fn reset() {
        TRUNK_FORWARD_COUNT.with(|count| count.set(0));
        TRUNK_FORWARD_TOKENS.with(|calls| calls.borrow_mut().clear());
        FAIL_TARGET_CALL.with(|failure| failure.set(None));
        FAIL_ACCEPTED_CATCHUP.with(|failure| failure.set(false));
    }

    fn new() -> Self {
        Self::reset();
        Self
    }
}

impl Drop for TargetHooks {
    fn drop(&mut self) {
        Self::reset();
    }
}

fn assert_calls(expected: &[&[u32]]) {
    let expected = expected
        .iter()
        .map(|tokens| tokens.to_vec())
        .collect::<Vec<_>>();
    TRUNK_FORWARD_TOKENS.with(|calls| assert_eq!(*calls.borrow(), expected));
    TRUNK_FORWARD_COUNT.with(|count| assert_eq!(count.get(), expected.len()));
}

fn fixture() -> (Qwen4ExpWeights, Qwen4ExpMtpWeights) {
    let root = PathBuf::from(
        std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").expect("synthetic MTP oracle directory"),
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let text = &config["text_config"];
    // Do not let a real pack turn this bounded local regression into inference.
    assert_eq!(text["hidden_size"].as_u64(), Some(16));
    assert_eq!(text["num_hidden_layers"].as_u64(), Some(4));
    assert_eq!(text["vocab_size"].as_u64(), Some(32));
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
    let mut trunk = crate::weights::qwen4_exp::load_with_paging_policy(
        &root,
        &manifest,
        crate::expert_stream::StreamExpertsMode::Off,
        1,
    )
    .unwrap();
    let head = crate::weights::qwen4_exp_mtp::load(&root, &manifest, &trunk).unwrap();
    trunk.target_schedule = Qwen4ExpTargetSchedule::CanonicalSingleton;
    assert!(matches!(
        trunk.token_embedding.weight.dtype(),
        MlxDtype::Float32 | MlxDtype::Bfloat16
    ));
    (trunk, head)
}

fn ordinary(
    trunk: &Qwen4ExpWeights,
    tokens: &[u32],
    state: &Qwen4ExpState,
    owner: u64,
) -> Qwen4ExpOutput {
    // This oracle never goes through the verifier or its scheduling helper.
    qwen4_exp::forward(trunk, tokens, state, owner, ProjectionBatchPolicy::Shared).unwrap()
}

fn state_bytes(state: &Qwen4ExpState, layers: usize) -> Vec<u8> {
    let mut cache = MlxKVCache::new_contiguous(layers);
    cache.qwen4_exp = Some(state.clone());
    cache.advance(state.position());
    // The real codec includes PLE token history, which arrays() omits.
    cache.serialize_to_bytes()
}

fn bits(array: &MlxArray) -> Vec<u32> {
    let widened = mlx_sys::contiguous(&astype(array, MlxDtype::Float32, None), None);
    try_eval(&[&widened]).unwrap();
    assert!(widened.data_f32().iter().all(|value| value.is_finite()));
    widened
        .data_f32()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn assert_array(actual: &MlxArray, expected: &MlxArray) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(bits(actual), bits(expected));
}

fn assert_output(actual: &Qwen4ExpOutput, expected: &Qwen4ExpOutput, layers: usize) {
    assert_array(&actual.logits, &expected.logits);
    assert_array(&actual.hidden, &expected.hidden);
    assert_array(&actual.stream_hidden, &expected.stream_hidden);
    assert_eq!(
        state_bytes(&actual.state, layers),
        state_bytes(&expected.state, layers)
    );
}

fn populated_prefix(
    trunk: &Qwen4ExpWeights,
    head: &Qwen4ExpMtpWeights,
    owner: u64,
) -> (Qwen4ExpOutput, Qwen4ExpDraftCursor) {
    let prefix = ordinary(
        trunk,
        &[1, 2, 3, 4, 5, 6, 7, 8],
        &Qwen4ExpState::new(trunk, owner),
        owner,
    );
    let mut cursor = Qwen4ExpDraftCursor::new(head, owner);
    cursor
        .absorb(head, &[1, 2, 3, 4, 5, 6, 7, 8], &prefix.stream_hidden)
        .unwrap();
    let last = ordinary(trunk, &[9], &prefix.state, owner);
    cursor.absorb(head, &[9], &last.stream_hidden).unwrap();
    assert_eq!(last.state.position(), 9);
    assert_eq!(cursor.draft_state.position(), 8);
    assert!(cursor.aligned(&last.state));
    let descriptors = last.state.array_descriptors();
    let arrays = last.state.arrays();
    assert_eq!(descriptors.len(), arrays.len());
    for kind in [
        "gdn.conv",
        "gdn.recurrent",
        "qsa.k",
        "qsa.v",
        "qsa.index",
        "ple.conv",
    ] {
        assert!(
            descriptors.iter().any(|entry| {
                entry.kind == kind
                    && bits(arrays[entry.index])
                        .iter()
                        .any(|value| value & 0x7fff_ffff != 0)
            }),
            "missing nonzero {kind} prefix state"
        );
    }
    (last, cursor)
}

// Only the vocabulary projection is controlled. The complete trunk/head,
// including their nonzero attention, MoE, HC and PLE computation, still runs.
fn set_constant_head(weight: &mut QuantizedWeight, maxima: &[usize]) {
    let shape = weight.weight.shape();
    assert_eq!(shape, [32, 16]);
    let dtype = weight.weight.dtype();
    let mut bias = vec![-2.0; shape[0] as usize];
    for &token in maxima {
        bias[token] = 2.0;
    }
    *weight = QuantizedWeight::new(mlx_sys::zeros(&shape, dtype, None), None, None)
        .with_linear_bias(Some(astype(&MlxArray::from_f32_slice(&bias), dtype, None)));
}

#[derive(Debug, PartialEq)]
struct CursorSnapshot {
    state: Vec<u8>,
    hidden: Option<(Vec<i32>, MlxDtype, Vec<u32>)>,
    owner: u64,
    proposed: usize,
    accepted: usize,
}

fn snapshot(cursor: &Qwen4ExpDraftCursor) -> CursorSnapshot {
    CursorSnapshot {
        state: state_bytes(&cursor.draft_state, 1),
        hidden: cursor
            .stream_hidden
            .as_ref()
            .map(|array| (array.shape(), array.dtype(), bits(array))),
        owner: cursor.owner,
        proposed: cursor.proposed,
        accepted: cursor.accepted,
    }
}

fn assert_cursor(
    cursor: &Qwen4ExpDraftCursor,
    expected_head: &Qwen4ExpOutput,
    expected_trunk: &Qwen4ExpOutput,
    proposed: usize,
    accepted: usize,
) {
    assert_eq!(
        state_bytes(&cursor.draft_state, 1),
        state_bytes(&expected_head.state, 1)
    );
    assert_array(
        cursor.stream_hidden.as_ref().unwrap(),
        &expected_trunk.stream_hidden,
    );
    assert_eq!(cursor.proposed, proposed);
    assert_eq!(cursor.accepted, accepted);
    assert!(cursor.aligned(&expected_trunk.state));
}

#[test]
#[ignore = "requires bounded synthetic F32/BF16 Flash Next MTP artifacts and Metal"]
fn canonical_verify_preserves_real_singleton_outputs_and_full_checkpoints() {
    let _hooks = TargetHooks::new();
    let _ordinary_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let (trunk, head) = fixture();
    let owner = 3201;
    let layers = trunk.layers.len();
    let (prefix, _) = populated_prefix(&trunk, &head, owner);
    let checkpoint = prefix.state.clone();
    let before = state_bytes(&checkpoint, layers);
    let primary = next_token(&prefix).unwrap();
    let a = ordinary(&trunk, &[primary], &checkpoint, owner);
    let draft = next_token(&a).unwrap();
    let a_before = state_bytes(&a.state, layers);
    let a_logits_before = bits(&a.logits);
    let a_hidden_before = bits(&a.hidden);
    let a_stream_hidden_before = bits(&a.stream_hidden);
    let b = ordinary(&trunk, &[draft], &a.state, owner);
    assert_eq!(state_bytes(&a.state, layers), a_before);
    assert_eq!(bits(&a.logits), a_logits_before);
    assert_eq!(bits(&a.hidden), a_hidden_before);
    assert_eq!(bits(&a.stream_hidden), a_stream_hidden_before);
    let accepted = verify_one(&trunk, &checkpoint, owner, primary, draft, 3, &[]).unwrap();
    assert_eq!(state_bytes(&a.state, layers), a_before);
    assert_eq!(bits(&a.logits), a_logits_before);
    assert_eq!(bits(&a.hidden), a_hidden_before);
    assert_eq!(bits(&a.stream_hidden), a_stream_hidden_before);
    assert_calls(&[&[primary], &[draft]]);
    assert!(accepted.accepted);
    assert_eq!(accepted.target_schedule, "canonical_singleton");
    assert!(accepted.verification_logits.is_none());
    assert_eq!(accepted.committed, [primary, draft]);
    assert_output(&accepted.after_primary, &a, layers);
    assert_output(accepted.after_draft.as_ref().unwrap(), &b, layers);
    assert_eq!(accepted.next_primary, next_token(&b).unwrap());
    assert_eq!(
        accepted.after_primary.state.position(),
        checkpoint.position() + 1
    );
    assert_eq!(
        accepted.after_draft.as_ref().unwrap().state.position(),
        checkpoint.position() + 2
    );
    assert_eq!(accepted.rejection_wall_us, 0);

    let rejected_draft = (draft + 1) % 32;
    for (candidate, budget, terminal) in [
        (rejected_draft, 3, vec![]),
        (draft, 3, vec![primary]),
        (draft, 3, vec![draft]),
        (draft, 1, vec![]),
    ] {
        TargetHooks::reset();
        let step = verify_one(
            &trunk,
            &checkpoint,
            owner,
            primary,
            candidate,
            budget,
            &terminal,
        )
        .unwrap();
        assert_calls(&[&[primary]]);
        assert!(!step.accepted);
        assert!(step.after_draft.is_none());
        assert!(step.verification_logits.is_none());
        assert_eq!(step.committed, [primary]);
        assert_eq!(step.next_primary, draft);
        assert_eq!(step.bonus_wall_us, 0);
        assert_eq!(step.rejection_wall_us, 0);
        assert_output(&step.after_primary, &a, layers);
    }
    TargetHooks::reset();
    assert!(verify_one(&trunk, &checkpoint, owner, primary, draft, 0, &[]).is_err());
    assert_calls(&[]);

    let continuation = next_token(&b).unwrap();
    let expected = ordinary(&trunk, &[continuation], &b.state, owner);
    let actual = ordinary(
        &trunk,
        &[continuation],
        &accepted.after_draft.as_ref().unwrap().state,
        owner,
    );
    assert_output(&actual, &expected, layers);
    assert_eq!(state_bytes(&checkpoint, layers), before);
    assert_eq!(state_bytes(&prefix.state, layers), before);
    assert_eq!(state_bytes(&a.state, layers), a_before);
    // Reusing the retained checkpoint also exercises its PLE n-gram history.
    assert_output(
        &ordinary(&trunk, &[primary], &checkpoint, owner),
        &a,
        layers,
    );

    // Selector unit control: use separated Float32 ties across reduction blocks.
    // These rows are not substituted into verify_one or a cursor transaction.
    let mut values = vec![-3.0_f32; 2 * 4097];
    for index in [17, 2049, 4096] {
        values[index] = 7.0;
    }
    for index in [2049, 4096] {
        values[4097 + index] = 7.0;
    }
    let logits = reshape(&MlxArray::from_f32_slice(&values), &[2, 4097], None);
    assert_eq!(token_at_row(&logits, 0).unwrap(), 17);
    assert_eq!(token_at_row(&logits, 1).unwrap(), 2049);
    let selector_output = Qwen4ExpOutput {
        logits,
        hidden: prefix.hidden,
        stream_hidden: prefix.stream_hidden,
        state: prefix.state,
    };
    assert_eq!(next_token(&selector_output).unwrap(), 2049);
    assert_eq!(top_two_margin(&selector_output.logits, 0).unwrap(), 0.0);
}

#[test]
#[ignore = "requires bounded synthetic F32/BF16 Flash Next MTP artifacts and Metal"]
fn canonical_cursor_uses_primary_hidden_for_catchup_and_bonus_hidden_for_continuation() {
    let _hooks = TargetHooks::new();
    let _ordinary_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let (mut trunk, mut head) = fixture();
    set_constant_head(&mut trunk.lm_head, &[3, 17]);
    set_constant_head(&mut head.graph.lm_head, &[3]);
    let owner = 3202;
    let layers = trunk.layers.len();
    let (prefix, initial) = populated_prefix(&trunk, &head, owner);
    let before = snapshot(&initial);
    let trunk_before = state_bytes(&prefix.state, layers);
    let mut cursor = initial.clone();
    let mut state = prefix.state.clone();
    let mut primary = next_token(&prefix).unwrap();
    assert_eq!(primary, 3);

    for count in 1..=2 {
        let a = ordinary(&trunk, &[primary], &state, owner);
        let draft = next_token(&a).unwrap();
        let b = ordinary(&trunk, &[draft], &a.state, owner);
        assert_ne!(bits(&a.stream_hidden), bits(&b.stream_hidden));
        assert_eq!(draft, 3);
        assert_eq!(next_token(&b).unwrap(), 3);
        assert_eq!(top_two_margin(&a.logits, 0).unwrap(), 0.0);
        assert_eq!(top_two_margin(&b.logits, 0).unwrap(), 0.0);
        let proposed = head_forward(
            &head,
            cursor.stream_hidden.as_ref().unwrap(),
            &[primary],
            &cursor.draft_state,
            cursor.owner,
        )
        .unwrap();
        assert_eq!(next_token(&proposed).unwrap(), draft);
        // Independent full head, not the cache-only catch-up helper under test.
        let caught_up = head_forward(
            &head,
            &a.stream_hidden,
            &[draft],
            &proposed.state,
            cursor.owner,
        )
        .unwrap();
        TargetHooks::reset();
        let result = cursor
            .step(&trunk, &head, &state, owner, primary, 3, &[])
            .unwrap();
        assert_calls(&[&[primary], &[draft]]);
        assert!(result.accepted);
        assert_eq!(result.committed_len, 2);
        assert_eq!(result.emitted, [draft, 3]);
        assert_eq!(result.correction_margin, 0.0);
        assert_eq!(result.bonus_margin, 0.0);
        assert_eq!(result.rejection_wall_us, 0);
        assert_eq!(
            state_bytes(&result.trunk_state, layers),
            state_bytes(&b.state, layers)
        );
        assert_cursor(&cursor, &caught_up, &b, count, count);
        state = result.trunk_state;
        primary = result.emitted[1];
    }
    assert_eq!(snapshot(&initial), before);
    assert_eq!(state_bytes(&prefix.state, layers), trunk_before);

    // Rejection still computes a real draft, but commits only ordinary A.
    set_constant_head(&mut head.graph.lm_head, &[5]);
    let a = ordinary(&trunk, &[3], &prefix.state, owner);
    let proposed = head_forward(
        &head,
        initial.stream_hidden.as_ref().unwrap(),
        &[3],
        &initial.draft_state,
        initial.owner,
    )
    .unwrap();
    assert_eq!(next_token(&proposed).unwrap(), 5);
    let mut rejected = initial.clone();
    TargetHooks::reset();
    let result = rejected
        .advance(&trunk, &head, &prefix.state, owner, 3, 3, &[])
        .unwrap();
    assert_calls(&[&[3]]);
    assert!(!result.accepted);
    assert_eq!(result.consumed, [3]);
    assert_eq!(result.next_primary, 3);
    assert_eq!(result.observation.draft_token, Some(5));
    assert_eq!(result.observation.target_schedule, "canonical_singleton");
    assert!(result.observation.verification_logits.is_none());
    assert_array(
        result
            .observation
            .canonical_correction_logits
            .as_ref()
            .unwrap(),
        &a.logits,
    );
    assert_array(&result.observation.next_logits, &a.logits);
    assert_eq!(
        state_bytes(&result.trunk_state, layers),
        state_bytes(&a.state, layers)
    );
    assert_cursor(&rejected, &proposed, &a, 1, 0);

    // A one-slot request skips the proposal logits yet preserves shifted head history.
    let mut final_slot = initial.clone();
    TargetHooks::reset();
    let result = final_slot
        .advance(&trunk, &head, &prefix.state, owner, 3, 1, &[])
        .unwrap();
    assert_calls(&[&[3]]);
    assert!(!result.accepted);
    assert_eq!(result.consumed, [3]);
    assert_eq!(result.observation.draft_token, None);
    assert!(result.observation.verification_logits.is_none());
    assert_array(
        result
            .observation
            .canonical_correction_logits
            .as_ref()
            .unwrap(),
        &a.logits,
    );
    assert_cursor(&final_slot, &proposed, &a, 0, 0);

    TargetHooks::reset();
    let zero_before = snapshot(&final_slot);
    assert!(
        final_slot
            .step(&trunk, &head, &a.state, owner, 3, 0, &[])
            .is_err()
    );
    assert_calls(&[]);
    assert_eq!(snapshot(&final_slot), zero_before);
    assert_eq!(snapshot(&initial), before);
    assert_eq!(state_bytes(&prefix.state, layers), trunk_before);
}

#[test]
#[ignore = "requires bounded synthetic F32/BF16 Flash Next MTP artifacts and Metal"]
fn canonical_cursor_materialized_failures_preserve_transaction_and_retry() {
    let _hooks = TargetHooks::new();
    let _ordinary_scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let (mut trunk, mut head) = fixture();
    set_constant_head(&mut trunk.lm_head, &[3, 17]);
    set_constant_head(&mut head.graph.lm_head, &[3]);
    let owner = 3203;
    let layers = trunk.layers.len();
    let (prefix, mut initial) = populated_prefix(&trunk, &head, owner);
    let warm = initial
        .step(&trunk, &head, &prefix.state, owner, 3, 3, &[])
        .unwrap();
    assert!(warm.accepted);
    assert_eq!((initial.proposed, initial.accepted), (1, 1));
    let state = warm.trunk_state;
    let trunk_before = state_bytes(&state, layers);
    let before = snapshot(&initial);
    let a = ordinary(&trunk, &[3], &state, owner);
    let b = ordinary(&trunk, &[3], &a.state, owner);
    let proposed = head_forward(
        &head,
        initial.stream_hidden.as_ref().unwrap(),
        &[3],
        &initial.draft_state,
        initial.owner,
    )
    .unwrap();
    let caught_up = head_forward(
        &head,
        &a.stream_hidden,
        &[3],
        &proposed.state,
        initial.owner,
    )
    .unwrap();

    for (target_failure, catchup_failure, calls) in
        [(Some(1), false, 1), (Some(2), false, 2), (None, true, 2)]
    {
        let mut cursor = initial.clone();
        TargetHooks::reset();
        FAIL_TARGET_CALL.with(|failure| failure.set(target_failure));
        FAIL_ACCEPTED_CATCHUP.with(|failure| failure.set(catchup_failure));
        let failure = cursor
            .step(&trunk, &head, &state, owner, 3, 3, &[])
            .err()
            .expect("injected failure");
        assert!(failure.contains("after materialized"));
        assert_calls(&vec![&[3][..]; calls]);
        assert_eq!(snapshot(&cursor), before);
        assert_eq!(snapshot(&initial), before);
        assert_eq!(state_bytes(&state, layers), trunk_before);
        assert!(cursor.aligned(&state));

        TargetHooks::reset();
        let retried = cursor
            .step(&trunk, &head, &state, owner, 3, 3, &[])
            .unwrap();
        assert_calls(&[&[3], &[3]]);
        assert!(retried.accepted);
        assert_eq!(retried.committed_len, 2);
        assert_eq!(retried.emitted, [3, 3]);
        assert_eq!(
            state_bytes(&retried.trunk_state, layers),
            state_bytes(&b.state, layers)
        );
        assert_cursor(&cursor, &caught_up, &b, 2, 2);
        assert_eq!(snapshot(&initial), before);
        assert_eq!(state_bytes(&state, layers), trunk_before);
        assert_output(&ordinary(&trunk, &[3], &state, owner), &a, layers);
    }
}
