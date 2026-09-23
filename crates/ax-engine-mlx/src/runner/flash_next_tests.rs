//! Artifact-backed tests of the actual scheduler-facing Flash Next MTP route.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::model::qwen4_exp::Qwen4ExpState;
use crate::model::qwen4_exp_mtp::mtp_parity;
use crate::model::shared::ProjectionBatchPolicy;
use crate::weights::qwen4_exp::Qwen4ExpTargetSchedule;
use ax_engine_core::{
    ExecutionBatch, ExecutionItem, PositionRange, RouteMetadata, StepId, WorkUnitKind,
};
use std::path::PathBuf;

mod aligned_state_tests;

fn artifacts() -> NativeModelArtifacts {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_MTP_ORACLE_DIR").unwrap());
    let mut manifest = ax_engine_core::convert::convert_hf_model_dir(&root).unwrap();
    manifest.weight_sanitize = ax_engine_core::WeightSanitize::HfToMlx;
    manifest.runtime_status = ax_engine_core::NativeRuntimeStatus::default();
    NativeModelArtifacts::from_manifest_and_root(root, manifest).unwrap()
}

fn context(id: u64, prompt_len: usize, budget: u32) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(id),
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: 0,
        generated_len: 0,
        max_output_tokens: budget,
        seed: 42,
        deterministic_argmax_sampling: true,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: None,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: true,
        tool_call_mode: false,
        structured_output_mode: false,
        max_think_tokens: None,
        answer_reserve_tokens: None,
    }
}

fn execute(
    runner: &MlxRunner,
    ctx: RunnerRequestContext,
    tokens: &[u32],
    mode: ExecutionMode,
) -> RunnerOutput {
    execute_with_block_size(runner, ctx, tokens, mode, 4)
}

fn execute_with_block_size(
    runner: &MlxRunner,
    ctx: RunnerRequestContext,
    tokens: &[u32],
    mode: ExecutionMode,
    block_size_tokens: u32,
) -> RunnerOutput {
    let position = if mode == ExecutionMode::Prefill {
        ctx.processed_prompt_tokens
    } else {
        ctx.prompt_len + ctx.generated_len - 1
    };
    runner.run(RunnerInput {
        block_size_tokens,
        memory_pressure: None,
        block_tables: Vec::new(),
        request_contexts: vec![ctx],
        request_multimodal_inputs: Vec::new(),
        execution_batch: ExecutionBatch {
            step_id: StepId(u64::from(position)),
            model_id: "flash-next-candidate-test".into(),
            execution_plan_ref: None,
            total_scheduled_tokens: tokens.len() as u32,
            route_metadata: RouteMetadata {
                execution_plan: None,
                attention_route: None,
                kv_mode: None,
                prefix_cache_path: None,
                barrier_mode: None,
                crossover_decisions: Vec::new(),
            },
            items: vec![ExecutionItem {
                request_id: ctx.request_id,
                mode,
                planned_work_unit: if mode == ExecutionMode::Prefill {
                    WorkUnitKind::PrefillChunk
                } else {
                    WorkUnitKind::TokenDecode
                },
                input_token_slice: tokens.to_vec(),
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: position,
                    end_exclusive: position + tokens.len() as u32,
                },
                scheduled_token_count: tokens.len() as u32,
                block_table_ref: ctx.request_id,
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
        },
    })
}

#[derive(Clone)]
struct Generation {
    prompt_len: usize,
    tokens: Vec<u32>,
    routes: Vec<Vec<(String, u32)>>,
    prefill_seconds: f64,
    decode_seconds: f64,
    prefill_state: Option<Qwen4ExpState>,
}

fn snapshot_trunk(runner: &MlxRunner, request_id: RequestId) -> Option<Qwen4ExpState> {
    runner
        .states
        .lock()
        .get(&request_id)
        .and_then(|state| state.cache.qwen4_exp.clone())
}

fn flash_runner_state_bytes(state: &Qwen4ExpState, layers: usize) -> Vec<u8> {
    let mut cache = MlxKVCache::new_contiguous(layers);
    cache.qwen4_exp = Some(state.clone());
    cache.advance(state.position());
    cache.serialize_to_bytes()
}

#[test]
fn flash_next_finite_exact_ties_choose_first_token_on_cpu_and_mlx() {
    // Separate ties across reduction chunks in a full-size vocabulary. These
    // are exact Float32 ties, without NaNs, infinities, or a margin waiver.
    const VOCAB: usize = 262_144;
    let _scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    for ties in [
        [17, 65_553, 131_089, 196_625],
        [65_553, 131_089, 196_625, 262_143],
    ] {
        let mut values = vec![-2.0f32; VOCAB];
        for index in ties {
            values[index] = 3.0;
        }
        assert!(values.iter().all(|value| value.is_finite()));
        let sampling = MlxSamplingParams::greedy();
        let mut rng = Xorshift64::new(42);
        let cpu = crate::sampling::sample_categorical(&values, sampling, &[], &mut rng);
        assert_eq!(cpu, ties[0] as u32);
        let logits = MlxArray::from_raw_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values.as_slice()),
            &[VOCAB as i32],
            MlxDtype::Float32,
        );
        mlx_sys::eval(&[&logits]);
        // Direct prefill uses [vocab]; the verifier reduces a [1, vocab] row
        // with the same MLX argmax operation. Exercise both actual shapes.
        for row in [
            logits.clone(),
            mlx_sys::reshape(&logits, &[1, VOCAB as i32], None),
        ] {
            let token = crate::generate::sample_token_from_prefill_logits(
                &row,
                MlxSamplingRequest::new(sampling, &[]),
                &mut rng,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut Vec::new(),
            );
            assert_eq!(token, cpu, "shape={:?}, ties={ties:?}", row.shape());
        }
    }
}

#[test]
fn flash_next_exact_profile_is_false_and_nested_scope_restores_on_error() {
    fn failed_request(selected: bool) -> Result<(), &'static str> {
        let _call = crate::fastpath::scoped_qwen_linear_mtp_exact(selected);
        assert!(!crate::fastpath::qwen_linear_mtp_exact_enabled());
        {
            let _nested = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
            assert!(crate::fastpath::qwen_linear_mtp_exact_enabled());
        }
        assert!(!crate::fastpath::qwen_linear_mtp_exact_enabled());
        Err("synthetic request failure")
    }

    let original = crate::fastpath::qwen_linear_mtp_exact_enabled();
    {
        let _outer = crate::fastpath::scoped_qwen_linear_mtp_exact(true);
        let eligible = qwen_linear_mtp_exact_model_eligible("qwen4_exp", true, 1, &[]);
        assert!(!eligible);
        for explicit in [None, Some(false), Some(true)] {
            let (resolved, _) =
                crate::fastpath::resolve_qwen_linear_mtp_exact_with_override(eligible, explicit);
            assert!(!resolved);
            for mtp_requested in [false, true] {
                let selected = qwen_linear_mtp_exact_scope_for_request(resolved, mtp_requested);
                let result = failed_request(selected);
                assert_eq!(result, Err("synthetic request failure"));
                assert!(crate::fastpath::qwen_linear_mtp_exact_enabled());
            }
        }
    }
    assert_eq!(crate::fastpath::qwen_linear_mtp_exact_enabled(), original);
}

fn runner_identity_admitted(
    schedule: &Qwen4ExpTargetSchedule,
    identity: &mtp_parity::GreedyIdentityReport,
) -> bool {
    match schedule {
        Qwen4ExpTargetSchedule::CanonicalSingleton => identity.greedy_identity,
        Qwen4ExpTargetSchedule::LegacyBatched => identity.identity_until_first_tie,
        Qwen4ExpTargetSchedule::Unavailable(_) => false,
    }
}

#[test]
fn canonical_runner_evidence_rejects_a_tie_waiver_without_changing_legacy_admission() {
    let tied = mtp_parity::greedy_identity_until_tie(&[17, 31], &[17, 37], 0.0, 0.0).unwrap();
    assert!(!tied.greedy_identity);
    assert!(tied.identity_until_first_tie);
    assert!(!runner_identity_admitted(
        &Qwen4ExpTargetSchedule::CanonicalSingleton,
        &tied
    ));
    assert!(runner_identity_admitted(
        &Qwen4ExpTargetSchedule::LegacyBatched,
        &tied
    ));
    let exact = mtp_parity::greedy_identity_until_tie(&[17, 31], &[17, 31], 0.0, 0.0).unwrap();
    assert!(runner_identity_admitted(
        &Qwen4ExpTargetSchedule::CanonicalSingleton,
        &exact
    ));
    assert!(!runner_identity_admitted(
        &Qwen4ExpTargetSchedule::Unavailable("test".into()),
        &exact
    ));
}

/// Expected `Qwen4ExpState::position` after one production `MlxRunner::run`
/// prefill for this file's runner setup: `from_artifacts(..., true)` disables
/// n-gram, Flash Next has no generic MTP, sampling is greedy.
///
/// Direct policy primes `start_direct_pipeline` with the first generated
/// token when `max_output > 1`, so the snapshot is `prompt_len + 1`. A live
/// Flash Next MTP cursor skips that prime (`flash_next_cursor_owns_decode`)
/// and stays at `prompt_len`. The extra token is the mlx_lm-style double-
/// buffer lookahead, not a round-up of `block_size_tokens`, QSA ratio, or
/// PLE/n-gram window.
fn flash_next_run_prefill_snapshot_len(
    prompt_len: usize,
    max_output: u32,
    mtp_requested: bool,
) -> usize {
    let session_direct = true;
    let request_ngram_disabled = max_output < NGRAM_MIN_OUTPUT_FOR_ACCELERATION;
    let bootstrap_pipeline = should_bootstrap_direct_pipeline(
        session_direct,
        request_ngram_disabled,
        false,
        false,
        mtp_requested,
    );
    let flash_next_cursor_owns_decode = mtp_requested;
    let primes = bootstrap_pipeline && !flash_next_cursor_owns_decode && max_output > 1;
    prompt_len + usize::from(primes)
}

/// AXKB header: magic[4] + version u32 + seq_len u64. Byte 8 is the first
/// byte of `seq_len`. Values 5 versus 6 mean the snapshots are at different
/// token counts, not a later tensor payload difference.
#[test]
fn axkb_header_seq_len_is_little_endian_u64_at_byte_8() {
    let mut five = MlxKVCache::new_contiguous(1);
    five.advance(5);
    let mut six = MlxKVCache::new_contiguous(1);
    six.advance(6);
    let five_bytes = five.serialize_to_bytes();
    let six_bytes = six.serialize_to_bytes();
    assert_eq!(&five_bytes[..4], b"AXKB");
    assert_eq!(&five_bytes[4..8], &six_bytes[4..8]);
    assert_eq!(&five_bytes[8..16], &5u64.to_le_bytes());
    assert_eq!(&six_bytes[8..16], &6u64.to_le_bytes());
    assert_ne!(five_bytes[8], six_bytes[8]);
}

#[test]
fn flash_next_direct_prefill_snapshot_is_prompt_plus_pipeline_bootstrap() {
    // The real-pack failure at prompt_len=69 / block_size=4 was 70 vs 69 on
    // the direct policy only. MTP at the same schedule stayed at 69. That
    // extra position is `start_direct_pipeline` after greedy prefill with
    // max_output=3, not a partial-block round-up: every length in 65..=72
    // (both sides of the 4-token and 69-token boundaries) gets the same +1
    // on direct and +0 on MTP.
    const BLOCK: usize = 4;
    const MAX_OUTPUT: u32 = 3;
    for prompt_len in 65..=72 {
        let head = MlxRunner::linear_boundary_capture_head_len(BLOCK, 0, prompt_len);
        if prompt_len.is_multiple_of(BLOCK) {
            assert_eq!(head, None, "aligned prompt_len={prompt_len}");
        } else {
            assert_eq!(
                head,
                Some(prompt_len - prompt_len % BLOCK),
                "unaligned prompt_len={prompt_len}"
            );
        }
        assert_eq!(
            flash_next_run_prefill_snapshot_len(prompt_len, MAX_OUTPUT, false),
            prompt_len + 1,
            "direct greedy prefill snapshot at prompt_len={prompt_len}"
        );
        assert_eq!(
            flash_next_run_prefill_snapshot_len(prompt_len, MAX_OUTPUT, true),
            prompt_len,
            "MTP cursor prefill snapshot at prompt_len={prompt_len}"
        );
        assert_eq!(
            flash_next_run_prefill_snapshot_len(prompt_len, 1, false),
            prompt_len,
            "max_output=1 skips the direct-pipeline prime at prompt_len={prompt_len}"
        );
    }
    assert_eq!(flash_next_run_prefill_snapshot_len(5, 3, false), 6);
    assert_eq!(flash_next_run_prefill_snapshot_len(5, 3, true), 5);
    assert_eq!(flash_next_run_prefill_snapshot_len(69, 3, false), 70);
    assert_eq!(flash_next_run_prefill_snapshot_len(69, 3, true), 69);
}

impl Generation {
    fn maximum(&self, key: &str) -> u32 {
        self.routes
            .iter()
            .flat_map(|route| route.iter())
            .filter(|(name, _)| name == key)
            .map(|(_, value)| *value)
            .max()
            .unwrap_or_default()
    }

    fn min_correction_margin(&self) -> Option<f32> {
        self.routes
            .iter()
            .flat_map(|route| route.iter())
            .filter(|(name, value)| {
                name == "ax_mlx_flash_next_mtp_min_correction_margin_milli" && *value != u32::MAX
            })
            .map(|(_, value)| *value)
            .min()
            .map(|milli| milli as f32 / 1000.0)
    }
}

// The direct pipeline may already have consumed the first generated token.
// An empty suffix has no retained logits to inspect, so fail closed.
fn direct_margin_replay_tokens(
    position: usize,
    prompt_len: usize,
    matching_prefix: &[u32],
) -> Option<&[u32]> {
    let consumed = position.checked_sub(prompt_len)?;
    let suffix = matching_prefix.get(consumed..)?;
    (!suffix.is_empty()).then_some(suffix)
}

#[test]
fn direct_margin_replay_respects_pipeline_lookahead() {
    let prefix = [11751, 13];
    assert_eq!(
        direct_margin_replay_tokens(6, 5, &prefix),
        Some(&prefix[1..])
    );
    assert_eq!(
        direct_margin_replay_tokens(5, 5, &prefix),
        Some(&prefix[..])
    );
    assert_eq!(direct_margin_replay_tokens(6, 5, &prefix[..1]), None);
    assert_eq!(direct_margin_replay_tokens(8, 5, &prefix), None);
    assert_eq!(direct_margin_replay_tokens(4, 5, &prefix), None);
}

fn direct_candidate_gap(logits: &[f32], direct: u32, candidate: u32) -> Option<f32> {
    let direct = *logits.get(direct as usize)?;
    let candidate = *logits.get(candidate as usize)?;
    let gap = direct - candidate;
    (direct.is_finite() && candidate.is_finite() && gap.is_finite() && gap >= 0.0).then_some(gap)
}

#[test]
fn direct_margin_uses_the_actual_candidate_not_an_unrelated_runner_up() {
    assert_eq!(direct_candidate_gap(&[5.0, 4.9, 1.0], 0, 2), Some(4.0));
    assert_eq!(direct_candidate_gap(&[5.0, 5.0], 0, 1), Some(0.0));
    assert_eq!(direct_candidate_gap(&[5.0], 0, 1), None);
    assert_eq!(direct_candidate_gap(&[5.0, f32::NAN], 0, 1), None);
    assert_eq!(direct_candidate_gap(&[f32::INFINITY, 1.0], 0, 1), None);
    assert_eq!(direct_candidate_gap(&[1.0, 5.0], 0, 1), None);
}

#[test]
fn flash_next_prefill_cursor_action_precedence_prefers_a_live_cursor_over_a_stash() {
    use FlashNextPrefillCursorAction as Action;
    // A cold cache always initializes, whatever cursor state exists.
    assert_eq!(
        flash_next_prefill_cursor_action(0, false, false, false),
        Action::Initialize
    );
    assert_eq!(
        flash_next_prefill_cursor_action(0, true, true, true),
        Action::Initialize
    );
    // A live aligned cursor is kept even when a restored payload is stashed:
    // the stash is only consulted when there is no live cursor to begin with.
    assert_eq!(
        flash_next_prefill_cursor_action(8, true, true, true),
        Action::Keep
    );
    // A live unaligned cursor is dropped even when a restored payload is
    // stashed.
    assert_eq!(
        flash_next_prefill_cursor_action(8, true, false, true),
        Action::Drop
    );
    // No live cursor: a decoded-and-verified payload resumes the cursor.
    assert_eq!(
        flash_next_prefill_cursor_action(8, false, false, true),
        Action::ResumeWithRestoredCursor
    );
    // No live cursor and no stash: resume without a cursor.
    assert_eq!(
        flash_next_prefill_cursor_action(8, false, false, false),
        Action::ResumeWithoutCursor
    );
}

#[test]
fn flash_next_mtp_telemetry_merges_and_emits_cursor_restored() {
    let mut telemetry = FlashNextMtpTelemetry {
        cursor_restored: 2,
        resumed_without_cursor: 3,
        ..Default::default()
    };
    let other = FlashNextMtpTelemetry {
        cursor_restored: 1,
        resumed_without_cursor: 4,
        ..Default::default()
    };
    telemetry.merge_from(other);
    assert_eq!(telemetry.cursor_restored, 3);
    assert_eq!(telemetry.resumed_without_cursor, 7);
    let mut decisions: Vec<(String, u32)> = Vec::new();
    telemetry.append_route_decisions(&mut decisions);
    let restored = decisions
        .iter()
        .find(|(key, _)| key == "ax_mlx_flash_next_mtp_cursor_restored")
        .map(|(_, value)| *value);
    assert_eq!(restored, Some(3));
    let resumed = decisions
        .iter()
        .find(|(key, _)| key == "ax_mlx_flash_next_mtp_resumed_without_cursor")
        .map(|(_, value)| *value);
    assert_eq!(resumed, Some(7));
}

fn direct_margin_at_divergence(
    runner: &MlxRunner,
    prefill: Option<&Qwen4ExpState>,
    prompt_len: usize,
    matching_prefix: &[u32],
    expected_token: u32,
    candidate_token: u32,
) -> Option<f32> {
    let trunk = runner.weights.qwen4_exp.as_ref()?;
    let prefill = prefill?;
    let replay = direct_margin_replay_tokens(prefill.position(), prompt_len, matching_prefix)?;
    let owner = runner.cfg.compile_cache_identity;
    let mut state = prefill.clone();
    let mut output = None;
    for &token in replay {
        let next = crate::model::qwen4_exp::forward(
            trunk,
            &[token],
            &state,
            owner,
            ProjectionBatchPolicy::Shared,
        )
        .ok()?;
        state = next.state.clone();
        output = Some(next);
    }
    let output = output?;
    let row = output.logits.shape().first().copied()?.saturating_sub(1);
    if row != 0 {
        return None;
    }
    let token = mlx_sys::argmax(&output.logits, None);
    mlx_sys::try_eval(&[&token]).ok()?;
    if token.data_u32().first().copied() != Some(expected_token) {
        return None;
    }
    let logits = mlx_sys::astype(&output.logits, mlx_sys::MlxDtype::Float32, None);
    mlx_sys::try_eval(&[&logits]).ok()?;
    direct_candidate_gap(logits.data_f32(), expected_token, candidate_token)
}

fn runner_greedy_identity(
    direct: &Generation,
    mtp: &Generation,
    runner: &MlxRunner,
) -> mtp_parity::GreedyIdentityReport {
    let tie_margin = mtp_parity::mtp_tie_margin();
    let shared = direct.tokens.len().min(mtp.tokens.len());
    let mut position = shared;
    for index in 0..shared {
        if direct.tokens[index] != mtp.tokens[index] {
            position = index;
            break;
        }
    }
    if position == shared && direct.tokens.len() == mtp.tokens.len() {
        let mut report = mtp_parity::GreedyIdentityReport::exact();
        report.compared_positions = shared;
        return report;
    }
    let direct_margin = if position < shared {
        direct_margin_at_divergence(
            runner,
            direct.prefill_state.as_ref(),
            direct.prompt_len,
            &direct.tokens[..position],
            direct.tokens[position],
            mtp.tokens[position],
        )
    } else {
        None
    };
    // A correction margin from another verifier position cannot certify this tie.
    let margin = direct_margin.unwrap_or(f32::INFINITY);
    mtp_parity::greedy_identity_until_tie(&direct.tokens, &mtp.tokens, margin, tie_margin)
        .unwrap_or_else(|error| panic!("{error}"))
}

fn generate(
    runner: &MlxRunner,
    prompt: &[u32],
    quantum: usize,
    ctx: RunnerRequestContext,
) -> Generation {
    generate_with_block_size(runner, prompt, quantum, ctx, 4)
}

fn generate_with_block_size(
    runner: &MlxRunner,
    prompt: &[u32],
    quantum: usize,
    ctx: RunnerRequestContext,
    block_size: u32,
) -> Generation {
    generate_observed(runner, prompt, quantum, ctx, block_size, |_| {})
}

fn generate_observed(
    runner: &MlxRunner,
    prompt: &[u32],
    quantum: usize,
    mut ctx: RunnerRequestContext,
    block_size: u32,
    mut observe: impl FnMut(&Qwen4ExpState),
) -> Generation {
    let mut result = Generation {
        prompt_len: prompt.len(),
        tokens: Vec::new(),
        routes: Vec::new(),
        prefill_seconds: 0.0,
        decode_seconds: 0.0,
        prefill_state: None,
    };
    for chunk in prompt.chunks(quantum) {
        let prefill_started = Instant::now();
        let output =
            execute_with_block_size(runner, ctx, chunk, ExecutionMode::Prefill, block_size);
        result.prefill_seconds += prefill_started.elapsed().as_secs_f64();
        let update = &output.request_updates[0];
        assert!(update.error.is_none(), "{:?}", update.error);
        result.tokens.extend(update.output_token);
        result.tokens.extend_from_slice(&update.output_tokens);
        result
            .routes
            .push(output.route_metadata.crossover_decisions);
        if let Some(state) = snapshot_trunk(runner, ctx.request_id) {
            observe(&state);
            result.prefill_state = Some(state);
        }
        ctx.processed_prompt_tokens += chunk.len() as u32;
        if ctx.processed_prompt_tokens < ctx.prompt_len {
            assert!(result.tokens.is_empty());
        }
        if update.stop_reason.is_some() {
            return result;
        }
    }
    assert_eq!(result.tokens.len(), 1);
    while result.tokens.len() < ctx.max_output_tokens as usize {
        ctx.generated_len = result.tokens.len() as u32;
        let decode_started = Instant::now();
        let output = execute_with_block_size(
            runner,
            ctx,
            &[*result.tokens.last().unwrap()],
            ExecutionMode::Decode,
            block_size,
        );
        result.decode_seconds += decode_started.elapsed().as_secs_f64();
        let update = &output.request_updates[0];
        assert!(update.error.is_none(), "{:?}", update.error);
        assert!(update.output_token.is_some());
        result.tokens.extend(update.output_token);
        result.tokens.extend_from_slice(&update.output_tokens);
        result
            .routes
            .push(output.route_metadata.crossover_decisions);
        assert!(result.tokens.len() <= ctx.max_output_tokens as usize);
        if let Some(state) = runner.states.lock().get(&ctx.request_id) {
            if let Some(trunk) = state.cache.qwen4_exp.as_ref() {
                observe(trunk);
            }
            if let Some(cursor) = &state.flash_next_mtp.cursor {
                let trunk = state.cache.qwen4_exp.as_ref().unwrap();
                assert_eq!(trunk.position(), state.cache.seq_len());
                assert!(cursor.aligned(trunk));
            }
        }
        if update.stop_reason.is_some() {
            break;
        }
    }
    assert!(
        !runner.states.lock().contains_key(&ctx.request_id),
        "finished request retained state"
    );
    result
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts and explicit candidate attachment"]
fn flash_next_runner_mtp_matches_direct_across_prefill_quanta_and_budgets() {
    let artifacts = artifacts();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2, true).unwrap();
    assert!(
        runner.has_mtp(),
        "synthetic Flash Next artifacts must include mtp.safetensors"
    );
    assert!(runner.weights.mtp.is_none());
    assert!(!runner.mtp_model_policy.certified_default_on());
    let mut id = 100;
    for prompt in [vec![1], vec![1, 2, 3, 4, 5, 6, 7, 8, 9]] {
        for quantum in [1, 3, 100] {
            for budget in [1, 2, 3, 8] {
                // Clear only the test's private prefix stores so both runs exercise cold prefill.
                *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
                    max_bytes: 64 * 1024 * 1024,
                    max_entries: 128,
                });
                *runner.native_prefix_cache.lock() =
                    MlxNativePrefixCache::new(MlxPrefixCachePolicy {
                        max_bytes: 64 * 1024 * 1024,
                        max_entries: 128,
                    });
                runner.set_mtp_requested(false);
                let direct = generate(&runner, &prompt, quantum, context(id, prompt.len(), budget));
                id += 1;
                *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
                    max_bytes: 64 * 1024 * 1024,
                    max_entries: 128,
                });
                *runner.native_prefix_cache.lock() =
                    MlxNativePrefixCache::new(MlxPrefixCachePolicy {
                        max_bytes: 64 * 1024 * 1024,
                        max_entries: 128,
                    });
                runner.set_mtp_requested(true);
                let candidate =
                    generate(&runner, &prompt, quantum, context(id, prompt.len(), budget));
                id += 1;
                assert_eq!(
                    candidate.tokens, direct.tokens,
                    "quantum={quantum}, budget={budget}"
                );
                assert_eq!(
                    runner_greedy_identity(&direct, &candidate, &runner).compared_positions,
                    direct.tokens.len()
                );
                assert_eq!(direct.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
                if budget > 1 {
                    assert!(candidate.maximum("ax_mlx_flash_next_mtp_verified_steps") > 0);
                }
                assert_eq!(candidate.maximum("ax_mlx_mtp_model_policy"), 10);
                assert_eq!(candidate.maximum("ax_mlx_flash_next_mtp_step_errors"), 0);
            }
        }
    }
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts; explicitly selects canonical scheduling"]
fn flash_next_canonical_runner_matches_aligned_states_and_continuation() {
    let _scope = crate::fastpath::scoped_qwen_linear_mtp_exact(false);
    let artifacts = artifacts();
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    for force_accept in [false, true] {
        let mut weights = crate::weights::load_weights(&artifacts).unwrap();
        // The synthetic affine/BF16 fixtures exercise scheduling mechanics;
        // overriding their schedule here does not certify a real MXFP4 pack.
        weights.qwen4_exp.as_mut().unwrap().target_schedule =
            Qwen4ExpTargetSchedule::CanonicalSingleton;
        if force_accept {
            // Keep the nonzero stateful trunk, but force exact greedy ties in
            // both heads so acceptance and bonus emission are exercised.
            let shape = [
                artifacts.manifest().vocab_size as i32,
                artifacts.manifest().hidden_size as i32,
            ];
            let data = vec![0.0f32; (shape[0] * shape[1]) as usize];
            let head = crate::weights::QuantizedWeight::new(
                MlxArray::from_raw_data(
                    data.as_ptr().cast(),
                    std::mem::size_of_val(data.as_slice()),
                    &shape,
                    MlxDtype::Float32,
                ),
                None,
                None,
            );
            weights.lm_head = head.clone();
            weights.qwen4_exp.as_mut().unwrap().lm_head = head.clone();
            weights.qwen4_exp_mtp.as_mut().unwrap().graph.lm_head = head;
        }
        let shared = MlxSharedWeightsCell::new();
        shared.publish(Arc::new(weights));
        let mut runner = MlxRunner::from_artifacts_with_runtime_shares(
            &artifacts,
            2,
            true,
            true,
            None,
            Some(&shared),
        )
        .unwrap();
        assert!(runner.has_mtp());
        for quantum in [3, 100] {
            let mut snapshots = [
                std::collections::BTreeMap::new(),
                std::collections::BTreeMap::new(),
            ];
            let mut generations = Vec::new();
            for (mode, states) in snapshots.iter_mut().enumerate() {
                *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
                    max_bytes: 64 * 1024 * 1024,
                    max_entries: 128,
                });
                *runner.native_prefix_cache.lock() =
                    MlxNativePrefixCache::new(MlxPrefixCachePolicy {
                        max_bytes: 64 * 1024 * 1024,
                        max_entries: 128,
                    });
                runner.set_mtp_requested(mode == 1);
                let generation = generate_observed(
                    &runner,
                    &prompt,
                    quantum,
                    context(1200 + mode as u64, prompt.len(), 8),
                    4,
                    |state| {
                        if state.position() >= prompt.len() {
                            states.insert(state.position(), state.clone());
                        }
                    },
                );
                generations.push(generation);
            }
            let direct = &generations[0];
            let candidate = &generations[1];
            assert_eq!(direct.tokens.len(), 8);
            assert_eq!(
                candidate.tokens, direct.tokens,
                "quantum={quantum}, force_accept={force_accept}"
            );
            assert!(runner_greedy_identity(direct, candidate, &runner).greedy_identity);
            assert!(candidate.maximum("ax_mlx_flash_next_mtp_verified_steps") > 0);
            assert_eq!(candidate.maximum("ax_mlx_flash_next_mtp_step_errors"), 0);
            if force_accept {
                assert_eq!(candidate.tokens, vec![0; 8]);
                assert!(candidate.maximum("ax_mlx_flash_next_mtp_accepted_steps") > 0);
            }
            let trunk = runner.weights.qwen4_exp.as_ref().unwrap();
            let mut aligned = Vec::new();
            for (&position, direct_state) in &snapshots[0] {
                let Some(candidate_state) = snapshots[1].get(&position) else {
                    continue;
                };
                let consumed = position - prompt.len();
                assert_eq!(&direct.tokens[..consumed], &candidate.tokens[..consumed]);
                assert_eq!(
                    flash_runner_state_bytes(direct_state, trunk.layers.len()),
                    flash_runner_state_bytes(candidate_state, trunk.layers.len()),
                    "aligned consumed prefix={consumed}, quantum={quantum}, force_accept={force_accept}"
                );
                aligned.push((position, direct_state, candidate_state));
            }
            assert!(
                aligned.len() >= 2,
                "must compare live decode states, not only prefill"
            );
            let (position, direct_state, candidate_state) = aligned.last().unwrap();
            let consumed = position - prompt.len();
            assert!(consumed > 0 && consumed + 1 < direct.tokens.len());
            let token = direct.tokens[consumed];
            // Resume ordinary singleton execution from each retained state at
            // the same consumed prefix, including PLE history and QSA caches.
            let continue_from = |state| {
                crate::model::qwen4_exp::forward(
                    trunk,
                    &[token],
                    state,
                    runner.cfg.compile_cache_identity,
                    ProjectionBatchPolicy::Shared,
                )
                .unwrap()
            };
            let continued_direct = continue_from(direct_state);
            let continued_candidate = continue_from(candidate_state);
            assert_eq!(
                flash_runner_state_bytes(&continued_direct.state, trunk.layers.len()),
                flash_runner_state_bytes(&continued_candidate.state, trunk.layers.len()),
            );
            for output in [&continued_direct, &continued_candidate] {
                let next = mlx_sys::argmax(&output.logits, None);
                mlx_sys::eval(&[&next]);
                assert_eq!(next.data_u32()[0], direct.tokens[consumed + 1]);
            }
        }
    }
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts and explicit candidate attachment"]
fn flash_next_runner_mtp_request_release_drops_draft_history() {
    let artifacts = artifacts();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2, true).unwrap();
    runner.set_mtp_requested(true);
    assert!(runner.has_mtp());
    let prompt = [1, 2, 3, 4, 5];
    let ctx = context(501, prompt.len(), 8);
    let output = execute(&runner, ctx, &prompt, ExecutionMode::Prefill);
    assert!(output.request_updates[0].error.is_none());
    assert!(
        runner
            .states
            .lock()
            .get(&ctx.request_id)
            .unwrap()
            .flash_next_mtp
            .cursor
            .is_some()
    );
    runner.release_request_state(ctx.request_id);
    assert!(!runner.states.lock().contains_key(&ctx.request_id));
    let recovered = generate(&runner, &[6, 7], 100, context(502, 2, 3));
    assert_eq!(recovered.tokens.len(), 3);
}

#[test]
#[ignore = "requires a real isolated Flash Next candidate and explicit MTP attachment"]
fn flash_next_real_runner_mtp_matches_recorded_resident_control() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let expected: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_EXPECTED_IDS").unwrap()).unwrap();
    assert!((3..=8).contains(&expected.len()));
    let started = Instant::now();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    let load_seconds = started.elapsed().as_secs_f64();
    assert!(runner.has_mtp());
    runner.set_mtp_requested(true);
    assert!(runner.mtp_requested());
    let generation_started = Instant::now();
    // The resident record used server-default 16-token blocks. Four-token
    // blocks intentionally split this five-token prompt at a recurrent prefix
    // snapshot boundary (3+1+1), changing quantized arithmetic from 4+1.
    let result = generate_with_block_size(
        &runner,
        &prompt,
        usize::MAX,
        context(701, prompt.len(), expected.len() as u32),
        16,
    );
    let tie_margin = mtp_parity::mtp_tie_margin();
    let margin = result.min_correction_margin().unwrap_or(f32::INFINITY);
    let identity =
        mtp_parity::greedy_identity_until_tie(&expected, &result.tokens, margin, tie_margin)
            .unwrap_or_else(|error| panic!("{error}"));
    let greedy_identity = identity.greedy_identity;
    let trunk = runner.weights.qwen4_exp.as_ref().unwrap();
    let admitted = runner_identity_admitted(&trunk.target_schedule, &identity);
    let evidence = serde_json::json!({
        "qualification":false, "route":"production_flash_next_mtp_candidate",
        "target_schedule": crate::model::qwen4_exp_mtp::target_schedule_name(trunk),
        "block_size_tokens":16,
        "load_seconds":load_seconds,"generation_seconds":generation_started.elapsed().as_secs_f64(),
        "tokens":result.tokens,"expected_ids":expected,"routes":result.routes,
        "greedy_identity": greedy_identity,
        "identity_until_first_tie": identity.identity_until_first_tie,
        "tie_divergences": identity
            .tie_divergences
            .iter()
            .map(mtp_parity::TieDivergence::to_json)
            .collect::<Vec<_>>(),
        "within_tolerance": admitted,
        "mlx_peak_bytes":mlx_sys::get_peak_memory(),
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_RESULT_PATH") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
    eprintln!("{evidence}");
    assert!(
        admitted,
        "MTP greedy identity failed the selected target schedule contract"
    );
    assert!(result.maximum("ax_mlx_flash_next_mtp_verified_steps") > 0);
    assert_eq!(result.maximum("ax_mlx_flash_next_mtp_step_errors"), 0);
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts and explicit candidate attachment"]
fn flash_next_runner_mtp_acceptance_budget_and_prefix_fallback() {
    let artifacts = artifacts();
    let mut weights = crate::weights::load_weights(&artifacts).unwrap();
    let shape = [
        artifacts.manifest().vocab_size as i32,
        artifacts.manifest().hidden_size as i32,
    ];
    let data = vec![0.0f32; (shape[0] * shape[1]) as usize];
    let head = crate::weights::QuantizedWeight::new(
        MlxArray::from_raw_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data.as_slice()),
            &shape,
            MlxDtype::Float32,
        ),
        None,
        None,
    );
    weights.lm_head = head.clone();
    weights.qwen4_exp.as_mut().unwrap().lm_head = head.clone();
    weights.qwen4_exp_mtp.as_mut().unwrap().graph.lm_head = head;
    let shared = MlxSharedWeightsCell::new();
    shared.publish(Arc::new(weights));
    let mut runner = MlxRunner::from_artifacts_with_runtime_shares(
        &artifacts,
        2,
        true,
        true,
        None,
        Some(&shared),
    )
    .unwrap();
    runner.set_mtp_requested(true);
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let first = generate(&runner, &prompt, 100, context(801, prompt.len(), 6));
    assert_eq!(first.tokens, vec![0; 6]);
    assert_eq!(first.maximum("ax_mlx_flash_next_mtp_accepted_steps"), 2);
    assert_eq!(first.maximum("ax_mlx_flash_next_mtp_verified_steps"), 3);
    let repeated = generate(&runner, &prompt, 100, context(802, prompt.len(), 6));
    assert_eq!(repeated.tokens, first.tokens);
    assert_eq!(repeated.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
    assert!(repeated.maximum("ax_mlx_flash_next_mtp_resumed_without_cursor") > 0);
    assert!(repeated.maximum("ax_mlx_flash_next_mtp_direct_fallback_steps") > 0);
    assert!(repeated.maximum("ax_mtp_direct_fallback_steps") > 0);
    assert_eq!(
        repeated.maximum("ax_mlx_flash_next_mtp_cursor_initialized"),
        0
    );
}

#[test]
#[ignore = "requires synthetic Flash Next MTP artifacts and explicit candidate attachment"]
fn flash_next_runner_mtp_terminal_and_processor_fallback_match_direct() {
    let artifacts = artifacts();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2, true).unwrap();
    let prompt = [1, 2, 3, 4, 5, 6, 7];
    runner.set_mtp_requested(false);
    let reference = generate(&runner, &prompt, 100, context(901, prompt.len(), 8));
    assert_ne!(reference.tokens[0], reference.tokens[1]);
    runner.terminal_token_ids = vec![reference.tokens[1]];
    // A different model ID is not needed: reset the private test stores so
    // this terminal case exercises a live candidate rather than prefix fallback.
    *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 0,
        max_entries: 0,
    });
    *runner.native_prefix_cache.lock() = MlxNativePrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 0,
        max_entries: 0,
    });
    runner.set_mtp_requested(true);
    let ctx = RunnerRequestContext {
        ignore_eos: false,
        ..context(902, prompt.len(), 8)
    };
    let terminal = generate(&runner, &prompt, 100, ctx);
    assert_eq!(terminal.tokens, reference.tokens[..2]);
    assert!(terminal.maximum("ax_mlx_flash_next_mtp_verified_steps") > 0);
    let processor_ctx = RunnerRequestContext {
        repetition_penalty: 1.1,
        ..context(903, prompt.len(), 4)
    };
    runner.set_mtp_requested(false);
    let direct = generate(&runner, &prompt, 100, processor_ctx);
    runner.set_mtp_requested(true);
    let candidate = generate(
        &runner,
        &prompt,
        100,
        RunnerRequestContext {
            request_id: RequestId(904),
            ..processor_ctx
        },
    );
    assert_eq!(candidate.tokens, direct.tokens);
    assert_eq!(candidate.maximum("ax_mlx_flash_next_mtp_verified_steps"), 0);
    assert_eq!(
        candidate.maximum("ax_mlx_flash_next_mtp_cursor_initialized"),
        0
    );
}

#[test]
#[ignore = "requires a real isolated Flash Next candidate and explicit MTP attachment"]
fn flash_next_real_runner_mtp_matches_same_schedule_direct() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    assert!(runner.has_mtp());
    runner.set_mtp_requested(false);
    let direct = generate_with_block_size(
        &runner,
        &prompt,
        usize::MAX,
        context(1001, prompt.len(), 3),
        4,
    );
    *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 64 * 1024 * 1024,
        max_entries: 128,
    });
    *runner.native_prefix_cache.lock() = MlxNativePrefixCache::new(MlxPrefixCachePolicy {
        max_bytes: 64 * 1024 * 1024,
        max_entries: 128,
    });
    runner.set_mtp_requested(true);
    let candidate = generate_with_block_size(
        &runner,
        &prompt,
        usize::MAX,
        context(1002, prompt.len(), 3),
        4,
    );
    let layers = runner.weights.qwen4_exp.as_ref().unwrap().layers.len();
    let dtype = runner
        .weights
        .qwen4_exp
        .as_ref()
        .unwrap()
        .token_embedding
        .weight
        .dtype();
    let direct_prefill = direct.prefill_state.as_ref().unwrap();
    let mtp_prefill = candidate.prefill_state.as_ref().unwrap();
    // Byte 8 of the AXKB snapshot is seq_len (u64 LE). MTP prefill is n-1
    // plus a singleton of the last prompt token and stays at prompt.len():
    // a live draft cursor owns the first decode step, so
    // `initialize_generation_state` skips `start_direct_pipeline`. Direct
    // greedy prefill uses the same n-1 / singleton split, then primes the
    // mlx_lm-style double-buffer with the first generated token whenever
    // max_output > 1. That lookahead is independent of prompt length and of
    // `block_size_tokens` (see
    // `flash_next_direct_prefill_snapshot_is_prompt_plus_pipeline_bootstrap`).
    // Historical affine batched verification allowed arithmetic differences
    // at the same position. Canonical verification requires exact identity at
    // aligned consumed prefixes, but these lookahead snapshots are not aligned.
    // Positions are compared against the production `run()` contract.
    assert_eq!(
        mtp_prefill.position(),
        flash_next_run_prefill_snapshot_len(prompt.len(), 3, true),
        "MTP prefill snapshot seq_len (AXKB byte 8) must equal the prompt"
    );
    assert_eq!(
        direct_prefill.position(),
        flash_next_run_prefill_snapshot_len(prompt.len(), 3, false),
        "direct prefill snapshot seq_len (AXKB byte 8) includes the pipeline bootstrap token"
    );
    let mtp_bytes = flash_runner_state_bytes(mtp_prefill, layers);
    let direct_bytes = flash_runner_state_bytes(direct_prefill, layers);
    assert_eq!(
        &mtp_bytes[8..16],
        &(mtp_prefill.position() as u64).to_le_bytes()
    );
    assert_eq!(
        &direct_bytes[8..16],
        &(direct_prefill.position() as u64).to_le_bytes()
    );
    // Recurrent QSA/GDN/PLE state is not comparable across the extra
    // generated token the direct pipeline already committed.
    let same_position = mtp_prefill.position() == direct_prefill.position();
    let prefill_records = if same_position {
        mtp_parity::mtp_state_array_records(mtp_prefill, direct_prefill)
    } else {
        Vec::new()
    };
    mtp_parity::eprint_mtp_state_array_table("runner prefill", &prefill_records);
    let prefill_divergence = prefill_records
        .iter()
        .map(|record| record.divergence)
        .fold(mtp_parity::MtpDivergence::zero(), |a, b| a.max_relative(b));
    let identity = runner_greedy_identity(&direct, &candidate, &runner);
    let greedy_identity = identity.greedy_identity;
    let trunk = runner.weights.qwen4_exp.as_ref().unwrap();
    let canonical = matches!(
        trunk.target_schedule,
        Qwen4ExpTargetSchedule::CanonicalSingleton
    );
    let admitted = runner_identity_admitted(&trunk.target_schedule, &identity);
    // The runner drops request state on the terminal step, so decode-state
    // divergence is measured by the CandidateSession controls, not here.
    let tolerance = mtp_parity::mtp_run_tolerance(dtype);
    if same_position {
        if canonical {
            assert_eq!(mtp_bytes, direct_bytes);
        } else {
            mtp_parity::assert_mtp_state_close(mtp_prefill, direct_prefill, dtype);
        }
    }
    let within_tolerance = if same_position {
        admitted
            && if canonical {
                mtp_bytes == direct_bytes
            } else {
                prefill_divergence.relative <= tolerance.limit
            }
    } else {
        admitted
    };
    let evidence = serde_json::json!({
        "qualification":false,"prompt_ids":prompt,"block_size_tokens":4,"direct_ids":direct.tokens,
        "target_schedule": crate::model::qwen4_exp_mtp::target_schedule_name(trunk),
        "mtp_ids":candidate.tokens,"direct_routes":direct.routes,"mtp_routes":candidate.routes,
        "greedy_identity": greedy_identity,
        "identity_until_first_tie": identity.identity_until_first_tie,
        "tie_divergences": identity
            .tie_divergences
            .iter()
            .map(mtp_parity::TieDivergence::to_json)
            .collect::<Vec<_>>(),
        "prefill_state_exact": canonical && same_position && mtp_bytes == direct_bytes,
        "prefill_state_compared": same_position,
        "prefill_state_byte_exact_is_non_goal": !canonical,
        "prefill_seq_len_field": "AXKB header seq_len u64 LE at bytes 8..16",
        "prefill_position": mtp_prefill.position(),
        "direct_prefill_position": direct_prefill.position(),
        "direct_prefill_includes_pipeline_bootstrap_token": true,
        "decode_state_compared": false,
        "logit_scale": serde_json::Value::Null,
        "max_logit_abs_difference": serde_json::Value::Null,
        "max_logit_relative_divergence": serde_json::Value::Null,
        "max_state_abs_difference": prefill_divergence.max_abs,
        "max_state_relative_divergence": prefill_divergence.relative,
        "tolerance": tolerance.limit,
        "tolerance_source": tolerance.source,
        "tie_margin": mtp_parity::mtp_tie_margin(),
        "state_tolerance": tolerance.limit,
        "within_tolerance": within_tolerance,
        "state_arrays": mtp_parity::mtp_state_arrays_json(&prefill_records),
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_RESULT_PATH") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
    assert!(
        admitted,
        "MTP greedy identity failed the selected target schedule contract"
    );
    assert!(candidate.maximum("ax_mlx_flash_next_mtp_verified_steps") > 0);
}

#[test]
#[ignore = "requires a real Flash Next pack and explicit MTP attachment; records paired timings"]
fn flash_next_real_runner_mtp_paired_cost() {
    let root = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_REAL_PACK").unwrap());
    let artifacts = NativeModelArtifacts::from_dir(&root).unwrap();
    let prompt: Vec<u32> =
        serde_json::from_str(&std::env::var("AX_FLASH_NEXT_PROMPT_IDS").unwrap()).unwrap();
    assert!(!prompt.is_empty() && prompt.len() <= 128);
    let output_path = PathBuf::from(std::env::var_os("AX_FLASH_NEXT_RESULT_PATH").unwrap());
    let load_started = Instant::now();
    let mut runner = MlxRunner::from_artifacts(&artifacts, 2048, true).unwrap();
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert!(runner.has_mtp());
    assert!(!runner.mtp_model_policy.certified_default_on());
    let pair_count = std::env::var("AX_FLASH_NEXT_PAIRED_PAIRS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(14usize);
    assert!(
        pair_count >= 12,
        "paired cost needs at least 12 pairs, got {pair_count}"
    );
    let warmup_pairs = 2usize;
    let mut evidence = serde_json::json!({
        "qualification": false,
        "target_schedule": crate::model::qwen4_exp_mtp::target_schedule_name(runner.weights.qwen4_exp.as_ref().unwrap()),
        "kind": "paired_production_runner_cost",
        "completed": false,
        "prompt_ids": prompt,
        "output_budget": 32,
        "block_size_tokens": 16,
        "prefill_quantum": "whole_prompt",
        "load_seconds": load_seconds,
        "pair_count": pair_count,
        "warmup_pairs": warmup_pairs,
        "head_attached_in_both_modes": true,
        "prefix_stores": "cleared_before_every_request",
        "timing_boundary": "runner_call_with_host_visible_tokens",
        "limitations": [
            "A single short prompt does not qualify MTP profitability.",
            "Direct control includes attached-head memory; it does not measure head loading cost.",
            "No mlx_lm baseline or independent trained-head oracle is supplied.",
            "Prefix resets and per-step state-alignment assertions are outside or between runner calls."
        ],
        "samples": [],
    });
    let mut direct_reference: Option<Vec<u32>> = None;
    let mut mtp_reference: Option<Vec<u32>> = None;
    let mut last_direct: Option<Generation> = None;
    let mut last_mtp: Option<Generation> = None;
    let mut pair_identity = mtp_parity::GreedyIdentityReport::exact();
    // Two warmup pairs, then at least twelve measured pairs. Reverse order on
    // alternate pairs to avoid assigning all later, warmer requests to one mode.
    for pair in 0..pair_count {
        for mode in 0..2usize {
            let candidate = (pair + mode) % 2 == 1;
            *runner.prefix_cache.lock() = MlxPrefixCache::new(MlxPrefixCachePolicy {
                max_bytes: 64 * 1024 * 1024,
                max_entries: 128,
            });
            *runner.native_prefix_cache.lock() = MlxNativePrefixCache::new(MlxPrefixCachePolicy {
                max_bytes: 64 * 1024 * 1024,
                max_entries: 128,
            });
            runner.set_mtp_requested(candidate);
            let started = Instant::now();
            let result = generate_with_block_size(
                &runner,
                &prompt,
                usize::MAX,
                context(1100 + pair as u64 * 2 + mode as u64, prompt.len(), 32),
                16,
            );
            let total_seconds = started.elapsed().as_secs_f64();
            if candidate {
                let reference = mtp_reference.get_or_insert_with(|| result.tokens.clone());
                assert_eq!(
                    result.tokens, *reference,
                    "paired MTP output changed at pair={pair}"
                );
                last_mtp = Some(result.clone());
            } else {
                let reference = direct_reference.get_or_insert_with(|| result.tokens.clone());
                assert_eq!(
                    result.tokens, *reference,
                    "paired direct output changed at pair={pair}"
                );
                last_direct = Some(result.clone());
            }
            if let (Some(direct_run), Some(mtp_run)) = (&last_direct, &last_mtp) {
                pair_identity = runner_greedy_identity(direct_run, mtp_run, &runner);
            }
            let verified = result.maximum("ax_mlx_flash_next_mtp_verified_steps");
            let errors = result.maximum("ax_mlx_flash_next_mtp_step_errors");
            evidence["samples"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "pair": pair, "warmup": pair < warmup_pairs,
                    "mtp_requested": candidate,
                    "total_seconds": total_seconds,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "generated_ids": result.tokens,
                    "token_parity": pair_identity.greedy_identity,
                    "greedy_identity": pair_identity.greedy_identity,
                    "identity_until_first_tie": pair_identity.identity_until_first_tie,
                    "tie_divergences": pair_identity
                        .tie_divergences
                        .iter()
                        .map(mtp_parity::TieDivergence::to_json)
                        .collect::<Vec<_>>(),
                    "min_correction_margin": result.min_correction_margin(),
                    "routes": result.routes,
                    "verified_steps": verified,
                    "accepted": result.maximum("ax_mlx_flash_next_mtp_accepted_steps"),
                    "emitted_tokens": result.maximum("ax_mlx_flash_next_mtp_emitted_tokens"),
                    "correction_wall_us": result.maximum("ax_mlx_flash_next_mtp_correction_wall_us"),
                    "bonus_wall_us": result.maximum("ax_mlx_flash_next_mtp_bonus_wall_us"),
                    "rejection_wall_us": result.maximum("ax_mlx_flash_next_mtp_rejection_wall_us"),
                    "verify_wall_us": result.maximum("ax_mtp_verify_forward_wall_us"),
                    "mlx_buffer_cache_bytes": mlx_sys::get_cache_memory(),
                    "mlx_peak_bytes": mlx_sys::get_peak_memory(),
                }));
            std::fs::write(&output_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
            eprintln!(
                "pair={pair} candidate={candidate} total_seconds={total_seconds:.6} identity_until_first_tie={}",
                pair_identity.identity_until_first_tie
            );
            assert!(
                runner_identity_admitted(
                    &runner.weights.qwen4_exp.as_ref().unwrap().target_schedule,
                    &pair_identity
                ),
                "MTP greedy identity failed the selected target schedule contract at pair={pair} candidate={candidate}"
            );
            assert_eq!(result.tokens.len(), 32);
            assert_eq!(errors, 0);
            assert_eq!(verified > 0, candidate);
        }
    }
    evidence["completed"] = true.into();
    evidence["greedy_identity_all_requests"] = runner_identity_admitted(
        &runner.weights.qwen4_exp.as_ref().unwrap().target_schedule,
        &pair_identity,
    )
    .into();
    evidence["identity_until_first_tie"] = pair_identity.identity_until_first_tie.into();
    evidence["tie_divergences"] = serde_json::json!(
        pair_identity
            .tie_divergences
            .iter()
            .map(mtp_parity::TieDivergence::to_json)
            .collect::<Vec<_>>()
    );
    evidence["tie_count"] = pair_identity.tie_divergences.len().into();
    std::fs::write(output_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
}
