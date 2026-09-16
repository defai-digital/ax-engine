//! Artifact-backed tests of the actual scheduler-facing Flash Next MTP route.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use ax_engine_core::{
    ExecutionBatch, ExecutionItem, PositionRange, RouteMetadata, StepId, WorkUnitKind,
};
use std::path::PathBuf;

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

struct Generation {
    tokens: Vec<u32>,
    routes: Vec<Vec<(String, u32)>>,
    prefill_seconds: f64,
    decode_seconds: f64,
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
    mut ctx: RunnerRequestContext,
    block_size: u32,
) -> Generation {
    let mut result = Generation {
        tokens: Vec::new(),
        routes: Vec::new(),
        prefill_seconds: 0.0,
        decode_seconds: 0.0,
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
        if let Some(state) = runner.states.lock().get(&ctx.request_id)
            && let Some(cursor) = &state.flash_next_mtp.cursor
        {
            let trunk = state.cache.qwen4_exp.as_ref().unwrap();
            assert_eq!(trunk.position(), state.cache.seq_len());
            assert!(cursor.aligned(trunk));
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
        "run with AX_MLX_FLASH_NEXT_MTP_CANDIDATE=1"
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
    let evidence = serde_json::json!({
        "qualification":false, "route":"production_flash_next_mtp_candidate",
        "block_size_tokens":16,
        "load_seconds":load_seconds,"generation_seconds":generation_started.elapsed().as_secs_f64(),
        "tokens":result.tokens,"expected_ids":expected,"routes":result.routes,
        "mlx_peak_bytes":mlx_sys::get_peak_memory(),
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_RESULT_PATH") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
    eprintln!("{evidence}");
    assert_eq!(result.tokens, expected);
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
    let evidence = serde_json::json!({
        "qualification":false,"block_size_tokens":4,"direct_ids":direct.tokens,
        "mtp_ids":candidate.tokens,"direct_routes":direct.routes,"mtp_routes":candidate.routes,
    });
    if let Some(path) = std::env::var_os("AX_FLASH_NEXT_RESULT_PATH") {
        std::fs::write(path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
    }
    assert_eq!(candidate.tokens, direct.tokens);
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
    let mut evidence = serde_json::json!({
        "qualification": false,
        "kind": "paired_production_runner_cost",
        "completed": false,
        "prompt_ids": prompt,
        "output_budget": 32,
        "block_size_tokens": 16,
        "prefill_quantum": "whole_prompt",
        "load_seconds": load_seconds,
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
    let mut expected = None;
    // Pair zero warms both routes. Reverse order on alternate pairs to avoid
    // assigning all later, warmer requests to one mode.
    for pair in 0..7 {
        for mode in 0..2 {
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
                context(1100 + pair * 2 + mode, prompt.len(), 32),
                16,
            );
            let total_seconds = started.elapsed().as_secs_f64();
            let reference = expected.get_or_insert_with(|| result.tokens.clone());
            let parity = result.tokens == *reference;
            let verified = result.maximum("ax_mlx_flash_next_mtp_verified_steps");
            let errors = result.maximum("ax_mlx_flash_next_mtp_step_errors");
            evidence["samples"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({
                    "pair": pair, "warmup": pair == 0,
                    "mtp_requested": candidate,
                    "total_seconds": total_seconds,
                    "prefill_seconds": result.prefill_seconds,
                    "decode_seconds": result.decode_seconds,
                    "generated_ids": result.tokens,
                    "token_parity": parity,
                    "routes": result.routes,
                    "verified_steps": verified,
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
                "pair={pair} candidate={candidate} total_seconds={total_seconds:.6} parity={parity}"
            );
            assert!(
                parity,
                "paired runner output changed at pair={pair} candidate={candidate}"
            );
            assert_eq!(result.tokens.len(), 32);
            assert_eq!(errors, 0);
            assert_eq!(verified > 0, candidate);
        }
    }
    evidence["completed"] = true.into();
    std::fs::write(output_path, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
}
