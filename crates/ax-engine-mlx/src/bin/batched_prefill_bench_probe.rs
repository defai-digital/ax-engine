//! Padded batched prefill benchmark — the P3 completion-condition probe.
//!
//! Reports, in one run: per-request TTFT, the decode gap a running request
//! experiences while other requests prefill, aggregate decode throughput,
//! and MLX peak memory. Drive it twice (default flags, then
//! `AX_MLX_BATCHED_PREFILL=1`) to compare sequential vs padded batched
//! prefill on the same workload; `batched_prefill_rows` in the output
//! proves whether the batched path actually engaged.
//!
//! Workload shape:
//! 1. Request 0 prefills alone, then decodes a few tokens — its median
//!    inter-token latency is the decode-gap baseline.
//! 2. One mixed step: request 0's next decode item shares the step with
//!    `AX_BATCH - 1` fresh prefill items. The step wall time is request 0's
//!    decode gap and the joiners' TTFT.
//! 3. All requests decode together for `AX_GEN` tokens — aggregate
//!    throughput.
//!
//! Usage:
//!   cargo run --release --bin batched_prefill_bench_probe -- <model_dir>
//! Env: AX_BATCH (default 4), AX_PROMPT_LEN (default 384), AX_GEN
//! (default 12), AX_PROMPT_SEED (default 0).

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::{
    ExecutionBatch, ExecutionItem, ExecutionMode, ExecutionRunner, NativeModelArtifacts,
    PositionRange, RequestId, RouteMetadata, RunnerInput, StepId,
};
use ax_engine_mlx::{MlxRunner, generate::DEFAULT_PREFILL_CHUNK, model::ModelConfig};

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => Ok(default),
    }
}

fn greedy_ctx(
    request: u64,
    prompt_len: usize,
    generated_len: usize,
    max_output: usize,
) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(request),
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: prompt_len as u32,
        generated_len: generated_len as u32,
        max_output_tokens: max_output as u32,
        seed: 1 + request,
        deterministic_argmax_sampling: true,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: true,
        tool_call_mode: false,
        structured_output_mode: false,
        min_p: None,
        max_think_tokens: None,
        answer_reserve_tokens: None,
    }
}

fn item(request: u64, mode: ExecutionMode, tokens: Vec<u32>, start: usize) -> ExecutionItem {
    let count = tokens.len() as u32;
    ExecutionItem {
        request_id: RequestId(request),
        mode,
        planned_work_unit: ax_engine_core::work_unit_for_execution_mode(mode),
        input_token_slice: tokens,
        reused_prefix_token_slice: Vec::new(),
        position_range: PositionRange {
            start: start as u32,
            end_exclusive: start as u32 + count,
        },
        scheduled_token_count: count,
        block_table_ref: RequestId(request),
        prefix_tokens_reused: 0,
        prefix_blocks_reused: 0,
    }
}

fn runner_input(
    step: u64,
    items: Vec<ExecutionItem>,
    contexts: Vec<RunnerRequestContext>,
) -> RunnerInput {
    let total: u32 = items.iter().map(|item| item.scheduled_token_count).sum();
    RunnerInput {
        block_size_tokens: 16,
        memory_pressure: None,
        execution_batch: ExecutionBatch {
            step_id: StepId(step),
            model_id: "batched-prefill-bench".into(),
            execution_plan_ref: None,
            items,
            total_scheduled_tokens: total,
            route_metadata: RouteMetadata::empty(),
        },
        block_tables: Vec::new(),
        request_contexts: contexts,
        request_multimodal_inputs: Vec::new(),
    }
}

struct StepResult {
    tokens: Vec<(u64, u32)>,
    batched_prefill_rows: u32,
    wall_ms: f64,
}

fn run_step(runner: &MlxRunner, input: RunnerInput) -> Result<StepResult, String> {
    let started = Instant::now();
    let output = runner.run(input);
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut tokens = Vec::new();
    for update in &output.request_updates {
        if let Some(error) = &update.error {
            return Err(format!("request {} failed: {error}", update.request_id.0));
        }
        let token = update
            .output_token
            .or_else(|| update.output_tokens.first().copied())
            .ok_or_else(|| format!("request {} produced no token", update.request_id.0))?;
        tokens.push((update.request_id.0, token));
    }
    let batched_prefill_rows = output
        .route_metadata
        .crossover_decisions
        .iter()
        .find(|(key, _)| key == "ax_mlx_batched_prefill_rows")
        .map(|(_, value)| *value)
        .unwrap_or(0);
    Ok(StepResult {
        tokens,
        batched_prefill_rows,
        wall_ms,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    values[values.len() / 2]
}

fn run() -> Result<ExitCode, String> {
    let model_dir = env::args()
        .nth(1)
        .ok_or_else(|| "usage: batched_prefill_bench_probe <model_dir>".to_string())?;
    let batch = env_usize("AX_BATCH", 4)?;
    let prompt_len = env_usize("AX_PROMPT_LEN", 384)?;
    let gen_len = env_usize("AX_GEN", 12)?;
    let prompt_seed = env_usize("AX_PROMPT_SEED", 0)?;
    if batch < 2 || prompt_len == 0 || gen_len == 0 {
        return Err("AX_BATCH must be >= 2 and AX_PROMPT_LEN/AX_GEN positive".to_string());
    }
    let baseline_decode = 4usize;
    let max_output = gen_len + baseline_decode + 4;

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut runner = MlxRunner::from_artifacts(&artifacts, DEFAULT_PREFILL_CHUNK, true)
        .map_err(|error| format!("failed to create runner: {error}"))?;
    runner.set_mtp_requested(false);
    mlx_sys::reset_peak_memory();

    let modulus = (cfg.vocab_size.saturating_sub(1)).max(1) as u128;
    // AX_RAGGED=1 shortens each joiner differently so the padded batched
    // path really pads (equal-length cohorts never exercise the mask).
    let ragged = env::var_os("AX_RAGGED").is_some();
    let prompt = |request: usize| -> Vec<u32> {
        let row_len = if ragged && request > 0 {
            prompt_len
                .saturating_sub(request * 29)
                .max(prompt_len / 2)
                .max(1)
        } else {
            prompt_len
        };
        (0..row_len)
            .map(|index| {
                ((prompt_seed as u128 + request as u128 * 31 + index as u128 * 7 + 3) % modulus + 1)
                    as u32
            })
            .collect()
    };

    let mut step_id = 0u64;
    let mut next_step = |items, contexts| {
        let input = runner_input(step_id, items, contexts);
        step_id += 1;
        input
    };

    // Stage 1: request 0 prefills alone and decodes a baseline.
    let prompt0 = prompt(0);
    let stage1_prefill = run_step(
        &runner,
        next_step(
            vec![item(0, ExecutionMode::Prefill, prompt0.clone(), 0)],
            vec![greedy_ctx(0, prompt_len, 0, max_output)],
        ),
    )?;
    let mut current: Vec<u32> = vec![stage1_prefill.tokens[0].1];
    let mut generated0 = 1usize;
    let mut baseline_itl = Vec::new();
    for _ in 0..baseline_decode {
        let step = run_step(
            &runner,
            next_step(
                vec![item(
                    0,
                    ExecutionMode::Decode,
                    vec![current[0]],
                    prompt_len + generated0 - 1,
                )],
                vec![greedy_ctx(0, prompt_len, generated0, max_output)],
            ),
        )?;
        current[0] = step.tokens[0].1;
        generated0 += 1;
        baseline_itl.push(step.wall_ms);
    }

    // Stage 2: one mixed step — request 0 decodes while the others prefill.
    let mut items = vec![item(
        0,
        ExecutionMode::Decode,
        vec![current[0]],
        prompt_len + generated0 - 1,
    )];
    let mut contexts = vec![greedy_ctx(0, prompt_len, generated0, max_output)];
    for request in 1..batch {
        let tokens = prompt(request);
        let row_len = tokens.len();
        items.push(item(request as u64, ExecutionMode::Prefill, tokens, 0));
        contexts.push(greedy_ctx(request as u64, row_len, 0, max_output));
    }
    let mixed = run_step(&runner, next_step(items, contexts))?;
    generated0 += 1;
    let mut generated = vec![1usize; batch];
    generated[0] = generated0;
    current = vec![0; batch];
    for &(request, token) in &mixed.tokens {
        current[request as usize] = token;
    }

    // Stage 3: everyone decodes together.
    let prompt_lens: Vec<usize> = (0..batch).map(|request| prompt(request).len()).collect();
    let stage3_started = Instant::now();
    let mut stage3_steps = Vec::new();
    for _ in 0..gen_len {
        let items = (0..batch)
            .map(|request| {
                item(
                    request as u64,
                    ExecutionMode::Decode,
                    vec![current[request]],
                    prompt_lens[request] + generated[request] - 1,
                )
            })
            .collect();
        let contexts = (0..batch)
            .map(|request| {
                greedy_ctx(
                    request as u64,
                    prompt_lens[request],
                    generated[request],
                    max_output,
                )
            })
            .collect();
        let step = run_step(&runner, next_step(items, contexts))?;
        for &(request, token) in &step.tokens {
            current[request as usize] = token;
            generated[request as usize] += 1;
        }
        stage3_steps.push(step.wall_ms);
    }
    let stage3_wall_s = stage3_started.elapsed().as_secs_f64();

    let mut baseline = baseline_itl.clone();
    let mut stage3_sorted = stage3_steps.clone();
    let report = serde_json::json!({
        "model_family": cfg.model_family,
        "batch": batch,
        "prompt_len": prompt_len,
        "batched_prefill_flag": ax_engine_mlx::fastpath::batched_prefill_enabled(),
        "batched_prefill_rows_engaged": mixed.batched_prefill_rows,
        "ttft_first_request_ms": stage1_prefill.wall_ms,
        "ttft_joining_requests_ms": mixed.wall_ms,
        "decode_gap_during_joining_prefill_ms": mixed.wall_ms,
        "baseline_itl_ms": median(&mut baseline),
        "batched_decode_itl_ms": median(&mut stage3_sorted),
        "aggregate_decode_tokens_per_s": (batch * gen_len) as f64 / stage3_wall_s,
        "peak_memory_mb": mlx_sys::get_peak_memory() as f64 / (1024.0 * 1024.0),
        "final_tokens": current,
    });
    println!("{report}");
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("batched_prefill_bench_probe: {message}");
            ExitCode::FAILURE
        }
    }
}
