//! Load-time kernel warm-up TTFT probe (the P1-B measurement gate).
//!
//! Loads a model through the production `MlxRunner` constructor — the path
//! that runs the load-time Metal kernel warm-up — then drives one greedy
//! request (a single prefill `run()` followed by per-step decode `run()`
//! calls) and reports wall-clock timings as JSON. Compare a run with the
//! warm-up enabled (default) against `AX_MLX_LOAD_KERNEL_WARMUP=0` to
//! measure the first-request latency the warm-up removes.
//!
//! Usage:
//!   cargo run --release --bin load_warmup_ttft_probe -- <model_dir>
//! Env: AX_PROMPT_LEN (default 64), AX_GEN (default 8), AX_PROMPT_SEED
//! (default 0).

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

const REQUEST: u64 = 1;

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => Ok(default),
    }
}

fn greedy_ctx(prompt_len: usize, generated_len: usize, max_output: usize) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(REQUEST),
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: prompt_len as u32,
        generated_len: generated_len as u32,
        max_output_tokens: max_output as u32,
        seed: 1,
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
    }
}

fn single_item_input(
    step: u64,
    mode: ExecutionMode,
    tokens: Vec<u32>,
    start: usize,
    context: RunnerRequestContext,
) -> RunnerInput {
    let count = tokens.len() as u32;
    RunnerInput {
        block_size_tokens: 16,
        memory_pressure: None,
        execution_batch: ExecutionBatch {
            step_id: StepId(step),
            model_id: "ttft-probe".into(),
            execution_plan_ref: None,
            items: vec![ExecutionItem {
                request_id: RequestId(REQUEST),
                mode,
                planned_work_unit: ax_engine_core::work_unit_for_execution_mode(mode),
                input_token_slice: tokens,
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: start as u32,
                    end_exclusive: start as u32 + count,
                },
                scheduled_token_count: count,
                block_table_ref: RequestId(REQUEST),
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: count,
            route_metadata: RouteMetadata::empty(),
        },
        block_tables: Vec::new(),
        request_contexts: vec![context],
        request_multimodal_inputs: Vec::new(),
    }
}

fn first_output_token(runner: &MlxRunner, input: RunnerInput) -> Result<u32, String> {
    let output = runner.run(input);
    output
        .request_updates
        .iter()
        .find(|update| update.request_id == RequestId(REQUEST))
        .and_then(|update| {
            update
                .output_token
                .or_else(|| update.output_tokens.first().copied())
        })
        .ok_or_else(|| "runner step produced no output token".to_string())
}

fn run() -> Result<ExitCode, String> {
    let model_dir = env::args()
        .nth(1)
        .ok_or_else(|| "usage: load_warmup_ttft_probe <model_dir>".to_string())?;
    let prompt_len = env_usize("AX_PROMPT_LEN", 64)?;
    let gen_len = env_usize("AX_GEN", 8)?;
    let prompt_seed = env_usize("AX_PROMPT_SEED", 0)?;
    if prompt_len == 0 || gen_len == 0 {
        return Err("AX_PROMPT_LEN and AX_GEN must be greater than zero".to_string());
    }

    let warmup_enabled = ax_engine_mlx::fastpath::load_kernel_warmup_enabled();

    let load_started = Instant::now();
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut runner = MlxRunner::from_artifacts(&artifacts, DEFAULT_PREFILL_CHUNK, true)
        .map_err(|error| format!("failed to create runner: {error}"))?;
    runner.set_mtp_requested(false);
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;

    let modulus = (cfg.vocab_size.saturating_sub(1)).max(1) as u128;
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|index| ((prompt_seed as u128 + index as u128 * 7 + 3) % modulus + 1) as u32)
        .collect();
    let max_output = gen_len + 4;

    let mut step_id = 0u64;
    let prefill_started = Instant::now();
    let mut token = first_output_token(
        &runner,
        single_item_input(
            step_id,
            ExecutionMode::Prefill,
            prompt.clone(),
            0,
            greedy_ctx(prompt_len, 0, max_output),
        ),
    )?;
    let ttft_prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;

    let mut decode_step_ms = Vec::with_capacity(gen_len);
    for generated in 1..=gen_len {
        step_id += 1;
        let step_started = Instant::now();
        token = first_output_token(
            &runner,
            single_item_input(
                step_id,
                ExecutionMode::Decode,
                vec![token],
                prompt_len + generated - 1,
                greedy_ctx(prompt_len, generated, max_output),
            ),
        )?;
        decode_step_ms.push(step_started.elapsed().as_secs_f64() * 1e3);
    }

    let report = serde_json::json!({
        "model_family": cfg.model_family,
        "warmup_enabled": warmup_enabled,
        "prompt_len": prompt_len,
        "load_ms": load_ms,
        "ttft_prefill_ms": ttft_prefill_ms,
        "decode_step_ms": decode_step_ms,
    });
    println!("{report}");
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("load_warmup_ttft_probe: {message}");
            ExitCode::FAILURE
        }
    }
}
