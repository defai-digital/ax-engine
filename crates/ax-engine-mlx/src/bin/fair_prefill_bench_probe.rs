//! Scheduler-split prefill correctness and TTFT probe.
//!
//! Drives the production [`MlxRunner`] with one prompt split across multiple
//! prefill execution items. This is deliberately below the service scheduler:
//! it isolates the cost and sampling semantics of the runner boundary without
//! adding HTTP, tokenization, or arbiter noise.
//!
//! Run the same model and seed twice:
//!
//! ```text
//! AX_PREFILL_QUANTUM=0  cargo run --release --bin fair_prefill_bench_probe -- <model>
//! AX_PREFILL_QUANTUM=64 cargo run --release --bin fair_prefill_bench_probe -- <model>
//! ```
//!
//! `AX_PREFILL_QUANTUM=0` submits the whole prompt in one item. Greedy runs use
//! `output_tokens` as a cross-quantum exactness check. Sampled runs are compared
//! at a fixed quantum: changing the chunk shape can legitimately perturb bf16
//! logits and therefore the categorical CDF, but non-terminal items must never
//! consume RNG or sample. `cache_only_continuations_per_prompt` proves that
//! continuation route engaged.
//!
//! Environment:
//! - `AX_PROMPT_LEN` (default 2048)
//! - `AX_PREFILL_QUANTUM` (default 64; 0 means the whole prompt)
//! - `AX_WARMUPS` (default 1)
//! - `AX_REPETITIONS` (default 5)
//! - `AX_PROMPT_SEED` (default 0)
//! - `AX_SAMPLING_SEED` (default 42)
//! - `AX_TEMPERATURE` (default 0.8)
//! - `AX_TOP_P` (default 0.95)
//! - `AX_TOP_K` (default 20)
//!
//! For stable cold-prefill measurements, set
//! `AX_MLX_PREFIX_CACHE_MAX_ENTRIES=0`.

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

fn env_u64(name: &str, default: u64) -> Result<u64, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<u64>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => Ok(default),
    }
}

fn env_u32(name: &str, default: u32) -> Result<u32, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<u32>()
            .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}")),
        Err(_) => Ok(default),
    }
}

fn env_f32(name: &str, default: f32) -> Result<f32, String> {
    match env::var(name) {
        Ok(value) => value
            .parse::<f32>()
            .map_err(|_| format!("{name} must be a finite number, got {value:?}"))
            .and_then(|parsed| {
                parsed
                    .is_finite()
                    .then_some(parsed)
                    .ok_or_else(|| format!("{name} must be finite, got {value:?}"))
            }),
        Err(_) => Ok(default),
    }
}

#[derive(Clone, Copy)]
struct SamplingConfig {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    seed: u64,
}

fn request_context(
    request_id: RequestId,
    prompt_len: usize,
    processed_prompt_tokens: usize,
    sampling: SamplingConfig,
) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id,
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: processed_prompt_tokens as u32,
        generated_len: 0,
        max_output_tokens: 1,
        seed: sampling.seed,
        deterministic_argmax_sampling: sampling.temperature <= 0.0
            && sampling.top_k == 0
            && sampling.top_p >= 1.0,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: true,
        tool_call_mode: false,
        structured_output_mode: false,
    }
}

/// Run `decode_tokens` single-token decode steps after a completed prefill so
/// repeated-request probes exercise the same decode-then-prefill sequence the
/// serving path produces (`AX_DECODE_TOKENS`). Returns the last sampled token.
fn run_decode_steps(
    runner: &MlxRunner,
    prompt_len: usize,
    first_token: u32,
    decode_tokens: usize,
    request_id: RequestId,
    sampling: SamplingConfig,
    next_step_id: &mut u64,
) -> Result<u32, String> {
    let mut token = first_token;
    for step in 0..decode_tokens {
        let position = prompt_len + step;
        let mut ctx = request_context(request_id, prompt_len, prompt_len, sampling);
        ctx.generated_len = (step + 1) as u32;
        ctx.max_output_tokens = (decode_tokens + 2) as u32;
        let input = RunnerInput {
            block_size_tokens: 16,
            memory_pressure: None,
            execution_batch: ExecutionBatch {
                step_id: StepId(*next_step_id),
                model_id: "fair-prefill-bench-probe".into(),
                execution_plan_ref: None,
                items: vec![ExecutionItem {
                    request_id,
                    mode: ExecutionMode::Decode,
                    planned_work_unit: ax_engine_core::WorkUnitKind::TokenDecode,
                    input_token_slice: vec![token],
                    reused_prefix_token_slice: Vec::new(),
                    position_range: PositionRange {
                        start: position as u32,
                        end_exclusive: position as u32 + 1,
                    },
                    scheduled_token_count: 1,
                    block_table_ref: request_id,
                    prefix_tokens_reused: 0,
                    prefix_blocks_reused: 0,
                }],
                total_scheduled_tokens: 1,
                route_metadata: RouteMetadata::empty(),
            },
            block_tables: Vec::new(),
            request_contexts: vec![ctx],
            request_multimodal_inputs: Vec::new(),
        };
        *next_step_id = next_step_id.saturating_add(1);
        let output = runner.run(input);
        let update = output
            .request_updates
            .iter()
            .find(|update| update.request_id == request_id)
            .ok_or_else(|| format!("decode step for request {} lost", request_id.0))?;
        if let Some(error) = &update.error {
            return Err(format!(
                "decode step for request {} failed: {error}",
                request_id.0
            ));
        }
        token = update
            .output_token
            .or_else(|| update.output_tokens.first().copied())
            .unwrap_or(token);
    }
    Ok(token)
}

fn prefill_input(
    step_id: u64,
    request_id: RequestId,
    prompt_len: usize,
    tokens: Vec<u32>,
    processed_prompt_tokens: usize,
    sampling: SamplingConfig,
) -> RunnerInput {
    let count = tokens.len() as u32;
    RunnerInput {
        block_size_tokens: 16,
        memory_pressure: None,
        execution_batch: ExecutionBatch {
            step_id: StepId(step_id),
            model_id: "fair-prefill-bench-probe".into(),
            execution_plan_ref: None,
            items: vec![ExecutionItem {
                request_id,
                mode: ExecutionMode::Prefill,
                planned_work_unit: ax_engine_core::WorkUnitKind::PrefillChunk,
                input_token_slice: tokens,
                reused_prefix_token_slice: Vec::new(),
                position_range: PositionRange {
                    start: processed_prompt_tokens as u32,
                    end_exclusive: processed_prompt_tokens as u32 + count,
                },
                scheduled_token_count: count,
                block_table_ref: request_id,
                prefix_tokens_reused: 0,
                prefix_blocks_reused: 0,
            }],
            total_scheduled_tokens: count,
            route_metadata: RouteMetadata::empty(),
        },
        block_tables: Vec::new(),
        request_contexts: vec![request_context(
            request_id,
            prompt_len,
            processed_prompt_tokens,
            sampling,
        )],
        request_multimodal_inputs: Vec::new(),
    }
}

struct PrefillResult {
    output_token: u32,
    wall_ms: f64,
    execution_items: usize,
    cache_only_continuations: u32,
}

fn run_prefill(
    runner: &MlxRunner,
    prompt: &[u32],
    quantum: usize,
    request_id: RequestId,
    sampling: SamplingConfig,
    next_step_id: &mut u64,
) -> Result<PrefillResult, String> {
    let started = Instant::now();
    let quantum = if quantum == 0 {
        prompt.len()
    } else {
        quantum.max(1)
    };
    let mut processed = 0usize;
    let mut output_token = None;
    let mut execution_items = 0usize;
    let mut cache_only_continuations = 0u32;

    while processed < prompt.len() {
        let end = processed.saturating_add(quantum).min(prompt.len());
        let input = prefill_input(
            *next_step_id,
            request_id,
            prompt.len(),
            prompt[processed..end].to_vec(),
            processed,
            sampling,
        );
        *next_step_id = next_step_id.saturating_add(1);
        let output = runner.run(input);
        let update = output
            .request_updates
            .iter()
            .find(|update| update.request_id == request_id)
            .ok_or_else(|| format!("request {} produced no runner update", request_id.0))?;
        if let Some(error) = &update.error {
            return Err(format!("request {} failed: {error}", request_id.0));
        }
        cache_only_continuations = cache_only_continuations.max(
            output
                .route_metadata
                .crossover_decisions
                .iter()
                .find(|(key, _)| key == "ax_mlx_prefill_cache_only_continuations")
                .map(|(_, count)| *count)
                .unwrap_or(0),
        );
        if update.tokens_executed != (end - processed) as u32 {
            return Err(format!(
                "request {} executed {} tokens, expected {}",
                request_id.0,
                update.tokens_executed,
                end - processed
            ));
        }
        let visible_token = update
            .output_token
            .or_else(|| update.output_tokens.first().copied());
        if end < prompt.len() && visible_token.is_some() {
            return Err(format!(
                "request {} emitted a token from non-terminal prefill item {}",
                request_id.0, execution_items
            ));
        }
        if end == prompt.len() {
            output_token = visible_token;
        }
        processed = end;
        execution_items += 1;
    }

    let output_token = output_token
        .ok_or_else(|| format!("request {} final prefill emitted no token", request_id.0))?;
    Ok(PrefillResult {
        output_token,
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
        execution_items,
        cache_only_continuations,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.0;
    }
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn run() -> Result<ExitCode, String> {
    let model_dir = env::args()
        .nth(1)
        .ok_or_else(|| "usage: fair_prefill_bench_probe <model_dir>".to_string())?;
    let prompt_len = env_usize("AX_PROMPT_LEN", 2048)?;
    let quantum = env_usize("AX_PREFILL_QUANTUM", 64)?;
    let warmups = env_usize("AX_WARMUPS", 1)?;
    let repetitions = env_usize("AX_REPETITIONS", 5)?;
    let prompt_seed = env_u64("AX_PROMPT_SEED", 0)?;
    let sampling_seed = env_u64("AX_SAMPLING_SEED", 42)?;
    let decode_tokens = env_usize("AX_DECODE_TOKENS", 0)?;
    let release_state = env_usize("AX_RELEASE_REQUEST_STATE", 1)? != 0;
    let sampling = SamplingConfig {
        temperature: env_f32("AX_TEMPERATURE", 0.8)?,
        top_p: env_f32("AX_TOP_P", 0.95)?,
        top_k: env_u32("AX_TOP_K", 20)?,
        seed: sampling_seed,
    };
    if prompt_len == 0 || repetitions == 0 {
        return Err("AX_PROMPT_LEN and AX_REPETITIONS must be greater than zero".to_string());
    }
    if sampling.temperature < 0.0 || !(0.0..=1.0).contains(&sampling.top_p) {
        return Err("AX_TEMPERATURE must be >= 0 and AX_TOP_P must be in [0, 1]".to_string());
    }

    let load_started = Instant::now();
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut runner = MlxRunner::from_artifacts(&artifacts, DEFAULT_PREFILL_CHUNK, true)
        .map_err(|error| format!("failed to create runner: {error}"))?;
    runner.set_mtp_requested(false);
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;

    let modulus = (cfg.vocab_size.saturating_sub(1)).max(1) as u128;
    let mut next_step_id = 0u64;
    let total_runs = warmups.saturating_add(repetitions);
    let mut measured_ms = Vec::with_capacity(repetitions);
    let mut output_tokens = Vec::with_capacity(repetitions);
    let mut execution_items = 0usize;
    let mut cache_only_continuations = 0u32;

    for run_index in 0..total_runs {
        let run_prompt_seed = prompt_seed.wrapping_add(run_index as u64 * 1_000_003);
        let prompt: Vec<u32> = (0..prompt_len)
            .map(|index| ((run_prompt_seed as u128 + index as u128 * 7 + 3) % modulus + 1) as u32)
            .collect();
        let request_id = RequestId(run_index as u64 + 1);
        let result = run_prefill(
            &runner,
            &prompt,
            quantum,
            request_id,
            sampling,
            &mut next_step_id,
        )?;
        execution_items = result.execution_items;
        cache_only_continuations = result.cache_only_continuations;
        if decode_tokens > 0 {
            run_decode_steps(
                &runner,
                prompt_len,
                result.output_token,
                decode_tokens,
                request_id,
                sampling,
                &mut next_step_id,
            )?;
        }
        if release_state {
            runner.release_request_state(request_id);
        }
        if run_index >= warmups {
            measured_ms.push(result.wall_ms);
            output_tokens.push(result.output_token);
        }
    }

    let mut sorted_ms = measured_ms.clone();
    let median_ttft_ms = median(&mut sorted_ms);
    let report = serde_json::json!({
        "model_family": cfg.model_family,
        "prompt_len": prompt_len,
        "prefill_quantum": quantum,
        "execution_items_per_prompt": execution_items,
        "cache_only_continuations_per_prompt": cache_only_continuations,
        "warmups": warmups,
        "repetitions": repetitions,
        "sampling": {
            "seed": sampling.seed,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
        },
        "load_ms": load_ms,
        "ttft_ms": measured_ms,
        "median_ttft_ms": median_ttft_ms,
        "median_prefill_tokens_per_s": prompt_len as f64 / (median_ttft_ms / 1e3),
        "output_tokens": output_tokens,
    });
    println!("{report}");
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(message) => {
            eprintln!("fair_prefill_bench_probe: {message}");
            ExitCode::FAILURE
        }
    }
}
