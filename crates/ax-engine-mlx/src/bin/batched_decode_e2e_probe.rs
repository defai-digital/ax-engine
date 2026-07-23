//! Batched decode — end-to-end serving harness (the 2c validation gate).
//!
//! Drives the real `MlxRunner::run()` with prefill + multi-request decode
//! `RunnerInput`s and checks two things:
//!
//! 1. **Harness fidelity.** A request driven through `run()` produces the same
//!    greedy stream as the known-correct `model::forward` reference — proving the
//!    harness constructs `RunnerInput`s the way the engine does (so its verdicts
//!    are trustworthy).
//! 2. **Batched == sequential.** N requests decoded with all their decode items
//!    in one `run()` call per step produce the same per-request streams as
//!    decoding them one at a time.
//! 3. **Shared-forward engagement.** Route telemetry proves that the comparison
//!    exercised the batched forward rather than a structurally rejected
//!    per-item fallback. Token-exact output plus route engagement is the
//!    certification gate.
//!
//! The MLX runner keys per-request KV by `request_id` in its own `MlxKVCache`
//! and does NOT read `input.block_tables`, so the harness passes an empty block
//! table vector.
//!
//! Usage:
//!   cargo run --release --bin batched_decode_e2e_probe -- <dense_model_dir>
//! Env: AX_BATCH (default 3), AX_PROMPT_LEN (default 24), AX_GEN (default 16),
//! AX_PROMPT_SEED (default 0), and AX_RAGGED=1 for unequal prompt lengths.
//!
//! Sampling (gate 3): `AX_SAMPLING` selects the per-request sampler — `greedy`
//! (default), `topp` (temp 0.7, top-p 0.9), `topk` (temp 0.7, top-k 40), or
//! `rep` (temp 0, repetition penalty 1.3). Exercising the numerically uncertified
//! batched forward requires both `AX_MLX_BATCHED_DECODE=1` and
//! `AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED=1`. Sampled batching additionally
//! requires `AX_MLX_BATCHED_DECODE_SAMPLING=1`; without every required flag the
//! runner falls back to per-item decode and the comparison is vacuous. For any
//! non-greedy sampler the model::forward greedy
//! oracle does not apply, so the harness instead decodes every request ALONE on
//! a second runner (each single-item step stays on the per-item path even with
//! the flags on) and checks the batched streams are token-exact with that
//! per-item oracle — the promotion gate for mixed greedy/sampled cohorts.
//! Per-request seeds come from `AX_SEED_BASE` (default 1) so the two runners
//! advance identical RNGs.

use std::env;
use std::path::Path;
use std::process::ExitCode;

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::{
    ExecutionBatch, ExecutionItem, ExecutionMode, ExecutionRunner, NativeModelArtifacts,
    PositionRange, RequestId, RouteMetadata, RunnerInput, StepId,
};
use ax_engine_mlx::{
    MlxRunner,
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{ModelConfig, forward},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{argmax, eval};

const MODEL_ID: &str = "harness";

/// Per-request sampler applied to every request in a run (the harness keeps the
/// cohort homogeneous in *kind* but each request seeds its own RNG).
#[derive(Clone, Copy)]
struct SamplingCfg {
    deterministic: bool,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repetition_penalty: f32,
    seed_base: u64,
    label: &'static str,
}

impl SamplingCfg {
    fn from_env() -> Result<Self, String> {
        let seed_base = optional_env("AX_SEED_BASE")?
            .map(|value| {
                value.parse::<u64>().map_err(|_| {
                    format!("AX_SEED_BASE must be a non-negative integer, got {value:?}")
                })
            })
            .transpose()?
            .unwrap_or(1);
        let sampling = optional_env("AX_SAMPLING")?;
        let config = match sampling.as_deref().unwrap_or("greedy") {
            "topp" => Self {
                deterministic: false,
                temperature: 0.7,
                top_p: 0.9,
                top_k: 0,
                repetition_penalty: 1.0,
                seed_base,
                label: "topp(temp0.7,p0.9)",
            },
            "topk" => Self {
                deterministic: false,
                temperature: 0.7,
                top_p: 1.0,
                top_k: 40,
                repetition_penalty: 1.0,
                seed_base,
                label: "topk(temp0.7,k40)",
            },
            "rep" => Self {
                deterministic: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.3,
                seed_base,
                label: "rep(penalty1.3)",
            },
            "greedy" => Self {
                deterministic: true,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                seed_base,
                label: "greedy",
            },
            other => {
                return Err(format!(
                    "AX_SAMPLING must be one of greedy, topp, topk, or rep; got {other:?}"
                ));
            }
        };
        Ok(config)
    }

    /// Greedy scenarios can be checked against the model::forward oracle; any
    /// non-greedy sampler must use the per-item sequential oracle instead.
    fn is_greedy(&self) -> bool {
        self.deterministic
            || (self.temperature == 0.0
                && self.top_k == 0
                && self.top_p >= 1.0
                && self.repetition_penalty == 1.0)
    }
}

fn ctx(
    req: u64,
    prompt_len: usize,
    generated_len: usize,
    max_output: usize,
    sampling: &SamplingCfg,
) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(req),
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: prompt_len as u32,
        generated_len: generated_len as u32,
        max_output_tokens: max_output as u32,
        // Distinct per-request seed so both runners advance identical RNGs and
        // the batched stream can be compared token-exact against the per-item one.
        seed: sampling.seed_base.wrapping_add(req),
        deterministic_argmax_sampling: sampling.deterministic,
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        repetition_penalty: sampling.repetition_penalty,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        ignore_eos: true, // fixed-length streams for comparison
        tool_call_mode: false,
        structured_output_mode: false,
    }
}

fn item(req: u64, mode: ExecutionMode, tokens: Vec<u32>, start: usize) -> ExecutionItem {
    let n = tokens.len() as u32;
    ExecutionItem {
        request_id: RequestId(req),
        mode,
        planned_work_unit: ax_engine_core::work_unit_for_execution_mode(mode),
        input_token_slice: tokens,
        reused_prefix_token_slice: Vec::new(),
        position_range: PositionRange {
            start: start as u32,
            end_exclusive: start as u32 + n,
        },
        scheduled_token_count: n,
        block_table_ref: RequestId(req),
        prefix_tokens_reused: 0,
        prefix_blocks_reused: 0,
    }
}

fn runner_input(
    step: u64,
    items: Vec<ExecutionItem>,
    contexts: Vec<RunnerRequestContext>,
) -> RunnerInput {
    let total: u32 = items.iter().map(|i| i.scheduled_token_count).sum();
    RunnerInput {
        block_size_tokens: 16,
        memory_pressure: None,
        execution_batch: ExecutionBatch {
            step_id: StepId(step),
            model_id: MODEL_ID.into(),
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

/// `request_id → output token` for one `run()` call.
fn run_outputs(runner: &MlxRunner, input: RunnerInput) -> Vec<(u64, u32)> {
    run_outputs_with_decode_route_rows(runner, input).0
}

#[derive(Clone, Copy, Debug, Default)]
struct DecodeRouteRows {
    tensor_forward: u32,
    row_exact: u32,
    row_exact_forward: u32,
}

/// Output tokens plus the number of rows that entered each production decode
/// coalescing route. Tensor-forward and row-exact counters remain distinct so a
/// row-exact pass can never accidentally mint a tensor-batch certificate.
fn run_outputs_with_decode_route_rows(
    runner: &MlxRunner,
    input: RunnerInput,
) -> (Vec<(u64, u32)>, DecodeRouteRows) {
    let out = runner.run(input);
    let route_value = |target: &str| {
        out.route_metadata
            .crossover_decisions
            .iter()
            .filter(|(name, _)| name == target)
            .map(|(_, rows)| *rows)
            .fold(0u32, u32::saturating_add)
    };
    let route_rows = DecodeRouteRows {
        tensor_forward: route_value("ax_mlx_batched_decode_forward_rows"),
        row_exact: route_value("ax_mlx_row_exact_coalesced_decode_rows"),
        row_exact_forward: route_value("ax_mlx_row_exact_coalesced_decode_forward_rows"),
    };
    let tokens = out
        .request_updates
        .iter()
        .filter_map(|u| u.output_token.map(|t| (u.request_id.0, t)))
        .collect();
    (tokens, route_rows)
}

/// Reference: greedy decode via `model::forward` (the 2a-probe oracle path).
fn model_reference(
    cfg: &ModelConfig,
    w: &ModelWeights,
    prompt: &[u32],
    gen_len: usize,
) -> Vec<u32> {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (tok0, _h) = chunked_prefill_with_final_hidden(
        cfg,
        w,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );
    // `MlxRunner::from_artifacts(..., disable_ngram_acceleration=true)` latches
    // direct greedy Gemma decode onto the bounded physical sliding-KV ring
    // after prefill. Mirror that physical topology in the independent
    // `model::forward` oracle before crossing the sliding-window boundary.
    // Otherwise the two mathematically equivalent paths can choose different
    // greedy tokens at a near tie solely because one retains ordered K/V while
    // the other rotates the same window into slot order.
    cache.set_rotating_sliding_decode(ax_engine_mlx::fastpath::rotating_sliding_decode_enabled());
    let mut stream = vec![tok0];
    let mut tok = tok0;
    for _ in 0..gen_len {
        let offset = cache.seq_len();
        let logits = forward(cfg, w, &[tok], &mut cache, offset);
        let idx = argmax(&logits, None);
        eval(&[&idx]);
        cache.advance(1);
        tok = idx.data_u32()[0];
        stream.push(tok);
    }
    stream
}

/// Per-item oracle: decode every request ALONE (one request's items per `run()`
/// call). A single-item decode step never satisfies the batched interception's
/// `>= 2 eligible` condition, so it stays on the per-item path even with
/// `AX_MLX_BATCHED_DECODE=1` — giving the canonical single-sequence stream for
/// each request under the exact same sampler and seed the batched runner uses.
fn decode_sequential(
    runner: &MlxRunner,
    prompts: &[Vec<u32>],
    sampling: &SamplingCfg,
    gen_len: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let mut streams = Vec::with_capacity(prompts.len());
    let mut step_id = 1_000_000u64; // distinct step-id space from the batched run
    for (r, prompt) in prompts.iter().enumerate() {
        let req = r as u64;
        let prefill = runner_input(
            step_id,
            vec![item(req, ExecutionMode::Prefill, prompt.clone(), 0)],
            vec![ctx(req, prompt.len(), 0, gen_len + 4, sampling)],
        );
        step_id += 1;
        let mut tok = run_outputs(runner, prefill)
            .iter()
            .find(|(id, _)| *id == req)
            .map(|(_, t)| *t)
            .ok_or_else(|| format!("sequential prefill returned no token for request {req}"))?;
        let mut stream = vec![tok];
        for s in 0..gen_len {
            let generated_len = s + 1;
            let input = runner_input(
                step_id,
                vec![item(
                    req,
                    ExecutionMode::Decode,
                    vec![tok],
                    prompt.len() + s,
                )],
                vec![ctx(req, prompt.len(), generated_len, gen_len + 4, sampling)],
            );
            step_id += 1;
            tok = run_outputs(runner, input)
                .iter()
                .find(|(id, _)| *id == req)
                .map(|(_, t)| *t)
                .ok_or_else(|| {
                    format!("sequential decode returned no token for request {req} at step {s}")
                })?;
            stream.push(tok);
        }
        streams.push(stream);
    }
    Ok(streams)
}

fn optional_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    optional_env(name)?
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("{name} must be a non-negative integer, got {value:?}"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn build_prompts(batch: usize, len: usize, vocab: usize) -> Result<Vec<Vec<u32>>, String> {
    if vocab <= 1 || vocab > u32::MAX as usize {
        return Err(format!(
            "model vocabulary size must be between 2 and {}, got {vocab}",
            u32::MAX
        ));
    }
    let prompt_seed = env_usize("AX_PROMPT_SEED", 0)?;
    let ragged = env::var_os("AX_RAGGED").is_some();
    let modulus = (vocab - 1) as u128;
    (0..batch)
        .map(|r| {
            let row_len = if ragged {
                len.saturating_sub(3usize.saturating_mul(r))
                    .max(len / 2)
                    .max(1)
            } else {
                len
            };
            (0..row_len)
                .map(|i| {
                    let value =
                        (prompt_seed as u128 + r as u128 * 31 + i as u128 * 7 + 3) % modulus + 1;
                    u32::try_from(value)
                        .map_err(|_| "generated prompt token exceeds u32".to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect()
}

fn run() -> Result<ExitCode, String> {
    let mut model_dir: Option<String> = None;
    let mut certification_context_json = false;
    for argument in env::args().skip(1) {
        if argument == "--certification-context-json" {
            certification_context_json = true;
        } else if model_dir.is_none() {
            model_dir = Some(argument);
        } else {
            return Err(format!("unexpected argument: {argument}"));
        }
    }
    let model_dir =
        model_dir.ok_or_else(|| "usage: batched_decode_e2e_probe <dense_model_dir>".to_string())?;

    if certification_context_json {
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .map_err(|error| format!("failed to load model artifacts: {error}"))?;
        let context =
            ax_engine_mlx::batched_decode_certification::batched_decode_certification_context(
                &artifacts,
            )
            .map_err(|error| format!("failed to build certification context: {error:?}"))?;
        println!(
            "{}",
            serde_json::to_string_pretty(&context)
                .map_err(|error| format!("failed to serialize certification context: {error}"))?
        );
        return Ok(ExitCode::SUCCESS);
    }

    let batch = env_usize("AX_BATCH", 3)?;
    let prompt_len = env_usize("AX_PROMPT_LEN", 24)?;
    let gen_len = env_usize("AX_GEN", 16)?;
    if batch < 2 {
        return Err("AX_BATCH must be at least 2 to exercise a shared decode forward".to_string());
    }
    if prompt_len == 0 {
        return Err("AX_PROMPT_LEN must be greater than zero".to_string());
    }
    if gen_len == 0 {
        return Err("AX_GEN must be greater than zero".to_string());
    }
    let max_output = gen_len
        .checked_add(4)
        .ok_or_else(|| "AX_GEN is too large".to_string())?;
    let maximum_position = prompt_len
        .checked_add(gen_len)
        .ok_or_else(|| "AX_PROMPT_LEN + AX_GEN overflows usize".to_string())?;
    for (name, value) in [
        ("AX_BATCH", batch),
        ("AX_PROMPT_LEN + AX_GEN", maximum_position),
        ("max_output_tokens", max_output),
    ] {
        if u32::try_from(value).is_err() {
            return Err(format!("{name} exceeds the u32 runner contract"));
        }
    }
    let _ = batch
        .checked_mul(gen_len)
        .ok_or_else(|| "AX_BATCH * AX_GEN overflows usize".to_string())?;

    let sampling = SamplingCfg::from_env()?;
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load model artifacts: {error}"))?;
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights =
        load_weights(&artifacts).map_err(|error| format!("failed to load weights: {error}"))?;
    let prompts = build_prompts(batch, prompt_len, cfg.vocab_size)?;

    let batched_flag = ax_engine_mlx::batched_decode_session::batched_decode_enabled();
    let uncertified_override =
        ax_engine_mlx::batched_decode_session::batched_decode_allow_uncertified();
    let sampling_flag = ax_engine_mlx::batched_decode_session::batched_decode_sampling_enabled();
    println!("# batched decode E2E serving harness");
    println!(
        "model_family {}  batch {batch}  prompt_len {prompt_len}  gen_len {gen_len}  sampling {}  batched_flag {batched_flag}  uncertified_override {uncertified_override}  sampling_flag {sampling_flag}",
        cfg.model_family, sampling.label,
    );
    if batched_flag && !uncertified_override {
        eprintln!(
            "  NOTE: AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED is off — an uncertified \
             Qwen/Gemma model uses row-exact coalescing, never the tensor-batch path."
        );
    }
    if !sampling.is_greedy() && batched_flag && !sampling_flag {
        eprintln!(
            "  WARNING: AX_SAMPLING={} but AX_MLX_BATCHED_DECODE_SAMPLING is off — sampled \
             requests decode per-item in BOTH runners, so BATCHED==SEQUENTIAL is vacuous. \
             Set AX_MLX_BATCHED_DECODE_SAMPLING=1 to exercise the batched sampler.",
            sampling.label
        );
    }

    // Greedy model::forward oracle (only valid for the greedy sampler).
    let refs: Option<Vec<Vec<u32>>> = sampling.is_greedy().then(|| {
        prompts
            .iter()
            .map(|p| model_reference(&cfg, &weights, p, gen_len))
            .collect()
    });

    // Certification covers the direct target path. An attached MTP/assistant
    // artifact is orthogonal: strict MTP requests remain on their speculative
    // route and are profiled/certified separately.
    let mut runner = MlxRunner::from_artifacts(&artifacts, DEFAULT_PREFILL_CHUNK, true)
        .map_err(|error| format!("failed to create batched runner: {error}"))?;
    runner.set_mtp_requested(false);

    // Prefill every request (one run() per prefill item); record the first token.
    let mut cur: Vec<u32> = Vec::with_capacity(batch);
    let mut streams: Vec<Vec<u32>> = Vec::with_capacity(batch);
    let mut step_id = 0u64;
    for (r, prompt) in prompts.iter().enumerate() {
        let input = runner_input(
            step_id,
            vec![item(r as u64, ExecutionMode::Prefill, prompt.clone(), 0)],
            vec![ctx(r as u64, prompt.len(), 0, gen_len + 4, &sampling)],
        );
        step_id += 1;
        let outs = run_outputs(&runner, input);
        let tok0 = outs
            .iter()
            .find(|(id, _)| *id == r as u64)
            .map(|(_, t)| *t)
            .ok_or_else(|| format!("prefill returned no token for request {r}"))?;
        cur.push(tok0);
        streams.push(vec![tok0]);
    }

    // Batched decode: ALL requests' decode items in ONE run() call per step.
    // Timed: with the flag on the group runs one shared forward; with it off the
    // runner processes them per item (B sequential forwards) — the runner-level
    // A/B (run the harness once per flag value, same batch, and compare).
    let decode_started = std::time::Instant::now();
    let mut batched_forward_steps = 0usize;
    let mut batched_forward_row_mismatch = false;
    let mut row_exact_steps = 0usize;
    let mut row_exact_row_mismatch = false;
    let mut row_exact_forward_rows = 0usize;
    for s in 0..gen_len {
        let generated_len = s + 1; // prefill produced token #1
        let items: Vec<ExecutionItem> = (0..batch)
            .map(|r| {
                item(
                    r as u64,
                    ExecutionMode::Decode,
                    vec![cur[r]],
                    prompts[r].len() + s,
                )
            })
            .collect();
        let contexts: Vec<RunnerRequestContext> = (0..batch)
            .map(|r| {
                ctx(
                    r as u64,
                    prompts[r].len(),
                    generated_len,
                    gen_len + 4,
                    &sampling,
                )
            })
            .collect();
        let (outs, route_rows) =
            run_outputs_with_decode_route_rows(&runner, runner_input(step_id, items, contexts));
        if route_rows.tensor_forward > 0 {
            batched_forward_steps += 1;
            batched_forward_row_mismatch |= route_rows.tensor_forward != batch as u32;
        }
        if route_rows.row_exact > 0 {
            row_exact_steps += 1;
            row_exact_row_mismatch |= route_rows.row_exact != batch as u32;
            row_exact_forward_rows =
                row_exact_forward_rows.saturating_add(route_rows.row_exact_forward as usize);
        }
        step_id += 1;
        let mut seen = vec![false; batch];
        for (id, tok) in outs {
            let r = id as usize;
            if r >= batch {
                return Err(format!("decode returned unexpected request id {id}"));
            }
            if seen[r] {
                return Err(format!(
                    "decode returned duplicate output for request {id} at step {s}"
                ));
            }
            seen[r] = true;
            streams[r].push(tok);
            cur[r] = tok;
        }
        if let Some(missing) = seen.iter().position(|seen| !seen) {
            return Err(format!(
                "decode returned no output for request {missing} at step {s}"
            ));
        }
    }
    let decode_s = decode_started.elapsed().as_secs_f64();
    println!(
        "decode: {decode_s:.3}s for {} tokens = {:.1} agg tok/s (batched_flag {})",
        batch * gen_len,
        (batch * gen_len) as f64 / decode_s,
        ax_engine_mlx::batched_decode_session::batched_decode_enabled()
    );

    let mut failed = false;
    let minimum_batched_forward_steps = gen_len.saturating_sub(1).max(1);
    if batched_forward_steps >= minimum_batched_forward_steps && !batched_forward_row_mismatch {
        println!(
            "BATCHED-PATH: PASS ({batched_forward_steps}/{gen_len} decode steps used the shared {batch}-row forward)"
        );
    } else if batched_forward_steps > 0 || uncertified_override {
        println!(
            "BATCHED-PATH: FAIL ({batched_forward_steps}/{gen_len} shared-forward steps, expected at least {minimum_batched_forward_steps}; row_mismatch={batched_forward_row_mismatch})"
        );
        failed = true;
    } else {
        println!("BATCHED-PATH: SKIP (certification or diagnostic override did not engage it)");
    }
    if row_exact_steps >= minimum_batched_forward_steps && !row_exact_row_mismatch {
        println!(
            "ROW-EXACT-COALESCED-PATH: PASS ({row_exact_steps}/{gen_len} decode steps, \
             {row_exact_forward_rows} independently shaped rows submitted in groups)"
        );
    } else if row_exact_steps > 0 {
        println!(
            "ROW-EXACT-COALESCED-PATH: FAIL ({row_exact_steps}/{gen_len} coalesced steps, \
             expected at least {minimum_batched_forward_steps}; \
             row_mismatch={row_exact_row_mismatch})"
        );
        failed = true;
    } else {
        println!("ROW-EXACT-COALESCED-PATH: SKIP (tensor batch or per-item route selected)");
    }

    // Verdict 1 (greedy only): harness fidelity vs the model::forward oracle.
    if let Some(refs) = &refs {
        let mut ok = true;
        for r in 0..batch {
            if streams[r] != refs[r] {
                ok = false;
                let first = (0..streams[r].len().max(refs[r].len()))
                    .find(|&i| streams[r].get(i) != refs[r].get(i));
                eprintln!(
                    "  req {r} MISMATCH at {first:?}:\n    runner {:?}\n    ref    {:?}",
                    streams[r], refs[r]
                );
            }
        }
        if ok {
            println!(
                "HARNESS-FIDELITY: PASS ({batch}/{batch} runner streams == model::forward reference)"
            );
        } else {
            println!("HARNESS-FIDELITY: FAIL");
            failed = true;
        }
    } else {
        println!("HARNESS-FIDELITY: SKIP (non-greedy sampler has no model::forward oracle)");
    }

    // Verdict 2 (all samplers): batched streams == per-item sequential oracle.
    // This is the gate-3 promotion check — for sampled cohorts it proves the
    // batched sampler is token-exact with the single-sequence sampler, RNG and
    // tie-break included. A second runner keeps its own per-request KV/RNG state.
    let mut seq_runner = MlxRunner::from_artifacts(&artifacts, DEFAULT_PREFILL_CHUNK, true)
        .map_err(|error| format!("failed to create sequential runner: {error}"))?;
    seq_runner.set_mtp_requested(false);
    let seq_streams = decode_sequential(&seq_runner, &prompts, &sampling, gen_len)?;
    let mut ok = true;
    for r in 0..batch {
        if streams[r] != seq_streams[r] {
            ok = false;
            let first = (0..streams[r].len().max(seq_streams[r].len()))
                .find(|&i| streams[r].get(i) != seq_streams[r].get(i));
            eprintln!(
                "  req {r} MISMATCH at {first:?}:\n    batched    {:?}\n    sequential {:?}",
                streams[r], seq_streams[r]
            );
        }
    }
    if ok {
        println!(
            "BATCHED==SEQUENTIAL: PASS ({batch}/{batch} batched streams == per-item oracle, sampling {})",
            sampling.label
        );
    } else {
        println!("BATCHED==SEQUENTIAL: FAIL");
        failed = true;
    }

    if failed {
        return Ok(ExitCode::from(1));
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(exit_code) => exit_code,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
