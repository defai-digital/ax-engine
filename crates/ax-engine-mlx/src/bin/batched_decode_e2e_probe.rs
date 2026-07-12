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

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::{
    ExecutionBatch, ExecutionItem, ExecutionMode, ExecutionRunner, KvCompressionConfig,
    NativeModelArtifacts, PositionRange, RequestId, RouteMetadata, RunnerInput, StepId,
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
    fn from_env() -> Self {
        let seed_base = env::var("AX_SEED_BASE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        match env::var("AX_SAMPLING").as_deref() {
            Ok("topp") => Self {
                deterministic: false,
                temperature: 0.7,
                top_p: 0.9,
                top_k: 0,
                repetition_penalty: 1.0,
                seed_base,
                label: "topp(temp0.7,p0.9)",
            },
            Ok("topk") => Self {
                deterministic: false,
                temperature: 0.7,
                top_p: 1.0,
                top_k: 40,
                repetition_penalty: 1.0,
                seed_base,
                label: "topk(temp0.7,k40)",
            },
            Ok("rep") => Self {
                deterministic: false,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.3,
                seed_base,
                label: "rep(penalty1.3)",
            },
            _ => Self {
                deterministic: true,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                seed_base,
                label: "greedy",
            },
        }
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
    run_outputs_with_batched_forward_rows(runner, input).0
}

/// Output tokens plus the number of rows that entered the shared forward.
fn run_outputs_with_batched_forward_rows(
    runner: &MlxRunner,
    input: RunnerInput,
) -> (Vec<(u64, u32)>, u32) {
    let out = runner.run(input);
    let batched_forward_rows = out
        .route_metadata
        .crossover_decisions
        .iter()
        .filter(|(name, _)| name == "ax_mlx_batched_decode_forward_rows")
        .map(|(_, rows)| *rows)
        .sum();
    let tokens = out
        .request_updates
        .iter()
        .filter_map(|u| u.output_token.map(|t| (u.request_id.0, t)))
        .collect();
    (tokens, batched_forward_rows)
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
    let mut stream = vec![tok0];
    let mut tok = tok0;
    for _ in 0..gen_len {
        let offset = cache.seq_len;
        let logits = forward(cfg, w, &[tok], &mut cache, offset);
        let idx = argmax(&logits, None);
        eval(&[&idx]);
        cache.seq_len += 1;
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
) -> Vec<Vec<u32>> {
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
            .expect("sequential prefill token");
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
                .expect("sequential decode token");
            stream.push(tok);
        }
        streams.push(stream);
    }
    streams
}

fn build_prompts(batch: usize, len: usize, vocab: usize) -> Vec<Vec<u32>> {
    let prompt_seed = env::var("AX_PROMPT_SEED")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let ragged = env::var_os("AX_RAGGED").is_some();
    (0..batch)
        .map(|r| {
            let row_len = if ragged {
                len.saturating_sub(3 * r).max(len / 2).max(1)
            } else {
                len
            };
            (0..row_len)
                .map(|i| ((prompt_seed + r * 31 + i * 7 + 3) % (vocab - 1)) as u32 + 1)
                .collect()
        })
        .collect()
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let model_dir = args.first().cloned().unwrap_or_else(|| {
        eprintln!("usage: batched_decode_e2e_probe <dense_model_dir>");
        std::process::exit(2);
    });
    let certification_context_json = args
        .iter()
        .any(|argument| argument == "--certification-context-json");
    let batch: usize = env::var("AX_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let prompt_len: usize = env::var("AX_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let gen_len: usize = env::var("AX_GEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let sampling = SamplingCfg::from_env();

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    if certification_context_json {
        let context =
            ax_engine_mlx::batched_decode_certification::batched_decode_certification_context(
                &artifacts,
            )
            .expect("certification context");
        println!(
            "{}",
            serde_json::to_string_pretty(&context).expect("serialize certification context")
        );
        return;
    }
    let weights = load_weights(&artifacts).expect("weights");
    let prompts = build_prompts(batch, prompt_len, cfg.vocab_size);

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
            "  WARNING: AX_MLX_BATCHED_DECODE_ALLOW_UNCERTIFIED is off — the runner will use the per-item path, so BATCHED==SEQUENTIAL is vacuous."
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

    // Build a runner. run() uses interior mutability, so one runner drives all.
    let runner = MlxRunner::from_artifacts(
        &artifacts,
        DEFAULT_PREFILL_CHUNK,
        true,
        KvCompressionConfig::disabled(),
    )
    .expect("runner");

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
            .expect("prefill token");
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
        let (outs, batched_forward_rows) =
            run_outputs_with_batched_forward_rows(&runner, runner_input(step_id, items, contexts));
        if batched_forward_rows > 0 {
            batched_forward_steps += 1;
            batched_forward_row_mismatch |= batched_forward_rows != batch as u32;
        }
        step_id += 1;
        for (id, tok) in outs {
            let r = id as usize;
            streams[r].push(tok);
            cur[r] = tok;
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
    let seq_runner = MlxRunner::from_artifacts(
        &artifacts,
        DEFAULT_PREFILL_CHUNK,
        true,
        KvCompressionConfig::disabled(),
    )
    .expect("sequential runner");
    let seq_streams = decode_sequential(&seq_runner, &prompts, &sampling, gen_len);
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
        std::process::exit(1);
    }
}
