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
//!    in ONE `run()` call per step produce the same per-request streams as
//!    decoding them one at a time. Today the runner processes both the same way
//!    (per-item loop), so this is a baseline; once the `run()` batched-decode
//!    interception lands (2b-ii, behind `AX_MLX_BATCHED_DECODE`), the SAME test
//!    with the flag on validates it — token-exact is the promotion gate.
//!
//! The MLX runner keys per-request KV by `request_id` in its own `MlxKVCache`
//! and does NOT read `input.block_tables`, so the harness passes an empty block
//! table vector.
//!
//! Usage:
//!   cargo run --release --bin batched_decode_e2e_probe -- <dense_model_dir>
//! Env: AX_BATCH (default 3), AX_PROMPT_LEN (default 24), AX_GEN (default 16).

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

fn ctx(
    req: u64,
    prompt_len: usize,
    generated_len: usize,
    max_output: usize,
) -> RunnerRequestContext {
    RunnerRequestContext {
        request_id: RequestId(req),
        prompt_len: prompt_len as u32,
        processed_prompt_tokens: prompt_len as u32,
        generated_len: generated_len as u32,
        max_output_tokens: max_output as u32,
        seed: 0,
        deterministic_argmax_sampling: true,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
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
    let out = runner.run(input);
    out.request_updates
        .iter()
        .filter_map(|u| u.output_token.map(|t| (u.request_id.0, t)))
        .collect()
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

fn build_prompts(batch: usize, len: usize, vocab: usize) -> Vec<Vec<u32>> {
    (0..batch)
        .map(|r| {
            (0..len)
                .map(|i| ((r * 31 + i * 7 + 3) % (vocab - 1)) as u32 + 1)
                .collect()
        })
        .collect()
}

fn main() {
    let model_dir = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: batched_decode_e2e_probe <dense_model_dir>");
        std::process::exit(2);
    });
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

    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let weights = load_weights(&artifacts).expect("weights");
    let prompts = build_prompts(batch, prompt_len, cfg.vocab_size);

    println!("# batched decode E2E serving harness");
    println!(
        "model_family {}  batch {batch}  prompt_len {prompt_len}  gen_len {gen_len}  batched_flag {}",
        cfg.model_family,
        ax_engine_mlx::batched_decode_session::batched_decode_enabled()
    );

    // Reference streams via model::forward.
    let refs: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| model_reference(&cfg, &weights, p, gen_len))
        .collect();

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
            vec![ctx(r as u64, prompt.len(), 0, gen_len + 4)],
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
    for s in 0..gen_len {
        let generated_len = s + 1; // prefill produced token #1
        let items: Vec<ExecutionItem> = (0..batch)
            .map(|r| {
                item(
                    r as u64,
                    ExecutionMode::Decode,
                    vec![cur[r]],
                    prompt_len + s,
                )
            })
            .collect();
        let contexts: Vec<RunnerRequestContext> = (0..batch)
            .map(|r| ctx(r as u64, prompt_len, generated_len, gen_len + 4))
            .collect();
        let outs = run_outputs(&runner, runner_input(step_id, items, contexts));
        step_id += 1;
        for (id, tok) in outs {
            let r = id as usize;
            streams[r].push(tok);
            cur[r] = tok;
        }
    }

    // Verdict 1: harness fidelity — runner streams == model::forward reference.
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
        println!(
            "(with AX_MLX_BATCHED_DECODE=1 this same batched-step path exercises the run() interception)"
        );
    } else {
        println!("HARNESS-FIDELITY: FAIL");
        std::process::exit(1);
    }
}
