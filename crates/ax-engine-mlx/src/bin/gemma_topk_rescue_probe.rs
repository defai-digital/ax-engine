//! Gemma 4 assistant-MTP top-k position-1 rescue headroom probe.
//!
//! Question: how much decode headroom would a k-way position-1 candidate draft
//! (tree verify) add over the shipped depth-2 policy? Production drafts a
//! single argmax chain gated at 0.85 (first position) / 0.999 (deep). A miss
//! still commits the target's correction token, so top-k rescue only helps on
//! (a) steps where the greedy draft misses but the target token sits in the
//! drafter's top-k, and (b) steps the first gate blocks entirely, where a
//! k-candidate submission could still accept one token.
//!
//! Method: commit the TARGET-greedy trajectory (policy-independent under
//! greedy acceptance), and at every step run the real ungated assistant draft
//! (`gemma4_assistant_forward_one`) recurrently to depth 2, recording per
//! step: drafter confidence, the committed target token's rank inside the
//! drafter's top-8, and the same for the depth-2 position conditional on a
//! depth-1 hit. Policies (gates, top-k widths, chains) are then simulated
//! exactly from the JSONL records offline — one model pass measures every
//! policy.
//!
//! Usage:
//!   cargo run --release --bin gemma_topk_rescue_probe -- <mtp_package_dir> [committed_tokens]
//!
//! Env:
//!   AX_TOPK_PROMPT_FILE  comma/whitespace-separated u32 token ids of a REAL
//!                        chat-templated prompt (required for realistic rates).
//!   AX_TOPK_OUT          output JSONL path (default: stdout).

use std::env;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    gemma4_assistant_mtp::load_gemma4_assistant_mtp_status,
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{
        Gemma4AssistantSharedKvLayers, ModelConfig, forward_all_positions_with_post_norm,
        gemma4_assistant_forward_one,
    },
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::load_weights,
};
use mlx_sys::{MlxArray, MlxDtype, argmax, astype, clear_cache, enable_compile, eval, slice};

const TOP_K: usize = 8;

fn slice_hidden_row(post_norm_all: &MlxArray, row: usize, hidden: usize) -> MlxArray {
    let r = row as i32;
    let h = hidden as i32;
    slice(post_norm_all, &[0, r, 0], &[1, r + 1, h], &[1, 1, 1], None)
}

/// Top-k token ids by logit plus the T=1.0 softmax probability of the argmax.
fn topk_with_confidence(logits: &[f32], k: usize) -> Result<(Vec<u32>, f32), String> {
    if logits.is_empty() {
        return Err("assistant produced empty logits".to_string());
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("assistant produced non-finite logits".to_string());
    }
    if k == 0 {
        return Err("top-k width must be greater than zero".to_string());
    }
    let keep = k.min(logits.len());
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(keep - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
    let mut top: Vec<usize> = idx[..keep].to_vec();
    top.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let best_v = logits[top[0]];
    let mut sum = 0f64;
    for &v in logits {
        sum += ((v - best_v) as f64).exp();
    }
    let conf = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    Ok((top.into_iter().map(|i| i as u32).collect(), conf))
}

/// Rank (1-based) of `token` inside `top`, or 0 when absent.
fn rank_of(top: &[u32], token: u32) -> usize {
    top.iter().position(|&t| t == token).map_or(0, |p| p + 1)
}

struct PendingDepth2 {
    top: Vec<u32>,
    conf: f32,
    /// Depth-1 draft hit, so the depth-2 chain conditioning matches the
    /// committed trajectory and the record is meaningful.
    valid: bool,
}

fn parse_positive_usize(label: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{label} must be a positive integer, got {value:?}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

fn parse_token_ids(raw: &str) -> Result<Vec<u32>, String> {
    let ids = raw
        .split(|character: char| character == ',' || character.is_whitespace())
        .filter(|token| !token.trim().is_empty())
        .map(|token| {
            token
                .trim()
                .parse::<u32>()
                .map_err(|_| format!("invalid prompt token id {token:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.is_empty() {
        return Err("prompt token list must not be empty".to_string());
    }
    Ok(ids)
}

fn optional_env(name: &str) -> Result<Option<String>, String> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {name}: {error}")),
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let model_dir = args.next().ok_or_else(|| {
        "usage: gemma_topk_rescue_probe <mtp_package_dir> [committed_tokens]".to_string()
    })?;
    let target_tokens = args
        .next()
        .map(|value| parse_positive_usize("committed_tokens", &value))
        .transpose()?
        .unwrap_or(400);
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }

    println!("Loading target model from {model_dir}...");
    let target_artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load target artifacts: {error}"))?;
    let target_cfg = ModelConfig::from_manifest(target_artifacts.manifest());
    let target_weights = Arc::new(
        load_weights(&target_artifacts)
            .map_err(|error| format!("failed to load target weights: {error}"))?,
    );

    let status =
        load_gemma4_assistant_mtp_status(Path::new(&model_dir), target_artifacts.manifest());
    let config = status.config.ok_or_else(|| {
        format!(
            "no usable gemma4 assistant MTP contract found next to target model ({:?})",
            status.disable_reason
        )
    })?;
    println!(
        "assistant: {} (path {})",
        config.assistant_model_id,
        config.assistant_path.display()
    );
    let assistant_artifacts = NativeModelArtifacts::from_dir(&config.assistant_path)
        .map_err(|error| format!("failed to load assistant artifacts: {error}"))?;
    let assistant_cfg = ModelConfig::from_manifest(assistant_artifacts.manifest());
    let assistant_weights = Arc::new(
        load_weights(&assistant_artifacts)
            .map_err(|error| format!("failed to load assistant weights: {error}"))?,
    );
    let shared_layers: Gemma4AssistantSharedKvLayers =
        target_cfg.gemma4_assistant_shared_kv_layers();

    enable_compile();

    let prompt: Vec<u32> = match optional_env("AX_TOPK_PROMPT_FILE")? {
        Some(path) => {
            let raw = std::fs::read_to_string(&path)
                .map_err(|error| format!("failed to read AX_TOPK_PROMPT_FILE {path:?}: {error}"))?;
            parse_token_ids(&raw)?
        }
        None => (1..=48u32).collect(),
    };
    let mut out: Box<dyn Write> = match optional_env("AX_TOPK_OUT")? {
        Some(path) => Box::new(
            std::fs::File::create(&path)
                .map_err(|error| format!("failed to create AX_TOPK_OUT {path:?}: {error}"))?,
        ),
        None => Box::new(std::io::stdout()),
    };
    println!(
        "prompt_tokens={}  target_committed={target_tokens}",
        prompt.len()
    );

    let mut cache = MlxKVCache::new(target_cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (mut primary, mut hidden) = chunked_prefill_with_final_hidden(
        &target_cfg,
        &target_weights,
        &prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), &prompt),
        &mut rng,
    );

    let mut committed = 0usize;
    let mut pending_d2: Option<PendingDepth2> = None;
    let t0 = Instant::now();

    while committed < target_tokens {
        let base_position = cache.seq_len();

        // Ungated recurrent assistant draft to depth 2, capturing top-k.
        let hidden_bf16 = astype(&hidden, MlxDtype::Bfloat16, None);
        let (d1_top, d1_conf, d2_record) = match gemma4_assistant_forward_one(
            &assistant_cfg,
            &assistant_weights,
            &target_cfg,
            &target_weights,
            &cache,
            shared_layers,
            primary,
            &hidden_bf16,
            base_position,
        ) {
            Ok((logits, projected_hidden)) => {
                eval(&[&logits]);
                let (d1_top, d1_conf) = topk_with_confidence(logits.data_f32(), TOP_K)?;
                let d1_argmax = d1_top[0];
                let cur_hidden = astype(&projected_hidden, MlxDtype::Bfloat16, None);
                let (logits2, _) = gemma4_assistant_forward_one(
                    &assistant_cfg,
                    &assistant_weights,
                    &target_cfg,
                    &target_weights,
                    &cache,
                    shared_layers,
                    d1_argmax,
                    &cur_hidden,
                    base_position + 1,
                )
                .map_err(|error| {
                    format!("assistant depth-2 draft failed at step {committed}: {error}")
                })?;
                eval(&[&logits2]);
                let d2 = Some(topk_with_confidence(logits2.data_f32(), TOP_K)?);
                (d1_top, d1_conf, d2)
            }
            Err(err) => {
                return Err(format!("assistant draft failed at step {committed}: {err}"));
            }
        };

        // Commit the target-greedy next token (1 target forward).
        let token_offset = cache.seq_len();
        let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
            &target_cfg,
            &target_weights,
            &[primary],
            &mut cache,
            token_offset,
        );
        cache.advance(1);
        let predicted_arr = argmax(&logits_all, None);
        eval(&[&predicted_arr, &post_norm_all]);
        let t_next = predicted_arr.data_u32()[0];

        // Resolve last step's depth-2 record: its chain target is this step's
        // committed token.
        if let Some(p) = pending_d2.take()
            && p.valid
        {
            let rank2 = rank_of(&p.top, t_next);
            writeln!(
                out,
                "{{\"pos\":2,\"conf\":{:.6},\"rank\":{rank2},\"target\":{t_next}}}",
                p.conf
            )
            .map_err(|error| format!("failed to write depth-2 record: {error}"))?;
        }

        let rank1 = rank_of(&d1_top, t_next);
        writeln!(
            out,
            "{{\"pos\":1,\"conf\":{:.6},\"rank\":{rank1},\"target\":{t_next}}}",
            d1_conf
        )
        .map_err(|error| format!("failed to write depth-1 record: {error}"))?;
        pending_d2 = d2_record.map(|(top, conf)| PendingDepth2 {
            top,
            conf,
            valid: rank1 == 1,
        });

        let next_hidden = slice_hidden_row(&post_norm_all, 0, target_cfg.hidden_size);
        eval(&[&next_hidden]);
        primary = t_next;
        hidden = next_hidden;
        committed += 1;

        if committed.is_multiple_of(100) {
            println!(
                "  {committed}/{target_tokens} committed ({:.1} tok/s probe pace)",
                committed as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    clear_cache();
    println!(
        "done: committed={committed} wall={:.1}s",
        t0.elapsed().as_secs_f64()
    );
    out.flush()
        .map_err(|error| format!("failed to flush probe output: {error}"))?;
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
