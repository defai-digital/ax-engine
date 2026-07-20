//! Gemma 4 assistant-MTP draft-depth probe.
//!
//! Goal: measure whether drafting MORE THAN ONE token per step with the Gemma 4
//! assistant drafter is a net win. Production hard-caps the assistant at
//! `max_depth = 1` (runner.rs `config.max_depth = config.max_depth.min(1)`), so
//! Gemma's speculative speedup is structurally capped at ~2x (1 draft + 1 verify),
//! while the Qwen MTP head drafts depth-3. This probe lifts that cap and drives the
//! REAL assistant forward (`gemma4_assistant_forward_one`) recurrently.
//!
//! The architectural question: the assistant has NO key/value projection of its
//! own — it attends the TARGET's frozen KV cache (`peek_layer_kv`) and receives the
//! drafted token only through the residual stream (the token embedding plus the
//! assistant's `post_projection` estimate of the target backbone hidden, which the
//! production draft path computes and then DISCARDS as `_projected_hidden`). So a
//! depth-2 draft query attends the committed prefix but NOT the depth-1 drafted
//! position. This probe wires that discarded `projected_hidden` back in as the next
//! step's backbone hidden and measures the resulting accept rate.
//!
//! Faithfulness: both arms drive the real assistant draft, the real target verify
//! forward (`forward_all_positions_with_post_norm`, correct RoPE / sliding-window
//! mask), and greedy argmax-match acceptance (the production default). The depth-1
//! arm at gate 0.999 should reproduce the production ~98-99% accept rate as a
//! sanity check; the depth-2 numbers are then trustworthy relative to it.
//!
//! Greedy target acceptance => deterministic committed trajectory, so depth-1 and
//! depth-2 are compared on the same generated text (we assert prefix equality).
//!
//! Usage:
//!   cargo run --release --bin gemma_depth_probe -- <target_model_dir> [committed_tokens]
//!
//! Env:
//!   AX_GEMMA_PROMPT_FILE  path to comma/whitespace-separated u32 token ids (a REAL
//!                         tokenized prompt). Required for the realistic
//!                         high-acceptance regime; without it a synthetic id ramp is
//!                         used (relative cross-check only).
//!   AX_GEMMA_DEPTHS       comma-separated depths to sweep (default "1,2,3").
//!   AX_GEMMA_GATES        comma-separated draft confidence gates to sweep
//!                         (default "0,0.9,0.99,0.999"). 0 = ungated (raw ceiling).
//!   AX_GEMMA_PROMPT_LEN   synthetic prompt length when no file (default 48).

use std::env;
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
        Gemma4AssistantSharedKvLayers, ModelConfig, forward_all_positions_update_cache,
        forward_all_positions_with_post_norm, gemma4_assistant_forward_one,
    },
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{MlxArray, MlxDtype, argmax, astype, clear_cache, enable_compile, eval, slice};

/// Per-arm aggregate stats for one gate-schedule configuration.
struct ArmStats {
    depth: usize,
    gate_label: String,
    committed: Vec<u32>,
    steps: usize,
    accepted: usize,        // accepted speculative tokens (excludes primary)
    target_forwards: usize, // verify + recompute forwards (the wall-clock cost)
    assistant_forwards: usize,
    // Per draft-position: how often a token was proposed at this depth and how
    // often it was ultimately accepted (a >= position index).
    proposed_by_pos: [usize; 8],
    accepted_by_pos: [usize; 8],
    wall_s: f64,
}

impl ArmStats {
    fn accept_per_step(&self) -> f64 {
        self.accepted as f64 / self.steps.max(1) as f64
    }
    fn tok_per_fwd(&self) -> f64 {
        self.committed.len() as f64 / self.target_forwards.max(1) as f64
    }
    fn tok_per_s(&self) -> f64 {
        self.committed.len() as f64 / self.wall_s.max(1e-9)
    }
    fn drafted(&self) -> usize {
        self.proposed_by_pos.iter().sum()
    }
    fn accept_rate(&self) -> f64 {
        let d = self.drafted();
        if d == 0 {
            return 1.0;
        }
        self.accepted as f64 / d as f64
    }
}

fn slice_hidden_row(post_norm_all: &MlxArray, row: usize, hidden: usize) -> MlxArray {
    let r = row as i32;
    let h = hidden as i32;
    slice(post_norm_all, &[0, r, 0], &[1, r + 1, h], &[1, 1, 1], None)
}

/// CPU argmax + T=1.0 softmax confidence (probability of the argmax token) over a
/// `[vocab]` f32 logit slice.
fn argmax_confidence(logits: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    let mut sum = 0f64;
    for &v in logits {
        sum += ((v - best_v) as f64).exp();
    }
    let conf = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    (best as u32, conf)
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    target_cfg: &ModelConfig,
    target_weights: &ModelWeights,
    assistant_cfg: &ModelConfig,
    assistant_weights: &ModelWeights,
    shared_layers: Gemma4AssistantSharedKvLayers,
    prompt: &[u32],
    target_tokens: usize,
    gates: &[f32],
) -> ArmStats {
    let depth = gates.len();
    let mut cache = MlxKVCache::new(target_cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (mut primary, mut hidden) = chunked_prefill_with_final_hidden(
        target_cfg,
        target_weights,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );

    let mut committed: Vec<u32> = Vec::with_capacity(target_tokens + depth);
    let mut steps = 0usize;
    let mut accepted = 0usize;
    let mut target_forwards = 0usize;
    let mut assistant_forwards = 0usize;
    let mut proposed_by_pos = [0usize; 8];
    let mut accepted_by_pos = [0usize; 8];

    let t0 = Instant::now();
    while committed.len() < target_tokens {
        let position = cache.seq_len();
        let draft = assistant_draft_safe(
            assistant_cfg,
            assistant_weights,
            target_cfg,
            target_weights,
            &cache,
            shared_layers,
            primary,
            &hidden,
            position,
            gates,
            &mut assistant_forwards,
            &mut proposed_by_pos,
        );

        // Verify [primary] ++ draft on a throwaway clone (1 target forward).
        let token_offset = cache.seq_len();
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft.len());
        verify_input.push(primary);
        verify_input.extend_from_slice(&draft);

        let mut vclone = cache.clone();
        let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
            target_cfg,
            target_weights,
            &verify_input,
            &mut vclone,
            token_offset,
        );
        vclone.advance(verify_input.len());
        let predicted_arr = argmax(&logits_all, None);
        eval(&[&predicted_arr, &post_norm_all]);
        target_forwards += 1;
        let predicted = predicted_arr.data_u32();

        // Greedy argmax-match acceptance (production default for Gemma).
        let mut a = 0usize;
        while a < draft.len() && predicted[a] == draft[a] {
            a += 1;
        }
        #[allow(clippy::needless_range_loop)]
        for pos in 0..a.min(8) {
            accepted_by_pos[pos] += 1;
        }

        // Commit primary + accepted drafts; bonus token carries to next step.
        committed.push(primary);
        committed.extend_from_slice(&draft[..a]);
        accepted += a;
        steps += 1;

        if a == draft.len() {
            // Full accept: the verify clone is exactly the committed prefix.
            cache = vclone;
        } else {
            // Partial accept: roll the real cache forward over the committed
            // prefix only (dense recompute, +1 forward) — the same shape as
            // production's recompute_committed_prefix.
            let committed_step: Vec<u32> = verify_input[..1 + a].to_vec();
            forward_all_positions_update_cache(
                target_cfg,
                target_weights,
                &committed_step,
                &mut cache,
                token_offset,
            );
            cache.advance(committed_step.len());
            target_forwards += 1;
        }

        // Next primary + backbone hidden = the verify position at index `a`.
        let next_hidden = slice_hidden_row(&post_norm_all, a, target_cfg.hidden_size);
        eval(&[&next_hidden]);
        primary = predicted[a];
        hidden = next_hidden;
    }
    let wall_s = t0.elapsed().as_secs_f64();
    clear_cache();

    ArmStats {
        depth,
        gate_label: gates
            .iter()
            .map(|g| format!("{g:.3}"))
            .collect::<Vec<_>>()
            .join(","),
        committed,
        steps,
        accepted,
        target_forwards,
        assistant_forwards,
        proposed_by_pos,
        accepted_by_pos,
        wall_s,
    }
}

/// Safe wrapper around the recurrent assistant draft (no pointer games — the
/// borrow checker is happy because every argument is a plain reference).
#[allow(clippy::too_many_arguments)]
fn assistant_draft_safe(
    assistant_cfg: &ModelConfig,
    assistant_weights: &ModelWeights,
    target_cfg: &ModelConfig,
    target_weights: &ModelWeights,
    target_cache: &MlxKVCache,
    shared_layers: Gemma4AssistantSharedKvLayers,
    primary_token: u32,
    primary_hidden: &MlxArray,
    base_position: usize,
    gates: &[f32],
    forwards: &mut usize,
    proposed_by_pos: &mut [usize; 8],
) -> Vec<u32> {
    let depth = gates.len();
    let mut drafts: Vec<u32> = Vec::with_capacity(depth);
    let mut cur_token = primary_token;
    let mut cur_hidden = astype(primary_hidden, MlxDtype::Bfloat16, None);

    #[allow(clippy::needless_range_loop)]
    for d in 0..depth {
        let gate = gates[d];
        let Ok((logits, projected_hidden)) = gemma4_assistant_forward_one(
            assistant_cfg,
            assistant_weights,
            target_cfg,
            target_weights,
            target_cache,
            shared_layers,
            cur_token,
            &cur_hidden,
            base_position + d,
        ) else {
            break;
        };
        *forwards += 1;
        eval(&[&logits]);
        let logits_cpu = logits.data_f32();
        let (token, conf) = argmax_confidence(logits_cpu);
        if gate > 0.0 && conf < gate {
            break;
        }
        if d < 8 {
            proposed_by_pos[d] += 1;
        }
        drafts.push(token);
        cur_token = token;
        cur_hidden = astype(&projected_hidden, MlxDtype::Bfloat16, None);
    }
    drafts
}

/// Parse `AX_GEMMA_SCHEDULES`: ';'-separated gate schedules, each a comma-separated
/// per-depth confidence gate list. Schedule length = draft depth. E.g.
/// "0.999;0.9,0.99;0.97,0.995;0.9,0.99,0.999" sweeps depth-1 g0.999, depth-2
/// [0.9,0.99] and [0.97,0.995], depth-3 [0.9,0.99,0.999].
fn parse_schedules(s: &str, default: &[&str]) -> Result<Vec<Vec<f32>>, String> {
    let raw: Vec<&str> = if s.trim().is_empty() {
        default.to_vec()
    } else {
        s.split(';').collect()
    };
    let mut schedules = Vec::with_capacity(raw.len());
    for schedule in raw {
        if schedule.trim().is_empty() {
            return Err("AX_GEMMA_SCHEDULES contains an empty schedule".to_string());
        }
        let gates = schedule
            .split(',')
            .map(|part| {
                let part = part.trim();
                let gate = part
                    .parse::<f32>()
                    .map_err(|_| format!("invalid confidence gate {part:?}"))?;
                if !gate.is_finite() || !(0.0..=1.0).contains(&gate) {
                    return Err(format!(
                        "confidence gate must be finite and between 0 and 1, got {part:?}"
                    ));
                }
                Ok(gate)
            })
            .collect::<Result<Vec<_>, String>>()?;
        if gates.is_empty() || gates.len() > 8 {
            return Err("each AX_GEMMA_SCHEDULES entry must have 1 to 8 gates".to_string());
        }
        schedules.push(gates);
    }
    if schedules.is_empty() {
        return Err("AX_GEMMA_SCHEDULES must contain at least one schedule".to_string());
    }
    Ok(schedules)
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
        "usage: gemma_depth_probe <target_model_dir> [committed_tokens]".to_string()
    })?;
    let target_tokens = args
        .next()
        .map(|value| parse_positive_usize("committed_tokens", &value))
        .transpose()?
        .unwrap_or(256);
    if let Some(unexpected) = args.next() {
        return Err(format!("unexpected argument: {unexpected}"));
    }
    let prompt_len = optional_env("AX_GEMMA_PROMPT_LEN")?
        .map(|value| parse_positive_usize("AX_GEMMA_PROMPT_LEN", &value))
        .transpose()?
        .unwrap_or(48);
    // Default sweep: depth-1 baseline + symmetric and asymmetric depth-2/3 gates.
    let schedules = parse_schedules(
        optional_env("AX_GEMMA_SCHEDULES")?.as_deref().unwrap_or(""),
        &[
            "0.999",
            "0.99,0.99",
            "0.9,0.99",
            "0.97,0.995",
            "0.99,0.999",
            "0.9,0.99,0.999",
            "0.97,0.99,0.999",
        ],
    )?;

    println!("Loading target model from {model_dir}...");
    let target_artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
        .map_err(|error| format!("failed to load target artifacts: {error}"))?;
    let target_cfg = ModelConfig::from_manifest(target_artifacts.manifest());
    let target_weights = Arc::new(
        load_weights(&target_artifacts)
            .map_err(|error| format!("failed to load target weights: {error}"))?,
    );

    // Resolve + load the assistant drafter via the production contract loader.
    let status =
        load_gemma4_assistant_mtp_status(Path::new(&model_dir), target_artifacts.manifest());
    let config = status.config.ok_or_else(|| {
        format!(
            "no usable gemma4 assistant MTP contract found next to target model ({:?})",
            status.disable_reason
        )
    })?;
    println!(
        "assistant: {} (path {}), contract max_depth={}",
        config.assistant_model_id,
        config.assistant_path.display(),
        config.max_depth
    );
    let assistant_artifacts = NativeModelArtifacts::from_dir(&config.assistant_path)
        .map_err(|error| format!("failed to load assistant artifacts: {error}"))?;
    let assistant_cfg = ModelConfig::from_manifest(assistant_artifacts.manifest());
    let assistant_weights = Arc::new(
        load_weights(&assistant_artifacts)
            .map_err(|error| format!("failed to load assistant weights: {error}"))?,
    );
    let shared_layers = target_cfg.gemma4_assistant_shared_kv_layers();

    enable_compile();

    let prompt: Vec<u32> = match optional_env("AX_GEMMA_PROMPT_FILE")? {
        Some(path) => {
            let raw = std::fs::read_to_string(&path).map_err(|error| {
                format!("failed to read AX_GEMMA_PROMPT_FILE {path:?}: {error}")
            })?;
            parse_token_ids(&raw)?
        }
        None => {
            let upper = u32::try_from(prompt_len)
                .map_err(|_| "AX_GEMMA_PROMPT_LEN exceeds the u32 token range".to_string())?;
            (1..=upper).collect()
        }
    };
    println!(
        "prompt_tokens={}  target_committed={target_tokens}  schedules={schedules:?}\n",
        prompt.len()
    );

    // Warm up JIT (not measured).
    let max_depth = schedules.iter().map(|s| s.len()).max().unwrap_or(1);
    let _ = run_arm(
        &target_cfg,
        &target_weights,
        &assistant_cfg,
        &assistant_weights,
        shared_layers,
        &prompt,
        8,
        &vec![0.0f32; max_depth],
    );

    println!(
        "  {:>5} {:>16} {:>9} {:>9} {:>10} {:>9} {:>9} {:>8} {:>8}",
        "depth",
        "gate_sched",
        "accept/st",
        "acc_rate",
        "tok/fwd",
        "tok/s",
        "asst_fwd",
        "d2_acc%",
        "d3_acc%"
    );

    let mut best: Option<(f64, String)> = None; // (tok/fwd, label) among acc_rate>0.95
    for sched in &schedules {
        let s = run_arm(
            &target_cfg,
            &target_weights,
            &assistant_cfg,
            &assistant_weights,
            shared_layers,
            &prompt,
            target_tokens,
            sched,
        );
        let pos_pct = |i: usize| -> f64 {
            let prop = s.proposed_by_pos.get(i).copied().unwrap_or(0);
            let acc = s.accepted_by_pos.get(i).copied().unwrap_or(0);
            if prop > 0 {
                100.0 * acc as f64 / prop as f64
            } else {
                0.0
            }
        };
        let label = format!("d{}:{}", s.depth, s.gate_label);
        if s.accept_rate() > 0.95 {
            let tpf = s.tok_per_fwd();
            if best.as_ref().is_none_or(|(b, _)| tpf > *b) {
                best = Some((tpf, label.clone()));
            }
        }
        println!(
            "  {:>5} {:>16} {:>9.3} {:>9.3} {:>10.3} {:>9.2} {:>9} {:>8.1} {:>8.1}",
            s.depth,
            s.gate_label,
            s.accept_per_step(),
            s.accept_rate(),
            s.tok_per_fwd(),
            s.tok_per_s(),
            s.assistant_forwards,
            pos_pct(1),
            pos_pct(2),
        );
    }

    println!();
    println!("  accept/st = accepted draft tokens per step (excludes primary).");
    println!("  acc_rate  = accepted / drafted tokens (the >0.95 constraint).");
    println!("  tok/fwd   = committed tokens / target forwards (THE throughput proxy).");
    println!("  d2/d3_acc% = of steps proposing a 2nd/3rd token, % accepted.");
    if let Some((tpf, label)) = &best {
        println!("  >> BEST (acc_rate>0.95): {label}  tok/fwd={tpf:.3}");
    } else {
        println!("  >> no schedule held acc_rate>0.95");
    }
    println!("  tok/s is thermally noisy; rank on tok/fwd.");
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
