//! Tree-draft speculative-decoding prototype (Phase A).
//!
//! Goal: measure whether a *tree* of MTP drafts breaks the linear depth-D
//! acceptance ceiling on the dense 27B path — i.e. accepts meaningfully more
//! tokens per target verify forward than the current linear argmax chain.
//!
//! Design (faithful, no FFI changes):
//!   * Both arms drive the model's *real* MTP head and the *real* target verify
//!     forward (`forward_all_positions_with_post_norm`), so RoPE positions are
//!     correct — every candidate path is verified as an ordinary linear sequence
//!     `[primary] ++ path` at sequential positions.
//!   * The linear arm is the tree arm with branch schedule `[1,1,...]` (a single
//!     greedy candidate), so the two arms are byte-for-byte the same code path
//!     and differ only in how many candidates they verify.
//!   * Greedy (argmax) target acceptance ⇒ the committed output is deterministic
//!     and identical across schedules; we assert prefix equality as a correctness
//!     check, then compare accepted-tokens/step.
//!
//! What Phase A proves and does NOT prove:
//!   * PROVES: the acceptance uplift — mean accepted drafts/step for the tree vs
//!     the linear chain on the same trajectory. This is the ceiling-break signal.
//!   * Does NOT prove wall-clock yet: this prototype verifies each candidate in
//!     its OWN forward (so it pays `leaves` forwards/step). The real win needs the
//!     Phase B single-forward tree verify (tree attention mask + per-token RoPE
//!     positions, which requires extending the `mlx-sys` rope FFI). We therefore
//!     report the PROJECTED single-forward throughput: under a single-forward
//!     implementation the tree costs 1 forward/step, so projected effective
//!     tokens/forward = 1 + mean_accepted, and projected speedup over linear =
//!     (1 + mean_accepted_tree) / (1 + mean_accepted_linear).
//!
//! Usage:
//!   cargo run --release --bin tree_draft_probe -- <model_dir> [committed_tokens]
//!
//! Env:
//!   AX_TREE_SCHEDULES   ';'-separated tree schedules to sweep; each is a
//!                       comma-separated per-depth branch factor list
//!                       (e.g. "2,2,1,1,1;2,2,2,1,1"). Each is compared against a
//!                       linear chain of the same depth. Falls back to AX_TREE_BRANCH.
//!   AX_TREE_BRANCH      single schedule (default "2,2,1,1,1") when no sweep set.
//!   AX_TREE_PROMPT_FILE path to a file of comma/whitespace-separated u32 token
//!                       ids (a REAL tokenized prompt). Required to measure in the
//!                       realistic high-acceptance regime; without it a synthetic
//!                       id ramp is used (relative cross-check only).
//!   AX_TREE_PROMPT_LEN  synthetic prompt length when no file (default 48).

use std::env;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ax_engine_core::NativeModelArtifacts;
use ax_engine_mlx::{
    generate::{DEFAULT_PREFILL_CHUNK, chunked_prefill_with_final_hidden},
    kv_cache::MlxKVCache,
    model::{
        ModelConfig, forward_all_positions, forward_all_positions_update_cache,
        forward_all_positions_with_post_norm,
    },
    mtp::{mtp_draft_tokens, mtp_draft_tokens_gated, mtp_head_step},
    sampling::{MlxSamplingParams, MlxSamplingRequest, Xorshift64},
    weights::{ModelWeights, load_weights},
};
use mlx_sys::{MlxArray, argmax, clear_cache, enable_compile, eval, slice};

/// One root-to-leaf candidate draft path plus the per-depth head confidence
/// (probability the head assigned to each chosen token, for optional gating).
struct Candidate {
    tokens: Vec<u32>,
}

/// Aggregate statistics for one decode arm.
#[allow(dead_code)] // label/forwards/leaves retained for diagnostics
struct ArmStats {
    label: String,
    committed: Vec<u32>,
    steps: usize,
    accepted_drafts: usize, // accepted speculative tokens (excludes primary + bonus)
    forwards: usize,        // target verify forwards actually executed
    leaves: usize,          // candidate paths verified across the run
    wall_s: f64,
}

impl ArmStats {
    fn mean_accepted(&self) -> f64 {
        self.accepted_drafts as f64 / self.steps.max(1) as f64
    }
    /// Projected tokens/forward if each step used a single tree-mask forward.
    fn effective_tpf_projected(&self) -> f64 {
        1.0 + self.mean_accepted()
    }
}

fn slice_hidden_row(post_norm_all: &MlxArray, row: usize, hidden: usize) -> MlxArray {
    // post_norm_all: [1, seq, hidden] -> [1, 1, hidden] at `row`.
    let r = row as i32;
    let h = hidden as i32;
    slice(post_norm_all, &[0, r, 0], &[1, r + 1, h], &[1, 1, 1], None)
}

/// Build the candidate set by branching the real MTP head per `branch[d]`.
///
/// Returns every root-to-leaf path of length `branch.len()`. `[1,1,..]` yields
/// exactly one greedy chain (the linear baseline).
fn draft_tree(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    primary_hidden: &MlxArray,
    primary_token: u32,
    branch: &[usize],
    vocab: usize,
) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    let head_cache = MlxKVCache::new(1);
    expand(
        weights,
        cfg,
        primary_hidden,
        primary_token,
        head_cache,
        branch,
        0,
        &mut Vec::new(),
        &mut out,
        vocab,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn expand(
    weights: &ModelWeights,
    cfg: &ModelConfig,
    main_hidden: &MlxArray,
    prev_token: u32,
    mut cache: MlxKVCache,
    branch: &[usize],
    depth: usize,
    path: &mut Vec<u32>,
    out: &mut Vec<Candidate>,
    vocab: usize,
) {
    if depth == branch.len() {
        out.push(Candidate {
            tokens: path.clone(),
        });
        return;
    }
    // One head step from this node; logits drive the branching at this depth.
    let Some((post_norm, logits)) =
        mtp_head_step(weights, cfg, main_hidden, prev_token, &mut cache)
    else {
        // No MTP head — emit whatever prefix we have.
        out.push(Candidate {
            tokens: path.clone(),
        });
        return;
    };
    let k = branch[depth].max(1);
    let children = topk_tokens(&logits, k, vocab);
    for (ci, &child) in children.iter().enumerate() {
        path.push(child);
        // Each sibling needs its own KV state diverging at this node; the last
        // child can consume `cache` directly to save one clone.
        let branch_cache = if ci + 1 == children.len() {
            std::mem::replace(&mut cache, MlxKVCache::new(1))
        } else {
            cache.clone()
        };
        expand(
            weights,
            cfg,
            &post_norm,
            child,
            branch_cache,
            branch,
            depth + 1,
            path,
            out,
            vocab,
        );
        path.pop();
    }
}

/// Top-k token ids from a `[vocab]` f32 logit array (CPU; k is tiny).
fn topk_tokens(logits: &MlxArray, k: usize, vocab: usize) -> Vec<u32> {
    if k <= 1 {
        let a = argmax(logits, None);
        eval(&[&a]);
        return vec![a.data_u32()[0]];
    }
    eval(&[logits]);
    let data = logits.data_f32();
    let n = vocab.min(data.len());
    // Partial selection of the k largest, returned in descending-logit order.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| data[b].partial_cmp(&data[a]).unwrap());
    idx.truncate(k);
    idx.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());
    idx.into_iter().map(|i| i as u32).collect()
}

/// Verify one candidate path as a linear sequence on a *throwaway* cache clone.
///
/// Returns the greedily-accepted prefix length: the count of leading draft
/// tokens whose target argmax matches. The clone is discarded — the real cache
/// is advanced separately by exactly the committed tokens (see `commit_forward`),
/// which is correct for both dense AND linear-attention layers (the recurrent
/// linear/conv state cannot be trimmed back to a prefix, so we must not adopt a
/// clone that ran past the accepted point).
fn verify_candidate(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    base_cache: &MlxKVCache,
    primary_token: u32,
    cand: &[u32],
    token_offset: usize,
) -> usize {
    let mut verify_input: Vec<u32> = Vec::with_capacity(1 + cand.len());
    verify_input.push(primary_token);
    verify_input.extend_from_slice(cand);

    let mut cache = base_cache.clone();
    let logits_all = forward_all_positions(cfg, weights, &verify_input, &mut cache, token_offset);
    let predicted_arr = argmax(&logits_all, None); // [verify_len]
    eval(&[&predicted_arr]);
    let predicted = predicted_arr.data_u32();

    // predicted[i] = target argmax after verify_input[i]; it should equal
    // verify_input[i+1] = cand[i] when the draft is correct.
    let mut accepted = 0usize;
    while accepted < cand.len() && predicted[accepted] == cand[accepted] {
        accepted += 1;
    }
    accepted
}

/// Advance the real cache over exactly the committed tokens `[primary] ++ accepted`
/// and return `(bonus_token, next_hidden)` for the following step.
///
/// One forward over the committed prefix is the linear-attention-safe analogue
/// of production's `recompute_committed_prefix`: it leaves the recurrent state
/// exactly at the committed boundary. The last logit row predicts the bonus
/// token; the last post-norm hidden row seeds the next MTP draft.
fn commit_forward(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    committed_step: &[u32],
    token_offset: usize,
) -> (u32, MlxArray) {
    let (logits_all, post_norm_all) =
        forward_all_positions_with_post_norm(cfg, weights, committed_step, cache, token_offset);
    cache.seq_len += committed_step.len();
    let last = committed_step.len() - 1;
    let bonus_arr = argmax(&logits_all, None);
    let hidden = slice_hidden_row(&post_norm_all, last, cfg.hidden_size);
    eval(&[&bonus_arr, &hidden]);
    (bonus_arr.data_u32()[last], hidden)
}

/// Run a full greedy decode generating `target_tokens` committed tokens with the
/// given branch schedule, drafting via the real MTP head and verifying each
/// candidate faithfully.
fn run_arm(
    label: &str,
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt: &[u32],
    target_tokens: usize,
    branch: &[usize],
) -> ArmStats {
    let vocab = cfg.vocab_size;
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);

    let (mut primary, mut hidden) = chunked_prefill_with_final_hidden(
        cfg,
        weights,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );

    let mut committed: Vec<u32> = Vec::with_capacity(target_tokens + branch.len());
    let mut steps = 0usize;
    let mut accepted_drafts = 0usize;
    let mut forwards = 0usize;
    let mut leaves = 0usize;

    let t0 = Instant::now();
    while committed.len() < target_tokens {
        // 1. Draft a tree of candidate paths from (primary, hidden).
        let candidates = draft_tree(weights, cfg, &hidden, primary, branch, vocab);

        // 2. Verify every candidate (throwaway clones); keep the longest
        //    greedily-accepted prefix.
        let token_offset = cache.seq_len;
        let mut best_accepted = 0usize;
        let mut best_tokens: &[u32] = &[];
        for cand in &candidates {
            let a = verify_candidate(cfg, weights, &cache, primary, &cand.tokens, token_offset);
            forwards += 1;
            leaves += 1;
            if best_tokens.is_empty() || a > best_accepted {
                best_accepted = a;
                best_tokens = &cand.tokens;
            }
        }

        // 3. Commit: primary + accepted drafts; bonus carries to next step.
        let mut committed_step: Vec<u32> = Vec::with_capacity(1 + best_accepted);
        committed_step.push(primary);
        committed_step.extend_from_slice(&best_tokens[..best_accepted]);
        committed.extend_from_slice(&committed_step);
        accepted_drafts += best_accepted;
        steps += 1;

        // 4. Advance the real cache over exactly the committed tokens (linear-
        //    attention-safe), seeding next step from the bonus token + its hidden.
        let (bonus, next_hidden) =
            commit_forward(cfg, weights, &mut cache, &committed_step, token_offset);
        forwards += 1; // the commit forward (both arms pay it equally)
        primary = bonus;
        hidden = next_hidden;
    }
    let wall_s = t0.elapsed().as_secs_f64();
    clear_cache();

    ArmStats {
        label: label.to_string(),
        committed,
        steps,
        accepted_drafts,
        forwards,
        leaves,
        wall_s,
    }
}

/// Real-throughput stats for a production-faithful linear MTP chain at one depth.
struct DepthStats {
    depth: usize,
    committed: usize,
    steps: usize,
    accepted: usize,
    target_forwards: usize, // verify + recompute forwards (the wall-clock cost)
    wall_s: f64,
}

/// Production-faithful linear MTP decode at a fixed `max_depth`, measuring real
/// wall-clock throughput. Mirrors the runner's linear-attention path:
///   * drafts with the real `mtp_draft_tokens` (confidence gate + greedy argmax
///     drafting under default env);
///   * verifies `[primary] ++ draft` on a throwaway clone (1 forward);
///   * on FULL accept adopts the clone (recurrent state already correct);
///   * on PARTIAL accept recomputes the committed prefix on the real cache
///     (`forward_all_positions_update_cache`, the linear-attention-safe rollback,
///     +1 forward) — exactly what `recompute_committed_prefix` does in production.
///
/// Greedy target acceptance ⇒ deterministic trajectory, so depths are comparable.
///
/// The MTP head cache is reset per step (no cross-step head history). This is
/// applied identically at every depth, so the depth-vs-throughput comparison is
/// fair; only the absolute accept rate is a touch below the persistent-cache
/// production value.
fn run_linear_realistic(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt: &[u32],
    target_tokens: usize,
    max_depth: usize,
) -> DepthStats {
    // Uses the process-global env gate (default 0.90) via the real drafter.
    run_fixed_gate(
        cfg,
        weights,
        prompt,
        target_tokens,
        max_depth,
        f32::NAN, // sentinel: use env default
    )
}

/// Like `run_linear_realistic` but with an explicit fixed gate (`gate.is_nan()`
/// falls back to the env default), so multiple gate values can be compared in one
/// process without the env `OnceLock` pinning a single value.
fn run_fixed_gate(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt: &[u32],
    target_tokens: usize,
    max_depth: usize,
    gate: f32,
) -> DepthStats {
    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (mut primary, mut hidden) = chunked_prefill_with_final_hidden(
        cfg,
        weights,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );

    let mut committed = 0usize;
    let mut steps = 0usize;
    let mut accepted = 0usize;
    let mut target_forwards = 0usize;

    let t0 = Instant::now();
    while committed < target_tokens {
        // Draft via the real production drafter (fresh head cache each step).
        let mut head_cache = MlxKVCache::new(1);
        let (draft, _lp, _dist, _added, _m) = if gate.is_nan() {
            mtp_draft_tokens(
                weights,
                cfg,
                &hidden,
                primary,
                &mut head_cache,
                Some(max_depth),
                &mut rng,
            )
        } else {
            mtp_draft_tokens_gated(
                weights,
                cfg,
                &hidden,
                primary,
                &mut head_cache,
                Some(max_depth),
                &mut rng,
                gate,
            )
        };

        let token_offset = cache.seq_len;
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft.len());
        verify_input.push(primary);
        verify_input.extend_from_slice(&draft);

        let mut vclone = cache.clone();
        let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
            cfg,
            weights,
            &verify_input,
            &mut vclone,
            token_offset,
        );
        vclone.seq_len += verify_input.len();
        let predicted_arr = argmax(&logits_all, None);
        eval(&[&predicted_arr, &post_norm_all]);
        target_forwards += 1;
        let predicted = predicted_arr.data_u32();

        let mut a = 0usize;
        while a < draft.len() && predicted[a] == draft[a] {
            a += 1;
        }

        // Commit primary + accepted drafts; bonus carries forward.
        committed += 1 + a;
        accepted += a;
        steps += 1;

        if a == draft.len() {
            // Full accept: the verify clone's recurrent state is exactly the
            // committed prefix — adopt it (no extra forward).
            cache = vclone;
        } else {
            // Partial accept: roll the real cache forward over the committed
            // prefix only (linear-attention-safe recompute, +1 forward).
            let committed_step: Vec<u32> = verify_input[..1 + a].to_vec();
            forward_all_positions_update_cache(
                cfg,
                weights,
                &committed_step,
                &mut cache,
                token_offset,
            );
            cache.seq_len += committed_step.len();
            target_forwards += 1;
        }

        // Next primary + hidden = the verify position at index `a`.
        let next_hidden = slice_hidden_row(&post_norm_all, a, cfg.hidden_size);
        eval(&[&next_hidden]);
        primary = predicted[a];
        hidden = next_hidden;
    }
    let wall_s = t0.elapsed().as_secs_f64();
    clear_cache();

    DepthStats {
        depth: max_depth,
        committed,
        steps,
        accepted,
        target_forwards,
        wall_s,
    }
}

/// Result of an adaptive-gate run.
struct AdaptiveStats {
    stats: DepthStats,
    final_gate: f32,
    min_gate_seen: f32,
    max_gate_seen: f32,
}

/// Adaptive draft-confidence-gate controller (the candidate "best practice").
///
/// Auto-tunes the gate per workload by hill-climbing on a deterministic,
/// thermal-noise-free throughput proxy `committed / (target_forwards + steps*depth*r)`
/// (r = head/target forward cost ratio ≈ 0.10), bounded to [0.80, 0.95]. The
/// proxy uses only step counts, so the controller is reproducible and not fooled
/// by thermal drift. Correctness is never at risk — the gate only changes how far
/// ahead each step verifies.
fn run_linear_adaptive(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    prompt: &[u32],
    target_tokens: usize,
    max_depth: usize,
) -> AdaptiveStats {
    const GATE_MIN: f32 = 0.80;
    const GATE_MAX: f32 = 0.95;
    const STEP: f32 = 0.02;
    const WINDOW: usize = 24; // steps per hill-climb decision
    const R: f64 = 0.10; // head-forward cost relative to a target forward

    let mut cache = MlxKVCache::new(cfg.layer_count);
    let mut rng = Xorshift64::new(0);
    let (mut primary, mut hidden) = chunked_prefill_with_final_hidden(
        cfg,
        weights,
        prompt,
        &mut cache,
        DEFAULT_PREFILL_CHUNK,
        MlxSamplingRequest::new(MlxSamplingParams::greedy(), prompt),
        &mut rng,
    );

    let mut committed = 0usize;
    let mut steps = 0usize;
    let mut accepted = 0usize;
    let mut target_forwards = 0usize;

    let mut gate = 0.90f32;
    let mut dir = -STEP; // start by loosening
    let mut prev_score = -1.0f64;
    let (mut win_committed, mut win_forwards, mut win_steps) = (0usize, 0usize, 0usize);
    let (mut min_gate, mut max_gate) = (gate, gate);

    let t0 = Instant::now();
    while committed < target_tokens {
        let mut head_cache = MlxKVCache::new(1);
        let (draft, _lp, _dist, _added, _m) = mtp_draft_tokens_gated(
            weights,
            cfg,
            &hidden,
            primary,
            &mut head_cache,
            Some(max_depth),
            &mut rng,
            gate,
        );

        let token_offset = cache.seq_len;
        let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft.len());
        verify_input.push(primary);
        verify_input.extend_from_slice(&draft);

        let mut vclone = cache.clone();
        let (logits_all, post_norm_all) = forward_all_positions_with_post_norm(
            cfg,
            weights,
            &verify_input,
            &mut vclone,
            token_offset,
        );
        vclone.seq_len += verify_input.len();
        let predicted_arr = argmax(&logits_all, None);
        eval(&[&predicted_arr, &post_norm_all]);
        target_forwards += 1;
        win_forwards += 1;
        let predicted = predicted_arr.data_u32();

        let mut a = 0usize;
        while a < draft.len() && predicted[a] == draft[a] {
            a += 1;
        }

        committed += 1 + a;
        accepted += a;
        steps += 1;
        win_committed += 1 + a;
        win_steps += 1;

        if a == draft.len() {
            cache = vclone;
        } else {
            let committed_step: Vec<u32> = verify_input[..1 + a].to_vec();
            forward_all_positions_update_cache(
                cfg,
                weights,
                &committed_step,
                &mut cache,
                token_offset,
            );
            cache.seq_len += committed_step.len();
            target_forwards += 1;
            win_forwards += 1;
        }

        let next_hidden = slice_hidden_row(&post_norm_all, a, cfg.hidden_size);
        eval(&[&next_hidden]);
        primary = predicted[a];
        hidden = next_hidden;

        // Hill-climb the gate once per window on the deterministic throughput proxy.
        if win_steps >= WINDOW {
            let denom = win_forwards as f64 + win_steps as f64 * max_depth as f64 * R;
            let score = win_committed as f64 / denom;
            if prev_score > 0.0 && score < prev_score {
                dir = -dir; // window worsened → reverse search direction
            }
            prev_score = score;
            gate = (gate + dir).clamp(GATE_MIN, GATE_MAX);
            min_gate = min_gate.min(gate);
            max_gate = max_gate.max(gate);
            win_committed = 0;
            win_forwards = 0;
            win_steps = 0;
        }
    }
    let wall_s = t0.elapsed().as_secs_f64();
    clear_cache();

    AdaptiveStats {
        stats: DepthStats {
            depth: max_depth,
            committed,
            steps,
            accepted,
            target_forwards,
            wall_s,
        },
        final_gate: gate,
        min_gate_seen: min_gate,
        max_gate_seen: max_gate,
    }
}

fn parse_branch(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|p| p.trim().parse::<usize>().ok())
        .map(|v| v.max(1))
        .collect()
}

fn main() {
    let model_dir = env::args()
        .nth(1)
        .expect("Usage: tree_draft_probe <model_dir> [committed_tokens]");
    let target_tokens: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let prompt_len: usize = env::var("AX_TREE_PROMPT_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    // One or more tree schedules to sweep, ';'-separated (each comma-separated
    // per-depth branch factors). Each is compared against a linear chain of the
    // same depth.
    let schedules: Vec<Vec<usize>> = env::var("AX_TREE_SCHEDULES")
        .unwrap_or_else(|_| env::var("AX_TREE_BRANCH").unwrap_or_else(|_| "2,2,1,1,1".to_string()))
        .split(';')
        .map(parse_branch)
        .filter(|s| !s.is_empty())
        .collect();

    println!("Loading model from {model_dir}...");
    let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir)).expect("load artifacts");
    let cfg = ModelConfig::from_manifest(artifacts.manifest());
    let mut weights_owned = load_weights(&artifacts).expect("load weights");
    assert!(
        weights_owned.mtp.is_some(),
        "model has no MTP head — tree-draft probe requires an MTP sidecar"
    );
    let sidecar_depth = weights_owned.mtp.as_ref().map(|m| m.max_depth).unwrap_or(0);
    // The sidecar caps depth at its tuned `mtp_depth_max` (3 for 27B). The head is
    // a single recurrent module, so it can be applied deeper — raise the ceiling so
    // the depth sweep can probe past the conservative default.
    if let Ok(raw) = env::var("AX_PROBE_HEAD_MAX_DEPTH")
        && let Ok(d) = raw.trim().parse::<usize>()
        && let Some(mtp) = weights_owned.mtp.as_mut()
    {
        mtp.max_depth = d;
        println!("raised head.max_depth {sidecar_depth} -> {d}");
    }
    let weights = Arc::new(weights_owned);
    enable_compile();

    // Prompt: real tokenized prompt from AX_TREE_PROMPT_FILE (whitespace/comma
    // separated u32 token ids) when set — needed to land in the realistic
    // high-acceptance regime. Otherwise a synthetic id ramp (only valid for a
    // relative cross-check, NOT for absolute acceptance).
    let prompt: Vec<u32> = match env::var("AX_TREE_PROMPT_FILE") {
        Ok(path) => {
            let raw = std::fs::read_to_string(&path).expect("read AX_TREE_PROMPT_FILE");
            raw.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect()
        }
        Err(_) => (1..=(prompt_len as u32)).collect(),
    };
    assert!(!prompt.is_empty(), "empty prompt");
    println!(
        "prompt_tokens={}  target_committed={target_tokens}",
        prompt.len()
    );

    // ── Adaptive-gate validation (candidate best practice) ──────────────────
    // AX_ADAPTIVE_GATE=1 compares the adaptive-gate controller against fixed
    // gates {0.80, 0.85, 0.90, 0.98} at the same depth, to check the controller
    // auto-lands at/above the per-suite fixed optimum.
    if env::var("AX_ADAPTIVE_GATE").is_ok() {
        let depth: usize = env::var("AX_DEPTH_SWEEP")
            .ok()
            .and_then(|s| s.split(',').next().and_then(|x| x.trim().parse().ok()))
            .unwrap_or(2);
        println!("\n=== Adaptive-gate vs fixed gates (depth {depth}) ===");
        let _ = run_linear_realistic(&cfg, &weights, &prompt, 8, depth); // warmup
        let proxy = |s: &DepthStats| {
            s.committed as f64 / (s.target_forwards as f64 + s.steps as f64 * depth as f64 * 0.10)
        };
        println!(
            "  {:>12} {:>10} {:>10} {:>10}",
            "config", "tok/fwd", "proxy", "accept/st"
        );
        for g in [0.80f32, 0.85, 0.90, 0.98] {
            // Per-process OnceLock would pin the env gate; instead drive the gate
            // explicitly through the adaptive runner pinned to a constant.
            let s = run_fixed_gate(&cfg, &weights, &prompt, target_tokens, depth, g);
            println!(
                "  {:>12} {:>10.3} {:>10.4} {:>10.3}",
                format!("fixed {g:.2}"),
                s.committed as f64 / s.target_forwards.max(1) as f64,
                proxy(&s),
                s.accepted as f64 / s.steps.max(1) as f64,
            );
        }
        let a = run_linear_adaptive(&cfg, &weights, &prompt, target_tokens, depth);
        println!(
            "  {:>12} {:>10.3} {:>10.4} {:>10.3}   final_gate={:.2} range=[{:.2},{:.2}]",
            "ADAPTIVE",
            a.stats.committed as f64 / a.stats.target_forwards.max(1) as f64,
            proxy(&a.stats),
            a.stats.accepted as f64 / a.stats.steps.max(1) as f64,
            a.final_gate,
            a.min_gate_seen,
            a.max_gate_seen,
        );
        return;
    }

    // ── Depth-throughput sweep (the workable solution) ──────────────────────
    // AX_DEPTH_SWEEP="2,3,4,5,6" measures real tok/s for the production-faithful
    // linear MTP chain at each depth, to find the throughput-optimal draft depth.
    if let Ok(spec) = env::var("AX_DEPTH_SWEEP") {
        let sweep: Vec<usize> = spec
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .filter(|&d| d >= 1)
            .collect();
        println!("\n=== Depth-throughput sweep (production-faithful linear MTP) ===");
        // Warm up JIT (not measured).
        let _ = run_linear_realistic(
            &cfg,
            &weights,
            &prompt,
            8,
            *sweep.iter().max().unwrap_or(&3),
        );
        let results: Vec<DepthStats> = sweep
            .iter()
            .map(|&d| run_linear_realistic(&cfg, &weights, &prompt, target_tokens, d))
            .collect();
        let base_tps = results
            .iter()
            .find(|s| s.depth == 3)
            .map(|s| s.committed as f64 / s.wall_s);
        println!(
            "  {:>5} {:>10} {:>12} {:>10} {:>10} {:>8} {:>8}",
            "depth", "tok/s", "tok/fwd", "accept/st", "fwds/step", "wall_s", "vs_d3"
        );
        for s in &results {
            let tps = s.committed as f64 / s.wall_s;
            let tok_per_fwd = s.committed as f64 / s.target_forwards.max(1) as f64;
            let accept_st = s.accepted as f64 / s.steps.max(1) as f64;
            let fwds_step = s.target_forwards as f64 / s.steps.max(1) as f64;
            let vs = base_tps
                .map(|b| format!("{:.3}x", tps / b))
                .unwrap_or_default();
            println!(
                "  {:>5} {:>10.2} {:>12.3} {:>10.3} {:>10.3} {:>8.2} {:>8}",
                s.depth, tps, tok_per_fwd, accept_st, fwds_step, s.wall_s, vs
            );
        }
        let best = results
            .iter()
            .max_by(|a, b| {
                (a.committed as f64 / a.wall_s)
                    .partial_cmp(&(b.committed as f64 / b.wall_s))
                    .unwrap()
            })
            .unwrap();
        println!(
            "\n  throughput-optimal depth = {} ({:.2} tok/s){}",
            best.depth,
            best.committed as f64 / best.wall_s,
            base_tps
                .map(|b| format!(
                    ", {:.3}x vs depth-3",
                    (best.committed as f64 / best.wall_s) / b
                ))
                .unwrap_or_default(),
        );
        return;
    }

    let depths: Vec<usize> = {
        let mut d: Vec<usize> = schedules.iter().map(|s| s.len()).collect();
        d.sort_unstable();
        d.dedup();
        d
    };
    println!("schedules={schedules:?}  linear depths={depths:?}\n");

    // Warm up JIT (not measured).
    let _ = run_arm("warmup", &cfg, &weights, &prompt, 8, &[1usize]);

    // One linear baseline per distinct depth.
    let mut linear_by_depth: std::collections::HashMap<usize, ArmStats> =
        std::collections::HashMap::new();
    for &d in &depths {
        let arm = run_arm(
            &format!("linear-d{d}"),
            &cfg,
            &weights,
            &prompt,
            target_tokens,
            &vec![1usize; d],
        );
        linear_by_depth.insert(d, arm);
    }

    println!("\n=== Results (projected = single-forward tree, 1 verify/step) ===");
    println!(
        "  {:<14} {:>6} {:>10} {:>12} {:>8} {:>12}",
        "schedule", "leaves", "accept/stp", "proj_tok/fwd", "vs_lin", "wall_s"
    );
    for d in &depths {
        let lin = &linear_by_depth[d];
        println!(
            "  {:<14} {:>6} {:>10.3} {:>12.3} {:>8} {:>12.2}",
            format!("linear-d{d}"),
            1,
            lin.mean_accepted(),
            lin.effective_tpf_projected(),
            "1.000x",
            lin.wall_s,
        );
    }
    for sched in &schedules {
        let leaves: usize = sched.iter().product();
        let tree = run_arm("tree", &cfg, &weights, &prompt, target_tokens, sched);
        let lin = &linear_by_depth[&sched.len()];
        let n = lin.committed.len().min(tree.committed.len());
        let identical = lin.committed[..n] == tree.committed[..n];
        let ratio = tree.effective_tpf_projected() / lin.effective_tpf_projected();
        println!(
            "  {:<14} {:>6} {:>10.3} {:>12.3} {:>7.3}x {:>12.2}  {}",
            format!("{sched:?}"),
            leaves,
            tree.mean_accepted(),
            tree.effective_tpf_projected(),
            ratio,
            tree.wall_s,
            if identical { "ok" } else { "TRAJ-MISMATCH" },
        );
    }
    println!();
    println!("  vs_lin = projected single-forward tree tok/fwd ÷ same-depth linear tok/fwd.");
    println!("  Phase A pays `leaves` real forwards/step; projected assumes Phase B tree-mask.");
}
