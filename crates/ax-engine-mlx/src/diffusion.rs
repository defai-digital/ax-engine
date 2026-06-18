//! DiffusionGemma block-autoregressive generation.
//!
//! DiffusionGemma uses the same Gemma4 MoE backbone in two modes:
//! - **Encoder** (causal): prompt prefill and block commit — identical to AR forward.
//! - **Denoiser** (bidirectional): iterative refinement of a fixed-size canvas.
//!
//! Per-block generation loop:
//! 1. Prefill prompt through causal encoder (writes KV cache).
//! 2. Initialize random canvas of `canvas_size` token IDs.
//! 3. Denoise loop: run bidirectional forward over canvas, apply entropy-bound
//!    sampling, check convergence.
//! 4. Commit: run causal encoder pass over canvas, write KV, emit tokens.
//! 5. Repeat from step 2 for the next block.

use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, astype, divide, eval, log, multiply, negative, rms_norm,
    softmax, sum_axis,
};

use crate::kv_cache::MlxKVCache;
use crate::model::{
    DiffusionConfig, FinalLogitsMode, ModelConfig, compute_per_layer_inputs_arr, embed_tokens,
    finalize_lm_head_logits, layer_forward_bidirectional, shared,
};
use crate::sampling::Xorshift64;
use crate::weights::ModelWeights;

/// Mutable state of the canvas being denoised.
pub struct DiffusionCanvas {
    /// Current canvas token IDs.
    pub tokens: Vec<u32>,
    /// Canvas size.
    pub canvas_size: usize,
    /// Best-guess argmax prediction from the last denoise step.
    pub argmax_canvas: Vec<u32>,
    /// Consecutive steps where `argmax_canvas` has been unchanged.
    pub stable_count: usize,
    /// Mean per-position entropy from the last denoise step.
    pub mean_entropy: f32,
    /// Current denoise step index.
    pub step: usize,
    /// Whether the canvas has converged.
    pub converged: bool,
    /// Probability-weighted token embeddings from last denoise step
    /// (used for self-conditioning feedback; `None` on first step).
    pub prev_self_cond_embed: Option<MlxArray>,
}

/// Phase of the diffusion generation state machine.
pub enum DiffusionPhase {
    /// Causal encoder pass over the prompt (prefill).
    Prefill,
    /// Bidirectional denoiser loop over the canvas.
    Denoise,
    /// Causal encoder pass to write KV + emit committed tokens.
    Commit,
}

/// High-level state of the diffusion generator.
pub struct DiffusionState {
    pub phase: DiffusionPhase,
    pub canvas: DiffusionCanvas,
    pub committed_tokens: Vec<u32>,
}

impl DiffusionState {
    pub fn new(canvas_size: usize) -> Self {
        Self {
            phase: DiffusionPhase::Prefill,
            canvas: DiffusionCanvas::empty(canvas_size),
            committed_tokens: Vec::new(),
        }
    }
}

impl DiffusionCanvas {
    fn empty(canvas_size: usize) -> Self {
        Self {
            tokens: Vec::new(),
            canvas_size,
            argmax_canvas: Vec::new(),
            stable_count: 0,
            mean_entropy: f32::MAX,
            step: 0,
            converged: false,
            prev_self_cond_embed: None,
        }
    }
}

/// Initialize a canvas with uniformly random token IDs.
pub fn init_canvas(canvas_size: usize, vocab_size: usize, rng: &mut Xorshift64) -> DiffusionCanvas {
    let tokens: Vec<u32> = (0..canvas_size)
        .map(|_| (rng.next_u64() % vocab_size as u64) as u32)
        .collect();
    DiffusionCanvas {
        tokens,
        canvas_size,
        argmax_canvas: vec![0; canvas_size],
        stable_count: 0,
        mean_entropy: f32::MAX,
        step: 0,
        converged: false,
        prev_self_cond_embed: None,
    }
}

/// Check whether the canvas has converged.
///
/// Convergence requires:
/// 1. `argmax_canvas` unchanged for `convergence_steps` consecutive steps, AND
/// 2. Mean per-position entropy below `entropy_threshold`.
pub fn check_convergence(canvas: &DiffusionCanvas, cfg: &DiffusionConfig) -> bool {
    canvas.stable_count >= cfg.convergence_steps && canvas.mean_entropy < cfg.entropy_threshold
}

/// Linear temperature schedule from `temp_start` (step 0) to `temp_end` (max steps).
fn temperature_at_step(step: usize, cfg: &DiffusionConfig) -> f32 {
    let t = step as f32 / cfg.max_denoise_steps.max(1) as f32;
    cfg.temp_start + (cfg.temp_end - cfg.temp_start) * t
}

/// Run one denoise step: bidirectional forward → per-position logits →
/// entropy-bound sampling → convergence check → self-conditioning.
#[allow(clippy::too_many_arguments)]
pub fn denoise_step(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &MlxKVCache,
    canvas: &mut DiffusionCanvas,
    step: usize,
    token_offset: usize,
    rng: &mut Xorshift64,
) {
    let canvas_size = canvas.canvas_size;
    let temperature = temperature_at_step(step, diff_cfg);

    // Bidirectional forward over canvas → logits [1, canvas_size, vocab_size].
    let logits = forward_bidirectional(cfg, weights, &canvas.tokens, cache, token_offset);

    // Temperature-scale: divide by T.
    let scaled = if (temperature - 1.0).abs() > 1e-6 {
        divide(&logits, &MlxArray::from_f32(temperature), None)
    } else {
        logits
    };

    // Softmax over vocab (last axis) → [1, canvas_size, vocab_size].
    let prob = softmax(&scaled, -1, None);

    // Per-position entropy: H(p) = -sum(p * log(p)) along vocab axis.
    // Add epsilon before log to avoid log(0).
    let eps = MlxArray::from_f32(1e-10);
    let log_prob = log(&add(&prob, &eps, None), None);
    let p_log_p = multiply(&prob, &log_prob, None);
    let entropy = negative(&sum_axis(&p_log_p, -1, false, None), None);

    // Argmax per position → [1, canvas_size] (argmax reduces last axis).
    let argmax_tokens = argmax(&prob, None);

    // Materialize entropy and argmax for Rust-side sampling logic.
    eval(&[&entropy, &argmax_tokens]);
    let entropy_data = entropy.data_f32();
    let argmax_data = argmax_tokens.data_u32();

    // Entropy-bound position selection:
    // Sort positions by entropy ascending (most confident first).
    let mut position_order: Vec<usize> = (0..canvas_size).collect();
    position_order.sort_by(|&a, &b| entropy_data[a].total_cmp(&entropy_data[b]));

    // Greedily accept lowest-entropy positions until cumulative entropy
    // exceeds the budget.
    let mut accept_mask = vec![false; canvas_size];
    let mut cumulative_entropy = 0.0_f32;
    for &pos in &position_order {
        if cumulative_entropy + entropy_data[pos] > diff_cfg.entropy_bound {
            break;
        }
        accept_mask[pos] = true;
        cumulative_entropy += entropy_data[pos];
    }
    // Always accept at least one position to guarantee progress.
    if !accept_mask.iter().any(|&v| v) {
        accept_mask[position_order[0]] = true;
    }

    // Renoise rejected positions with fresh random tokens.
    let mut new_tokens = canvas.tokens.clone();
    for pos in 0..canvas_size {
        if !accept_mask[pos] {
            new_tokens[pos] = (rng.next_u64() % cfg.vocab_size as u64) as u32;
        }
    }

    // Update argmax canvas and check stability.
    let new_argmax: Vec<u32> = (0..canvas_size).map(|i| argmax_data[i]).collect();
    let unchanged = new_argmax == canvas.argmax_canvas;
    canvas.stable_count = if unchanged {
        canvas.stable_count + 1
    } else {
        0
    };
    canvas.argmax_canvas = new_argmax;

    // Compute mean entropy.
    let total_entropy: f32 = entropy_data.iter().sum();
    canvas.mean_entropy = total_entropy / canvas_size as f32;
    canvas.tokens = new_tokens;
    canvas.step = step;
    canvas.converged = check_convergence(canvas, diff_cfg);

    // Self-conditioning: compute probability-weighted token embeddings.
    // weighted_embed = sum(prob * embed_tokens, axis=vocab)
    // Shape: [1, canvas_size, hidden_size]
    //
    // This embedding is fed back through a gated MLP (when self-conditioning
    // weights are available in the checkpoint) and added to canvas embeddings
    // before the next denoise step. Without checkpoint-specific weights the
    // feedback is stored but not applied — the structural path is ready for
    // when DiffusionGemma weights ship.
    if diff_cfg.self_conditioning {
        canvas.prev_self_cond_embed = compute_self_conditioning_embed(&prob, weights, cfg);
    }
}

/// Compute the self-conditioning embedding: probability-weighted average of
/// token embeddings.
///
/// Returns `[1, canvas_size, hidden_size]` — the mean predicted embedding per
/// position. When self-conditioning gate MLP weights are available in the
/// checkpoint, the caller should pass this through the gate and add the result
/// to canvas embeddings before the next denoise step.
fn compute_self_conditioning_embed(
    prob: &MlxArray,
    weights: &ModelWeights,
    cfg: &ModelConfig,
) -> Option<MlxArray> {
    // prob: [1, canvas_size, vocab_size] (f32, materialized after softmax).
    let canvas_size = prob.shape()[1] as usize;
    let vocab_size = cfg.vocab_size;
    let hidden_size = cfg.hidden_size;
    let prob_data = prob.data_f32();

    // Build full embedding table on CPU: [vocab_size, hidden_size].
    let all_ids: Vec<u32> = (0..vocab_size as u32).collect();
    let embed_all = embed_tokens(&all_ids, &weights.token_embedding, hidden_size);
    let embed_all = astype(&embed_all, MlxDtype::Float32, None);
    eval(&[&embed_all]);
    let embed_data = embed_all.data_f32();
    // embed_all shape: [1, vocab_size, hidden_size] → flatten to [vocab_size, hidden_size].
    let embed_table = &embed_data[..vocab_size * hidden_size];

    // Weighted sum: for each canvas position, sum_v(prob[v] * embed[v]).
    let mut weighted = vec![0.0_f32; canvas_size * hidden_size];
    for pos in 0..canvas_size {
        let p_offset = pos * vocab_size;
        let w_offset = pos * hidden_size;
        for v in 0..vocab_size {
            let p = prob_data[p_offset + v];
            if p == 0.0 {
                continue;
            }
            let e_offset = v * hidden_size;
            for h in 0..hidden_size {
                weighted[w_offset + h] += p * embed_table[e_offset + h];
            }
        }
    }

    let result = MlxArray::from_raw_data(
        weighted.as_ptr() as *const u8,
        weighted.len() * std::mem::size_of::<f32>(),
        &[1, canvas_size as i32, hidden_size as i32],
        MlxDtype::Float32,
    );
    Some(result)
}

/// Commit the canvas via a causal encoder pass.
///
/// Runs the standard causal forward over the canvas tokens, writes KV cache,
/// and returns the committed token IDs (the canvas tokens themselves).
pub fn commit_block(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    canvas: &DiffusionCanvas,
    token_offset: usize,
) -> Vec<u32> {
    let _logits = crate::model::forward(cfg, weights, &canvas.tokens, cache, token_offset);
    canvas.tokens.clone()
}

/// Generate one diffusion block: denoise → commit.
///
/// The prompt is assumed to be already prefilled into the cache.
/// Returns the committed tokens from this block.
pub fn generate_diffusion_block(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    rng: &mut Xorshift64,
    token_offset: usize,
) -> Vec<u32> {
    let mut canvas = init_canvas(diff_cfg.canvas_size, cfg.vocab_size, rng);

    for step in 0..diff_cfg.max_denoise_steps {
        denoise_step(
            cfg,
            diff_cfg,
            weights,
            cache,
            &mut canvas,
            step,
            token_offset,
            rng,
        );
        if canvas.converged {
            break;
        }
    }

    commit_block(cfg, weights, cache, &canvas, token_offset)
}

/// Entry point for diffusion-based generation.
///
/// Prefills the prompt, then generates blocks until `max_tokens` is reached.
pub fn generate_diffusion(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    prompt_tokens: &[u32],
    max_tokens: usize,
    rng: &mut Xorshift64,
) -> Vec<u32> {
    // Phase 1: Prefill prompt (causal encoder pass, writes KV cache).
    if !prompt_tokens.is_empty() {
        let _ = crate::model::forward(cfg, weights, prompt_tokens, cache, 0);
    }

    let mut all_tokens = Vec::with_capacity(max_tokens);
    let mut token_offset = prompt_tokens.len();

    // Phase 2: Generate blocks until max_tokens is reached.
    while all_tokens.len() < max_tokens {
        let block = generate_diffusion_block(cfg, diff_cfg, weights, cache, rng, token_offset);
        token_offset += block.len();
        all_tokens.extend_from_slice(&block);
    }

    all_tokens.truncate(max_tokens);
    all_tokens
}

/// Run the bidirectional forward pass over canvas tokens.
///
/// Unlike the causal `forward()`, this uses `layer_forward_bidirectional` for
/// each layer — no KV cache writes, bidirectional attention over the canvas.
/// Returns per-position logits `[1, canvas_size, vocab_size]`.
pub fn forward_bidirectional(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &[u32],
    cache: &MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    let _seq = token_ids.len();

    // Embed tokens.
    let mut hidden = embed_tokens(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = crate::model::shared::scale_hidden_pub(&hidden, scale);
    }

    // Compute per-layer inputs (Gemma4 per-layer embeddings).
    let ids_1d = MlxArray::from_raw_data(
        token_ids.as_ptr() as *const u8,
        std::mem::size_of_val(token_ids),
        &[token_ids.len() as i32],
        MlxDtype::Uint32,
    );
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, &ids_1d, &hidden);

    // Run each layer with bidirectional attention.
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        hidden = layer_forward_bidirectional(cfg, layer_w, &hidden, cache, li, token_offset, pli);
    }

    // Final norm + LM head → logits [1, seq, vocab_size].
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = shared::qw(&normed, &weights.lm_head);
    finalize_lm_head_logits(cfg, &logits, FinalLogitsMode::Full)
}

/// Generate one block of diffusion tokens and return them as a queue.
///
/// This is the integration entry point for the runner: it generates a full
/// block of `canvas_size` tokens via the diffusion process and returns them
/// for the runner to buffer and drain one at a time.
pub fn generate_diffusion_token_batch(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    rng: &mut Xorshift64,
    token_offset: usize,
) -> Vec<u32> {
    generate_diffusion_block(cfg, diff_cfg, weights, cache, rng, token_offset)
}
