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
//!
//! ## GPU-native sampling
//!
//! The denoise step keeps token state and sampling logic on the GPU to avoid
//! per-step GPU→CPU synchronisation. Entropy-bound acceptance is computed via
//! `argsort` + `cumsum` + `where_cond`, and convergence stability is measured
//! with `not_equal` + `sum_axis`, materialising only scalar counters.
//! Convergence is checked every `convergence_check_interval` steps (default 4)
//! to further reduce sync overhead.

use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxDtype, add, argmax, argsort_axis, astype, cumsum, divide, eval, less_equal, log,
    matmul, multiply, negative, not_equal, reshape, rms_norm, softmax, sum_axis, take_along_axis,
    where_cond,
};

use crate::kv_cache::MlxKVCache;
use crate::model::{
    DiffusionConfig, FinalLogitsMode, ModelConfig, compute_per_layer_inputs_arr, embed_tokens,
    embed_tokens_arr, finalize_lm_head_logits, layer_forward_bidirectional, shared,
};
use crate::sampling::Xorshift64;
use crate::weights::ModelWeights;

/// Result of generating one diffusion block, including telemetry.
pub(crate) struct DiffusionBlockResult {
    /// The committed token IDs from this block.
    pub tokens: Vec<u32>,
    /// Number of denoise steps executed (may be < max if converged early).
    pub denoise_steps: u32,
    /// Whether the canvas converged before hitting max_denoise_steps.
    pub converged: bool,
    /// Total wall time spent in the denoise loop (microseconds).
    pub denoise_wall_us: u32,
    /// Wall time for the causal commit pass (microseconds).
    pub commit_wall_us: u32,
    /// Total wall time for the entire block generation (microseconds).
    pub block_wall_us: u32,
}

// Mutable state of the canvas being denoised.
//
// Token and argmax state is kept on GPU as `MlxArray` to avoid per-step
// GPU→CPU→GPU round-trips. Only scalar convergence counters cross the
// device boundary.
struct DiffusionCanvas {
    /// Current token IDs, 1-D `[canvas_size]` u32 on GPU.
    tokens_gpu: MlxArray,
    canvas_size: usize,
    /// Previous step's argmax predictions, 1-D `[canvas_size]` u32 on GPU.
    /// `None` on the very first step (no previous prediction to compare).
    argmax_canvas: Option<MlxArray>,
    stable_count: usize,
    mean_entropy: f32,
    step: usize,
    converged: bool,
    prev_self_cond_embed: Option<MlxArray>,
    /// Number of positions accepted in the last denoise step.
    accepted_count: usize,
    /// Fraction of positions accepted (accepted_count / canvas_size).
    /// Used for adaptive convergence detection.
    acceptance_rate: f32,
    /// Mean entropy from the previous check step, for plateau detection.
    prev_mean_entropy: f32,
}

// Initialize a canvas with uniformly random token IDs on GPU.
fn init_canvas(canvas_size: usize, vocab_size: usize, rng: &mut Xorshift64) -> DiffusionCanvas {
    let tokens: Vec<u32> = (0..canvas_size)
        .map(|_| (rng.next_u64() % vocab_size as u64) as u32)
        .collect();
    let tokens_gpu = MlxArray::from_raw_data(
        tokens.as_ptr() as *const u8,
        std::mem::size_of_val(&tokens[..]),
        &[canvas_size as i32],
        MlxDtype::Uint32,
    );
    DiffusionCanvas {
        tokens_gpu,
        canvas_size,
        argmax_canvas: None,
        stable_count: 0,
        mean_entropy: f32::MAX,
        step: 0,
        converged: false,
        prev_self_cond_embed: None,
        accepted_count: 0,
        acceptance_rate: 1.0, // start at 100% (all positions changing)
        prev_mean_entropy: f32::MAX,
    }
}

// Check whether the canvas has converged.
//
// Convergence triggers when ANY of these criteria are met:
//
// 1. **Strict criteria** (original): argmax unchanged for `convergence_steps`
//    consecutive check steps AND mean entropy below `entropy_threshold`.
//
// 2. **Acceptance rate criteria** (adaptive): acceptance rate drops below
//    `acceptance_rate_threshold` (default 1%). When almost no positions are
//    being updated, the model has converged regardless of absolute entropy.
//
// 3. **Entropy plateau criteria**: entropy has stopped decreasing significantly
//    (delta < 0.001) after step 16, indicating diminishing returns.
fn check_convergence(canvas: &DiffusionCanvas, cfg: &DiffusionConfig) -> bool {
    // Strict criteria: traditional argmax stability + low entropy.
    let strict_converged =
        canvas.stable_count >= cfg.convergence_steps && canvas.mean_entropy < cfg.entropy_threshold;

    // Acceptance rate criteria: almost no positions being updated.
    let acceptance_converged = canvas.acceptance_rate < cfg.acceptance_rate_threshold;

    // Entropy plateau criteria: entropy stalled after warmup period.
    // Only check after step 16 to allow initial exploration.
    let entropy_delta = (canvas.prev_mean_entropy - canvas.mean_entropy).abs();
    let plateau_converged = entropy_delta < 0.001 && canvas.step >= 16;

    strict_converged || acceptance_converged || plateau_converged
}

// Linear temperature schedule from `temp_start` (step 0) to `temp_end` (max steps).
fn temperature_at_step(step: usize, cfg: &DiffusionConfig) -> f32 {
    let t = step as f32 / cfg.max_denoise_steps.max(1) as f32;
    cfg.temp_start + (cfg.temp_end - cfg.temp_start) * t
}

// Run one denoise step: bidirectional forward → per-position logits →
// GPU-native entropy-bound sampling → convergence check → self-conditioning.
//
// Token state stays on GPU throughout; only scalar convergence counters
// (`changed_count`, `mean_entropy`) are materialised to CPU. On non-check
// steps (controlled by `convergence_check_interval`), the convergence
// materialisation is skipped entirely.
#[allow(clippy::too_many_arguments)]
fn denoise_step(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &MlxKVCache,
    canvas: &mut DiffusionCanvas,
    step: usize,
    token_offset: usize,
    embed_table: Option<&MlxArray>,
) {
    let temperature = temperature_at_step(step, diff_cfg);
    let is_check_step = step.is_multiple_of(diff_cfg.convergence_check_interval);

    // Bidirectional forward over canvas → logits [1, canvas_size, vocab_size].
    let logits = forward_bidirectional(cfg, weights, &canvas.tokens_gpu, cache, token_offset);

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
    // entropy: [1, canvas_size]
    let entropy = negative(&sum_axis(&p_log_p, -1, false, None), None);

    // Argmax per position → [1, canvas_size] (argmax reduces last axis).
    let argmax_2d = argmax(&prob, None);
    // Reshape to 1-D [canvas_size] for token-state tracking.
    let argmax_1d = reshape(&argmax_2d, &[canvas.canvas_size as i32], None);

    // ── GPU-native entropy-bound sampling ────────────────────────────
    //
    // Sort positions by entropy ascending (most confident first), then
    // accept positions greedily until cumulative entropy exceeds the
    // budget. This is equivalent to the previous CPU-side sort+loop but
    // runs entirely as lazy MLX graph nodes.

    // sorted_positions: [1, canvas_size] — indices that sort entropy ascending.
    let sorted_positions = argsort_axis(&entropy, -1, None);

    // sorted_entropy: [1, canvas_size] — entropy values in sorted order.
    let sorted_entropy = take_along_axis(&entropy, &sorted_positions, -1, None);

    // cum_entropy: [1, canvas_size] — inclusive prefix sum of sorted entropy.
    let cum_entropy = cumsum(&sorted_entropy, -1, false, true, None);

    // bound_scalar: [] f32 — the entropy budget.
    let bound_scalar = MlxArray::from_f32(diff_cfg.entropy_bound);

    // accepted_sorted: [1, canvas_size] bool — positions within budget.
    // Using <= ensures at least the first (lowest-entropy) position is
    // accepted when its entropy ≤ bound, guaranteeing progress.
    let accepted_sorted = less_equal(&cum_entropy, &bound_scalar, None);

    // Inverse-sort the acceptance mask back to original position order.
    // inverse_sort[i] = rank of position i in the sorted ordering.
    let inverse_sort = argsort_axis(&sorted_positions, -1, None);
    let accept_mask = take_along_axis(&accepted_sorted, &inverse_sort, -1, None);
    let accept_mask_1d = reshape(&accept_mask, &[canvas.canvas_size as i32], None);

    // Token update: accepted positions keep current tokens, rejected
    // positions adopt the model's argmax prediction.
    let new_tokens = where_cond(&accept_mask_1d, &canvas.tokens_gpu, &argmax_1d, None);

    // ── Acceptance rate tracking ─────────────────────────────────────
    //
    // Count positions accepted (where accept_mask is true) for adaptive
    // convergence detection. When acceptance rate drops near zero, the
    // model has converged even if absolute entropy is still above threshold.
    let accepted_f32 = astype(&accept_mask_1d, MlxDtype::Float32, None);
    let accepted_sum = sum_axis(&accepted_f32, -1, false, None);
    eval(&[&accepted_sum]);
    canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
    canvas.acceptance_rate = canvas.accepted_count as f32 / canvas.canvas_size as f32;

    // ── Convergence detection (check steps only) ─────────────────────
    //
    // On check steps, materialise two scalars (changed_count, mean_entropy)
    // to update the stability counter and convergence flag. On non-check
    // steps, skip the materialisation entirely.

    if is_check_step {
        // Argmax stability: count positions that differ from previous step.
        if let Some(prev_argmax) = &canvas.argmax_canvas {
            let changed = not_equal(&argmax_1d, prev_argmax, None);
            let changed_f32 = astype(&changed, MlxDtype::Float32, None);
            let changed_count = sum_axis(&changed_f32, -1, false, None);

            // Mean entropy: scalar mean of per-position entropy.
            let total_entropy = sum_axis(&entropy, -1, false, None);
            let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
            let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);

            eval(&[&changed_count, &mean_entropy]);

            let changed_val = changed_count.data_f32()[0];
            canvas.stable_count = if changed_val == 0.0 {
                canvas.stable_count + 1
            } else {
                0
            };
            canvas.prev_mean_entropy = canvas.mean_entropy;
            canvas.mean_entropy = mean_entropy.data_f32()[0];
        } else {
            // First step: no previous argmax to compare; force unstable.
            canvas.stable_count = 0;
            let total_entropy = sum_axis(&entropy, -1, false, None);
            let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
            let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);
            eval(&[&mean_entropy]);
            canvas.prev_mean_entropy = canvas.mean_entropy;
            canvas.mean_entropy = mean_entropy.data_f32()[0];
        }
        canvas.converged = check_convergence(canvas, diff_cfg);
    }

    // Store current argmax for next step's stability comparison.
    canvas.argmax_canvas = Some(argmax_1d);
    canvas.tokens_gpu = new_tokens;
    canvas.step = step;

    // Self-conditioning: GPU matmul of prob × embed_table.
    // prob: [1, canvas_size, vocab_size]
    // embed_table: [vocab_size, hidden_size]
    // result: [1, canvas_size, hidden_size]
    //
    // This embedding is fed back through a gated MLP (when self-conditioning
    // weights are available in the checkpoint) and added to canvas embeddings
    // before the next denoise step. Without checkpoint-specific weights the
    // feedback is stored but not applied.
    if diff_cfg.self_conditioning
        && let Some(embed) = embed_table
    {
        canvas.prev_self_cond_embed = Some(matmul(&prob, embed, None));
    }
}

// Pre-compute the full embedding table once for reuse across all denoise steps.
//
// Returns `[vocab_size, hidden_size]` as a bf16 MLX array on GPU.
// This avoids re-dequantizing the ~3.5 GB embedding table on every step.
fn compute_embed_table(weights: &ModelWeights, cfg: &ModelConfig) -> MlxArray {
    let all_ids: Vec<u32> = (0..cfg.vocab_size as u32).collect();
    let embed_all = embed_tokens(&all_ids, &weights.token_embedding, cfg.hidden_size);
    astype(&embed_all, MlxDtype::Bfloat16, None)
}

// Commit the canvas via a causal encoder pass.
//
// Materialises GPU tokens to CPU for the causal forward (which requires
// `&[u32]`), writes KV cache, and returns the committed token IDs.
fn commit_block(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    canvas: &DiffusionCanvas,
    token_offset: usize,
) -> Vec<u32> {
    // Materialise GPU tokens to CPU for the causal forward.
    eval(&[&canvas.tokens_gpu]);
    let tokens: Vec<u32> = canvas.tokens_gpu.data_u32().to_vec();
    let _logits = crate::model::forward(cfg, weights, &tokens, cache, token_offset);
    tokens
}

/// Generate one diffusion block: denoise → commit.
///
/// The prompt is assumed to be already prefilled into the cache.
/// Returns telemetry along with the committed tokens.
pub(crate) fn generate_diffusion_block(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    rng: &mut Xorshift64,
    token_offset: usize,
    embed_table_cache: &mut Option<MlxArray>,
) -> DiffusionBlockResult {
    let block_start = Instant::now();
    let mut canvas = init_canvas(diff_cfg.canvas_size, cfg.vocab_size, rng);

    if diff_cfg.self_conditioning && embed_table_cache.is_none() {
        *embed_table_cache = Some(compute_embed_table(weights, cfg));
    }
    let embed_table = if diff_cfg.self_conditioning {
        embed_table_cache.as_ref()
    } else {
        None
    };

    let denoise_start = Instant::now();
    let mut steps_executed = 0_u32;
    for step in 0..diff_cfg.max_denoise_steps {
        denoise_step(
            cfg,
            diff_cfg,
            weights,
            cache,
            &mut canvas,
            step,
            token_offset,
            embed_table,
        );
        steps_executed += 1;
        if canvas.converged {
            break;
        }
    }
    let denoise_wall_us = elapsed_us(denoise_start);

    let commit_start = Instant::now();
    let tokens = commit_block(cfg, weights, cache, &canvas, token_offset);
    let commit_wall_us = elapsed_us(commit_start);
    let block_wall_us = elapsed_us(block_start);

    DiffusionBlockResult {
        tokens,
        denoise_steps: steps_executed,
        converged: canvas.converged,
        denoise_wall_us,
        commit_wall_us,
        block_wall_us,
    }
}

fn elapsed_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

// Run the bidirectional forward pass over canvas tokens.
//
// Unlike the causal `forward()`, this uses `layer_forward_bidirectional` for
// each layer — no KV cache writes, bidirectional attention over the canvas.
// Accepts a 1-D `[canvas_size]` u32 `MlxArray` (may be GPU-resident/lazy).
// Returns per-position logits `[1, canvas_size, vocab_size]`.
fn forward_bidirectional(
    cfg: &ModelConfig,
    weights: &ModelWeights,
    token_ids: &MlxArray,
    cache: &MlxKVCache,
    token_offset: usize,
) -> MlxArray {
    // Embed tokens directly from GPU array.
    let mut hidden = embed_tokens_arr(token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = crate::model::shared::scale_hidden_pub(&hidden, scale);
    }

    // Compute per-layer inputs (Gemma4 per-layer embeddings).
    let per_layer_inputs = compute_per_layer_inputs_arr(cfg, weights, token_ids, &hidden);

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
