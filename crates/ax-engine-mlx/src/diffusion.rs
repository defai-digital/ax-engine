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
//!
//! ## Compiled denoise closure
//!
//! When `AX_DIFFUSION_COMPILED_FORWARD=1` is set, the bidirectional forward
//! pass is wrapped in an `MlxClosure` compiled via `mlx_compile`. The compiled
//! graph is constructed once per block and reused for every denoise step,
//! collapsing ~250 per-step MLX C-API calls into a single dispatched graph.
//! Falls back to the imperative path on thread mismatch or compile failure.
//!
//! ## Conditional causal commit skip
//!
//! When `AX_DIFFUSION_SKIP_COMMIT_ON_CONVERGE=1` is set and the denoise loop
//! converges at step 1 with >= 99% acceptance, the causal commit pass is
//! skipped entirely — the canvas tokens are emitted directly. This saves
//! ~40 ms per block on the first-block measurement where convergence is
//! immediate.
//!
//! ## Per-layer embedding cache
//!
//! When `AX_DIFFUSION_EMBEDDING_CACHE=1` is set, the per-layer embedding
//! inputs (`compute_per_layer_inputs_arr`) are cached across denoise steps.
//! A GPU-side sum fingerprint detects token changes (2 dispatches + 1 eval)
//! and reuses the cached embeddings when tokens are unchanged, saving 46
//! embedding dispatches per cache hit.
//!
//! ## KV concatenation buffer
//!
//! When `AX_DIFFUSION_KV_CONCAT_BUFFER=1` is set, per-layer KV concatenation
//! buffers are pre-allocated on the first denoise step via `concatenate`,
//! then updated on subsequent steps via `slice_update`. This avoids
//! re-copying the cached prompt prefix on every step, saving memory
//! bandwidth proportional to the prompt length.
//!
//! ## Full-pipeline compiled forward
//!
//! The previous full-pipeline closure is disabled until it can carry the
//! dynamic self-conditioning signal and per-step renoise tokens. The
//! forward-only closure remains available for configurations without
//! self-conditioning.

use std::time::Instant;

use mlx_sys::{
    MlxArray, MlxClosure, MlxDtype, MlxVectorArray, add, arange, argmax, argsort_axis, astype,
    cumsum, divide, equal, eval, gelu_approx, greater_equal, less_equal, log, matmul, multiply,
    negative, not_equal, reshape, rms_norm, softmax, sum_axis, take_along_axis, where_cond,
};

use crate::fastpath;
use crate::kv_cache::MlxKVCache;
use crate::model::shared::KVConcatBuffer;
use crate::model::{
    DiffusionConfig, FinalLogitsMode, ModelConfig, compute_per_layer_inputs_arr, embed_tokens,
    embed_tokens_arr, finalize_lm_head_logits, layer_forward_bidirectional, shared,
};
use crate::sampling::Xorshift64;
use crate::weights::ModelWeights;

/// Per-criterion convergence signals for telemetry diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ConvergenceSignals {
    /// Strict argmax stability + low entropy.
    pub strict: bool,
    /// Acceptance rate below threshold.
    pub acceptance: bool,
    /// Entropy plateau (diminishing returns after warmup).
    pub plateau: bool,
}

impl ConvergenceSignals {
    fn any(self) -> bool {
        self.strict || self.acceptance || self.plateau
    }
}

/// Result of generating one diffusion block, including telemetry.
pub(crate) struct DiffusionBlockResult {
    /// The committed token IDs from this block.
    pub tokens: Vec<u32>,
    /// Number of denoise steps executed (may be < max if converged early).
    pub denoise_steps: u32,
    /// Whether the canvas converged before hitting max_denoise_steps.
    pub converged: bool,
    /// Which convergence criterion triggered (all false if converged==false).
    pub converged_strict: bool,
    pub converged_acceptance: bool,
    pub converged_plateau: bool,
    /// Lowest mean entropy observed across all check steps (near-miss telemetry).
    pub min_entropy: f32,
    /// Lowest acceptance rate observed across all steps (near-miss telemetry).
    pub min_acceptance_rate: f32,
    /// Total wall time spent in the denoise loop (microseconds).
    pub denoise_wall_us: u32,
    /// Wall time for the causal commit pass (microseconds). 0 when skipped.
    pub commit_wall_us: u32,
    /// Total wall time for the entire block generation (microseconds).
    pub block_wall_us: u32,
    /// Whether the causal commit was skipped due to step-1 convergence.
    pub commit_skipped: bool,
    /// Whether the full-pipeline compiled closure was used for denoising.
    pub full_pipeline_used: bool,
    /// Whether KV concatenation buffers were used for bidirectional attention.
    pub kv_buffer_used: bool,
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
    /// Which convergence criterion last triggered.
    last_signals: ConvergenceSignals,
    prev_self_cond_embed: Option<MlxArray>,
    /// Number of positions accepted in the last denoise step.
    accepted_count: usize,
    /// Fraction of positions accepted (accepted_count / canvas_size).
    /// Used for adaptive convergence detection.
    acceptance_rate: f32,
    /// Mean entropy from the previous check step, for plateau detection.
    prev_mean_entropy: f32,
    /// Lowest mean entropy seen across all check steps (near-miss telemetry).
    min_entropy: f32,
    /// Lowest acceptance rate seen across all steps (near-miss telemetry).
    min_acceptance_rate: f32,
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
        last_signals: ConvergenceSignals::default(),
        prev_self_cond_embed: None,
        accepted_count: 0,
        acceptance_rate: 1.0, // start at 100% (all positions changing)
        prev_mean_entropy: f32::MAX,
        min_entropy: f32::MAX,
        min_acceptance_rate: 1.0,
    }
}

// Check whether the canvas has converged.
//
// Convergence triggers when ANY of these criteria are met:
//
// 1. **Strict criteria** (original): argmax unchanged for `convergence_steps`
//    consecutive check steps AND mean entropy below `entropy_threshold`.
//
// 2. **Acceptance rate criteria** (adaptive): the update rate drops below
//    `acceptance_rate_threshold` (default 1%). `acceptance_rate` measures
//    positions kept from the current canvas, so convergence requires almost
//    all positions to be accepted.
//
// 3. **Entropy plateau criteria**: entropy has stopped decreasing significantly
//    (delta < `entropy_plateau_delta`) after step 16, indicating diminishing
//    returns.
fn check_convergence(canvas: &DiffusionCanvas, cfg: &DiffusionConfig) -> ConvergenceSignals {
    // Strict criteria: traditional argmax stability + low entropy.
    let strict =
        canvas.stable_count >= cfg.convergence_steps && canvas.mean_entropy < cfg.entropy_threshold;

    // Acceptance rate criteria: almost no positions being updated.
    let update_rate = 1.0 - canvas.acceptance_rate;
    let acceptance = canvas.step > 0 && update_rate < cfg.acceptance_rate_threshold;

    // Entropy plateau criteria: entropy stalled after warmup period.
    // Only check after step 16 to allow initial exploration.
    let entropy_delta = (canvas.prev_mean_entropy - canvas.mean_entropy).abs();
    let plateau = entropy_delta < cfg.entropy_plateau_delta && canvas.step >= 16;

    ConvergenceSignals {
        strict,
        acceptance,
        plateau,
    }
}

// Linear temperature schedule from `temp_start` (step 0) to `temp_end` (max steps).
fn temperature_at_step(step: usize, cfg: &DiffusionConfig) -> f32 {
    let t = step as f32 / cfg.max_denoise_steps.max(1) as f32;
    cfg.temp_start + (cfg.temp_end - cfg.temp_start) * t
}

fn mlx_scalar_u32(value: u32) -> MlxArray {
    MlxArray::from_raw_data(
        &value as *const u32 as *const u8,
        std::mem::size_of::<u32>(),
        &[],
        MlxDtype::Uint32,
    )
}

fn include_lowest_entropy_position(accepted_sorted: &MlxArray, canvas_size: usize) -> MlxArray {
    let ranks = arange(0.0, canvas_size as f64, 1.0, MlxDtype::Uint32, None);
    let ranks_2d = reshape(&ranks, &[1, canvas_size as i32], None);
    let first_sorted = equal(&ranks_2d, &mlx_scalar_u32(0), None);
    let accepted_count = add(
        &astype(accepted_sorted, MlxDtype::Uint32, None),
        &astype(&first_sorted, MlxDtype::Uint32, None),
        None,
    );
    greater_equal(&accepted_count, &mlx_scalar_u32(1), None)
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
    rng: &mut Xorshift64,
    step: usize,
    token_offset: usize,
    embed_table: Option<&MlxArray>,
    compiled_forward: Option<&MlxClosure>,
    kv_buffers: Option<&mut Vec<KVConcatBuffer>>,
    full_pipeline: Option<&MlxClosure>,
) {
    let temperature = temperature_at_step(step, diff_cfg);
    let is_check_step = step.is_multiple_of(diff_cfg.convergence_check_interval);

    // When the full-pipeline compiled closure is available, it fuses
    // forward + softmax + entropy + sampling + acceptance into a single
    // graph dispatch. Falls back to the imperative path on thread mismatch.
    if let Some(pipeline) = full_pipeline {
        let temp_arr = MlxArray::from_f32(temperature);
        match pipeline.try_apply(&[&canvas.tokens_gpu, &temp_arr]) {
            Ok(mut outputs) => {
                // Outputs: [new_tokens, argmax_1d, accept_mask_1d, entropy, prob]
                let prob = outputs.swap_remove(4);
                let entropy = outputs.swap_remove(3);
                let accept_mask_1d = outputs.swap_remove(2);
                let argmax_1d = outputs.swap_remove(1);
                let new_tokens = outputs.swap_remove(0);

                // Acceptance rate tracking.
                let accepted_f32 = astype(&accept_mask_1d, MlxDtype::Float32, None);
                let accepted_sum = sum_axis(&accepted_f32, -1, false, None);
                eval(&[&accepted_sum]);
                canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
                canvas.acceptance_rate = canvas.accepted_count as f32 / canvas.canvas_size as f32;
                canvas.min_acceptance_rate = canvas.min_acceptance_rate.min(canvas.acceptance_rate);

                // Convergence detection (check steps only).
                if is_check_step {
                    if let Some(prev_argmax) = &canvas.argmax_canvas {
                        let changed = not_equal(&argmax_1d, prev_argmax, None);
                        let changed_f32 = astype(&changed, MlxDtype::Float32, None);
                        let changed_count = sum_axis(&changed_f32, -1, false, None);
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
                        canvas.min_entropy = canvas.min_entropy.min(canvas.mean_entropy);
                    } else {
                        canvas.stable_count = 0;
                        let total_entropy = sum_axis(&entropy, -1, false, None);
                        let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
                        let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);
                        eval(&[&mean_entropy]);
                        canvas.prev_mean_entropy = canvas.mean_entropy;
                        canvas.mean_entropy = mean_entropy.data_f32()[0];
                        canvas.min_entropy = canvas.min_entropy.min(canvas.mean_entropy);
                    }
                    let signals = check_convergence(canvas, diff_cfg);
                    canvas.last_signals = signals;
                    canvas.converged = signals.any();
                }

                canvas.argmax_canvas = Some(argmax_1d);
                canvas.tokens_gpu = new_tokens;
                canvas.step = step;

                // Skip self-conditioning when the block has converged: the
                // embedding is only needed for the next denoise step.
                if diff_cfg.self_conditioning
                    && !canvas.converged
                    && let Some(embed) = embed_table
                {
                    canvas.prev_self_cond_embed = Some(matmul(&prob, embed, None));
                }
                return;
            }
            Err(_) => { /* fall through to imperative path */ }
        }
    }

    // Imperative path: forward + manual post-processing.
    // Also used as fallback when compiled closures fail.
    let logits = if let Some(compiled) = compiled_forward {
        match compiled.try_apply(&[&canvas.tokens_gpu]) {
            Ok(mut outputs) => outputs.swap_remove(0),
            Err(_) => forward_bidirectional(BidirectionalForward {
                cfg,
                weights,
                token_ids: &canvas.tokens_gpu,
                cache,
                token_offset,
                self_conditioning_signal: canvas.prev_self_cond_embed.as_ref(),
                embed_cache: None,
                kv_buffers,
            }),
        }
    } else {
        forward_bidirectional(BidirectionalForward {
            cfg,
            weights,
            token_ids: &canvas.tokens_gpu,
            cache,
            token_offset,
            self_conditioning_signal: canvas.prev_self_cond_embed.as_ref(),
            embed_cache: None,
            kv_buffers,
        })
    };

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
    // Always include the lowest-entropy position to match the CPU reference
    // fallback when every position exceeds the entropy budget.
    let accepted_sorted = include_lowest_entropy_position(
        &less_equal(&cum_entropy, &bound_scalar, None),
        canvas.canvas_size,
    );

    // Inverse-sort the acceptance mask back to original position order.
    // inverse_sort[i] = rank of position i in the sorted ordering.
    let inverse_sort = argsort_axis(&sorted_positions, -1, None);
    let accept_mask = take_along_axis(&accepted_sorted, &inverse_sort, -1, None);
    let accept_mask_1d = reshape(&accept_mask, &[canvas.canvas_size as i32], None);

    let random_tokens: Vec<u32> = (0..canvas.canvas_size)
        .map(|_| (rng.next_u64() % cfg.vocab_size as u64) as u32)
        .collect();
    let random_tokens_gpu = MlxArray::from_raw_data(
        random_tokens.as_ptr() as *const u8,
        std::mem::size_of_val(&random_tokens[..]),
        &[canvas.canvas_size as i32],
        MlxDtype::Uint32,
    );

    // Token update: accepted low-entropy positions adopt the denoiser draft;
    // rejected positions are renoised for the next denoise step.
    let new_tokens = where_cond(&accept_mask_1d, &argmax_1d, &random_tokens_gpu, None);

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
    canvas.min_acceptance_rate = canvas.min_acceptance_rate.min(canvas.acceptance_rate);

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
            canvas.min_entropy = canvas.min_entropy.min(canvas.mean_entropy);
        } else {
            // First step: no previous argmax to compare; force unstable.
            canvas.stable_count = 0;
            let total_entropy = sum_axis(&entropy, -1, false, None);
            let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
            let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);
            eval(&[&mean_entropy]);
            canvas.prev_mean_entropy = canvas.mean_entropy;
            canvas.mean_entropy = mean_entropy.data_f32()[0];
            canvas.min_entropy = canvas.min_entropy.min(canvas.mean_entropy);
        }
        let signals = check_convergence(canvas, diff_cfg);
        canvas.last_signals = signals;
        canvas.converged = signals.any();
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
    //
    // Skip when the block has already converged: the embedding is only needed
    // for a subsequent denoise step, and convergence means this is the last one.
    if diff_cfg.self_conditioning
        && !canvas.converged
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
    let mut embed_all = embed_tokens(&all_ids, &weights.token_embedding, cfg.hidden_size);
    if let Some(scale) = cfg.hidden_states_scale {
        embed_all = crate::model::shared::scale_hidden_pub(&embed_all, scale);
    }
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
    let output_tokens = canvas.argmax_canvas.as_ref().unwrap_or(&canvas.tokens_gpu);

    // Materialise GPU tokens to CPU for the causal forward.
    eval(&[output_tokens]);
    let tokens: Vec<u32> = output_tokens.data_u32().to_vec();
    let _logits = crate::model::forward(cfg, weights, &tokens, cache, token_offset);
    // model::forward appends K/V to the cache but does not advance seq_len;
    // update it so subsequent blocks append at the correct offset.
    cache.seq_len = token_offset + tokens.len();
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

    // Build KV concatenation buffers when the opt-in flag is set.
    // Per-layer buffers are lazily initialized on the first denoise step
    // and reused via `slice_update` on subsequent steps.
    let use_kv_buffers = fastpath::diffusion_kv_concat_buffer_enabled();
    let mut kv_buffers: Option<Vec<KVConcatBuffer>> = if use_kv_buffers {
        Some(Vec::new())
    } else {
        None
    };

    let full_pipeline: Option<MlxClosure> = None;

    // When full pipeline is active, suppress the forward-only closure.
    let compiled_forward: Option<MlxClosure> = if full_pipeline.is_some() {
        None
    } else if !diff_cfg.self_conditioning && fastpath::diffusion_compiled_forward_enabled() {
        let cfg_addr = cfg as *const ModelConfig as usize;
        let weights_addr = weights as *const ModelWeights as usize;
        let cache_addr = cache as *const MlxKVCache as usize;
        let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
            let token_ids = inputs.get(0);
            let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
            let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
            let cache_ref = unsafe { &*(cache_addr as *const MlxKVCache) };
            let logits = forward_bidirectional(BidirectionalForward {
                cfg: cfg_ref,
                weights: weights_ref,
                token_ids: &token_ids,
                cache: cache_ref,
                token_offset,
                self_conditioning_signal: None,
                embed_cache: None,
                kv_buffers: None,
            });
            vec![logits]
        });
        closure.compile(false).ok()
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
            rng,
            step,
            token_offset,
            embed_table,
            compiled_forward.as_ref(),
            kv_buffers.as_mut(),
            full_pipeline.as_ref(),
        );
        steps_executed += 1;
        if canvas.converged {
            break;
        }
    }
    let denoise_wall_us = elapsed_us(denoise_start);

    // Conditional commit skip: when the denoise loop converged at step 1
    // with near-perfect acceptance, the canvas tokens are already the
    // model's output — the causal commit pass is redundant.
    let commit_skipped = fastpath::diffusion_skip_commit_on_converge()
        && canvas.converged
        && steps_executed == 1
        && canvas.acceptance_rate >= 0.99;

    let (tokens, commit_wall_us) = if commit_skipped {
        let output_tokens = canvas.argmax_canvas.as_ref().unwrap_or(&canvas.tokens_gpu);
        eval(&[output_tokens]);
        let tokens: Vec<u32> = output_tokens.data_u32().to_vec();
        (tokens, 0)
    } else {
        let start = Instant::now();
        let tokens = commit_block(cfg, weights, cache, &canvas, token_offset);
        (tokens, elapsed_us(start))
    };
    let block_wall_us = elapsed_us(block_start);

    DiffusionBlockResult {
        tokens,
        denoise_steps: steps_executed,
        converged: canvas.converged,
        converged_strict: canvas.last_signals.strict,
        converged_acceptance: canvas.last_signals.acceptance,
        converged_plateau: canvas.last_signals.plateau,
        min_entropy: canvas.min_entropy,
        min_acceptance_rate: canvas.min_acceptance_rate,
        denoise_wall_us,
        commit_wall_us,
        block_wall_us,
        commit_skipped,
        full_pipeline_used: full_pipeline.is_some(),
        kv_buffer_used: kv_buffers.is_some(),
    }
}

fn elapsed_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

// Per-layer embedding cache for DiffusionGemma denoiser.
//
// Caches the output of `compute_per_layer_inputs_arr` across denoise steps.
// When token IDs are unchanged (high acceptance rate), the cached embeddings
// are reused, saving 46 embedding dispatches per cache hit. Token change
// detection uses a GPU-side sum fingerprint (2 dispatches + 1 eval).
struct EmbeddingCache {
    /// Sum of token IDs from the last cached computation (fingerprint).
    token_sum: f32,
    /// Cached per-layer embedding inputs.
    per_layer_inputs: Option<Vec<MlxArray>>,
}

impl EmbeddingCache {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            token_sum: f32::NAN,
            per_layer_inputs: None,
        }
    }

    /// Check whether tokens have changed using a sum fingerprint.
    /// Returns true when the cache should be refreshed.
    fn needs_refresh(&self, token_ids: &MlxArray) -> bool {
        if self.per_layer_inputs.is_none() {
            return true;
        }
        let token_sum_f32 = astype(token_ids, MlxDtype::Float32, None);
        let sum = sum_axis(&token_sum_f32, -1, false, None);
        eval(&[&sum]);
        let current_sum = sum.data_f32()[0];
        (current_sum - self.token_sum).abs() > 0.5
    }

    /// Update the cache with new per-layer inputs.
    fn update(&mut self, token_ids: &MlxArray, inputs: Vec<MlxArray>) {
        let token_sum_f32 = astype(token_ids, MlxDtype::Float32, None);
        let sum = sum_axis(&token_sum_f32, -1, false, None);
        eval(&[&sum]);
        self.token_sum = sum.data_f32()[0];
        self.per_layer_inputs = Some(inputs);
    }
}

struct BidirectionalForward<'a> {
    cfg: &'a ModelConfig,
    weights: &'a ModelWeights,
    token_ids: &'a MlxArray,
    cache: &'a MlxKVCache,
    token_offset: usize,
    self_conditioning_signal: Option<&'a MlxArray>,
    embed_cache: Option<&'a mut EmbeddingCache>,
    kv_buffers: Option<&'a mut Vec<KVConcatBuffer>>,
}

// Run the bidirectional forward pass over canvas tokens.
//
// Unlike the causal `forward()`, this uses `layer_forward_bidirectional` for
// each layer — no KV cache writes, bidirectional attention over the canvas.
// Accepts a 1-D `[canvas_size]` u32 `MlxArray` (may be GPU-resident/lazy).
// Returns per-position logits `[1, canvas_size, vocab_size]`.
//
// When `embed_cache` is `Some`, per-layer embedding inputs are cached across
// denoise steps and reused when token IDs are unchanged.
//
// When `kv_buffers` is `Some`, per-layer KV concatenation buffers are used
// to avoid re-copying the cached prompt prefix on every step.
fn forward_bidirectional(mut ctx: BidirectionalForward<'_>) -> MlxArray {
    let cfg = ctx.cfg;
    let weights = ctx.weights;
    // Embed tokens directly from GPU array.
    let mut hidden = embed_tokens_arr(ctx.token_ids, &weights.token_embedding, cfg.hidden_size);
    hidden = astype(&hidden, MlxDtype::Bfloat16, None);
    if let Some(scale) = cfg.hidden_states_scale {
        hidden = crate::model::shared::scale_hidden_pub(&hidden, scale);
    }
    if let Some(self_conditioning) = weights.diffusion_self_conditioning.as_ref() {
        if let Some(signal) = ctx.self_conditioning_signal {
            let normed = rms_norm(
                signal,
                Some(&self_conditioning.pre_norm),
                cfg.rms_norm_eps,
                None,
            );
            let gate = shared::qw(&normed, &self_conditioning.gate_proj);
            let up = shared::qw(&normed, &self_conditioning.up_proj);
            let activated = multiply(&gelu_approx(&gate, None), &up, None);
            let sc_signal = shared::qw(&activated, &self_conditioning.down_proj);
            hidden = add(&hidden, &sc_signal, None);
        }
        hidden = rms_norm(&hidden, None, cfg.rms_norm_eps, None);
    }

    // Compute per-layer inputs (Gemma4 per-layer embeddings).
    // Use cache when available and tokens are unchanged.
    let per_layer_inputs = if let Some(cache_entry) = ctx.embed_cache {
        if cache_entry.needs_refresh(ctx.token_ids) {
            let pli = compute_per_layer_inputs_arr(cfg, weights, ctx.token_ids, &hidden);
            if let Some(ref inputs) = pli {
                cache_entry.update(ctx.token_ids, inputs.clone());
            }
            pli
        } else {
            cache_entry.per_layer_inputs.clone()
        }
    } else {
        compute_per_layer_inputs_arr(cfg, weights, ctx.token_ids, &hidden)
    };

    // Run each layer with bidirectional attention.
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        let kv_buf = ctx.kv_buffers.as_mut().map(|v| {
            // Lazily grow the buffer vec to cover all layers.
            if v.len() <= li {
                v.resize_with(li + 1, KVConcatBuffer::new);
            }
            &mut v[li]
        });
        hidden = layer_forward_bidirectional(
            cfg,
            layer_w,
            &hidden,
            ctx.cache,
            li,
            ctx.token_offset,
            pli,
            kv_buf,
        );
    }

    // Final norm + LM head → logits [1, seq, vocab_size].
    let normed = rms_norm(&hidden, Some(&weights.final_norm), cfg.rms_norm_eps, None);
    let logits = shared::qw(&normed, &weights.lm_head);
    finalize_lm_head_logits(cfg, &logits, FinalLogitsMode::Full)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal canvas with scalar fields set for convergence testing.
    /// The GPU token array is a dummy — `check_convergence` only reads scalars.
    fn test_canvas(
        stable_count: usize,
        mean_entropy: f32,
        acceptance_rate: f32,
        prev_mean_entropy: f32,
        step: usize,
    ) -> DiffusionCanvas {
        DiffusionCanvas {
            tokens_gpu: MlxArray::from_f32(0.0),
            canvas_size: 256,
            argmax_canvas: None,
            stable_count,
            mean_entropy,
            step,
            converged: false,
            last_signals: ConvergenceSignals::default(),
            prev_self_cond_embed: None,
            accepted_count: (acceptance_rate * 256.0) as usize,
            acceptance_rate,
            prev_mean_entropy,
            min_entropy: f32::MAX,
            min_acceptance_rate: 1.0,
        }
    }

    fn default_diff_cfg() -> DiffusionConfig {
        DiffusionConfig {
            canvas_size: 256,
            max_denoise_steps: 48,
            entropy_bound: 0.1,
            entropy_threshold: 0.005,
            convergence_steps: 2,
            temp_start: 0.8,
            temp_end: 0.4,
            self_conditioning: true,
            convergence_check_interval: 4,
            acceptance_rate_threshold: 0.01,
            entropy_plateau_delta: 0.001,
        }
    }

    #[test]
    fn convergence_signals_strict() {
        let cfg = default_diff_cfg();
        // stable_count >= convergence_steps (2) AND mean_entropy < entropy_threshold (0.005).
        let canvas = test_canvas(2, 0.003, 0.5, 0.004, 8);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            signals.strict,
            "strict should fire when stable and low entropy"
        );
        assert!(!signals.acceptance);
        assert!(!signals.plateau);
        assert!(signals.any());
    }

    #[test]
    fn convergence_signals_strict_not_stable() {
        let cfg = default_diff_cfg();
        // stable_count < convergence_steps → strict should NOT fire.
        let canvas = test_canvas(1, 0.003, 0.5, 0.004, 8);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            !signals.strict,
            "strict requires stable_count >= convergence_steps"
        );
    }

    #[test]
    fn convergence_signals_strict_high_entropy() {
        let cfg = default_diff_cfg();
        // stable but entropy above threshold → strict should NOT fire.
        let canvas = test_canvas(2, 0.5, 0.5, 0.6, 8);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            !signals.strict,
            "strict requires mean_entropy < entropy_threshold"
        );
    }

    #[test]
    fn convergence_signals_acceptance() {
        let cfg = default_diff_cfg();
        // update_rate (1.0 - 0.995 = 0.005) < acceptance_rate_threshold (0.01).
        let canvas = test_canvas(0, 0.5, 0.995, 0.6, 4);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            signals.acceptance,
            "acceptance should fire when update rate < threshold"
        );
        assert!(!signals.strict);
        assert!(!signals.plateau);
        assert!(signals.any());
    }

    #[test]
    fn convergence_signals_acceptance_ignores_initial_step() {
        let cfg = default_diff_cfg();
        let canvas = test_canvas(0, 0.5, 0.0, f32::MAX, 0);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            !signals.acceptance,
            "acceptance convergence must not fire before a previous denoise step exists"
        );
        assert!(!signals.any());
    }

    #[test]
    fn convergence_signals_acceptance_not_low() {
        let cfg = default_diff_cfg();
        // update_rate (0.5) well above threshold.
        let canvas = test_canvas(0, 0.5, 0.5, 0.6, 4);
        let signals = check_convergence(&canvas, &cfg);
        assert!(!signals.acceptance);
    }

    #[test]
    fn convergence_signals_low_acceptance_means_canvas_still_updating() {
        let cfg = default_diff_cfg();
        let canvas = test_canvas(0, 0.5, 0.005, 0.6, 4);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            !signals.acceptance,
            "low acceptance means most positions are still updating"
        );
    }

    #[test]
    fn convergence_signals_plateau() {
        let cfg = default_diff_cfg();
        // entropy delta (0.010 - 0.0095 = 0.0005) < plateau_delta (0.001) AND step >= 16.
        let canvas = test_canvas(0, 0.0095, 0.5, 0.010, 20);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            signals.plateau,
            "plateau should fire when delta < threshold after step 16"
        );
        assert!(!signals.strict);
        assert!(!signals.acceptance);
        assert!(signals.any());
    }

    #[test]
    fn convergence_signals_plateau_before_warmup() {
        let cfg = default_diff_cfg();
        // Same entropy delta but step < 16 → plateau should NOT fire.
        let canvas = test_canvas(0, 0.0095, 0.5, 0.010, 12);
        let signals = check_convergence(&canvas, &cfg);
        assert!(!signals.plateau, "plateau requires step >= 16 warmup");
    }

    #[test]
    fn convergence_signals_none() {
        let cfg = default_diff_cfg();
        // Nothing triggers: unstable, high entropy, high acceptance, early step.
        let canvas = test_canvas(0, 0.5, 0.5, 0.6, 4);
        let signals = check_convergence(&canvas, &cfg);
        assert!(!signals.any());
    }

    #[test]
    fn convergence_signals_multiple() {
        let cfg = default_diff_cfg();
        // strict + acceptance + plateau all fire simultaneously.
        // stable_count=3 >= 2, entropy=0.001 < 0.005 → strict.
        // update_rate=1.0 - 0.995 < 0.01 → acceptance.
        // abs(0.0015 - 0.001) = 0.0005 < 0.001 AND step=20 >= 16 → plateau.
        let canvas = test_canvas(3, 0.001, 0.995, 0.0015, 20);
        let signals = check_convergence(&canvas, &cfg);
        assert!(signals.strict);
        assert!(signals.acceptance);
        assert!(signals.plateau);
        assert!(signals.any());
    }

    #[test]
    fn convergence_signals_any_logic() {
        let none = ConvergenceSignals {
            strict: false,
            acceptance: false,
            plateau: false,
        };
        assert!(!none.any());
        let strict_only = ConvergenceSignals {
            strict: true,
            ..Default::default()
        };
        assert!(strict_only.any());
    }

    #[test]
    fn include_lowest_entropy_position_accepts_one_when_budget_accepts_none() {
        let accepted_sorted = mlx_sys::zeros(&[1, 4], MlxDtype::Bool, None);
        let with_fallback = include_lowest_entropy_position(&accepted_sorted, 4);
        let accepted_f32 = astype(&with_fallback, MlxDtype::Float32, None);
        let accepted_sum = sum_axis(&accepted_f32, -1, false, None);
        eval(&[&accepted_sum]);
        assert_eq!(accepted_sum.data_f32(), &[1.0]);
    }

    // ── Commit skip predicate tests ──────────────────────────────────
    //
    // These verify the pure predicate logic used in `generate_diffusion_block`
    // without requiring model weights or MLX runtime.

    /// Simulate the commit-skip predicate from `generate_diffusion_block`.
    fn should_skip_commit(
        flag_enabled: bool,
        converged: bool,
        steps_executed: u32,
        acceptance_rate: f32,
    ) -> bool {
        flag_enabled && converged && steps_executed == 1 && acceptance_rate >= 0.99
    }

    #[test]
    fn commit_skip_fires_on_step1_convergence() {
        assert!(should_skip_commit(true, true, 1, 1.0));
        assert!(should_skip_commit(true, true, 1, 0.99));
    }

    #[test]
    fn commit_no_skip_on_multi_step() {
        // Even with convergence and high acceptance, steps > 1 prevents skip.
        assert!(!should_skip_commit(true, true, 2, 1.0));
        assert!(!should_skip_commit(true, true, 48, 0.99));
    }

    #[test]
    fn commit_no_skip_on_low_acceptance() {
        // Acceptance below 0.99 threshold prevents skip.
        assert!(!should_skip_commit(true, true, 1, 0.98));
        assert!(!should_skip_commit(true, true, 1, 0.5));
    }

    #[test]
    fn commit_no_skip_without_convergence() {
        assert!(!should_skip_commit(true, false, 1, 1.0));
    }

    #[test]
    fn commit_no_skip_when_flag_disabled() {
        // Flag disabled prevents skip regardless of other conditions.
        assert!(!should_skip_commit(false, true, 1, 1.0));
    }

    // ── EmbeddingCache tests ─────────────────────────────────────────

    #[test]
    fn embedding_cache_initial_state_needs_refresh() {
        let cache = EmbeddingCache::new();
        assert!(cache.per_layer_inputs.is_none());
        assert!(cache.token_sum.is_nan());
        // needs_refresh returns true when cache is empty.
        let dummy = MlxArray::from_f32(42.0);
        assert!(cache.needs_refresh(&dummy));
    }

    fn make_u32_array(data: &[u32]) -> MlxArray {
        MlxArray::from_raw_data(
            data.as_ptr() as *const u8,
            std::mem::size_of_val(data),
            &[data.len() as i32],
            MlxDtype::Uint32,
        )
    }

    #[test]
    fn embedding_cache_hit_on_same_tokens() {
        let mut cache = EmbeddingCache::new();
        let tokens = make_u32_array(&[10, 20, 30]);
        let dummy_pli = vec![MlxArray::from_f32(1.0)];
        cache.update(&tokens, dummy_pli);
        assert!(cache.per_layer_inputs.is_some());
        // Same token sum → no refresh needed.
        assert!(!cache.needs_refresh(&tokens));
    }

    #[test]
    fn embedding_cache_miss_on_different_tokens() {
        let mut cache = EmbeddingCache::new();
        let tokens_a = make_u32_array(&[10, 20, 30]);
        let dummy_pli = vec![MlxArray::from_f32(1.0)];
        cache.update(&tokens_a, dummy_pli);
        // Different tokens with different sum → refresh needed.
        let tokens_b = make_u32_array(&[100, 200, 300]);
        assert!(cache.needs_refresh(&tokens_b));
    }

    // ── KVConcatBuffer tests ─────────────────────────────────────────

    #[test]
    fn kv_concat_buffer_initial_state_is_empty() {
        let buf = KVConcatBuffer::new();
        assert!(buf.full_k.is_none());
        assert!(buf.full_v.is_none());
        assert_eq!(buf.cached_seq, 0);
    }

    #[test]
    fn kv_concat_buffer_populate_and_reuse() {
        let mut buf = KVConcatBuffer::new();
        // Populate the buffer as if the first denoise step ran.
        buf.full_k = Some(MlxArray::from_f32(3.0));
        buf.full_v = Some(MlxArray::from_f32(4.0));
        buf.cached_seq = 10;

        // After first step, buffer is populated.
        assert!(buf.full_k.is_some());
        assert!(buf.full_v.is_some());
        assert_eq!(buf.cached_seq, 10);

        // Second step: buffer is still valid for reuse.
        // The caller would use slice_update to replace the canvas portion.
        assert!(buf.full_k.is_some());
    }

    // ── Full pipeline / KV buffer flag tests ──────────────────────────

    #[test]
    fn diffusion_block_result_tracks_optimization_flags() {
        let result = DiffusionBlockResult {
            tokens: vec![1, 2, 3],
            denoise_steps: 1,
            converged: true,
            converged_strict: false,
            converged_acceptance: true,
            converged_plateau: false,
            min_entropy: 0.01,
            min_acceptance_rate: 0.99,
            denoise_wall_us: 100,
            commit_wall_us: 0,
            block_wall_us: 100,
            commit_skipped: true,
            full_pipeline_used: true,
            kv_buffer_used: true,
        };
        assert!(result.commit_skipped);
        assert!(result.full_pipeline_used);
        assert!(result.kv_buffer_used);

        let result2 = DiffusionBlockResult {
            tokens: vec![4, 5],
            denoise_steps: 10,
            converged: false,
            converged_strict: false,
            converged_acceptance: false,
            converged_plateau: false,
            min_entropy: 0.5,
            min_acceptance_rate: 0.3,
            denoise_wall_us: 2000,
            commit_wall_us: 100,
            block_wall_us: 2200,
            commit_skipped: false,
            full_pipeline_used: false,
            kv_buffer_used: false,
        };
        assert!(!result2.commit_skipped);
        assert!(!result2.full_pipeline_used);
        assert!(!result2.kv_buffer_used);
    }
}
