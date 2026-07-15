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
//! Convergence is checked every `convergence_check_interval` steps (default 1)
//! to further reduce sync overhead.
//!
//! ## Performance optimizations
//!
//! The default decode path is the **full-pipeline compiled closure** (see
//! below). Mutable secondary opts (embedding cache, KV concat buffer) stay
//! **outside** `mlx_compile` graphs: compile traces a pure graph and must not
//! depend on step-varying control flow or external buffer mutation (same
//! constraint as the MTP compiled head). Those opts therefore apply on the
//! imperative / forward-only fallback, where they are **auto-enabled** when
//! the full pipeline is disabled.
//!
//! - **Conditional causal commit skip** (default ON, opt-out:
//!   `AX_DIFFUSION_NO_SKIP_COMMIT=1`): when the denoise loop converges with at
//!   least 99% acceptance, the causal commit pass is skipped when the block
//!   terminates the request — saving ~40 ms of causal commit weight traffic.
//!
//! - **Compiled forward closure** (default ON when the full pipeline is off,
//!   opt-out: `AX_DIFFUSION_NO_COMPILED_FORWARD=1`): wraps the bidirectional
//!   forward in an `MlxClosure`, collapsing ~250 per-step C-API calls into one
//!   dispatched graph. Self-conditioning flows in as an explicit input.
//!
//! - **Per-layer embedding cache** (auto-ON when full pipeline is off; force
//!   via `AX_DIFFUSION_EMBEDDING_CACHE=1`): caches
//!   `compute_per_layer_inputs_arr` across denoise steps when a token
//!   fingerprint is unchanged. Output-neutral; not used inside full-pipeline
//!   compile (purity).
//!
//! - **KV concatenation buffer** (auto-ON when full pipeline is off; force via
//!   `AX_DIFFUSION_KV_CONCAT_BUFFER=1`): pre-allocates per-layer KV buffers and
//!   updates the canvas slice via `slice_update` + `contiguous`/`eval` so the
//!   buffer path matches re-`concatenate` bit-for-bit across steps. Not used
//!   inside full-pipeline compile (purity).
//!
//! - **Resumable denoise workspace** (Phase B): `open` / `advance` / `commit`
//!   split monoblock generation so a step budget
//!   (`AX_DIFFUSION_STEPS_PER_ENGINE_STEP`) can drive partial DenoiseStep
//!   progress. Default (unset) remains monoblock denoise+commit in one call.
//!
//! - **Stage profile** (Phase C, opt-in `AX_DIFFUSION_PROFILE=1`): accumulates
//!   forward vs sample vs commit wall time for denoise observability.
//!
//! - **Random-token host buffer reuse** (Phase C): renoise samples reuse a
//!   pre-sized `Vec<u32>` on the workspace instead of allocating every step.
//!
//! ## Full-pipeline compiled forward
//!
//! The full-pipeline closure compiles the entire denoise step (forward, softmax,
//! entropy, sampling, and acceptance) into a single MLX graph, collapsing ~280
//! per-step dispatches into one. **Default ON**; opt-out via
//! `AX_DIFFUSION_NO_FULL_PIPELINE=1` (falls back to the compiled/imperative path
//! with embed cache + KV buffer auto-enabled).
//!
//! The closure accepts four inputs:
//!   - `[0]` token_ids: `[canvas_size]` u32
//!   - `[1]` self_cond: `[1, canvas_size, hidden_size]` bf16 (zeros on step 0)
//!   - `[2]` temperature: scalar f32
//!   - `[3]` random_tokens: `[canvas_size]` u32 (for renoising rejected positions)
//!
//! And outputs five arrays:
//!   - `[0]` new_tokens: `[canvas_size]` u32
//!   - `[1]` argmax_1d: `[canvas_size]` u32
//!   - `[2]` accept_mask_1d: `[canvas_size]` bool
//!   - `[3]` entropy: `[1, canvas_size]` f32
//!   - `[4]` prob: `[1, canvas_size, vocab_size]` f32

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ax_engine_core::DiffusionScheduleUpdate;

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
//    `acceptance_rate_threshold` (default 7.5%). `acceptance_rate` measures
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
    // Only check after step 8 to allow initial exploration.
    let entropy_delta = (canvas.prev_mean_entropy - canvas.mean_entropy).abs();
    let plateau = entropy_delta < cfg.entropy_plateau_delta && canvas.step >= 8;

    ConvergenceSignals {
        strict,
        acceptance,
        plateau,
    }
}

// Temperature schedule from `temp_start` (step 0) to `temp_end` (max steps).
//
// Supports two schedules:
// - **Linear**: `temp = start + (end - start) * t` — constant cooling rate.
// - **Exponential**: `temp = start * (end / start)^t` — drops temperature faster
//   in early steps, which can reduce denoise iterations on in-distribution
//   prompts by reaching exploitation temperature sooner.
fn temperature_at_step(step: usize, cfg: &DiffusionConfig) -> f32 {
    let t = step as f32 / cfg.max_denoise_steps.max(1) as f32;
    match cfg.temperature_schedule {
        crate::model::DiffusionTemperatureSchedule::Linear => {
            cfg.temp_start + (cfg.temp_end - cfg.temp_start) * t
        }
        crate::model::DiffusionTemperatureSchedule::Exponential => {
            if cfg.temp_start <= 0.0 {
                cfg.temp_end
            } else if cfg.temp_end <= 0.0 {
                // Geometric interpolation is undefined for a non-positive end
                // temperature: temp_end == 0.0 makes (end/start)^t collapse to
                // 0.0 for every t > 0 (an instant cliff, not a gradual decay),
                // and temp_end < 0.0 raises a negative ratio to a fractional
                // power, producing NaN. Either poisons the softmax via
                // divide-by-zero/NaN-propagation downstream. Fall back to
                // linear interpolation, which stays finite for any temp_end.
                cfg.temp_start + (cfg.temp_end - cfg.temp_start) * t
            } else {
                cfg.temp_start * (cfg.temp_end / cfg.temp_start).powf(t)
            }
        }
    }
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
    embed_cache: Option<&mut EmbeddingCache>,
    random_token_scratch: &mut Vec<u32>,
    zero_sc_signal: Option<&MlxArray>,
) {
    let temperature = temperature_at_step(step, diff_cfg);
    let is_check_step = step.is_multiple_of(diff_cfg.convergence_check_interval);
    let forward_started = Instant::now();

    // When the full-pipeline compiled closure is available, it fuses
    // forward + softmax + entropy + sampling + acceptance into a single
    // graph dispatch. Falls back to the imperative path on thread mismatch.
    if let Some(pipeline) = full_pipeline {
        // Prefer workspace-reused zero placeholder; allocate only as last resort.
        let owned_zero;
        let sc_input: &MlxArray = if let Some(prev) = canvas.prev_self_cond_embed.as_ref() {
            prev
        } else if let Some(z) = zero_sc_signal {
            z
        } else if diff_cfg.self_conditioning {
            owned_zero = mlx_sys::zeros(
                &[1, canvas.canvas_size as i32, cfg.hidden_size as i32],
                MlxDtype::Bfloat16,
                None,
            );
            &owned_zero
        } else {
            &canvas.tokens_gpu
        };

        // Reuse host scratch for renoise random tokens (Phase C hot-path).
        let random_tokens_gpu = random_tokens_gpu_from_scratch(
            random_token_scratch,
            canvas.canvas_size,
            cfg.vocab_size,
            rng,
        );

        let temp_arr = MlxArray::from_f32(temperature);
        match pipeline.try_apply(&[&canvas.tokens_gpu, sc_input, &temp_arr, &random_tokens_gpu]) {
            Ok(mut outputs) => {
                // Outputs: [new_tokens, argmax_1d, accept_mask_1d, entropy, prob]
                let prob = outputs.swap_remove(4);
                let entropy = outputs.swap_remove(3);
                let accept_mask_1d = outputs.swap_remove(2);
                let argmax_1d = outputs.swap_remove(1);
                let new_tokens = outputs.swap_remove(0);

                // Acceptance rate + convergence detection (batched evals).
                let accepted_f32 = astype(&accept_mask_1d, MlxDtype::Float32, None);
                let accepted_sum = sum_axis(&accepted_f32, -1, false, None);

                if is_check_step {
                    if let Some(prev_argmax) = &canvas.argmax_canvas {
                        let changed = not_equal(&argmax_1d, prev_argmax, None);
                        let changed_f32 = astype(&changed, MlxDtype::Float32, None);
                        let changed_count = sum_axis(&changed_f32, -1, false, None);
                        let total_entropy = sum_axis(&entropy, -1, false, None);
                        let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
                        let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);
                        // Single batched eval.
                        eval(&[&accepted_sum, &changed_count, &mean_entropy]);
                        canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
                        canvas.acceptance_rate =
                            canvas.accepted_count as f32 / canvas.canvas_size as f32;
                        canvas.min_acceptance_rate =
                            canvas.min_acceptance_rate.min(canvas.acceptance_rate);
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
                        let total_entropy = sum_axis(&entropy, -1, false, None);
                        let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
                        let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);
                        eval(&[&accepted_sum, &mean_entropy]);
                        canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
                        canvas.acceptance_rate =
                            canvas.accepted_count as f32 / canvas.canvas_size as f32;
                        canvas.min_acceptance_rate =
                            canvas.min_acceptance_rate.min(canvas.acceptance_rate);
                        canvas.stable_count = 0;
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
                    && canvas.acceptance_rate < diff_cfg.sc_skip_acceptance_rate
                    && let Some(embed) = embed_table
                {
                    // Skip when acceptance_rate >= sc_skip_acceptance_rate:
                    // the canvas is mostly stable (>95% positions accepted),
                    // so the self-conditioning signal barely changes and the
                    // expensive `prob × embed_table` matmul (~5% per step)
                    // is wasted.
                    //
                    // Cast back to bf16: `prob` is f32 (finalized logits), so
                    // the matmul result would otherwise propagate f32 through
                    // every layer of the next step's forward — including a
                    // per-layer re-upcast of the cached prompt KV at the
                    // attention concat — and retrace the compiled pipeline
                    // (the step-0 zero signal is bf16).
                    canvas.prev_self_cond_embed = Some(astype(
                        &matmul(&prob, embed, None),
                        MlxDtype::Bfloat16,
                        None,
                    ));
                }
                record_denoise_stage_forward_us(elapsed_us(forward_started));
                return;
            }
            Err(_) => { /* fall through to imperative path */ }
        }
    }

    // Imperative path: forward + manual post-processing.
    // Also used as fallback when compiled closures fail.
    let logits = if let Some(compiled) = compiled_forward {
        let owned_zero;
        let sc_input: &MlxArray = if let Some(prev) = canvas.prev_self_cond_embed.as_ref() {
            prev
        } else if let Some(z) = zero_sc_signal {
            z
        } else if diff_cfg.self_conditioning {
            owned_zero = mlx_sys::zeros(
                &[1, canvas.canvas_size as i32, cfg.hidden_size as i32],
                MlxDtype::Bfloat16,
                None,
            );
            &owned_zero
        } else {
            &canvas.tokens_gpu
        };
        match compiled.try_apply(&[&canvas.tokens_gpu, sc_input]) {
            Ok(mut outputs) => outputs.swap_remove(0),
            Err(_) => forward_bidirectional(BidirectionalForward {
                cfg,
                weights,
                token_ids: &canvas.tokens_gpu,
                cache,
                token_offset,
                self_conditioning_signal: canvas.prev_self_cond_embed.as_ref(),
                embed_cache,
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
            embed_cache,
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

    // Argmax per position → [1, canvas_size] (argmax reduces last axis).
    let argmax_2d = argmax(&prob, None);
    // Reshape to 1-D [canvas_size] for token-state tracking.
    let argmax_1d = reshape(&argmax_2d, &[canvas.canvas_size as i32], None);

    // ── Sampler-dependent acceptance mask ──────────────────────────────
    //
    // Two strategies are supported. The sampler choice dominates denoise
    // throughput: confidence-threshold avoids argsort/cumsum/inverse-sort
    // and is 4–5× faster with equivalent output quality (mlx-optiq evidence).
    //
    // Entropy is only needed for convergence detection (check steps). With
    // the confidence-threshold sampler on non-check steps the expensive
    // log/multiply/sum entropy computation is skipped entirely.
    let entropy: Option<MlxArray>;
    let accept_mask_1d: MlxArray;

    match diff_cfg.sampler {
        crate::model::DiffusionSampler::EntropyBound => {
            // Per-position entropy: H(p) = -sum(p * log(p)) along vocab axis.
            let eps = MlxArray::from_f32(1e-10);
            let log_prob = log(&add(&prob, &eps, None), None);
            let p_log_p = multiply(&prob, &log_prob, None);
            let ent = negative(&sum_axis(&p_log_p, -1, false, None), None);

            // Sort positions by entropy ascending (most confident first),
            // accept greedily until cumulative entropy exceeds the budget.
            let sorted_positions = argsort_axis(&ent, -1, None);
            let sorted_entropy = take_along_axis(&ent, &sorted_positions, -1, None);
            let cum_entropy = cumsum(&sorted_entropy, -1, false, true, None);
            let bound_scalar = MlxArray::from_f32(diff_cfg.entropy_bound);
            let accepted_sorted = include_lowest_entropy_position(
                &less_equal(&cum_entropy, &bound_scalar, None),
                canvas.canvas_size,
            );
            let inverse_sort = argsort_axis(&sorted_positions, -1, None);
            let accept_mask = take_along_axis(&accepted_sorted, &inverse_sort, -1, None);
            accept_mask_1d = reshape(&accept_mask, &[canvas.canvas_size as i32], None);
            entropy = Some(ent);
        }
        crate::model::DiffusionSampler::ConfidenceThreshold => {
            // Accept positions whose peak softmax probability >= threshold.
            // No sorting required — just one comparison per position.
            let argmax_3d = reshape(&argmax_2d, &[1, canvas.canvas_size as i32, 1], None);
            let peak_prob = take_along_axis(&prob, &argmax_3d, -1, None);
            let peak_2d = reshape(&peak_prob, &[1, canvas.canvas_size as i32], None);
            let threshold = MlxArray::from_f32(diff_cfg.confidence_threshold);
            let accept_mask = greater_equal(&peak_2d, &threshold, None);
            accept_mask_1d = reshape(&accept_mask, &[canvas.canvas_size as i32], None);

            // Entropy is only needed for convergence detection on check steps.
            if is_check_step {
                let eps = MlxArray::from_f32(1e-10);
                let log_prob = log(&add(&prob, &eps, None), None);
                let p_log_p = multiply(&prob, &log_prob, None);
                entropy = Some(negative(&sum_axis(&p_log_p, -1, false, None), None));
            } else {
                entropy = None;
            }
        }
    }

    record_denoise_stage_forward_us(elapsed_us(forward_started));
    let sample_started = Instant::now();
    let random_tokens_gpu = random_tokens_gpu_from_scratch(
        random_token_scratch,
        canvas.canvas_size,
        cfg.vocab_size,
        rng,
    );

    // Token update: accepted low-entropy positions adopt the denoiser draft;
    // rejected positions are renoised for the next denoise step.
    let new_tokens = where_cond(&accept_mask_1d, &argmax_1d, &random_tokens_gpu, None);

    // ── Acceptance rate tracking + convergence detection ─────────────────
    //
    // Reduce host-side overhead (llama.cpp #24529 evidence: 10–20% gain):
    // - On non-check steps: skip acceptance rate materialization entirely
    //   (the value is only needed for convergence detection on check steps
    //   and for the commit-skip decision after the denoise loop).
    // - On check steps: batch accepted_sum with changed_count + mean_entropy
    //   in a single eval() call to minimize GPU→CPU syncs.

    let accepted_f32 = astype(&accept_mask_1d, MlxDtype::Float32, None);
    let accepted_sum = sum_axis(&accepted_f32, -1, false, None);

    if is_check_step {
        // Entropy is always computed on check steps (both samplers).
        let ent = entropy
            .as_ref()
            .expect("entropy must be present on check steps");
        // Argmax stability: count positions that differ from previous step.
        if let Some(prev_argmax) = &canvas.argmax_canvas {
            let changed = not_equal(&argmax_1d, prev_argmax, None);
            let changed_f32 = astype(&changed, MlxDtype::Float32, None);
            let changed_count = sum_axis(&changed_f32, -1, false, None);

            // Mean entropy: scalar mean of per-position entropy.
            let total_entropy = sum_axis(ent, -1, false, None);
            let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
            let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);

            // Single batched eval for all three scalars.
            eval(&[&accepted_sum, &changed_count, &mean_entropy]);

            canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
            canvas.acceptance_rate = canvas.accepted_count as f32 / canvas.canvas_size as f32;
            canvas.min_acceptance_rate = canvas.min_acceptance_rate.min(canvas.acceptance_rate);

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
            let total_entropy = sum_axis(ent, -1, false, None);
            let canvas_size_f32 = MlxArray::from_f32(canvas.canvas_size as f32);
            let mean_entropy = divide(&total_entropy, &canvas_size_f32, None);

            // Batch accepted_sum + mean_entropy eval.
            eval(&[&accepted_sum, &mean_entropy]);

            canvas.accepted_count = accepted_sum.data_f32()[0] as usize;
            canvas.acceptance_rate = canvas.accepted_count as f32 / canvas.canvas_size as f32;
            canvas.min_acceptance_rate = canvas.min_acceptance_rate.min(canvas.acceptance_rate);

            canvas.stable_count = 0;
            canvas.prev_mean_entropy = canvas.mean_entropy;
            canvas.mean_entropy = mean_entropy.data_f32()[0];
            canvas.min_entropy = canvas.min_entropy.min(canvas.mean_entropy);
        }
        let signals = check_convergence(canvas, diff_cfg);
        canvas.last_signals = signals;
        canvas.converged = signals.any();
    }
    // On non-check steps: acceptance_rate is left unchanged from the last
    // check step. This is safe because:
    // - check_convergence() is only called on check steps
    // - The commit-skip decision after the loop uses the final check step's value

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
    //
    // Also skip when acceptance_rate >= sc_skip_acceptance_rate: the canvas
    // is mostly stable, so the self-conditioning signal barely changes and
    // the expensive `prob × embed_table` matmul (~5% per step) is wasted.
    if diff_cfg.self_conditioning
        && !canvas.converged
        && canvas.acceptance_rate < diff_cfg.sc_skip_acceptance_rate
        && let Some(embed) = embed_table
    {
        // Cast back to bf16 — see the compiled-pipeline site above: an f32
        // signal propagates f32 activations through the whole next forward
        // and re-upcasts the cached prompt KV per layer per step.
        canvas.prev_self_cond_embed = Some(astype(
            &matmul(&prob, embed, None),
            MlxDtype::Bfloat16,
            None,
        ));
    }
    record_denoise_stage_sample_us(elapsed_us(sample_started));
}

/// Fill `scratch` with `n` random vocab ids and upload as a GPU `[n]` u32 array.
///
/// Reuses the host vector capacity across denoise steps (Phase C hot-path).
pub(crate) fn random_tokens_gpu_from_scratch(
    scratch: &mut Vec<u32>,
    n: usize,
    vocab_size: usize,
    rng: &mut Xorshift64,
) -> MlxArray {
    scratch.clear();
    scratch.reserve(n);
    let vocab = vocab_size.max(1) as u64;
    for _ in 0..n {
        scratch.push((rng.next_u64() % vocab) as u32);
    }
    MlxArray::from_raw_data(
        scratch.as_ptr() as *const u8,
        std::mem::size_of_val(scratch.as_slice()),
        &[n as i32],
        MlxDtype::Uint32,
    )
}

// ── Denoise stage profile (Phase C, opt-in AX_DIFFUSION_PROFILE) ───────────

static DENOISE_STAGE_FORWARD_US: AtomicU64 = AtomicU64::new(0);
static DENOISE_STAGE_SAMPLE_US: AtomicU64 = AtomicU64::new(0);
static DENOISE_STAGE_COMMIT_US: AtomicU64 = AtomicU64::new(0);
static DENOISE_STAGE_STEPS: AtomicU64 = AtomicU64::new(0);

fn record_denoise_stage_forward_us(us: u32) {
    if !fastpath::diffusion_profile_enabled() {
        return;
    }
    DENOISE_STAGE_FORWARD_US.fetch_add(u64::from(us), Ordering::Relaxed);
    DENOISE_STAGE_STEPS.fetch_add(1, Ordering::Relaxed);
}

fn record_denoise_stage_sample_us(us: u32) {
    if !fastpath::diffusion_profile_enabled() {
        return;
    }
    DENOISE_STAGE_SAMPLE_US.fetch_add(u64::from(us), Ordering::Relaxed);
}

fn record_denoise_stage_commit_us(us: u32) {
    if !fastpath::diffusion_profile_enabled() {
        return;
    }
    DENOISE_STAGE_COMMIT_US.fetch_add(u64::from(us), Ordering::Relaxed);
}

/// Snapshot of cumulative denoise stage timers (microseconds).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub struct DenoiseStageProfile {
    pub forward_us: u64,
    pub sample_us: u64,
    pub commit_us: u64,
    pub steps: u64,
}

/// Read and zero denoise stage profile counters (test / tooling).
#[allow(dead_code)]
pub fn take_denoise_stage_profile() -> DenoiseStageProfile {
    DenoiseStageProfile {
        forward_us: DENOISE_STAGE_FORWARD_US.swap(0, Ordering::Relaxed),
        sample_us: DENOISE_STAGE_SAMPLE_US.swap(0, Ordering::Relaxed),
        commit_us: DENOISE_STAGE_COMMIT_US.swap(0, Ordering::Relaxed),
        steps: DENOISE_STAGE_STEPS.swap(0, Ordering::Relaxed),
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
    cache.set_seq_len(token_offset + tokens.len());
    tokens
}

/// Runner-supplied policy for deciding whether a converged block may skip the
/// causal commit pass.
///
/// `commit_block` is the only path that appends the canvas K/V to the cache
/// and advances `cache.seq_len()`, so the commit may only be skipped when the
/// block terminates the request — a skipped commit on a non-final block would
/// generate the next block with no memory of this one, at the same absolute
/// positions.
pub(crate) struct DiffusionCommitPolicy<'a> {
    /// Token ids at which the runner truncates the served block queue
    /// (dInfer-style EOS early termination; the runner-global terminal set).
    pub truncation_terminal_ids: &'a [u32],
    /// Token ids that actually terminate this request (empty when the request
    /// sets `ignore_eos`).
    pub request_terminal_ids: &'a [u32],
    /// Remaining output-token budget for the request, when known. `None`
    /// conservatively forces the commit pass.
    pub remaining_output_budget: Option<u32>,
}

impl DiffusionCommitPolicy<'_> {
    /// Whether serving this block's tokens ends the request — i.e. no further
    /// block will be generated, so this block's KV is never read again.
    fn block_terminates_request(&self, tokens: &[u32]) -> bool {
        let served_len = tokens
            .iter()
            .position(|tok| self.truncation_terminal_ids.contains(tok))
            .map(|pos| pos + 1)
            .unwrap_or(tokens.len());
        if tokens[..served_len]
            .iter()
            .any(|tok| self.request_terminal_ids.contains(tok))
        {
            return true;
        }
        self.remaining_output_budget
            .is_some_and(|budget| budget as usize <= served_len)
    }
}

/// Clamp how many denoise steps to run in one engine/call budget.
///
/// Pure control helper used by the resumable workspace and unit-tested
/// without a loaded model.
pub(crate) fn clamp_denoise_step_budget(
    steps_this_call: usize,
    max_denoise_steps: usize,
    steps_already_done: u32,
) -> usize {
    let remaining = max_denoise_steps.saturating_sub(steps_already_done as usize);
    if steps_this_call == usize::MAX {
        remaining
    } else {
        steps_this_call.min(remaining)
    }
}

/// Build schedule feedback for the engine from workspace progress.
pub(crate) fn schedule_update_from_progress(
    denoise_steps_in_block: u32,
    commit_ready: bool,
    block_committed: bool,
) -> DiffusionScheduleUpdate {
    DiffusionScheduleUpdate {
        denoise_steps_in_block,
        commit_ready,
        block_committed,
    }
}

/// Resumable denoise state for one diffusion block (Phase B).
///
/// Holds canvas + counters across engine steps when
/// `AX_DIFFUSION_STEPS_PER_ENGINE_STEP` is set. Compiled pipelines are rebuilt
/// each `advance` call so raw pointer captures stay valid for the call scope.
pub(crate) struct DiffusionBlockWorkspace {
    canvas: DiffusionCanvas,
    effective_cfg: DiffusionConfig,
    steps_executed: u32,
    denoise_wall_us: u32,
    token_offset: usize,
    /// Pre-sized host buffer for renoise random tokens (Phase C hot-path).
    random_token_scratch: Vec<u32>,
    /// Reused zero self-conditioning placeholder for step 0 / missing prior.
    zero_sc_signal: Option<MlxArray>,
    full_pipeline_used: bool,
    kv_buffer_used: bool,
    block_start: Instant,
}

/// Progress after one `advance_diffusion_workspace` call.
///
/// Extra fields beyond schedule feedback are retained for telemetry and
/// unit tests even when the runner only consumes `commit_ready` /
/// `steps_total`.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct DiffusionAdvanceProgress {
    pub steps_this_call: u32,
    pub steps_total: u32,
    pub converged: bool,
    /// True when denoise is finished and the block may be committed.
    pub commit_ready: bool,
    pub acceptance_rate: f32,
    pub min_entropy: f32,
    pub min_acceptance_rate: f32,
    pub last_signals: ConvergenceSignals,
}

impl DiffusionAdvanceProgress {
    pub fn schedule_update(self) -> DiffusionScheduleUpdate {
        schedule_update_from_progress(self.steps_total, self.commit_ready, false)
    }
}

/// Open a new diffusion block workspace (canvas initialized, zero steps).
pub(crate) fn open_diffusion_block(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    rng: &mut Xorshift64,
    token_offset: usize,
    embed_table_cache: &mut Option<MlxArray>,
) -> DiffusionBlockWorkspace {
    let mut effective_cfg = diff_cfg.clone();
    if effective_cfg.self_conditioning && weights.diffusion_self_conditioning.is_none() {
        effective_cfg.self_conditioning = false;
    }
    if effective_cfg.self_conditioning && embed_table_cache.is_none() {
        *embed_table_cache = Some(compute_embed_table(weights, cfg));
    }
    let canvas = init_canvas(effective_cfg.canvas_size, cfg.vocab_size, rng);
    let canvas_size = canvas.canvas_size;
    DiffusionBlockWorkspace {
        canvas,
        effective_cfg,
        steps_executed: 0,
        denoise_wall_us: 0,
        token_offset,
        random_token_scratch: Vec::with_capacity(canvas_size),
        zero_sc_signal: None,
        full_pipeline_used: false,
        kv_buffer_used: false,
        block_start: Instant::now(),
    }
}

/// Run up to `max_steps_this_call` denoise steps on an open workspace.
///
/// When `max_steps_this_call` is `usize::MAX`, runs until convergence or
/// `max_denoise_steps` (monoblock denoise phase).
#[allow(clippy::too_many_arguments)]
pub(crate) fn advance_diffusion_workspace(
    ws: &mut DiffusionBlockWorkspace,
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &MlxKVCache,
    rng: &mut Xorshift64,
    embed_table_cache: &Option<MlxArray>,
    max_steps_this_call: usize,
) -> DiffusionAdvanceProgress {
    let diff_cfg = &ws.effective_cfg;
    let budget = clamp_denoise_step_budget(
        max_steps_this_call,
        diff_cfg.max_denoise_steps,
        ws.steps_executed,
    );
    if budget == 0 || ws.canvas.converged {
        let commit_ready =
            ws.canvas.converged || ws.steps_executed as usize >= diff_cfg.max_denoise_steps;
        return DiffusionAdvanceProgress {
            steps_this_call: 0,
            steps_total: ws.steps_executed,
            converged: ws.canvas.converged,
            commit_ready,
            acceptance_rate: ws.canvas.acceptance_rate,
            min_entropy: ws.canvas.min_entropy,
            min_acceptance_rate: ws.canvas.min_acceptance_rate,
            last_signals: ws.canvas.last_signals,
        };
    }

    let embed_table = if diff_cfg.self_conditioning {
        embed_table_cache.as_ref()
    } else {
        None
    };

    // Build compiled pipelines for this advance call (pointer lifetime = call).
    let full_pipeline_requested = !fastpath::diffusion_no_full_pipeline();
    let token_offset = ws.token_offset;
    let full_pipeline: Option<MlxClosure> = if full_pipeline_requested {
        let cfg_addr = cfg as *const ModelConfig as usize;
        let weights_addr = weights as *const ModelWeights as usize;
        let cache_addr = cache as *const MlxKVCache as usize;
        let canvas_size = ws.canvas.canvas_size;
        let has_self_cond = diff_cfg.self_conditioning;
        let sampler = diff_cfg.sampler;
        let entropy_bound = diff_cfg.entropy_bound;
        let confidence_threshold = diff_cfg.confidence_threshold;

        // SAFETY: The closure captures raw pointers to `cfg`, `weights`, and `cache`
        // because `MlxClosure::new_dyn()` requires `'static` captures. The referenced
        // objects outlive this advance call — the closure is invoked synchronously
        // and never stored beyond this scope.
        //
        // Embed cache / KV buffers are intentionally *not* threaded into the
        // compiled graph: mlx_compile requires a pure traced function.
        let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
            let token_ids = inputs.get(0);
            let self_cond_signal = inputs.get(1);
            let temperature = inputs.get(2);
            let random_tokens = inputs.get(3);

            let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
            let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
            let cache_ref = unsafe { &*(cache_addr as *const MlxKVCache) };
            let signal_ref = if has_self_cond {
                Some(&self_cond_signal)
            } else {
                None
            };

            let logits = forward_bidirectional(BidirectionalForward {
                cfg: cfg_ref,
                weights: weights_ref,
                token_ids: &token_ids,
                cache: cache_ref,
                token_offset,
                self_conditioning_signal: signal_ref,
                embed_cache: None,
                kv_buffers: None,
            });

            let scaled = divide(&logits, &temperature, None);
            let prob = softmax(&scaled, -1, None);
            let argmax_2d = argmax(&prob, None);
            let argmax_1d = reshape(&argmax_2d, &[canvas_size as i32], None);

            let (accept_mask_1d, entropy) = match sampler {
                crate::model::DiffusionSampler::EntropyBound => {
                    let eps = MlxArray::from_f32(1e-10);
                    let log_prob = log(&add(&prob, &eps, None), None);
                    let p_log_p = multiply(&prob, &log_prob, None);
                    let ent = negative(&sum_axis(&p_log_p, -1, false, None), None);
                    let sorted_positions = argsort_axis(&ent, -1, None);
                    let sorted_entropy = take_along_axis(&ent, &sorted_positions, -1, None);
                    let cum_entropy = cumsum(&sorted_entropy, -1, false, true, None);
                    let bound_scalar = MlxArray::from_f32(entropy_bound);
                    let accepted_sorted = include_lowest_entropy_position(
                        &less_equal(&cum_entropy, &bound_scalar, None),
                        canvas_size,
                    );
                    let inverse_sort = argsort_axis(&sorted_positions, -1, None);
                    let accept_mask = take_along_axis(&accepted_sorted, &inverse_sort, -1, None);
                    let mask_1d = reshape(&accept_mask, &[canvas_size as i32], None);
                    (mask_1d, ent)
                }
                crate::model::DiffusionSampler::ConfidenceThreshold => {
                    let argmax_3d = reshape(&argmax_2d, &[1, canvas_size as i32, 1], None);
                    let peak_prob = take_along_axis(&prob, &argmax_3d, -1, None);
                    let peak_2d = reshape(&peak_prob, &[1, canvas_size as i32], None);
                    let threshold = MlxArray::from_f32(confidence_threshold);
                    let accept_mask = greater_equal(&peak_2d, &threshold, None);
                    let mask_1d = reshape(&accept_mask, &[canvas_size as i32], None);
                    let eps = MlxArray::from_f32(1e-10);
                    let log_prob = log(&add(&prob, &eps, None), None);
                    let p_log_p = multiply(&prob, &log_prob, None);
                    let ent = negative(&sum_axis(&p_log_p, -1, false, None), None);
                    (mask_1d, ent)
                }
            };

            let new_tokens = where_cond(&accept_mask_1d, &argmax_1d, &random_tokens, None);
            vec![new_tokens, argmax_1d, accept_mask_1d, entropy, prob]
        });
        closure.compile(false).ok()
    } else {
        None
    };

    let mut kv_buffers: Option<Vec<KVConcatBuffer>> = if fastpath::diffusion_no_kv_concat_buffer() {
        None
    } else {
        Some(Vec::new())
    };
    let mut embed_cache: Option<EmbeddingCache> = if fastpath::diffusion_no_embedding_cache() {
        None
    } else {
        Some(EmbeddingCache::new())
    };

    let compiled_forward: Option<MlxClosure> = if full_pipeline.is_some() {
        None
    } else if !fastpath::diffusion_no_compiled_forward() {
        let cfg_addr = cfg as *const ModelConfig as usize;
        let weights_addr = weights as *const ModelWeights as usize;
        let cache_addr = cache as *const MlxKVCache as usize;
        let has_self_cond = diff_cfg.self_conditioning;
        let closure = MlxClosure::new_dyn(move |inputs: &MlxVectorArray| -> Vec<MlxArray> {
            let token_ids = inputs.get(0);
            let self_cond_signal = inputs.get(1);
            let cfg_ref = unsafe { &*(cfg_addr as *const ModelConfig) };
            let weights_ref = unsafe { &*(weights_addr as *const ModelWeights) };
            let cache_ref = unsafe { &*(cache_addr as *const MlxKVCache) };
            let signal_ref = if has_self_cond {
                Some(&self_cond_signal)
            } else {
                None
            };
            let logits = forward_bidirectional(BidirectionalForward {
                cfg: cfg_ref,
                weights: weights_ref,
                token_ids: &token_ids,
                cache: cache_ref,
                token_offset,
                self_conditioning_signal: signal_ref,
                embed_cache: None,
                kv_buffers: None,
            });
            vec![logits]
        });
        closure.compile(false).ok()
    } else {
        None
    };

    if full_pipeline.is_some() {
        ws.full_pipeline_used = true;
    }
    if full_pipeline.is_none() && kv_buffers.is_some() {
        ws.kv_buffer_used = true;
    }

    // Ensure zero self-cond placeholder exists when needed (reused across steps).
    if diff_cfg.self_conditioning && ws.zero_sc_signal.is_none() {
        ws.zero_sc_signal = Some(mlx_sys::zeros(
            &[1, ws.canvas.canvas_size as i32, cfg.hidden_size as i32],
            MlxDtype::Bfloat16,
            None,
        ));
    }

    let denoise_start = Instant::now();
    let mut steps_this_call = 0_u32;
    for _ in 0..budget {
        let step = ws.steps_executed as usize;
        denoise_step(
            cfg,
            diff_cfg,
            weights,
            cache,
            &mut ws.canvas,
            rng,
            step,
            token_offset,
            embed_table,
            compiled_forward.as_ref(),
            kv_buffers.as_mut(),
            full_pipeline.as_ref(),
            embed_cache.as_mut(),
            &mut ws.random_token_scratch,
            ws.zero_sc_signal.as_ref(),
        );
        ws.steps_executed = ws.steps_executed.saturating_add(1);
        steps_this_call = steps_this_call.saturating_add(1);
        if ws.canvas.converged {
            break;
        }
    }
    let denoise_elapsed = elapsed_us(denoise_start);
    ws.denoise_wall_us = ws.denoise_wall_us.saturating_add(denoise_elapsed);

    let commit_ready =
        ws.canvas.converged || ws.steps_executed as usize >= diff_cfg.max_denoise_steps;
    DiffusionAdvanceProgress {
        steps_this_call,
        steps_total: ws.steps_executed,
        converged: ws.canvas.converged,
        commit_ready,
        acceptance_rate: ws.canvas.acceptance_rate,
        min_entropy: ws.canvas.min_entropy,
        min_acceptance_rate: ws.canvas.min_acceptance_rate,
        last_signals: ws.canvas.last_signals,
    }
}

/// Commit a finished denoise workspace into tokens + KV (or skip commit).
pub(crate) fn commit_diffusion_workspace(
    ws: DiffusionBlockWorkspace,
    cfg: &ModelConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    commit_policy: DiffusionCommitPolicy<'_>,
) -> DiffusionBlockResult {
    let commit_start = Instant::now();
    // Conditional commit skip: when the denoise loop converged with
    // near-perfect acceptance, the canvas tokens are already the model's
    // output — the causal commit pass (~40 ms) is redundant *for tokens*.
    // Enabled by default; opt-out via `AX_DIFFUSION_NO_SKIP_COMMIT=1`.
    //
    // The commit pass is still load-bearing for the KV cache: it is the only
    // path that appends this block's K/V and advances `cache.seq_len()`. The
    // skip is therefore restricted to blocks that terminate the request
    // (EOS in the served tokens, or the remaining output budget is exhausted);
    // otherwise the next block would be generated with no memory of this one,
    // at the same absolute positions.
    let commit_skip_eligible = !fastpath::diffusion_no_skip_commit()
        && ws.canvas.converged
        && ws.canvas.acceptance_rate >= 0.99;

    let (tokens, commit_wall_us, commit_skipped) = if commit_skip_eligible {
        let output_tokens = ws
            .canvas
            .argmax_canvas
            .as_ref()
            .unwrap_or(&ws.canvas.tokens_gpu);
        eval(&[output_tokens]);
        let tokens: Vec<u32> = output_tokens.data_u32().to_vec();
        if commit_policy.block_terminates_request(&tokens) {
            (tokens, 0, true)
        } else {
            let start = Instant::now();
            let tokens = commit_block(cfg, weights, cache, &ws.canvas, ws.token_offset);
            (tokens, elapsed_us(start), false)
        }
    } else {
        let start = Instant::now();
        let tokens = commit_block(cfg, weights, cache, &ws.canvas, ws.token_offset);
        (tokens, elapsed_us(start), false)
    };
    record_denoise_stage_commit_us(elapsed_us(commit_start));
    let block_wall_us = elapsed_us(ws.block_start);

    DiffusionBlockResult {
        tokens,
        denoise_steps: ws.steps_executed,
        converged: ws.canvas.converged,
        converged_strict: ws.canvas.last_signals.strict,
        converged_acceptance: ws.canvas.last_signals.acceptance,
        converged_plateau: ws.canvas.last_signals.plateau,
        min_entropy: ws.canvas.min_entropy,
        min_acceptance_rate: ws.canvas.min_acceptance_rate,
        denoise_wall_us: ws.denoise_wall_us,
        commit_wall_us,
        block_wall_us,
        commit_skipped,
        full_pipeline_used: ws.full_pipeline_used,
        kv_buffer_used: ws.kv_buffer_used,
    }
}

/// Generate one diffusion block: denoise → commit (monoblock default).
///
/// The prompt is assumed to be already prefilled into the cache.
/// Returns telemetry along with the committed tokens.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_diffusion_block(
    cfg: &ModelConfig,
    diff_cfg: &DiffusionConfig,
    weights: &ModelWeights,
    cache: &mut MlxKVCache,
    rng: &mut Xorshift64,
    token_offset: usize,
    embed_table_cache: &mut Option<MlxArray>,
    commit_policy: DiffusionCommitPolicy<'_>,
) -> DiffusionBlockResult {
    let mut ws = open_diffusion_block(cfg, diff_cfg, weights, rng, token_offset, embed_table_cache);
    // Monoblock: run until converge or max_denoise_steps.
    let _ = advance_diffusion_workspace(
        &mut ws,
        cfg,
        weights,
        cache,
        rng,
        embed_table_cache,
        usize::MAX,
    );
    commit_diffusion_workspace(ws, cfg, weights, cache, commit_policy)
}

fn elapsed_us(started: Instant) -> u32 {
    started.elapsed().as_micros().min(u32::MAX as u128) as u32
}

// ── Embedding cache telemetry ──────────────────────────────────────────────
//
// Global hit/miss counters for the per-layer embedding cache in the
// bidirectional denoiser. Incremented from `forward_bidirectional` when
// `EmbeddingCache::needs_refresh` is consulted.
static EMBEDDING_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static EMBEDDING_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Returns `(hits, misses)` for the DiffusionGemma embedding cache.
#[allow(dead_code)]
pub fn embedding_cache_counters() -> (u64, u64) {
    (
        EMBEDDING_CACHE_HITS.load(Ordering::Relaxed),
        EMBEDDING_CACHE_MISSES.load(Ordering::Relaxed),
    )
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
            EMBEDDING_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
            let pli = compute_per_layer_inputs_arr(cfg, weights, ctx.token_ids, &hidden);
            if let Some(ref inputs) = pli {
                cache_entry.update(ctx.token_ids, inputs.clone());
            }
            pli
        } else {
            EMBEDDING_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            cache_entry.per_layer_inputs.clone()
        }
    } else {
        compute_per_layer_inputs_arr(cfg, weights, ctx.token_ids, &hidden)
    };

    // Run each layer with bidirectional attention.
    let profile_enabled = fastpath::diffusion_profile_enabled();
    for (li, layer_w) in weights.layers.iter().enumerate() {
        let pli = per_layer_inputs.as_ref().map(|v| &v[li]);
        let kv_buf = ctx.kv_buffers.as_mut().map(|v| {
            // Lazily grow the buffer vec to cover all layers.
            if v.len() <= li {
                v.resize_with(li + 1, KVConcatBuffer::new);
            }
            &mut v[li]
        });
        let layer_started = profile_enabled.then(Instant::now);
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
        if let Some(started) = layer_started {
            let elapsed = started.elapsed().as_micros();
            eprintln!(
                "[diffusion-profile] layer {li}/{num_layers} wall_us={elapsed}",
                num_layers = weights.layers.len(),
            );
        }
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
            entropy_threshold: 0.02,
            convergence_steps: 2,
            temp_start: 0.8,
            temp_end: 0.4,
            self_conditioning: true,
            convergence_check_interval: 2,
            acceptance_rate_threshold: 0.075,
            entropy_plateau_delta: 0.005,
            sampler: crate::model::DiffusionSampler::EntropyBound,
            confidence_threshold: 0.9,
            temperature_schedule: crate::model::DiffusionTemperatureSchedule::Linear,
            sc_skip_acceptance_rate: 0.95,
        }
    }

    #[test]
    fn convergence_signals_strict() {
        let cfg = default_diff_cfg();
        // stable_count >= convergence_steps (2) AND mean_entropy < entropy_threshold (0.02).
        // Use step=4 to stay below plateau warmup (step >= 8).
        let canvas = test_canvas(2, 0.003, 0.5, 0.004, 4);
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
        // update_rate (1.0 - 0.995 = 0.005) < acceptance_rate_threshold (0.075).
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
        // entropy delta (0.010 - 0.0095 = 0.0005) < plateau_delta (0.005) AND step >= 8.
        let canvas = test_canvas(0, 0.0095, 0.5, 0.010, 20);
        let signals = check_convergence(&canvas, &cfg);
        assert!(
            signals.plateau,
            "plateau should fire when delta < threshold after step 8"
        );
        assert!(!signals.strict);
        assert!(!signals.acceptance);
        assert!(signals.any());
    }

    #[test]
    fn convergence_signals_plateau_before_warmup() {
        let cfg = default_diff_cfg();
        // Same entropy delta but step < 8 → plateau should NOT fire.
        let canvas = test_canvas(0, 0.0095, 0.5, 0.010, 4);
        let signals = check_convergence(&canvas, &cfg);
        assert!(!signals.plateau, "plateau requires step >= 8 warmup");
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

    /// Simulate the commit-skip *eligibility* predicate from
    /// `generate_diffusion_block`. Eligibility fires on any convergence (not
    /// just step 1) with high acceptance; the actual skip additionally
    /// requires `DiffusionCommitPolicy::block_terminates_request` (tested
    /// separately below), because skipping the commit drops the block's KV.
    fn should_skip_commit(
        flag_enabled: bool,
        converged: bool,
        _steps_executed: u32,
        acceptance_rate: f32,
    ) -> bool {
        flag_enabled && converged && acceptance_rate >= 0.99
    }

    #[test]
    fn commit_skip_fires_on_convergence() {
        assert!(should_skip_commit(true, true, 1, 1.0));
        assert!(should_skip_commit(true, true, 1, 0.99));
    }

    #[test]
    fn commit_skip_fires_on_multi_step_convergence() {
        // Multi-step convergence with high acceptance now skips commit.
        assert!(should_skip_commit(true, true, 2, 1.0));
        assert!(should_skip_commit(true, true, 48, 0.99));
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

    // ── Commit-policy termination tests ──────────────────────────────

    const EOS: u32 = 106;

    fn policy(request_terminal: bool, budget: Option<u32>) -> DiffusionCommitPolicy<'static> {
        DiffusionCommitPolicy {
            truncation_terminal_ids: &[EOS],
            request_terminal_ids: if request_terminal { &[EOS] } else { &[] },
            remaining_output_budget: budget,
        }
    }

    #[test]
    fn non_final_block_requires_commit() {
        // No EOS, plenty of budget left: another block follows, so the KV
        // append must not be skipped.
        assert!(!policy(true, Some(1000)).block_terminates_request(&[1, 2, 3, 4]));
        // Unknown budget is conservative: commit.
        assert!(!policy(true, None).block_terminates_request(&[1, 2, 3, 4]));
    }

    #[test]
    fn eos_in_block_terminates_request() {
        assert!(policy(true, Some(1000)).block_terminates_request(&[1, 2, EOS, 4]));
        // EOS as the very first served token still terminates.
        assert!(policy(true, Some(1000)).block_terminates_request(&[EOS, 2, 3, 4]));
    }

    #[test]
    fn budget_exhaustion_terminates_request() {
        assert!(policy(true, Some(4)).block_terminates_request(&[1, 2, 3, 4]));
        assert!(policy(true, Some(2)).block_terminates_request(&[1, 2, 3, 4]));
        assert!(!policy(true, Some(5)).block_terminates_request(&[1, 2, 3, 4]));
    }

    #[test]
    fn ignore_eos_truncated_block_can_still_continue() {
        // ignore_eos: the runner truncates the served queue at EOS (position 2,
        // served_len 3) but the request does not terminate there — the request
        // continues iff budget outlasts the served tokens.
        assert!(!policy(false, Some(10)).block_terminates_request(&[1, 2, EOS, 4]));
        assert!(policy(false, Some(3)).block_terminates_request(&[1, 2, EOS, 4]));
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

    // ── Temperature schedule tests ─────────────────────────────────────

    #[test]
    fn temperature_linear_schedule_midpoint() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.8;
        cfg.temp_end = 0.4;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Linear;
        // At step 24 (midpoint), t = 0.5 → temp = 0.8 + (0.4 - 0.8) * 0.5 = 0.6
        let temp = temperature_at_step(24, &cfg);
        assert!((temp - 0.6).abs() < 1e-6, "linear midpoint: got {temp}");
    }

    #[test]
    fn temperature_exponential_schedule_midpoint() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.8;
        cfg.temp_end = 0.4;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Exponential;
        // At step 24 (midpoint), t = 0.5 → temp = 0.8 * (0.5)^0.5 ≈ 0.5657
        let temp = temperature_at_step(24, &cfg);
        let expected = 0.8_f32 * (0.5_f32).sqrt();
        assert!(
            (temp - expected).abs() < 1e-4,
            "exponential midpoint: got {temp}, expected {expected}"
        );
        // Exponential drops faster early: at midpoint it should be below linear midpoint.
        assert!(temp < 0.6, "exponential should drop faster than linear");
    }

    #[test]
    fn temperature_exponential_endpoints() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.8;
        cfg.temp_end = 0.4;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Exponential;
        // At step 0, t = 0 → temp = 0.8 * (0.5)^0 = 0.8
        let temp0 = temperature_at_step(0, &cfg);
        assert!((temp0 - 0.8).abs() < 1e-6, "exp step 0: got {temp0}");
        // At step max, t = 1 → temp = 0.8 * (0.5)^1 = 0.4
        let temp48 = temperature_at_step(48, &cfg);
        assert!((temp48 - 0.4).abs() < 1e-6, "exp step max: got {temp48}");
    }

    #[test]
    fn temperature_exponential_zero_start() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.0;
        cfg.temp_end = 0.4;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Exponential;
        // temp_start = 0 → should return temp_end to avoid 0 * (inf)^t.
        let temp = temperature_at_step(24, &cfg);
        assert!((temp - 0.4).abs() < 1e-6, "exp zero start: got {temp}");
    }

    #[test]
    fn temperature_exponential_zero_end_falls_back_to_linear_and_stays_finite() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.8;
        cfg.temp_end = 0.0;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Exponential;
        // temp_end = 0 must not collapse to 0.0 immediately after step 0
        // (0^t == 0 for any t > 0); linear fallback decays gradually instead.
        let temp_step1 = temperature_at_step(1, &cfg);
        assert!(
            temp_step1 > 0.0 && temp_step1.is_finite(),
            "exp zero end step 1 must stay finite and positive: got {temp_step1}"
        );
        let temp_mid = temperature_at_step(24, &cfg);
        assert!(
            (temp_mid - 0.4).abs() < 1e-6,
            "exp zero end midpoint should match linear fallback: got {temp_mid}"
        );
        let temp_end = temperature_at_step(48, &cfg);
        assert!(
            (temp_end - 0.0).abs() < 1e-6,
            "exp zero end step max: got {temp_end}"
        );
    }

    #[test]
    fn temperature_exponential_negative_end_stays_finite() {
        let mut cfg = default_diff_cfg();
        cfg.max_denoise_steps = 48;
        cfg.temp_start = 0.8;
        cfg.temp_end = -0.4;
        cfg.temperature_schedule = crate::model::DiffusionTemperatureSchedule::Exponential;
        // A negative ratio raised to a fractional power is NaN in the naive
        // geometric formula; the linear fallback must stay finite everywhere.
        for step in [0, 1, 24, 47, 48] {
            let temp = temperature_at_step(step, &cfg);
            assert!(
                temp.is_finite(),
                "exp negative end step {step} must stay finite: got {temp}"
            );
        }
    }

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

    // ── Phase B: step budget + schedule progress ─────────────────────

    #[test]
    fn clamp_denoise_step_budget_limits_remaining() {
        assert_eq!(clamp_denoise_step_budget(4, 48, 0), 4);
        assert_eq!(clamp_denoise_step_budget(4, 48, 46), 2);
        assert_eq!(clamp_denoise_step_budget(4, 48, 48), 0);
        assert_eq!(clamp_denoise_step_budget(usize::MAX, 48, 10), 38);
        assert_eq!(clamp_denoise_step_budget(0, 48, 0), 0);
    }

    #[test]
    fn schedule_update_from_progress_maps_fields() {
        let mid = schedule_update_from_progress(5, false, false);
        assert_eq!(mid.denoise_steps_in_block, 5);
        assert!(!mid.commit_ready);
        assert!(!mid.block_committed);

        let ready = schedule_update_from_progress(12, true, false);
        assert!(ready.commit_ready);
        assert!(!ready.block_committed);

        let done = schedule_update_from_progress(12, false, true);
        assert!(done.block_committed);
        assert!(!done.commit_ready);
    }

    #[test]
    fn advance_progress_schedule_update_is_not_committed() {
        let progress = DiffusionAdvanceProgress {
            steps_this_call: 2,
            steps_total: 6,
            converged: false,
            commit_ready: false,
            acceptance_rate: 0.5,
            min_entropy: 1.0,
            min_acceptance_rate: 0.4,
            last_signals: ConvergenceSignals::default(),
        };
        let update = progress.schedule_update();
        assert_eq!(update.denoise_steps_in_block, 6);
        assert!(!update.commit_ready);
        assert!(!update.block_committed);

        let ready = DiffusionAdvanceProgress {
            commit_ready: true,
            ..progress
        };
        let update_ready = ready.schedule_update();
        assert!(update_ready.commit_ready);
        assert!(!update_ready.block_committed);
    }

    fn tiny_diffusion_cfg(canvas: usize, max_steps: usize) -> (ModelConfig, DiffusionConfig) {
        let model = ModelConfig {
            compile_cache_identity: 42,
            rope_mscale: 1.0,
            model_family: "diffusion_gemma".to_string(),
            layer_count: 0,
            hidden_size: 8,
            intermediate_size: 16,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 4,
            vocab_size: 16,
            rope_theta: 10_000.0,
            rope_dims: 4,
            attn_output_gate: false,
            query_scale: 1.0,
            final_logit_softcapping: None,
            moe_expert_count: 0,
            moe_experts_per_token: 0,
            moe_expert_intermediate_size: 0,
            layer_configs: Vec::new(),
            global_sliding_window: None,
            gemma4_moe_router: false,
            uses_geglu: false,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            linear_attention: None,
            mla_attention: None,
            glm_router: None,
            rms_norm_eps: 1e-6,
            rope_freqs: None,
            no_rope_layer_interval: 0,
            attn_temperature_floor: 8192.0,
            attn_temperature_scale: 0.1,
            intermediate_size_mlp: 0,
            moe_layer_freq: 1,
            moe_first_dense_layers: 0,
            moe_shared_expert_count: 0,
            moe_sigmoid_routing: false,
            moe_routed_scaling_factor: 1.0,
            moe_n_group: 1,
            moe_topk_group: 1,
            think_start_token_id: None,
            think_end_token_id: None,
            diffusion: None,
            gpt_oss_uses_mxfp4_experts: false,
            generation_kind: ax_engine_core::GenerationKind::BlockDiffusion,
        };
        let mut diff = default_diff_cfg();
        diff.canvas_size = canvas;
        diff.max_denoise_steps = max_steps;
        diff.self_conditioning = false;
        diff.convergence_check_interval = 1;
        // Force max steps so multi-budget path reaches commit_ready without
        // relying on model quality for convergence.
        diff.entropy_threshold = -1.0;
        diff.acceptance_rate_threshold = -1.0;
        diff.entropy_plateau_delta = -1.0;
        (model, diff)
    }

    fn tiny_diffusion_weights(hidden: usize, vocab: usize) -> ModelWeights {
        use crate::weights::QuantizedWeight;
        use mlx_sys::zeros;
        ModelWeights {
            token_embedding: QuantizedWeight::new(
                zeros(&[vocab as i32, hidden as i32], MlxDtype::Float32, None),
                None,
                None,
            ),
            final_norm: zeros(&[hidden as i32], MlxDtype::Float32, None),
            lm_head: QuantizedWeight::new(
                zeros(&[vocab as i32, hidden as i32], MlxDtype::Float32, None),
                None,
                None,
            ),
            layers: Vec::new(),
            per_layer_embed: None,
            per_layer_model_proj: None,
            per_layer_proj_norm: None,
            mtp: None,
            glm_mtp: None,
            gemma4_assistant_mtp: Default::default(),
            assistant_pre_projection: None,
            assistant_post_projection: None,
            embedding_dense_0: None,
            embedding_dense_1: None,
            gemma4_unified_vision: None,
            gemma4_unified_audio: None,
            diffusion_self_conditioning: None,
        }
    }

    /// Drive the real open/advance/commit surface across partial step budgets.
    #[test]
    fn open_advance_commit_across_partial_step_budgets() {
        let (cfg, diff_cfg) = tiny_diffusion_cfg(4, 3);
        let weights = tiny_diffusion_weights(cfg.hidden_size, cfg.vocab_size);
        let mut cache = MlxKVCache::new(0);
        let mut rng = Xorshift64::new(11);
        let mut embed_table: Option<MlxArray> = None;

        let mut ws = open_diffusion_block(
            &cfg,
            &diff_cfg,
            &weights,
            &mut rng,
            /*token_offset*/ 0,
            &mut embed_table,
        );
        assert_eq!(ws.steps_executed, 0);

        let p1 =
            advance_diffusion_workspace(&mut ws, &cfg, &weights, &cache, &mut rng, &embed_table, 1);
        assert_eq!(p1.steps_this_call, 1);
        assert_eq!(p1.steps_total, 1);
        assert!(!p1.commit_ready);
        let sched1 = p1.schedule_update();
        assert_eq!(sched1.denoise_steps_in_block, 1);
        assert!(!sched1.commit_ready);
        assert!(!sched1.block_committed);

        let p2 =
            advance_diffusion_workspace(&mut ws, &cfg, &weights, &cache, &mut rng, &embed_table, 1);
        assert_eq!(p2.steps_this_call, 1);
        assert_eq!(p2.steps_total, 2);
        assert!(!p2.commit_ready);

        let p3 =
            advance_diffusion_workspace(&mut ws, &cfg, &weights, &cache, &mut rng, &embed_table, 1);
        assert_eq!(p3.steps_this_call, 1);
        assert_eq!(p3.steps_total, 3);
        assert!(
            p3.commit_ready,
            "max_denoise_steps=3 must make commit_ready after three advances"
        );

        let result = commit_diffusion_workspace(
            ws,
            &cfg,
            &weights,
            &mut cache,
            DiffusionCommitPolicy {
                truncation_terminal_ids: &[],
                request_terminal_ids: &[],
                remaining_output_budget: Some(4),
            },
        );
        assert_eq!(result.denoise_steps, 3);
        assert_eq!(result.tokens.len(), 4);
    }

    // ── Phase C: random scratch reuse + stage profile ────────────────

    #[test]
    fn random_tokens_gpu_from_scratch_reuses_capacity() {
        let mut rng = Xorshift64::new(7);
        let mut scratch = Vec::new();
        let a = random_tokens_gpu_from_scratch(&mut scratch, 16, 1000, &mut rng);
        eval(&[&a]);
        assert_eq!(a.shape(), vec![16]);
        assert_eq!(scratch.len(), 16);
        let cap_after_first = scratch.capacity();
        assert!(cap_after_first >= 16);

        let b = random_tokens_gpu_from_scratch(&mut scratch, 16, 1000, &mut rng);
        eval(&[&b]);
        assert_eq!(b.shape(), vec![16]);
        assert_eq!(scratch.len(), 16);
        // Capacity must not shrink; reuse is the hot-path contract.
        assert!(
            scratch.capacity() >= cap_after_first,
            "scratch capacity regressed: before={cap_after_first} after={}",
            scratch.capacity()
        );
        // Values must be in-vocab.
        for &tok in &scratch {
            assert!(tok < 1000, "token {tok} out of vocab");
        }
    }

    #[test]
    fn denoise_stage_profile_take_zeros_counters() {
        // Force-write via the same atomics the production recorders use.
        DENOISE_STAGE_FORWARD_US.store(11, Ordering::Relaxed);
        DENOISE_STAGE_SAMPLE_US.store(22, Ordering::Relaxed);
        DENOISE_STAGE_COMMIT_US.store(33, Ordering::Relaxed);
        DENOISE_STAGE_STEPS.store(2, Ordering::Relaxed);
        let snap = take_denoise_stage_profile();
        assert_eq!(
            snap,
            DenoiseStageProfile {
                forward_us: 11,
                sample_us: 22,
                commit_us: 33,
                steps: 2,
            }
        );
        let empty = take_denoise_stage_profile();
        assert_eq!(empty, DenoiseStageProfile::default());
    }

    #[test]
    fn record_denoise_stage_is_noop_without_profile_env() {
        // Without AX_DIFFUSION_PROFILE, recorders must not accumulate.
        let _ = take_denoise_stage_profile();
        record_denoise_stage_forward_us(100);
        record_denoise_stage_sample_us(100);
        record_denoise_stage_commit_us(100);
        let snap = take_denoise_stage_profile();
        assert_eq!(snap, DenoiseStageProfile::default());
    }
}
