use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use mlx_sys::{
    MlxArray, MlxStream, clear_cache, enable_compile, max_recommended_working_set_size,
    set_wired_limit,
};

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;
use ax_engine_core::{
    ExecutionRunner, ExecutionStatus, KvWriteSummary, NativeModelArtifacts,
    NativeModelBindingSummary, NativeModelManifest, NativeTensorRole, RequestExecutionUpdate,
    RequestId, RunnerInput, RunnerOutput, StopReason,
};

use crate::generate::{
    advance_greedy_pipeline, chunked_prefill, decode_step, start_greedy_pipeline,
};
use crate::kv_cache::{MlxKVCache, MlxKVCacheUsage};
use crate::model::ModelConfig;
use crate::sampling::Xorshift64;
use crate::speculative::{
    DEFAULT_DRAFT_LEN, DRAFT_CONFIDENCE_THRESHOLD, LINEAR_MIN_NGRAM_SUPPORT, MAX_DRAFT_LEN,
    NgramTable, single_decode, speculative_decode_step,
};
use crate::weights::{ModelWeights, load_weights};

/// Beta prior counts for the speculation accept-rate gate.
///
/// Beta(3, 1) → initial posterior mean = 0.75, above the accept threshold,
/// so speculation is enabled optimistically from the first step and is only
/// suppressed once the posterior accumulates evidence of a low accept rate.
const SPEC_BETA_PRIOR_ALPHA: f32 = 3.0;
const SPEC_BETA_PRIOR_BETA: f32 = 1.0;

/// Cap total Beta observations to ~100 to bound the "memory" of the gate
/// and allow the posterior to adapt if token statistics change mid-sequence.
/// Equivalent to an EMA span of roughly 100 speculative steps.
const SPEC_BETA_MAX_TOTAL: f32 = 100.0;

const SPEC_ACCEPT_THRESHOLD: f32 = 0.5;
const SPEC_RETRY_INTERVAL: u32 = 8;
/// Steps to suppress speculation after a complete miss (0 draft tokens accepted)
/// on a linear-attention model.  Recompute cost is O(1) token regardless of context
/// length, so 128 was far too conservative; 16 gives the n-gram table time to
/// recover without sacrificing the whole generation window.
const LINEAR_SPEC_RETRY_INTERVAL: u32 = 16;
/// Steps to suppress after a *partial* accept (≥1 draft token accepted but not all).
/// Partial accept means the n-gram is close — retry quickly.
const LINEAR_SPEC_PARTIAL_RETRY_INTERVAL: u32 = 4;
/// Maximum number of prompt tail tokens fed into the n-gram table.
/// Long prompts (especially random-token benchmarks) would otherwise fill the
/// table with useless bigrams that trigger false-positive speculation and force
/// expensive recompute on the very first speculative attempt.
const NGRAM_PROMPT_FEED_MAX: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SpeculationTelemetry {
    draft_attempts: u32,
    draft_tokens: u32,
    accepted_tokens: u32,
    rejected_tokens: u32,
    full_accepts: u32,
    partial_rejects: u32,
    complete_misses: u32,
    no_draft_steps: u32,
    cooldown_steps: u32,
    cooldown_events: u32,
    cooldown_steps_scheduled: u32,
}

impl SpeculationTelemetry {
    fn record_draft(&mut self, draft_len: usize, accept_count: usize) {
        self.draft_attempts = self.draft_attempts.saturating_add(1);
        self.draft_tokens = self.draft_tokens.saturating_add(saturating_u32(draft_len));
        self.accepted_tokens = self
            .accepted_tokens
            .saturating_add(saturating_u32(accept_count));
        self.rejected_tokens = self
            .rejected_tokens
            .saturating_add(saturating_u32(draft_len.saturating_sub(accept_count)));

        if accept_count == draft_len {
            self.full_accepts = self.full_accepts.saturating_add(1);
        } else if accept_count == 0 {
            self.complete_misses = self.complete_misses.saturating_add(1);
        } else {
            self.partial_rejects = self.partial_rejects.saturating_add(1);
        }
    }

    fn record_no_draft(&mut self) {
        self.no_draft_steps = self.no_draft_steps.saturating_add(1);
    }

    fn record_cooldown_step(&mut self) {
        self.cooldown_steps = self.cooldown_steps.saturating_add(1);
    }

    fn record_cooldown_event(&mut self, disabled_steps: u32) {
        self.cooldown_events = self.cooldown_events.saturating_add(1);
        self.cooldown_steps_scheduled =
            self.cooldown_steps_scheduled.saturating_add(disabled_steps);
    }

    fn merge_from(&mut self, other: Self) {
        self.draft_attempts = self.draft_attempts.saturating_add(other.draft_attempts);
        self.draft_tokens = self.draft_tokens.saturating_add(other.draft_tokens);
        self.accepted_tokens = self.accepted_tokens.saturating_add(other.accepted_tokens);
        self.rejected_tokens = self.rejected_tokens.saturating_add(other.rejected_tokens);
        self.full_accepts = self.full_accepts.saturating_add(other.full_accepts);
        self.partial_rejects = self.partial_rejects.saturating_add(other.partial_rejects);
        self.complete_misses = self.complete_misses.saturating_add(other.complete_misses);
        self.no_draft_steps = self.no_draft_steps.saturating_add(other.no_draft_steps);
        self.cooldown_steps = self.cooldown_steps.saturating_add(other.cooldown_steps);
        self.cooldown_events = self.cooldown_events.saturating_add(other.cooldown_events);
        self.cooldown_steps_scheduled = self
            .cooldown_steps_scheduled
            .saturating_add(other.cooldown_steps_scheduled);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        let entries = [
            ("ax_spec_draft_attempts", self.draft_attempts),
            ("ax_spec_draft_tokens", self.draft_tokens),
            ("ax_spec_accepted_tokens", self.accepted_tokens),
            ("ax_spec_rejected_tokens", self.rejected_tokens),
            ("ax_spec_full_accepts", self.full_accepts),
            ("ax_spec_partial_rejects", self.partial_rejects),
            ("ax_spec_complete_misses", self.complete_misses),
            ("ax_spec_no_draft_steps", self.no_draft_steps),
            ("ax_spec_cooldown_steps", self.cooldown_steps),
            ("ax_spec_cooldown_events", self.cooldown_events),
            (
                "ax_spec_cooldown_steps_scheduled",
                self.cooldown_steps_scheduled,
            ),
        ];

        decisions.extend(
            entries
                .into_iter()
                .filter(|(_, value)| *value > 0)
                .map(|(key, value)| (key.to_string(), value)),
        );
    }
}

fn saturating_u32(value: usize) -> u32 {
    value.min(u32::MAX as usize) as u32
}

fn saturating_u32_from_u64(value: u64) -> u32 {
    value.min(u32::MAX as u64) as u32
}

fn kib_ceil(bytes: u64) -> u32 {
    if bytes == 0 {
        0
    } else {
        saturating_u32_from_u64(bytes.saturating_add(1023) / 1024)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct KvCacheTelemetry {
    request_snapshots: u32,
    logical_tokens: u64,
    capacity_tokens: u64,
    logical_bytes: u64,
    capacity_bytes: u64,
    full_attention_layers: u64,
    linear_state_layers: u64,
    linear_state_bytes: u64,
    growth_count: u64,
}

impl KvCacheTelemetry {
    fn merge_from(&mut self, usage: MlxKVCacheUsage) {
        self.request_snapshots = self.request_snapshots.saturating_add(1);
        self.logical_tokens = self.logical_tokens.saturating_add(usage.logical_tokens as u64);
        self.capacity_tokens = self
            .capacity_tokens
            .saturating_add(usage.capacity_tokens as u64);
        self.logical_bytes = self.logical_bytes.saturating_add(usage.logical_bytes);
        self.capacity_bytes = self.capacity_bytes.saturating_add(usage.capacity_bytes);
        self.full_attention_layers = self
            .full_attention_layers
            .saturating_add(usage.full_attention_layers as u64);
        self.linear_state_layers = self
            .linear_state_layers
            .saturating_add(usage.linear_state_layers as u64);
        self.linear_state_bytes = self
            .linear_state_bytes
            .saturating_add(usage.linear_state_bytes);
        self.growth_count = self.growth_count.saturating_add(usage.growth_count);
    }

    fn append_route_decisions(&self, decisions: &mut Vec<(String, u32)>) {
        if self.request_snapshots == 0 {
            return;
        }

        let entries = [
            ("ax_mlx_kv_request_snapshots", self.request_snapshots),
            (
                "ax_mlx_kv_logical_tokens",
                saturating_u32_from_u64(self.logical_tokens),
            ),
            (
                "ax_mlx_kv_capacity_tokens",
                saturating_u32_from_u64(self.capacity_tokens),
            ),
            ("ax_mlx_kv_logical_kib", kib_ceil(self.logical_bytes)),
            ("ax_mlx_kv_capacity_kib", kib_ceil(self.capacity_bytes)),
            (
                "ax_mlx_kv_full_attention_layers",
                saturating_u32_from_u64(self.full_attention_layers),
            ),
            (
                "ax_mlx_kv_linear_state_layers",
                saturating_u32_from_u64(self.linear_state_layers),
            ),
            (
                "ax_mlx_kv_linear_state_kib",
                kib_ceil(self.linear_state_bytes),
            ),
            (
                "ax_mlx_kv_growth_count",
                saturating_u32_from_u64(self.growth_count),
            ),
        ];

        decisions.extend(
            entries
                .into_iter()
                .filter(|(_, value)| *value > 0)
                .map(|(key, value)| (key.to_string(), value)),
        );
    }
}

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    ngram: NgramTable,
    /// Per-request PRNG for temperature sampling.  Seeded from request_id so
    /// deterministic seeds produce reproducible outputs.
    rng: Xorshift64,
    /// Beta-Bernoulli posterior α for the speculation accept-rate gate.
    /// Incremented by accepted draft tokens each speculative step.
    spec_beta_alpha: f32,
    /// Beta-Bernoulli posterior β for the speculation accept-rate gate.
    /// Incremented by rejected draft tokens each speculative step.
    spec_beta_beta: f32,
    /// Steps remaining before re-enabling speculation (0 = speculation allowed).
    spec_disabled_steps: u32,
    /// Pre-verified bonus tokens ready to serve without a model run.
    bonus_queue: VecDeque<u32>,
    /// The token to use as `last_token` for the next model run.
    /// None on the very first decode step (use framework-supplied input instead).
    next_model_last_token: Option<u32>,
    /// Lazy token from the previous greedy decode step (double-buffer pipeline).
    ///
    /// When `Some`, the next call to `decode_one` uses `advance_greedy_pipeline`
    /// to materialise this token while simultaneously submitting the next step
    /// to the GPU — eliminating the GPU idle gap between steps.
    ///
    /// Only set when `no_speculative = true` and `temperature == 0.0`.
    pending_greedy: Option<MlxArray>,
    /// Cumulative per-request counters surfaced through route metadata for
    /// benchmark auditability.
    speculation: SpeculationTelemetry,
}

impl RequestState {
    fn new(num_layers: usize, request_id: RequestId) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            ngram: NgramTable::new(),
            rng: Xorshift64::new(request_id.0),
            spec_beta_alpha: SPEC_BETA_PRIOR_ALPHA,
            spec_beta_beta: SPEC_BETA_PRIOR_BETA,
            spec_disabled_steps: 0,
            bonus_queue: VecDeque::new(),
            next_model_last_token: None,
            pending_greedy: None,
            speculation: SpeculationTelemetry::default(),
        }
    }

    fn spec_posterior_mean(&self) -> f32 {
        self.spec_beta_alpha / (self.spec_beta_alpha + self.spec_beta_beta)
    }
}

/// ExecutionRunner backed by the MLX inference path.
pub struct MlxRunner {
    cfg: ModelConfig,
    weights: Arc<ModelWeights>,
    prefill_chunk: usize,
    binding_summary: NativeModelBindingSummary,
    states: Mutex<HashMap<RequestId, RequestState>>,
    /// Dedicated GPU stream kept alive for the runner's lifetime.
    _stream: MlxStream,
    /// When true, always use single-token decode (disables n-gram speculation).
    /// Set via `AX_NO_SPEC=1` environment variable for benchmarking isolation.
    no_speculative: bool,
}

impl fmt::Debug for MlxRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxRunner")
            .field("layers", &self.cfg.layer_count)
            .field("vocab", &self.cfg.vocab_size)
            .finish()
    }
}

impl MlxRunner {
    pub fn from_artifacts(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
        no_speculative: bool,
    ) -> Result<Self, MlxRunnerError> {
        // Enable MLX compute-graph compilation globally.
        // This caches and reuses compiled Metal shaders across calls with the same
        // graph structure — the equivalent of mlx_lm's per-step mx.compile() JIT.
        // Without this, MLX rebuilds the dispatch graph on every decode step,
        // causing measurable CPU overhead (~10-15% throughput gap vs mlx_lm).
        enable_compile();

        // Dedicated GPU stream — mirrors mlx_lm's `mx.new_stream(mx.default_device())`.
        // Setting it as default avoids implicit cross-stream synchronization on the
        // shared default stream.
        let stream = MlxStream::new_gpu();
        stream.set_as_default();

        // Wire weights into GPU memory to prevent paging between requests.
        // Use Metal's recommendedMaxWorkingSetSize — values above this are rejected.
        let wired_cap = max_recommended_working_set_size();
        if wired_cap > 0 {
            set_wired_limit(wired_cap);
        }

        validate_mlx_supported_manifest(artifacts)?;

        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        let weights = load_weights(artifacts).map_err(MlxRunnerError::Weights)?;

        let binding_summary = binding_summary_from_specs(artifacts.tensor_specs());

        let weights = Arc::new(weights);

        // JIT warm-up: trigger Metal shader compilation for both decode and prefill paths.
        {
            let mut dummy_cache = MlxKVCache::new(cfg.layer_count);
            let mut dummy_rng = Xorshift64::new(0);
            decode_step(&cfg, &weights, 0, &mut dummy_cache, 0.0, &mut dummy_rng);
            dummy_cache.reset();
            let dummy_tokens: Vec<u32> = vec![0u32; 8];
            chunked_prefill(
                &cfg,
                &weights,
                &dummy_tokens,
                &mut dummy_cache,
                prefill_chunk,
            );
        }

        // AX_NO_SPEC=1 env var overrides the config flag (for benchmarking/debugging).
        // Qwen3.5 linear-attention uses `speculative_decode_step_linear_safe` which
        // clones the cache for verification and recomputes the committed prefix on
        // partial accept — so speculative is safe to enable for linear-attention models.
        let no_speculative = no_speculative
            || std::env::var("AX_NO_SPEC")
                .map(|v| v == "1")
                .unwrap_or(false);

        Ok(Self {
            cfg,
            weights,
            prefill_chunk: prefill_chunk.max(1),
            binding_summary,
            states: Mutex::new(HashMap::new()),
            _stream: stream,
            no_speculative,
        })
    }
}

impl ExecutionRunner for MlxRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let step_id = input.execution_batch.step_id;
        let mut request_updates = Vec::new();
        let logits_handles = Vec::new();
        let logits_outputs = Vec::new();

        let mut route_metadata = input.execution_batch.route_metadata.clone();
        let mut speculation = SpeculationTelemetry::default();
        let mut kv_cache = KvCacheTelemetry::default();

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let result = self.run_item(item, ctx);
            speculation.merge_from(result.speculation);
            kv_cache.merge_from(result.kv_usage);
            request_updates.push(result.update);
        }
        speculation.append_route_decisions(&mut route_metadata.crossover_decisions);
        kv_cache.append_route_decisions(&mut route_metadata.crossover_decisions);

        let tokens_written: u32 = input
            .execution_batch
            .items
            .iter()
            .map(|i| i.scheduled_token_count)
            .sum();

        RunnerOutput {
            step_id,
            request_updates,
            logits_handles,
            logits_outputs,
            kv_write_summary: KvWriteSummary {
                tokens_written,
                blocks_touched: 0,
            },
            route_metadata,
            execution_status: ExecutionStatus::Success,
        }
    }

    fn native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        Some(self.binding_summary)
    }
}

impl MlxRunner {
    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
    ) -> MlxItemRun {
        let token_ids = &item.input_token_slice;
        if token_ids.is_empty() {
            return MlxItemRun {
                update: RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    stop_reason: None,
                    error: Some("empty token slice".into()),
                },
                speculation: SpeculationTelemetry::default(),
                kv_usage: MlxKVCacheUsage::default(),
            };
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);

        // Extract per-request state from the map and release the lock before GPU
        // work.  This ensures a long prefill for one request does not block state
        // access for any other request: the mutex is held only for the O(1)
        // HashMap remove and subsequent insert, never across a GPU forward pass.
        //
        // Concurrency contract: the scheduler must not route the same request_id
        // to two concurrent run() calls — otherwise one call would create a fresh
        // empty state from None while the other holds the extracted state.
        let mut state = {
            let mut states = self.states.lock().unwrap();
            states
                .remove(&item.request_id)
                .unwrap_or_else(|| RequestState::new(self.cfg.layer_count, item.request_id))
        };

        let temperature = ctx.map(|c| c.temperature).unwrap_or(0.0);

        // GPU work — mutex is NOT held during prefill, decode, or speculative steps.
        let sampled_token = match item.mode {
            ExecutionMode::Prefill => {
                let tok = chunked_prefill(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    &mut state.cache,
                    self.prefill_chunk,
                );
                // Seed the n-gram table with the tail of the prompt.
                // Only the last NGRAM_PROMPT_FEED_MAX tokens are fed: long prompts
                // (e.g. random-token benchmarks with 512+ tokens) would otherwise
                // inject hundreds of useless bigrams, causing a false-positive spec
                // attempt on the very first decode step and disabling speculation
                // for LINEAR_SPEC_RETRY_INTERVAL steps — wiping out most of the
                // generation window.
                let feed_start = token_ids.len().saturating_sub(NGRAM_PROMPT_FEED_MAX);
                state.ngram.feed(&token_ids[feed_start..]);
                // Reset bonus state for this new generation.
                state.bonus_queue.clear();
                state.next_model_last_token = None;
                state.pending_greedy = None;

                // Bootstrap the double-buffer greedy pipeline: submit the second token's
                // forward pass to the GPU asynchronously so the first decode step can
                // materialise it while the GPU is already computing the third token.
                // Only for greedy (temperature = 0) non-speculative runs.
                if self.no_speculative && temperature == 0.0 {
                    state.pending_greedy = Some(start_greedy_pipeline(
                        &self.cfg,
                        &self.weights,
                        tok,
                        &mut state.cache,
                    ));
                }

                tok
            }
            ExecutionMode::Decode => self.decode_one(&mut state, token_ids, temperature),
        };

        let stop_reason = if generated_len + 1 >= max_output {
            Some(StopReason::MaxOutputTokens)
        } else {
            None
        };

        // Re-insert state only if the request continues — lock held briefly.
        let speculation = state.speculation;
        let kv_usage = state.cache.usage_snapshot();
        if stop_reason.is_none() {
            let mut states = self.states.lock().unwrap();
            states.insert(item.request_id, state);
        } else {
            // Free MLX's intermediate graph and compute cache after each completed
            // request.  Mirrors mlx_lm's mx.metal.clear_cache() at end of generation;
            // reclaims GPU memory that would otherwise persist until the next request.
            clear_cache();
        }

        MlxItemRun {
            update: RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: item.scheduled_token_count,
                output_token: Some(sampled_token),
                stop_reason,
                error: None,
            },
            speculation,
            kv_usage,
        }
    }

    /// Produce one output token for a decode step.
    ///
    /// Pops from the bonus queue when pre-verified tokens are available.
    /// Uses the double-buffer greedy pipeline when `no_speculative = true` and
    /// `temperature == 0.0` (bootstrapped during prefill).
    /// Otherwise runs a speculative or single-token decode pass.
    fn decode_one(&self, state: &mut RequestState, input_tokens: &[u32], temperature: f32) -> u32 {
        // Serve pre-verified bonus tokens without re-running the model.
        // (Bonus tokens only exist on the speculative path; the greedy pipeline
        // never populates the bonus queue.)
        if let Some(tok) = state.bonus_queue.pop_front() {
            return tok;
        }

        // Double-buffer greedy pipeline: materialise the pending lazy token while
        // simultaneously submitting the next step to the GPU.  This mirrors
        // mlx_lm's `_step(y)` → `async_eval(next_y)` → `eval(y)` loop and
        // eliminates the GPU idle gap between consecutive greedy decode steps.
        if self.no_speculative
            && temperature == 0.0
            && let Some(pending) = state.pending_greedy.take()
        {
            let (tok, next_pending) =
                advance_greedy_pipeline(&self.cfg, &self.weights, &pending, &mut state.cache);
            state.pending_greedy = Some(next_pending);
            return tok;
        }

        let last_token = state
            .next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);

        let result = self.run_model_decode(state, last_token, temperature);

        // result[0] is the output for this step.
        // result[1..last] are bonus tokens (KVs already in cache).
        // result[last] is the starting point for the next model run (KV not yet in cache).
        let output = result[0];

        // Queue bonus tokens (intermediate accepted drafts).
        // result[1..len-1] is empty for len<=2; use get() to avoid panic on len=1.
        for &t in result.get(1..result.len().saturating_sub(1)).unwrap_or(&[]) {
            state.bonus_queue.push_back(t);
        }

        // The last element drives the next model run.
        state.next_model_last_token = result.last().copied();

        output
    }

    /// Run one model decode step (speculative or single), updating the Beta-Bernoulli gate.
    fn run_model_decode(
        &self,
        state: &mut RequestState,
        last_token: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // Runtime opt-out via AX_NO_SPEC=1 for benchmarking isolation.
        if self.no_speculative {
            return single_decode(
                &self.cfg,
                &self.weights,
                &mut state.cache,
                &mut state.ngram,
                last_token,
                temperature,
                &mut state.rng,
            );
        }

        // Speculation disabled: count down and use single decode.
        if state.spec_disabled_steps > 0 {
            state.spec_disabled_steps -= 1;
            state.speculation.record_cooldown_step();
            return single_decode(
                &self.cfg,
                &self.weights,
                &mut state.cache,
                &mut state.ngram,
                last_token,
                temperature,
                &mut state.rng,
            );
        }

        let draft = speculative_draft(&state.ngram, self.cfg.linear_attention.is_some());
        if draft.is_empty() {
            state.speculation.record_no_draft();
            return single_decode(
                &self.cfg,
                &self.weights,
                &mut state.cache,
                &mut state.ngram,
                last_token,
                temperature,
                &mut state.rng,
            );
        }

        let draft_len = draft.len();
        let result = speculative_decode_step(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            &draft,
            temperature,
            &mut state.rng,
        );

        // Beta-Bernoulli posterior update.
        // accept_count = result.len() - 1 (last element is next model input, not bonus).
        let accept_count = result.len().saturating_sub(1);
        state.speculation.record_draft(draft_len, accept_count);
        state.spec_beta_alpha += accept_count as f32;
        state.spec_beta_beta += (draft_len - accept_count) as f32;

        // Normalise to keep total observations bounded — prevents the posterior
        // from becoming overconfident and unable to adapt to changing statistics.
        let total = state.spec_beta_alpha + state.spec_beta_beta;
        if total > SPEC_BETA_MAX_TOTAL {
            let scale = SPEC_BETA_MAX_TOTAL / total;
            state.spec_beta_alpha *= scale;
            state.spec_beta_beta *= scale;
        }

        if let Some(disabled_steps) = speculative_disabled_steps(
            self.cfg.linear_attention.is_some(),
            accept_count,
            draft_len,
            state.spec_posterior_mean(),
        ) {
            state.spec_disabled_steps = disabled_steps;
            state.speculation.record_cooldown_event(disabled_steps);
        }

        result
    }
}

struct MlxItemRun {
    update: RequestExecutionUpdate,
    speculation: SpeculationTelemetry,
    kv_usage: MlxKVCacheUsage,
}

fn speculative_disabled_steps(
    has_linear_attention: bool,
    accept_count: usize,
    draft_len: usize,
    posterior_mean: f32,
) -> Option<u32> {
    if draft_len == 0 {
        return None;
    }

    if has_linear_attention {
        // Linear-attention recurrent state cannot be rolled back with trim_to; any
        // partial reject pays branch verification + committed-prefix recompute.
        // Recompute cost is O(accepted+1) tokens — bounded at DEFAULT_DRAFT_LEN+1,
        // not O(context length) — so a large retry interval is unwarranted.
        //
        // Differentiate complete miss from partial accept: a partial accept means
        // the n-gram was directionally correct; retry quickly.  A complete miss
        // means the table prediction is off; back off longer.
        if accept_count == 0 {
            return Some(LINEAR_SPEC_RETRY_INTERVAL); // complete miss: 16 steps
        }
        return (accept_count < draft_len).then_some(LINEAR_SPEC_PARTIAL_RETRY_INTERVAL); // partial: 4 steps
    }

    (posterior_mean < SPEC_ACCEPT_THRESHOLD).then_some(SPEC_RETRY_INTERVAL)
}

fn speculative_draft(ngram: &NgramTable, has_linear_attention: bool) -> Vec<u32> {
    if has_linear_attention {
        // Dense rollback is O(1); linear-attention partial-reject pays
        // branch/recompute, so cap at DEFAULT_DRAFT_LEN to bound recompute cost.
        ngram.predict_with_confidence(
            DEFAULT_DRAFT_LEN,
            LINEAR_MIN_NGRAM_SUPPORT,
            DRAFT_CONFIDENCE_THRESHOLD,
        )
    } else {
        // Dense models extend up to MAX_DRAFT_LEN when the n-gram chain is
        // high-confidence; the confidence gate stops the chain early otherwise.
        ngram.predict_with_confidence(MAX_DRAFT_LEN, 1, DRAFT_CONFIDENCE_THRESHOLD)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxRunnerError {
    #[error("MLX model feature is not supported: {0}")]
    UnsupportedFeature(String),
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}

fn validate_mlx_supported_manifest(artifacts: &NativeModelArtifacts) -> Result<(), MlxRunnerError> {
    let manifest = artifacts.manifest();
    if manifest.linear_attention.is_enabled() || has_linear_attention_tensors(artifacts) {
        validate_qwen_gated_delta_linear_attention(manifest)?;
    }
    if manifest.sliding_window_size.is_some()
        || !manifest.layer_types.is_empty()
        || !manifest.kv_shared_source_layers.is_empty()
        || manifest.global_head_dim.is_some()
        || manifest.rope_theta_swa.is_some()
    {
        validate_gemma4_interleaved_attention(manifest)?;
    }
    Ok(())
}

fn has_linear_attention_tensors(artifacts: &NativeModelArtifacts) -> bool {
    artifacts.tensor_specs().iter().any(|tensor| {
        matches!(
            tensor.role,
            NativeTensorRole::LinearAttentionInProjQkv
                | NativeTensorRole::LinearAttentionInProjZ
                | NativeTensorRole::LinearAttentionInProjA
                | NativeTensorRole::LinearAttentionInProjB
                | NativeTensorRole::LinearAttentionConv1d
                | NativeTensorRole::LinearAttentionDtBias
                | NativeTensorRole::LinearAttentionALog
                | NativeTensorRole::LinearAttentionNorm
                | NativeTensorRole::LinearAttentionOutProj
        )
    })
}

fn binding_summary_from_specs(
    specs: &[ax_engine_core::NativeTensorSpec],
) -> NativeModelBindingSummary {
    let mut summary = NativeModelBindingSummary {
        bindings_prepared: true,
        buffers_bound: true,
        buffer_count: specs.len().min(u32::MAX as usize) as u32,
        buffer_bytes: 0,
        source_quantized_binding_count: 0,
        source_q4_k_binding_count: 0,
        source_q5_k_binding_count: 0,
        source_q6_k_binding_count: 0,
        source_q8_0_binding_count: 0,
    };

    for spec in specs {
        summary.buffer_bytes = summary.buffer_bytes.saturating_add(spec.length_bytes);
        if !spec.source_quantized {
            continue;
        }
        summary.source_quantized_binding_count =
            summary.source_quantized_binding_count.saturating_add(1);
        match spec.source_tensor_type.as_deref() {
            Some("q4_k") => {
                summary.source_q4_k_binding_count =
                    summary.source_q4_k_binding_count.saturating_add(1);
            }
            Some("q5_k") => {
                summary.source_q5_k_binding_count =
                    summary.source_q5_k_binding_count.saturating_add(1);
            }
            Some("q6_k") => {
                summary.source_q6_k_binding_count =
                    summary.source_q6_k_binding_count.saturating_add(1);
            }
            Some("q8_0") => {
                summary.source_q8_0_binding_count =
                    summary.source_q8_0_binding_count.saturating_add(1);
            }
            _ => {}
        }
    }

    summary
}

fn validate_qwen_gated_delta_linear_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if !matches!(manifest.model_family.as_str(), "qwen3_5" | "qwen3_next") {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention is currently supported only for qwen3_5/qwen3_next MLX manifests"
                .to_string(),
        ));
    }
    let Some(key_head_dim) = manifest.linear_attention.key_head_dim else {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention.key_head_dim must be configured".to_string(),
        ));
    };
    if key_head_dim % 32 != 0 {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "linear_attention.key_head_dim {key_head_dim} must be divisible by 32 for the MLX gated-delta kernel"
        )));
    }
    Ok(())
}

fn validate_gemma4_interleaved_attention(
    manifest: &NativeModelManifest,
) -> Result<(), MlxRunnerError> {
    if manifest.model_family != "gemma4" {
        return Err(MlxRunnerError::UnsupportedFeature(
            "interleaved sliding/full attention is only implemented for Gemma4 manifests"
                .to_string(),
        ));
    }
    if manifest.layer_types.len() != manifest.layer_count as usize {
        return Err(MlxRunnerError::UnsupportedFeature(format!(
            "Gemma4 interleaved attention requires one layer_type per layer, got {} for {} layers",
            manifest.layer_types.len(),
            manifest.layer_count
        )));
    }

    for (idx, layer_type) in manifest.layer_types.iter().enumerate() {
        if layer_type != "sliding_attention" && layer_type != "full_attention" {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 layer {idx} uses unsupported layer_type {layer_type:?}"
            )));
        }
    }

    let has_sliding = manifest
        .layer_types
        .iter()
        .any(|layer_type| layer_type == "sliding_attention");
    if has_sliding && manifest.sliding_window_size.is_none() {
        return Err(MlxRunnerError::UnsupportedFeature(
            "Gemma4 sliding_attention layers require sliding_window_size".to_string(),
        ));
    }

    for (&layer, &source) in &manifest.kv_shared_source_layers {
        if layer >= manifest.layer_count || source >= manifest.layer_count || source >= layer {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 KV-shared layer {layer} has invalid source layer {source}"
            )));
        }
        let layer_type = &manifest.layer_types[layer as usize];
        let source_type = &manifest.layer_types[source as usize];
        if layer_type != source_type {
            return Err(MlxRunnerError::UnsupportedFeature(format!(
                "Gemma4 KV-shared layer {layer} type {layer_type:?} cannot reuse source {source} type {source_type:?}"
            )));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ax_engine_core::{
        AX_NATIVE_MODEL_MANIFEST_FILE, NativeLinearAttentionConfig, NativeModelManifest,
        NativeMoeConfig, NativeRuntimeStatus, NativeTensorDataType, NativeTensorFormat,
        NativeTensorSpec,
    };
    use std::fs;
    use std::path::{Path, PathBuf};

    // Verify that the extract-work-reinsert mutex pattern correctly isolates
    // per-request state without GPU execution required.
    #[test]
    fn state_extraction_isolates_concurrent_requests() {
        let mut states: HashMap<RequestId, RequestState> = HashMap::new();
        let a = RequestId(1);
        let b = RequestId(2);

        // Extract A from the map (simulates the lock-brief-remove step).
        // While A is extracted, B's slot is accessible without contention.
        let state_a = states.remove(&a).unwrap_or_else(|| RequestState::new(2, a));
        let state_b = states.remove(&b).unwrap_or_else(|| RequestState::new(2, b));

        // GPU work would run here with state_a / state_b outside the map.
        // Verify B can be reinserted independently of A.
        states.insert(b, state_b);
        states.insert(a, state_a);

        assert_eq!(states.len(), 2);
        assert!(states.contains_key(&a));
        assert!(states.contains_key(&b));
    }

    #[test]
    fn completed_request_state_is_not_reinserted() {
        let mut states: HashMap<RequestId, RequestState> = HashMap::new();
        let id = RequestId(42);
        states.insert(id, RequestState::new(2, id));

        // Extract and simulate a completed request (stop_reason.is_some()).
        // The state should not be reinserted, mirroring the run_item control flow.
        let _state = states.remove(&id).unwrap();
        // No states.insert here — dropped at end of scope.

        assert!(
            !states.contains_key(&id),
            "completed request must not leave orphaned state"
        );
    }

    #[test]
    fn prefill_clears_bonus_and_last_token() {
        let mut state = RequestState::new(2, RequestId(0));
        state.bonus_queue.push_back(99);
        state.bonus_queue.push_back(100);
        state.next_model_last_token = Some(5);

        // Simulate the prefill reset branch of run_item.
        state.bonus_queue.clear();
        state.next_model_last_token = None;

        assert!(
            state.bonus_queue.is_empty(),
            "bonus queue must be cleared on prefill"
        );
        assert!(
            state.next_model_last_token.is_none(),
            "last_token pointer must be reset on prefill"
        );
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ax-mlx-runner-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn tensor(
        name: &str,
        role: NativeTensorRole,
        layer_index: Option<u32>,
        shape: Vec<u64>,
    ) -> NativeTensorSpec {
        NativeTensorSpec {
            name: name.to_string(),
            role,
            layer_index,
            dtype: NativeTensorDataType::F16,
            source_tensor_type: None,
            source_quantized: false,
            quantization: None,
            quantized_source: None,
            shape,
            file: PathBuf::from("model.safetensors"),
            offset_bytes: 0,
            length_bytes: 32,
        }
    }

    #[test]
    fn binding_summary_reports_manifest_bytes_and_quantized_sources() {
        let dense = tensor("dense", NativeTensorRole::AttentionNorm, Some(0), vec![4]);
        let mut q4 = tensor("q4", NativeTensorRole::AttentionQ, Some(0), vec![8, 4]);
        q4.source_quantized = true;
        q4.source_tensor_type = Some("q4_k".to_string());
        q4.length_bytes = 64;
        let mut u32_affine = tensor("u32", NativeTensorRole::AttentionO, Some(0), vec![4, 1]);
        u32_affine.source_quantized = true;
        u32_affine.length_bytes = 16;

        let summary = binding_summary_from_specs(&[dense, q4, u32_affine]);

        assert!(summary.bindings_prepared);
        assert!(summary.buffers_bound);
        assert_eq!(summary.buffer_count, 3);
        assert_eq!(summary.buffer_bytes, 112);
        assert_eq!(summary.source_quantized_binding_count, 2);
        assert_eq!(summary.source_q4_k_binding_count, 1);
        assert_eq!(summary.source_q5_k_binding_count, 0);
        assert_eq!(summary.source_q6_k_binding_count, 0);
        assert_eq!(summary.source_q8_0_binding_count, 0);
    }

    fn dense_manifest() -> NativeModelManifest {
        NativeModelManifest {
            schema_version: ax_engine_core::AX_NATIVE_MODEL_MANIFEST_SCHEMA_VERSION.to_string(),
            model_family: "test_dense".to_string(),
            tensor_format: NativeTensorFormat::Safetensors,
            source_quantization: None,
            runtime_status: NativeRuntimeStatus::default(),
            layer_count: 1,
            hidden_size: 4,
            intermediate_size: 8,
            attention_head_count: 1,
            attention_head_dim: 4,
            kv_head_count: 1,
            vocab_size: 16,
            tie_word_embeddings: false,
            rope_theta: None,
            rope_theta_swa: None,
            query_pre_attn_scalar: None,
            attention_logit_softcap: None,
            attn_output_gate: false,
            partial_rotary_factor: None,
            attention_value_from_key_layers: Vec::new(),
            attention_v_norm_no_scale_layers: Vec::new(),
            global_head_dim: None,
            sliding_window_size: None,
            layer_types: Vec::new(),
            kv_shared_source_layers: Default::default(),
            final_logit_softcapping: None,
            hidden_states_scale: None,
            moe_norm_topk_prob: false,
            hidden_size_per_layer_input: 0,
            vocab_size_per_layer_input: None,
            linear_attention: NativeLinearAttentionConfig::default(),
            moe: NativeMoeConfig::default(),
            tensors: vec![
                tensor(
                    "model.embed_tokens.weight",
                    NativeTensorRole::TokenEmbedding,
                    None,
                    vec![16, 4],
                ),
                tensor(
                    "model.norm.weight",
                    NativeTensorRole::FinalNorm,
                    None,
                    vec![4],
                ),
                tensor(
                    "lm_head.weight",
                    NativeTensorRole::LmHead,
                    None,
                    vec![16, 4],
                ),
                tensor(
                    "model.layers.0.input_layernorm.weight",
                    NativeTensorRole::AttentionNorm,
                    Some(0),
                    vec![4],
                ),
                tensor(
                    "model.layers.0.self_attn.q_proj.weight",
                    NativeTensorRole::AttentionQ,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.k_proj.weight",
                    NativeTensorRole::AttentionK,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.v_proj.weight",
                    NativeTensorRole::AttentionV,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.self_attn.o_proj.weight",
                    NativeTensorRole::AttentionO,
                    Some(0),
                    vec![4, 4],
                ),
                tensor(
                    "model.layers.0.mlp.norm.weight",
                    NativeTensorRole::FfnNorm,
                    Some(0),
                    vec![4],
                ),
                tensor(
                    "model.layers.0.mlp.gate_proj.weight",
                    NativeTensorRole::FfnGate,
                    Some(0),
                    vec![8, 4],
                ),
                tensor(
                    "model.layers.0.mlp.up_proj.weight",
                    NativeTensorRole::FfnUp,
                    Some(0),
                    vec![8, 4],
                ),
                tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    NativeTensorRole::FfnDown,
                    Some(0),
                    vec![4, 8],
                ),
            ],
        }
    }

    fn write_artifacts(manifest: NativeModelManifest) -> NativeModelArtifacts {
        let dir = unique_test_dir("manifest");
        fs::create_dir_all(&dir).expect("fixture directory should create");
        fs::write(dir.join("model.safetensors"), vec![0_u8; 4096]).expect("weights should write");
        fs::write(
            dir.join(AX_NATIVE_MODEL_MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).expect("manifest should serialize"),
        )
        .expect("manifest should write");
        NativeModelArtifacts::from_dir(&dir).expect("fixture manifest should validate")
    }

    fn qwen35_linear_manifest() -> NativeModelManifest {
        let mut manifest = dense_manifest();
        manifest.model_family = "qwen3_5".to_string();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: None,
            num_value_heads: Some(1),
            num_key_heads: Some(1),
            key_head_dim: Some(32),
            value_head_dim: Some(4),
            conv_kernel_dim: Some(4),
        };
        manifest.tensors.retain(|tensor| {
            !matches!(
                tensor.role,
                NativeTensorRole::AttentionQ
                    | NativeTensorRole::AttentionK
                    | NativeTensorRole::AttentionV
                    | NativeTensorRole::AttentionO
            )
        });
        manifest.tensors.extend([
            tensor(
                "model.layers.0.linear_attn.in_proj_qkv.weight",
                NativeTensorRole::LinearAttentionInProjQkv,
                Some(0),
                vec![68, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_z.weight",
                NativeTensorRole::LinearAttentionInProjZ,
                Some(0),
                vec![4, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_a.weight",
                NativeTensorRole::LinearAttentionInProjA,
                Some(0),
                vec![1, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.in_proj_b.weight",
                NativeTensorRole::LinearAttentionInProjB,
                Some(0),
                vec![1, 4],
            ),
            tensor(
                "model.layers.0.linear_attn.conv1d.weight",
                NativeTensorRole::LinearAttentionConv1d,
                Some(0),
                vec![68, 4, 1],
            ),
            tensor(
                "model.layers.0.linear_attn.dt_bias",
                NativeTensorRole::LinearAttentionDtBias,
                Some(0),
                vec![1],
            ),
            tensor(
                "model.layers.0.linear_attn.A_log",
                NativeTensorRole::LinearAttentionALog,
                Some(0),
                vec![1],
            ),
            tensor(
                "model.layers.0.linear_attn.norm.weight",
                NativeTensorRole::LinearAttentionNorm,
                Some(0),
                vec![4],
            ),
            tensor(
                "model.layers.0.linear_attn.out_proj.weight",
                NativeTensorRole::LinearAttentionOutProj,
                Some(0),
                vec![4, 4],
            ),
        ]);
        manifest
    }

    #[test]
    fn mlx_manifest_validation_rejects_linear_attention_for_non_qwen35() {
        let mut manifest = dense_manifest();
        manifest.linear_attention = NativeLinearAttentionConfig {
            full_attention_interval: Some(4),
            num_value_heads: Some(1),
            num_key_heads: Some(1),
            key_head_dim: Some(4),
            value_head_dim: Some(4),
            conv_kernel_dim: Some(4),
        };
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("linear attention should fail closed");

        assert!(error.to_string().contains("qwen3_5/qwen3_next"));
    }

    #[test]
    fn mlx_manifest_validation_allows_qwen35_linear_attention() {
        let artifacts = write_artifacts(qwen35_linear_manifest());

        validate_mlx_supported_manifest(&artifacts)
            .expect("Qwen3.5 linear attention is wired for the MLX path");
    }

    #[test]
    fn real_mlx_manifest_resolves_qwen35_linear_interval_when_configured() {
        let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
            return;
        };
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .expect("real MLX manifest should load");

        validate_mlx_supported_manifest(&artifacts).expect("real MLX manifest should be supported");
        let binding = binding_summary_from_specs(artifacts.tensor_specs());
        assert!(binding.buffer_bytes > 0);
        assert!(binding.source_quantized_binding_count > 0);
        let cfg = ModelConfig::from_manifest(artifacts.manifest());

        assert_eq!(
            cfg.linear_attention
                .as_ref()
                .expect("real manifest should configure linear attention")
                .full_attention_interval,
            4
        );
        assert!(cfg.is_linear_attention_layer(0));
        assert!(!cfg.is_linear_attention_layer(3));
    }

    #[test]
    fn real_mlx_runner_warms_up_qwen35_when_configured() {
        if std::env::var("AX_ENGINE_MLX_RUN_REAL_FORWARD").as_deref() != Ok("1") {
            return;
        }
        let Ok(model_dir) = std::env::var("AX_ENGINE_MLX_REAL_MODEL_DIR") else {
            return;
        };
        let artifacts = NativeModelArtifacts::from_dir(Path::new(&model_dir))
            .expect("real MLX manifest should load");

        MlxRunner::from_artifacts(&artifacts, 8, true)
            .expect("real Qwen3.5 MLX runner should warm up");
    }

    #[test]
    fn linear_attention_speculation_cools_down_after_reject() {
        // Complete miss (0 accepted): long cooldown.
        assert_eq!(
            speculative_disabled_steps(true, 0, DEFAULT_DRAFT_LEN, 0.95),
            Some(LINEAR_SPEC_RETRY_INTERVAL)
        );
        // Partial accept (some but not all): short cooldown to retry quickly.
        assert_eq!(
            speculative_disabled_steps(true, 3, DEFAULT_DRAFT_LEN, 0.95),
            Some(LINEAR_SPEC_PARTIAL_RETRY_INTERVAL)
        );
        // Full accept: no cooldown.
        assert_eq!(
            speculative_disabled_steps(true, DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN, 0.25),
            None
        );
    }

    #[test]
    fn speculation_telemetry_records_acceptance_and_cooldown_counters() {
        let mut telemetry = SpeculationTelemetry::default();

        telemetry.record_no_draft();
        telemetry.record_draft(DEFAULT_DRAFT_LEN, DEFAULT_DRAFT_LEN);
        telemetry.record_draft(DEFAULT_DRAFT_LEN, 0);
        telemetry.record_draft(DEFAULT_DRAFT_LEN, 2);
        telemetry.record_cooldown_step();
        telemetry.record_cooldown_event(4);

        let mut decisions = Vec::new();
        telemetry.append_route_decisions(&mut decisions);
        let decisions = decisions
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();

        assert_eq!(decisions.get("ax_spec_no_draft_steps"), Some(&1));
        assert_eq!(decisions.get("ax_spec_draft_attempts"), Some(&3));
        assert_eq!(
            decisions.get("ax_spec_draft_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 * 3))
        );
        assert_eq!(
            decisions.get("ax_spec_accepted_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 + 2))
        );
        assert_eq!(
            decisions.get("ax_spec_rejected_tokens"),
            Some(&(DEFAULT_DRAFT_LEN as u32 * 2 - 2))
        );
        assert_eq!(decisions.get("ax_spec_full_accepts"), Some(&1));
        assert_eq!(decisions.get("ax_spec_complete_misses"), Some(&1));
        assert_eq!(decisions.get("ax_spec_partial_rejects"), Some(&1));
        assert_eq!(decisions.get("ax_spec_cooldown_steps"), Some(&1));
        assert_eq!(decisions.get("ax_spec_cooldown_events"), Some(&1));
        assert_eq!(decisions.get("ax_spec_cooldown_steps_scheduled"), Some(&4));
    }

    #[test]
    fn dense_speculation_uses_beta_posterior_gate() {
        // Posterior mean above threshold → no cooldown.
        assert_eq!(
            speculative_disabled_steps(false, 3, DEFAULT_DRAFT_LEN, 0.95),
            None
        );
        // Posterior mean below threshold → cooldown period.
        assert_eq!(
            speculative_disabled_steps(false, 0, DEFAULT_DRAFT_LEN, 0.49),
            Some(SPEC_RETRY_INTERVAL)
        );
    }

    #[test]
    fn linear_attention_draft_requires_repeated_ngram_evidence() {
        let mut ngram = NgramTable::new();
        ngram.feed(&[1, 2, 3, 1, 2, 3]);

        // Dense: 3-token cycle builds high-confidence bigrams → draft up to MAX_DRAFT_LEN.
        let dense_draft = speculative_draft(&ngram, false);
        assert!(!dense_draft.is_empty(), "dense draft should be non-empty");
        assert!(
            dense_draft.len() <= MAX_DRAFT_LEN,
            "dense draft must not exceed MAX_DRAFT_LEN"
        );

        // Linear-attention: min_support=2 filters one-off n-grams.
        assert!(
            speculative_draft(&ngram, true).is_empty(),
            "linear attention should not probe one-off prompt n-grams"
        );

        ngram.feed(&[1, 2, 3]);
        let lin_draft = speculative_draft(&ngram, true);
        assert!(
            !lin_draft.is_empty(),
            "linear attention draft should be non-empty after second repeat"
        );
        assert!(
            lin_draft.len() <= DEFAULT_DRAFT_LEN,
            "linear attention draft must not exceed DEFAULT_DRAFT_LEN"
        );
    }

    #[test]
    fn mlx_manifest_validation_rejects_unsupported_linear_key_dim() {
        let mut manifest = qwen35_linear_manifest();
        manifest.linear_attention.key_head_dim = Some(4);
        for tensor in &mut manifest.tensors {
            match tensor.role {
                NativeTensorRole::LinearAttentionInProjQkv => tensor.shape = vec![12, 4],
                NativeTensorRole::LinearAttentionConv1d => tensor.shape = vec![12, 4, 1],
                _ => {}
            }
        }
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("Dk must match the gated-delta kernel contract");

        assert!(error.to_string().contains("divisible by 32"));
    }

    #[test]
    fn mlx_manifest_validation_allows_attn_output_gate() {
        let mut manifest = dense_manifest();
        manifest.attn_output_gate = true;
        manifest
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == NativeTensorRole::AttentionQ)
            .expect("q tensor should exist")
            .shape = vec![8, 4];
        let artifacts = write_artifacts(manifest);

        validate_mlx_supported_manifest(&artifacts)
            .expect("attention output gate is implemented in the MLX model graph");
    }

    #[test]
    fn mlx_manifest_validation_allows_gemma4_interleaved_attention() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec!["sliding_attention".to_string()];
        manifest.global_head_dim = Some(8);
        let artifacts = write_artifacts(manifest);

        validate_mlx_supported_manifest(&artifacts)
            .expect("Gemma4 interleaved attention is implemented in the MLX model graph");
    }

    #[test]
    fn mlx_manifest_validation_rejects_unknown_interleaved_attention() {
        let mut manifest = dense_manifest();
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec!["sliding_attention".to_string()];
        manifest.global_head_dim = Some(8);
        let artifacts = write_artifacts(manifest);

        let error = validate_mlx_supported_manifest(&artifacts)
            .expect_err("non-Gemma4 interleaved attention should fail closed");

        assert!(error.to_string().contains("Gemma4"));
    }

    #[test]
    fn mlx_manifest_validation_allows_valid_gemma4_kv_shared_layers() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.layer_count = 2;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "sliding_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);

        validate_gemma4_interleaved_attention(&manifest)
            .expect("same-type Gemma4 KV sharing should be supported");
    }

    #[test]
    fn mlx_manifest_validation_rejects_cross_type_gemma4_kv_shared_layers() {
        let mut manifest = dense_manifest();
        manifest.model_family = "gemma4".to_string();
        manifest.layer_count = 2;
        manifest.sliding_window_size = Some(1024);
        manifest.layer_types = vec![
            "sliding_attention".to_string(),
            "full_attention".to_string(),
        ];
        manifest.kv_shared_source_layers.insert(1, 0);

        let error = validate_gemma4_interleaved_attention(&manifest)
            .expect_err("cross-type KV sharing should fail closed");

        assert!(error.to_string().contains("cannot reuse"));
    }
}
