use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use mlx_sys::{MlxStream, max_recommended_working_set_size, set_wired_limit};

use ax_engine_core::{
    ExecutionRunner, ExecutionStatus, KvWriteSummary, NativeModelArtifacts,
    NativeModelBindingSummary, RequestExecutionUpdate, RequestId,
    RouteMetadata, RunnerInput, RunnerOutput, StopReason,
};
use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;

use crate::generate::{chunked_prefill, decode_step};
use crate::kv_cache::MlxKVCache;
use crate::model::ModelConfig;
use crate::speculative::{NgramTable, speculative_decode_step, single_decode, DEFAULT_DRAFT_LEN};
use crate::weights::{ModelWeights, load_weights};

const EMA_ALPHA: f32 = 0.1;
const SPEC_ACCEPT_THRESHOLD: f32 = 0.5;
const SPEC_RETRY_INTERVAL: u32 = 8;

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    ngram: NgramTable,
    /// EMA of speculative accept rate (1.0 = always accept, 0.0 = never).
    spec_ema: f32,
    /// Steps remaining before re-enabling speculation (0 = speculation allowed).
    spec_disabled_steps: u32,
    /// Pre-verified bonus tokens ready to serve without a model run.
    bonus_queue: VecDeque<u32>,
    /// The token to use as `last_token` for the next model run.
    /// None on the very first decode step (use framework-supplied input instead).
    next_model_last_token: Option<u32>,
}

impl RequestState {
    fn new(num_layers: usize) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            ngram: NgramTable::new(),
            spec_ema: 1.0,
            spec_disabled_steps: 0,
            bonus_queue: VecDeque::new(),
            next_model_last_token: None,
        }
    }
}

/// ExecutionRunner backed by the MLX native inference path.
pub struct MlxNativeRunner {
    cfg: ModelConfig,
    weights: Arc<ModelWeights>,
    prefill_chunk: usize,
    binding_summary: NativeModelBindingSummary,
    states: Mutex<HashMap<RequestId, RequestState>>,
    /// Dedicated GPU stream kept alive for the runner's lifetime.
    _stream: MlxStream,
}

impl fmt::Debug for MlxNativeRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlxNativeRunner")
            .field("layers", &self.cfg.layer_count)
            .field("vocab", &self.cfg.vocab_size)
            .finish()
    }
}

impl MlxNativeRunner {
    pub fn from_artifacts(
        artifacts: &NativeModelArtifacts,
        prefill_chunk: usize,
    ) -> Result<Self, MlxRunnerError> {
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

        let cfg = ModelConfig::from_manifest(artifacts.manifest());
        let weights = load_weights(artifacts).map_err(MlxRunnerError::Weights)?;

        let binding_summary = NativeModelBindingSummary {
            bindings_prepared: true,
            buffers_bound: true,
            buffer_count: artifacts.tensor_specs().len() as u32,
            buffer_bytes: 0,
            source_quantized_binding_count: 0,
            source_q4_k_binding_count: 0,
            source_q5_k_binding_count: 0,
            source_q6_k_binding_count: 0,
            source_q8_0_binding_count: 0,
        };

        let weights = Arc::new(weights);

        // JIT warm-up: trigger Metal shader compilation for both decode and prefill paths.
        {
            let mut dummy_cache = MlxKVCache::new(cfg.layer_count);
            decode_step(&cfg, &weights, 0, &mut dummy_cache);
            dummy_cache.reset();
            let dummy_tokens: Vec<u32> = vec![0u32; 8];
            chunked_prefill(&cfg, &weights, &dummy_tokens, &mut dummy_cache, prefill_chunk);
        }

        Ok(Self {
            cfg,
            weights,
            prefill_chunk: prefill_chunk.max(1),
            binding_summary,
            states: Mutex::new(HashMap::new()),
            _stream: stream,
        })
    }
}

impl ExecutionRunner for MlxNativeRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let step_id = input.execution_batch.step_id;
        let mut request_updates = Vec::new();
        let logits_handles = Vec::new();
        let logits_outputs = Vec::new();

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let update = self.run_item(item, ctx);
            request_updates.push(update);
        }

        let tokens_written: u32 = input.execution_batch.items.iter()
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
            route_metadata: RouteMetadata::empty(),
            execution_status: ExecutionStatus::Success,
        }
    }

    fn native_model_binding_summary(&self) -> Option<NativeModelBindingSummary> {
        Some(self.binding_summary)
    }
}

impl MlxNativeRunner {
    fn run_item(
        &self,
        item: &ax_engine_core::ExecutionItem,
        ctx: Option<&RunnerRequestContext>,
    ) -> RequestExecutionUpdate {
        let token_ids = &item.input_token_slice;
        if token_ids.is_empty() {
            return RequestExecutionUpdate {
                request_id: item.request_id,
                tokens_executed: 0,
                output_token: None,
                stop_reason: None,
                error: Some("empty token slice".into()),
            };
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);

        let mut states = self.states.lock().unwrap();
        let state = states
            .entry(item.request_id)
            .or_insert_with(|| RequestState::new(self.cfg.layer_count));

        let sampled_token = match item.mode {
            ExecutionMode::Prefill => {
                let tok = chunked_prefill(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    &mut state.cache,
                    self.prefill_chunk,
                );
                // Seed n-gram table with prompt tokens for better early speculation.
                state.ngram.feed(token_ids);
                // Reset bonus state for this new generation.
                state.bonus_queue.clear();
                state.next_model_last_token = None;
                tok
            }
            ExecutionMode::Decode => {
                self.decode_one(state, token_ids)
            }
        };

        let stop_reason = if generated_len + 1 >= max_output {
            Some(StopReason::MaxOutputTokens)
        } else {
            None
        };

        if stop_reason.is_some() {
            states.remove(&item.request_id);
        }

        RequestExecutionUpdate {
            request_id: item.request_id,
            tokens_executed: item.scheduled_token_count,
            output_token: Some(sampled_token),
            stop_reason,
            error: None,
        }
    }

    /// Produce one output token for a decode step.
    ///
    /// Pops from the bonus queue when pre-verified tokens are available.
    /// Otherwise runs a speculative or single-token decode pass.
    fn decode_one(&self, state: &mut RequestState, input_tokens: &[u32]) -> u32 {
        // Serve pre-verified bonus tokens without re-running the model.
        if let Some(tok) = state.bonus_queue.pop_front() {
            return tok;
        }

        let last_token = state.next_model_last_token
            .or_else(|| input_tokens.last().copied())
            .unwrap_or(0);

        let result = self.run_model_decode(state, last_token);

        // result[0] is the output for this step.
        // result[1..last] are bonus tokens (KVs already in cache).
        // result[last] is the starting point for the next model run (KV not yet in cache).
        let output = result[0];

        // Queue bonus tokens (intermediate accepted drafts).
        for &t in &result[1..result.len().saturating_sub(1)] {
            state.bonus_queue.push_back(t);
        }

        // The last element drives the next model run.
        state.next_model_last_token = result.last().copied();

        output
    }

    /// Run one model decode step (speculative or single), updating EMA gating.
    fn run_model_decode(&self, state: &mut RequestState, last_token: u32) -> Vec<u32> {
        // Speculation disabled: count down and use single decode.
        if state.spec_disabled_steps > 0 {
            state.spec_disabled_steps -= 1;
            return single_decode(&self.cfg, &self.weights, &mut state.cache, &mut state.ngram, last_token);
        }

        let draft = state.ngram.predict(DEFAULT_DRAFT_LEN);
        if draft.is_empty() {
            return single_decode(&self.cfg, &self.weights, &mut state.cache, &mut state.ngram, last_token);
        }

        let draft_len = draft.len();
        let result = speculative_decode_step(
            &self.cfg,
            &self.weights,
            &mut state.cache,
            &mut state.ngram,
            last_token,
            draft_len,
        );

        // Update EMA: accept_count = result.len() - 1 (excluding next last_token).
        let accept_count = result.len().saturating_sub(1);
        let accept_rate = accept_count as f32 / draft_len as f32;
        state.spec_ema = state.spec_ema * (1.0 - EMA_ALPHA) + accept_rate * EMA_ALPHA;
        if state.spec_ema < SPEC_ACCEPT_THRESHOLD {
            state.spec_disabled_steps = SPEC_RETRY_INTERVAL;
        }

        result
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxRunnerError {
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}
