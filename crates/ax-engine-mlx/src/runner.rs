use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use ax_engine_core::{
    ExecutionRunner, ExecutionStatus, KvWriteSummary, NativeModelArtifacts,
    NativeModelBindingSummary, RequestExecutionUpdate, RequestId, RequestLogitsOutput,
    RouteMetadata, RunnerInput, RunnerOutput, StopReason,
};
use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;

use crate::generate::{chunked_prefill, decode_step};
use crate::kv_cache::MlxKVCache;
use crate::model::ModelConfig;
use crate::weights::{ModelWeights, load_weights};

/// ExecutionRunner backed by the MLX native inference path.
pub struct MlxNativeRunner {
    cfg: ModelConfig,
    weights: Arc<ModelWeights>,
    prefill_chunk: usize,
    binding_summary: NativeModelBindingSummary,
    /// KV cache persisted across prefill → decode steps, keyed by request_id.
    caches: Mutex<HashMap<RequestId, MlxKVCache>>,
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

        // JIT warm-up: one forward pass to trigger Metal shader compilation before first request.
        {
            let mut dummy_cache = MlxKVCache::new(cfg.layer_count);
            decode_step(&cfg, &weights, 0, &mut dummy_cache);
        }

        Ok(Self {
            cfg,
            weights,
            prefill_chunk: prefill_chunk.max(1),
            binding_summary,
            caches: Mutex::new(HashMap::new()),
        })
    }
}

impl ExecutionRunner for MlxNativeRunner {
    fn run(&self, input: RunnerInput) -> RunnerOutput {
        let step_id = input.execution_batch.step_id;
        let mut request_updates = Vec::new();
        let mut logits_outputs = Vec::new();
        let mut logits_handles = Vec::new();

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let (update, logits) = self.run_item(item, ctx);

            if let Some(lo) = logits {
                logits_handles.push(item.request_id);
                logits_outputs.push(lo);
            }
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
    ) -> (RequestExecutionUpdate, Option<RequestLogitsOutput>) {
        let token_ids = &item.input_token_slice;
        if token_ids.is_empty() {
            return (
                RequestExecutionUpdate {
                    request_id: item.request_id,
                    tokens_executed: 0,
                    output_token: None,
                    stop_reason: None,
                    error: Some("empty token slice".into()),
                },
                None,
            );
        }

        let max_output = ctx.map(|c| c.max_output_tokens).unwrap_or(1);
        let generated_len = ctx.map(|c| c.generated_len).unwrap_or(0);

        // Take the cache out of the map (or create a fresh one for prefill).
        let mut cache = {
            let mut caches = self.caches.lock().unwrap();
            caches.remove(&item.request_id)
                .unwrap_or_else(|| MlxKVCache::new(self.cfg.layer_count))
        };

        let sampled_token = match item.mode {
            ExecutionMode::Prefill => {
                chunked_prefill(
                    &self.cfg,
                    &self.weights,
                    token_ids,
                    &mut cache,
                    self.prefill_chunk,
                )
            }
            ExecutionMode::Decode => {
                let last_token = *token_ids.last().unwrap_or(&0);
                decode_step(&self.cfg, &self.weights, last_token, &mut cache)
            }
        };

        let stop_reason = if generated_len + 1 >= max_output {
            Some(StopReason::MaxOutputTokens)
        } else {
            None
        };

        // Persist cache for the next decode step unless generation is finished.
        if stop_reason.is_none() {
            let mut caches = self.caches.lock().unwrap();
            caches.insert(item.request_id, cache);
        }

        let update = RequestExecutionUpdate {
            request_id: item.request_id,
            tokens_executed: item.scheduled_token_count,
            output_token: Some(sampled_token),
            stop_reason,
            error: None,
        };

        (update, None)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MlxRunnerError {
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}
