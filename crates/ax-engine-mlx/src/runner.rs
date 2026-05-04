use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use mlx_sys::{MlxStream, enable_compile, max_recommended_working_set_size, set_wired_limit};

use ax_engine_core::runner::RunnerRequestContext;
use ax_engine_core::scheduler::ExecutionMode;
use ax_engine_core::{
    ExecutionRunner, ExecutionStatus, KvWriteSummary, NativeModelArtifacts,
    NativeModelBindingSummary, NativeModelManifest, NativeTensorRole, RequestExecutionUpdate,
    RequestId, RouteMetadata, RunnerInput, RunnerOutput, StopReason,
};

use crate::generate::{chunked_prefill, decode_step};
use crate::kv_cache::MlxKVCache;
use crate::model::ModelConfig;
use crate::sampling::Xorshift64;
use crate::speculative::{DEFAULT_DRAFT_LEN, NgramTable, single_decode, speculative_decode_step};
use crate::weights::{ModelWeights, load_weights};

const EMA_ALPHA: f32 = 0.1;
const SPEC_ACCEPT_THRESHOLD: f32 = 0.5;
const SPEC_RETRY_INTERVAL: u32 = 8;

/// Per-request mutable state persisted across prefill → decode steps.
struct RequestState {
    cache: MlxKVCache,
    ngram: NgramTable,
    /// Per-request PRNG for temperature sampling.  Seeded from request_id so
    /// deterministic seeds produce reproducible outputs.
    rng: Xorshift64,
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
    fn new(num_layers: usize, request_id: RequestId) -> Self {
        Self {
            cache: MlxKVCache::new(num_layers),
            ngram: NgramTable::new(),
            rng: Xorshift64::new(request_id.0),
            spec_ema: 1.0,
            spec_disabled_steps: 0,
            bonus_queue: VecDeque::new(),
            next_model_last_token: None,
        }
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

        for item in &input.execution_batch.items {
            let ctx = input
                .request_contexts
                .iter()
                .find(|c| c.request_id == item.request_id);

            let update = self.run_item(item, ctx);
            request_updates.push(update);
        }

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
            route_metadata: RouteMetadata::empty(),
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
                // Seed n-gram table with prompt tokens for better early speculation.
                state.ngram.feed(token_ids);
                // Reset bonus state for this new generation.
                state.bonus_queue.clear();
                state.next_model_last_token = None;
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
        if stop_reason.is_none() {
            let mut states = self.states.lock().unwrap();
            states.insert(item.request_id, state);
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
    fn decode_one(&self, state: &mut RequestState, input_tokens: &[u32], temperature: f32) -> u32 {
        // Serve pre-verified bonus tokens without re-running the model.
        if let Some(tok) = state.bonus_queue.pop_front() {
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

    /// Run one model decode step (speculative or single), updating EMA gating.
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

        let draft = state.ngram.predict(DEFAULT_DRAFT_LEN);
        if draft.is_empty() {
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
            draft_len,
            temperature,
            &mut state.rng,
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
    #[error("MLX model feature is not supported: {0}")]
    UnsupportedFeature(String),
    #[error("weight loading failed: {0}")]
    Weights(#[from] crate::weights::WeightLoadError),
}

fn validate_mlx_supported_manifest(artifacts: &NativeModelArtifacts) -> Result<(), MlxRunnerError> {
    let manifest = artifacts.manifest();
    if manifest.linear_attention.is_enabled()
        || artifacts.tensor_specs().iter().any(|tensor| {
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
    {
        return Err(MlxRunnerError::UnsupportedFeature(
            "linear_attention layers require a dedicated MLX linear-attention implementation"
                .to_string(),
        ));
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
    use std::path::PathBuf;

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

    #[test]
    fn mlx_manifest_validation_rejects_linear_attention() {
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

        assert!(error.to_string().contains("linear_attention"));
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
