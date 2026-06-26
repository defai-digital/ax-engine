use super::*;

pub(crate) fn validate_llama_cpp_benchmark_runtime(
    manifest: &BenchmarkManifest,
    runtime: &RuntimeConfig,
) -> Result<(), CliError> {
    match runtime.backend_adapter.as_ref() {
        Some(adapter) if adapter.supports_stepwise_benchmark() => Ok(()),
        Some(adapter) if adapter.supports_blocking_benchmark() => {
            if manifest.class != ManifestClass::Scenario {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters currently support scenario manifests only"
                        .to_string(),
                ));
            }
            let shape = manifest.shape.as_ref().ok_or_else(|| {
                CliError::Contract("scenario manifest must contain a shape object".to_string())
            })?;
            if shape.concurrency != 1 {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters currently require shape.concurrency=1"
                        .to_string(),
                ));
            }
            if manifest.checks.require_prefix_reuse {
                return Err(CliError::Contract(
                    "blocking llama.cpp benchmark adapters do not support require_prefix_reuse"
                        .to_string(),
                ));
            }
            Ok(())
        }
        Some(BackendAdapterManifest::LlamaCppCli { .. }) => Err(CliError::Contract(
            "ax-engine-bench llama.cpp execution does not support CLI adapters; use a server-backed llama.cpp adapter instead".to_string(),
        )),
        Some(_) => Err(CliError::Contract(
            "llama.cpp benchmark execution requires a supported backend adapter".to_string(),
        )),
        None => Err(CliError::Contract(
            "llama.cpp benchmark execution requires runtime.backend_adapter".to_string(),
        )),
    }
}

pub(crate) fn request_snapshot_for_bench(
    engine: &EngineCore,
    request_id: RequestId,
) -> Result<RequestSnapshot, CliError> {
    engine
        .request_manager()
        .snapshot(request_id)
        .ok_or_else(|| CliError::Runtime(format!("missing request snapshot {:?}", request_id)))
}

pub(crate) fn has_live_requests(
    engine: &EngineCore,
    request_ids: &[RequestId],
) -> Result<bool, CliError> {
    for request_id in request_ids {
        let snapshot = request_snapshot_for_bench(engine, *request_id)?;
        if !snapshot.state.is_terminal() {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn llama_cpp_reports_for_session(
    session: &EngineSession,
    request_ids: &[RequestId],
) -> Result<BTreeMap<RequestId, SessionRequestReport>, CliError> {
    let mut reports = BTreeMap::new();

    for request_id in request_ids {
        let report = session.request_report(request_id.0).ok_or_else(|| {
            CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
        })?;
        reports.insert(*request_id, report);
    }

    Ok(reports)
}

pub(crate) fn llama_cpp_reports_changed(
    before: &BTreeMap<RequestId, SessionRequestReport>,
    after: &BTreeMap<RequestId, SessionRequestReport>,
) -> bool {
    before != after
}

pub(crate) fn llama_cpp_session_has_live_requests(
    session: &EngineSession,
    request_ids: &[RequestId],
) -> Result<bool, CliError> {
    for request_id in request_ids {
        let report = session.request_report(request_id.0).ok_or_else(|| {
            CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
        })?;
        if !llama_cpp_request_is_terminal(report.state) {
            return Ok(true);
        }
    }

    Ok(false)
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ManifestClass {
    Scenario,
    Replay,
}

impl ManifestClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Scenario => "scenario",
            Self::Replay => "replay",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchmarkManifest {
    pub(crate) schema_version: String,
    pub(crate) id: String,
    pub(crate) class: ManifestClass,
    pub(crate) scenario: String,
    pub(crate) model: ModelManifest,
    pub(crate) runtime: RuntimeManifest,
    pub(crate) sampling: SamplingManifest,
    #[serde(default)]
    pub(crate) shape: Option<ScenarioShape>,
    #[serde(default)]
    pub(crate) events: Vec<ReplayEventManifest>,
    #[serde(default)]
    pub(crate) source: Option<ManifestSource>,
    #[serde(default)]
    pub(crate) checks: ManifestChecks,
    #[serde(default)]
    pub(crate) notes: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MatrixClass {
    ScenarioMatrix,
}

impl MatrixClass {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::ScenarioMatrix => "scenario_matrix",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchmarkMatrixManifest {
    pub(crate) schema_version: String,
    pub(crate) id: String,
    pub(crate) class: MatrixClass,
    pub(crate) members: Vec<BenchmarkMatrixMember>,
    #[serde(default)]
    pub(crate) notes: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct BenchmarkMatrixMember {
    pub(crate) manifest: String,
    #[serde(default)]
    pub(crate) label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ModelManifest {
    pub(crate) family: String,
    pub(crate) revision: String,
    pub(crate) quant: String,
    pub(crate) tokenizer_revision: String,
    pub(crate) chat_template_revision: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct RuntimeManifest {
    pub(crate) selected_backend: SelectedBackend,
    pub(crate) support_tier: SupportTier,
    pub(crate) resolution_policy: ResolutionPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) backend_adapter: Option<BackendAdapterManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) llama_cpp_preset: Option<LlamaCppPresetManifest>,
    #[serde(default = "default_true")]
    pub(crate) deterministic: bool,
    #[serde(default)]
    pub(crate) max_batch_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) kv_total_blocks: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mlx_model_artifacts_dir: Option<PathBuf>,
    #[serde(default)]
    pub(crate) flags: RuntimeFlags,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) struct LlamaCppPresetManifest {
    #[serde(default = "default_llama_cpp_preset_name")]
    pub(crate) name: String,
    #[serde(default = "default_llama_cpp_parallel_slots")]
    pub(crate) parallel_slots: u32,
    #[serde(default)]
    pub(crate) continuous_batching: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) logical_batch_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) physical_batch_size: Option<u32>,
    #[serde(default)]
    pub(crate) cache_prompt: bool,
    #[serde(default)]
    pub(crate) slot_save_path: Option<String>,
    #[serde(default)]
    pub(crate) slot_restore_path: Option<String>,
    #[serde(default = "default_llama_cpp_speculative_decode_mode")]
    pub(crate) speculative_decode_mode: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) metrics_endpoint: Option<String>,
}

impl Default for LlamaCppPresetManifest {
    fn default() -> Self {
        Self {
            name: default_llama_cpp_preset_name(),
            parallel_slots: default_llama_cpp_parallel_slots(),
            continuous_batching: true,
            logical_batch_size: Some(2048),
            physical_batch_size: Some(512),
            cache_prompt: true,
            slot_save_path: None,
            slot_restore_path: None,
            speculative_decode_mode: default_llama_cpp_speculative_decode_mode(),
            metrics_endpoint: Some("server:/metrics".to_string()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum BackendAdapterManifest {
    LlamaCppCli {
        cli_path: PathBuf,
        model_path: PathBuf,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_args: Vec<String>,
    },
    LlamaCppServerCompletion {
        server_url: String,
    },
}

impl BackendAdapterManifest {
    pub(crate) fn selected_backend(&self) -> SelectedBackend {
        match self {
            Self::LlamaCppCli { .. } | Self::LlamaCppServerCompletion { .. } => {
                SelectedBackend::LlamaCpp
            }
        }
    }

    pub(crate) fn to_sdk_config(&self) -> LlamaCppConfig {
        match self {
            Self::LlamaCppCli {
                cli_path,
                model_path,
                extra_args,
            } => {
                let mut config = ax_engine_sdk::LlamaCppCliConfig::new(cli_path, model_path);
                config.extra_args = extra_args.clone();
                LlamaCppConfig::Cli(config)
            }
            Self::LlamaCppServerCompletion { server_url } => {
                LlamaCppConfig::server_completion(server_url.clone())
            }
        }
    }

    pub(crate) fn supports_stepwise_benchmark(&self) -> bool {
        matches!(self, Self::LlamaCppServerCompletion { .. })
    }

    pub(crate) fn supports_blocking_benchmark(&self) -> bool {
        matches!(self, Self::LlamaCppServerCompletion { .. })
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct RuntimeFlags {
    #[serde(default)]
    pub(crate) prefix_cache: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SamplingManifest {
    #[serde(default)]
    pub(crate) temperature: f32,
    #[serde(default = "default_top_p")]
    pub(crate) top_p: f32,
    #[serde(default)]
    pub(crate) top_k: u32,
    #[serde(default)]
    pub(crate) seed: u64,
    #[serde(default)]
    pub(crate) ignore_eos: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ScenarioShape {
    pub(crate) input_tokens_target: u32,
    pub(crate) output_tokens_target: u32,
    pub(crate) concurrency: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ManifestSource {
    #[serde(default)]
    pub(crate) dataset_id: Option<String>,
    #[serde(default)]
    pub(crate) prompt_set: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub(crate) struct ManifestChecks {
    #[serde(default)]
    pub(crate) expect_deterministic: bool,
    #[serde(default)]
    pub(crate) require_prefix_reuse: bool,
    #[serde(default)]
    pub(crate) require_no_allocator_churn_failure: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReplayEventKind {
    Submit,
    Cancel,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ReplayEventManifest {
    pub(crate) t_ms: u64,
    #[serde(rename = "type")]
    pub(crate) kind: ReplayEventKind,
    #[serde(default)]
    pub(crate) request_id: Option<String>,
    #[serde(default)]
    pub(crate) prompt_ref: Option<String>,
    #[serde(default)]
    pub(crate) output_tokens_target: Option<u32>,
    #[serde(default)]
    pub(crate) prefix_group: Option<String>,
    #[serde(default)]
    pub(crate) body_group: Option<String>,
}

pub(crate) const fn default_true() -> bool {
    true
}

pub(crate) const fn default_top_p() -> f32 {
    1.0
}

pub(crate) fn default_llama_cpp_preset_name() -> String {
    "safe_stepwise_server".to_string()
}

pub(crate) const fn default_llama_cpp_parallel_slots() -> u32 {
    1
}

pub(crate) fn default_llama_cpp_speculative_decode_mode() -> String {
    "disabled".to_string()
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeConfig {
    pub(crate) deterministic: bool,
    pub(crate) max_batch_tokens: u32,
    pub(crate) block_size_tokens: u32,
    pub(crate) kv_total_blocks: Option<u32>,
    pub(crate) flags: RuntimeFlags,
    pub(crate) llama_cpp_preset: Option<LlamaCppPresetManifest>,
    pub(crate) backend_policy: BackendPolicy,
    pub(crate) resolved_backend: ResolvedBackend,
    pub(crate) backend_adapter: Option<BackendAdapterManifest>,
    pub(crate) mlx_model_artifacts_dir: Option<PathBuf>,
    pub(crate) mlx_model_artifacts_source: Option<NativeModelArtifactsSource>,
}

impl RuntimeConfig {
    pub(crate) fn uses_mlx_runtime(&self) -> bool {
        self.resolved_backend.selected_backend.is_mlx()
    }

    pub(crate) fn uses_metal_bringup_runtime(&self) -> bool {
        false
    }

    pub(crate) fn mlx_runtime_report(&self) -> Option<NativeRuntimeReport> {
        if !self.uses_mlx_runtime() {
            return None;
        }

        None
    }

    pub(crate) fn native_model_report(&self) -> Option<NativeModelReport> {
        if !self.uses_mlx_runtime() {
            return None;
        }

        let model_dir = self.mlx_model_artifacts_dir.as_ref()?;
        if model_dir.is_file() {
            return None;
        }
        let summary = NativeModelArtifacts::from_dir(model_dir).ok()?.summary();
        let source = self
            .mlx_model_artifacts_source
            .unwrap_or(NativeModelArtifactsSource::ExplicitConfig);
        let binding = self.resolved_native_model_binding_summary();

        Some(NativeModelReport::from_summary(source, summary, binding))
    }

    pub(crate) fn resolved_native_model_binding_summary(
        &self,
    ) -> Option<NativeModelBindingSummary> {
        None
    }

    pub(crate) fn tool_mode(&self) -> &'static str {
        if self.uses_metal_bringup_runtime() {
            "engine_bringup_runtime"
        } else if self.uses_mlx_runtime() {
            "mlx_runtime"
        } else if self
            .backend_adapter
            .as_ref()
            .is_some_and(BackendAdapterManifest::supports_stepwise_benchmark)
        {
            "llama_cpp_stepwise_runtime"
        } else {
            "llama_cpp_blocking_runtime"
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SyntheticRequestSpec {
    pub(crate) external_id: String,
    pub(crate) request_id: RequestId,
    pub(crate) arrival_sequence: SequenceNo,
    pub(crate) model_family: String,
    pub(crate) prompt_token_target: u32,
    pub(crate) input_tokens: Vec<u32>,
    pub(crate) input_text: Option<String>,
    pub(crate) max_output_tokens: u32,
    pub(crate) sampling_params: SamplingParams,
    pub(crate) metadata: Option<String>,
}

impl SyntheticRequestSpec {
    pub(crate) fn into_submission(self) -> RequestSubmission {
        RequestSubmission {
            request_id: self.request_id,
            model_id: ax_engine_core::ModelId(self.model_family),
            input_tokens: self.input_tokens,
            multimodal_inputs: Default::default(),
            sampling_params: self.sampling_params,
            max_output_tokens: self.max_output_tokens,
            arrival_sequence: self.arrival_sequence,
            metadata: self.metadata,
        }
    }

    pub(crate) fn with_input_text(mut self, input_text: String) -> Self {
        self.input_tokens.clear();
        self.input_text = Some(input_text);
        self
    }
}

#[derive(Clone, Debug)]
pub(crate) enum ReplayEvent {
    Submit {
        t_ms: u64,
        spec: SyntheticRequestSpec,
    },
    Cancel {
        t_ms: u64,
        request_id: RequestId,
    },
}

impl ReplayEvent {
    pub(crate) fn t_ms(&self) -> u64 {
        match self {
            Self::Submit { t_ms, .. } | Self::Cancel { t_ms, .. } => *t_ms,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GateStatus {
    pub(crate) passed: bool,
    pub(crate) reason: Option<String>,
}

impl GateStatus {
    pub(crate) fn pass() -> Self {
        Self {
            passed: true,
            reason: None,
        }
    }

    pub(crate) fn fail(reason: impl Into<String>) -> Self {
        Self {
            passed: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeResult {
    pub(crate) tool_mode: &'static str,
    pub(crate) runtime: RuntimeConfig,
    pub(crate) observation: RuntimeObservation,
    pub(crate) correctness: GateStatus,
    pub(crate) determinism: GateStatus,
}

impl RuntimeResult {
    pub(crate) fn status_label(&self) -> &'static str {
        if self.correctness.passed && self.determinism.passed {
            "ok"
        } else {
            "completed_with_failures"
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RuntimeObservation {
    pub(crate) step_count: u64,
    pub(crate) e2e_latency_ms: u64,
    pub(crate) ttft_ms: Option<u64>,
    pub(crate) prefill_tokens: u64,
    pub(crate) decode_tokens: u64,
    pub(crate) prefill_steps: u64,
    pub(crate) decode_steps: u64,
    pub(crate) total_scheduled_tokens: u64,
    pub(crate) total_selected_requests: u64,
    pub(crate) prefix_hits: u64,
    pub(crate) memory_blocked_steps: u64,
    pub(crate) memory_blocked_request_events: u64,
    pub(crate) total_cpu_time_us: u64,
    pub(crate) total_runner_time_us: u64,
    pub(crate) total_prefill_runner_time_us: u64,
    pub(crate) total_decode_runner_time_us: u64,
    pub(crate) kv_peak_blocks: u32,
    pub(crate) memory_peak_mb: f64,
    pub(crate) evictions: u64,
    pub(crate) preempted_requests: u64,
    pub(crate) preempted_tokens: u64,
    pub(crate) cleanup_count: usize,
    pub(crate) llama_cpp_processing_request_events: u64,
    pub(crate) llama_cpp_deferred_request_events: u64,
    pub(crate) token_accounting_sources: BTreeMap<String, u64>,
    pub(crate) route_metadata: RouteMetadata,
    pub(crate) step_trace: Vec<StepTraceEntry>,
    pub(crate) final_requests: Vec<FinalRequestState>,
    pub(crate) cancelled_requests: BTreeSet<RequestId>,
    pub(crate) digest: Value,
    pub(crate) execution_plans_seen: BTreeSet<String>,
    pub(crate) attention_routes_seen: BTreeSet<String>,
    pub(crate) kv_modes_seen: BTreeSet<String>,
    pub(crate) barrier_modes_seen: BTreeSet<String>,
    pub(crate) runtime_report: Option<RuntimeReport>,
}

pub(crate) const LEGACY_KV_CACHE_ESTIMATED_BYTES_PER_TOKEN: f64 = 16.0;

impl RuntimeObservation {
    pub(crate) fn record_token_accounting_source(&mut self, phase: &str, source: &str) {
        let key = format!("{phase}.{source}");
        let entry = self.token_accounting_sources.entry(key).or_insert(0);
        *entry = entry.saturating_add(1);
    }

    pub(crate) fn observe_step(
        &mut self,
        engine: &EngineCore,
        outcome: &ax_engine_core::EngineStepOutcome,
        current_time_ms: u64,
    ) {
        self.step_count += 1;
        self.total_selected_requests += outcome.schedule_plan.selected_requests.len() as u64;
        self.total_scheduled_tokens += u64::from(outcome.metrics.scheduled_tokens);
        self.prefix_hits += u64::from(outcome.metrics.prefix_hits);
        if !outcome.schedule_plan.memory_blocked_requests.is_empty() {
            self.memory_blocked_steps += 1;
            self.memory_blocked_request_events +=
                outcome.schedule_plan.memory_blocked_requests.len() as u64;
        }
        self.total_cpu_time_us += outcome.metrics.cpu_time_us;
        self.total_runner_time_us += outcome.metrics.runner_time_us;
        self.evictions = self
            .evictions
            .saturating_add(u64::from(outcome.metrics.evictions));
        self.preempted_requests = self
            .preempted_requests
            .saturating_add(u64::from(outcome.metrics.preempted_requests));
        self.preempted_tokens = self
            .preempted_tokens
            .saturating_add(u64::from(outcome.metrics.preempted_tokens));
        self.kv_peak_blocks = self
            .kv_peak_blocks
            .max(engine.kv_manager().used_block_count());
        self.cleanup_count += outcome.cleanup_results.len();

        if self.ttft_ms.is_none() && outcome.metrics.ttft_events > 0 {
            self.ttft_ms = Some(current_time_ms);
        }

        if let Some(batch) = &outcome.schedule_plan.execution_batch {
            let mut saw_prefill = false;
            let mut saw_decode = false;
            let mut step_prefill_tokens = 0u64;
            let mut step_decode_tokens = 0u64;
            for item in &batch.items {
                match item.mode {
                    ax_engine_core::ExecutionMode::Prefill => {
                        let scheduled_tokens = u64::from(item.scheduled_token_count);
                        self.prefill_tokens += scheduled_tokens;
                        step_prefill_tokens += scheduled_tokens;
                        saw_prefill = true;
                    }
                    ax_engine_core::ExecutionMode::Decode => {
                        let scheduled_tokens = u64::from(item.scheduled_token_count);
                        self.decode_tokens += scheduled_tokens;
                        step_decode_tokens += scheduled_tokens;
                        saw_decode = true;
                    }
                }
            }
            let step_phase_tokens = step_prefill_tokens.saturating_add(step_decode_tokens);
            if step_phase_tokens > 0 {
                self.total_prefill_runner_time_us = self
                    .total_prefill_runner_time_us
                    .saturating_add(proportional_time_us(
                        outcome.metrics.runner_time_us,
                        step_prefill_tokens,
                        step_phase_tokens,
                    ));
                self.total_decode_runner_time_us =
                    self.total_decode_runner_time_us
                        .saturating_add(proportional_time_us(
                            outcome.metrics.runner_time_us,
                            step_decode_tokens,
                            step_phase_tokens,
                        ));
            }
            if saw_prefill {
                self.prefill_steps += 1;
            }
            if saw_decode {
                self.decode_steps += 1;
            }
        }

        if let Some(runner_output) = &outcome.runner_output {
            self.merge_route_metadata(&runner_output.route_metadata);
        }

        self.step_trace
            .push(StepTraceEntry::capture(engine, outcome, current_time_ms));
    }

    pub(crate) fn observe_llama_cpp_session_step(
        &mut self,
        reports_before: &BTreeMap<RequestId, SessionRequestReport>,
        reports_after: &BTreeMap<RequestId, SessionRequestReport>,
        step: &EngineStepReport,
        current_time_ms: u64,
    ) {
        self.step_count += 1;
        self.total_selected_requests += u64::from(step.scheduled_requests);
        self.total_scheduled_tokens += u64::from(step.scheduled_tokens);
        self.prefix_hits += u64::from(step.prefix_hits);
        self.decode_tokens += u64::from(step.scheduled_tokens);
        if step.scheduled_tokens > 0 {
            self.decode_steps += 1;
        }
        self.total_cpu_time_us += step.cpu_time_us;
        self.total_runner_time_us += step.runner_time_us;
        self.evictions = self.evictions.saturating_add(u64::from(step.evictions));
        self.preempted_requests = self
            .preempted_requests
            .saturating_add(u64::from(step.preempted_requests));
        self.preempted_tokens = self
            .preempted_tokens
            .saturating_add(u64::from(step.preempted_tokens));
        self.kv_peak_blocks = self.kv_peak_blocks.max(step.kv_usage_blocks);

        if self.ttft_ms.is_none() && step.ttft_events > 0 {
            self.ttft_ms = Some(current_time_ms);
        }

        let mut selected_request_ids = Vec::new();
        let mut items = Vec::new();
        let mut step_route_metadata = RouteMetadata::empty();
        let mut saw_prefill_progress = false;
        let live_request_ids = reports_after
            .iter()
            .filter_map(|(request_id, report)| {
                (!llama_cpp_request_is_terminal(report.state)).then_some(*request_id)
            })
            .collect::<BTreeSet<_>>();

        for (request_id, report_after) in reports_after {
            let report_before = reports_before.get(request_id);
            let output_delta = report_before
                .map(|report| report_after.output_len.saturating_sub(report.output_len))
                .unwrap_or(report_after.output_len);
            let prompt_delta = report_before
                .map(|report| {
                    report_after
                        .processed_prompt_tokens
                        .saturating_sub(report.processed_prompt_tokens)
                })
                .unwrap_or(report_after.processed_prompt_tokens);
            let state_changed = report_before
                .map(|report| report.state != report_after.state)
                .unwrap_or(true);

            if output_delta == 0 && prompt_delta == 0 && !state_changed {
                continue;
            }

            selected_request_ids.push(*request_id);
            if prompt_delta > 0 {
                saw_prefill_progress = true;
            }
            if output_delta > 0 {
                items.push(StepTraceItem::capture_llama_cpp_decode(
                    *request_id,
                    report_after.output_len,
                    output_delta,
                ));
            }

            let route_metadata = route_metadata_from_generate_route(&report_after.route);
            self.merge_route_metadata(&route_metadata);
            merge_step_route_metadata(&mut step_route_metadata, &route_metadata);
        }

        if saw_prefill_progress {
            self.prefill_steps += 1;
        }
        let selected_request_set = selected_request_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let deferred_request_ids = live_request_ids
            .difference(&selected_request_set)
            .copied()
            .collect::<Vec<_>>();
        self.llama_cpp_processing_request_events = self
            .llama_cpp_processing_request_events
            .saturating_add(live_request_ids.len() as u64);
        self.llama_cpp_deferred_request_events = self
            .llama_cpp_deferred_request_events
            .saturating_add(deferred_request_ids.len() as u64);

        self.step_trace
            .push(StepTraceEntry::capture_llama_cpp_shared(
                selected_request_ids,
                deferred_request_ids,
                step,
                current_time_ms,
                step_route_metadata,
                items,
            ));
    }

    pub(crate) fn finalize(
        &mut self,
        engine: &EngineCore,
        request_ids: &[RequestId],
        external_ids: BTreeMap<RequestId, String>,
        current_time_ms: u64,
        block_size_tokens: u32,
    ) -> Result<(), CliError> {
        self.e2e_latency_ms = current_time_ms;
        self.memory_peak_mb = (f64::from(self.kv_peak_blocks)
            * f64::from(block_size_tokens)
            * LEGACY_KV_CACHE_ESTIMATED_BYTES_PER_TOKEN)
            / 1024.0;

        let mut final_requests = Vec::new();
        for request_id in request_ids {
            let snapshot = request_snapshot_for_bench(engine, *request_id)?;
            final_requests.push(FinalRequestState {
                external_id: external_ids
                    .get(request_id)
                    .cloned()
                    .unwrap_or_else(|| format!("req-{}", request_id.0)),
                request_id: *request_id,
                state: format!("{:?}", snapshot.state),
                processed_prompt_tokens: snapshot.processed_prompt_tokens,
                generated_tokens: snapshot.generated_tokens,
                cancel_requested: snapshot.cancel_requested,
                last_error: snapshot.last_error,
            });
        }
        final_requests.sort_by_key(|request| request.request_id);
        self.digest = json!({
            "step_count": self.step_count,
            "ttft_ms": Value::Null,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefix_hits": self.prefix_hits,
            "preempted_requests": self.preempted_requests,
            "preempted_tokens": self.preempted_tokens,
            "memory_blocked_steps": self.memory_blocked_steps,
            "memory_blocked_request_events": self.memory_blocked_request_events,
            "cleanup_count": self.cleanup_count,
            "llama_cpp_processing_request_events": self.llama_cpp_processing_request_events,
            "llama_cpp_deferred_request_events": self.llama_cpp_deferred_request_events,
            "route": serialize_route_metadata(&self.route_metadata),
            "requests": final_requests.iter().map(FinalRequestState::digest_json).collect::<Vec<_>>()
        });
        self.final_requests = final_requests;
        Ok(())
    }

    pub(crate) fn finalize_llama_cpp(
        &mut self,
        final_reports: Vec<(SyntheticRequestSpec, SessionRequestReport)>,
        elapsed_ms: u64,
    ) {
        self.e2e_latency_ms = elapsed_ms;
        self.memory_peak_mb = 0.0;
        self.prefill_tokens = final_reports
            .iter()
            .map(|(_, report)| u64::from(report.prompt_len))
            .sum();
        let delegated_cached_tokens_total = final_reports
            .iter()
            .map(|(_, report)| delegated_cached_tokens_from_generate_route(&report.route))
            .sum::<u32>();
        if delegated_cached_tokens_total > 0 {
            if self.route_metadata.prefix_cache_path.is_none() {
                self.route_metadata.prefix_cache_path = Some("delegated_prompt_cache".to_string());
            }
            upsert_route_decision(
                &mut self.route_metadata.crossover_decisions,
                "delegated_cached_tokens",
                delegated_cached_tokens_total,
            );
        }
        let mut final_requests = final_reports
            .into_iter()
            .map(|(spec, report)| FinalRequestState {
                external_id: spec.external_id,
                request_id: spec.request_id,
                state: llama_cpp_final_request_state_label(report.state).to_string(),
                processed_prompt_tokens: report.processed_prompt_tokens,
                generated_tokens: report.output_tokens.clone(),
                cancel_requested: report.cancel_requested,
                last_error: report.last_error.clone(),
            })
            .collect::<Vec<_>>();
        final_requests.sort_by_key(|request| request.request_id);
        self.final_requests = final_requests;
        self.digest = json!({
            "step_count": self.step_count,
            "ttft_ms": Value::Null,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "prefix_hits": self.prefix_hits,
            "preempted_requests": self.preempted_requests,
            "preempted_tokens": self.preempted_tokens,
            "memory_blocked_steps": self.memory_blocked_steps,
            "memory_blocked_request_events": self.memory_blocked_request_events,
            "cleanup_count": self.cleanup_count,
            "llama_cpp_processing_request_events": self.llama_cpp_processing_request_events,
            "llama_cpp_deferred_request_events": self.llama_cpp_deferred_request_events,
            "route": serialize_route_metadata(&self.route_metadata),
            "requests": self
                .final_requests
                .iter()
                .map(FinalRequestState::digest_json)
                .collect::<Vec<_>>()
        });
    }

    pub(crate) fn prefix_hit_rate(&self) -> f64 {
        if self.total_selected_requests == 0 {
            0.0
        } else {
            (self.prefix_hits as f64 / self.total_selected_requests as f64) * 100.0
        }
    }

    /// Prefill throughput from runner execution time when available.
    /// Synthetic core paths can fall back to step counts; blocking delegated
    /// adapters leave this at 0 because they only expose request round-trip time.
    pub(crate) fn prefill_tok_s(&self) -> f64 {
        if self.prefill_tokens == 0 {
            0.0
        } else if self.total_prefill_runner_time_us > 0 {
            tokens_per_second_from_micros(self.prefill_tokens, self.total_prefill_runner_time_us)
        } else if self.prefill_steps == 0 {
            0.0
        } else {
            (self.prefill_tokens as f64 * 1000.0) / self.prefill_steps as f64
        }
    }

    /// Decode throughput from runner execution time when available.
    /// Returns 0.0 when the runtime did not report decode runner time.
    pub(crate) fn decode_tok_s(&self) -> f64 {
        if self.decode_tokens == 0 {
            0.0
        } else if self.total_decode_runner_time_us > 0 {
            tokens_per_second_from_micros(self.decode_tokens, self.total_decode_runner_time_us)
        } else {
            0.0
        }
    }

    pub(crate) fn cpu_time_per_token_us(&self) -> f64 {
        if self.total_scheduled_tokens == 0 {
            0.0
        } else {
            self.total_cpu_time_us as f64 / self.total_scheduled_tokens as f64
        }
    }

    pub(crate) fn runner_time_per_token_us(&self) -> f64 {
        if self.total_scheduled_tokens == 0 {
            0.0
        } else {
            self.total_runner_time_us as f64 / self.total_scheduled_tokens as f64
        }
    }

    pub(crate) fn runner_time_share_pct(&self) -> f64 {
        if self.total_cpu_time_us == 0 {
            0.0
        } else {
            (self.total_runner_time_us as f64 / self.total_cpu_time_us as f64) * 100.0
        }
    }

    pub(crate) fn scheduled_tokens_per_step(&self) -> f64 {
        if self.step_count == 0 {
            0.0
        } else {
            self.total_scheduled_tokens as f64 / self.step_count as f64
        }
    }

    pub(crate) fn request_state(&self, request_id: RequestId, expected_state: &str) -> bool {
        self.final_requests
            .iter()
            .find(|request| request.request_id == request_id)
            .map(|request| request.state == expected_state)
            .unwrap_or(false)
    }

    pub(crate) fn replay_status(&self) -> &'static str {
        if self.cancelled_requests.is_empty() {
            "not_applicable"
        } else if self
            .cancelled_requests
            .iter()
            .all(|request_id| self.request_state(*request_id, "Cancelled"))
        {
            "pass"
        } else {
            "fail"
        }
    }

    pub(crate) fn churn_status(&self) -> &'static str {
        if self
            .final_requests
            .iter()
            .any(|request| request.state == "Failed")
        {
            "fail"
        } else {
            "pass"
        }
    }

    pub(crate) fn merge_route_metadata(&mut self, route_metadata: &RouteMetadata) {
        record_route_variant(
            &mut self.execution_plans_seen,
            route_metadata.execution_plan.as_deref(),
        );
        record_route_variant(
            &mut self.attention_routes_seen,
            route_metadata.attention_route.as_deref(),
        );
        record_route_variant(&mut self.kv_modes_seen, route_metadata.kv_mode.as_deref());
        record_route_variant(
            &mut self.barrier_modes_seen,
            route_metadata.barrier_mode.as_deref(),
        );

        self.route_metadata.execution_plan =
            aggregate_route_variant(&self.execution_plans_seen, "mixed_step_plans");
        self.route_metadata.attention_route =
            aggregate_route_variant(&self.attention_routes_seen, "mixed_attention_routes");
        self.route_metadata.kv_mode =
            aggregate_route_variant(&self.kv_modes_seen, "mixed_kv_modes");
        self.route_metadata.barrier_mode =
            aggregate_route_variant(&self.barrier_modes_seen, "mixed_barrier_modes");

        let mut cumulative_decisions = self
            .route_metadata
            .crossover_decisions
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();
        for (key, value) in &route_metadata.crossover_decisions {
            let entry = cumulative_decisions.entry(key.clone()).or_insert(0);
            *entry = entry.saturating_add(*value);
        }
        cumulative_decisions.insert(
            "execution_plan_variants".into(),
            self.execution_plans_seen.len() as u32,
        );
        cumulative_decisions.insert(
            "attention_route_variants".into(),
            self.attention_routes_seen.len() as u32,
        );
        cumulative_decisions.insert("kv_mode_variants".into(), self.kv_modes_seen.len() as u32);
        cumulative_decisions.insert(
            "barrier_mode_variants".into(),
            self.barrier_modes_seen.len() as u32,
        );
        self.route_metadata.crossover_decisions = cumulative_decisions.into_iter().collect();

        let live_share_hits = self
            .route_metadata
            .crossover_decisions
            .iter()
            .find(|(key, _)| key == "live_share_hits")
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let retained_cache_hits = self
            .route_metadata
            .crossover_decisions
            .iter()
            .find(|(key, _)| key == "retained_cache_hits")
            .map(|(_, value)| *value)
            .unwrap_or(0);

        self.route_metadata.prefix_cache_path = Some(
            match (live_share_hits > 0, retained_cache_hits > 0) {
                (true, true) => "mixed_live_and_retained",
                (true, false) => "live_request_share",
                (false, true) => "retained_prompt_prefix_cache",
                (false, false) => route_metadata
                    .prefix_cache_path
                    .as_deref()
                    .unwrap_or("metadata_lookup"),
            }
            .to_string(),
        );
    }
}

impl Default for RuntimeObservation {
    fn default() -> Self {
        Self {
            step_count: 0,
            e2e_latency_ms: 0,
            ttft_ms: None,
            prefill_tokens: 0,
            decode_tokens: 0,
            prefill_steps: 0,
            decode_steps: 0,
            total_scheduled_tokens: 0,
            total_selected_requests: 0,
            prefix_hits: 0,
            memory_blocked_steps: 0,
            memory_blocked_request_events: 0,
            total_cpu_time_us: 0,
            total_runner_time_us: 0,
            total_prefill_runner_time_us: 0,
            total_decode_runner_time_us: 0,
            kv_peak_blocks: 0,
            memory_peak_mb: 0.0,
            evictions: 0,
            preempted_requests: 0,
            preempted_tokens: 0,
            cleanup_count: 0,
            llama_cpp_processing_request_events: 0,
            llama_cpp_deferred_request_events: 0,
            token_accounting_sources: BTreeMap::new(),
            route_metadata: RouteMetadata::empty(),
            step_trace: Vec::new(),
            final_requests: Vec::new(),
            cancelled_requests: BTreeSet::new(),
            digest: Value::Null,
            execution_plans_seen: BTreeSet::new(),
            attention_routes_seen: BTreeSet::new(),
            kv_modes_seen: BTreeSet::new(),
            barrier_modes_seen: BTreeSet::new(),
            runtime_report: None,
        }
    }
}

pub(crate) fn record_route_variant(seen: &mut BTreeSet<String>, value: Option<&str>) {
    if let Some(value) = value {
        seen.insert(value.to_string());
    }
}

pub(crate) fn aggregate_route_variant(
    seen: &BTreeSet<String>,
    mixed_label: &str,
) -> Option<String> {
    match seen.len() {
        0 => None,
        1 => seen.iter().next().cloned(),
        _ => Some(mixed_label.to_string()),
    }
}

pub(crate) fn merge_route_variant(
    current: &Option<String>,
    next: &Option<String>,
    mixed_label: &str,
) -> Option<String> {
    match (current.as_deref(), next.as_deref()) {
        (None, None) => None,
        (Some(current), None) | (None, Some(current)) => Some(current.to_string()),
        (Some(current), Some(next)) if current == next => Some(current.to_string()),
        _ => Some(mixed_label.to_string()),
    }
}

pub(crate) fn merge_step_route_metadata(target: &mut RouteMetadata, next: &RouteMetadata) {
    target.execution_plan = merge_route_variant(
        &target.execution_plan,
        &next.execution_plan,
        "mixed_step_plans",
    );
    target.attention_route = merge_route_variant(
        &target.attention_route,
        &next.attention_route,
        "mixed_attention_routes",
    );
    target.kv_mode = merge_route_variant(&target.kv_mode, &next.kv_mode, "mixed_kv_modes");
    target.prefix_cache_path = merge_route_variant(
        &target.prefix_cache_path,
        &next.prefix_cache_path,
        "mixed_prefix_cache_paths",
    );
    target.barrier_mode = merge_route_variant(
        &target.barrier_mode,
        &next.barrier_mode,
        "mixed_barrier_modes",
    );

    let mut decisions = target
        .crossover_decisions
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    for (key, value) in &next.crossover_decisions {
        let entry = decisions.entry(key.clone()).or_insert(0);
        *entry = entry.saturating_add(*value);
    }
    target.crossover_decisions = decisions.into_iter().collect();
}

pub(crate) fn delegated_cached_tokens_from_generate_route(route: &GenerateRouteReport) -> u32 {
    route
        .crossover_decisions
        .iter()
        .find(|(key, _)| key.as_str() == "delegated_cached_tokens")
        .map(|(_, value)| *value)
        .unwrap_or(0)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct MetalDispatchValidationStepTrace {
    pub(crate) expected_key_cache_checksum: u64,
    pub(crate) expected_attention_output_checksum: u64,
    pub(crate) expected_gather_output_checksum: u64,
    pub(crate) expected_copy_output_checksum: u64,
    pub(crate) attention_max_abs_diff_microunits: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct MetalDispatchNumericStepTrace {
    pub(crate) key_cache_checksum: u64,
    pub(crate) attention_output_checksum: u64,
    pub(crate) gather_output_checksum: u64,
    pub(crate) copy_output_checksum: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) validation: Option<MetalDispatchValidationStepTrace>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct MetalDispatchKernelStepTrace {
    pub(crate) function_name: String,
    pub(crate) element_count: u32,
    pub(crate) threads_per_grid_width: u64,
    pub(crate) threads_per_threadgroup_width: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct MetalDispatchStepTrace {
    pub(crate) command_queue_label: String,
    pub(crate) command_buffer_label: String,
    pub(crate) command_buffer_status: String,
    pub(crate) runtime_device_name: String,
    pub(crate) runtime_required_pipeline_count: u32,
    pub(crate) runtime_max_thread_execution_width: u64,
    pub(crate) runtime_model_conditioned_inputs: bool,
    pub(crate) runtime_real_model_tensor_inputs: bool,
    pub(crate) runtime_complete_model_forward_supported: bool,
    pub(crate) runtime_model_bindings_prepared: bool,
    pub(crate) runtime_model_buffers_bound: bool,
    pub(crate) runtime_model_buffer_count: u32,
    pub(crate) runtime_model_buffer_bytes: u64,
    pub(crate) runtime_model_family: Option<String>,
    pub(crate) execution_direct_decode_token_count: u32,
    pub(crate) execution_direct_decode_checksum_lo: u32,
    pub(crate) execution_logits_output_count: u32,
    pub(crate) execution_remaining_logits_handle_count: u32,
    pub(crate) execution_model_bound_ffn_decode: bool,
    pub(crate) execution_real_model_forward_completed: bool,
    pub(crate) execution_prefix_native_dispatch_count: u32,
    pub(crate) execution_prefix_cpu_reference_dispatch_count: u32,
    pub(crate) execution_qkv_projection_token_count: u32,
    pub(crate) execution_layer_continuation_token_count: u32,
    pub(crate) execution_logits_projection_token_count: u32,
    pub(crate) execution_logits_vocab_scan_row_count: u32,
    pub(crate) execution_prefix_native_projection_row_count: u32,
    pub(crate) execution_prefix_cpu_projection_row_count: u32,
    pub(crate) execution_prefix_native_rms_norm_element_count: u32,
    pub(crate) execution_prefix_cpu_rms_norm_element_count: u32,
    pub(crate) execution_prefix_native_ffn_activation_element_count: u32,
    pub(crate) execution_prefix_cpu_ffn_activation_element_count: u32,
    pub(crate) execution_prefix_native_residual_add_element_count: u32,
    pub(crate) execution_prefix_cpu_residual_add_element_count: u32,
    pub(crate) execution_prefix_native_scale_element_count: u32,
    pub(crate) execution_prefix_cpu_scale_element_count: u32,
    pub(crate) execution_direct_decode_native_projection_row_count: u32,
    pub(crate) execution_direct_decode_cpu_projection_row_count: u32,
    pub(crate) execution_direct_decode_native_rms_norm_element_count: u32,
    pub(crate) execution_direct_decode_cpu_rms_norm_element_count: u32,
    pub(crate) execution_direct_decode_native_ffn_activation_element_count: u32,
    pub(crate) execution_direct_decode_cpu_ffn_activation_element_count: u32,
    pub(crate) execution_direct_decode_native_residual_add_element_count: u32,
    pub(crate) execution_direct_decode_cpu_residual_add_element_count: u32,
    pub(crate) execution_direct_decode_native_scale_element_count: u32,
    pub(crate) execution_direct_decode_cpu_scale_element_count: u32,
    pub(crate) execution_direct_decode_batched_logits_group_count: u32,
    pub(crate) execution_direct_decode_batched_logits_token_count: u32,
    pub(crate) execution_direct_decode_batched_group_fallback_count: u32,
    pub(crate) execution_direct_decode_batched_group_fallback_token_count: u32,
    pub(crate) binary_archive_state: String,
    pub(crate) binary_archive_attached_pipeline_count: u32,
    pub(crate) binary_archive_serialized: bool,
    pub(crate) arena_token_capacity: u32,
    pub(crate) arena_slot_capacity: u32,
    pub(crate) arena_attention_ref_capacity: u32,
    pub(crate) arena_gather_ref_capacity: u32,
    pub(crate) arena_gather_output_capacity: u32,
    pub(crate) arena_copy_pair_capacity: u32,
    pub(crate) arena_sequence_capacity: u32,
    pub(crate) arena_reused_existing: bool,
    pub(crate) arena_grew_existing: bool,
    pub(crate) kernels: Vec<MetalDispatchKernelStepTrace>,
    pub(crate) numeric: MetalDispatchNumericStepTrace,
}

impl MetalDispatchStepTrace {
    pub(crate) fn from_trace(trace: &MetalDispatchTrace) -> Self {
        Self {
            command_queue_label: trace.command_queue_label.clone(),
            command_buffer_label: trace.command_buffer_label.clone(),
            command_buffer_status: json_string_label(trace.command_buffer_status),
            runtime_device_name: trace.runtime.device_name.clone(),
            runtime_required_pipeline_count: trace.runtime.required_pipeline_count,
            runtime_max_thread_execution_width: trace.runtime.max_thread_execution_width,
            runtime_model_conditioned_inputs: trace.runtime.model_conditioned_inputs,
            runtime_real_model_tensor_inputs: trace.runtime.real_model_tensor_inputs,
            runtime_complete_model_forward_supported: trace
                .runtime
                .complete_model_forward_supported,
            runtime_model_bindings_prepared: trace.runtime.model_bindings_prepared,
            runtime_model_buffers_bound: trace.runtime.model_buffers_bound,
            runtime_model_buffer_count: trace.runtime.model_buffer_count,
            runtime_model_buffer_bytes: trace.runtime.model_buffer_bytes,
            runtime_model_family: trace
                .runtime
                .model
                .as_ref()
                .map(|model| model.model_family.clone()),
            execution_direct_decode_token_count: trace.execution.direct_decode_token_count,
            execution_direct_decode_checksum_lo: trace.execution.direct_decode_checksum_lo,
            execution_logits_output_count: trace.execution.logits_output_count,
            execution_remaining_logits_handle_count: trace.execution.remaining_logits_handle_count,
            execution_model_bound_ffn_decode: trace.execution.model_bound_ffn_decode,
            execution_real_model_forward_completed: trace.execution.real_model_forward_completed,
            execution_prefix_native_dispatch_count: trace.execution.prefix_native_dispatch_count,
            execution_prefix_cpu_reference_dispatch_count: trace
                .execution
                .prefix_cpu_reference_dispatch_count,
            execution_qkv_projection_token_count: trace.execution.qkv_projection_token_count,
            execution_layer_continuation_token_count: trace
                .execution
                .layer_continuation_token_count,
            execution_logits_projection_token_count: trace.execution.logits_projection_token_count,
            execution_logits_vocab_scan_row_count: trace.execution.logits_vocab_scan_row_count,
            execution_prefix_native_projection_row_count: trace
                .execution
                .prefix_native_projection_row_count,
            execution_prefix_cpu_projection_row_count: trace
                .execution
                .prefix_cpu_projection_row_count,
            execution_prefix_native_rms_norm_element_count: trace
                .execution
                .prefix_native_rms_norm_element_count,
            execution_prefix_cpu_rms_norm_element_count: trace
                .execution
                .prefix_cpu_rms_norm_element_count,
            execution_prefix_native_ffn_activation_element_count: trace
                .execution
                .prefix_native_ffn_activation_element_count,
            execution_prefix_cpu_ffn_activation_element_count: trace
                .execution
                .prefix_cpu_ffn_activation_element_count,
            execution_prefix_native_residual_add_element_count: trace
                .execution
                .prefix_native_residual_add_element_count,
            execution_prefix_cpu_residual_add_element_count: trace
                .execution
                .prefix_cpu_residual_add_element_count,
            execution_prefix_native_scale_element_count: trace
                .execution
                .prefix_native_scale_element_count,
            execution_prefix_cpu_scale_element_count: trace
                .execution
                .prefix_cpu_scale_element_count,
            execution_direct_decode_native_projection_row_count: trace
                .execution
                .direct_decode_native_projection_row_count,
            execution_direct_decode_cpu_projection_row_count: trace
                .execution
                .direct_decode_cpu_projection_row_count,
            execution_direct_decode_native_rms_norm_element_count: trace
                .execution
                .direct_decode_native_rms_norm_element_count,
            execution_direct_decode_cpu_rms_norm_element_count: trace
                .execution
                .direct_decode_cpu_rms_norm_element_count,
            execution_direct_decode_native_ffn_activation_element_count: trace
                .execution
                .direct_decode_native_ffn_activation_element_count,
            execution_direct_decode_cpu_ffn_activation_element_count: trace
                .execution
                .direct_decode_cpu_ffn_activation_element_count,
            execution_direct_decode_native_residual_add_element_count: trace
                .execution
                .direct_decode_native_residual_add_element_count,
            execution_direct_decode_cpu_residual_add_element_count: trace
                .execution
                .direct_decode_cpu_residual_add_element_count,
            execution_direct_decode_native_scale_element_count: trace
                .execution
                .direct_decode_native_scale_element_count,
            execution_direct_decode_cpu_scale_element_count: trace
                .execution
                .direct_decode_cpu_scale_element_count,
            execution_direct_decode_batched_logits_group_count: trace
                .execution
                .direct_decode_batched_logits_group_count,
            execution_direct_decode_batched_logits_token_count: trace
                .execution
                .direct_decode_batched_logits_token_count,
            execution_direct_decode_batched_group_fallback_count: trace
                .execution
                .direct_decode_batched_group_fallback_count,
            execution_direct_decode_batched_group_fallback_token_count: trace
                .execution
                .direct_decode_batched_group_fallback_token_count,
            binary_archive_state: json_string_label(trace.runtime.binary_archive.state),
            binary_archive_attached_pipeline_count: trace
                .runtime
                .binary_archive
                .attached_pipeline_count,
            binary_archive_serialized: trace.runtime.binary_archive.serialized,
            arena_token_capacity: trace.arena.token_capacity,
            arena_slot_capacity: trace.arena.slot_capacity,
            arena_attention_ref_capacity: trace.arena.attention_ref_capacity,
            arena_gather_ref_capacity: trace.arena.gather_ref_capacity,
            arena_gather_output_capacity: trace.arena.gather_output_capacity,
            arena_copy_pair_capacity: trace.arena.copy_pair_capacity,
            arena_sequence_capacity: trace.arena.sequence_capacity,
            arena_reused_existing: trace.arena.reused_existing,
            arena_grew_existing: trace.arena.grew_existing,
            kernels: trace
                .kernels
                .iter()
                .map(|kernel| MetalDispatchKernelStepTrace {
                    function_name: kernel.function_name.clone(),
                    element_count: kernel.element_count,
                    threads_per_grid_width: kernel.threads_per_grid.width,
                    threads_per_threadgroup_width: kernel.threads_per_threadgroup.width,
                })
                .collect(),
            numeric: MetalDispatchNumericStepTrace {
                key_cache_checksum: trace.numeric.key_cache_checksum,
                attention_output_checksum: trace.numeric.attention_output_checksum,
                gather_output_checksum: trace.numeric.gather_output_checksum,
                copy_output_checksum: trace.numeric.copy_output_checksum,
                validation: trace.numeric.validation.as_ref().map(|validation| {
                    MetalDispatchValidationStepTrace {
                        expected_key_cache_checksum: validation.expected_key_cache_checksum,
                        expected_attention_output_checksum: validation
                            .expected_attention_output_checksum,
                        expected_gather_output_checksum: validation.expected_gather_output_checksum,
                        expected_copy_output_checksum: validation.expected_copy_output_checksum,
                        attention_max_abs_diff_microunits: validation
                            .attention_max_abs_diff_microunits,
                    }
                }),
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepTraceEntry {
    pub(crate) t_ms: u64,
    pub(crate) step_id: Option<StepId>,
    pub(crate) admitted_request_ids: Vec<RequestId>,
    pub(crate) selected_request_ids: Vec<RequestId>,
    pub(crate) deferred_request_ids: Vec<RequestId>,
    pub(crate) memory_blocked_request_ids: Vec<RequestId>,
    pub(crate) cleanup_request_ids: Vec<RequestId>,
    pub(crate) scheduled_tokens: u32,
    pub(crate) prefix_hits: u32,
    pub(crate) cpu_time_us: u64,
    pub(crate) runner_time_us: u64,
    pub(crate) kv_usage_blocks: u32,
    pub(crate) evictions: u32,
    pub(crate) preempted_requests: u32,
    pub(crate) preempted_tokens: u32,
    pub(crate) runner_executed: bool,
    pub(crate) route_metadata: RouteMetadata,
    pub(crate) metal_dispatch: Option<MetalDispatchStepTrace>,
    pub(crate) items: Vec<StepTraceItem>,
}

impl StepTraceEntry {
    pub(crate) fn capture(
        engine: &EngineCore,
        outcome: &ax_engine_core::EngineStepOutcome,
        current_time_ms: u64,
    ) -> Self {
        let (route_metadata, items) = if let Some(batch) = &outcome.schedule_plan.execution_batch {
            (
                batch.route_metadata.clone(),
                batch.items.iter().map(StepTraceItem::capture).collect(),
            )
        } else {
            (RouteMetadata::empty(), Vec::new())
        };

        Self {
            t_ms: current_time_ms,
            step_id: outcome.metrics.step_id,
            admitted_request_ids: outcome.admitted_requests.clone(),
            selected_request_ids: outcome.schedule_plan.selected_requests.clone(),
            deferred_request_ids: outcome.schedule_plan.deferred_requests.clone(),
            memory_blocked_request_ids: outcome.schedule_plan.memory_blocked_requests.clone(),
            cleanup_request_ids: outcome
                .cleanup_results
                .iter()
                .map(|result| result.request_id)
                .collect(),
            scheduled_tokens: outcome.metrics.scheduled_tokens,
            prefix_hits: outcome.metrics.prefix_hits,
            cpu_time_us: outcome.metrics.cpu_time_us,
            runner_time_us: outcome.metrics.runner_time_us,
            kv_usage_blocks: engine.kv_manager().used_block_count(),
            evictions: outcome.metrics.evictions,
            preempted_requests: outcome.metrics.preempted_requests,
            preempted_tokens: outcome.metrics.preempted_tokens,
            runner_executed: outcome.runner_output.is_some(),
            route_metadata,
            metal_dispatch: outcome
                .runner_output
                .as_ref()
                .and_then(|_| engine.last_metal_dispatch())
                .map(|trace| MetalDispatchStepTrace::from_trace(&trace)),
            items,
        }
    }

    pub(crate) fn json(&self) -> Value {
        let mut value = json!({
            "t_ms": self.t_ms,
            "step_id": self.step_id.map(|step_id| step_id.0),
            "admitted_request_ids": self.admitted_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "selected_request_ids": self.selected_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "deferred_request_ids": self.deferred_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "memory_blocked_request_ids": self.memory_blocked_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "cleanup_request_ids": self.cleanup_request_ids.iter().map(|request_id| request_id.0).collect::<Vec<_>>(),
            "scheduled_tokens": self.scheduled_tokens,
            "prefix_hits": self.prefix_hits,
            "cpu_time_us": self.cpu_time_us,
            "runner_time_us": self.runner_time_us,
            "kv_usage_blocks": self.kv_usage_blocks,
            "evictions": self.evictions,
            "preempted_requests": self.preempted_requests,
            "preempted_tokens": self.preempted_tokens,
            "runner_executed": self.runner_executed,
            "route": serialize_route_metadata(&self.route_metadata),
            "items": self.items.iter().map(StepTraceItem::json).collect::<Vec<_>>()
        });
        if let Some(metal_dispatch) = self.metal_dispatch.as_ref() {
            value
                .as_object_mut()
                .expect("step trace json should serialize as object")
                .insert(
                    "metal_dispatch".to_string(),
                    serde_json::to_value(metal_dispatch)
                        .expect("metal dispatch summary should serialize"),
                );
        }
        value
    }

    pub(crate) fn capture_llama_cpp_shared(
        selected_request_ids: Vec<RequestId>,
        deferred_request_ids: Vec<RequestId>,
        step: &EngineStepReport,
        current_time_ms: u64,
        route_metadata: RouteMetadata,
        items: Vec<StepTraceItem>,
    ) -> Self {
        Self {
            t_ms: current_time_ms,
            step_id: step.step_id.map(StepId),
            admitted_request_ids: Vec::new(),
            selected_request_ids,
            deferred_request_ids,
            memory_blocked_request_ids: Vec::new(),
            cleanup_request_ids: Vec::new(),
            scheduled_tokens: step.scheduled_tokens,
            prefix_hits: step.prefix_hits,
            cpu_time_us: step.cpu_time_us,
            runner_time_us: step.runner_time_us,
            kv_usage_blocks: step.kv_usage_blocks,
            evictions: step.evictions,
            preempted_requests: step.preempted_requests,
            preempted_tokens: step.preempted_tokens,
            runner_executed: false,
            route_metadata,
            metal_dispatch: None,
            items,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct StepTraceItem {
    pub(crate) request_id: RequestId,
    pub(crate) mode: ax_engine_core::ExecutionMode,
    pub(crate) position_start: u32,
    pub(crate) position_end_exclusive: u32,
    pub(crate) scheduled_token_count: u32,
    pub(crate) input_token_count: usize,
    pub(crate) prefix_tokens_reused: u32,
    pub(crate) prefix_blocks_reused: u32,
}

impl StepTraceItem {
    pub(crate) fn capture(item: &ax_engine_core::ExecutionItem) -> Self {
        Self {
            request_id: item.request_id,
            mode: item.mode,
            position_start: item.position_range.start,
            position_end_exclusive: item.position_range.end_exclusive,
            scheduled_token_count: item.scheduled_token_count,
            input_token_count: item.input_token_slice.len(),
            prefix_tokens_reused: item.prefix_tokens_reused,
            prefix_blocks_reused: item.prefix_blocks_reused,
        }
    }

    pub(crate) fn json(&self) -> Value {
        json!({
            "request_id": self.request_id.0,
            "mode": match self.mode {
                ax_engine_core::ExecutionMode::Prefill => "Prefill",
                ax_engine_core::ExecutionMode::Decode => "Decode",
            },
            "position_start": self.position_start,
            "position_end_exclusive": self.position_end_exclusive,
            "scheduled_token_count": self.scheduled_token_count,
            "input_token_count": self.input_token_count,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "prefix_blocks_reused": self.prefix_blocks_reused
        })
    }

    pub(crate) fn capture_llama_cpp_decode(
        request_id: RequestId,
        output_len: u32,
        scheduled_tokens: u32,
    ) -> Self {
        let position_start = output_len.saturating_sub(scheduled_tokens);
        Self {
            request_id,
            mode: ax_engine_core::ExecutionMode::Decode,
            position_start,
            position_end_exclusive: output_len,
            scheduled_token_count: scheduled_tokens,
            input_token_count: 0,
            prefix_tokens_reused: 0,
            prefix_blocks_reused: 0,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FinalRequestState {
    pub(crate) external_id: String,
    pub(crate) request_id: RequestId,
    pub(crate) state: String,
    pub(crate) processed_prompt_tokens: u32,
    pub(crate) generated_tokens: Vec<u32>,
    pub(crate) cancel_requested: bool,
    pub(crate) last_error: Option<String>,
}

impl FinalRequestState {
    pub(crate) fn digest_json(&self) -> Value {
        json!({
            "external_id": self.external_id,
            "request_id": self.request_id.0,
            "state": self.state,
            "processed_prompt_tokens": self.processed_prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "cancel_requested": self.cancel_requested,
            "last_error": self.last_error
        })
    }
}
