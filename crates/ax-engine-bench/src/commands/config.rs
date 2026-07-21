use super::*;

pub(crate) fn runtime_config_from_manifest(
    manifest: &BenchmarkManifest,
) -> Result<RuntimeConfig, CliError> {
    let backend_policy = BackendPolicy::new(manifest.runtime.resolution_policy);
    let resolved_backend = ResolvedBackend::new(
        manifest.runtime.selected_backend,
        manifest.runtime.support_tier,
        manifest.runtime.fallback_reason.clone(),
    );
    resolved_backend
        .validate_against(&backend_policy)
        .map_err(|error| {
            CliError::Contract(format!(
                "invalid benchmark runtime backend-resolution contract: {error}"
            ))
        })?;

    if resolved_backend.selected_backend == SelectedBackend::Mlx {
        if let Some(backend_adapter) = manifest.runtime.backend_adapter.as_ref() {
            return Err(CliError::Contract(format!(
                "runtime.backend_adapter must be omitted when selected_backend is mlx; found adapter={:?}",
                backend_adapter.selected_backend()
            )));
        }
        if manifest.runtime.llama_cpp_preset.is_some() {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset must be omitted when selected_backend is mlx".to_string(),
            ));
        }
    } else {
        let backend_adapter = manifest.runtime.backend_adapter.as_ref().ok_or_else(|| {
            CliError::Contract(
                "runtime.backend_adapter is required when selected_backend is llama_cpp"
                    .to_string(),
            )
        })?;
        if backend_adapter.selected_backend() != resolved_backend.selected_backend {
            return Err(CliError::Contract(format!(
                "runtime.backend_adapter kind does not match selected_backend: adapter={:?}, selected_backend={:?}",
                backend_adapter.selected_backend(),
                resolved_backend.selected_backend
            )));
        }
    }
    let llama_cpp_preset = if resolved_backend.selected_backend == SelectedBackend::LlamaCpp {
        let preset = manifest
            .runtime
            .llama_cpp_preset
            .clone()
            .unwrap_or_default();
        validate_llama_cpp_preset(&preset, manifest)?;
        Some(preset)
    } else {
        None
    };

    let default_session_config = EngineSessionConfig::default();
    let mlx_model_artifacts_dir = manifest
        .runtime
        .mlx_model_artifacts_dir
        .clone()
        .or(default_session_config.mlx_model_artifacts_dir.clone());
    let mlx_model_artifacts_source = manifest
        .runtime
        .mlx_model_artifacts_dir
        .as_ref()
        .map(|_| NativeModelArtifactsSource::ExplicitConfig)
        .or(default_session_config.mlx_model_artifacts_source);

    Ok(RuntimeConfig {
        deterministic: manifest.runtime.deterministic,
        max_batch_tokens: manifest.runtime.max_batch_tokens,
        block_size_tokens: 16,
        kv_total_blocks: manifest.runtime.kv_total_blocks,
        flags: manifest.runtime.flags.clone(),
        llama_cpp_preset,
        backend_policy,
        resolved_backend,
        backend_adapter: manifest.runtime.backend_adapter.clone(),
        mlx_model_artifacts_dir,
        mlx_model_artifacts_source,
    })
}

pub(crate) fn validate_llama_cpp_preset(
    preset: &LlamaCppPresetManifest,
    manifest: &BenchmarkManifest,
) -> Result<(), CliError> {
    if preset.parallel_slots == 0 {
        return Err(CliError::Contract(
            "runtime.llama_cpp_preset.parallel_slots must be greater than zero".to_string(),
        ));
    }
    if let Some(logical_batch_size) = preset.logical_batch_size {
        if logical_batch_size == 0 {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.logical_batch_size must be greater than zero".to_string(),
            ));
        }
    }
    if let Some(physical_batch_size) = preset.physical_batch_size {
        if physical_batch_size == 0 {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.physical_batch_size must be greater than zero"
                    .to_string(),
            ));
        }
    }
    if let (Some(logical), Some(physical)) = (preset.logical_batch_size, preset.physical_batch_size)
    {
        if physical > logical {
            return Err(CliError::Contract(
                "runtime.llama_cpp_preset.physical_batch_size must not exceed logical_batch_size"
                    .to_string(),
            ));
        }
    }
    if manifest.checks.require_prefix_reuse && !preset.cache_prompt {
        return Err(CliError::Contract(
            "runtime.llama_cpp_preset.cache_prompt must be true when checks.require_prefix_reuse is true"
                .to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn normalize_runtime_for_test_execution(
    runtime: RuntimeConfig,
    _preserve_native_artifacts_for_test: bool,
) -> RuntimeConfig {
    #[cfg(test)]
    {
        // Unit tests should remain stable across local machines and must not
        // silently flip from deterministic placeholder execution into repo
        // auto-detected Metal bring-up just because native artifacts happen to
        // exist in the checkout.
        if runtime.uses_mlx_runtime() && !_preserve_native_artifacts_for_test {
            return RuntimeConfig {
                mlx_model_artifacts_dir: None,
                mlx_model_artifacts_source: None,
                ..runtime
            };
        }
    }

    runtime
}

pub(crate) fn sampling_from_manifest(
    manifest: &BenchmarkManifest,
) -> Result<SamplingParams, CliError> {
    Ok(SamplingParams {
        temperature: manifest.sampling.temperature,
        top_p: manifest.sampling.top_p,
        top_k: manifest.sampling.top_k,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        no_repeat_ngram_size: 0,
        ngram_window: 128,
        seed: manifest.sampling.seed,
        deterministic: manifest.runtime.deterministic,
        ignore_eos: manifest.sampling.ignore_eos,
    })
}

pub(crate) fn generate_request_from_spec(
    spec: &SyntheticRequestSpec,
    manifest: &BenchmarkManifest,
) -> GenerateRequest {
    GenerateRequest {
        model_id: spec.model_family.clone(),
        input_tokens: spec.input_tokens.clone(),
        input_text: spec.input_text.clone(),
        multimodal_inputs: Default::default(),
        max_output_tokens: spec.max_output_tokens,
        sampling: GenerateSampling {
            temperature: manifest.sampling.temperature,
            top_p: manifest.sampling.top_p,
            top_k: manifest.sampling.top_k,
            min_p: None,
            repetition_penalty: 1.0,
            repetition_context_size: None,
            no_repeat_ngram_size: 0,
            ngram_window: 128,
            seed: manifest.sampling.seed,
            deterministic: Some(manifest.runtime.deterministic),
            ignore_eos: manifest.sampling.ignore_eos,
        },
        stop_sequences: Vec::new(),
        metadata: spec.metadata.clone(),
    }
}

pub(crate) fn manifest_expect_deterministic(
    manifest: &BenchmarkManifest,
) -> Result<bool, CliError> {
    Ok(manifest.checks.expect_deterministic)
}

pub(crate) fn direct_decode_batching_opportunity_observed(step_trace: &[StepTraceEntry]) -> bool {
    step_trace.iter().any(|step| {
        step.items
            .iter()
            .filter(|item| item.mode == ax_engine_core::ExecutionMode::Decode)
            .take(2)
            .count()
            > 1
    })
}

pub(crate) fn annotate_route_json_with_decode_batching_opportunity(
    route_json: &mut Value,
    step_trace: &[StepTraceEntry],
) {
    let Some(route_object) = route_json.as_object_mut() else {
        return;
    };
    route_object.insert(
        "metal_direct_decode_batching_opportunity_observed".to_string(),
        json!(direct_decode_batching_opportunity_observed(step_trace)),
    );
}

#[allow(
    clippy::expect_used,
    reason = "runtime metadata uses controlled object literals and infallible report serializers"
)]
pub(crate) fn serialize_runtime_metadata(
    runtime: &RuntimeConfig,
    actual_runtime: Option<&RuntimeReport>,
) -> Value {
    let mut runtime_json = json!({
        "selected_backend": runtime.resolved_backend.selected_backend,
        "support_tier": runtime.resolved_backend.support_tier,
        "resolution_policy": runtime.backend_policy.resolution_policy,
        "deterministic": runtime.deterministic,
        "max_batch_tokens": runtime.max_batch_tokens,
        "block_size_tokens": runtime.block_size_tokens,
        "flags": {
            "prefix_cache": runtime.flags.prefix_cache
        },
        "host": current_host_report(),
        "metal_toolchain": current_metal_toolchain_report()
    });

    if let Some(preset) = runtime.llama_cpp_preset.as_ref() {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert(
                "llama_cpp_preset".to_string(),
                serde_json::to_value(preset).expect("llama.cpp preset should serialize"),
            );
    }

    if let Some(actual_runtime) = actual_runtime {
        let runtime_object = runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object");
        runtime_object.insert(
            "selected_backend".to_string(),
            serde_json::to_value(actual_runtime.selected_backend)
                .expect("runtime selected_backend should serialize"),
        );
        runtime_object.insert(
            "support_tier".to_string(),
            serde_json::to_value(actual_runtime.support_tier)
                .expect("runtime support_tier should serialize"),
        );
        runtime_object.insert(
            "resolution_policy".to_string(),
            serde_json::to_value(actual_runtime.resolution_policy)
                .expect("runtime resolution_policy should serialize"),
        );
        runtime_object.insert(
            "capabilities".to_string(),
            serde_json::to_value(&actual_runtime.capabilities)
                .expect("runtime capabilities should serialize"),
        );
        runtime_object.insert(
            "host".to_string(),
            serde_json::to_value(&actual_runtime.host).expect("runtime host should serialize"),
        );
        runtime_object.insert(
            "metal_toolchain".to_string(),
            serde_json::to_value(&actual_runtime.metal_toolchain)
                .expect("runtime metal_toolchain should serialize"),
        );
        if let Some(fallback_reason) = actual_runtime.fallback_reason.as_ref() {
            runtime_object.insert("fallback_reason".to_string(), json!(fallback_reason));
        } else {
            runtime_object.remove("fallback_reason");
        }
        if let Some(native_runtime) = actual_runtime.mlx_runtime.as_ref() {
            runtime_object.insert(
                "mlx_runtime".to_string(),
                serde_json::to_value(native_runtime).expect("MLX runtime should serialize"),
            );
        }
        if let Some(native_model) = actual_runtime.mlx_model.as_ref() {
            runtime_object.insert(
                "mlx_model".to_string(),
                serde_json::to_value(native_model).expect("MLX model should serialize"),
            );
        }
    }

    if let Some(kv_total_blocks) = runtime.kv_total_blocks {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert("kv_total_blocks".to_string(), json!(kv_total_blocks));
    }

    if let Some(backend_adapter) = runtime_backend_adapter_for_report(runtime, actual_runtime) {
        runtime_json
            .as_object_mut()
            .expect("runtime metadata should serialize as object")
            .insert(
                "backend_adapter".to_string(),
                serde_json::to_value(backend_adapter).expect("backend adapter should serialize"),
            );
    }

    if actual_runtime.is_none() {
        if let Some(fallback_reason) = runtime.resolved_backend.fallback_reason.as_deref() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert("fallback_reason".to_string(), json!(fallback_reason));
        }
    }

    if actual_runtime.is_none() {
        if let Some(native_runtime) = runtime.mlx_runtime_report() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert(
                    "mlx_runtime".to_string(),
                    serde_json::to_value(native_runtime).expect("MLX runtime should serialize"),
                );
        }
    }

    if actual_runtime.is_none() {
        if let Some(native_model) = runtime.native_model_report() {
            runtime_json
                .as_object_mut()
                .expect("runtime metadata should serialize as object")
                .insert(
                    "mlx_model".to_string(),
                    serde_json::to_value(native_model).expect("MLX model should serialize"),
                );
        }
    }

    runtime_json
}

pub(crate) fn runtime_backend_adapter_for_report<'a>(
    runtime: &'a RuntimeConfig,
    actual_runtime: Option<&RuntimeReport>,
) -> Option<&'a BackendAdapterManifest> {
    let selected_backend = actual_runtime
        .map(|report| report.selected_backend)
        .unwrap_or(runtime.resolved_backend.selected_backend);

    runtime
        .backend_adapter
        .as_ref()
        .filter(|adapter| adapter.selected_backend() == selected_backend)
}

pub(crate) fn replay_events_from_manifest(
    manifest: &BenchmarkManifest,
) -> Result<Vec<ReplayEvent>, CliError> {
    let events = if manifest.events.is_empty() {
        return Err(CliError::Contract(
            "replay manifest must contain an events array".to_string(),
        ));
    } else {
        &manifest.events
    };

    let mut replay_events = Vec::with_capacity(events.len());
    let mut next_request_id = 1u64;
    let mut request_ids_by_external = BTreeMap::new();

    for event in events {
        match event.kind {
            ReplayEventKind::Submit => {
                let external_id = event
                    .request_id
                    .as_deref()
                    .ok_or_else(|| {
                        CliError::Contract("submit event missing request_id".to_string())
                    })?
                    .to_string();
                let request_id = RequestId(next_request_id);
                next_request_id += 1;
                request_ids_by_external.insert(external_id.clone(), request_id);

                let prompt_ref = event.prompt_ref.as_deref().ok_or_else(|| {
                    CliError::Contract("submit event missing prompt_ref".to_string())
                })?;
                let prefix_group = event.prefix_group.as_deref();
                let body_group = event.body_group.as_deref();
                let prompt_len = replay_prompt_target(prompt_ref);

                replay_events.push(ReplayEvent::Submit {
                    t_ms: event.t_ms,
                    spec: SyntheticRequestSpec {
                        external_id,
                        request_id,
                        arrival_sequence: SequenceNo(request_id.0),
                        model_family: manifest.model.family.clone(),
                        prompt_token_target: prompt_len,
                        input_tokens: synthetic_prompt_tokens(
                            prompt_len,
                            Some(prompt_ref),
                            prefix_group,
                            // Replay fixtures are frozen: keep the historical
                            // 64-token shared prefix so replayed token streams
                            // and their determinism digests stay comparable
                            // with recorded baselines.
                            64,
                            body_group,
                            request_id.0 as u32,
                        ),
                        input_text: None,
                        max_output_tokens: event.output_tokens_target.ok_or_else(|| {
                            CliError::Contract(
                                "submit event missing output_tokens_target".to_string(),
                            )
                        })?,
                        sampling_params: sampling_from_manifest(manifest)?,
                        metadata: prefix_group.map(str::to_string),
                    },
                });
            }
            ReplayEventKind::Cancel => {
                let external_id = event.request_id.as_deref().ok_or_else(|| {
                    CliError::Contract("cancel event missing request_id".to_string())
                })?;
                let request_id = *request_ids_by_external.get(external_id).ok_or_else(|| {
                    CliError::Contract(format!(
                        "cancel event references request_id before submit: {external_id}"
                    ))
                })?;
                replay_events.push(ReplayEvent::Cancel {
                    t_ms: event.t_ms,
                    request_id,
                });
            }
        }
    }

    for window in replay_events.windows(2) {
        if window[1].t_ms() < window[0].t_ms() {
            return Err(CliError::Contract(format!(
                "replay events must be sorted by t_ms: found t_ms={} after t_ms={}",
                window[1].t_ms(),
                window[0].t_ms()
            )));
        }
    }

    Ok(replay_events)
}
