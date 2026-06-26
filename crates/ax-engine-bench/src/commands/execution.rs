use super::*;

pub(crate) fn execute_manifest_with_runtime(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    if runtime.uses_mlx_runtime() {
        execute_manifest_once(manifest, runtime)
    } else {
        execute_manifest_llama_cpp_once(manifest, runtime)
    }
}

pub(crate) fn execute_manifest_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    match manifest.class {
        ManifestClass::Scenario => execute_scenario_once(manifest, runtime),
        ManifestClass::Replay => execute_replay_once(manifest, runtime),
    }
}

pub(crate) fn execute_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?;
    run_scenario_workload(runtime, specs)
}

pub(crate) fn scenario_specs_from_manifest(
    manifest: &BenchmarkManifest,
) -> Result<Vec<SyntheticRequestSpec>, CliError> {
    let shape = manifest.shape.as_ref().ok_or_else(|| {
        CliError::Contract("scenario manifest must contain a shape object".to_string())
    })?;
    if shape.input_tokens_target == 0 {
        return Err(CliError::Contract(
            "scenario manifest shape.input_tokens_target must be greater than zero".to_string(),
        ));
    }
    if shape.output_tokens_target == 0 {
        return Err(CliError::Contract(
            "scenario manifest shape.output_tokens_target must be greater than zero".to_string(),
        ));
    }
    let shared_prefix = manifest.scenario == "shared_prefix";

    let mut specs = Vec::new();
    for ordinal in 0..shape.concurrency {
        let request_id = RequestId(u64::from(ordinal) + 1);
        let prefix_group = if shared_prefix {
            Some("scenario-shared")
        } else {
            None
        };
        specs.push(SyntheticRequestSpec {
            external_id: format!("req-{}", ordinal + 1),
            request_id,
            arrival_sequence: SequenceNo(u64::from(ordinal) + 1),
            model_family: manifest.model.family.clone(),
            prompt_token_target: shape.input_tokens_target,
            input_tokens: synthetic_prompt_tokens(
                shape.input_tokens_target,
                Some("scenario"),
                prefix_group,
                None,
                ordinal,
            ),
            input_text: None,
            max_output_tokens: shape.output_tokens_target,
            sampling_params: sampling_from_manifest(manifest)?,
            metadata: prefix_group.map(str::to_string),
        });
    }

    Ok(specs)
}

pub(crate) fn execute_manifest_llama_cpp_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    validate_llama_cpp_benchmark_runtime(manifest, &runtime)?;

    match runtime.backend_adapter.as_ref() {
        Some(adapter) if adapter.supports_stepwise_benchmark() => match manifest.class {
            ManifestClass::Scenario => execute_llama_cpp_scenario_once(manifest, runtime),
            ManifestClass::Replay => execute_llama_cpp_replay_once(manifest, runtime),
        },
        Some(adapter) if adapter.supports_blocking_benchmark() => match manifest.class {
            ManifestClass::Scenario => execute_llama_cpp_blocking_scenario_once(manifest, runtime),
            ManifestClass::Replay => Err(CliError::Contract(
                "blocking llama.cpp benchmark adapters currently support scenario manifests only"
                    .to_string(),
            )),
        },
        _ => Err(CliError::Contract(
            "llama.cpp benchmark execution requires a supported backend adapter".to_string(),
        )),
    }
}

pub(crate) fn execute_llama_cpp_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?;
    if specs.is_empty() {
        return Err(CliError::Contract(
            "scenario manifest produced no requests".to_string(),
        ));
    }
    run_llama_cpp_scenario_workload(manifest, runtime, specs)
}

pub(crate) fn execute_llama_cpp_blocking_scenario_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    let specs = scenario_specs_from_manifest(manifest)?
        .into_iter()
        .map(|spec| {
            let prompt_text = synthetic_prompt_text(
                spec.prompt_token_target,
                Some(&spec.external_id),
                spec.metadata.as_deref(),
                None,
                spec.request_id.0 as u32,
            );
            spec.with_input_text(prompt_text)
        })
        .collect::<Vec<_>>();
    if specs.is_empty() {
        return Err(CliError::Contract(
            "scenario manifest produced no requests".to_string(),
        ));
    }
    run_llama_cpp_blocking_scenario_workload(manifest, runtime, specs)
}

pub(crate) fn execute_llama_cpp_replay_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    run_llama_cpp_replay_workload(manifest, runtime, replay_events_from_manifest(manifest)?)
}

pub(crate) fn run_llama_cpp_scenario_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let prefill_tokens = specs
        .iter()
        .map(|spec| spec.input_tokens.len() as u64)
        .sum();
    let mut observation = RuntimeObservation {
        prefill_tokens,
        total_scheduled_tokens: prefill_tokens,
        ..RuntimeObservation::default()
    };
    let request_ids = specs.iter().map(|spec| spec.request_id).collect::<Vec<_>>();
    let max_steps = workload_step_guard_from_specs(&specs);
    let mut session = build_session(&runtime, &specs)?;
    observation.runtime_report = Some(session.runtime_report());

    for spec in &specs {
        session
            .submit_generate_with_request_id(
                spec.request_id.0,
                generate_request_from_spec(spec, manifest),
            )
            .map_err(|error| {
                CliError::Runtime(format!(
                    "llama.cpp benchmark request failed through the SDK contract: {error}"
                ))
            })?;
    }

    loop {
        if !llama_cpp_session_has_live_requests(&session, &request_ids)? {
            let elapsed_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
            let final_reports = specs
                .iter()
                .map(|spec| {
                    let report = session.request_report(spec.request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!(
                            "missing llama.cpp request {}",
                            spec.request_id.0
                        ))
                    })?;
                    Ok((spec.clone(), report))
                })
                .collect::<Result<Vec<_>, CliError>>()?;
            observation.finalize_llama_cpp(final_reports, elapsed_ms.max(1));
            return Ok(observation);
        }

        let reports_before = llama_cpp_reports_for_session(&session, &request_ids)?;
        let step = session.step_report().map_err(|error| {
            CliError::Runtime(format!(
                "llama.cpp benchmark step failed through the SDK contract: {error}"
            ))
        })?;
        let current_time_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1;
        let reports_after = llama_cpp_reports_for_session(&session, &request_ids)?;
        observation.observe_llama_cpp_session_step(
            &reports_before,
            &reports_after,
            &step,
            current_time_ms,
        );
        let progress_made = step.scheduled_requests > 0
            || step.scheduled_tokens > 0
            || llama_cpp_reports_changed(&reports_before, &reports_after);

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "llama.cpp scenario exceeded delegated step guard".to_string(),
            ));
        }

        if !progress_made {
            return Err(CliError::Runtime(
                "llama.cpp scenario stalled with a live delegated request".to_string(),
            ));
        }
    }
}

pub(crate) fn run_llama_cpp_blocking_scenario_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let mut observation = RuntimeObservation::default();
    let mut session = build_session(&runtime, &specs)?;
    observation.runtime_report = Some(session.runtime_report());

    let mut final_reports = Vec::new();
    for spec in specs {
        let request_started = Instant::now();
        let response = session
            .generate_with_request_id(
                spec.request_id.0,
                generate_request_from_spec(&spec, manifest),
            )
            .map_err(|error| {
                CliError::Runtime(format!(
                    "blocking llama.cpp benchmark request failed through the SDK contract: {error}"
                ))
            })?;
        let elapsed_ms = request_started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64
            + 1;
        let prompt_token_count = response
            .known_prompt_token_count()
            .unwrap_or(spec.prompt_token_target);
        let known_output_token_count = response.known_output_token_count();
        observation.record_token_accounting_source(
            "prompt",
            prompt_token_count_source(&response, response.known_prompt_token_count().is_none()),
        );
        observation.step_count += 1;
        observation.total_selected_requests += 1;
        observation.prefill_tokens += u64::from(prompt_token_count);
        observation.total_scheduled_tokens += u64::from(prompt_token_count);
        if observation.ttft_ms.is_none() {
            observation.ttft_ms = Some(elapsed_ms);
        }

        let mut used_synthetic_output_estimate = false;
        let output_tokens = if response.output_tokens.is_empty() {
            if known_output_token_count.is_some() {
                Vec::new()
            } else {
                used_synthetic_output_estimate = response.output_text.is_some();
                response
                    .output_text
                    .as_deref()
                    .map(|text| synthetic_text_output_tokens(text, spec.max_output_tokens))
                    .unwrap_or_default()
            }
        } else {
            response.output_tokens.clone()
        };
        let output_token_count = known_output_token_count.unwrap_or(output_tokens.len() as u32);
        observation.record_token_accounting_source(
            "output",
            output_token_count_source(&response, used_synthetic_output_estimate),
        );
        observation.decode_tokens += u64::from(output_token_count);
        observation.total_scheduled_tokens += u64::from(output_token_count);

        let route_metadata = route_metadata_from_generate_route(&response.route);
        observation.merge_route_metadata(&route_metadata);

        let final_state = match response.status {
            GenerateStatus::Pending => SessionRequestState::Running,
            GenerateStatus::Finished => SessionRequestState::Finished,
            GenerateStatus::Cancelled => SessionRequestState::Cancelled,
            GenerateStatus::Failed => SessionRequestState::Failed,
        };
        let max_output_tokens = spec.max_output_tokens;
        let output_token_logprobs = if response.output_tokens.is_empty() {
            Vec::new()
        } else {
            response.output_token_logprobs.clone()
        };
        final_reports.push((
            spec,
            SessionRequestReport {
                request_id: response.request_id,
                model_id: response.model_id,
                state: final_state,
                prompt_tokens: response.prompt_tokens,
                processed_prompt_tokens: prompt_token_count,
                output_tokens,
                output_token_logprobs,
                prompt_len: prompt_token_count,
                output_len: output_token_count,
                max_output_tokens,
                cancel_requested: false,
                execution_plan_ref: response.route.execution_plan.clone(),
                route: response.route,
                finish_reason: response.finish_reason,
                terminal_stop_reason: stop_reason_from_generate_finish_reason(
                    response.finish_reason,
                ),
                last_error: None,
            },
        ));
    }

    observation.e2e_latency_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64 + 1;
    observation.finalize_llama_cpp(final_reports, observation.e2e_latency_ms.max(1));
    Ok(observation)
}

pub(crate) fn run_llama_cpp_replay_workload(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
    events: Vec<ReplayEvent>,
) -> Result<RuntimeObservation, CliError> {
    let specs = events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut session = build_session(&runtime, &specs)?;
    let spec_by_request = specs
        .iter()
        .cloned()
        .map(|spec| (spec.request_id, spec))
        .collect::<BTreeMap<_, _>>();
    let mut submitted_request_ids = Vec::new();

    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut cancelled_requests = BTreeSet::new();
    let mut event_index = 0usize;
    let mut current_time_ms = 0u64;
    let max_steps = replay_step_guard(&events);

    while event_index < events.len()
        || llama_cpp_session_has_live_requests(&session, &submitted_request_ids)?
    {
        while let Some(event) = events.get(event_index) {
            if event.t_ms() > current_time_ms {
                break;
            }

            match event {
                ReplayEvent::Submit { spec, .. } => {
                    if submitted_request_ids.contains(&spec.request_id) {
                        return Err(CliError::Runtime(format!(
                            "llama.cpp replay request {:?} was submitted more than once",
                            spec.request_id
                        )));
                    }

                    session
                        .submit_generate_with_request_id(
                            spec.request_id.0,
                            generate_request_from_spec(spec, manifest),
                        )
                        .map_err(|error| {
                            CliError::Runtime(format!(
                                "failed to submit llama.cpp replay benchmark request: {error}"
                            ))
                        })?;
                    submitted_request_ids.push(spec.request_id);
                    observation.prefill_tokens += spec.input_tokens.len() as u64;
                    observation.total_scheduled_tokens += spec.input_tokens.len() as u64;

                    let report = session.request_report(spec.request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!(
                            "missing llama.cpp request {}",
                            spec.request_id.0
                        ))
                    })?;
                    observation
                        .merge_route_metadata(&route_metadata_from_generate_route(&report.route));
                }
                ReplayEvent::Cancel { request_id, .. } => {
                    if !submitted_request_ids.contains(request_id) {
                        return Err(CliError::Runtime(format!(
                            "llama.cpp replay cancel arrived before submit for {:?}",
                            request_id
                        )));
                    }

                    cancelled_requests.insert(*request_id);
                    session.cancel_request(request_id.0).map_err(|error| {
                        CliError::Runtime(format!(
                            "failed to cancel llama.cpp replay benchmark request: {error}"
                        ))
                    })?;
                    let report = session.request_report(request_id.0).ok_or_else(|| {
                        CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
                    })?;
                    observation
                        .merge_route_metadata(&route_metadata_from_generate_route(&report.route));
                }
            }

            event_index += 1;
        }

        if !llama_cpp_session_has_live_requests(&session, &submitted_request_ids)? {
            if let Some(next_event) = events.get(event_index) {
                current_time_ms = next_event.t_ms();
                continue;
            }
            break;
        }

        let tick_time_ms = current_time_ms.saturating_add(1);
        let reports_before = llama_cpp_reports_for_session(&session, &submitted_request_ids)?;
        let step = session.step_report().map_err(|error| {
            CliError::Runtime(format!(
                "llama.cpp replay step failed through the SDK contract: {error}"
            ))
        })?;
        let reports_after = llama_cpp_reports_for_session(&session, &submitted_request_ids)?;
        observation.observe_llama_cpp_session_step(
            &reports_before,
            &reports_after,
            &step,
            tick_time_ms,
        );
        let progress_made = step.scheduled_requests > 0
            || step.scheduled_tokens > 0
            || llama_cpp_reports_changed(&reports_before, &reports_after);

        current_time_ms = tick_time_ms;

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "llama.cpp replay exceeded delegated step guard".to_string(),
            ));
        }

        if !progress_made {
            return Err(CliError::Runtime(
                "llama.cpp replay stalled with a live delegated request".to_string(),
            ));
        }
    }

    observation.cancelled_requests = cancelled_requests;
    let final_reports = submitted_request_ids
        .iter()
        .map(|request_id| {
            let spec = spec_by_request.get(request_id).cloned().ok_or_else(|| {
                CliError::Runtime(format!("missing llama.cpp replay spec {:?}", request_id))
            })?;
            let report = session.request_report(request_id.0).ok_or_else(|| {
                CliError::Runtime(format!("missing llama.cpp request {}", request_id.0))
            })?;
            Ok((spec, report))
        })
        .collect::<Result<Vec<_>, CliError>>()?;
    observation.finalize_llama_cpp(final_reports, current_time_ms.max(1));
    Ok(observation)
}

pub(crate) fn execute_replay_once(
    manifest: &BenchmarkManifest,
    runtime: RuntimeConfig,
) -> Result<RuntimeObservation, CliError> {
    run_replay_workload(runtime, replay_events_from_manifest(manifest)?)
}

pub(crate) fn run_scenario_workload(
    runtime: RuntimeConfig,
    specs: Vec<SyntheticRequestSpec>,
) -> Result<RuntimeObservation, CliError> {
    let started = Instant::now();
    let max_steps = workload_step_guard_from_specs(&specs);
    let mut session = build_session(&runtime, &specs)?;
    let mut submitted_request_ids = Vec::new();
    let mut external_ids = BTreeMap::new();
    let shared_prefix_staging = should_stage_shared_prefix_scenario(&specs);
    let mut pending_specs = Vec::new();

    for (index, spec) in specs.into_iter().enumerate() {
        external_ids.insert(spec.request_id, spec.external_id.clone());
        if shared_prefix_staging && index > 0 {
            pending_specs.push(spec);
            continue;
        }
        submit_scenario_spec(&mut session, &mut submitted_request_ids, spec)?;
    }

    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut current_time_ms = 0u64;

    while !pending_specs.is_empty() || has_live_requests(session.core(), &submitted_request_ids)? {
        if current_time_ms > 0 && !pending_specs.is_empty() {
            for spec in pending_specs.drain(..) {
                submit_scenario_spec(&mut session, &mut submitted_request_ids, spec)?;
            }
        }

        if submitted_request_ids.is_empty()
            || !has_live_requests(session.core(), &submitted_request_ids)?
        {
            break;
        }

        let outcome = session
            .step()
            .map_err(|error| CliError::Runtime(format!("engine session step failed: {error}")))?;
        current_time_ms = elapsed_ms_since(started);
        observation.observe_step(session.core(), &outcome, current_time_ms);

        if observation.step_count > max_steps {
            return Err(CliError::Runtime(
                "scenario workload exceeded engine step guard".to_string(),
            ));
        }

        if outcome.runner_output.is_none()
            && outcome.schedule_plan.memory_blocked_requests.is_empty()
            && outcome.schedule_plan.selected_requests.is_empty()
            && has_live_requests(session.core(), &submitted_request_ids)?
        {
            return Err(CliError::Runtime(
                "scenario workload stalled with live requests".to_string(),
            ));
        }
    }

    observation.finalize(
        session.core(),
        &submitted_request_ids,
        external_ids,
        elapsed_ms_since(started),
        runtime.block_size_tokens,
    )?;
    Ok(observation)
}

pub(crate) fn should_stage_shared_prefix_scenario(specs: &[SyntheticRequestSpec]) -> bool {
    specs.len() > 1
        && specs
            .iter()
            .all(|spec| spec.metadata.as_deref() == Some("scenario-shared"))
}

pub(crate) fn submit_scenario_spec(
    session: &mut EngineSession,
    submitted_request_ids: &mut Vec<RequestId>,
    spec: SyntheticRequestSpec,
) -> Result<(), CliError> {
    submitted_request_ids.push(spec.request_id);
    session.submit(spec.into_submission()).map_err(|error| {
        CliError::Runtime(format!("failed to submit benchmark request: {error}"))
    })?;
    Ok(())
}

pub(crate) fn run_replay_workload(
    runtime: RuntimeConfig,
    events: Vec<ReplayEvent>,
) -> Result<RuntimeObservation, CliError> {
    let specs = events
        .iter()
        .filter_map(|event| match event {
            ReplayEvent::Submit { spec, .. } => Some(spec.clone()),
            ReplayEvent::Cancel { .. } => None,
        })
        .collect::<Vec<_>>();
    let mut session = build_session(&runtime, &specs)?;
    let mut request_ids = Vec::new();
    let mut external_ids = BTreeMap::new();
    let mut cancelled_requests = BTreeSet::new();
    let mut observation = RuntimeObservation {
        runtime_report: Some(session.runtime_report()),
        ..RuntimeObservation::default()
    };
    let mut event_index = 0usize;
    let mut current_time_ms = 0u64;
    let max_steps = replay_step_guard(&events);

    while event_index < events.len() || has_live_requests(session.core(), &request_ids)? {
        while let Some(event) = events.get(event_index) {
            if event.t_ms() > current_time_ms {
                break;
            }

            match event {
                ReplayEvent::Submit { spec, .. } => {
                    request_ids.push(spec.request_id);
                    external_ids.insert(spec.request_id, spec.external_id.clone());
                    session
                        .submit(spec.clone().into_submission())
                        .map_err(|error| {
                            CliError::Runtime(format!(
                                "failed to submit replay benchmark request: {error}"
                            ))
                        })?;
                }
                ReplayEvent::Cancel { request_id, .. } => {
                    cancelled_requests.insert(*request_id);
                    session.cancel(*request_id).map_err(|error| {
                        CliError::Runtime(format!(
                            "failed to cancel replay benchmark request: {error}"
                        ))
                    })?;
                }
            }

            event_index += 1;
        }

        if !has_live_requests(session.core(), &request_ids)? {
            if let Some(next_event) = events.get(event_index) {
                current_time_ms = next_event.t_ms();
                continue;
            }
            break;
        }

        let outcome = session
            .step()
            .map_err(|error| CliError::Runtime(format!("engine session step failed: {error}")))?;
        current_time_ms += 1;
        observation.observe_step(session.core(), &outcome, current_time_ms);

        if current_time_ms > max_steps {
            return Err(CliError::Runtime(format!(
                "replay workload exceeded engine step guard: current_time_ms={current_time_ms}, max_steps={max_steps}"
            )));
        }

        if outcome.runner_output.is_none()
            && outcome.schedule_plan.memory_blocked_requests.is_empty()
            && outcome.schedule_plan.selected_requests.is_empty()
            && has_live_requests(session.core(), &request_ids)?
        {
            return Err(CliError::Runtime(
                "replay workload stalled with live requests".to_string(),
            ));
        }
    }

    observation.cancelled_requests = cancelled_requests;
    observation.finalize(
        session.core(),
        &request_ids,
        external_ids,
        current_time_ms,
        runtime.block_size_tokens,
    )?;
    Ok(observation)
}

pub(crate) const WORKLOAD_STEP_GUARD_MIN: u64 = 256;
pub(crate) const WORKLOAD_STEP_GUARD_SLACK: u64 = 512;

pub(crate) fn workload_step_guard_from_specs(specs: &[SyntheticRequestSpec]) -> u64 {
    output_step_guard(specs.iter().map(|spec| spec.max_output_tokens))
}

pub(crate) fn output_step_guard(max_output_tokens: impl Iterator<Item = u32>) -> u64 {
    max_output_tokens
        .map(u64::from)
        .max()
        .unwrap_or(0)
        .saturating_add(WORKLOAD_STEP_GUARD_SLACK)
        .max(WORKLOAD_STEP_GUARD_MIN)
}

pub(crate) fn replay_step_guard(events: &[ReplayEvent]) -> u64 {
    let latest_event_time = events.iter().map(ReplayEvent::t_ms).max().unwrap_or(0);
    let output_guard = output_step_guard(events.iter().filter_map(|event| match event {
        ReplayEvent::Submit { spec, .. } => Some(spec.max_output_tokens),
        ReplayEvent::Cancel { .. } => None,
    }));

    latest_event_time.saturating_add(output_guard)
}

pub(crate) fn session_config_from_runtime(
    runtime: &RuntimeConfig,
    specs: &[SyntheticRequestSpec],
) -> EngineSessionConfig {
    let estimated_total_blocks = specs
        .iter()
        .map(|spec| {
            u64::from(spec.prompt_token_target + spec.max_output_tokens)
                .div_ceil(u64::from(runtime.block_size_tokens))
        })
        .sum::<u64>()
        .max(1) as u32;
    let kv_total_blocks = runtime
        .kv_total_blocks
        .unwrap_or_else(|| estimated_total_blocks.saturating_add(8));

    let llama_backend = if runtime.resolved_backend.selected_backend.is_mlx() {
        None
    } else {
        runtime
            .backend_adapter
            .as_ref()
            .map(BackendAdapterManifest::to_sdk_config)
    };
    let mlx_runtime_artifacts_dir = if runtime.resolved_backend.selected_backend.is_mlx() {
        EngineSessionConfig::default_mlx_runtime_artifacts_dir().or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|dir| dir.join("build/metal"))
                .filter(|dir| dir.join("build_report.json").is_file())
        })
    } else {
        None
    };
    let mlx_runtime_artifacts_source = mlx_runtime_artifacts_dir
        .as_ref()
        .map(|_| NativeRuntimeArtifactsSource::RepoAutoDetect);
    EngineSessionConfig::from_resolved_request(ResolvedSessionConfigRequest {
        cache_group_id: CacheGroupId(0),
        block_size_tokens: runtime.block_size_tokens,
        total_blocks: kv_total_blocks,
        deterministic: runtime.deterministic,
        max_batch_tokens: runtime.max_batch_tokens,
        backend_policy: runtime.backend_policy,
        resolved_backend: runtime.resolved_backend.clone(),
        llama_backend,
        mlx_lm_backend: None,
        mlx_runtime_artifacts_dir,
        mlx_runtime_artifacts_source,
        mlx_model_artifacts_dir: runtime.mlx_model_artifacts_dir.clone(),
        mlx_model_artifacts_source: runtime.mlx_model_artifacts_source,
        mlx_disable_ngram_acceleration: false,
        mlx_mtp_disable_ngram_stacking: true,
        mlx_speculation_profile: None,
        mlx_kv_compression: ax_engine_sdk::KvCompressionConfig::disabled(),
        // Match mlx-lm's prefill_step_size=2048 default for like-for-like
        // bench comparisons. The runner auto-clamps for linear-attention
        // (512) and MLA (16) model families. See
        // crates/ax-engine-bench/src/inference_args.rs::BENCH_DEFAULT_MLX_PREFILL_CHUNK
        // for the rationale.
        mlx_prefill_chunk: Some(crate::inference_args::BENCH_DEFAULT_MLX_PREFILL_CHUNK),
    })
}

pub(crate) fn build_session(
    runtime: &RuntimeConfig,
    specs: &[SyntheticRequestSpec],
) -> Result<EngineSession, CliError> {
    let session_config = session_config_from_runtime(runtime, specs);

    EngineSession::new(session_config).map_err(|error| {
        CliError::Contract(format!(
            "failed to create benchmark SDK session from manifest runtime: {error}"
        ))
    })
}

pub(crate) fn evaluate_correctness(
    manifest: &BenchmarkManifest,
    observation: &RuntimeObservation,
) -> Result<GateStatus, CliError> {
    if observation
        .final_requests
        .iter()
        .any(|request| request.state == "Failed")
    {
        return Ok(GateStatus::fail(
            "one or more requests reached Failed in engine bring-up execution",
        ));
    }

    if observation
        .final_requests
        .iter()
        .any(|request| request.state != "Finished" && request.state != "Cancelled")
    {
        return Ok(GateStatus::fail(
            "engine execution completed with non-terminal live requests",
        ));
    }

    if manifest_requests_output(manifest)
        && observation.decode_tokens == 0
        && observation
            .final_requests
            .iter()
            .any(|request| request.state == "Finished")
    {
        return Ok(GateStatus::fail(
            "one or more finished requests produced zero output tokens",
        ));
    }

    let require_prefix_reuse = manifest.checks.require_prefix_reuse;
    if require_prefix_reuse && observation.prefix_hit_rate() <= f64::EPSILON {
        return Ok(GateStatus::fail(
            "manifest requires prefix reuse but runtime reported zero prefix hits",
        ));
    }

    if observation
        .cancelled_requests
        .iter()
        .any(|request_id| !observation.request_state(*request_id, "Cancelled"))
    {
        return Ok(GateStatus::fail(
            "cancelled replay request did not finish in Cancelled state",
        ));
    }

    if manifest.checks.require_no_allocator_churn_failure && observation.churn_status() == "fail" {
        return Ok(GateStatus::fail(
            "manifest requires no allocator churn failure but one or more requests reached Failed state",
        ));
    }

    Ok(GateStatus::pass())
}

pub(crate) fn manifest_requests_output(manifest: &BenchmarkManifest) -> bool {
    match manifest.class {
        ManifestClass::Scenario => manifest
            .shape
            .as_ref()
            .is_some_and(|shape| shape.output_tokens_target > 0),
        ManifestClass::Replay => manifest.events.iter().any(|event| match event {
            ReplayEventManifest {
                kind: ReplayEventKind::Submit,
                output_tokens_target,
                ..
            } => output_tokens_target.unwrap_or(0) > 0,
            ReplayEventManifest {
                kind: ReplayEventKind::Cancel,
                ..
            } => false,
        }),
    }
}

pub(crate) fn enforce_runtime_gates(execution: &RuntimeResult) -> Result<(), CliError> {
    if !execution.correctness.passed {
        return Err(CliError::Correctness(gate_failure_reason(
            &execution.correctness,
            "correctness gate failed",
        )));
    }

    if !execution.determinism.passed {
        return Err(CliError::Correctness(gate_failure_reason(
            &execution.determinism,
            "determinism gate failed",
        )));
    }

    Ok(())
}

pub(crate) fn gate_failure_reason(status: &GateStatus, fallback: &str) -> String {
    status
        .reason
        .clone()
        .unwrap_or_else(|| fallback.to_string())
}
