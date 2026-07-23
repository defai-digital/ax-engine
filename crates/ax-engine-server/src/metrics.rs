use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;

pub(crate) async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let metrics = state.metrics.as_ref();
    let lives = state.snapshots();
    let mut body = String::new();

    append_counter(
        &mut body,
        "ax_engine_http_requests_total",
        "Total HTTP requests observed by ax-engine-server.",
        metrics.http_requests_total.load(Ordering::Relaxed),
    );
    append_gauge(
        &mut body,
        "ax_engine_http_requests_in_flight",
        "HTTP requests currently executing inside ax-engine-server.",
        metrics.http_requests_in_flight.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_http_status_2xx_total",
        "HTTP responses with 2xx status.",
        metrics.http_status_2xx_total.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_http_status_4xx_total",
        "HTTP responses with 4xx status.",
        metrics.http_status_4xx_total.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_http_status_5xx_total",
        "HTTP responses with 5xx status.",
        metrics.http_status_5xx_total.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_grpc_requests_total",
        "Total gRPC requests observed by ax-engine-server.",
        metrics.grpc_requests_total.load(Ordering::Relaxed),
    );
    append_gauge(
        &mut body,
        "ax_engine_grpc_requests_in_flight",
        "gRPC requests currently executing inside ax-engine-server.",
        metrics.grpc_requests_in_flight.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_grpc_status_ok_total",
        "gRPC responses observed with an OK status. Successful unary and all streaming responses carry their real status in HTTP/2 trailers, invisible at this layer, so those are counted here by default.",
        metrics.grpc_status_ok_total.load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_grpc_status_error_total",
        "gRPC responses observed with a non-OK grpc-status response header (e.g. auth rejections).",
        metrics.grpc_status_error_total.load(Ordering::Relaxed),
    );
    append_gauge(
        &mut body,
        "ax_engine_jobs_in_flight",
        "Engine jobs holding shared HTTP and gRPC admission capacity.",
        state.admission.active_jobs() as u64,
    );
    append_model_gauge(
        &mut body,
        "ax_engine_model_jobs_in_flight",
        "Engine jobs holding one loaded model generation's admission scope.",
        &lives
            .iter()
            .map(|live| {
                (
                    live.model_id.as_ref().clone(),
                    live.admission.active_jobs() as u64,
                )
            })
            .collect::<Vec<_>>(),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_jobs_pending",
        "Commands and active streams owned by loaded model generation workers.",
        lives
            .iter()
            .map(|live| live.generation_service.pending_jobs() as u64)
            .sum(),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_commands_queued",
        "Commands waiting in loaded model generation workers' bounded queues.",
        lives
            .iter()
            .map(|live| live.generation_service.queued_commands() as u64)
            .sum(),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_command_queue_capacity",
        "Combined queue capacity of loaded model generation workers.",
        lives
            .iter()
            .map(|live| live.generation_service.command_queue_capacity() as u64)
            .sum(),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_active_streams",
        "Native streams owned by loaded model generation workers.",
        lives
            .iter()
            .map(|live| live.generation_service.active_streams() as u64)
            .sum(),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_buffered_stream_events",
        "Loaded model generation events buffered behind backpressured consumers.",
        lives
            .iter()
            .map(|live| live.generation_service.buffered_stream_events() as u64)
            .sum(),
    );
    append_counter(
        &mut body,
        "ax_engine_generation_saturated_commands_total",
        "Commands rejected because the bounded native generation queue was full.",
        metrics
            .generation_saturated_commands_total
            .load(Ordering::Relaxed),
    );
    append_counter(
        &mut body,
        "ax_engine_generation_stream_backlog_overflows_total",
        "Native streams cancelled after exceeding the bounded worker event backlog.",
        metrics
            .generation_stream_backlog_overflows_total
            .load(Ordering::Relaxed),
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_worker_ready",
        "Whether every loaded model generation worker can accept commands.",
        u64::from(!lives.is_empty() && lives.iter().all(|live| live.generation_service.is_ready())),
    );

    // Engine-step gauges come from the snapshot cached by the generation worker;
    // scraping must never call `step_report` itself
    // (that call advances the engine or consumes request stream chunks).
    // Each loaded model runs its own engine worker, so every series is
    // emitted per model (label `model`) plus an unlabeled aggregate for
    // dashboards written against the single-model layout.
    let loaded_model_ids = lives
        .iter()
        .map(|live| live.model_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut step_models = metrics.engine_step_gauges_per_model();
    step_models.retain(|(model_id, _)| loaded_model_ids.contains(model_id.as_str()));
    if !step_models.is_empty() {
        append_step_metric(
            &mut body,
            "ax_engine_steps_total",
            "Successful engine steps observed by the current server process (unlabeled: summed across loaded models).",
            "counter",
            &step_models,
            |step| step.steps_total,
        );
        append_step_metric(
            &mut body,
            "ax_engine_step_scheduled_requests",
            "Requests scheduled in the latest observed engine step (unlabeled: summed across loaded models).",
            "gauge",
            &step_models,
            |step| step.scheduled_requests,
        );
        append_step_metric(
            &mut body,
            "ax_engine_step_scheduled_tokens",
            "Tokens scheduled in the latest observed engine step (unlabeled: summed across loaded models).",
            "gauge",
            &step_models,
            |step| step.scheduled_tokens,
        );
        append_step_metric(
            &mut body,
            "ax_engine_step_kv_usage_blocks",
            "KV cache blocks used in the latest observed engine step (unlabeled: summed across loaded models).",
            "gauge",
            &step_models,
            |step| step.kv_usage_blocks,
        );
        append_step_metric(
            &mut body,
            "ax_engine_step_prefix_hits_total",
            "Prefix cache hits accumulated across observed engine steps (unlabeled: summed across loaded models).",
            "counter",
            &step_models,
            |step| step.prefix_hits_total,
        );
    }

    let mut arbiter_models = state.execution_arbiter_stats();
    arbiter_models.retain(|(model_id, _, _)| loaded_model_ids.contains(model_id.as_str()));
    if !arbiter_models.is_empty() {
        append_arbiter_metric(
            &mut body,
            "ax_engine_model_execution_turns_total",
            "Device execution turns acquired by model and work class.",
            "counter",
            &arbiter_models,
            |stats| stats.turns_total,
        );
        append_arbiter_metric(
            &mut body,
            "ax_engine_model_execution_wait_microseconds_total",
            "Total time spent waiting for the process-wide device arbiter.",
            "counter",
            &arbiter_models,
            |stats| stats.wait_us_total,
        );
        append_arbiter_metric(
            &mut body,
            "ax_engine_model_execution_wait_microseconds_max",
            "Longest observed wait for the process-wide device arbiter.",
            "gauge",
            &arbiter_models,
            |stats| stats.wait_us_max,
        );
        append_arbiter_metric(
            &mut body,
            "ax_engine_model_execution_hold_microseconds_total",
            "Total time spent holding the process-wide device arbiter.",
            "counter",
            &arbiter_models,
            |stats| stats.hold_us_total,
        );
        append_arbiter_metric(
            &mut body,
            "ax_engine_model_execution_hold_microseconds_max",
            "Longest observed process-wide device-arbiter turn.",
            "gauge",
            &arbiter_models,
            |stats| stats.hold_us_max,
        );
    }

    append_memory_metrics(&mut body, &lives, metrics);

    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
}

#[derive(Clone)]
struct ModelMemorySample {
    model_id: String,
    weight_artifact_bytes: u64,
    weight_artifact_available: u64,
    runner: Option<crate::app_state::ModelMemoryGauges>,
}

fn append_memory_metrics(
    body: &mut String,
    lives: &[crate::app_state::LiveState],
    metrics: &crate::app_state::ServerMetrics,
) {
    let runner_by_model = metrics
        .model_memory_gauges_per_model()
        .into_iter()
        .collect::<std::collections::BTreeMap<_, _>>();
    let samples = lives
        .iter()
        .map(|live| {
            let weight_artifact_bytes = live
                .session_config
                .mlx_model_artifacts_dir()
                .and_then(crate::model_load::model_weight_bytes);
            ModelMemorySample {
                model_id: live.model_id.as_ref().clone(),
                weight_artifact_bytes: weight_artifact_bytes.unwrap_or(0),
                weight_artifact_available: u64::from(weight_artifact_bytes.is_some()),
                runner: runner_by_model.get(live.model_id.as_ref()).copied(),
            }
        })
        .collect::<Vec<_>>();
    let active_bytes = mlx_sys::device_active_bytes();

    append_optional_gauge(
        body,
        "ax_engine_memory_mlx_active_bytes",
        "Process-wide active bytes measured by mlx_get_active_memory; not attributable to one model by MLX.",
        active_bytes,
    );
    append_optional_gauge(
        body,
        "ax_engine_memory_mlx_cache_bytes",
        "Process-wide reusable allocator-cache bytes measured by mlx_get_cache_memory.",
        mlx_sys::device_cache_bytes(),
    );
    append_optional_gauge(
        body,
        "ax_engine_memory_mlx_peak_bytes",
        "Process-wide peak active bytes measured by mlx_get_peak_memory.",
        mlx_sys::device_peak_bytes(),
    );
    append_optional_gauge(
        body,
        "ax_engine_memory_metal_recommended_working_set_bytes",
        "Metal recommended maximum working-set size used as the server memory budget.",
        mlx_sys::device_recommended_working_set_bytes(),
    );
    append_optional_gauge(
        body,
        "ax_engine_memory_host_resident_bytes",
        "Current process resident bytes from task_info; this is host RSS, not MLX-only memory.",
        mlx_sys::host_resident_bytes(),
    );

    append_model_gauge(
        body,
        "ax_engine_model_memory_weight_artifact_bytes",
        "Exact bytes of safetensors files in each loaded model artifact directory; an on-disk attribution input, not an allocator measurement.",
        &samples
            .iter()
            .map(|sample| (sample.model_id.clone(), sample.weight_artifact_bytes))
            .collect::<Vec<_>>(),
    );
    append_model_gauge(
        body,
        "ax_engine_model_memory_weight_artifact_available",
        "Whether exact safetensors artifact bytes are available for this model.",
        &samples
            .iter()
            .map(|sample| (sample.model_id.clone(), sample.weight_artifact_available))
            .collect::<Vec<_>>(),
    );
    append_model_gauge(
        body,
        "ax_engine_model_memory_kv_report_available",
        "Whether the model has produced a native runner KV memory report.",
        &samples
            .iter()
            .map(|sample| (sample.model_id.clone(), u64::from(sample.runner.is_some())))
            .collect::<Vec<_>>(),
    );

    let runner_value =
        |sample: &ModelMemorySample, value: fn(crate::app_state::ModelMemoryGauges) -> u64| {
            sample.runner.map(value).unwrap_or(0)
        };
    for (name, help, value) in [
        (
            "ax_engine_model_memory_kv_logical_bytes",
            "Logical KV bytes represented by request caches in the latest native runner report.",
            (|memory: crate::app_state::ModelMemoryGauges| memory.kv_logical_bytes)
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
        (
            "ax_engine_model_memory_kv_capacity_bytes",
            "Allocated contiguous KV capacity bytes in the latest native runner report; paged views may overlap the pool slab metric.",
            (|memory: crate::app_state::ModelMemoryGauges| memory.kv_capacity_bytes)
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
        (
            "ax_engine_model_memory_kv_linear_state_bytes",
            "Recurrent and convolution state bytes not included in attention KV capacity.",
            (|memory: crate::app_state::ModelMemoryGauges| memory.kv_linear_state_bytes)
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
        (
            "ax_engine_model_memory_kv_paged_pool_slab_bytes",
            "Real fixed-slab bytes reserved by a model's shared physical FA block pool.",
            (|memory: crate::app_state::ModelMemoryGauges| memory.kv_paged_pool_slab_bytes)
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
        (
            "ax_engine_model_memory_kv_physical_bytes",
            "Derived physical KV attribution: paged pool slabs or contiguous capacity, plus recurrent state, without double-counting paged views.",
            crate::app_state::ModelMemoryGauges::physical_kv_bytes
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
        (
            "ax_engine_model_memory_prefix_cache_payload_bytes",
            "Portable in-memory prefix-cache payload bytes reported by the model runner; excludes disk files.",
            (|memory: crate::app_state::ModelMemoryGauges| memory.prefix_cache_payload_bytes)
                as fn(crate::app_state::ModelMemoryGauges) -> u64,
        ),
    ] {
        append_model_gauge(
            body,
            name,
            help,
            &samples
                .iter()
                .map(|sample| (sample.model_id.clone(), runner_value(sample, value)))
                .collect::<Vec<_>>(),
        );
    }

    let attributed = samples
        .iter()
        .map(|sample| {
            (
                sample.model_id.clone(),
                sample.weight_artifact_bytes.saturating_add(
                    sample
                        .runner
                        .map(crate::app_state::ModelMemoryGauges::physical_kv_bytes)
                        .unwrap_or(0),
                ),
            )
        })
        .collect::<Vec<_>>();
    append_model_gauge(
        body,
        "ax_engine_model_memory_attributed_device_lower_bound_bytes",
        "Model-attributed lower bound from exact weight artifact bytes plus non-double-counted physical KV geometry; excludes graphs, temporary tensors, and allocator overhead.",
        &attributed,
    );
    for sample in &samples {
        if let Some(memory) = sample.runner {
            append_model_topology_info(body, &sample.model_id, memory);
        }
    }

    let attributed_total = attributed
        .iter()
        .map(|(_, bytes)| *bytes)
        .fold(0u64, u64::saturating_add);
    append_gauge(
        body,
        "ax_engine_memory_attributed_device_lower_bound_bytes",
        "Sum of per-model weight-artifact plus physical-KV lower-bound attribution.",
        attributed_total,
    );
    if let Some(active_bytes) = active_bytes {
        append_gauge(
            body,
            "ax_engine_memory_unattributed_active_bytes",
            "Measured MLX active bytes not covered by the model lower-bound attribution (graphs, temporaries, overhead, or unavailable artifacts).",
            active_bytes.saturating_sub(attributed_total),
        );
        append_gauge(
            body,
            "ax_engine_memory_attribution_excess_bytes",
            "Amount by which lower-bound attribution exceeds measured MLX active bytes; non-zero flags attribution assumptions that need review.",
            attributed_total.saturating_sub(active_bytes),
        );
        let coverage_basis_points = if active_bytes == 0 {
            0
        } else {
            ((u128::from(attributed_total) * 10_000) / u128::from(active_bytes)).min(10_000) as u64
        };
        append_gauge(
            body,
            "ax_engine_memory_attribution_coverage_basis_points",
            "Attributed lower-bound share of measured MLX active bytes, capped at 10000 basis points; inspect excess_bytes when capped.",
            coverage_basis_points,
        );
    }
}

fn append_model_topology_info(
    body: &mut String,
    model_id: &str,
    memory: crate::app_state::ModelMemoryGauges,
) {
    let name = "ax_engine_model_kv_topology_info";
    if !body.contains("# HELP ax_engine_model_kv_topology_info ") {
        body.push_str("# HELP ");
        body.push_str(name);
        body.push_str(
            " Typed hybrid-KV topology and rollback contract from the latest runner report.\n",
        );
        body.push_str("# TYPE ");
        body.push_str(name);
        body.push_str(" gauge\n");
    }
    body.push_str(name);
    body.push_str("{model=\"");
    body.push_str(&escape_label_value(model_id));
    body.push_str("\",attention_storage=\"");
    body.push_str(memory.attention_storage());
    body.push_str("\",sliding_storage=\"");
    body.push_str(memory.sliding_storage());
    body.push_str("\",recurrent_state=\"");
    body.push_str(if memory.linear_state_layers > 0 {
        "present"
    } else {
        "none"
    });
    body.push_str("\",rollback_strategy=\"");
    body.push_str(memory.rollback_strategy());
    body.push_str("\"} 1\n");
}

fn append_model_gauge(body: &mut String, name: &str, help: &str, values: &[(String, u64)]) {
    body.push_str("# HELP ");
    body.push_str(name);
    body.push(' ');
    body.push_str(help);
    body.push('\n');
    body.push_str("# TYPE ");
    body.push_str(name);
    body.push_str(" gauge\n");
    for (model_id, value) in values {
        body.push_str(name);
        body.push_str("{model=\"");
        body.push_str(&escape_label_value(model_id));
        body.push_str("\"} ");
        body.push_str(&value.to_string());
        body.push('\n');
    }
}

fn append_arbiter_metric(
    body: &mut String,
    name: &str,
    help: &str,
    metric_type: &str,
    values: &[(
        String,
        crate::generation::service::ExecutionWorkClass,
        crate::generation::service::ModelExecutionStats,
    )],
    value: impl Fn(&crate::generation::service::ModelExecutionStats) -> u64,
) {
    body.push_str("# HELP ");
    body.push_str(name);
    body.push(' ');
    body.push_str(help);
    body.push('\n');
    body.push_str("# TYPE ");
    body.push_str(name);
    body.push(' ');
    body.push_str(metric_type);
    body.push('\n');
    for (model_id, work_class, stats) in values {
        body.push_str(name);
        body.push_str("{model=\"");
        body.push_str(&escape_label_value(model_id));
        body.push_str("\",work_class=\"");
        body.push_str(work_class.as_str());
        body.push_str("\"} ");
        body.push_str(&value(stats).to_string());
        body.push('\n');
    }
}

/// One engine-step metric: HELP/TYPE once, then the unlabeled aggregate
/// followed by one `model`-labeled series per loaded model.
fn append_step_metric(
    body: &mut String,
    name: &str,
    help: &str,
    metric_type: &str,
    step_models: &[(String, crate::app_state::EngineStepGauges)],
    value: impl Fn(&crate::app_state::EngineStepGauges) -> u64,
) {
    body.push_str("# HELP ");
    body.push_str(name);
    body.push(' ');
    body.push_str(help);
    body.push('\n');
    body.push_str("# TYPE ");
    body.push_str(name);
    body.push(' ');
    body.push_str(metric_type);
    body.push('\n');
    let total: u64 = step_models.iter().map(|(_, step)| value(step)).sum();
    body.push_str(name);
    body.push(' ');
    body.push_str(&total.to_string());
    body.push('\n');
    for (model_id, step) in step_models {
        body.push_str(name);
        body.push_str("{model=\"");
        body.push_str(&escape_label_value(model_id));
        body.push_str("\"} ");
        body.push_str(&value(step).to_string());
        body.push('\n');
    }
}

fn escape_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn append_counter(body: &mut String, name: &str, help: &str, value: u64) {
    append_metric(body, name, help, "counter", value);
}

fn append_gauge(body: &mut String, name: &str, help: &str, value: u64) {
    append_metric(body, name, help, "gauge", value);
}

fn append_optional_gauge(body: &mut String, name: &str, help: &str, value: Option<u64>) {
    if let Some(value) = value {
        append_gauge(body, name, help, value);
    }
}

fn append_metric(body: &mut String, name: &str, help: &str, metric_type: &str, value: u64) {
    body.push_str("# HELP ");
    body.push_str(name);
    body.push(' ');
    body.push_str(help);
    body.push('\n');
    body.push_str("# TYPE ");
    body.push_str(name);
    body.push(' ');
    body.push_str(metric_type);
    body.push('\n');
    body.push_str(name);
    body.push(' ');
    body.push_str(&value.to_string());
    body.push('\n');
}
