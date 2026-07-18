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

    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
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
