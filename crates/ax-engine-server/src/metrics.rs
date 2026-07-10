use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;

pub(crate) async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let metrics = state.metrics.as_ref();
    let live = state.snapshot();
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
        "Commands and active streams owned by the current model generation worker.",
        live.generation_service.pending_jobs() as u64,
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_commands_queued",
        "Commands waiting in the current model generation's bounded worker queue.",
        live.generation_service.queued_commands() as u64,
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_command_queue_capacity",
        "Maximum commands that can wait in the current model generation worker queue.",
        live.generation_service.command_queue_capacity() as u64,
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_active_streams",
        "Native streams owned by the current model generation worker.",
        live.generation_service.active_streams() as u64,
    );
    append_gauge(
        &mut body,
        "ax_engine_generation_buffered_stream_events",
        "Current model generation events buffered behind backpressured consumers.",
        live.generation_service.buffered_stream_events() as u64,
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
        "Whether the current model generation worker can accept commands.",
        u64::from(live.generation_service.is_ready()),
    );

    // Engine-step gauges come from the snapshot cached by the generation worker;
    // scraping must never call `step_report` itself
    // (that call advances the engine or consumes request stream chunks).
    if let Some(step) = metrics.engine_step_gauges() {
        append_counter(
            &mut body,
            "ax_engine_steps_total",
            "Successful engine steps observed by the current server process.",
            step.steps_total,
        );
        append_gauge(
            &mut body,
            "ax_engine_step_scheduled_requests",
            "Requests scheduled in the latest observed engine step.",
            step.scheduled_requests,
        );
        append_gauge(
            &mut body,
            "ax_engine_step_scheduled_tokens",
            "Tokens scheduled in the latest observed engine step.",
            step.scheduled_tokens,
        );
        append_gauge(
            &mut body,
            "ax_engine_step_kv_usage_blocks",
            "KV cache blocks used in the latest observed engine step.",
            step.kv_usage_blocks,
        );
        append_counter(
            &mut body,
            "ax_engine_step_prefix_hits_total",
            "Prefix cache hits accumulated across observed engine steps.",
            step.prefix_hits_total,
        );
    }

    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
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
