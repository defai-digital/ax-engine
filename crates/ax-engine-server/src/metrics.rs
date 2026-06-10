use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;

pub(crate) async fn prometheus_metrics(State(state): State<AppState>) -> Response {
    let metrics = state.metrics.as_ref();
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

    // Engine-step gauges come from the snapshot cached by the endpoints that
    // actually drive steps; scraping must never call `step_report` itself
    // (that call advances the engine or consumes request stream chunks).
    if let Some(step) = metrics.engine_step_gauges() {
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
