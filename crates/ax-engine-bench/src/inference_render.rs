use ax_engine_sdk::{GenerateResponse, GenerateStreamEvent};

use crate::error::CliError;
use crate::labels::{
    generate_finish_reason_label, generate_status_label, optional_route_label, request_state_label,
    selected_backend_label, support_tier_label,
};

pub(crate) fn render_generate_response(
    response: &GenerateResponse,
    json_output: bool,
) -> Result<String, CliError> {
    if json_output {
        return serde_json::to_string_pretty(response)
            .map(|json| format!("{json}\n"))
            .map_err(|error| {
                CliError::Runtime(format!("failed to serialize generate response: {error}"))
            });
    }

    if let Some(output_text) = response.output_text.as_deref() {
        let mut rendered = output_text.to_string();
        if !rendered.ends_with('\n') {
            rendered.push('\n');
        }
        rendered.push_str(&format_generate_metadata_suffix(response));
        return Ok(rendered);
    }

    let rendered_tokens = response
        .output_tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let mut rendered = format!("{rendered_tokens}\n");
    rendered.push_str(&format_generate_metadata_suffix(response));
    Ok(rendered)
}

pub(crate) fn render_stream_event(
    event: &GenerateStreamEvent,
    json_output: bool,
) -> Result<String, CliError> {
    if json_output {
        return serde_json::to_string(event)
            .map(|json| format!("{json}\n"))
            .map_err(|error| {
                CliError::Runtime(format!("failed to serialize stream event: {error}"))
            });
    }

    let rendered = match event {
        GenerateStreamEvent::Request(payload) => format!(
            "request id={} backend={} support_tier={} state={} execution_plan={}\n",
            payload.request.request_id,
            selected_backend_label(payload.runtime.selected_backend),
            support_tier_label(payload.runtime.support_tier),
            request_state_label(payload.request.state),
            optional_route_label(payload.request.route.execution_plan.as_deref()),
        ),
        GenerateStreamEvent::Step(payload) => {
            let finish_reason = payload
                .request
                .finish_reason
                .map(generate_finish_reason_label)
                .unwrap_or("none");
            let delta_text = payload.delta_text.as_deref().unwrap_or("");
            format!(
                "step id={} state={} execution_plan={} delta_tokens={:?} delta_token_logprobs={:?} delta_text={delta_text:?} total_output_tokens={} finish_reason={finish_reason}\n",
                payload.request.request_id,
                request_state_label(payload.request.state),
                optional_route_label(payload.request.route.execution_plan.as_deref()),
                payload.delta_tokens,
                payload.delta_token_logprobs,
                payload.request.output_tokens.len(),
            )
        }
        GenerateStreamEvent::Response(payload) => {
            let finish_reason = payload
                .response
                .finish_reason
                .map(generate_finish_reason_label)
                .unwrap_or("none");
            if let Some(output_text) = payload.response.output_text.as_deref() {
                format!(
                    "response id={} status={} finish_reason={} execution_plan={} output_text={output_text:?} output_token_logprobs={:?}\n",
                    payload.response.request_id,
                    generate_status_label(payload.response.status),
                    finish_reason,
                    optional_route_label(payload.response.route.execution_plan.as_deref()),
                    payload.response.output_token_logprobs,
                )
            } else {
                format!(
                    "response id={} status={} finish_reason={} execution_plan={} output_tokens={:?} output_token_logprobs={:?}\n",
                    payload.response.request_id,
                    generate_status_label(payload.response.status),
                    finish_reason,
                    optional_route_label(payload.response.route.execution_plan.as_deref()),
                    payload.response.output_tokens,
                    payload.response.output_token_logprobs,
                )
            }
        }
    };

    Ok(rendered)
}

fn format_generate_metadata_suffix(response: &GenerateResponse) -> String {
    let finish_reason = response
        .finish_reason
        .map(generate_finish_reason_label)
        .unwrap_or("none");
    let execution_plan = optional_route_label(response.route.execution_plan.as_deref());
    format!(
        "request_id={}\nstatus={}\nfinish_reason={}\nexecution_plan={}\noutput_token_logprobs={:?}\n",
        response.request_id,
        generate_status_label(response.status),
        finish_reason,
        execution_plan,
        response.output_token_logprobs,
    )
}
