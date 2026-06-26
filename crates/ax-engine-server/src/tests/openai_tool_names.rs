use crate::openai::chat_requests::{
    render_openai_chat_prompt, render_openai_chat_prompt_with_tools,
};
use crate::openai::schema::OpenAiChatMessage;

use axum::http::StatusCode;
use serde_json::json;

#[test]
fn openai_chat_prompt_renderer_rejects_invalid_tool_declaration_name() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let error = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3-Coder-Next-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read file",
                    "parameters": {"type": "object"}
                }
            }
        ])),
        Some(&json!("auto")),
    )
    .expect_err("invalid tool declaration names should fail closed");

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(error.1.error.code.as_deref(), Some("invalid_request"));
    assert!(error.1.error.message.contains("tools[0].function.name"));
}

#[test]
fn openai_chat_prompt_renderer_rejects_invalid_tool_choice_name() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"}
    ]))
    .expect("sample messages should deserialize");

    let error = render_openai_chat_prompt_with_tools(
        "mlx-community/Qwen3-Coder-Next-4bit",
        &messages,
        Some(&json!([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"}
                }
            }
        ])),
        Some(&json!({"type": "function", "function": {"name": "read file"}})),
    )
    .expect_err("invalid forced tool_choice names should fail closed");

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(error.1.error.code.as_deref(), Some("invalid_request"));
    assert!(error.1.error.message.contains("tool_choice.function.name"));
}

#[test]
fn openai_chat_prompt_renderer_rejects_invalid_history_tool_call_name() {
    let messages: Vec<OpenAiChatMessage> = serde_json::from_value(json!([
        {"role": "user", "content": "Read README.md"},
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_123", "content": "AX Engine"}
    ]))
    .expect("tool replay messages should deserialize");

    let error = render_openai_chat_prompt("mlx-community/Qwen3-Coder-Next-4bit", &messages)
        .expect_err("invalid history tool call names should fail closed");

    assert_eq!(error.0, StatusCode::BAD_REQUEST);
    assert_eq!(error.1.error.code.as_deref(), Some("invalid_request"));
    assert!(
        error
            .1
            .error
            .message
            .contains("messages[1].tool_calls[0].function.name")
    );
}
