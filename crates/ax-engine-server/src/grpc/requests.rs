// `tonic::Status` is part of the tonic trait boundary and is returned directly
// from request-building helpers so call sites can preserve gRPC status codes.
#![allow(clippy::result_large_err)]

use ax_engine_sdk::{GenerateRequest, GenerateSampling};
use axum::Json;
use axum::http::StatusCode;
use tonic::Status;

use super::proto;
use crate::app_state::LiveState;
use crate::chat;
use crate::errors::ErrorResponse;
use crate::openai::requests::{
    default_native_mlx_openai_repetition_penalty, tokenize_native_mlx_text_input,
};

/// Render a chat prompt from plain (role, content) pairs.
fn render_grpc_chat_prompt(
    model_id: &str,
    messages: &[(String, String)],
) -> Result<String, String> {
    chat::render_prompt(model_id, messages)
}

/// Chat stop sequences for the gRPC service.
fn grpc_chat_stop_sequences(model_id: &str, stop: Vec<String>) -> Vec<String> {
    chat::stop_sequences(model_id, stop)
}

/// Map OpenAI HTTP error responses onto gRPC status codes for shared helpers.
pub(super) fn openai_error_to_status(
    (status, Json(body)): (StatusCode, Json<ErrorResponse>),
) -> Status {
    let message = body.error.message;
    match status {
        StatusCode::BAD_REQUEST => Status::invalid_argument(message),
        StatusCode::NOT_FOUND => Status::not_found(message),
        StatusCode::TOO_MANY_REQUESTS => Status::resource_exhausted(message),
        StatusCode::SERVICE_UNAVAILABLE => Status::unavailable(message),
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => Status::permission_denied(message),
        _ => Status::internal(message),
    }
}

/// Tokenize `input_text` for native MLX when required (parity with OpenAI HTTP).
pub(super) fn finalize_native_generate_request(
    live: &LiveState,
    mut request: GenerateRequest,
) -> Result<GenerateRequest, Status> {
    let (input_tokens, input_text) = tokenize_native_mlx_text_input(
        live,
        request.input_tokens,
        request.input_text,
    )
    .map_err(openai_error_to_status)?;
    request.input_tokens = input_tokens;
    request.input_text = input_text;
    Ok(request)
}

pub(super) fn build_chat_generate_request(
    live: &LiveState,
    req: &proto::ChatCompletionRequest,
) -> Result<GenerateRequest, Status> {
    chat::validate_native_chat_artifact(
        live.model_id.as_ref(),
        live.session_config.mlx_model_artifacts_dir.as_deref(),
    )
    .map_err(Status::invalid_argument)?;
    let pairs: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let input_text = render_grpc_chat_prompt(live.model_id.as_ref(), &pairs)
        .map_err(Status::invalid_argument)?;
    let max_output_tokens = if req.max_tokens == 0 {
        256
    } else {
        req.max_tokens
    };
    let default_repetition_penalty =
        default_native_mlx_openai_repetition_penalty(live, req.temperature);
    let sampling = GenerateSampling {
        temperature: req.temperature,
        top_p: 1.0,
        top_k: 0,
        min_p: None,
        repetition_penalty: default_repetition_penalty,
        repetition_context_size: None,
        seed: req.seed,
        deterministic: None,
        ignore_eos: false,
    };
    let stop_sequences = grpc_chat_stop_sequences(live.model_id.as_ref(), req.stop.clone());

    finalize_native_generate_request(
        live,
        GenerateRequest {
            model_id: live.model_id.to_string(),
            input_tokens: Vec::new(),
            input_text: Some(input_text),
            multimodal_inputs: Default::default(),
            max_output_tokens,
            sampling,
            stop_sequences,
            metadata: None,
        },
    )
}

pub(super) fn build_completion_generate_request(
    live: &LiveState,
    req: &proto::CompletionRequest,
) -> Result<GenerateRequest, Status> {
    let max_output_tokens = if req.max_tokens == 0 {
        256
    } else {
        req.max_tokens
    };
    let default_repetition_penalty =
        default_native_mlx_openai_repetition_penalty(live, req.temperature);
    let sampling = GenerateSampling {
        temperature: req.temperature,
        top_p: 1.0,
        top_k: 0,
        min_p: None,
        repetition_penalty: default_repetition_penalty,
        repetition_context_size: None,
        seed: req.seed,
        deterministic: None,
        ignore_eos: false,
    };
    finalize_native_generate_request(
        live,
        GenerateRequest {
            model_id: live.model_id.to_string(),
            input_tokens: Vec::new(),
            input_text: Some(req.prompt.clone()),
            multimodal_inputs: Default::default(),
            max_output_tokens,
            sampling,
            stop_sequences: req.stop.clone(),
            metadata: None,
        },
    )
}

pub(super) fn grpc_embedding_prompt_tokens(inputs: &[Vec<u32>]) -> u32 {
    inputs.iter().map(Vec::len).sum::<usize>() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_state::{AppState, build_app_state};
    use crate::args::ServerArgs;
    use clap::Parser;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn user(msg: &str) -> Vec<(String, String)> {
        vec![("user".to_string(), msg.to_string())]
    }

    fn test_app_state(model_id: &str, artifact_dir: &std::path::Path) -> AppState {
        let args = ServerArgs::parse_from([
            "ax-engine-server",
            "--model-id",
            model_id,
            // Explicit llama.cpp tier: support tiers are MLX-direct by
            // default (see "Make server direct-first by default"), which
            // makes construction eagerly require a real model-manifest.json
            // in --mlx-model-artifacts-dir. This test's artifact_dir is
            // deliberately manifest-less (it only wants to verify the
            // chat_template.jinja-specific error surfaced later, inside
            // build_chat_generate_request), so it must opt out of the
            // direct-first default to keep session construction lazy about
            // that directory's contents — matching the delegated
            // --llama-server-url this test also configures.
            "--support-tier",
            "llama-cpp",
            "--llama-server-url",
            "http://127.0.0.1:1",
            "--mlx-model-artifacts-dir",
            artifact_dir.to_str().expect("artifact dir should be UTF-8"),
        ]);
        let session_config = args.session_config().expect("session config should build");
        build_app_state(args.model_id.clone(), session_config).expect("app state should build")
    }

    #[test]
    fn grpc_chat_prompt_disables_thinking_for_qwen_reasoning_models() {
        // Mirror of the native OpenAI chat renderer: short chat responses should
        // not spend their output budget on visible Qwen thinking text.
        let no_thinking =
            render_grpc_chat_prompt("Qwen3.6-35B-A3B-4bit", &user("hi")).expect("render");
        assert!(
            no_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "thinking-disabled suffix should pre-close the think block: {no_thinking}"
        );

        let coder = render_grpc_chat_prompt("Qwen3-Coder-Next-4bit", &user("hi")).expect("render");
        assert!(
            coder.ends_with("<|im_start|>assistant\n"),
            "Coder-Next follows its non-thinking-only template without pre-closing a think block: {coder}"
        );
    }

    #[test]
    fn grpc_chat_prompt_keeps_no_think_suffix_for_ax_qwen3_alias() {
        let prompt = render_grpc_chat_prompt("qwen3", &user("hi")).expect("render");
        assert!(
            prompt.ends_with("<|im_start|>assistant\n"),
            "the ax qwen3 alias follows Qwen3-Coder-Next's no-think template: {prompt}"
        );
    }

    #[test]
    fn grpc_chat_prompt_rejects_empty_messages() {
        let err = render_grpc_chat_prompt("qwen3", &[]).expect_err("empty must fail");
        assert!(err.contains("at least one message"));
    }

    #[tokio::test]
    async fn grpc_chat_request_rejects_gemma4_artifact_without_chat_template() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("ax-engine-grpc-gemma4-base-artifact-{unique}"));
        fs::create_dir_all(&artifact_dir).expect("artifact dir should create");
        let state = test_app_state("gemma4", &artifact_dir);
        let req = proto::ChatCompletionRequest {
            model: "gemma4".to_string(),
            messages: vec![proto::ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: 8,
            temperature: 0.0,
            seed: 0,
            stop: Vec::new(),
        };

        let live = state.snapshot();
        let error = build_chat_generate_request(&live, &req)
            .expect_err("Gemma4 base artifact should fail closed for gRPC chat");
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
        assert!(error.message().contains("chat_template.jinja"));

        fs::remove_dir_all(artifact_dir).expect("artifact dir should clean up");
    }

    #[test]
    fn grpc_embedding_usage_counts_input_tokens() {
        let input_tokens = vec![vec![101, 102, 103], vec![201, 202]];
        let embedding_width = 768_u32;
        assert_eq!(grpc_embedding_prompt_tokens(&input_tokens), 5);
        assert_ne!(grpc_embedding_prompt_tokens(&input_tokens), embedding_width);
    }

    #[tokio::test]
    async fn grpc_chat_request_tokenizes_text_for_native_mlx() {
        // Regression: native MLX rejects bare input_text; gRPC chat must
        // tokenize like OpenAI HTTP before generate.
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let artifact_dir =
            std::env::temp_dir().join(format!("ax-engine-grpc-mlx-tokenize-{unique}"));
        fs::create_dir_all(&artifact_dir).expect("artifact dir should create");
        fs::write(
            artifact_dir.join("config.json"),
            r#"{"eos_token_id":2}"#,
        )
        .expect("config");
        fs::write(
            artifact_dir.join("tokenizer.json"),
            r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "<|im_start|>": 2,
      "user": 3,
      "assistant": 4,
      "im_end": 5
    },
    "unk_token": "[UNK]"
  }
}"#,
        )
        .expect("tokenizer");

        let state = test_app_state("qwen3", &artifact_dir);
        let mut live = state.snapshot();
        live.runtime_report.selected_backend = ax_engine_sdk::SelectedBackend::Mlx;
        live.session_config = std::sync::Arc::new(
            live.session_config
                .as_ref()
                .clone()
                .with_mlx_model_artifacts_dir(&artifact_dir),
        );

        let req = proto::ChatCompletionRequest {
            model: "qwen3".to_string(),
            messages: vec![proto::ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            max_tokens: 8,
            temperature: 0.0,
            seed: 0,
            stop: Vec::new(),
        };

        let generate = build_chat_generate_request(&live, &req)
            .expect("native MLX gRPC chat must tokenize successfully");
        assert!(
            !generate.input_tokens.is_empty(),
            "expected pre-tokenized input_tokens, got empty"
        );
        assert!(
            generate.input_text.is_none(),
            "native MLX must clear input_text after tokenization"
        );
        // Shared OpenAI MLX policy: qwen at temperature 0 uses 1.1.
        assert_eq!(generate.sampling.repetition_penalty, 1.1);

        let _ = fs::remove_dir_all(artifact_dir);
    }

    #[tokio::test]
    async fn grpc_completion_request_tokenizes_text_for_native_mlx() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be valid")
            .as_nanos();
        let artifact_dir = std::env::temp_dir()
            .join(format!("ax-engine-grpc-mlx-completion-tokenize-{unique}"));
        fs::create_dir_all(&artifact_dir).expect("artifact dir should create");
        fs::write(
            artifact_dir.join("config.json"),
            r#"{"eos_token_id":2}"#,
        )
        .expect("config");
        fs::write(
            artifact_dir.join("tokenizer.json"),
            r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "hello": 1,
      "world": 2
    },
    "unk_token": "[UNK]"
  }
}"#,
        )
        .expect("tokenizer");

        let state = test_app_state("completion-model", &artifact_dir);
        let mut live = state.snapshot();
        live.runtime_report.selected_backend = ax_engine_sdk::SelectedBackend::Mlx;
        live.session_config = std::sync::Arc::new(
            live.session_config
                .as_ref()
                .clone()
                .with_mlx_model_artifacts_dir(&artifact_dir),
        );

        let req = proto::CompletionRequest {
            model: "completion-model".to_string(),
            prompt: "hello world".to_string(),
            max_tokens: 8,
            temperature: 0.0,
            seed: 0,
            stop: Vec::new(),
        };

        let generate = build_completion_generate_request(&live, &req)
            .expect("native MLX gRPC completion must tokenize");
        assert_eq!(generate.input_tokens, vec![1, 2]);
        assert!(generate.input_text.is_none());

        let _ = fs::remove_dir_all(artifact_dir);
    }

    #[test]
    fn grpc_chat_prompt_closes_gemma4_turns() {
        // Without `<turn|>\n` after content, Gemma4 sees a single unterminated turn
        // and continues the user's message instead of producing an assistant reply.
        let messages = vec![
            ("user".to_string(), "hello".to_string()),
            ("assistant".to_string(), "hi".to_string()),
            ("user".to_string(), "again".to_string()),
        ];
        let prompt = render_grpc_chat_prompt("gemma-4-e2b", &messages).expect("render");
        assert_eq!(
            prompt,
            "<bos>\
             <|turn>user\nhello<turn|>\n\
             <|turn>model\nhi<turn|>\n\
             <|turn>user\nagain<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>",
        );
    }

    #[test]
    fn grpc_chat_prompt_preserves_glm47_tool_observation_shape() {
        // Mirror of `openai_glm_prompt_renderer_preserves_tool_observation_shape`.
        // GLM4.7 needs tool/function roles routed to observation/tool_response tags
        // and assistant turns to close thinking with the tokenizer-template suffix.
        let messages = vec![
            ("user".to_string(), "call tool".to_string()),
            (
                "assistant".to_string(),
                "<tool_call>x</tool_call>".to_string(),
            ),
            ("tool".to_string(), "tool result".to_string()),
            ("user".to_string(), "continue".to_string()),
        ];
        let prompt =
            render_grpc_chat_prompt("mlx-community/GLM-4.7-Flash-4bit", &messages).expect("render");
        assert_eq!(
            prompt,
            "[gMASK]<sop>\
             <|user|>call tool\
             <|assistant|></think><tool_call>x</tool_call>\
             <|observation|><tool_response>tool result</tool_response>\
             <|user|>continue\
             <|assistant|></think>",
        );
    }
}
