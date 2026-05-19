// `tonic::Status` is part of the tonic trait boundary and is returned directly
// from request-building helpers so call sites can preserve gRPC status codes.
#![allow(clippy::result_large_err)]

use ax_engine_sdk::{GenerateRequest, GenerateSampling};
use tonic::Status;

use super::proto;
use crate::{app_state::AppState, chat};

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

pub(super) fn build_chat_generate_request(
    state: &AppState,
    req: &proto::ChatCompletionRequest,
) -> Result<GenerateRequest, Status> {
    let pairs: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let input_text = render_grpc_chat_prompt(state.model_id.as_ref(), &pairs)
        .map_err(Status::invalid_argument)?;
    let max_output_tokens = if req.max_tokens == 0 {
        256
    } else {
        req.max_tokens
    };
    let sampling = GenerateSampling {
        temperature: req.temperature,
        top_p: 1.0,
        top_k: 0,
        min_p: None,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        seed: req.seed,
        deterministic: None,
        ignore_eos: false,
    };
    let stop_sequences = grpc_chat_stop_sequences(state.model_id.as_ref(), req.stop.clone());

    Ok(GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens: Vec::new(),
        input_text: Some(input_text),
        max_output_tokens,
        sampling,
        stop_sequences,
        metadata: None,
    })
}

pub(super) fn build_completion_generate_request(
    state: &AppState,
    req: &proto::CompletionRequest,
) -> GenerateRequest {
    let max_output_tokens = if req.max_tokens == 0 {
        256
    } else {
        req.max_tokens
    };
    let sampling = GenerateSampling {
        temperature: req.temperature,
        top_p: 1.0,
        top_k: 0,
        min_p: None,
        repetition_penalty: 1.0,
        repetition_context_size: None,
        seed: req.seed,
        deterministic: None,
        ignore_eos: false,
    };
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens: Vec::new(),
        input_text: Some(req.prompt.clone()),
        max_output_tokens,
        sampling,
        stop_sequences: req.stop.clone(),
        metadata: None,
    }
}

pub(super) fn grpc_embedding_prompt_tokens(input: &[u32]) -> u32 {
    input.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user(msg: &str) -> Vec<(String, String)> {
        vec![("user".to_string(), msg.to_string())]
    }

    #[test]
    fn grpc_chat_prompt_keeps_thinking_open_for_qwen_reasoning_models() {
        // Mirror of `native_chat_renderer_keeps_thinking_open_for_qwen_reasoning_models`.
        // Without this branch the gRPC path would reproduce #13: enable_thinking=false
        // pre-closes the `<think>` block, which causes Qwen3.6 / Qwen3-Next /
        // Qwen3-Coder-Next to truncate or loop on reasoning prompts.
        let thinking =
            render_grpc_chat_prompt("Qwen3.6-35B-A3B-4bit", &user("hi")).expect("render");
        assert!(
            thinking.ends_with("<|im_start|>assistant\n<think>\n"),
            "thinking-enabled suffix should be `<think>\\n` only: {thinking}"
        );
        assert!(
            !thinking.contains("</think>"),
            "thinking-enabled suffix must not pre-close the think block: {thinking}"
        );

        let coder = render_grpc_chat_prompt("Qwen3-Coder-Next-4bit", &user("hi")).expect("render");
        assert!(
            coder.ends_with("<|im_start|>assistant\n<think>\n"),
            "Coder-Next must also leave thinking open: {coder}"
        );
    }

    #[test]
    fn grpc_chat_prompt_keeps_no_thinking_for_older_qwen() {
        let prompt = render_grpc_chat_prompt("qwen3", &user("hi")).expect("render");
        assert!(
            prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            "non-thinking Qwen must keep the pre-closed think block: {prompt}"
        );
    }

    #[test]
    fn grpc_chat_prompt_rejects_empty_messages() {
        let err = render_grpc_chat_prompt("qwen3", &[]).expect_err("empty must fail");
        assert!(err.contains("at least one message"));
    }

    #[test]
    fn grpc_embedding_usage_counts_input_tokens() {
        let input_tokens = [101, 102, 103];
        let embedding_width = 768_u32;
        assert_eq!(grpc_embedding_prompt_tokens(&input_tokens), 3);
        assert_ne!(grpc_embedding_prompt_tokens(&input_tokens), embedding_width);
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
             <|turn>model\n",
        );
    }

    #[test]
    fn grpc_chat_prompt_preserves_glm47_tool_observation_shape() {
        // Mirror of `openai_glm_prompt_renderer_preserves_tool_observation_shape`.
        // GLM4.7 needs tool/function roles routed to observation/tool_response tags
        // and assistant turns to include `</think>` after the tag.
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
