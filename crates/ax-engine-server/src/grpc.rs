// `tonic::Status` is ~176 bytes — clippy::result_large_err fires on every helper
// returning `Result<_, Status>`. The trait surface comes from tonic and the
// helpers mirror that surface, so boxing per-call doesn't pay for itself. Allow
// at the module level rather than peppering individual functions.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use ax_engine_sdk::{EmbeddingPooling, GenerateRequest, GenerateSampling};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{app_state::AppState, chat};

mod conversions;
mod streams;

// Include the tonic-generated server code.
pub mod proto {
    tonic::include_proto!("ax_engine.v1");
}

use conversions::{finish_reason_str, proto_to_generate_request, sdk_response_to_proto, unix_now};
use proto::ax_engine_server::AxEngine;
use streams::{
    ChatChunkStream, CompletionChunkStream, GenerateEventStream, build_grpc_stream_state,
    run_blocking, spawn_grpc_chat_stream, spawn_grpc_completion_stream, spawn_grpc_generate_stream,
};

const GRPC_CHANNEL_CAPACITY: usize = 128;

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

// ─── Service struct ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AxEngineGrpcService {
    state: AppState,
}

impl AxEngineGrpcService {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    pub fn into_server(self) -> proto::ax_engine_server::AxEngineServer<Self> {
        proto::ax_engine_server::AxEngineServer::new(self)
    }
}

// ─── Request builders ─────────────────────────────────────────────────────────

fn build_chat_generate_request(
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

fn build_completion_generate_request(
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

// ─── Service implementation ───────────────────────────────────────────────────

#[tonic::async_trait]
impl AxEngine for AxEngineGrpcService {
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        Ok(Response::new(proto::HealthResponse {
            status: "ok".to_string(),
            service: "ax-engine-server".to_string(),
            model_id: self.state.model_id.to_string(),
        }))
    }

    async fn models(
        &self,
        _request: Request<proto::ModelsRequest>,
    ) -> Result<Response<proto::ModelsResponse>, Status> {
        Ok(Response::new(proto::ModelsResponse {
            object: "list".to_string(),
            data: vec![proto::ModelCard {
                id: self.state.model_id.to_string(),
                object: "model".to_string(),
                owned_by: "ax-engine-v4".to_string(),
            }],
        }))
    }

    // ── Generate unary ────────────────────────────────────────────────────────

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<proto::GenerateResponse>, Status> {
        let req = proto_to_generate_request(&self.state, request.into_inner());
        let request_id = self.state.allocate_request_id();
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let response = run_blocking(move || ctx.generate_with_request_id(request_id, req)).await?;
        Ok(Response::new(sdk_response_to_proto(response)))
    }

    // ── StreamGenerate ────────────────────────────────────────────────────────

    type StreamGenerateStream = GenerateEventStream;

    async fn stream_generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let req = proto_to_generate_request(&self.state, request.into_inner());
        let (ss, ctx) = build_grpc_stream_state(&self.state, req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_generate_stream(ss, tx, ctx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── ChatCompletion unary ──────────────────────────────────────────────────

    async fn chat_completion(
        &self,
        request: Request<proto::ChatCompletionRequest>,
    ) -> Result<Response<proto::ChatCompletionResponse>, Status> {
        let req = request.into_inner();
        let generate_req = build_chat_generate_request(&self.state, &req)?;
        let request_id = self.state.allocate_request_id();
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let r =
            run_blocking(move || ctx.generate_with_request_id(request_id, generate_req)).await?;
        let content = r.output_text.unwrap_or_default();
        let finish_reason = r.finish_reason.map(finish_reason_str).unwrap_or_default();
        Ok(Response::new(proto::ChatCompletionResponse {
            id: format!("chatcmpl-{}", r.request_id),
            object: "chat.completion".to_string(),
            created: unix_now(),
            model: r.model_id,
            choices: vec![proto::ChatCompletionChoice {
                index: 0,
                message: Some(proto::ChatMessage {
                    role: "assistant".to_string(),
                    content,
                }),
                finish_reason,
            }],
            usage: Some(proto::TokenUsage {
                prompt_tokens: r.prompt_tokens.len() as u32,
                completion_tokens: r.output_tokens.len() as u32,
                total_tokens: (r.prompt_tokens.len() + r.output_tokens.len()) as u32,
            }),
        }))
    }

    // ── StreamChatCompletion ──────────────────────────────────────────────────

    type StreamChatCompletionStream = ChatChunkStream;

    async fn stream_chat_completion(
        &self,
        request: Request<proto::ChatCompletionRequest>,
    ) -> Result<Response<Self::StreamChatCompletionStream>, Status> {
        let req = request.into_inner();
        let model_id = self.state.model_id.to_string();
        let generate_req = build_chat_generate_request(&self.state, &req)?;
        let (ss, ctx) = build_grpc_stream_state(&self.state, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_chat_stream(ss, model_id, tx, ctx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Completion unary ──────────────────────────────────────────────────────

    async fn completion(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<proto::CompletionResponse>, Status> {
        let req = request.into_inner();
        let generate_req = build_completion_generate_request(&self.state, &req);
        let request_id = self.state.allocate_request_id();
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let r =
            run_blocking(move || ctx.generate_with_request_id(request_id, generate_req)).await?;
        let text = r.output_text.unwrap_or_default();
        let finish_reason = r.finish_reason.map(finish_reason_str).unwrap_or_default();
        Ok(Response::new(proto::CompletionResponse {
            id: format!("cmpl-{}", r.request_id),
            object: "text_completion".to_string(),
            created: unix_now(),
            model: r.model_id,
            choices: vec![proto::CompletionChoice {
                index: 0,
                text,
                finish_reason,
            }],
            usage: Some(proto::TokenUsage {
                prompt_tokens: r.prompt_tokens.len() as u32,
                completion_tokens: r.output_tokens.len() as u32,
                total_tokens: (r.prompt_tokens.len() + r.output_tokens.len()) as u32,
            }),
        }))
    }

    // ── StreamCompletion ──────────────────────────────────────────────────────

    type StreamCompletionStream = CompletionChunkStream;

    async fn stream_completion(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<Self::StreamCompletionStream>, Status> {
        let req = request.into_inner();
        let model_id = self.state.model_id.to_string();
        let generate_req = build_completion_generate_request(&self.state, &req);
        let (ss, ctx) = build_grpc_stream_state(&self.state, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_completion_stream(ss, model_id, tx, ctx);
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    async fn embeddings(
        &self,
        request: Request<proto::EmbeddingsRequest>,
    ) -> Result<Response<proto::EmbeddingsResponse>, Status> {
        let req = request.into_inner();
        let model_id = self.state.model_id.to_string();
        let pooling = match req.pooling.as_str() {
            "mean" => EmbeddingPooling::Mean,
            "cls" => EmbeddingPooling::Cls,
            _ => EmbeddingPooling::Last,
        };
        let prompt_tokens = grpc_embedding_prompt_tokens(&req.input);
        let embedding = self
            .state
            .embedding_batcher
            .embed(req.input, pooling, req.normalize)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(proto::EmbeddingsResponse {
            object: "list".to_string(),
            data: vec![proto::EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: 0,
            }],
            model: model_id,
            usage: Some(proto::EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            }),
        }))
    }
}

fn grpc_embedding_prompt_tokens(input: &[u32]) -> u32 {
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
        // Mirror of `native_chat_renderer_keeps_thinking_open_for_qwen_reasoning_models`
        // in main.rs. Without this branch the gRPC path would have reproduced
        // #13 once gRPC was wired up: enable_thinking=false pre-closes the
        // `<think>` block, which causes Qwen3.6 / Qwen3-Next / Qwen3-Coder-Next
        // to truncate or loop on reasoning prompts.
        let thinking =
            render_grpc_chat_prompt("Qwen3.6-35B-A3B-5bit", &user("hi")).expect("render");
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
        // Mirror of `openai_glm_prompt_renderer_preserves_tool_observation_shape`
        // in main.rs. GLM4.7 needs tool/function roles routed to
        // `<|observation|><tool_response>...</tool_response>` and assistant turns
        // to include `</think>` after the tag.
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
