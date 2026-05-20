// `tonic::Status` is ~176 bytes — clippy::result_large_err fires on every helper
// returning `Result<_, Status>`. The trait surface comes from tonic and the
// helpers mirror that surface, so boxing per-call doesn't pay for itself. Allow
// at the module level rather than peppering individual functions.
#![allow(clippy::result_large_err)]

use std::sync::Arc;

use ax_engine_sdk::EmbeddingPooling;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::app_state::AppState;

mod conversions;
mod requests;
mod streams;

// Include the tonic-generated server code.
pub mod proto {
    tonic::include_proto!("ax_engine.v1");
}

use conversions::{finish_reason_str, proto_to_generate_request, sdk_response_to_proto, unix_now};
use proto::ax_engine_server::AxEngine;
use requests::{
    build_chat_generate_request, build_completion_generate_request, grpc_embedding_prompt_tokens,
};
use streams::{
    ChatChunkStream, CompletionChunkStream, GenerateEventStream, build_grpc_stream_state,
    run_blocking, spawn_grpc_chat_stream, spawn_grpc_completion_stream, spawn_grpc_generate_stream,
};

const GRPC_CHANNEL_CAPACITY: usize = 128;

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

async fn run_grpc_generate_request(
    state: &AppState,
    request_id: u64,
    request: ax_engine_sdk::GenerateRequest,
) -> Result<ax_engine_sdk::GenerateResponse, Status> {
    let ctx = Arc::clone(&state.stateless_generate_context);
    run_blocking(move || ctx.generate_with_request_id(request_id, request)).await
}

// ─── Service implementation ───────────────────────────────────────────────────

#[tonic::async_trait]
impl AxEngine for AxEngineGrpcService {
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let session_lock = self.state.request_session.try_lock();
        if session_lock.is_err() {
            return Err(Status::unavailable(
                "ax-engine-server has not finished initialising its inference session",
            ));
        }
        drop(session_lock);

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
        let response = run_grpc_generate_request(&self.state, request_id, req).await?;
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
        let r = run_grpc_generate_request(&self.state, request_id, generate_req).await?;
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
        let r = run_grpc_generate_request(&self.state, request_id, generate_req).await?;
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
