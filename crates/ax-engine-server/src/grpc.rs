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
use super::generation::service::GenerationServiceError;
use super::metadata::MODEL_OWNER;

mod conversions;
mod requests;
mod streams;

// Include the tonic-generated server code.
pub mod proto {
    tonic::include_proto!("ax_engine.v1");
}

use crate::admission::AdmissionError;
use conversions::{finish_reason_str, proto_to_generate_request, sdk_response_to_proto, unix_now};
use proto::ax_engine_server::AxEngine;
use requests::{
    build_chat_generate_request, build_completion_generate_request, grpc_embedding_prompt_tokens,
    openai_error_to_status,
};
use crate::openai::generation::populate_native_mlx_output_text;
use crate::openai::schema::OpenAiStreamKind;
use streams::{
    ChatChunkStream, CompletionChunkStream, GenerateEventStream, build_grpc_stream_state,
    native_grpc_stream_tokenizer, run_blocking, spawn_grpc_chat_stream,
    spawn_grpc_completion_stream, spawn_grpc_generate_stream,
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
    live: &crate::app_state::LiveState,
    request_id: u64,
    request: ax_engine_sdk::GenerateRequest,
) -> Result<ax_engine_sdk::GenerateResponse, Status> {
    let permit = state.try_admit(live).map_err(admission_status)?;
    if live.runtime_report.selected_backend.is_mlx() {
        let generation_service = live.generation_service.clone();
        return generation_service
            .generate(request_id, request, permit)
            .await
            .map_err(generation_service_status);
    }
    let ctx = Arc::clone(&live.stateless_generate_context);
    run_blocking(move || {
        let _permit = permit;
        ctx.generate_with_request_id(request_id, request)
    })
    .await
}

fn admission_status(error: AdmissionError) -> Status {
    match error {
        AdmissionError::Draining => Status::unavailable(
            "the current model is draining for replacement; retry after loading completes",
        ),
        AdmissionError::Saturated => Status::resource_exhausted(
            "server is at its maximum concurrent engine-job limit; retry shortly",
        ),
        AdmissionError::StaleGeneration => Status::unavailable(
            "the model changed while this request was being prepared; retry against the current model",
        ),
    }
}

fn generation_service_status(error: GenerationServiceError) -> Status {
    match error {
        GenerationServiceError::Engine(error) => streams::session_error_status(error),
        GenerationServiceError::Saturated => {
            Status::resource_exhausted("native generation command queue is saturated")
        }
        GenerationServiceError::Unavailable => {
            Status::unavailable("native generation worker is unavailable")
        }
    }
}

// ─── Service implementation ───────────────────────────────────────────────────

#[tonic::async_trait]
impl AxEngine for AxEngineGrpcService {
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let live = self.state.snapshot();
        if !live.generation_service.is_ready() {
            return Err(Status::unavailable(
                "the native generation worker is unavailable",
            ));
        }
        Ok(Response::new(proto::HealthResponse {
            status: "ok".to_string(),
            service: "ax-engine-server".to_string(),
            model_id: live.model_id.to_string(),
        }))
    }

    async fn models(
        &self,
        _request: Request<proto::ModelsRequest>,
    ) -> Result<Response<proto::ModelsResponse>, Status> {
        let live = self.state.snapshot();
        Ok(Response::new(proto::ModelsResponse {
            object: "list".to_string(),
            data: vec![proto::ModelCard {
                id: live.model_id.to_string(),
                object: "model".to_string(),
                owned_by: MODEL_OWNER.to_string(),
            }],
        }))
    }

    // ── Generate unary ────────────────────────────────────────────────────────

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<proto::GenerateResponse>, Status> {
        let live = self.state.snapshot();
        let req = proto_to_generate_request(&live, request.into_inner())?;
        let request_id = self.state.allocate_request_id();
        let mut response = run_grpc_generate_request(&self.state, &live, request_id, req).await?;
        // Decode native MLX tokens so unary Generate returns text (HTTP parity).
        populate_native_mlx_output_text(
            &live,
            &mut response,
            OpenAiStreamKind::Completion,
            false,
        )
        .map_err(openai_error_to_status)?;
        Ok(Response::new(sdk_response_to_proto(response)))
    }

    // ── StreamGenerate ────────────────────────────────────────────────────────

    type StreamGenerateStream = GenerateEventStream;

    async fn stream_generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let live = self.state.snapshot();
        let req = proto_to_generate_request(&live, request.into_inner())?;
        let stream = build_grpc_stream_state(&self.state, &live, req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_generate_stream(tx, stream).map_err(generation_service_status)?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── ChatCompletion unary ──────────────────────────────────────────────────

    async fn chat_completion(
        &self,
        request: Request<proto::ChatCompletionRequest>,
    ) -> Result<Response<proto::ChatCompletionResponse>, Status> {
        let live = self.state.snapshot();
        let req = request.into_inner();
        let generate_req = build_chat_generate_request(&live, &req)?;
        let request_id = self.state.allocate_request_id();
        let mut r = run_grpc_generate_request(&self.state, &live, request_id, generate_req).await?;
        // Native MLX leaves output_text unset; decode + strip channels like OpenAI.
        populate_native_mlx_output_text(
            &live,
            &mut r,
            OpenAiStreamKind::ChatCompletion,
            false,
        )
        .map_err(openai_error_to_status)?;
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
        let live = self.state.snapshot();
        let req = request.into_inner();
        let model_id = live.model_id.to_string();
        let generate_req = build_chat_generate_request(&live, &req)?;
        let tokenizer = native_grpc_stream_tokenizer(&live)?;
        let stream = build_grpc_stream_state(&self.state, &live, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_chat_stream(model_id, tx, stream, tokenizer)
            .map_err(generation_service_status)?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Completion unary ──────────────────────────────────────────────────────

    async fn completion(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<proto::CompletionResponse>, Status> {
        let live = self.state.snapshot();
        let req = request.into_inner();
        let generate_req = build_completion_generate_request(&live, &req)?;
        let request_id = self.state.allocate_request_id();
        let mut r = run_grpc_generate_request(&self.state, &live, request_id, generate_req).await?;
        populate_native_mlx_output_text(&live, &mut r, OpenAiStreamKind::Completion, false)
            .map_err(openai_error_to_status)?;
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
        let live = self.state.snapshot();
        let req = request.into_inner();
        let model_id = live.model_id.to_string();
        let generate_req = build_completion_generate_request(&live, &req)?;
        let tokenizer = native_grpc_stream_tokenizer(&live)?;
        let stream = build_grpc_stream_state(&self.state, &live, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_completion_stream(model_id, tx, stream, tokenizer)
            .map_err(generation_service_status)?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    async fn embeddings(
        &self,
        request: Request<proto::EmbeddingsRequest>,
    ) -> Result<Response<proto::EmbeddingsResponse>, Status> {
        let live = self.state.snapshot();
        let req = request.into_inner();
        let model_id = live.model_id.to_string();
        let pooling = match req.pooling.as_str() {
            "mean" => EmbeddingPooling::Mean,
            "cls" => EmbeddingPooling::Cls,
            _ => EmbeddingPooling::Last,
        };
        let batch = if req.inputs.is_empty() {
            vec![req.input]
        } else {
            req.inputs
                .into_iter()
                .map(|input| input.tokens)
                .collect::<Vec<_>>()
        };
        if batch.is_empty() || batch.iter().any(Vec::is_empty) {
            return Err(Status::invalid_argument(
                "embedding input must not be empty",
            ));
        }
        let permit = self.state.try_admit(&live).map_err(admission_status)?;
        let prompt_tokens = grpc_embedding_prompt_tokens(&batch);
        let embeddings = if batch.len() == 1 {
            let input = batch
                .into_iter()
                .next()
                .ok_or_else(|| Status::invalid_argument("embedding input must not be empty"))?;
            vec![
                live.embedding_batcher
                    .embed(input, pooling, req.normalize, permit)
                    .await
                    .map_err(|e| Status::internal(e.to_string()))?,
            ]
        } else {
            let generation_service = live.generation_service.clone();
            generation_service
                .execute(move |session| {
                    let _permit = permit;
                    session.embed_batch_flat(&batch, pooling, req.normalize)
                })
                .await
                .map_err(generation_service_status)
                .map(|matrix| {
                    (0..matrix.batch_size)
                        .map(|index| matrix.row(index).to_vec())
                        .collect::<Vec<_>>()
                })?
        };

        Ok(Response::new(proto::EmbeddingsResponse {
            object: "list".to_string(),
            data: embeddings
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| proto::EmbeddingData {
                    object: "embedding".to_string(),
                    embedding,
                    index: index as u32,
                })
                .collect(),
            model: model_id,
            usage: Some(proto::EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            }),
        }))
    }
}
