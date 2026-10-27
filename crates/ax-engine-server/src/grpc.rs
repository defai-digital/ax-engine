// `tonic::Status` is ~176 bytes — clippy::result_large_err fires on every helper
// returning `Result<_, Status>`. The trait surface comes from tonic and the
// helpers mirror that surface, so boxing per-call doesn't pay for itself. Allow
// at the module level rather than peppering individual functions.
#![allow(clippy::result_large_err)]

use crate::generation::native::reject_failed_generate_status;
use crate::openai::stop::{
    apply_client_stops_to_answer_only, apply_client_stops_to_generate_response,
    validate_client_stop_sequences,
};
use std::sync::Arc;

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
use crate::openai::generation::populate_native_mlx_output_text;
use crate::openai::schema::OpenAiStreamKind;
use conversions::{finish_reason_str, proto_to_generate_request, sdk_response_to_proto, unix_now};
use proto::ax_engine_server::AxEngine;
use requests::{
    build_chat_generate_request, build_completion_generate_request, grpc_embedding_prompt_tokens,
    openai_error_to_status,
};
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
            .generate(
                request_id,
                request,
                permit,
                state.limits.generate_max_duration,
            )
            .await
            .map_err(generation_service_status);
    }
    let ctx = Arc::clone(&live.stateless_generate_context);
    let generate = run_blocking(move || {
        let _permit = permit;
        ctx.generate_with_request_id(request_id, request)
    });
    // Same hang backstop as HTTP: a stalled delegated upstream must not hold
    // the call (and its admission permit) forever.
    match state.limits.generate_max_duration {
        Some(deadline) => tokio::time::timeout(deadline, generate)
            .await
            .unwrap_or_else(|_elapsed| {
                Err(generation_service_status(
                    GenerationServiceError::DeadlineExceeded {
                        request_id,
                        observed_event_count: 0,
                    },
                ))
            }),
        None => generate.await,
    }
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
        GenerationServiceError::DeadlineExceeded {
            request_id,
            observed_event_count,
        } => Status::deadline_exceeded(format!(
            "generation request {request_id} did not complete within the configured deadline \
             ({observed_event_count} stream event(s) observed)"
        )),
    }
}

fn grpc_model(state: &AppState, requested: &str) -> Result<crate::app_state::LiveState, Status> {
    let requested = (!requested.is_empty()).then_some(requested);
    let live = state.snapshot_for_model(requested).ok_or_else(|| {
        Status::not_found(format!(
            "requested model is not loaded; loaded models: {}",
            state.model_ids().join(", ")
        ))
    })?;
    // Activity before admission (see `select_model`).
    live.mark_used();
    Ok(live)
}

// ─── Service implementation ───────────────────────────────────────────────────

#[tonic::async_trait]
impl AxEngine for AxEngineGrpcService {
    async fn health(
        &self,
        _request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let unavailable = self.state.unavailable_model_ids();
        if !unavailable.is_empty() {
            return Err(Status::unavailable(format!(
                "native generation workers are unavailable for loaded models: {}",
                unavailable.join(", ")
            )));
        }
        let live = self.state.snapshot();
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
        let lives = self.state.snapshots();
        Ok(Response::new(proto::ModelsResponse {
            object: "list".to_string(),
            data: lives
                .iter()
                .map(|live| proto::ModelCard {
                    id: live.model_id.to_string(),
                    object: "model".to_string(),
                    owned_by: MODEL_OWNER.to_string(),
                })
                .collect(),
        }))
    }

    // ── Generate unary ────────────────────────────────────────────────────────

    async fn generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<proto::GenerateResponse>, Status> {
        let request = request.into_inner();
        let live = grpc_model(&self.state, &request.model)?;
        let req = proto_to_generate_request(&live, request)?;
        let request_id = self.state.allocate_request_id();
        let mut response = run_grpc_generate_request(&self.state, &live, request_id, req).await?;
        // Decode native MLX tokens so unary Generate returns text (HTTP parity).
        populate_native_mlx_output_text(&live, &mut response, OpenAiStreamKind::Completion, false)
            .map_err(openai_error_to_status)?;
        Ok(Response::new(sdk_response_to_proto(response)))
    }

    // ── StreamGenerate ────────────────────────────────────────────────────────

    type StreamGenerateStream = GenerateEventStream;

    async fn stream_generate(
        &self,
        request: Request<proto::GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let request = request.into_inner();
        let live = grpc_model(&self.state, &request.model)?;
        let req = proto_to_generate_request(&live, request)?;
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
        let req = request.into_inner();
        let live = grpc_model(&self.state, &req.model)?;
        validate_client_stop_sequences(&req.stop).map_err(openai_error_to_status)?;
        let generate_req = build_chat_generate_request(&live, &req)?;
        let request_id = self.state.allocate_request_id();
        let mut r = run_grpc_generate_request(&self.state, &live, request_id, generate_req).await?;
        reject_failed_generate_status(r.status, r.finish_reason).map_err(openai_error_to_status)?;
        // Native MLX leaves output_text unset; decode + strip channels like OpenAI.
        populate_native_mlx_output_text(&live, &mut r, OpenAiStreamKind::ChatCompletion, false)
            .map_err(openai_error_to_status)?;
        // The native session ignores stop_sequences; enforce client stops on
        // the decoded answer as the HTTP surfaces do.
        apply_client_stops_to_answer_only(&mut r, &req.stop);
        // Delegated backends report counts without materializing token
        // arrays, so `.len()` would report zero usage; use the SDK's
        // authoritative counts exactly like the HTTP surface.
        let usage = r
            .known_usage()
            .map(|(prompt_tokens, completion_tokens)| proto::TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
            });
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
            usage,
        }))
    }

    // ── StreamChatCompletion ──────────────────────────────────────────────────

    type StreamChatCompletionStream = ChatChunkStream;

    async fn stream_chat_completion(
        &self,
        request: Request<proto::ChatCompletionRequest>,
    ) -> Result<Response<Self::StreamChatCompletionStream>, Status> {
        let req = request.into_inner();
        let live = grpc_model(&self.state, &req.model)?;
        let model_id = live.model_id.to_string();
        validate_client_stop_sequences(&req.stop).map_err(openai_error_to_status)?;
        let generate_req = build_chat_generate_request(&live, &req)?;
        let tokenizer = native_grpc_stream_tokenizer(&live)?;
        // gRPC renders thinking off; DeepSeek then needs its `</think>` leak
        // cut on the stream, as the unary path's sanitize does.
        let deepseek_think_cut = live.runtime_report.selected_backend.is_mlx()
            && matches!(
                crate::chat::resolve_chat_template(
                    live.model_id.as_ref(),
                    crate::metadata::model_family_from_artifacts(&live).as_deref(),
                ),
                crate::chat::ChatPromptTemplate::DeepSeekChat
            );
        let stream = build_grpc_stream_state(&self.state, &live, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_chat_stream(
            model_id,
            tx,
            stream,
            tokenizer,
            req.stop.clone(),
            deepseek_think_cut,
        )
        .map_err(generation_service_status)?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Completion unary ──────────────────────────────────────────────────────

    async fn completion(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<proto::CompletionResponse>, Status> {
        let req = request.into_inner();
        let live = grpc_model(&self.state, &req.model)?;
        validate_client_stop_sequences(&req.stop).map_err(openai_error_to_status)?;
        let generate_req = build_completion_generate_request(&live, &req)?;
        let request_id = self.state.allocate_request_id();
        let mut r = run_grpc_generate_request(&self.state, &live, request_id, generate_req).await?;
        reject_failed_generate_status(r.status, r.finish_reason).map_err(openai_error_to_status)?;
        populate_native_mlx_output_text(&live, &mut r, OpenAiStreamKind::Completion, false)
            .map_err(openai_error_to_status)?;
        apply_client_stops_to_generate_response(&mut r, &req.stop);
        // Delegated backends report counts without materializing token
        // arrays, so `.len()` would report zero usage; use the SDK's
        // authoritative counts exactly like the HTTP surface.
        let usage = r
            .known_usage()
            .map(|(prompt_tokens, completion_tokens)| proto::TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens.saturating_add(completion_tokens),
            });
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
            usage,
        }))
    }

    // ── StreamCompletion ──────────────────────────────────────────────────────

    type StreamCompletionStream = CompletionChunkStream;

    async fn stream_completion(
        &self,
        request: Request<proto::CompletionRequest>,
    ) -> Result<Response<Self::StreamCompletionStream>, Status> {
        let req = request.into_inner();
        let live = grpc_model(&self.state, &req.model)?;
        let model_id = live.model_id.to_string();
        validate_client_stop_sequences(&req.stop).map_err(openai_error_to_status)?;
        let generate_req = build_completion_generate_request(&live, &req)?;
        let tokenizer = native_grpc_stream_tokenizer(&live)?;
        let stream = build_grpc_stream_state(&self.state, &live, generate_req).await?;
        let (tx, rx) = mpsc::channel(GRPC_CHANNEL_CAPACITY);
        spawn_grpc_completion_stream(model_id, tx, stream, tokenizer, req.stop.clone())
            .map_err(generation_service_status)?;
        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Embeddings ────────────────────────────────────────────────────────────

    async fn embeddings(
        &self,
        request: Request<proto::EmbeddingsRequest>,
    ) -> Result<Response<proto::EmbeddingsResponse>, Status> {
        let req = request.into_inner();
        let live = grpc_model(&self.state, &req.model)?;
        let model_id = live.model_id.to_string();
        // Same rails as /v1/embeddings: unknown pooling is an error, not a
        // silent last-token fallback, and the token cap applies.
        let pooling = crate::embeddings::parse_embedding_pooling(
            Some(req.pooling.as_str()).filter(|value| !value.is_empty()),
        )
        .map_err(Status::invalid_argument)?;
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
        // Same item-count cap as the HTTP surface.
        let max_items = crate::openai::embeddings::DEFAULT_EMBED_MAX_ITEMS;
        if batch.len() > max_items {
            return Err(Status::invalid_argument(format!(
                "input has {} items, exceeding the maximum of {max_items}",
                batch.len()
            )));
        }
        // Start-up-resolved AX_ENGINE_EMBED_MAX_TOKENS (default 8192;
        // see ServerEnvConfig for the accepted values). The cap is per item,
        // matching the HTTP surface: each input is embedded independently, so
        // a many-item batch of valid inputs must not fail. The running total
        // is bounded separately by the batch budget.
        let max_tokens = self.state.env.embed_max_tokens;
        let max_batch_tokens = self.state.env.embed_max_batch_tokens;
        let mut token_count = 0usize;
        for (i, ids) in batch.iter().enumerate() {
            token_count += ids.len();
            if ids.len() > max_tokens {
                return Err(Status::invalid_argument(format!(
                    "input[{i}] token count ({}) exceeds maximum ({max_tokens}); \
                     set AX_ENGINE_EMBED_MAX_TOKENS to override",
                    ids.len()
                )));
            }
        }
        if token_count > max_batch_tokens {
            return Err(Status::invalid_argument(format!(
                "batch token count ({token_count}) exceeds maximum ({max_batch_tokens}); \
                 set AX_ENGINE_EMBED_MAX_BATCH_TOKENS to override"
            )));
        }
        let permit = self.state.try_admit(&live).map_err(admission_status)?;
        let prompt_tokens = grpc_embedding_prompt_tokens(&batch);
        // Same bound as HTTP /v1/embeddings (AX_ENGINE_EMBED_TIMEOUT_MS).
        let embed_timeout = self.state.env.embed_timeout_ms;
        let timeout = std::time::Duration::from_millis(embed_timeout);
        let timed_out = || {
            Status::unavailable(format!(
                "embedding request timed out after {embed_timeout}ms"
            ))
        };
        let embeddings = if batch.len() == 1 {
            let input = batch
                .into_iter()
                .next()
                .ok_or_else(|| Status::invalid_argument("embedding input must not be empty"))?;
            vec![
                tokio::time::timeout(
                    timeout,
                    live.embedding_batcher
                        .embed(input, pooling, req.normalize, permit),
                )
                .await
                .map_err(|_| timed_out())?
                // Classified like HTTP and the batch branch (a bad token id
                // is INVALID_ARGUMENT, not INTERNAL).
                .map_err(streams::session_error_status)?,
            ]
        } else {
            let generation_service = live.generation_service.clone();
            tokio::time::timeout(
                timeout,
                generation_service.execute(move |session| {
                    let _permit = permit;
                    session.embed_batch_flat(&batch, pooling, req.normalize)
                }),
            )
            .await
            .map_err(|_| timed_out())?
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
