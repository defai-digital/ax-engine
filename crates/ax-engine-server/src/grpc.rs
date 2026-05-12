use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use ax_engine_sdk::{
    EmbeddingPooling, EngineSession, EngineSessionError, GenerateFinishReason, GenerateRequest,
    GenerateSampling, GenerateStreamEvent, GenerateStreamState, StatelessGenerateContext,
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use super::{AppState, StreamStateSource, render_grpc_chat_prompt, grpc_chat_stop_sequences};

// Include the tonic-generated server code.
pub mod proto {
    tonic::include_proto!("ax_engine.v1");
}

use proto::ax_engine_server::AxEngine;

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

// ─── Stream type aliases ──────────────────────────────────────────────────────

type GenerateEventStream = ReceiverStream<Result<proto::GenerateStreamEvent, Status>>;
type ChatChunkStream = ReceiverStream<Result<proto::ChatCompletionChunk, Status>>;
type CompletionChunkStream = ReceiverStream<Result<proto::CompletionChunk, Status>>;

// ─── Type conversions ─────────────────────────────────────────────────────────

fn proto_sampling_to_sdk(s: proto::GenerateSampling) -> GenerateSampling {
    GenerateSampling {
        temperature: s.temperature,
        top_p: if s.top_p == 0.0 { 1.0 } else { s.top_p },
        top_k: s.top_k,
        min_p: None,
        repetition_penalty: if s.repetition_penalty == 0.0 {
            1.0
        } else {
            s.repetition_penalty
        },
        repetition_context_size: None,
        seed: s.seed,
        deterministic: None,
    }
}

fn proto_to_generate_request(state: &AppState, req: proto::GenerateRequest) -> GenerateRequest {
    let sampling = req.sampling.map(proto_sampling_to_sdk).unwrap_or_default();
    GenerateRequest {
        model_id: state.model_id.to_string(),
        input_tokens: req.input_tokens,
        input_text: if req.input_text.is_empty() {
            None
        } else {
            Some(req.input_text)
        },
        max_output_tokens: if req.max_output_tokens == 0 {
            256
        } else {
            req.max_output_tokens
        },
        sampling,
        stop_sequences: Vec::new(),
        metadata: if req.metadata.is_empty() {
            None
        } else {
            Some(req.metadata)
        },
    }
}

fn sdk_response_to_proto(r: ax_engine_sdk::GenerateResponse) -> proto::GenerateResponse {
    proto::GenerateResponse {
        request_id: r.request_id,
        model_id: r.model_id,
        prompt_tokens: r.prompt_tokens,
        output_tokens: r.output_tokens,
        output_text: r.output_text.unwrap_or_default(),
        status: format!("{:?}", r.status),
        finish_reason: r.finish_reason.map(finish_reason_str).unwrap_or_default(),
        step_count: r.step_count,
    }
}

fn sdk_request_report_to_proto(
    r: &ax_engine_sdk::SessionRequestReport,
) -> proto::RequestReport {
    proto::RequestReport {
        request_id: r.request_id,
        model_id: r.model_id.clone(),
        state: format!("{:?}", r.state),
        prompt_tokens: r.prompt_tokens.clone(),
        processed_prompt_tokens: r.processed_prompt_tokens,
        output_tokens: r.output_tokens.clone(),
        prompt_len: r.prompt_len,
        output_len: r.output_len,
        max_output_tokens: r.max_output_tokens,
        cancel_requested: r.cancel_requested,
    }
}

fn sdk_stream_event_to_proto(event: GenerateStreamEvent) -> proto::GenerateStreamEvent {
    match event {
        GenerateStreamEvent::Request(payload) => proto::GenerateStreamEvent {
            event: "request".to_string(),
            request: Some(sdk_request_report_to_proto(&payload.request)),
            step: None,
            response: None,
        },
        GenerateStreamEvent::Step(payload) => proto::GenerateStreamEvent {
            event: "step".to_string(),
            request: None,
            step: Some(proto::GenerateStepEvent {
                request: Some(sdk_request_report_to_proto(&payload.request)),
                step: Some(proto::StepReport {
                    scheduled_requests: payload.step.scheduled_requests,
                    scheduled_tokens: payload.step.scheduled_tokens,
                    ttft_events: payload.step.ttft_events,
                    prefix_hits: payload.step.prefix_hits,
                    kv_usage_blocks: payload.step.kv_usage_blocks,
                    evictions: payload.step.evictions,
                    cpu_time_us: payload.step.cpu_time_us,
                    runner_time_us: payload.step.runner_time_us,
                }),
                delta_tokens: payload.delta_tokens,
                delta_text: payload.delta_text.unwrap_or_default(),
            }),
            response: None,
        },
        GenerateStreamEvent::Response(payload) => proto::GenerateStreamEvent {
            event: "response".to_string(),
            request: None,
            step: None,
            response: Some(sdk_response_to_proto(payload.response)),
        },
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn finish_reason_str(fr: GenerateFinishReason) -> String {
    match fr {
        GenerateFinishReason::Stop => "stop".to_string(),
        GenerateFinishReason::Length => "length".to_string(),
    }
}

// ─── Async task helper ────────────────────────────────────────────────────────

async fn run_blocking<F, T, E>(f: F) -> Result<T, Status>
where
    F: FnOnce() -> Result<T, E> + Send + 'static,
    T: Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Status::internal(e.to_string()))?
        .map_err(|e| Status::internal(e.to_string()))
}

fn next_request_id(state: &AppState) -> u64 {
    state.next_request_id.fetch_add(1, Ordering::AcqRel)
}

// ─── Stream state setup ───────────────────────────────────────────────────────

async fn build_grpc_stream_state(
    state: &AppState,
    request: GenerateRequest,
) -> Result<(GenerateStreamState, StreamStateSource), Status> {
    let request_id = next_request_id(state);

    if state
        .stateless_generate_context
        .supports_stateless_streaming()
    {
        let ctx = Arc::clone(&state.stateless_generate_context);
        let stream_ctx = Arc::clone(&ctx);
        let ss = run_blocking(move || {
            stream_ctx.stream_state_with_request_id(request_id, request)
        })
        .await?;
        return Ok((ss, StreamStateSource::Stateless(ctx)));
    }

    let session_config = state.session_config.as_ref().clone();
    let (session, ss) = run_blocking(move || {
        let mut session = EngineSession::new(session_config).map_err(|e| e.to_string())?;
        let ss = session
            .stream_generate_state_with_request_id(request_id, request)
            .map_err(|e| e.to_string())?;
        Ok::<_, String>((session, ss))
    })
    .await?;

    Ok((ss, StreamStateSource::Stateful(Box::new(session))))
}

// ─── Generic generate stream driver ──────────────────────────────────────────

fn spawn_grpc_generate_stream(
    stream_state: GenerateStreamState,
    tx: mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    stream_context: StreamStateSource,
) {
    tokio::task::spawn_blocking(move || {
        let mut ss = stream_state;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_generate_events(&mut ss, &tx, |s| ctx.next_stream_event(s))
            }
            StreamStateSource::Stateful(mut session) => {
                drive_grpc_generate_events(&mut ss, &tx, |s| session.next_stream_event(s))
            }
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_generate_events<N>(
    state: &mut GenerateStreamState,
    tx: &mpsc::Sender<Result<proto::GenerateStreamEvent, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(Some(event)) => {
                if tx.blocking_send(Ok(sdk_stream_event_to_proto(event))).is_err() {
                    return Ok(());
                }
            }
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
        }
    }
}

// ─── Chat stream driver ───────────────────────────────────────────────────────

fn spawn_grpc_chat_stream(
    stream_state: GenerateStreamState,
    model_id: String,
    tx: mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    stream_context: StreamStateSource,
) {
    tokio::task::spawn_blocking(move || {
        let mut ss = stream_state;
        let mut chat_role_emitted = false;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => drive_grpc_chat_events(
                &mut ss,
                &model_id,
                &mut chat_role_emitted,
                &tx,
                |s| ctx.next_stream_event(s),
            ),
            StreamStateSource::Stateful(mut session) => drive_grpc_chat_events(
                &mut ss,
                &model_id,
                &mut chat_role_emitted,
                &tx,
                |s| session.next_stream_event(s),
            ),
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_chat_events<N>(
    state: &mut GenerateStreamState,
    model_id: &str,
    chat_role_emitted: &mut bool,
    tx: &mpsc::Sender<Result<proto::ChatCompletionChunk, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
            Ok(Some(GenerateStreamEvent::Request(_) | GenerateStreamEvent::Response(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let Some(delta_text) = payload.delta_text else {
                    continue;
                };
                if delta_text.is_empty() {
                    continue;
                }
                let role = if *chat_role_emitted {
                    String::new()
                } else {
                    *chat_role_emitted = true;
                    "assistant".to_string()
                };
                let chunk = proto::ChatCompletionChunk {
                    id: format!("chatcmpl-{}", payload.request.request_id),
                    object: "chat.completion.chunk".to_string(),
                    created: unix_now(),
                    model: model_id.to_string(),
                    choices: vec![proto::ChatCompletionChunkChoice {
                        index: 0,
                        delta: Some(proto::ChatDelta { role, content: delta_text }),
                        finish_reason: String::new(),
                    }],
                };
                if tx.blocking_send(Ok(chunk)).is_err() {
                    return Ok(());
                }
            }
        }
    }
}

// ─── Completion stream driver ─────────────────────────────────────────────────

fn spawn_grpc_completion_stream(
    stream_state: GenerateStreamState,
    model_id: String,
    tx: mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    stream_context: StreamStateSource,
) {
    tokio::task::spawn_blocking(move || {
        let mut ss = stream_state;
        let result = match stream_context {
            StreamStateSource::Stateless(ctx) => {
                drive_grpc_completion_events(&mut ss, &model_id, &tx, |s| ctx.next_stream_event(s))
            }
            StreamStateSource::Stateful(mut session) => drive_grpc_completion_events(
                &mut ss,
                &model_id,
                &tx,
                |s| session.next_stream_event(s),
            ),
        };
        if let Err(status) = result {
            let _ = tx.blocking_send(Err(status));
        }
    });
}

fn drive_grpc_completion_events<N>(
    state: &mut GenerateStreamState,
    model_id: &str,
    tx: &mpsc::Sender<Result<proto::CompletionChunk, Status>>,
    mut next: N,
) -> Result<(), Status>
where
    N: FnMut(&mut GenerateStreamState) -> Result<Option<GenerateStreamEvent>, EngineSessionError>,
{
    loop {
        match next(state) {
            Ok(None) => return Ok(()),
            Err(e) => return Err(Status::internal(e.to_string())),
            Ok(Some(GenerateStreamEvent::Request(_) | GenerateStreamEvent::Response(_))) => {}
            Ok(Some(GenerateStreamEvent::Step(payload))) => {
                let Some(delta_text) = payload.delta_text else {
                    continue;
                };
                if delta_text.is_empty() {
                    continue;
                }
                let chunk = proto::CompletionChunk {
                    id: format!("cmpl-{}", payload.request.request_id),
                    object: "text_completion.chunk".to_string(),
                    created: unix_now(),
                    model: model_id.to_string(),
                    choices: vec![proto::CompletionChunkChoice {
                        index: 0,
                        text: delta_text,
                        finish_reason: String::new(),
                    }],
                };
                if tx.blocking_send(Ok(chunk)).is_err() {
                    return Ok(());
                }
            }
        }
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
    let max_output_tokens = if req.max_tokens == 0 { 256 } else { req.max_tokens };
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
    let max_output_tokens = if req.max_tokens == 0 { 256 } else { req.max_tokens };
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
        let request_id = next_request_id(&self.state);
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let response =
            run_blocking(move || ctx.generate_with_request_id(request_id, req)).await?;
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
        let request_id = next_request_id(&self.state);
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let r = run_blocking(move || ctx.generate_with_request_id(request_id, generate_req)).await?;
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
        let request_id = next_request_id(&self.state);
        let ctx = Arc::clone(&self.state.stateless_generate_context);
        let r = run_blocking(move || ctx.generate_with_request_id(request_id, generate_req)).await?;
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
        let embedding = self
            .state
            .embedding_batcher
            .embed(req.input, pooling, req.normalize)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let prompt_tokens = embedding.len() as u32;
        Ok(Response::new(proto::EmbeddingsResponse {
            object: "list".to_string(),
            data: vec![proto::EmbeddingData {
                object: "embedding".to_string(),
                embedding,
                index: 0,
            }],
            model: model_id,
            usage: proto::EmbeddingUsage {
                prompt_tokens,
                total_tokens: prompt_tokens,
            },
        }))
    }
}
