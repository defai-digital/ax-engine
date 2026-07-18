use std::collections::BTreeMap;
use std::time::Duration;

use ax_engine_sdk::{EngineTokenizer, EngineTokenizerError};
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app_state::{AppState, LiveState};
use crate::embeddings::{parse_embedding_pooling, parse_embedding_timeout_ms};
use crate::errors::{
    ErrorResponse, admission_error_response, error_response, map_generation_service_error,
};
use crate::openai::validation::select_model;

type HttpErrorResponse = (StatusCode, Json<ErrorResponse>);

const DEFAULT_CHUNK_MAX_TOKENS: usize = 512;
const DEFAULT_CHUNK_OVERLAP_TOKENS: usize = 0;
const MAX_RECORDS_PER_REQUEST: usize = 2048;
const MAX_CHUNKS_PER_REQUEST: usize = 8192;
const DEFAULT_EMBED_RECORDS_TIMEOUT_MS: u64 = 60_000;

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingRecordsRequest {
    #[serde(default)]
    model: Option<String>,
    records: Vec<EmbeddingRecordInput>,
    #[serde(default)]
    render_template: Option<String>,
    #[serde(default)]
    chunking: Option<EmbeddingChunking>,
    #[serde(default)]
    pooling: Option<String>,
    #[serde(default)]
    normalize: Option<bool>,
    #[serde(default = "default_add_eos")]
    add_eos: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingRecordInput {
    id: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    fields: BTreeMap<String, Value>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct EmbeddingChunking {
    #[serde(default = "default_chunk_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_chunk_overlap_tokens")]
    overlap_tokens: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingRecordsResponse {
    object: &'static str,
    data: Vec<EmbeddingRecordObject>,
    model: String,
    usage: EmbeddingRecordsUsage,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingRecordObject {
    object: &'static str,
    id: String,
    embedding: Vec<f32>,
    index: u32,
    record_index: u32,
    chunk_index: u32,
    token_start: u32,
    token_end: u32,
    token_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingRecordsUsage {
    prompt_tokens: usize,
    total_tokens: usize,
    records: usize,
    chunks: usize,
}

#[derive(Clone, Debug)]
struct PreparedEmbeddingChunk {
    id: String,
    record_index: usize,
    chunk_index: usize,
    token_start: usize,
    token_end: usize,
    token_count: usize,
    metadata: Option<Value>,
    token_ids: Vec<u32>,
}

fn default_add_eos() -> bool {
    true
}

fn default_chunk_max_tokens() -> usize {
    DEFAULT_CHUNK_MAX_TOKENS
}

fn default_chunk_overlap_tokens() -> usize {
    DEFAULT_CHUNK_OVERLAP_TOKENS
}

pub(crate) async fn embedding_records(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRecordsRequest>,
) -> Result<Json<EmbeddingRecordsResponse>, HttpErrorResponse> {
    let live = select_model(&state, request.model.as_deref())?;

    let pooling = parse_embedding_pooling(request.pooling.as_deref())
        .map_err(|message| error_response(StatusCode::BAD_REQUEST, "invalid_request", message))?;
    let normalize = request.normalize.unwrap_or(true);
    let model_id = live.model_id.as_ref().clone();
    let tokenizer = tokenizer_for_live(&live)?;
    let chunks = prepare_embedding_record_chunks(&request, &tokenizer)?;
    let batch = chunks
        .iter()
        .map(|chunk| chunk.token_ids.clone())
        .collect::<Vec<_>>();
    let token_count = chunks.iter().map(|chunk| chunk.token_count).sum();
    let generation_service = live.generation_service.clone();
    let timeout_ms = parse_embedding_timeout_ms(
        std::env::var("AX_ENGINE_EMBED_TIMEOUT_MS").ok(),
        DEFAULT_EMBED_RECORDS_TIMEOUT_MS,
    );
    let timeout = Duration::from_millis(timeout_ms);
    let permit = state.try_admit(&live).map_err(admission_error_response)?;

    let matrix = tokio::time::timeout(
        timeout,
        generation_service.execute(move |session| {
            let _permit = permit;
            session.embed_batch_flat(&batch, pooling, normalize)
        }),
    )
    .await
    .map_err(|_| {
        error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "service_unavailable",
            format!("embedding records timed out after {timeout_ms}ms"),
        )
    })?
    .map_err(map_generation_service_error)?;

    let data = chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| EmbeddingRecordObject {
            object: "embedding_record",
            id: chunk.id,
            embedding: matrix.row(index).to_vec(),
            index: index as u32,
            record_index: chunk.record_index as u32,
            chunk_index: chunk.chunk_index as u32,
            token_start: chunk.token_start as u32,
            token_end: chunk.token_end as u32,
            token_count: chunk.token_count as u32,
            metadata: chunk.metadata,
        })
        .collect::<Vec<_>>();
    let chunk_count = data.len();

    Ok(Json(EmbeddingRecordsResponse {
        object: "list",
        usage: EmbeddingRecordsUsage {
            prompt_tokens: token_count,
            total_tokens: token_count,
            records: request.records.len(),
            chunks: chunk_count,
        },
        data,
        model: model_id,
    }))
}

fn tokenizer_for_live(live: &LiveState) -> Result<EngineTokenizer, HttpErrorResponse> {
    let Some(model_dir) = live.session_config.mlx_model_artifacts_dir() else {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "embedding record ingestion requires mlx_model_artifacts_dir with tokenizer.json"
                .into(),
        ));
    };
    EngineTokenizer::from_model_dir_cached(model_dir).map_err(|error| {
        error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            format!("failed to load tokenizer: {error}"),
        )
    })
}

fn prepare_embedding_record_chunks(
    request: &EmbeddingRecordsRequest,
    tokenizer: &EngineTokenizer,
) -> Result<Vec<PreparedEmbeddingChunk>, HttpErrorResponse> {
    if request.records.is_empty() {
        return Err(invalid_request("records must not be empty"));
    }
    if request.records.len() > MAX_RECORDS_PER_REQUEST {
        return Err(invalid_request(format!(
            "records must contain at most {MAX_RECORDS_PER_REQUEST} items"
        )));
    }
    let chunking = normalize_chunking(request.chunking)?;
    let eos_token_id = if request.add_eos {
        tokenizer.eos_token_id()
    } else {
        None
    };
    let mut chunks = Vec::new();
    for (record_index, record) in request.records.iter().enumerate() {
        if record.id.trim().is_empty() {
            return Err(invalid_request(format!(
                "records[{record_index}].id must not be empty"
            )));
        }
        let rendered =
            render_record_text(record, request.render_template.as_deref(), record_index)?;
        let token_ids = tokenizer
            .encode(&rendered, false)
            .map_err(tokenization_error)?;
        if token_ids.is_empty() {
            return Err(invalid_request(format!(
                "records[{record_index}] rendered to no tokens"
            )));
        }
        for (chunk_index, (token_start, token_end)) in chunk_token_ranges(token_ids.len(), chunking)
            .into_iter()
            .enumerate()
        {
            let mut chunk_ids = token_ids[token_start..token_end].to_vec();
            if let Some(eos) = eos_token_id {
                chunk_ids.push(eos);
            }
            chunks.push(PreparedEmbeddingChunk {
                id: record.id.clone(),
                record_index,
                chunk_index,
                token_start,
                token_end,
                token_count: chunk_ids.len(),
                metadata: record.metadata.clone(),
                token_ids: chunk_ids,
            });
            if chunks.len() > MAX_CHUNKS_PER_REQUEST {
                return Err(invalid_request(format!(
                    "records produced more than {MAX_CHUNKS_PER_REQUEST} chunks"
                )));
            }
        }
    }
    Ok(chunks)
}

fn normalize_chunking(
    chunking: Option<EmbeddingChunking>,
) -> Result<EmbeddingChunking, HttpErrorResponse> {
    let chunking = chunking.unwrap_or(EmbeddingChunking {
        max_tokens: DEFAULT_CHUNK_MAX_TOKENS,
        overlap_tokens: DEFAULT_CHUNK_OVERLAP_TOKENS,
    });
    if chunking.max_tokens == 0 {
        return Err(invalid_request(
            "chunking.max_tokens must be greater than 0",
        ));
    }
    if chunking.overlap_tokens >= chunking.max_tokens {
        return Err(invalid_request(
            "chunking.overlap_tokens must be less than chunking.max_tokens",
        ));
    }
    Ok(chunking)
}

fn render_record_text(
    record: &EmbeddingRecordInput,
    render_template: Option<&str>,
    record_index: usize,
) -> Result<String, HttpErrorResponse> {
    if let Some(text) = record.text.as_deref() {
        if text.trim().is_empty() {
            return Err(invalid_request(format!(
                "records[{record_index}].text must not be empty"
            )));
        }
        return Ok(text.to_string());
    }
    if record.fields.is_empty() {
        return Err(invalid_request(format!(
            "records[{record_index}] must include text or fields"
        )));
    }
    let rendered = if let Some(template) = render_template {
        render_template_fields(template, &record.fields, record_index)?
    } else {
        render_default_fields(&record.fields)
    };
    if rendered.trim().is_empty() {
        return Err(invalid_request(format!(
            "records[{record_index}] rendered text must not be empty"
        )));
    }
    Ok(rendered)
}

fn render_template_fields(
    template: &str,
    fields: &BTreeMap<String, Value>,
    record_index: usize,
) -> Result<String, HttpErrorResponse> {
    // Single-pass replacement: each `{key}` in the template is resolved
    // exactly once against the field map. Field values are inserted verbatim
    // and never re-scanned for placeholders, preventing second-order
    // injection (e.g. a field value of `{other_key}` cannot cause a
    // subsequent replacement).
    let mut rendered = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        rendered.push_str(&rest[..start]);
        let after_open = &rest[start + 1..];
        if let Some(end) = after_open.find('}') {
            let key = &after_open[..end];
            match fields.get(key) {
                Some(value) => rendered.push_str(&render_field_value(value)),
                None => {
                    return Err(invalid_request(format!(
                        "records[{record_index}] render_template references missing field {key:?}"
                    )));
                }
            }
            rest = &after_open[end + 1..];
        } else {
            // No closing brace — keep the literal `{` and continue.
            rendered.push('{');
            rest = after_open;
        }
    }
    rendered.push_str(rest);
    Ok(rendered)
}

fn render_default_fields(fields: &BTreeMap<String, Value>) -> String {
    fields
        .iter()
        .map(|(key, value)| format!("{key}: {}", render_field_value(value)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_field_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn chunk_token_ranges(token_count: usize, chunking: EmbeddingChunking) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < token_count {
        let end = (start + chunking.max_tokens).min(token_count);
        ranges.push((start, end));
        if end == token_count {
            break;
        }
        start = end - chunking.overlap_tokens;
    }
    ranges
}

fn tokenization_error(error: EngineTokenizerError) -> HttpErrorResponse {
    error_response(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        format!("tokenization failed: {error}"),
    )
}

fn invalid_request(message: impl Into<String>) -> HttpErrorResponse {
    error_response(StatusCode::BAD_REQUEST, "invalid_request", message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_field_rendering_is_stable_and_sorted() {
        let mut fields = BTreeMap::new();
        fields.insert("body".to_string(), Value::String("World".to_string()));
        fields.insert("title".to_string(), Value::String("Hello".to_string()));

        assert_eq!(render_default_fields(&fields), "body: World\ntitle: Hello");
    }

    #[test]
    fn template_rendering_rejects_missing_fields() {
        let mut fields = BTreeMap::new();
        fields.insert("title".to_string(), Value::String("Hello".to_string()));

        let error = render_template_fields("{title}\n{body}", &fields, 0)
            .expect_err("missing placeholder should fail");

        assert_eq!(error.0, StatusCode::BAD_REQUEST);
        assert!(error.1.0.error.message.contains("missing field \"body\""));
    }

    #[test]
    fn chunk_ranges_include_overlap() {
        assert_eq!(
            chunk_token_ranges(
                10,
                EmbeddingChunking {
                    max_tokens: 4,
                    overlap_tokens: 1,
                },
            ),
            vec![(0, 4), (3, 7), (6, 10)]
        );
    }
}
