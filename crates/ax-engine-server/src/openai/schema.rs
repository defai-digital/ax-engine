use ax_engine_sdk::RequestMultimodalInputs;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Pre-tokenized input for the embedding endpoint. Mirrors OpenAI's
/// `input` shape support: one sequence (`[1,2,3]`) or a batch of
/// sequences (`[[1,2,3],[4,5,6]]`). Raw text inputs (`str` / `List[str]`)
/// are not accepted because the server does not run a tokenizer —
/// callers tokenize client-side and send token IDs.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum EmbeddingInput {
    /// A single sequence. Returns one embedding in `data[0]`.
    Single(Vec<u32>),
    /// A batch of sequences. Returns one embedding per input, in
    /// `data[i]` with `index == i`. The batch is dispatched directly
    /// to `embed_batch_flat`, bypassing the per-request microbatcher
    /// (the batch is already coalesced by the caller).
    Batch(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    pub(crate) fn into_batch(self) -> Vec<Vec<u32>> {
        match self {
            Self::Single(ids) => vec![ids],
            Self::Batch(b) => b,
        }
    }

    pub(crate) fn is_single(&self) -> bool {
        matches!(self, Self::Single(_))
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiEmbeddingRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    /// Pre-tokenized input: a single sequence (`[1,2,3]`) or a batch
    /// of sequences (`[[1,2,3],[4,5,6]]`). Raw text inputs are not
    /// accepted; tokenize client-side and send token IDs.
    pub(crate) input: EmbeddingInput,
    /// Ignored — always returns float32. Present for OpenAI API compatibility.
    #[serde(default)]
    #[serde(rename = "encoding_format")]
    pub(crate) _encoding_format: Option<String>,
    /// Pooling strategy: "last" (default), "mean", or "cls".
    ///
    /// "last" takes the final token's hidden state, which is the standard for
    /// decoder-only embedding models (Qwen3-Embedding, Gemma Embedding, etc.).
    /// The caller is responsible for appending an EOS token to the input when
    /// the model expects it.  "mean" averages all positions; "cls" takes the
    /// first token.
    #[serde(default)]
    pub(crate) pooling: Option<String>,
    /// Whether to L2-normalize the output vector to unit length (default true).
    ///
    /// Normalized embeddings allow cosine similarity to be computed as a simple
    /// dot product.  All major embedding APIs return normalized vectors.
    #[serde(default)]
    pub(crate) normalize: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiEmbeddingResponse {
    pub(crate) object: &'static str,
    pub(crate) data: Vec<OpenAiEmbeddingObject>,
    pub(crate) model: String,
    pub(crate) usage: OpenAiEmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiEmbeddingObject {
    pub(crate) object: &'static str,
    pub(crate) embedding: Vec<f32>,
    pub(crate) index: u32,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiEmbeddingUsage {
    pub(crate) prompt_tokens: usize,
    pub(crate) total_tokens: usize,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiCompletionHttpRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) prompt: OpenAiPromptInput,
    #[serde(default)]
    pub(crate) max_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) top_k: Option<u32>,
    #[serde(default)]
    pub(crate) min_p: Option<f32>,
    #[serde(default)]
    pub(crate) repetition_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) repetition_context_size: Option<u32>,
    #[serde(default)]
    pub(crate) stop: Option<OpenAiStopInput>,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) stream: bool,
    /// OpenAI legacy completions shape: an integer count of top alternatives
    /// to include. `0` requests sampled-token logprobs only; values above `0`
    /// are rejected until the runner emits top-N alternatives.
    #[serde(default)]
    pub(crate) logprobs: Option<u32>,
    #[serde(default)]
    pub(crate) top_logprobs: Option<u32>,
    #[serde(default)]
    pub(crate) metadata: Option<String>,
    #[serde(default)]
    pub(crate) multimodal_inputs: RequestMultimodalInputs,
    #[serde(default)]
    pub(crate) response_format: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChatCompletionHttpRequest {
    #[serde(default)]
    pub(crate) model: Option<String>,
    pub(crate) messages: Vec<OpenAiChatMessage>,
    #[serde(default)]
    pub(crate) input_tokens: Vec<u32>,
    #[serde(default)]
    pub(crate) max_tokens: Option<u32>,
    #[serde(default)]
    pub(crate) temperature: Option<f32>,
    #[serde(default)]
    pub(crate) top_p: Option<f32>,
    #[serde(default)]
    pub(crate) top_k: Option<u32>,
    #[serde(default)]
    pub(crate) min_p: Option<f32>,
    #[serde(default)]
    pub(crate) repetition_penalty: Option<f32>,
    #[serde(default)]
    pub(crate) repetition_context_size: Option<u32>,
    #[serde(default)]
    pub(crate) stop: Option<OpenAiStopInput>,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) stream: bool,
    #[serde(default)]
    pub(crate) logprobs: bool,
    #[serde(default)]
    pub(crate) top_logprobs: Option<u32>,
    #[serde(default)]
    pub(crate) reasoning: Option<Value>,
    #[serde(default)]
    pub(crate) metadata: Option<String>,
    #[serde(default)]
    pub(crate) multimodal_inputs: RequestMultimodalInputs,
    #[serde(default)]
    pub(crate) response_format: Option<Value>,
    #[serde(default)]
    pub(crate) tools: Option<Value>,
    #[serde(default)]
    pub(crate) tool_choice: Option<Value>,
}

/// OpenAI `stop` field: either a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OpenAiStopInput {
    Single(String),
    Multiple(Vec<String>),
}

impl OpenAiStopInput {
    pub(crate) fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OpenAiPromptInput {
    Text(String),
    TextBatch(Vec<String>),
    Tokens(Vec<u32>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChatMessage {
    pub(crate) role: String,
    #[serde(default)]
    pub(crate) content: Option<OpenAiChatContent>,
    #[serde(default)]
    pub(crate) tool_calls: Option<Value>,
    #[serde(default)]
    #[serde(rename = "tool_call_id")]
    pub(crate) _tool_call_id: Option<String>,
    #[serde(default)]
    #[serde(rename = "name")]
    pub(crate) _name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum OpenAiChatContent {
    Text(String),
    Parts(Vec<OpenAiChatContentPart>),
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChatContentPart {
    #[serde(rename = "type")]
    pub(crate) part_type: String,
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) image_url: Option<Value>,
    #[serde(default)]
    pub(crate) input_audio: Option<Value>,
    #[serde(default)]
    pub(crate) audio_url: Option<Value>,
    #[serde(default)]
    pub(crate) video_url: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) system_fingerprint: Option<&'static str>,
    pub(crate) choices: Vec<OpenAiCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionChoice {
    pub(crate) index: u32,
    pub(crate) text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<OpenAiCompletionLogprobs>,
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionLogprobs {
    pub(crate) tokens: Vec<String>,
    pub(crate) token_logprobs: Vec<Option<f32>>,
    pub(crate) top_logprobs: Vec<Option<Value>>,
    pub(crate) text_offset: Vec<u32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionResponse {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) system_fingerprint: Option<&'static str>,
    pub(crate) choices: Vec<OpenAiChatCompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<OpenAiUsage>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionChoice {
    pub(crate) index: u32,
    pub(crate) message: OpenAiChatMessageResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<OpenAiChatLogprobs>,
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatMessageResponse {
    pub(crate) role: &'static str,
    #[serde(serialize_with = "serialize_empty_as_null")]
    pub(crate) content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<OpenAiToolCall>>,
}

/// Serializes an empty string as JSON `null` so tool-call-only responses emit
/// `"content": null` rather than `"content": ""`, matching the OpenAI API
/// contract where `content` is `string | null`.
fn serialize_empty_as_null<S: serde::Serializer>(
    value: &str,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    if value.is_empty() {
        serializer.serialize_none()
    } else {
        serializer.serialize_str(value)
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiToolCall {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) tool_type: &'static str,
    pub(crate) function: OpenAiFunctionCall,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct OpenAiFunctionCall {
    pub(crate) name: String,
    pub(crate) arguments: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatLogprobs {
    pub(crate) content: Vec<OpenAiChatTokenLogprob>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatTokenLogprob {
    pub(crate) token: String,
    pub(crate) logprob: f32,
    pub(crate) bytes: Option<Vec<u8>>,
    pub(crate) top_logprobs: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionChunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) system_fingerprint: Option<&'static str>,
    pub(crate) choices: Vec<OpenAiCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiCompletionChunkChoice {
    pub(crate) index: u32,
    pub(crate) text: String,
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionChunk {
    pub(crate) id: String,
    pub(crate) object: &'static str,
    pub(crate) created: u64,
    pub(crate) model: String,
    pub(crate) system_fingerprint: Option<&'static str>,
    pub(crate) choices: Vec<OpenAiChatCompletionChunkChoice>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiChatCompletionChunkChoice {
    pub(crate) index: u32,
    pub(crate) delta: OpenAiChatDelta,
    pub(crate) finish_reason: Option<&'static str>,
}

#[derive(Debug, Default, Serialize)]
pub(crate) struct OpenAiChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiToolCallDelta {
    pub(crate) index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub(crate) tool_type: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) function: Option<OpenAiFunctionCallDelta>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiFunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) arguments: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiUsage {
    pub(crate) prompt_tokens: u32,
    pub(crate) completion_tokens: u32,
    pub(crate) total_tokens: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum OpenAiStreamKind {
    Completion,
    ChatCompletion,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn chat_message_response_serializes_empty_content_as_null() {
        // OpenAI API contract: when content is empty (tool-call-only response),
        // the JSON must emit "content": null, not "content": "".
        let msg = OpenAiChatMessageResponse {
            role: "assistant",
            content: String::new(),
            reasoning_content: None,
            tool_calls: Some(vec![OpenAiToolCall {
                id: "call_0".to_string(),
                tool_type: "function",
                function: OpenAiFunctionCall {
                    name: "lookup".to_string(),
                    arguments: r#"{"q":"hello"}"#.to_string(),
                },
            }]),
        };
        let json: serde_json::Value = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], serde_json::Value::Null,
            "empty content must serialize as null when tool_calls are present");
        assert!(json["tool_calls"].is_array());
    }

    #[test]
    fn chat_message_response_serializes_nonempty_content_as_string() {
        let msg = OpenAiChatMessageResponse {
            role: "assistant",
            content: "Hello!".to_string(),
            reasoning_content: None,
            tool_calls: None,
        };
        let json: serde_json::Value = serde_json::to_value(&msg).unwrap();
        assert_eq!(json["content"], json!("Hello!"));
    }
}
