use ax_engine_sdk::GenerateFinishReason;

use crate::openai::responses::{openai_finish_reason, unix_timestamp_secs};
use crate::openai::schema::{
    OpenAiChatCompletionChunk, OpenAiChatCompletionChunkChoice, OpenAiChatDelta,
    OpenAiCompletionChunk, OpenAiCompletionChunkChoice, OpenAiStreamKind,
};

pub(crate) fn completion_delta_chunk(
    request_id: u64,
    model: String,
    text: String,
) -> OpenAiCompletionChunk {
    OpenAiCompletionChunk {
        id: OpenAiStreamKind::Completion.response_id(request_id),
        object: OpenAiStreamKind::Completion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model,
        system_fingerprint: None,
        choices: vec![OpenAiCompletionChunkChoice {
            index: 0,
            text,
            finish_reason: None,
        }],
    }
}

pub(crate) fn completion_final_chunk(
    request_id: u64,
    model: String,
    finish_reason: Option<GenerateFinishReason>,
) -> OpenAiCompletionChunk {
    OpenAiCompletionChunk {
        id: OpenAiStreamKind::Completion.response_id(request_id),
        object: OpenAiStreamKind::Completion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model,
        system_fingerprint: None,
        choices: vec![OpenAiCompletionChunkChoice {
            index: 0,
            text: String::new(),
            finish_reason: openai_finish_reason(finish_reason),
        }],
    }
}

pub(crate) fn chat_delta_chunk(
    request_id: u64,
    model: String,
    role: Option<&'static str>,
    content: String,
) -> OpenAiChatCompletionChunk {
    OpenAiChatCompletionChunk {
        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model,
        system_fingerprint: None,
        choices: vec![OpenAiChatCompletionChunkChoice {
            index: 0,
            delta: OpenAiChatDelta {
                role,
                content: Some(content),
            },
            finish_reason: None,
        }],
    }
}

pub(crate) fn next_chat_delta_role(chat_role_emitted: &mut bool) -> Option<&'static str> {
    if *chat_role_emitted {
        None
    } else {
        *chat_role_emitted = true;
        Some("assistant")
    }
}

pub(crate) fn chat_final_chunk(
    request_id: u64,
    model: String,
    finish_reason: Option<GenerateFinishReason>,
) -> OpenAiChatCompletionChunk {
    OpenAiChatCompletionChunk {
        id: OpenAiStreamKind::ChatCompletion.response_id(request_id),
        object: OpenAiStreamKind::ChatCompletion.stream_chunk_object(),
        created: unix_timestamp_secs(),
        model,
        system_fingerprint: None,
        choices: vec![OpenAiChatCompletionChunkChoice {
            index: 0,
            delta: OpenAiChatDelta::default(),
            finish_reason: openai_finish_reason(finish_reason),
        }],
    }
}
