use axum::response::sse::Event;
use serde::Serialize;

use crate::errors::ErrorResponse;
use crate::generation::streaming::{StreamEventSender, send_stream_error};

pub(crate) fn send_openai_stream_chunk<T: Serialize>(tx: &StreamEventSender, payload: &T) -> bool {
    match serde_json::to_string(payload) {
        Ok(data) => tx.blocking_send(Ok(Event::default().data(data))).is_ok(),
        Err(error) => {
            send_stream_error(
                tx,
                ErrorResponse::server_error(format!(
                    "failed to serialize OpenAI stream chunk: {error}"
                )),
            );
            false
        }
    }
}
